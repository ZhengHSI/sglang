#!/usr/bin/env python3
"""
测试 MOE 中各个算子的精度误差
"""

import torch
import sys
import numpy as np
from typing import Tuple, Optional

# 添加 sglang 路径
sys.path.insert(0, "sglang/python")

def create_moe_test_data(
    batch_size: int = 128,
    seq_len: int = 256, 
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_experts: int = 8,
    topk: int = 2,
    device: str = "cuda"
) -> dict:
    """
    创建 MOE 测试数据
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度  
        hidden_size: 隐藏层大小
        intermediate_size: 中间层大小
        num_experts: 专家数量
        topk: 每个token选择的专家数量
        device: 设备
    
    Returns:
        包含所有测试数据的字典
    """
    # print(f"🏗️ 创建MOE测试数据...")
    # print(f"   批次大小: {batch_size}, 序列长度: {seq_len}")
    # print(f"   隐藏层大小: {hidden_size}, 中间层大小: {intermediate_size}")
    # print(f"   专家数量: {num_experts}, TopK: {topk}")
    
    M = batch_size * seq_len
    K = hidden_size
    N = intermediate_size
    
    # 1. 输入激活
    a = torch.randn(M, K, device=device, dtype=torch.half)
    print(f"   输入形状: {a.shape}")
    
    # 2. 量化权重 (模拟int4权重用int8存储)
    # W1: [num_experts, N*2, K//2] - gate和up权重合并, int4打包
    w1_q = torch.randint(-128, 127, (num_experts, N * 2, K // 2), 
                        device=device, dtype=torch.int8)
    
    # W2: [num_experts, K, N//2] - down权重, int4打包
    w2_q = torch.randint(-128, 127, (num_experts, K, N // 2), 
                        device=device, dtype=torch.int8)
    
    print(f"   W1权重形状: {w1_q.shape}")
    print(f"   W2权重形状: {w2_q.shape}")
    
    # 3. 权重scales (group-wise quantization)
    # W1 scale: [num_experts, K//512, N*8] 
    w1_scale = torch.rand(num_experts, K // 512, N * 8, 
                         device=device, dtype=torch.bfloat16) * 0.1
    
    # W2 scale: [num_experts, N//512, K*4]
    w2_scale = torch.rand(num_experts, N // 512, K * 4, 
                         device=device, dtype=torch.bfloat16) * 0.1
    
    print(f"   W1 scale形状: {w1_scale.shape}")
    print(f"   W2 scale形状: {w2_scale.shape}")
    
    # 4. Router输出 - 模拟top-k路由结果
    # 为每个token随机选择topk个专家
    topk_ids = torch.randint(0, num_experts, (M, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand(M, topk, device=device, dtype=torch.half)
    # 归一化权重
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    print(f"   TopK IDs形状: {topk_ids.shape}")
    print(f"   TopK权重形状: {topk_weights.shape}")
    
    # 5. 生成local_topk_ids (展平版本)
    local_topk_ids = topk_ids.view(-1)
    
    # 6. 激活量化scale (可选) - 必须是1D张量 [1]
    a1_scale = torch.tensor([0.01], device=device, dtype=torch.float32)
    a2_scale = torch.tensor([0.015], device=device, dtype=torch.float32)
    
    # 7. 专家偏移量 (用于CUTLASS grouped GEMM)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for i in range(num_experts):
        expert_counts[i] = (local_topk_ids == i).sum().item()
    
    expert_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device, dtype=torch.int32), expert_counts]), dim=0).to(torch.int32)
    
    # 8. Problem sizes for grouped GEMM
    problem_sizes1 = torch.zeros(num_experts, 3, dtype=torch.int32, device=device)
    problem_sizes2 = torch.zeros(num_experts, 3, dtype=torch.int32, device=device)
    
    for i in range(num_experts):
        count = expert_counts[i].item()
        # 第一个GEMM: [count, N*2, K]
        problem_sizes1[i] = torch.tensor([count, N * 2, K], dtype=torch.int32)
        # 第二个GEMM: [count, K, N] 
        problem_sizes2[i] = torch.tensor([count, K, N], dtype=torch.int32)
    
    # 9. Strides for grouped GEMM
    a_strides1 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    b_strides1 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    c_strides1 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    
    a_strides2 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    b_strides2 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)  
    c_strides2 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    
    # 设置strides
    for i in range(num_experts):
        # 第一个GEMM的strides
        a_strides1[i] = torch.tensor([K, 1, K], dtype=torch.int64)
        b_strides1[i] = torch.tensor([K//2, 1, K//2 * N * 2], dtype=torch.int64)
        c_strides1[i] = torch.tensor([N * 2, 1, N * 2], dtype=torch.int64)
        
        # 第二个GEMM的strides  
        a_strides2[i] = torch.tensor([N, 1, N], dtype=torch.int64)
        b_strides2[i] = torch.tensor([N//2, 1, N//2 * K], dtype=torch.int64)
        c_strides2[i] = torch.tensor([K, 1, K], dtype=torch.int64)
    
    # 10. Scale strides
    s_strides13 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    s_strides2 = torch.zeros(num_experts, 3, dtype=torch.int64, device=device)
    
    for i in range(num_experts):
        s_strides13[i] = torch.tensor([1, N * 8, 1], dtype=torch.int64)
        s_strides2[i] = torch.tensor([1, K * 4, 1], dtype=torch.int64)
    
    print(f"   专家偏移量: {expert_offsets}")
    print(f"   专家token计数: {expert_counts}")
    
    return {
        # 基础数据
        'a': a,
        'w1_q': w1_q,
        'w2_q': w2_q, 
        'w1_scale': w1_scale,
        'w2_scale': w2_scale,
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'local_topk_ids': local_topk_ids,
        'a1_scale': a1_scale,
        'a2_scale': a2_scale,
        
        # CUTLASS数据
        'expert_offsets': expert_offsets,
        'problem_sizes1': problem_sizes1,
        'problem_sizes2': problem_sizes2,
        'a_strides1': a_strides1,
        'b_strides1': b_strides1,
        'c_strides1': c_strides1,
        'a_strides2': a_strides2,
        'b_strides2': b_strides2,
        'c_strides2': c_strides2,
        's_strides13': s_strides13,
        's_strides2': s_strides2,
        
        # 元数据
        'M': M,
        'K': K,
        'N': N,
        'num_experts': num_experts,
        'topk': topk,
        'device': device
    }

def test_pre_reorder_kernel(data: dict):
    """测试pre_reorder_triton_kernel_for_cutlass_moe算子精度"""
    print("\n🔍 测试 pre_reorder_triton_kernel_for_cutlass_moe")
    
    try:
        from sglang.srt.layers.moe.ep_moe.kernels import pre_reorder_triton_kernel_for_cutlass_moe
        from sglang.srt.layers.moe.cutlass_w4a8_moe import run_cutlass_moe_ep_preproess
        
        M, K = data['M'], data['K']
        topk = data['topk']
        
        # 运行预处理获取src2dst映射
        _, src2dst, _ = run_cutlass_moe_ep_preproess(
            data['local_topk_ids'],
            data['num_experts'],
        )
        
        # 创建输出缓冲区
        gateup_input_int8 = torch.empty((M * topk, K), 
                                  device=data['device'], dtype=torch.int8)
        gateup_input_fp8 = torch.empty((M * topk, K), 
                                  device=data['device'], dtype=torch.int8)

        
        # 运行kernel
        pre_reorder_triton_kernel_for_cutlass_moe[(M,)](
            data['a'],
            gateup_input_int8,
            src2dst,
            data['local_topk_ids'],
            data['a1_scale'],
            data['num_experts'],
            topk,
            K,
            BLOCK_SIZE=512,
        )
        pre_reorder_triton_kernel_for_cutlass_moe[(M,)](
            data['a'],
            gateup_input_fp8,
            src2dst,
            data['local_topk_ids'],
            data['a1_scale'],
            data['num_experts'],
            topk,
            K,
            BLOCK_SIZE=512,
        )
        
        print(f"   ✅ 重排序完成")
        print(f"   输入形状: {data['a'].shape}")
        print(f"   输出形状: {gateup_input_int8.shape}")
        print(f"   输出数据类型: {gateup_input_int8.dtype}")
        print(f"   输出范围: [{gateup_input_int8.min().item()}, {gateup_input_int8.max().item()}]")
        # 简单精度检查 - 验证量化是否合理
        expected_range = 127 * data['a1_scale'].item()
        input_range = data['a'].abs().max().item()
        print(f"   输入数据范围: {input_range:.4f}")
        print(f"   预期量化范围: ±{expected_range:.4f}")
        
        # 验证量化数据的合理性
        dequant_range = gateup_input_int8.float().abs().max().item() * data['a1_scale'].item()
        print(f"   反量化数据范围: {dequant_range:.4f}")
        quantization_error = abs(input_range - dequant_range) / input_range * 100
        print(f"   量化误差: {quantization_error:.2f}%")

        #验证fp8和int8的差距
        diff = torch.mean(torch.abs(gateup_input_int8.float()*data['a1_scale'] - gateup_input_fp8.float()*data['a1_scale']))
        print("diff between int 8 and fp8: ", diff)
        torch.testing.assert_close(gateup_input_int8.float(), gateup_input_fp8.float(), rtol=0.01, atol=0.1)
        return gateup_input_int8, src2dst
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_cutlass_mm_data_generation(data: dict):
    """测试get_cutlass_w4a8_moe_mm_data算子"""
    print("\n🔍 测试 get_cutlass_w4a8_moe_mm_data")
    
    try:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import get_cutlass_w4a8_moe_mm_data
        
        M, K, N = data['M'], data['K'], data['N']
        num_experts = data['num_experts']
        
        # 创建输出映射
        a_map = torch.empty((data['local_topk_ids'].numel()), 
                           dtype=torch.int32, device=data['device'])
        c_map = torch.empty((data['local_topk_ids'].numel()), 
                           dtype=torch.int32, device=data['device'])
        
        # 运行数据生成
        get_cutlass_w4a8_moe_mm_data(
            data['local_topk_ids'],
            data['expert_offsets'],
            data['problem_sizes1'],
            data['problem_sizes2'],
            a_map,
            c_map,
            num_experts,
            N,
            K,
        )
        
        print(f"   ✅ CUTLASS数据生成完成")
        print(f"   A映射形状: {a_map.shape}")
        print(f"   C映射形状: {c_map.shape}")
        print(f"   A映射范围: [{a_map.min().item()}, {a_map.max().item()}]")
        print(f"   C映射范围: [{c_map.min().item()}, {c_map.max().item()}]")
        
        return a_map, c_map
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_per_tensor_quantization(data: dict):
    """测试per_tensor_quant_int8算子精度"""
    print("\n🔍 测试 per_tensor_quant_int8")
    
    try:
        from sglang.srt.layers.quantization.int8_kernel import per_tensor_quant_int8
        
        # 创建中间结果（模拟第一次GEMM+激活后的结果）
        M, N = data['M'], data['N']
        topk = data['topk']
        
        intermediate = torch.randn(M * topk, N, device=data['device'], dtype=torch.half)
        
        print(f"   中间结果形状: {intermediate.shape}")
        print(f"   中间结果范围: [{intermediate.min().item():.4f}, {intermediate.max().item():.4f}]")
        
        # 运行量化
        intermediate_q, scale = per_tensor_quant_int8(intermediate)
        
        print(f"   ✅ 量化完成")
        print(f"   量化结果形状: {intermediate_q.shape}")
        print(f"   量化结果类型: {intermediate_q.dtype}")
        print(f"   量化范围: [{intermediate_q.min().item()}, {intermediate_q.max().item()}]")
        print(f"   量化scale: {scale.item():.8f}")
        
        # 精度验证
        intermediate_dequant = intermediate_q.float() * scale
        mse = torch.mean((intermediate.float() - intermediate_dequant) ** 2)
        mae = torch.mean(torch.abs(intermediate.float() - intermediate_dequant))
        relative_error = mae / torch.mean(torch.abs(intermediate.float()))
        
        print(f"   MSE: {mse.item():.6f}")
        print(f"   MAE: {mae.item():.6f}")
        print(f"   相对误差: {relative_error.item():.4%}")
        
        return intermediate_q, scale
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_precision_tests():
    """运行所有精度测试"""
    print("🚀 开始MOE算子精度测试")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    # 创建测试数据
    data = create_moe_test_data(
        batch_size=32,
        seq_len=128, 
        hidden_size=4096,
        intermediate_size=11008,
        num_experts=8,
        topk=2
    )
    
    # 验证关键数据的形状
    # print(f"\n🔧 关键参数验证:")
    # print(f"   a1_scale形状: {data['a1_scale'].shape}")
    # print(f"   a2_scale形状: {data['a2_scale'].shape}")
    # print(f"   expert_offsets形状: {data['expert_offsets'].shape}")
    # print(f"   problem_sizes1形状: {data['problem_sizes1'].shape}")
    
    # print(f"\n📊 测试数据统计:")
    # print(f"   总token数: {data['M']}")
    # print(f"   总专家调用数: {data['M'] * data['topk']}")
    
    # 测试各个算子
    results = {}
    
    # 1. 测试预重排序
    gateup_input, src2dst = test_pre_reorder_kernel(data)
    results['pre_reorder'] = (gateup_input, src2dst)
    
    # # 2. 测试CUTLASS数据生成
    # a_map, c_map = test_cutlass_mm_data_generation(data)
    # results['cutlass_data'] = (a_map, c_map)
    
    # # 3. 测试量化算子
    # intermediate_q, scale = test_per_tensor_quantization(data)
    # results['quantization'] = (intermediate_q, scale)
    
    print(f"\n🎯 测试总结:")
    success_count = sum(1 for v in results.values() if v is not None and v[0] is not None)
    total_count = len(results)
    print(f"   成功: {success_count}/{total_count}")
    
    return results, data

if __name__ == "__main__":
    results, data = run_precision_tests() 