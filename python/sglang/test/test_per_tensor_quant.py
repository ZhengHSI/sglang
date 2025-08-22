#!/usr/bin/env python3
"""
测试 per-tensor INT8 量化算子
"""

import torch
import sys
import os

# 添加 sglang 路径
# sys.path.insert(0, "sglang/python")

def test_per_tensor_quant():
    """测试 per-tensor 量化功能"""
    print("🧪 开始测试 per-tensor INT8 量化算子")
    
    try:
        from sglang.srt.layers.quantization.int8_kernel import per_tensor_quant_int8
        print("✅ 成功导入 per_tensor_quant_int8 函数")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保在正确的环境中运行")
        return False
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，跳过测试（Triton 需要 CUDA）")
        return False
    
    device = "cuda"
    print(f"🔧 使用设备: {device}")
    
    # 基础功能测试
    print("\n📋 基础功能测试")
    test_shapes = [
        (128, 256),        # 2D tensor
        (64, 128, 256),    # 3D tensor
        (4, 64, 128, 256), # 4D tensor
        (1024,),           # 1D tensor
    ]
    
    for i, shape in enumerate(test_shapes):
        print(f"  测试 {i+1}: 形状 {shape}")
        
        # 创建测试数据
        x = torch.randn(shape, device=device, dtype=torch.float16)
        
        try:
            # 测试不带sum的量化
            x_q, scale = per_tensor_quant_int8(x, cal_sum=False)
            
            # 验证基本属性
            assert x_q.shape == x.shape, f"形状不匹配: {x_q.shape} vs {x.shape}"
            assert x_q.dtype == torch.int8, f"输出类型错误: {x_q.dtype}"
            assert scale.shape == torch.Size([1]), f"Scale形状错误: {scale.shape}"
            assert x_q.min() >= -128 and x_q.max() <= 127, "量化值超出int8范围"
            
            print(f"    ✅ 量化范围: [{x_q.min().item()}, {x_q.max().item()}], Scale: {scale.item():.6f}")
            
            # 测试带sum的量化
            x_q_sum, scale_sum, x_sum = per_tensor_quant_int8(x, cal_sum=True)
            
            # 验证一致性
            assert torch.allclose(x_q, x_q_sum), "带/不带sum的量化结果不一致"
            assert torch.allclose(scale, scale_sum), "带/不带sum的scale不一致"
            
            print(f"    ✅ Sum计算: {x_sum.item():.2f} (原始: {x.sum().item():.2f})")
            
        except Exception as e:
            print(f"    ❌ 测试失败: {e}")
            return False
    
    # 精度验证测试
    print("\n🎯 精度验证测试")
    x = torch.randn(512, 512, device=device, dtype=torch.float16)
    x_q, scale = per_tensor_quant_int8(x)
    
    # 反量化
    x_dequant = x_q.float() * scale
    
    # 计算误差
    mse = torch.mean((x.float() - x_dequant) ** 2)
    mae = torch.mean(torch.abs(x.float() - x_dequant))
    
    print(f"  MSE: {mse.item():.6f}")
    print(f"  MAE: {mae.item():.6f}")
    print(f"  相对误差: {(mae / torch.mean(torch.abs(x.float()))).item():.4%}")
    
    # 边界情况测试
    print("\n🔍 边界情况测试")
    
    # 测试全零张量
    x_zeros = torch.zeros(100, 100, device=device, dtype=torch.float16)
    x_q_zeros, scale_zeros = per_tensor_quant_int8(x_zeros)
    print(f"  全零张量: Scale = {scale_zeros.item():.8f}, 量化值全为0: {torch.all(x_q_zeros == 0).item()}")
    
    # 测试很小的值
    x_small = torch.full((100, 100), 1e-8, device=device, dtype=torch.float16)
    x_q_small, scale_small = per_tensor_quant_int8(x_small)
    print(f"  极小值张量: Scale = {scale_small.item():.8f}")
    
    # 测试很大的值
    x_large = torch.full((100, 100), 1000.0, device=device, dtype=torch.float16)
    x_q_large, scale_large = per_tensor_quant_int8(x_large)
    print(f"  大值张量: Scale = {scale_large.item():.6f}, Max量化值: {x_q_large.max().item()}")
    
    # 性能测试
    print("\n⚡ 性能测试")
    shape = (1024, 4096)
    x = torch.randn(shape, device=device, dtype=torch.float16)
    
    # 预热
    for _ in range(10):
        _ = per_tensor_quant_int8(x)
    
    # 计时
    torch.cuda.synchronize()
    import time
    start_time = time.time()
    
    num_runs = 100
    for _ in range(num_runs):
        _ = per_tensor_quant_int8(x)
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_time = total_time / num_runs * 1000  # ms
    
    elements = x.numel()
    throughput = elements * num_runs / total_time / 1e9  # GElements/s
    
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  吞吐量: {throughput:.2f} GElements/s")
    
    print("\n🎉 所有测试通过！Per-tensor 量化算子工作正常")
    return True

def compare_with_pytorch():
    """与PyTorch原生实现对比"""
    print("\n📊 与PyTorch原生实现对比")
    
    def pytorch_per_tensor_quant(x):
        """PyTorch原生per-tensor量化实现"""
        absmax = torch.max(torch.abs(x))
        scale = absmax / 127.0
        x_q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
        return x_q, scale
    
    device = "cuda"
    x = torch.randn(512, 512, device=device, dtype=torch.float16)
    
    # 我们的实现
    try:
        from sglang.srt.layers.quantization.int8_kernel import per_tensor_quant_int8
        x_q_ours, scale_ours = per_tensor_quant_int8(x)
    except:
        print("  ❌ 无法导入我们的实现")
        return
    
    # PyTorch实现
    x_q_pt, scale_pt = pytorch_per_tensor_quant(x)
    
    # 比较结果
    scale_diff = torch.abs(scale_ours - scale_pt)
    quant_diff = torch.mean(torch.abs(x_q_ours.float() - x_q_pt.float()))
    fp_diff = torch.mean(torch.abs(x_q_ours.float()*scale_ours - x.float()))

    print(f"  Scale差异: {scale_diff.item():.8f}")
    print(f"  量化值平均差异: {quant_diff.item():.6f}")
    print(f"  量化误差: {fp_diff.item():.6f}")
    # 性能比较
    import time
    
    # 预热
    for _ in range(10):
        _ = per_tensor_quant_int8(x)
        _ = pytorch_per_tensor_quant(x)
    
    # 我们的实现
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = per_tensor_quant_int8(x)
    torch.cuda.synchronize()
    our_time = time.time() - start
    
    # PyTorch实现
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = pytorch_per_tensor_quant(x)
    torch.cuda.synchronize()
    pt_time = time.time() - start
    
    speedup = pt_time / our_time
    print(f"  我们的实现: {our_time*1000:.2f} ms")
    print(f"  PyTorch实现: {pt_time*1000:.2f} ms")
    print(f"  加速比: {speedup:.2f}x")

if __name__ == "__main__":
    success = test_per_tensor_quant()
    if success:
        compare_with_pytorch() 