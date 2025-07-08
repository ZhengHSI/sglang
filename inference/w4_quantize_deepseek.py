import os
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import re

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

from kernel import weight_quant
import tensorrt_llm

group_s = 128

def quantize_bf16_to_int4(bf16_tensor):
    """Quantize bf16 tensor directly to int4 with block-wise scaling"""
    group_size = group_s
    # Reshape tensor to group_size blocks
    if bf16_tensor.shape[1] % group_size != 0:
        # Pad if necessary
        pad_size = group_size - (bf16_tensor.shape[1] % group_size)
        bf16_tensor = torch.nn.functional.pad(bf16_tensor, (0, pad_size))
    
    # Reshape to (out_features, num_groups, group_size)
    num_groups = bf16_tensor.shape[1] // group_size
    blocked_tensor = bf16_tensor.view(bf16_tensor.shape[0], num_groups, group_size)
    
    # Calculate scale for each group
    scale_tensor = torch.abs(blocked_tensor).max(dim=2).values / 7
    
    # Quantize to int4 (-8 to 7)
    quant_tensor = torch.clamp(torch.round(
        (blocked_tensor / scale_tensor.unsqueeze(-1))),
                               min=-8,
                               max=7)
    quant_tensor = quant_tensor.to(torch.int8)
    
    return quant_tensor.view(bf16_tensor.shape[0], bf16_tensor.shape[1]), scale_tensor

def quantize_fp8_block_scale_to_int4(fp8_tensor, fp8_scale):
    """将FP8块缩放量化转换为INT4量化"""
    group_size = group_s
    blocked_tensor = fp8_tensor.view(fp8_tensor.shape[0] // group_size, group_size,
                                     fp8_tensor.shape[1] // group_size,
                                     group_size).to(torch.float32)
    dequant_tensor = (blocked_tensor *
                      fp8_scale.unsqueeze(1).unsqueeze(3)).view(
                          fp8_tensor.shape[0],
                          fp8_tensor.shape[1] // group_size,
                          group_size).to(torch.bfloat16).to(torch.float32)
    scale_tensor = torch.abs(dequant_tensor).max(dim=2).values / 7
    quant_tensor = torch.clamp(torch.round(
        (dequant_tensor / scale_tensor.unsqueeze(-1))),
                               min=-8,
                               max=7)
    quant_tensor = quant_tensor.to(torch.int8)
    return quant_tensor.view(fp8_tensor.shape), scale_tensor

def main(bf16_path, w4_path, f8_path, model_name="deepseek-ai/DeepSeek-R1"):
    group_size = group_s
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(w4_path, exist_ok=True)
    model_index_file = os.path.join(w4_path, "model.safetensors.index.json")
    config_file = os.path.join(w4_path, "config.json")
    
    # 需要拷贝的文件列表
    files_to_copy = [
        "model.safetensors.index.json",
        "config.json",
        "configuration_deepseek.py",
        "generation_config.json",
        "modeling_deepseek.py",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    # 从BF16路径拷贝文件（如果不存在的话）
    for file_name in files_to_copy:
        src_file = os.path.join(f8_path, file_name)
        dst_file = os.path.join(w4_path, file_name)
        
        if os.path.exists(src_file) and not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {file_name} from {bf16_path} to {w4_path}")
        elif not os.path.exists(dst_file):
            print(f"Warning: {file_name} not found in {bf16_path}")
     
    # 如果关键文件仍然不存在，则从HuggingFace下载
    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        print(f"Downloading missing files from HuggingFace...")
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.safetensors"],
            local_dir=w4_path,
            local_dir_use_symlinks=False
        )
        print(f"model index file and config file downloaded to {w4_path}")

    # 修改config.json
    if os.path.exists(config_file):
        config = json.load(open(config_file))
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            quant_config.pop("fmt", None)
            quant_config["quant_method"] = "mixed_precision_w4"
            quant_config["weight_block_size"] = [group_size, group_size]
            quant_config["activation_scheme"] = "dynamic"
            quant_config["weight_group_size"] = group_size
        else:
            config["quantization_config"] = {
                "activation_scheme": "dynamic",
                "quant_method": "mixed_precision_w4",
                "weight_block_size": [group_size, group_size],
                "weight_group_size": group_size
            }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"config.json modified and saved to {config_file}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    expert_quant_count = len([key for key in weight_map.keys() 
                             if "mlp.experts" in key and key.endswith("weight")])
    block_int8_count = len([key for key in weight_map.keys() 
                           if key.endswith("_scale_inv") and "mlp.experts" not in key])
    
    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    
    expert_quantized = 0
    block_int8_quantized = 0
    
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}
        
        for weight_name, weight in state_dict.items():
            if "mlp.experts" in weight_name and weight_name.endswith("weight"):
                expert_quantized += 1
                # print("int4: ", weight_name)
                int8_weight, scale_tensor = quantize_bf16_to_int4(weight)
                
                # 打包为INT4格式
                # TODO 这里的tensor打包还是有点问题
                packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
                packed_weight = packer(int8_weight.cpu().contiguous())
                
                new_state_dict[weight_name] = packed_weight
                new_state_dict[f"{weight_name}_scale_inv"] = scale_tensor
                
            # # 检查是否为其他权重（使用block int8量化）
            # elif weight_name.endswith("_scale_inv") and "mlp.experts" not in weight_name:
            #     # 跳过scale_inv，因为我们会重新生成
            #     continue
            elif "mlp.experts" not in weight_name and weight_name.endswith("weight"):
                # 对其他权重使用block int8量化
                scale_inv_name = f"{weight_name}_scale_inv"
                if scale_inv_name in weight_map:
                    assert weight.element_size() == 2
                    print("int8: ", weight_name)
                    block_int8_quantized += 1
                    int8_weight, scale_inv = weight_quant(weight)
                    new_state_dict[weight_name] = int8_weight
                    new_state_dict[scale_inv_name] = scale_inv
                else:
                    new_state_dict[weight_name] = weight
            else:
                print("bf16: ", weight_name)
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(w4_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
    
    print(f"Expert weights quantized: {expert_quantized}")
    print(f"Block INT8 weights quantized: {block_int8_quantized}")
    
    # 创建量化配置文件
    create_quant_config(w4_path)

def create_quant_config(output_dir):
    """创建量化配置文件"""
    # 读取模型配置以获取层数
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    num_layers = config.get('num_hidden_layers', 61)
    
    # 创建quant_cfg.json
    quant_cfg = {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "quantized_layers": {}
    }
    
    # 定义注意力层名称
    attn_names = ["fused_a", "q_b_proj", "kv_b_proj", "o_proj"]
    mlp_names = ["gate_up_proj", "down_proj"]
    
    # 定义量化算法
    w4a8_awq = {"quant_algo": "W4A8_AWQ"}
    block_int8 = {"quant_algo": "BLOCK_INT8"}
    
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        
        # 注意力层使用BLOCK_INT8
        for attn_name in attn_names:
            quant_cfg["quantized_layers"][f"{prefix}.self_attn.{attn_name}"] = block_int8
        
        # 共享专家使用BLOCK_INT8
        for mlp_name in mlp_names:
            quant_cfg["quantized_layers"][f"{prefix}.mlp.shared_experts.{mlp_name}"] = block_int8
        
        # 前3层使用BLOCK_INT8，其余层使用W4A8_AWQ
        if layer_idx < 3:
            for mlp_name in mlp_names:
                quant_cfg["quantized_layers"][f"{prefix}.mlp.{mlp_name}"] = block_int8
        else:
            quant_cfg["quantized_layers"][f"{prefix}.mlp.experts"] = w4a8_awq
    
    # 保存quant_cfg.json
    quant_cfg_file = os.path.join(output_dir, "quant_cfg.json")
    with open(quant_cfg_file, 'w') as f:
        json.dump(quant_cfg, f, indent=4)
    print(f"quant_cfg.json saved to {quant_cfg_file}")
    
    # 创建hf_quant_config.json
    hf_quant_config = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None
        }
    }
    
    hf_quant_config_file = os.path.join(output_dir, "hf_quant_config.json")
    with open(hf_quant_config_file, 'w') as f:
        json.dump(hf_quant_config, f, indent=4)
    print(f"hf_quant_config.json saved to {hf_quant_config_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-w4-hf-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1")
    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_w4_hf_path, args.input_fp8_hf_path, args.model_name)
    print("done") 