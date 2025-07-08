import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import json

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

from kernel import weight_quant

def main(bf16_path, int8_path, model_name="deepseek-ai/DeepSeek-R1"):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)
    model_index_file = os.path.join(int8_path, "model.safetensors.index.json")
    config_file = os.path.join(int8_path, "config.json")
     
    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.safetensors"],
            local_dir=int8_path,
            local_dir_use_symlinks=False
        )
        print(f"model index file and config file downloaded to {int8_path}")

        # modify config.json and save it
        config = json.load(open(config_file))
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            quant_config.pop("fmt", None)
            quant_config["quant_method"] = "blockwise_int8"
            quant_config["weight_block_size"] = [
                128,
                128
            ]
            quant_config["activation_scheme"] = "dynamic"
        else:
            config["quantization_config"] = {
                "activation_scheme": "dynamic",
                "quant_method": "blockwise_int8",
                "weight_block_size": [
                    128,
                    128
                ]
            }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"config.json modified and saved to {config_file}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    scale_count = len([key for key in weight_map.keys() if key.endswith("_scale_inv")])
    
    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    quant_count = 0
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map:
                assert weight.element_size() == 2
                quant_count += 1
                int8_weight, scale_inv = weight_quant(weight)
                new_state_dict[weight_name] = int8_weight
                new_state_dict[scale_inv_name] = scale_inv
            else:
                new_state_dict[weight_name] = weight
        new_safetensor_file = os.path.join(int8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
    assert quant_count == scale_count
    print(f"{quant_count} weights are quantized.")
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1")
    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_int8_hf_path, args.model_name)
    print("done")
