#!/usr/bin/env python3
"""
Example usage of cutlass_w4a8_int8_moe_mm kernel

This example demonstrates how to use the new INT8 + INT4 MoE GEMM kernel
for efficient matrix multiplication with quantized weights and activations.
"""

import torch
import time
from sgl_kernel import cutlass_w4a8_int8_moe_mm, get_cutlass_w4a8_moe_mm_data


def create_moe_data(num_experts, batch_size, seq_len, hidden_size, intermediate_size, topk=2):
    """Create test data for MoE GEMM"""
    device = torch.device("cuda")
    
    # Create input activations (INT8)
    a_tensors = torch.randint(-128, 127, (batch_size * seq_len * topk, hidden_size), 
                             dtype=torch.int8, device=device)
    
    # Create weight matrices (packed INT4 as INT8)
    b_tensors = torch.randint(-8, 7, (num_experts, intermediate_size, hidden_size // 2), 
                             dtype=torch.int8, device=device)
    
    # Create scale tensors
    a_scales = torch.ones(1, dtype=torch.bfloat16, device=device)
    b_scales = torch.randn(num_experts, hidden_size // 512, intermediate_size * 4, 
                          dtype=torch.bfloat16, device=device)
    
    # Create expert offsets and problem sizes
    expert_offsets = torch.arange(0, batch_size * seq_len * topk, 
                                 batch_size * seq_len * topk // num_experts, 
                                 dtype=torch.int32, device=device)
    expert_offsets = torch.cat([expert_offsets, torch.tensor([batch_size * seq_len * topk], 
                                                           dtype=torch.int32, device=device)])
    
    problem_sizes = torch.tensor([[batch_size * seq_len * topk // num_experts, 
                                  intermediate_size, hidden_size]] * num_experts, 
                                dtype=torch.int32, device=device)
    
    # Create stride tensors
    a_strides = torch.tensor([hidden_size] * num_experts, dtype=torch.int64, device=device)
    b_strides = torch.tensor([hidden_size // 2] * num_experts, dtype=torch.int64, device=device)
    d_strides = torch.tensor([intermediate_size] * num_experts, dtype=torch.int64, device=device)
    s_strides = torch.tensor([intermediate_size * 4] * num_experts, dtype=torch.int64, device=device)
    
    return (a_tensors, b_tensors, a_scales, b_scales, expert_offsets, 
            problem_sizes, a_strides, b_strides, d_strides, s_strides)


def benchmark_int8_moe_mm(num_experts=8, batch_size=4, seq_len=8, hidden_size=4096, 
                         intermediate_size=7168, topk=2, num_runs=100):
    """Benchmark the INT8 MoE GEMM kernel"""
    
    print(f"Benchmarking INT8 MoE GEMM with:")
    print(f"  num_experts: {num_experts}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  topk: {topk}")
    print(f"  num_runs: {num_runs}")
    
    # Create test data
    (a_tensors, b_tensors, a_scales, b_scales, expert_offsets, 
     problem_sizes, a_strides, b_strides, d_strides, s_strides) = create_moe_data(
        num_experts, batch_size, seq_len, hidden_size, intermediate_size, topk
    )
    
    # Create output tensor
    d_tensors = torch.zeros(batch_size * seq_len * topk, intermediate_size, 
                           dtype=torch.float16, device=a_tensors.device)
    
    # Warm up
    for _ in range(10):
        cutlass_w4a8_int8_moe_mm(
            d_tensors,
            a_tensors,
            b_tensors,
            a_scales,
            b_scales,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            d_strides,
            s_strides,
            chunk_size=128,
            topk=topk,
        )
    
    # Synchronize
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        cutlass_w4a8_int8_moe_mm(
            d_tensors,
            a_tensors,
            b_tensors,
            a_scales,
            b_scales,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            d_strides,
            s_strides,
            chunk_size=128,
            topk=topk,
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = num_runs / total_time
    
    # Calculate FLOPs
    total_flops = (batch_size * seq_len * topk * hidden_size * intermediate_size * num_experts) * num_runs
    flops_per_sec = total_flops / total_time
    tflops = flops_per_sec / 1e12
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Average time per run: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} runs/second")
    print(f"  Performance: {tflops:.2f} TFLOPS")
    
    return avg_time, tflops


def compare_with_fp8_moe_mm():
    """Compare INT8 vs FP8 MoE GEMM performance"""
    print("Comparing INT8 vs FP8 MoE GEMM performance...")
    
    # Test parameters
    num_experts = 8
    batch_size = 4
    seq_len = 8
    hidden_size = 4096
    intermediate_size = 7168
    topk = 2
    num_runs = 50
    
    # Benchmark INT8 version
    print("\n" + "="*50)
    print("INT8 MoE GEMM Benchmark")
    print("="*50)
    int8_time, int8_tflops = benchmark_int8_moe_mm(
        num_experts, batch_size, seq_len, hidden_size, intermediate_size, topk, num_runs
    )
    
    # Note: FP8 version would require similar benchmarking
    # For now, we'll just show the INT8 results
    print(f"\nINT8 MoE GEMM achieved {int8_tflops:.2f} TFLOPS")
    print("FP8 comparison would require implementing similar benchmarking for the FP8 kernel")


def main():
    """Main function to run examples"""
    print("SGLang INT8 MoE GEMM Example")
    print("="*40)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return
    
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    
    # Run basic benchmark
    print("\n" + "="*50)
    print("Basic INT8 MoE GEMM Benchmark")
    print("="*50)
    benchmark_int8_moe_mm()
    
    # Run comparison
    compare_with_fp8_moe_mm()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 