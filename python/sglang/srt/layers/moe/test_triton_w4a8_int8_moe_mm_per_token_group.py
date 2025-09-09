import pytest
import torch
import sys
import traceback

# Import our Triton implementation
from triton_w4a8_int8_moe_mm import (
    triton_w4a8_int8_moe_mm,
)
# Try to import sglang quantization function
from sglang.srt.layers.quantization.int8_kernel import sglang_per_token_group_quant_int8

def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    """Pack int4 values into int8 tensor (CUTLASS format)."""
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)

    return packed_tensor.to(torch.int8)

def pack_interleave(num_experts, ref_weight, ref_scale):
    """Pack weight and scale tensors for Triton format."""
    # Use our Triton packing function
    w_q = pack_int4_values_to_int8(ref_weight)
    
    # Prepare scales in Triton format [E, K//chunk_size, N*4]
    K_groups = ref_scale.shape[2]  # K//128
    N = ref_scale.shape[1]
    
    w_scale = torch.zeros(num_experts, K_groups, N * 4, dtype=ref_scale.dtype, device=ref_scale.device)
    for i in range(num_experts):
        # Expand ref_scale from [N, K//chunk_size] to [K//chunk_size, N*4]
        scale_expanded = ref_scale[i].t().unsqueeze(-1).expand(-1, -1, 4).reshape(K_groups, N * 4)
        w_scale[i] = scale_expanded
    
    return w_q, w_scale

# def pack_interleave(num_experts, ref_weight, ref_scale):
#     n, k = ref_weight.shape[1], ref_weight.shape[2]

#     weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
#     w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
#     w_q = w_q.contiguous()

#     scale_interleaved = ref_scale.reshape(
#         ref_scale.shape[0], ref_scale.shape[1], (ref_scale.shape[2] // 4), 4
#     )  # [E, N, K/4, 4]
#     scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
#     scale_interleaved = scale_interleaved.reshape(
#         ref_scale.shape[0], ref_scale.shape[2] // 4, ref_scale.shape[1] * 4
#     )  # [E, K/4, N*4]
#     w_scale = scale_interleaved.contiguous()

#     return w_q, w_scale

def ref_per_token_group_gemm_fp(a, a_scales, w, w_scale, num_experts, experts_selection_result, scale_granularity):
    """Reference implementation for floating point computation (no quantization involved)."""
    dtype = torch.bfloat16
    batch_size, k = a.shape
    _, n, _ = w.shape
    c_ref = torch.zeros(batch_size, n, dtype=dtype, device=a.device)
    
    if scale_granularity == "K512":
        group_size = 512
    elif scale_granularity == "K128":
        group_size = 128
    else:
        raise ValueError("scale_granularity must be 'K512' or 'K128'")
    
    for i in range(num_experts):
        token_idx = torch.where(experts_selection_result == i)[0]
        if len(token_idx) == 0:
            continue
        
        # Get tokens for this expert (original floating point, no scaling)
        a_tok = a[token_idx]  # [num_tokens, k]
        
        # Prepare dequantized weights
        ref_w_scale_repeat = w_scale[i].repeat_interleave(128, dim=1).to(torch.float32)
        ref_w_scaled = (w[i].to(torch.float32) * ref_w_scale_repeat).to(dtype)
        
        # Direct floating point matrix multiplication (no activation scaling)
        c_tok = torch.matmul(a_tok.to(dtype), ref_w_scaled.t())
        c_ref[token_idx] = c_tok
    
    return c_ref


def ref_per_token_group_gemm(a_q, a_scales, w, w_scale, num_experts, experts_selection_result, scale_granularity):
    """Reference implementation for per-token-group quantized computation."""
    dtype = torch.bfloat16
    batch_size, k = a_q.shape
    _, n, _ = w.shape
    c_ref = torch.zeros(batch_size, n, dtype=dtype, device=a_q.device)
    
    if scale_granularity == "K512":
        group_size = 512
    elif scale_granularity == "K128":
        group_size = 128
    else:
        raise ValueError("scale_granularity must be 'K512' or 'K128'")
    
    num_groups = k // group_size
    
    for i in range(num_experts):
        token_idx = torch.where(experts_selection_result == i)[0]
        if len(token_idx) == 0:
            continue
        
        # Get quantized tokens and scales for this expert
        a_q_tok = a_q[token_idx]  # [num_tokens, k] - quantized activations
        a_scales_tok = a_scales[token_idx]  # [num_tokens, num_groups]
        
        # Prepare weight scales (repeat to match activation group granularity)
        ref_w_scale_repeat = w_scale[i].repeat_interleave(128, dim=1).to(torch.float32)
        ref_w_scaled = (w[i].to(torch.float32) * ref_w_scale_repeat).to(dtype)
        
        # For each token, dequantize and apply per-group scaling
        c_tok = torch.zeros(len(token_idx), n, dtype=dtype, device=a_q.device)
        
        for t_idx, (token_q, token_scales) in enumerate(zip(a_q_tok, a_scales_tok)):
            # Reshape quantized token to groups: [num_groups, group_size]
            token_q_grouped = token_q.view(num_groups, group_size)
            
            # Dequantize: apply per-group scaling to convert int8 back to float
            token_dequantized = torch.zeros_like(token_q, dtype=dtype)
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = (g + 1) * group_size
                token_dequantized[start_idx:end_idx] = token_q_grouped[g].to(dtype) * token_scales[g]
            
            # Compute matrix multiplication with dequantized activation
            c_tok[t_idx] = torch.matmul(token_dequantized, ref_w_scaled.t())
        
        c_ref[token_idx] = c_tok
    
    return c_ref


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("scale_granularity", ["K128"])
def test_triton_per_token_group_single_expert(batch_size, scale_granularity):
    """Test Triton per-token-group activation scaling with single expert."""
    # Test parameters
    num_experts = 1
    m = batch_size  # batch size
    k = 512  # input dimension
    n = 1024  # output dimension
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    debug = False

    print(f"\nTesting Triton per-token-group with batch_size={batch_size}, scale_granularity={scale_granularity}")

    # Create input tensors
    if debug:
        a = torch.ones(m, k, dtype=torch.bfloat16, device=device)
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device=device)
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
    else:
        a = torch.randn(m, k, dtype=dtype, device=device)
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        affine_coeff = 0.005
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device=device)
            * affine_coeff
        )

    # Pack weights for Triton format
    w_triton, w_scale_triton = pack_interleave(num_experts, ref_w, ref_w_scale)

    # Create expert offsets and problem sizes
    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)

    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = torch.full((num_experts, 3), k // 2, device=device, dtype=torch.int64)
    s_strides = torch.full((num_experts, 3), n * 4, device=device, dtype=torch.int64)

    # Quantize input with per-token-group scales
    a_q, a_scales = sglang_per_token_group_quant_int8(a, 128)
    
    # Create output tensor
    c_triton = torch.empty((m, n), dtype=torch.float16, device=device)
    
    triton_w4a8_int8_moe_mm(
        c_triton,
        a_q,
        w_triton,
        a_scales,  # Per-token-group scales
        w_scale_triton,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        128,
        8,
    )
    c_triton = c_triton.to(dtype)

    # Reference implementation
    experts_selection_result = torch.full((m,), 0, device=device)
    # c_ref_fp = ref_per_token_group_gemm_fp(
    #     a, a_scales, ref_w, ref_w_scale, num_experts, experts_selection_result, scale_granularity
    # )
    c_ref = ref_per_token_group_gemm_fp(
        a, a_scales, ref_w, ref_w_scale, num_experts, experts_selection_result, scale_granularity
    )

    # Compare Triton with reference
    try:
        torch.testing.assert_close(c_triton, c_ref, rtol=0.01, atol=0.1)
        print(f"  SUCCESS: Triton vs reference test passed for {scale_granularity}")
        print(f"    Ref tensor: {c_ref.flatten()[:10]}")
        print(f"    Triton tensor: {c_triton.flatten()[:10]}")
    except AssertionError as e:
        print(f"  FAILURE: Triton vs reference tensors are NOT close.")
        print(f"    Ref tensor: {c_ref.flatten()[:10]}")
        print(f"    Triton tensor: {c_triton.flatten()[:10]}")
        print(
            f"    Max absolute difference: {torch.max(torch.abs(c_triton.to(c_ref.dtype) - c_ref))}"
        )
        print(
            f"    Mean absolute difference: {torch.mean(torch.abs(c_triton.to(c_ref.dtype) - c_ref))}"
        )
        print(f"    AssertionError: {e}")
        raise


@pytest.mark.parametrize("batch_size", [4, 8, 16])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("num_experts", [2, 4])
@pytest.mark.parametrize("scale_granularity", ["K128"])
def test_triton_per_token_group_multi_experts(batch_size, k, n, num_experts, scale_granularity):
    """Test Triton per-token-group activation scaling with multiple experts."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"

    print(
        f"\nTesting Triton per-token-group multi-expert: batch_size={batch_size}, k={k}, "
        f"n={n}, num_experts={num_experts}, scale_granularity={scale_granularity}"
    )

    # Create input tensors
    a = torch.randn(batch_size, k, dtype=dtype, device=device)
    ref_w = torch.randint(-8, 8, (num_experts, n, k), dtype=torch.int8, device=device)
    affine_coeff = 0.005
    ref_w_scale = (
        torch.randn(num_experts, n, k // 128, dtype=dtype, device=device) * affine_coeff
    )

    # Pack weights for Triton format
    w_triton, w_scale_triton = pack_interleave(num_experts, ref_w, ref_w_scale)

    # Random expert selection
    experts_selection_result = torch.randint(0, num_experts, (batch_size,), device=device)
    permutation = torch.argsort(experts_selection_result)
    expert_token_counts = torch.bincount(experts_selection_result, minlength=num_experts)

    # Create problem sizes and offsets for active experts
    problem_sizes = []
    for i in range(num_experts):
        problem_sizes.append([n, expert_token_counts[i].item(), k])
    problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device=device)

    expert_offsets = []
    offset = 0
    for i in range(num_experts):
        expert_offsets.append(offset)
        offset += problem_sizes[i][1].item()
    expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=device)

    # Permute input and quantize
    a_perm = a[permutation]
    a_q_perm, a_scales_perm = sglang_per_token_group_quant_int8(a_perm, 128)


    # Create stride tensors
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = torch.full((num_experts, 3), k // 2, device=device, dtype=torch.int64)
    s_strides = torch.full((num_experts, 3), n * 4, device=device, dtype=torch.int64)

    # Test Triton
    c_perm = torch.empty((batch_size, n), dtype=torch.float16, device=device)
    try:
        triton_w4a8_int8_moe_mm(
            c_perm,
            a_q_perm,
            w_triton,
            a_scales_perm.float(),  # Per-token-group scales
            w_scale_triton,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            c_strides,
            s_strides,
            128,
            8,
        )
        
        # Un-permute the result
        c_triton = torch.empty_like(c_perm, dtype=dtype)
        c_perm = c_perm.to(dtype)
        c_triton[permutation] = c_perm
        
        print(f"  SUCCESS: Triton multi-expert test completed")
        
    except Exception as e:
        print(f"  FAILURE: Triton multi-expert test failed: {e}")
        traceback.print_exc()
        raise

    # Reference implementation
    c_ref = ref_per_token_group_gemm_fp(
        a_perm[torch.argsort(permutation)], a_scales_perm[torch.argsort(permutation)], 
        ref_w, ref_w_scale, num_experts, experts_selection_result, scale_granularity
    )

    # Compare results
    try:
        torch.testing.assert_close(c_triton, c_ref, rtol=0.001, atol=0.1)
        print(f"  SUCCESS: Triton multi-expert vs reference test passed")
    except AssertionError as e:
        print(f"  FAILURE: Triton multi-expert vs reference tensors are NOT close.")
        print(
            f"    Max absolute difference: {torch.max(torch.abs(c_triton.to(c_ref.dtype) - c_ref))}"
        )
        print(
            f"    Mean absolute difference: {torch.mean(torch.abs(c_triton.to(c_ref.dtype) - c_ref))}"
        )
        print(f"    AssertionError: {e}")
        raise


if __name__ == "__main__":
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
