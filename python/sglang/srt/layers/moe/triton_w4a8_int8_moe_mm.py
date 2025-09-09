import torch
import triton
import triton.language as tl
from typing import Optional


# @triton.jit
# def _w4a8_int8_moe_kernel(
#     # Pointers to matrices
#     a_ptr, b_ptr, c_ptr,
#     # Scale pointers
#     a_scale_ptr, b_scale_ptr,
#     # Matrix dimensions
#     M, N, K,
#     # Strides
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     # Scale strides
#     stride_a_scale_m, stride_a_scale_k,
#     stride_b_scale_k, stride_b_scale_n,
#     # Block sizes
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     # Group size for quantization
#     GROUP_SIZE: tl.constexpr,
#     # Whether to use per-token-group scaling for activations
#     USE_PER_TOKEN_GROUP_SCALE: tl.constexpr,
# ):
#     """
#     Triton kernel for W4A8 INT8 MoE matrix multiplication with per-token-group scaling.
    
#     This kernel performs:
#     - INT4 weight dequantization with per-group scaling (group size = 128)
#     - INT8 activation processing with per-token-group scaling
#     - Matrix multiplication with accumulation in FP32
#     - Output in FP16
    
#     Per-token-group scaling means each token has independent scales for each group of 128 elements.
#     """
#     # Program ID and grid calculation
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
#     # Calculate which block this program handles
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     # Block offsets
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
#     # Initialize accumulator
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # Loop over K dimension in chunks
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # Current K offset and range
#         k_offset = k * BLOCK_SIZE_K
#         offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        
#         # Load A (INT8 activations)
#         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
#         a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
#         # For packed INT4 weights, we need to handle the packing correctly
#         # B is stored as [N, K//2] where each element contains 2 INT4 values
#         k_packed = offs_k // 2  # Index into packed dimension
#         k_within_packed = offs_k % 2  # Which INT4 within the packed element
        
#         # Load packed B weights
#         b_ptrs = b_ptr + (k_packed[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#         b_mask = (k_packed[:, None] < (K // 2)) & (offs_bn[None, :] < N)
#         b_packed = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
#         # Unpack INT4 from INT8
#         # Extract low nibble (bits 0-3) and high nibble (bits 4-7)
#         low_nibble = b_packed & 0x0F
#         high_nibble = (b_packed >> 4) & 0x0F
        
#         # Select appropriate nibble based on k posit ion
#         b_unpacked = tl.where(
#             k_within_packed[:, None] == 0,
#             low_nibble,
#             high_nibble
#         )
        
#         # Convert unsigned 4-bit to signed 4-bit range [-8, 7]
#         b_unpacked = tl.where(b_unpacked > 7, b_unpacked - 16, b_unpacked)
        
#         # Load B scales for dequantization (per-group scaling)
#         group_idx = offs_k // GROUP_SIZE
#         b_scale_ptrs = b_scale_ptr + (group_idx[:, None] * stride_b_scale_k + offs_bn[None, :] * stride_b_scale_n)
#         b_scale_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
#         b_scales = tl.load(b_scale_ptrs, mask=b_scale_mask, other=1.0)
        
#         # Dequantize B weights
#         b_dequant = b_unpacked.to(tl.float32) * b_scales.to(tl.float32)
        
#         # Load A scales and dequantize A activations
#         if USE_PER_TOKEN_GROUP_SCALE:
#             # Per-token-group scaling: each token has independent scales for each group
#             # a_scales shape: [M, K//GROUP_SIZE]
            
#             # Calculate group indices for all K positions in this block
#             group_idx_k = offs_k // GROUP_SIZE
            
#             # Initialize dequantized activations
#             a_dequant = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
            
#             # Load scales for each group in this K block
#             # We need to handle the case where different K positions belong to different groups
#             a_scale_ptrs = a_scale_ptr + (offs_am[:, None] * stride_a_scale_m + group_idx_k[None, :] * stride_a_scale_k)
#             a_scale_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
#             a_scales_k = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)
            
#             # Apply per-token-group scaling
#             a_dequant = a.to(tl.float32) * a_scales_k.to(tl.float32)
#         else:
#             # Per-tensor scaling (single scale for all activations)
#             a_scales = tl.load(a_scale_ptr)
#             a_dequant = a.to(tl.float32) * a_scales.to(tl.float32)
        
#         # Perform matrix multiplication and accumulate
#         accumulator += tl.dot(a_dequant, b_dequant)
    
#     # Convert accumulator to output type and store
#     c = accumulator.to(tl.float16)
    
#     # Output pointers
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)

# # @triton.jit
# # def _w4a8_int8_moe_kernel(
# #     # Pointers to matrices
# #     a_ptr, b_ptr, c_ptr,
# #     # Scale pointers
# #     a_scale_ptr, b_scale_ptr,
# #     # Matrix dimensions
# #     M, N, K,
# #     # Strides
# #     stride_am, stride_ak,
# #     stride_bk, stride_bn,
# #     stride_cm, stride_cn,
# #     # Scale strides
# #     stride_a_scale_m, stride_a_scale_k,
# #     stride_b_scale_k, stride_b_scale_n,
# #     # Block sizes
# #     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
# #     # Group size for quantization
# #     GROUP_SIZE: tl.constexpr,
# #     # Whether to use per-token-group scaling for activations
# #     USE_PER_TOKEN_GROUP_SCALE: tl.constexpr,
# # ):
# #     """
# #     Triton kernel for W4A8 INT8 MoE matrix multiplication with per-token-group scaling.
    
# #     This kernel performs:
# #     - INT4 weight dequantization with per-group scaling (group size = 128)
# #     - INT8 activation processing with per-token-group scaling
# #     - Matrix multiplication with accumulation in FP32
# #     - Output in FP16
    
# #     Per-token-group scaling means each token has independent scales for each group of 128 elements.
# #     """
# #     # Program ID and grid calculation
# #     pid = tl.program_id(axis=0)
# #     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# #     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
# #     # Calculate which block this program handles
# #     pid_m = pid // num_pid_n
# #     pid_n = pid % num_pid_n

# #     # Block offsets
# #     # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
# #     # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
# #     offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
# #     offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

# #     # Initialize accumulator
# #     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
# #     # Loop over K dimension in chunks
# #     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
# #         # Current K offset and range
# #         k_offset = k * BLOCK_SIZE_K
# #         offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        
# #         # Load A (INT8 activations)
# #         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
# #         a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
# #         a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
# #         # For packed INT4 weights, we need to handle the packing correctly
# #         # B is stored as [N, K//2] where each element contains 2 INT4 values
# #         k_packed = offs_k // 2  # Index into packed dimension
# #         k_within_packed = offs_k % 2  # Which INT4 within the packed element
        
# #         # Load packed B weights
# #         b_ptrs = b_ptr + (k_packed[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
# #         b_mask = (k_packed[:, None] < (K // 2)) & (offs_bn[None, :] < N)
# #         b_packed = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
# #         # Unpack INT4 from INT8
# #         # Extract low nibble (bits 0-3) and high nibble (bits 4-7)
# #         low_nibble = b_packed & 0x0F
# #         high_nibble = (b_packed >> 4) & 0x0F
        
# #         # Select appropriate nibble based on k position
# #         b_unpacked = tl.where(
# #             k_within_packed[:, None] == 0,
# #             low_nibble,
# #             high_nibble
# #         )
        
# #         # Convert unsigned 4-bit to signed 4-bit range [-8, 7]
# #         b_unpacked = tl.where(b_unpacked > 7, b_unpacked - 16, b_unpacked)
        
# #         # Perform INT8 matrix multiplication first (before dequantization)
# #         # A is already INT8, B is converted from INT4 to INT8
# #         # tl.dot automatically uses INT32 accumulator for INT8 inputs
# #         block_result_int32 = tl.dot(a, b_unpacked)
        
# #         # Convert to float32 for dequantization
# #         block_result_fp32 = block_result_int32.to(tl.float32)
        
# #         # Load scales for dequantization
# #         # For per-group scaling, we need to get the scale for the current K group
# #         group_idx = k // (GROUP_SIZE // BLOCK_SIZE_K)  # Which group this K block belongs to
        
# #         # Load B scales (per-group scaling for weights)
# #         # B scales are [K//GROUP_SIZE, N], we need the scale for this group
# #         b_scale_ptrs = b_scale_ptr + (group_idx * stride_b_scale_k + offs_bn[None, :] * stride_b_scale_n)
# #         b_scale_mask = offs_bn[None, :] < N
# #         b_scales_group = tl.load(b_scale_ptrs, mask=b_scale_mask, other=1.0)
        
# #         # Load A scales based on scaling mode
# #         if USE_PER_TOKEN_GROUP_SCALE:
# #             # Per-token-group scaling: each token has independent scales for each group
# #             # a_scales shape: [M, K//GROUP_SIZE]
# #             a_scale_ptrs = a_scale_ptr + (offs_am[:, None] * stride_a_scale_m + group_idx * stride_a_scale_k)
# #             a_scale_mask = offs_am[:, None] < M
# #             a_scales_group = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)
# #         else:
# #             # Per-tensor scaling (single scale for all activations)
# #             a_scales_single = tl.load(a_scale_ptr)
# #             # Broadcast to match the output shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
# #             a_scales_group = tl.broadcast_to(a_scales_single, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        
# #         # Apply dequantization to the block result
# #         # The result needs to be scaled by both A and B scales
# #         # For matrix multiplication: C = A * B, so C_dequant = (A_scale * B_scale) * C_int
# #         combined_scales = a_scales_group.to(tl.float32) * b_scales_group.to(tl.float32)
# #         dequantized_block = block_result_fp32 * combined_scales
        
# #         # Accumulate the dequantized result
# #         accumulator += dequantized_block
    
# #     # Convert accumulator to output type and store
# #     c = accumulator.to(tl.float16)
    
# #     # Output pointers
# #     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
# #     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
# #     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
# #     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
# #     tl.store(c_ptrs, c, mask=c_mask)

# # def triton_w4a8_int8_moe_mm(
# #     d: torch.Tensor,
# #     a: torch.Tensor,
# #     b: torch.Tensor,
# #     a_scales: torch.Tensor,
# #     b_scales: torch.Tensor,
# #     expert_offsets: torch.Tensor,
# #     problem_sizes: torch.Tensor,
# #     a_strides: torch.Tensor,
# #     b_strides: torch.Tensor,
# #     d_strides: torch.Tensor,
# #     s_strides: torch.Tensor,
# #     chunk_size: int = 128,
# #     topk: int = 8,
# # ):
#     """
#     Triton implementation of W4A8 INT8 MoE matrix multiplication with per-token-group scaling.
    
#     This function replaces the CUTLASS implementation with a Triton kernel that:
#     1. Handles INT4 weight dequantization with per-group scaling
#     2. Processes INT8 activations with per-token-group scaling
#     3. Performs grouped GEMM for multiple experts
#     4. Outputs results in FP16
    
#     Args:
#         d: Output tensor [total_m, total_n] in FP16
#         a: Input activations [total_m, K] in INT8
#         b: Packed INT4 weights [E, N, K//2] in INT8 (packed)
#         a_scales: Activation scales [total_m, K//chunk_size] for per-token-group scaling
#         b_scales: Weight scales [E, K//chunk_size, N*4] in BF16
#         expert_offsets: Expert start offsets [E] in INT32
#         problem_sizes: Problem dimensions [E, 3] (N, M, K) in INT32
#         a_strides: A matrix strides [E, 3] in INT64
#         b_strides: B matrix strides [E, 3] in INT64
#         d_strides: D matrix strides [E, 3] in INT64
#         s_strides: Scale strides [E, 3] in INT64
#         chunk_size: Quantization group size (default: 128)
#         topk: Number of top-k experts (default: 8)
#     """
    
#     # Validate inputs
#     assert a.dtype == torch.int8, f"Expected INT8 activations, got {a.dtype}"
#     assert b.dtype == torch.int8, f"Expected packed INT4 weights, got {b.dtype}"
#     assert d.dtype == torch.float16, f"Expected FP16 output, got {d.dtype}"
    
#     # Determine if using per-token-group scaling
#     use_per_token_group_scale = a_scales.dim() > 1
    
#     if use_per_token_group_scale:
#         print(f"Using per-token-group scaling: a_scales shape = {a_scales.shape}")
#     else:
#         print(f"Using per-tensor scaling: a_scales shape = {a_scales.shape}")
    
#     # Get device
#     device = a.device
    
#     # Process each expert
#     num_experts = problem_sizes.shape[0]
    
#     for expert_idx in range(num_experts):
#         # Get problem size for this expert
#         N, M, K = problem_sizes[expert_idx].tolist()
        
#         if M == 0:  # Skip if no tokens for this expert
#             continue
            
#         # Get expert offset
#         offset = expert_offsets[expert_idx].item()
        
#         # Get pointers for this expert
#         a_expert = a[offset:offset+M, :]
#         b_expert = b[expert_idx, :, :]
#         d_expert = d[offset:offset+M, :]
        
#         # Get scales for this expert
#         if use_per_token_group_scale:
#             a_scale_expert = a_scales[offset:offset+M, :]
#         else:
#             a_scale_expert = a_scales
            
#         b_scale_expert = b_scales[expert_idx, :, :]
        
#         # Reshape b_scales from [K//chunk_size, N*4] to [K//chunk_size, N, 4] and take first scale
#         # This assumes the 4 scales are for different quantization schemes, we use the first one
#         K_groups = K // chunk_size
#         b_scale_reshaped = b_scale_expert.view(K_groups, N, 4)[:, :, 0]  # [K_groups, N]
        
#         # Launch kernel
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
        
#         _w4a8_int8_moe_kernel[grid](
#             # Pointers
#             a_expert, b_expert, d_expert,
#             a_scale_expert, b_scale_reshaped,
#             # Dimensions
#             M, N, K,
#             # # Strides for A (row-major)
#             # K, 1,
#             # # Strides for B (column-major, packed)
#             # 1, K // 2,
#             # # Strides for C (row-major)
#             # N, 1,
#             # Strides for A [M, K] - Dynamically get strides
#             a_expert.stride(0), a_expert.stride(1),
#             # Strides for B [N, K//2] - Dynamically get strides
#             b_expert.stride(0), b_expert.stride(1),
#             # Strides for C [M, N]
#             d_expert.stride(0), d_expert.stride(1),
#             # Scale strides
#             a_scale_expert.stride(0) if use_per_token_group_scale else 0,
#             a_scale_expert.stride(1) if use_per_token_group_scale else 0,
#             b_scale_reshaped.stride(0),
#             b_scale_reshaped.stride(1),
#             # Block sizes
#             BLOCK_SIZE_M=32,  # Smaller block size for better per-token-group handling
#             BLOCK_SIZE_N=64,
#             BLOCK_SIZE_K=32,
#             # Group size
#             GROUP_SIZE=chunk_size,
#             # Per-token-group scaling flag
#             USE_PER_TOKEN_GROUP_SCALE=use_per_token_group_scale,
#         )



# gpt修改

@triton.jit
def _w4a8_int8_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Scale pointers
    a_scale_ptr, b_scale_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bn, stride_bk,   # 注意顺序改了，匹配 `.stride()` 定义
    stride_cm, stride_cn,
    # Scale strides
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_k, stride_b_scale_n,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Group size for quantization
    GROUP_SIZE: tl.constexpr,
    # Whether to use per-token-group scaling for activations
    USE_PER_TOKEN_GROUP_SCALE: tl.constexpr,
):
    """
    Triton kernel for W4A8 INT8 MoE matrix multiplication with per-token-group scaling.
    """

    # Program ID and grid calculation
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 直接用 idx，不做 % wrap-around
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k_iter * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

        # Load activation A (int8)
        # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        # a = tl.load(a_ptrs, mask=a_mask, other=0)
        a = tl.load(
            a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0
        )

        # -------- B packed int4 load --------
        k_packed = offs_k // 2
        k_within_packed = offs_k % 2

        # 按照 b_expert.stride() 定义访问  
        # b 是 [N, K//2] 或 [K//2, N] 都能支持  
        b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + k_packed[:, None] * stride_bk)
        b_mask = (offs_bn[None, :] < N) & (k_packed[:, None] < (K // 2))
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)

        # Unpack INT4 → INT8
        low_nibble = b_packed & 0x0F
        high_nibble = (b_packed >> 4) & 0x0F
        b_unpacked = tl.where(k_within_packed[:, None] == 0, low_nibble, high_nibble)
        b_unpacked = tl.where(b_unpacked > 7, b_unpacked - 16, b_unpacked)  # signed int4

        # Dot product in int32
        block_result_int32 = tl.dot(a, b_unpacked)
        block_result_fp32 = block_result_int32.to(tl.float32)

        # -------- Load scales --------
        group_idx = k_iter // (GROUP_SIZE // BLOCK_SIZE_K)

        b_scale_ptrs = b_scale_ptr + (group_idx * stride_b_scale_k + offs_bn[None, :] * stride_b_scale_n)
        b_scale_mask = offs_bn[None, :] < N
        b_scales_group = tl.load(b_scale_ptrs, mask=b_scale_mask, other=1.0)

        if USE_PER_TOKEN_GROUP_SCALE:
            a_scale_ptrs = a_scale_ptr + (offs_am[:, None] * stride_a_scale_m + group_idx * stride_a_scale_k)
            a_scale_mask = offs_am[:, None] < M
            a_scales_group = tl.load(a_scale_ptrs, mask=a_scale_mask, other=1.0)
        else:
            a_scales_single = tl.load(a_scale_ptr)
            a_scales_group = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), a_scales_single, dtype=tl.float32)

        # Apply dequantization
        combined_scales = a_scales_group.to(tl.float32) * b_scales_group.to(tl.float32)
        accumulator += block_result_fp32 * combined_scales

    # Store FP16 result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_w4a8_int8_moe_mm(
    d: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    d_strides: torch.Tensor,
    s_strides: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 8,
):
    """
    Triton implementation of W4A8 INT8 MoE matrix multiplication with per-token-group scaling.
    """

    assert a.dtype == torch.int8
    assert b.dtype == torch.int8
    assert d.dtype == torch.float16

    use_per_token_group_scale = a_scales.dim() > 1
    num_tokens_total = a.shape[0]
    num_experts = problem_sizes.shape[0]
    for expert_idx in range(num_experts):
        N, M, K = problem_sizes[expert_idx].tolist()
        if M == 0:
            continue

        offset = expert_offsets[expert_idx].item()
        # 越界保护
        if M <= 0:
            continue
        
        # offset 越界保护
        if offset < 0 or offset >= num_tokens_total or offset + M > num_tokens_total:
            # 保留输出 layout，不跑 kernel，把对应区域清零
            d[offset:offset+M, :].zero_()
            continue

        a_expert = a[offset:offset+M, :]
        b_expert = b[expert_idx, :, :]
        d_expert = d[offset:offset+M, :]

        if use_per_token_group_scale:
            a_scale_expert = a_scales[offset:offset+M, :]
        else:
            a_scale_expert = a_scales

        b_scale_expert = b_scales[expert_idx, :, :]
        K_groups = K // chunk_size
        b_scale_reshaped = b_scale_expert.view(K_groups, N, 4)[:, :, 0]

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        _w4a8_int8_moe_kernel[grid](
            a_expert, b_expert, d_expert,
            a_scale_expert, b_scale_reshaped,
            M, N, K,
            # A strides
            a_expert.stride(0), a_expert.stride(1),
            # B strides: 注意现在顺序是 (stride_bn, stride_bk)  
            b_expert.stride(0), b_expert.stride(1),
            # C strides
            d_expert.stride(0), d_expert.stride(1),
            # Scale strides
            a_scale_expert.stride(0) if use_per_token_group_scale else 0,
            a_scale_expert.stride(1) if use_per_token_group_scale else 0,
            b_scale_reshaped.stride(0),
            b_scale_reshaped.stride(1),
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE=chunk_size,
            USE_PER_TOKEN_GROUP_SCALE=use_per_token_group_scale,
        )


# @triton.autotune(
#     configs=[
#         # 较小 tile: 适合小 M/N
#         triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32,  'GROUP_SIZE': 128, 'USE_PER_TOKEN_GROUP_SCALE': 1}, num_stages=2, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE': 128, 'USE_PER_TOKEN_GROUP_SCALE': 1}, num_stages=2, num_warps=4),

#         # Hopper 推荐 tile：适合大算子
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE': 128, 'USE_PER_TOKEN_GROUP_SCALE': 1}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE': 128, 'USE_PER_TOKEN_GROUP_SCALE': 1}, num_stages=4, num_warps=8),

#         # 更大 K tile: Hopper L2 吃得下
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64,  'GROUP_SIZE': 128, 'USE_PER_TOKEN_GROUP_SCALE': 1}, num_stages=4, num_warps=8),
#     ],
#     key=['M', 'N', 'K']
# )

# @triton.jit
# def _w4a8_int8_moe_kernel(
#     a_ptr, b_ptr, c_ptr,
#     a_scale_ptr, b_scale_ptr,
#     M, N, K,
#     stride_am, stride_ak,
#     stride_bn, stride_bk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_b_scale_k, stride_b_scale_n,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE: tl.constexpr,
#     USE_PER_TOKEN_GROUP_SCALE: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
#     pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

#     for k_iter in range(0, num_k_tiles):
#         offs_k = k_iter * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

#         # A load
#         a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
#         a = tl.load(
#             a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
#             mask=a_mask, other=0
#         )

#         # B load (int4 packed in int8)
#         k_packed = offs_k // 2
#         k_within_packed = offs_k % 2
#         b_mask = (offs_n[None, :] < N) & (k_packed[:, None] < (K // 2))
#         b_packed = tl.load(
#             b_ptr + (offs_n[None, :] * stride_bn + k_packed[:, None] * stride_bk),
#             mask=b_mask, other=0
#         )
#         low = b_packed & 0x0F
#         high = (b_packed >> 4) & 0x0F
#         b_unpacked = tl.where(k_within_packed[:, None] == 0, low, high)
#         b_unpacked = tl.where(b_unpacked > 7, b_unpacked - 16, b_unpacked)

#         # int8 dot int4 → int32
#         acc_int32 = tl.dot(a, b_unpacked)
#         acc_fp32 = acc_int32.to(tl.float32)

#         # scale load & apply
#         group_idx = k_iter // (GROUP_SIZE // BLOCK_SIZE_K)
#         b_scale = tl.load(
#             b_scale_ptr + (group_idx * stride_b_scale_k + offs_n[None, :] * stride_b_scale_n),
#             mask=(offs_n[None, :] < N),
#             other=1.0
#         )
#         if USE_PER_TOKEN_GROUP_SCALE:
#             a_scale = tl.load(
#                 a_scale_ptr + (offs_m[:, None] * stride_a_scale_m + group_idx * stride_a_scale_k),
#                 mask=(offs_m[:, None] < M),
#                 other=1.0
#             )
#         else:
#             a_scale_val = tl.load(a_scale_ptr)
#             a_scale = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), a_scale_val, dtype=tl.float32)

#         accumulator += acc_fp32 * (a_scale * b_scale)

#     # store
#     c = accumulator.to(tl.float16)
#     tl.store(
#         c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
#         c,
#         mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
#     )


# def triton_w4a8_int8_moe_mm(
#     d, a, b, a_scales, b_scales,
#     expert_offsets, problem_sizes,
#     a_strides, b_strides, d_strides, s_strides,

#     chunk_size=128, topk=8
# ):
#     """
#     改进版: 单次 kernel 遍历一个 expert 的 GEMM，支持 cuda graph capture
#     """
#     assert a.dtype == torch.int8
#     assert b.dtype == torch.int8
#     assert d.dtype == torch.float16

#     use_per_token_group_scale = a_scales.dim() > 1
#     num_experts = problem_sizes.shape[0]

#     for expert_idx in range(num_experts):
#         N, M, K = problem_sizes[expert_idx].tolist()
#         offset = expert_offsets[expert_idx].item()
#         # 不做 Python 分支退出，mask 在 kernel 内部做
#         a_expert = a[offset:offset + M, :]
#         b_expert = b[expert_idx, :, :]
#         d_expert = d[offset:offset + M, :]

#         if use_per_token_group_scale:
#             a_scale_expert = a_scales[offset:offset + M, :]
#         else:
#             a_scale_expert = a_scales

#         b_scale_expert = b_scales[expert_idx, :, :]
#         K_groups = max(1, K // chunk_size)
#         b_scale_reshaped = b_scale_expert.view(K_groups, N, 4)[:, :, 0]

#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )

#         # _w4a8_int8_moe_kernel[grid](
#         #     a_expert, b_expert, d_expert,
#         #     a_scale_expert, b_scale_reshaped,
#         #     M, N, K,
#         #     a_expert.stride(0), a_expert.stride(1),
#         #     b_expert.stride(0), b_expert.stride(1),
#         #     d_expert.stride(0), d_expert.stride(1),
#         #     a_scale_expert.stride(0) if use_per_token_group_scale else 0,
#         #     a_scale_expert.stride(1) if use_per_token_group_scale else 0,
#         #     b_scale_reshaped.stride(0),
#         #     b_scale_reshaped.stride(1),
#         #     BLOCK_SIZE_M=64,
#         #     BLOCK_SIZE_N=64,
#         #     BLOCK_SIZE_K=32,
#         #     GROUP_SIZE=chunk_size,
#         #     USE_PER_TOKEN_GROUP_SCALE=use_per_token_group_scale,
#         #     num_stages=2,
#         #     num_warps=4
#         # )
#         _w4a8_int8_moe_kernel[grid](
#             a_expert, b_expert, d_expert,
#             a_scale_expert, b_scale_reshaped,
#             M, N, K,
#             a_expert.stride(0), a_expert.stride(1),
#             b_expert.stride(0), b_expert.stride(1),
#             d_expert.stride(0), d_expert.stride(1),
#             a_scale_expert.stride(0) if use_per_token_group_scale else 0,
#             a_scale_expert.stride(1) if use_per_token_group_scale else 0,
#             b_scale_reshaped.stride(0),
#             b_scale_reshaped.stride(1),
#             # 这里 BLOCK_SIZE_*, GROUP_SIZE 等由 autotune 决定
#         )

