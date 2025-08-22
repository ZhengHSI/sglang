#pragma once

/**
 * @file w4a8_grouped_mm_c3x.cuh
 * @brief Implementation of grouped GEMM operation with int4 and INT8 mixed
 * precision
 *
 * This file implements a grouped GEMM operation that multiplies INT8 matrices
 * (A) with quantized INT4 matrices (B), applying per-block scaling factors.
 * The implementation is optimized for NVIDIA Hopper GPUs, leveraging Tensor
 * Cores for mixed precision arithmetic.
 *
 * Key features:
 * - Supports grouped GEMM operations with multiple experts
 * - Uses INT8 for matrix A
 * - Uses INT4 quantization for matrix B with per-block scaling
 * - Implements preprocessing for INT4 encoding and scale packing
 * - Optimized for Hopper architecture with Tensor Core operations
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "w4a8_get_group_starts.cuh"

using namespace cute;

namespace {

// Type definitions
using MmaType = cutlass::int8_t;     // INT8 type
using QuantType = cutlass::int4b_t;        // 4-bit integer type
using ElementAccumulator = float;        // INT8 needs int32_t accumulator
using ElementScale = cutlass::bfloat16_t;  // Scale type
using ElementScalePacked = cutlass::Array<ElementScale, 4>;
using ElementC = cutlass::half_t;  // Default output type (FP16)
using ElementD = ElementC;         // Default output type (INT32)
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// Architecture-specific configurations
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
// constexpr int TileShapeK = 512;
// using TileShape = Shape<_128, _32, cute::Int<TileShapeK>>;
// using ClusterShape = Shape<_1, _1, _1>;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

// Transposed layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

// Alignments
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
struct cutlass_3x_w4a8_int8_group_gemm {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC_Transpose*,
      AlignmentC,
      ElementD,
      LayoutD_Transpose*,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,
      LayoutB_Transpose*,
      AlignmentB,
      MmaType,
      LayoutA_Transpose*,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  // Define the final kernel and GEMM operation types
  using GemmKernelScaleOnly =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;

  using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

/**
 * @brief Main function to run int4 * int8 grouped GEMM from PyTorch
 *
 * This function performs multiple GEMM operations in parallel where each
 * operation multiplies an INT8 matrix (A) with a quantized INT4 matrix (B),
 * applying per-channel scaling factors. It's designed for efficient execution
 * on NVIDIA Hopper GPUs, leveraging Tensor Cores for optimal performance with
 * mixed precision arithmetic.
 *
 * The function includes preprocessing steps for both INT4 tensors and scale
 * factors to ensure optimal performance and correct operation.
 *
 * @param d_tensors Output tensor D with shape [total_m, total_n]
 * @param a_tensors Tensor containing all A matrices (INT8) with shape
 * [total_m, K]
 * @param b_tensors Tensor containing all B matrices (int4 packed as int8) with
 * shape [E, N, K/2]
 * @param a_scales Tensor containing A matrix scale factors
 * @param b_scales Tensor containing B matrix scale factors with shape [E,
 * K//512, N*4]
 * @param expert_offsets Tensor containing expert offsets for determining group
 * boundaries (int32)
 * @param problem_sizes Tensor containing problem sizes with shape [num_experts,
 * 3] (M, N, K for each group) (int32)
 * @param a_strides Stride information for A tensors
 * @param b_strides Stride information for B tensors
 * @param d_strides Stride information for D tensors
 * @param s_strides Stride information for scale tensors
 * @param chunk_size Size of each chunk for scales (K / number of scale chunks)
 */
// template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
template <typename Gemm>
void cutlass_w4a8_int8_group_gemm_caller(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size) {
  //   using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  // Check inputs
  TORCH_CHECK(a_tensors.dim() == 2, "A tensor must be 2D");
  TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
  TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
  TORCH_CHECK(
      a_scales.dim() == 1 || a_scales.dim() == 2,
      "A Scale tensor must be 1D [1] or 2D [total_m, K//512] for per-token-group scaling");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  // Check tensor shapes
  TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (N, M, K)");
  TORCH_CHECK(b_tensors.size(0) == num_experts, "B tensor first dimension must match number of groups");
  TORCH_CHECK(b_scales.size(0) == num_experts, "Scale tensor first dimension must match number of groups");
  TORCH_CHECK(b_tensors.size(2) * 2 == a_tensors.size(1), "B tensor K/2 dimension must match A tensor K dimension");
  TORCH_CHECK(b_scales.size(1) == a_tensors.size(1) / 512, "Scale tensor second dimension must be K//512");
  TORCH_CHECK(b_scales.size(2) == 4 * b_tensors.size(1), "Scale tensor last dimension must be 4*N");
  if (a_scales.dim() == 2) {
    TORCH_CHECK(
        a_scales.size(0) == a_tensors.size(0),
        "Per-token-group a_scales first dim must equal total_m (rows of A)");
    // Allow a_scales to be either K//512 (matches b_scales) or K//128 (4x finer). In the latter case we will repack.
    TORCH_CHECK(
        a_scales.size(1) == b_scales.size(1) || a_scales.size(1) == 4 * b_scales.size(1),
        "Per-token-group a_scales second dim must equal K//512 or K//128 (4x K//512)");
  }

  // Check tensor types
  TORCH_CHECK(a_tensors.scalar_type() == torch::kInt8, "A tensor must be int8 type");
  TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32 type");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32 type");

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0;
  // If using per-token-group activation scales (handled inside mainloop via fused scales),
  // disable epilogue alpha_ptr to avoid double scaling.
  fusion_args.alpha_ptr = (a_scales.dim() == 2) ? nullptr : a_scales.data_ptr<float>();
  // fusion_args.alpha_ptr = nullptr;
  ;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());

  run_int4_int8_get_group_gemm_starts(
      expert_offsets,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      a_tensors,
      b_tensors,
      d_tensors,
      a_scales,
      b_scales);

  // Build fused per-token-group scales if requested: fused_scales[m, k_group, n_pack] = a_scales[m, k_group] * b_scales[e, k_group, n_pack]
  torch::Tensor fused_scales;
  torch::Tensor fused_scales_ptrs;
  const bool use_per_token_group = per_act_token && (a_scales.dim() == 2);
  if (use_per_token_group) {
    const int64_t total_m = a_tensors.size(0);
    const int64_t scale_k = b_scales.size(1);            // K//512 (as used by weight scales after packing)
    const int64_t packed = b_scales.size(2);             // N*4

    fused_scales = torch::empty({total_m, scale_k, packed}, b_scales.options());

    // Accumulate per-expert row ranges using problem_sizes[:, 1] (M per expert)
    auto problem_sizes_cpu = problem_sizes.to(torch::kCPU);
    int64_t m_offset = 0;
    for (int e = 0; e < num_experts; ++e) {
      int32_t m_e = problem_sizes_cpu[e][1].item<int32_t>();  // M for expert e
      if (m_e == 0) {
        continue;
      }
      // b_scales slice: [scale_k, packed] -> [1, scale_k, packed]
      auto b_seg = b_scales[e].unsqueeze(0);
      // Build activation scales aligned to [m_e, scale_k, packed]
      auto a_rows = a_scales.narrow(0, m_offset, m_e);
      torch::Tensor a_aligned; // [m_e, scale_k, packed]
      if (a_rows.size(1) == scale_k) {
        // a_scales already in K//512 granularity: [m_e, scale_k] -> [m_e, scale_k, 1] -> broadcast
        a_aligned = a_rows.unsqueeze(-1);
        if (a_aligned.scalar_type() != b_scales.scalar_type()) {
          a_aligned = a_aligned.to(b_scales.scalar_type());
        }
        fused_scales.narrow(0, m_offset, m_e).copy_(a_aligned * b_seg);
      } else {
        // a_scales in K//128 granularity: pack 4 consecutive groups into the N*4 lanes
        TORCH_CHECK(
            a_rows.size(1) == 4 * scale_k,
            "a_scales second dim must be K//512 or K//128 (4x K//512)");
        // reshape to [m_e, scale_k, 4]
        auto a_view = a_rows.view({m_e, scale_k, 4});
        // build lane indices [0..packed-1] % 4 on the same device
        auto lane_idx = torch::arange(packed, options_int).to(a_rows.device()).remainder(4).to(torch::kLong);
        // select along last dim -> [m_e, scale_k, packed]
        a_aligned = torch::index_select(a_view, /*dim=*/2, lane_idx);
        if (a_aligned.scalar_type() != b_scales.scalar_type()) {
          a_aligned = a_aligned.to(b_scales.scalar_type());
        }
        fused_scales.narrow(0, m_offset, m_e).copy_(a_aligned * b_seg);
      }
      m_offset += m_e;
    }

    // Build expert-wise pointers into fused_scales (device tensor of int64 addresses)
    fused_scales_ptrs = torch::empty(num_experts, options_int);
    auto fused_scales_ptrs_cpu = torch::empty(num_experts, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    const auto* base_ptr = reinterpret_cast<uint8_t*>(fused_scales.data_ptr());
    const int64_t row_bytes = static_cast<int64_t>(fused_scales.stride(0)) * static_cast<int64_t>(fused_scales.element_size());

    m_offset = 0;
    for (int e = 0; e < num_experts; ++e) {
      int32_t m_e = problem_sizes_cpu[e][1].item<int32_t>();
      const uint8_t* ptr_e = base_ptr + m_offset * row_bytes;
      fused_scales_ptrs_cpu.index_put_({e}, static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr_e)));
      m_offset += m_e;
    }
    fused_scales_ptrs.copy_(fused_scales_ptrs_cpu, /*non_blocking=*/true);
  }

  auto& scale_ptrs_to_use = use_per_token_group ? fused_scales_ptrs : b_scales_ptrs;

  arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      {static_cast<const QuantType**>(b_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
       static_cast<const MmaType**>(a_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideA*>(a_strides.data_ptr())
       ,static_cast<const ElementScalePacked**>(scale_ptrs_to_use.data_ptr()),
       static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
       static_cast<int>(chunk_size)
       },
      {fusion_args,
       nullptr,
       nullptr,
       static_cast<ElementD**>(out_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
      hw_info};

  // Instantiate and run GEMM
  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "GEMM implementation not supported");
  }

  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "GEMM initialization failed");
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "GEMM execution failed");
  }
}

}  // namespace
