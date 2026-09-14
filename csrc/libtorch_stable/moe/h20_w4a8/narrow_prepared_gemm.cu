// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <vector>
#include <tuple>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

// vllm includes
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/cutlass_extensions/torch_utils.hpp"
#include "libtorch_stable/cutlass_extensions/common.hpp"


#include "libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"


namespace h20_native_prepared_n8 {

using namespace cute;

// -------------------------------------------------------------------------------------
// Static configuration shared across all instantiations
// -------------------------------------------------------------------------------------
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group
using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;

constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;
static int constexpr PackFactor = 8;  // 8 int4 packed into int32

// A matrix configuration
using ElementA = MmaType;
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 /
    cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of
                                            // elements (up to 16 bytes)

// B matrix configuration
using ElementB = QuantType;  // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<
              ElementB>::value;  // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input
// layouts
using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// Need to pass a pointer type to make the 3rd dimension of Stride be _0
using StrideA =
    cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB =
    cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

// Define the CuTe layout for reoredered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in
// contiguous locations in global memory. It specifies the reordering within a
// single warp's fragment
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(
    LayoutAtomQuant{}, Layout<Shape<int, int, Int<1>>, StrideB>{}));

using ElementScale = cutlass::float_e4m3_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using ElementC =
    cutlass::bfloat16_t;  // Element type for C and D matrix operands
using LayoutC =
    cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<
              ElementC>::value;  // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

// D matrix configuration
using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator = float;     // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
using StageCountType =
    cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based
                                                // on the tile size

// per-channel and per-token scales for epilogue
using ElementSChannel = float;

template <class TileShape_MN, class ClusterShape_MNK, class KernelSchedule,
          class EpilogueSchedule>
struct W4A8GroupedGemmKernel {
  using TileShape =
      decltype(cute::append(TileShape_MN{}, cute::Int<TileShapeK>{}));
  using ClusterShape = ClusterShape_MNK;

  // per-channel, per-token scales epilogue
  using ChTokScalesEpilogue =
      typename vllm::c3x::ScaledEpilogueArray<ElementAccumulator, ElementD,
                                              TileShape>;
  using EVTCompute = typename ChTokScalesEpilogue::EVTCompute;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementSChannel, ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*, AlignmentC,
          ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type*,
          AlignmentD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  // =========================================================== MIXED INPUT
  // WITH SCALES
  // ===========================================================================
  // The Scale information must get paired with the operand that will be scaled.
  // In this example, B is scaled so we make a tuple of B's information and the
  // scale information.
  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>,
          LayoutB_Reordered*, AlignmentB, ElementA, LayoutA_Transpose*,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloopShuffled, CollectiveEpilogue>;

  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

  using StrideC = typename GemmKernelShuffled::InternalStrideC;
  using StrideD = typename GemmKernelShuffled::InternalStrideD;

  using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
  using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;
  using StrideS = typename CollectiveMainloopShuffled::StrideScale;
  using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

  // static asserts for passing in strides/layouts
  // pack to 2x int64
  static_assert(sizeof(StrideS) == 2 * sizeof(int64_t));
  // pack to 3xint32,
  static_assert(sizeof(LayoutB_Reordered) % sizeof(int32_t) == 0,
                "LayoutB_Reordered size must be divisible by 4 bytes");

  static void grouped_mm(torch::stable::Tensor& out_tensors,
                         const torch::stable::Tensor& a_tensors,
                         const torch::stable::Tensor& b_tensors,
                         const torch::stable::Tensor& a_scales,
                         const torch::stable::Tensor& b_scales,
                         const torch::stable::Tensor& b_group_scales,
                         const int64_t b_group_size,
                         const torch::stable::Tensor& expert_offsets,
                         const torch::stable::Tensor& problem_sizes_torch,
                         const torch::stable::Tensor& a_strides,
                         const torch::stable::Tensor& b_strides,
                         const torch::stable::Tensor& c_strides,
                         const torch::stable::Tensor& group_scale_strides,
                         const torch::stable::Tensor& prepared_ptrs) {
    auto device = a_tensors.device();
    auto device_id = device.index();
    const torch::stable::accelerator::DeviceGuard device_guard(device_id);
    auto stream = get_current_cuda_stream(device_id);

    int num_experts = static_cast<int>(problem_sizes_torch.size(0));
    int n = static_cast<int>(b_tensors.size(1));
    int k = static_cast<int>(b_tensors.size(2)) * PackFactor;

    STD_TORCH_CHECK(prepared_ptrs.scalar_type() ==
                    torch::headeronly::ScalarType::Long);
    STD_TORCH_CHECK(prepared_ptrs.is_cuda() && prepared_ptrs.is_contiguous());
    STD_TORCH_CHECK(prepared_ptrs.dim() == 2 && prepared_ptrs.size(0) == 6 &&
                    prepared_ptrs.size(1) >= num_experts &&
                    prepared_ptrs.size(1) % 2 == 0);
    auto ptrs = static_cast<int64_t*>(prepared_ptrs.data_ptr());
    auto ptr_stride = prepared_ptrs.stride(0);

    // construct args
    using Args = typename GemmShuffled::Arguments;
    using MainloopArguments = typename GemmKernelShuffled::MainloopArguments;
    using EpilogueArguments = typename GemmKernelShuffled::EpilogueArguments;
    Args arguments;

    ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
        static_cast<ProblemShape::UnderlyingProblemShape*>(
            problem_sizes_torch.data_ptr());
    ProblemShape prob_shape{num_experts, problem_sizes_as_shapes, nullptr};

    // SwapAB so B operands come first
    MainloopArguments mainloop_arguments{
        static_cast<const QuantType**>(static_cast<void*>(ptrs + 1 * ptr_stride)),
        static_cast<LayoutB_Reordered*>(b_strides.data_ptr()),
        static_cast<const MmaType**>(static_cast<void*>(ptrs + 0 * ptr_stride)),
        static_cast<StrideA*>(a_strides.data_ptr()),
        static_cast<const cutlass::Array<ElementScale, 8>**>(
            static_cast<void*>(ptrs + 5 * ptr_stride)),
        static_cast<StrideS*>(group_scale_strides.data_ptr()),
        static_cast<int>(b_group_size)};

    EpilogueArguments epilogue_arguments{
        // since we are doing SwapAB the channel scales comes first, then token
        // scales
        ChTokScalesEpilogue::prepare_args(  // see ScaledEpilogueArray
            static_cast<const ElementAccumulator**>(
                static_cast<void*>(ptrs + 4 * ptr_stride)),  // per-channel
            static_cast<const ElementAccumulator**>(
                static_cast<void*>(ptrs + 3 * ptr_stride)),  // per-token
            true, true),
        nullptr,                                       // C
        static_cast<StrideC*>(c_strides.data_ptr()),   // C
        static_cast<ElementD**>(static_cast<void*>(ptrs + 2 * ptr_stride)),  // D
        static_cast<StrideC*>(c_strides.data_ptr())    // D
    };

    static const cutlass::KernelHardwareInfo hw_info{
        device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            device_id)};

    arguments = Args{cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape,
                     mainloop_arguments, epilogue_arguments, hw_info};

    // Allocate workspace
    size_t workspace_size = GemmShuffled::get_workspace_size(arguments);
    torch::stable::Tensor workspace = torch::stable::empty(
        workspace_size, torch::headeronly::ScalarType::Byte, std::nullopt,
        device);

    // Run GEMM
    GemmShuffled gemm;
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
    CUTLASS_CHECK(gemm.run(stream));
  }
};

// ----------------------------------------------------------------------------

using Coop = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using CoopEpi = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using K128 = W4A8GroupedGemmKernel<Shape<_128, _8>, Shape<_1, _1, _1>, Coop, CoopEpi>;
using K256 = W4A8GroupedGemmKernel<Shape<_256, _8>, Shape<_1, _1, _1>, Coop, CoopEpi>;

void mm(torch::stable::Tensor& out,
        const torch::stable::Tensor& a, const torch::stable::Tensor& b,
        const torch::stable::Tensor& token, const torch::stable::Tensor& channel,
        const torch::stable::Tensor& scale, int64_t group_size,
        const torch::stable::Tensor& offsets, const torch::stable::Tensor& problems,
        const torch::stable::Tensor& astride, const torch::stable::Tensor& bstride,
        const torch::stable::Tensor& cstride, const torch::stable::Tensor& sstride,
        const torch::stable::Tensor& ptrs, int64_t tile) {
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(problems.scalar_type() == torch::headeronly::ScalarType::Int);
  STD_TORCH_CHECK(problems.is_contiguous() && problems.dim() == 2 && problems.size(1) == 3);
  STD_TORCH_CHECK(problems.size(0) > 0 && problems.size(0) <= 256);
  if (tile == 128) {
    K128::grouped_mm(out, a, b, token, channel, scale, group_size, offsets,
                    problems, astride, bstride, cstride, sstride, ptrs);
  }
  else if (tile == 256) {
    K256::grouped_mm(out, a, b, token, channel, scale, group_size, offsets,
                    problems, astride, bstride, cstride, sstride, ptrs);
  }
  else { STD_TORCH_CHECK(false, "Unsupported selected H20 tile"); }
}

STABLE_TORCH_LIBRARY(h20_native_prepared_n8, m) {
  m.def("mm(Tensor! out, Tensor a, Tensor b, Tensor token, Tensor channel, "
        "Tensor scale, int group_size, Tensor offsets, Tensor problems, "
        "Tensor astride, Tensor bstride, Tensor cstride, Tensor sstride, "
        "Tensor ptrs, int tile) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_native_prepared_n8, CUDA, m) {
  m.impl("mm", TORCH_BOX(&mm));
}
}  // namespace h20_native_prepared_n8
