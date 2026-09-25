/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/trace.h"

#include "cutlass/cuda_host_adapter.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// CUTLASS grouped epilogue with a warp-local transpose for short-K BF16 output.
// Other shapes and dtypes retain the official scalar output path.
template <
  class ElementC_,
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_, int TileTokens_
>
class MacroGroup32WideOutputEpilogue {
public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy = EpilogueSchedule_;

  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;

  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>; // prevents void ref breakages

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(cute::is_same_v<EpilogueSchedule, PtrArrayNoSmemWarpSpecialized> || cute::is_same_v<EpilogueSchedule, PtrArrayDefault> || cute::is_same_v<EpilogueSchedule, PtrArrayNoSmemWarpSpecializedTransposed>, "Incompatible epilogue schedule.");
  static_assert(rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage { alignas(128) uint16_t output[64 * TileTokens_]; };

  // This epilogue writes global vectors directly and has no TMA descriptors.
  struct TensorMapStorage {};

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const&,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  MacroGroup32WideOutputEpilogue(Params const& params_)
      : params(params_) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    // For Ptr-Array or Grouped Gemm we cannot determine if source is needed based on first beta.
    return true;
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      [[maybe_unused]] ResidueMNK,
      int thread_idx,
      [[maybe_unused]] char* smem_buffer)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    // Batches are managed by using appropriate pointers to C and D matrices
    const int32_t mock_L = 1;
    const int32_t mock_l_coord = 0;
    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

    // If scalar alpha/beta are provided, i.e., same alpha/beta applies to all batches/groups.
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups,
    // we get the correct alpha/beta values for the current batch/group using group index.
    ThreadEpilogueOp epilogue_op = ThreadEpilogueOp(params.thread, l_coord);

    if (epilogue_op.is_source_needed() && params.dC == nullptr) {
      // Beta value is non-zero while pointer to C is a nullptr
      assert(0);
    }

    auto [stride_c, stride_d] = [&, l = l_coord]() {
      if constexpr (!cute::is_same_v<InternalStrideC, StrideC>) {
        // If grouped gemm
        if (epilogue_op.is_source_needed()) {
            return make_tuple(
                detail::get_epilogue_stride<EpilogueSchedule>(params.dC[l]),
                detail::get_epilogue_stride<EpilogueSchedule>(params.dD[l])
            );
        }
        else {
          return make_tuple(
              InternalStrideC{},
              detail::get_epilogue_stride<EpilogueSchedule>(params.dD[l])
          );
        }
      }
      else {
        return make_tuple(
            detail::get_epilogue_stride<EpilogueSchedule>(params.dC),
            detail::get_epilogue_stride<EpilogueSchedule>(params.dD)
        );
      }
    }();

    // Represent the full output tensor
    ElementC const* ptr_C_l = nullptr;
    if (epilogue_op.is_source_needed()) {
      ptr_C_l = params.ptr_C[l_coord];
    }
    Tensor mC_mnl = make_tensor(make_gmem_ptr<GmemElementC>(ptr_C_l), make_shape(M,N,mock_L), stride_c);     // (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D[l_coord]), make_shape(M,N,mock_L), stride_d);     // (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)

    Tensor gC = gC_mnl(_,_,m_coord,n_coord, mock_l_coord);                                             // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord, mock_l_coord);                                             // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                                         // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC);                                                         // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(blk_shape_MNK), make_coord(m_coord, n_coord));         // (BLK_M,BLK_N)
    Tensor tCcD = thr_mma.partition_C(cD_mn);                                                      // (VEC,THR_M,THR_N)

    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_shape(M,N))) {
          tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
        }
      }
    }
    // source is not needed, avoid load
    else {
      // The operator requires channel count divisible by 16 and aligned output.
      // Each warp owns 16 channels; the XOR swizzle preserves disjoint stripes.
      if constexpr (cute::is_same_v<ElementD, cutlass::bfloat16_t> &&
                    size<0>(BlockShapeMNK{}) == 64 && size(TiledMma{}) == 128) {
        if (TileTokens_ == 32 || TileTokens_ == 128 || get<2>(problem_shape_mnkl) == 256) {
          auto* scratch = reinterpret_cast<SharedStorage*>(smem_buffer)->output;
          // Materialize lane-dependent output addresses only after the GEMM.
          // Volatile PTX reads the same CTA thread ID; it changes no arithmetic.
          uint32_t late_tid;
          asm volatile("mov.u32 %0, %%tid.x;" : "=r"(late_tid) :: "memory");
          int late_thread_idx = int(late_tid) % 128;
          auto late_thr_mma = tiled_mma.get_thread_slice(late_thread_idx);
          auto coords = late_thr_mma.partition_C(make_identity_tensor(take<0,2>(blk_shape_MNK)));
          // Use the SM90 matrix store to transpose BF16 fragments into the
          // existing XOR-swizzled scratch, preserving the vector global store.
          using StoreAtom = cute::Copy_Atom<cute::SM90_U16x8_STSM_T, ElementD>;
          auto store_copy = cute::make_tiled_copy_C(StoreAtom{}, tiled_mma);
          auto store_thread = store_copy.get_slice(late_thread_idx);
          auto converted = cute::make_tensor<ElementD>(accumulators.layout());
          using PairOp = cutlass::epilogue::thread::LinearCombination<
              ElementD, 2, ElementAccumulator, ElementCompute,
              ThreadEpilogueOp::kScale, ThreadEpilogueOp::kRound,
              typename ThreadEpilogueOp::ElementSource>;
          typename PairOp::Params pair_params;
          pair_params.alpha = params.thread.alpha;
          pair_params.beta = params.thread.beta;
          pair_params.alpha_ptr = params.thread.alpha_ptr;
          pair_params.beta_ptr = params.thread.beta_ptr;
          pair_params.alpha_ptr_array = params.thread.alpha_ptr_array;
          pair_params.beta_ptr_array = params.thread.beta_ptr_array;
          PairOp pair_op(pair_params, l_coord);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(accumulators); i += 2) {
            typename PairOp::FragmentAccumulator pair;
            pair[0] = accumulators(i);
            pair[1] = accumulators(i + 1);
            auto result = pair_op(pair);
            converted(i) = result[0];
            converted(i + 1) = result[1];
          }
          auto shared_layout = cute::composition(cute::Swizzle<3,3,3>{},
              cute::make_layout(cute::make_shape(cute::_64{},cute::Int<TileTokens_>{}),
                                cute::make_stride(cute::_1{},cute::_64{})));
          auto shared_output = cute::make_tensor(
              cute::make_smem_ptr(reinterpret_cast<ElementD*>(scratch)),shared_layout);
          auto source = store_thread.retile_S(converted);
          auto target = store_thread.partition_D(shared_output);
          cute::copy(store_copy,source,target);
          __syncwarp();
          const int lane = late_thread_idx % 32;
          const int channel = (late_thread_idx / 32) * 16 + (lane % 2) * 8;
          CUTLASS_PRAGMA_UNROLL
          for (int it = 0; it < TileTokens_ / 16; ++it) {
            const int token = it * 16 + lane / 2;
            const int global_channel = int(m_coord) * 64 + channel;
            const int global_token = int(n_coord) * TileTokens_ + token;
            if (global_channel < M && global_token < N) {
              uint4 data = *reinterpret_cast<const uint4*>(scratch + token * 64 + (channel ^ ((token & 7) * 8)));
              auto* dst = &mD_mnl(global_channel, global_token, 0);
              *reinterpret_cast<uint4*>(dst) = data;
            }
          }
          // Every warp owns disjoint channel stripes, including after swizzle.
          __syncwarp();
          return;
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_shape(M,N))) {
          tCgD(i) = epilogue_op(accumulators(i));
        }
      }
    }
  }

private:
  Params params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Keep the official store/pipeline adapter, but do not allocate output scratch
// a second time for its unused descriptor interface.
template <class EpilogueOp>
class MacroGroup32WideOutputAdapter
    : public detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp> {
  using Base = detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp>;
 public:
  using Base::Base;
  using TensorMapStorage = typename EpilogueOp::TensorMapStorage;

  template <bool... Flags>
  CUTLASS_DEVICE auto tensormaps_init(
      typename EpilogueOp::Params const&, TensorMapStorage&,
      int32_t, int32_t, int32_t = 0) {
    return static_cast<cute::TmaDescriptor*>(nullptr);
  }
  template <bool... Flags>
  CUTLASS_DEVICE auto load_init(
      typename EpilogueOp::Params const&, TensorMapStorage&, int32_t, int32_t) {
    return cute::make_tuple(static_cast<cute::TmaDescriptor*>(nullptr));
  }
  template <bool... Flags>
  CUTLASS_DEVICE auto store_init(
      typename EpilogueOp::Params const&, TensorMapStorage&,
      int32_t, int32_t, int32_t = 0) {
    return cute::make_tuple(static_cast<cute::TmaDescriptor*>(nullptr));
  }
  template <bool IsLoad, bool Wait = true, class ProblemShape>
  CUTLASS_DEVICE void tensormaps_perform_update(
      TensorMapStorage&, typename EpilogueOp::Params const&,
      cute::TmaDescriptor const*, ProblemShape, int32_t, int32_t = 0) {}
  template <bool IsLoad, bool Wait = true>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(
      TensorMapStorage&, cute::TmaDescriptor const*, int32_t = 0) {}
};

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
