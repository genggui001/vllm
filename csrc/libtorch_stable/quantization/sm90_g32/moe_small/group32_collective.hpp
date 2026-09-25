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
// Modifications copyright contributors to the vLLM project.
#pragma once

namespace h20_exact_groupwise {
using namespace cute;

#ifndef H20_GROUP32_SCALE_PREFETCH
#define H20_GROUP32_SCALE_PREFETCH 1
#endif
#ifndef H20_GROUP32_OVERLAP
#define H20_GROUP32_OVERLAP 1
#endif

template <class Base>
struct ExactGroup32Mainloop : Base {
  using WeightScale = cutlass::bfloat16_t;
  using typename Base::MainloopPipeline;
  using typename Base::PipelineState;
  using typename Base::SmemLayoutA;
  using typename Base::SmemLayoutB;
  using typename Base::SwappedSmemLayoutAtomA;
  using typename Base::SwappedSmemLayoutAtomB;
  using typename Base::SwappedSmemCopyAtomA;
  using typename Base::SwappedSmemCopyAtomB;
  using typename Base::TiledMma;
  using typename Base::InternalSwappedStrideA;
  using typename Base::RealSwappedElementA;
  using typename Base::DispatchPolicy;
  using typename Base::TileShape;
  using Utils = cutlass::gemm::collective::detail::MixedInputUtils<Base>;
  static constexpr int NumThreadsPerWarpGroup = 128;
  static constexpr int ScaleRows = size<1>(TileShape{});
  static constexpr int ScaleWarpGroups = size(TiledMma{}) / 128;
  static constexpr bool UseAuxiliaryScale = H20_GROUP32_SCALE_PREFETCH;
  static constexpr int NumProducerThreadEvents = Base::NumProducerThreadEvents +
      (UseAuxiliaryScale ? 64 : 0);
  static constexpr int ScaleStages = UseAuxiliaryScale ? DispatchPolicy::Stages : 2;
  struct ScaleStorage {
    alignas(16) float4 activation[ScaleWarpGroups][ScaleStages][ScaleRows];
    alignas(128) WeightScale weight[ScaleWarpGroups][ScaleStages][4][64];
  };
  struct EmptyScaleStorage {};
  struct TensorStorage : Base::TensorStorage {
    [[no_unique_address]] cute::conditional_t<H20_GROUP32_SCALE_PREFETCH,
        ScaleStorage, EmptyScaleStorage> group_scale_storage;
    alignas(8) uint64_t scale_empty[ScaleStages];
  };

  struct ScaleArguments {
    const float* activation = nullptr;
    const WeightScale* weight = nullptr;
    const int* offsets = nullptr;
    int channels = 0, rows = 0;
    int activation_stride = 0, weight_stride = 0;
    int activation_group32 = 0, weight_group_shift = 0;
    bool weight_group_major = false;
    alignas(64) CUtensorMap weight_map{};
  };
  struct Arguments : Base::Arguments { ScaleArguments group_scales; };
  struct Params : Base::Params { ScaleArguments group_scales; };

  CUTLASS_DEVICE static float4 load_scale_quad(const float* ptr, bool group32) {
    if (!group32) {
      const float value = __ldg(ptr);
      return make_float4(value, value, value, value);
    }
    if ((reinterpret_cast<uintptr_t>(ptr) & 15) == 0) {
      return __ldg(reinterpret_cast<const float4*>(ptr));
    }
    return make_float4(__ldg(ptr), __ldg(ptr + 1),
                      __ldg(ptr + 2), __ldg(ptr + 3));
  }

  CUTLASS_DEVICE static float4 load_weight_quad(const WeightScale* ptr, bool group32) {
    float x=float(ptr[0]);
    return group32 ? make_float4(x,float(ptr[1]),float(ptr[2]),float(ptr[3]))
                   : make_float4(x,x,x,x);
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape shapes,
                                        Arguments const& args, void* workspace) {
    Params p;
    static_cast<typename Base::Params&>(p) =
        Base::to_underlying_arguments(shapes, args, workspace);
    p.group_scales = args.group_scales;
    return p;
  }

  template <class... TMs, class ProblemShape>
  CUTLASS_DEVICE void tensormaps_perform_update(
      typename Base::TensorMapStorage& storage, Params const& params,
      cute::tuple<TMs...> const& maps, ProblemShape shape, int32_t expert) {
    // The descriptor copy uses ordinary shared reads from every producer lane.
    // Finish those reads before the elected lane rewrites it for another group.
    __syncwarp();
    Base::tensormaps_perform_update(storage, params, maps, shape, expert);
  }

  template <class LoadInputs, class KIterator, class BlockCoord, class ProblemShape>
  CUTLASS_DEVICE void load_auxiliary(
      Params const& params, MainloopPipeline pipeline, PipelineState state,
      LoadInputs const&, BlockCoord const& coord, KIterator k_iter,
      int k_tiles, int lane, uint32_t, TensorStorage& storage,
      ProblemShape problem_shape, int expert) {
    const auto scales = params.group_scales;
    const int row_base = int(get<1>(coord)) * ScaleRows;
    const int channel_base = int(get<0>(coord)) * size<0>(TileShape{});
    const int begin = max(0, min(scales.rows, scales.offsets[expert]));
    const int rows = int(get<1>(problem_shape));
    for (int kt = 0; kt < k_tiles; ++kt, ++state, ++k_iter) {
      const int bank = state.index();
      cutlass::arch::ClusterBarrier::wait(&storage.scale_empty[bank], state.phase());
      pipeline.producer_acquire(state);
      const int group = int(*k_iter) * 4;
      CUTLASS_PRAGMA_UNROLL
      for (int wg = 0; wg < ScaleWarpGroups; ++wg) {
        auto& ss = storage.group_scale_storage;
        for (int row = lane; row < ScaleRows; row += 32) {
          float4& dst = ss.activation[wg][bank][row];
          if (row_base + row < rows) {
            const float* src = scales.activation +
                int64_t(begin + row_base + row) * scales.activation_stride +
                group * scales.activation_group32;
            if (scales.activation_group32 && (reinterpret_cast<uintptr_t>(src)&15)==0) {
              SM80_CP_ASYNC_CACHEALWAYS<uint4>::copy(
                  *reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4&>(dst));
            } else dst = load_scale_quad(src, scales.activation_group32);
          } else dst = make_float4(0.f,0.f,0.f,0.f);
        }
        if (lane==0) {
          auto* ready=pipeline.producer_get_barrier(state);
          cutlass::arch::ClusterTransactionBarrier::expect_transaction(ready,4*64*2);
          cute::SM90_TMA_LOAD_3D::copy(&params.group_scales.weight_map,ready,0,
              &ss.weight[wg][bank][0][0],channel_base+wg*64,group,expert);
        }
      }
      // Async payload completion and ordinary shared writes each publish an
      // arrival. The independent local empty barrier protects stage reuse.
      cp_async_fence();
      auto* barrier = pipeline.producer_get_barrier(state);
      cutlass::arch::cpasync_barrier_arrive_noinc(barrier);
      cutlass::arch::ClusterBarrier::arrive(barrier);
    }
  }

  template <class FrgTensorC, class BlockCoord, class ProblemShape>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                          FrgTensorC& accum, int k_tile_count, int thread_idx,
                          TensorStorage& shared_tensors, Params const& mainloop_params,
                          BlockCoord const& blk_coord, ProblemShape problem_shape,
                          int expert) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SwappedSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<SwappedSmemCopyAtomA>,
      "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SwappedSmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = cutlass::canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;


    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);                                      // (BLK_M,BLK_K,PIPE)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)


    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCrA_mma = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = [&]{
      if constexpr (not is_layout<InternalSwappedStrideA>::value) {
        // Make register tensor with MMA layout
        return make_fragment_like<RealSwappedElementA>(tCrA_mma);
      }
      else {
        // Make register tensor matching smem layout, converter will take care of de-swizzling
        return make_tensor_like<RealSwappedElementA>(tCsA(_,_,_,Int<0>{}));
      }
    }();
    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);                                        // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);                                  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);                                  // (CPY,CPY_M,CPY_K)

    // Partition of thread -> shared and thread -> RF
    auto partitioned_extra_info = Utils::partition_extra_mma_info(mma_thread_slice, shared_tensors);
    auto copy_partitions_extra_info = Utils::retile_extra_mma_info(tiled_mma, partitioned_extra_info, warp_group_thread_idx);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));                                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //


    auto coordinates = mma_thread_slice.partition_C(
        make_identity_tensor(take<0,2>(TileShape{})));
    auto partial = make_fragment_like(accum);
    auto partial_next = make_fragment_like(accum);
    clear(accum);
    clear(partial);
    if constexpr (H20_GROUP32_OVERLAP) clear(partial_next);
    const auto scales = mainloop_params.group_scales;
    const int channel_base = int(get<0>(blk_coord)) * size<0>(TileShape{});
    const int row_base = int(get<1>(blk_coord)) * size<1>(TileShape{});
    const int expert_rows = int(get<1>(problem_shape));
    const int global_row_base = max(0, min(scales.rows, scales.offsets[expert]));
    constexpr int KBlocks = size<2>(tCrA_load);
    static_assert(KBlocks == size<2>(TileShape{}) / 32);
    constexpr int AccumSize = decltype(size(accum))::value;
    static_assert(AccumSize == size<1>(TileShape{}) / 2);
    const int channel0 = channel_base + get<0>(coordinates(0));
    const int channel1 = channel_base + get<0>(coordinates(2));
    const int64_t weight_base0 =
        (int64_t(expert) * scales.channels + channel0) * scales.weight_stride;
    const int64_t weight_base1 =
        (int64_t(expert) * scales.channels + channel1) * scales.weight_stride;
    auto weight_quad = [&](int channel, int group) {
      if (channel >= scales.channels) return make_float4(0.f,0.f,0.f,0.f);
      if (!scales.weight_group_major) {
        return load_weight_quad(scales.weight +
            (int64_t(expert) * scales.channels + channel) * scales.weight_stride +
            (group >> scales.weight_group_shift), scales.weight_group_shift == 0);
      }
      const WeightScale* ptr = scales.weight + int64_t(expert) *
          scales.weight_stride * scales.channels + channel;
      return make_float4(
          float(ptr[((group + 0) >> scales.weight_group_shift) * scales.channels]),
          float(ptr[((group + 1) >> scales.weight_group_shift) * scales.channels]),
          float(ptr[((group + 2) >> scales.weight_group_shift) * scales.channels]),
          float(ptr[((group + 3) >> scales.weight_group_shift) * scales.channels]));
    };
    auto prefetch_scales = [](int) {};
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int kt = 0; kt < k_tile_count; ++kt) {
      auto token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, token);
      const int stage = smem_pipe_read.index();
      if constexpr (H20_GROUP32_SCALE_PREFETCH && !UseAuxiliaryScale) {
        if (kt + 1 < k_tile_count) cp_async_wait<1>();
        else cp_async_wait<0>();
        cutlass::arch::NamedBarrier::sync(128, uint32_t(warp_group_idx));
      }
      // Load the staged K128 tile before applying its four K32 transforms.
      CUTLASS_PRAGMA_UNROLL
      for (int load_k = 0; load_k < KBlocks; ++load_k) {
        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,
            partitioned_extra_info, copy_partitions_extra_info, load_k, stage);
      }
      constexpr bool VectorScales = !H20_GROUP32_SCALE_PREFETCH;
      float4 activation_quads[AccumSize / 2];
      float4 weight_quad0, weight_quad1;
      if constexpr (VectorScales) {
        const int ai = kt * KBlocks * scales.activation_group32;
        const int wi = (kt * KBlocks) >> scales.weight_group_shift;
        weight_quad0 = weight_quad(channel0, kt * KBlocks);
        weight_quad1 = weight_quad(channel1, kt * KBlocks);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < AccumSize / 4; ++j) {
          const int row0 = row_base + get<1>(coordinates(j * 4));
          const int row1 = row_base + get<1>(coordinates(j * 4 + 1));
          const int64_t ai0 = int64_t(global_row_base + row0) *
              scales.activation_stride + ai;
          const int64_t ai1 = int64_t(global_row_base + row1) *
              scales.activation_stride + ai;
          activation_quads[j * 2] = row0 < expert_rows ? load_scale_quad(
              scales.activation + ai0, scales.activation_group32) :
              make_float4(0.f, 0.f, 0.f, 0.f);
          activation_quads[j * 2 + 1] = row1 < expert_rows ? load_scale_quad(
              scales.activation + ai1, scales.activation_group32) :
              make_float4(0.f, 0.f, 0.f, 0.f);
        }
      }
      auto issue = [&](int kb, auto& target) {
        Utils::dequantize_A_kblock(tCrA_load, tCrA_mma, partitioned_extra_info, kb);
        warpgroup_fence_operand(target);
        warpgroup_arrive();
        gemm(tiled_mma, tCrA_mma(_,_,kb), tCrB(_,_,kb,stage), target);
        warpgroup_commit_batch();
      };
      if constexpr (H20_GROUP32_OVERLAP) issue(0, partial);
      CUTLASS_PRAGMA_UNROLL
      for (int kb = 0; kb < KBlocks; ++kb) {
        auto& current_partial = H20_GROUP32_OVERLAP && (kb & 1) ? partial_next : partial;
        if constexpr (H20_GROUP32_OVERLAP) {
          if (kb + 1 < KBlocks) {
            auto& next_partial = (kb & 1) ? partial : partial_next;
            issue(kb + 1, next_partial);
            warpgroup_wait<1>();
          } else {
            warpgroup_wait<0>();
          }
          warpgroup_fence_operand(current_partial);
        } else {
          issue(kb, current_partial);
        }
        const int group = kt * KBlocks + kb;
        const int activation_index = group * scales.activation_group32;
        const int weight_index = group >> scales.weight_group_shift;
        float ws0, ws1;
        if constexpr (H20_GROUP32_SCALE_PREFETCH) {
          ws0 = shared_tensors.group_scale_storage.weight[warp_group_idx][UseAuxiliaryScale ? stage : (kt&1)][kb][channel0-channel_base-warp_group_idx*64];
          ws1 = shared_tensors.group_scale_storage.weight[warp_group_idx][UseAuxiliaryScale ? stage : (kt&1)][kb][channel1-channel_base-warp_group_idx*64];
        } else {
          ws0 = reinterpret_cast<const float*>(&weight_quad0)[kb];
          ws1 = reinterpret_cast<const float*>(&weight_quad1)[kb];
        }
        float activation_scales[AccumSize / 2];
        // CLayout_64xN: values (2,2,N/8), strides (64,8,512).
        // Four adjacent values share two token scales and two channel scales.
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(accum) / 4; ++j) {
          const int row0 = row_base + get<1>(coordinates(j * 4));
          const int row1 = row_base + get<1>(coordinates(j * 4 + 1));
          const int64_t ai0 = int64_t(global_row_base + row0) *
              scales.activation_stride + activation_index;
          const int64_t ai1 = int64_t(global_row_base + row1) *
              scales.activation_stride + activation_index;
          if constexpr (H20_GROUP32_SCALE_PREFETCH) {
            activation_scales[j * 2] = reinterpret_cast<const float*>(
                &shared_tensors.group_scale_storage.activation[warp_group_idx][UseAuxiliaryScale ? stage : (kt&1)][row0-row_base])[kb];
            activation_scales[j * 2 + 1] = reinterpret_cast<const float*>(
                &shared_tensors.group_scale_storage.activation[warp_group_idx][UseAuxiliaryScale ? stage : (kt&1)][row1-row_base])[kb];
          } else {
            activation_scales[j * 2] = reinterpret_cast<const float*>(&activation_quads[j * 2])[kb];
            activation_scales[j * 2 + 1] = reinterpret_cast<const float*>(&activation_quads[j * 2 + 1])[kb];
          }
        }
        if constexpr (!H20_GROUP32_OVERLAP) {
          warpgroup_wait<0>();
          warpgroup_fence_operand(current_partial);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(accum) / 4; ++j) {
          const float as0 = activation_scales[j * 2];
          const float as1 = activation_scales[j * 2 + 1];
          accum(j * 4) = __fadd_rn(accum(j * 4),
              __fmul_rn(__fmul_rn(current_partial(j * 4), as0), ws0));
          accum(j * 4 + 1) = __fadd_rn(accum(j * 4 + 1),
              __fmul_rn(__fmul_rn(current_partial(j * 4 + 1), as1), ws0));
          accum(j * 4 + 2) = __fadd_rn(accum(j * 4 + 2),
              __fmul_rn(__fmul_rn(current_partial(j * 4 + 2), as0), ws1));
          accum(j * 4 + 3) = __fadd_rn(accum(j * 4 + 3),
              __fmul_rn(__fmul_rn(current_partial(j * 4 + 3), as1), ws1));
        }
      }
      if constexpr (UseAuxiliaryScale) {
        // Scale stages are local to this CTA. Every reader releases its own
        // accesses instead of relying on the matrix multicast release mask.
        cutlass::arch::ClusterBarrier::arrive(&shared_tensors.scale_empty[stage]);
      }
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
      if constexpr (H20_GROUP32_SCALE_PREFETCH && !UseAuxiliaryScale) {
        cutlass::arch::NamedBarrier::sync(128, uint32_t(warp_group_idx));
        if (kt + 2 < k_tile_count) prefetch_scales(kt + 2);
      }
    }
    warpgroup_fence_operand(accum);
  }

  // This consumer releases every stage after its K32 results have completed.
  CUTLASS_DEVICE void mma_tail(MainloopPipeline, PipelineState, int) {}
};
}  // namespace h20_exact_groupwise
