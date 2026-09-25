// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include <cfloat>
#include <optional>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "libtorch_stable/torch_utils.h"

namespace h20_groupwise {
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;
using Fp8 = cutlass::float_e4m3_t;
using namespace cute;
constexpr int kPipelineStages = 1;

__device__ uint64_t load_weight_word(const uint32_t* source) {
  if ((reinterpret_cast<uintptr_t>(source) & 7) == 0)
    return *reinterpret_cast<const uint64_t*>(source);
  return uint64_t(source[0]) | (uint64_t(source[1]) << 32);
}

__device__ uint64_t load_register_packed_word(const uint32_t* tile, int channel,
                                              int k_half) {
  const int thread = (channel % 8) * 4 + (channel / 16) * 32;
  const int shift = ((channel % 16) / 8) * 16;
  uint64_t bits = 0;
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    bits |= uint64_t((tile[(thread + i) * 2 + k_half] >> shift) & 0xffff)
            << (i * 16);
  return bits;
}

__device__ uint4 load_activation_vector(const Fp8* source) {
  if ((reinterpret_cast<uintptr_t>(source) & 15) == 0)
    return *reinterpret_cast<const uint4*>(source);
  uint32_t words[4] = {};
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i)
    words[i / 4] |= uint32_t(source[i].raw()) << ((i % 4) * 8);
  return make_uint4(words[0], words[1], words[2], words[3]);
}

template <bool Silu>
__device__ float read_input(const __nv_bfloat16* input, int64_t index,
                            int width) {
  if constexpr (!Silu) return __bfloat162float(input[index]);
  int64_t row = index / width;
  int column = index % width;
  float gate = __bfloat162float(input[row * (2 * width) + column]);
  float up = __bfloat162float(input[row * (2 * width) + width + column]);
  float activation =
      __bfloat162float(__float2bfloat16_rn(gate / (1.f + expf(-gate))));
  return __bfloat162float(__float2bfloat16_rn(activation * up));
}

template <bool Silu>
__global__ void quantize_kernel(const __nv_bfloat16* input, Fp8* output,
                                float* scales, int64_t groups, int group_size,
                                int width, const float* row_scale) {
  const int lane = threadIdx.x % 32;
  const int64_t group = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (group >= groups) return;
  float maximum = 0.f;
  bool nan = false;
  for (int k = lane; k < group_size; k += 32) {
    float x = read_input<Silu>(input, group * group_size + k, width);
    maximum = fmaxf(maximum, fabsf(x));
    nan |= isnan(x);
  }
  const bool any_nan = __any_sync(0xffffffff, nan);
  for (int d = 16; d; d >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
  float scale = any_nan ? __int_as_float(0x7fc00000)
                        : fmaxf(maximum * (1.f / 448.f), FLT_MIN);
  if (lane == 0)
    scales[group] =
        row_scale ? __fmul_rn(scale, row_scale[group / (width / group_size)])
                  : scale;
  for (int k = lane; k < group_size; k += 32) {
    float x = __fdiv_rn(read_input<Silu>(input, group * group_size + k, width),
                        scale);
    if (!isnan(x)) x = fminf(448.f, fmaxf(-448.f, x));
    output[group * group_size + k] = Fp8(x);
  }
}

template <int Tokens>
__global__ void tile_prefix_kernel(const int* offsets, int* prefix, int experts,
                                   int rows) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage storage;
  int e = threadIdx.x;
  int count = 0;
  if (e < experts) {
    int begin = max(0, min(rows, offsets[e]));
    int end = max(begin, min(rows, offsets[e + 1]));
    count = (end - begin + Tokens - 1) / Tokens;
  }
  int start, total;
  Scan(storage).ExclusiveSum(count, start, total);
  if (e < experts) prefix[e] = start;
  if (e == 0) prefix[experts] = total;
}

template <bool Silu>
__global__ void quantize_group32_kernel(const __nv_bfloat16* input, Fp8* output,
                                        float* scales, int64_t groups,
                                        int width, const float* row_scale) {
  const int lane = threadIdx.x % 4;
  const unsigned mask = 0xfu << (threadIdx.x % 32 / 4 * 4);
  const int64_t group = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 4;
  if (group >= groups) return;
  const int64_t row = group / (width / 32);
  const int column = (group % (width / 32)) * 32 + lane * 8;
  const int64_t offset = row * width * (Silu ? 2 : 1) + column;
  const uint4 words = *reinterpret_cast<const uint4*>(input + offset);
  const uint64_t packed[2] = {uint64_t(words.x) | (uint64_t(words.y) << 32),
                              uint64_t(words.z) | (uint64_t(words.w) << 32)};
  uint4 up_words = make_uint4(0, 0, 0, 0);
  if constexpr (Silu)
    up_words = *reinterpret_cast<const uint4*>(input + offset + width);
  const uint64_t packed_up[2] = {
      uint64_t(up_words.x) | (uint64_t(up_words.y) << 32),
      uint64_t(up_words.z) | (uint64_t(up_words.w) << 32)};
  cutlass::Array<float, 8> values;
  float maximum = 0.f;
  bool nan = false;
  CUTE_UNROLL
  for (int j = 0; j < 8; ++j) {
    float value = __bfloat162float(
        __ushort_as_bfloat16((packed[j / 4] >> ((j % 4) * 16)) & 0xffff));
    if constexpr (Silu) {
      float up = __bfloat162float(
          __ushort_as_bfloat16((packed_up[j / 4] >> ((j % 4) * 16)) & 0xffff));
      value =
          __bfloat162float(__float2bfloat16_rn(value / (1.f + expf(-value))));
      value = __bfloat162float(__float2bfloat16_rn(value * up));
    }
    values[j] = value;
    maximum = fmaxf(maximum, fabsf(value));
    nan |= isnan(value);
  }
  const bool any_nan = __any_sync(mask, nan);
  for (int d = 2; d; d >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, d, 4));
  const float scale = any_nan ? __int_as_float(0x7fc00000)
                              : fmaxf(maximum * (1.f / 448.f), FLT_MIN);
  if (lane == 0)
    scales[group] = row_scale ? __fmul_rn(scale, row_scale[row]) : scale;
  // BF16 values can use reciprocal multiplication for normal E4M3 results.
  // Subnormal rounding boundaries and nonfinite groups need exact division.
  const float inverse = __frcp_rn(scale);
  CUTE_UNROLL
  for (int j = 0; j < 8; ++j) {
    float value = __fmul_rn(values[j], inverse);
    if (!isfinite(scale) || fabsf(value) < 0x1p-6f)
      value = __fdiv_rn(values[j], scale);
    values[j] = value;
  }
  // The converter already saturates finite values to the E4M3 range.
  const auto converted =
      cutlass::NumericArrayConverter<Fp8, float, 8>{}(values);
  uint64_t codes;
  memcpy(&codes, &converted, sizeof(codes));
  *reinterpret_cast<uint64_t*>(output + group * 32 + lane * 8) = codes;
}

template <int Tokens>
struct MmaSelector;
template <>
struct MmaSelector<8> {
  using Op = SM90_64x8x32_F32E4M3E4M3_SS_TN<>;
};
template <>
struct MmaSelector<32> {
  using Op = SM90_64x32x32_F32E4M3E4M3_SS_TN<>;
};
template <>
struct MmaSelector<64> {
  using Op = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
};

// Weights occupy the GMMA M dimension; tokens occupy its N dimension.
// Each K=32 dot product is promoted with that token/weight group's scales.
template <int Tokens, bool Prepacked = false, bool RegisterPacked = false>
__global__ void grouped_gemm_kernel(const Fp8* a, const uint32_t* weight,
                                    const float* a_scale, const float* w_scale,
                                    const int* offsets, const int* tile_prefix,
                                    void* output, int rows, int n, int k,
                                    int experts, int a_group, int w_group,
                                    bool output_float) {
  using WLayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Fp8>{},
                                         make_shape(Int<64>{}, Int<32>{})));
  using ALayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Fp8>{},
                                         make_shape(Int<Tokens>{}, Int<32>{})));
  constexpr int Stages = kPipelineStages;
  __shared__ __align__(128) Fp8 sw[Stages * cosize_v<WLayout>];
  __shared__ __align__(128) Fp8 sa[Stages * cosize_v<ALayout>];
  __shared__ float shared_w_scale[Stages][64];
  __shared__ float shared_a_scale[Stages][Tokens];
  auto mma = make_tiled_mma(typename MmaSelector<Tokens>::Op{});
  auto thread_mma = mma.get_slice(threadIdx.x);
  auto coordinates = thread_mma.partition_C(
      make_identity_tensor(make_shape(Int<64>{}, Int<Tokens>{})));
  auto prototype = thread_mma.partition_fragment_C(
      make_identity_tensor(make_shape(Int<64>{}, Int<Tokens>{})));
  auto partial = make_tensor<float>(append(shape(prototype), Int<Stages>{}));
  auto total = make_tensor<float>(shape(prototype));
  const int n_tiles = (n + 63) / 64;
  constexpr int k_tiles = 1;
  const int work = tile_prefix[experts] * n_tiles * k_tiles;
  for (int task = blockIdx.x; task < work; task += gridDim.x) {
    const int row_channel_tile = task / k_tiles;
    const int token_tile = row_channel_tile / n_tiles;
    const int channel_start = (row_channel_tile % n_tiles) * 64;
    const int k_begin = 0;
    const int k_end = k;
    int lo = 0, hi = experts;
    while (lo + 1 < hi) {
      int mid = (lo + hi) / 2;
      if (tile_prefix[mid] <= token_tile)
        lo = mid;
      else
        hi = mid;
    }
    const int expert = lo;
    const int begin = max(0, min(rows, offsets[expert]));
    const int end = max(begin, min(rows, offsets[expert + 1]));
    const int row_start = begin + (token_tile - tile_prefix[expert]) * Tokens;
    clear(total);
    for (int base_k = k_begin; base_k < k_end; base_k += Stages * 32) {
      CUTE_UNROLL
      for (int stage = 0; stage < Stages; ++stage) {
        const int kk = base_k + stage * 32;
        auto w_smem = make_tensor(make_smem_ptr(sw + stage * cosize_v<WLayout>),
                                  WLayout{});
        auto a_smem = make_tensor(make_smem_ptr(sa + stage * cosize_v<ALayout>),
                                  ALayout{});
        if (threadIdx.x < 64) {
          int channel = channel_start + threadIdx.x;
          shared_w_scale[stage][threadIdx.x] =
              channel < n && kk < k
                  ? w_scale[Prepacked
                                ? ((int64_t(expert) * n_tiles +
                                    channel_start / 64) *
                                       (k / w_group) +
                                   (w_group == 32 ? kk / 32 : kk / 128)) *
                                          64 +
                                      threadIdx.x
                                : (int64_t(expert) * n + channel) *
                                          (k / w_group) +
                                      (w_group == 32 ? kk / 32 : kk / 128)]
                  : 0.f;
        }
        if (threadIdx.x < Tokens) {
          int row = row_start + threadIdx.x;
          shared_a_scale[stage][threadIdx.x] =
              row < end && kk < k ? a_scale[int64_t(row) * (k / a_group) +
                                            (a_group == 32 ? kk / 32 : 0)]
                                  : 0.f;
        }
        // Vectorize sixteen INT4 values into one shared-memory store.
        for (int i = threadIdx.x * 2; i < 64 * 4; i += blockDim.x * 2) {
          int channel = channel_start + i / 4;
          int col = kk + (i % 4) * 8;
          int64_t weight_index =
              Prepacked ? ((int64_t(expert) * n_tiles + channel_start / 64) *
                               (k / 32) +
                           kk / 32) *
                                  256 +
                              i
                        : (int64_t(expert) * n + channel) * (k / 8) + col / 8;
          uint64_t bits = 0;
          if (channel < n && kk < k) {
            if constexpr (RegisterPacked) {
              const uint32_t* tile =
                  weight +
                  ((int64_t(expert) * n_tiles + channel_start / 64) * (k / 32) +
                   kk / 32) *
                      256;
              bits = load_register_packed_word(tile, i / 4, (i % 4) / 2);
            } else {
              bits = load_weight_word(weight + weight_index);
            }
          }
          cutlass::Array<cutlass::int4b_t, 16> quantized;
          memcpy(&quantized, &bits, sizeof(bits));
          auto converted =
              cutlass::NumericArrayConverter<Fp8, cutlass::int4b_t, 16>{}(
                  quantized);
          uint4 packed;
          memcpy(&packed, &converted, sizeof(packed));
          *reinterpret_cast<uint4*>(&w_smem(i / 4, (i % 4) * 8)) = packed;
        }
        for (int i = threadIdx.x * 2; i < Tokens * 4; i += blockDim.x * 2) {
          int row = row_start + i / 4;
          uint4 value = (row < end && kk < k)
                            ? load_activation_vector(a + int64_t(row) * k + kk +
                                                     (i % 4) * 8)
                            : make_uint4(0, 0, 0, 0);
          *reinterpret_cast<uint4*>(&a_smem(i / 4, (i % 4) * 8)) = value;
        }
      }
      __syncthreads();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      clear(partial);
      warpgroup_arrive();
      CUTE_UNROLL
      for (int stage = 0; stage < Stages; ++stage) {
        auto w_smem = make_tensor(make_smem_ptr(sw + stage * cosize_v<WLayout>),
                                  WLayout{});
        auto a_smem = make_tensor(make_smem_ptr(sa + stage * cosize_v<ALayout>),
                                  ALayout{});
        auto fragment_w = thread_mma.partition_fragment_A(w_smem);
        auto fragment_a = thread_mma.partition_fragment_B(a_smem);
        auto accum = partial(_, _, _, stage);
        gemm(mma, fragment_w, fragment_a, accum);
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      CUTE_UNROLL
      for (int i = 0; i < size(total); ++i) {
        int channel = channel_start + get<0>(coordinates(i));
        int row = row_start + get<1>(coordinates(i));
        if (channel < n && row < end) {
          CUTE_UNROLL
          for (int stage = 0; stage < Stages; ++stage) {
            int kk = base_k + stage * 32;
            if (kk < k) {
              float as = shared_a_scale[stage][get<1>(coordinates(i))];
              float ws = shared_w_scale[stage][get<0>(coordinates(i))];
              auto accum = partial(_, _, _, stage);
              float contribution = __fmul_rn(__fmul_rn(accum(i), as), ws);
              total(i) = __fadd_rn(total(i), contribution);
            }
          }
        }
      }
      __syncthreads();
    }
    {
      CUTE_UNROLL
      for (int i = 0; i < size(total); ++i) {
        int channel = channel_start + get<0>(coordinates(i));
        int row = row_start + get<1>(coordinates(i));
        if (channel < n && row < end) {
          if (output_float)
            static_cast<float*>(output)[int64_t(row) * n + channel] = total(i);
          else
            static_cast<__nv_bfloat16*>(output)[int64_t(row) * n + channel] =
                __float2bfloat16_rn(total(i));
        }
      }
    }
    __syncthreads();
  }
}

// Keep multiple global-to-shared transfers outstanding, while retaining only
// one K32 accumulator fragment and its exact ordered FP32 promotion.
template <int Tokens>
struct RegisterMmaSelector;
template <>
struct RegisterMmaSelector<8> {
  using Op = SM90_64x8x32_F32E4M3E4M3_RS_TN<>;
};
template <>
struct RegisterMmaSelector<32> {
  using Op = SM90_64x32x32_F32E4M3E4M3_RS_TN<>;
};
template <>
struct RegisterMmaSelector<64> {
  using Op = SM90_64x64x32_F32E4M3E4M3_RS_TN<>;
};

template <int Tokens, bool SplitK = false, bool BothGroup32 = false,
          bool RegisterPacked = false, bool VectorScales = false>
__device__ __forceinline__ void grouped_gemm_pipeline_impl(
    const Fp8* a, const uint32_t* weight, const float* a_scale,
    const float* w_scale, const int* offsets, const int* tile_prefix,
    void* output, int rows, int n, int k, int experts, int a_group, int w_group,
    bool output_float) {
  using WLayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Fp8>{},
                                         make_shape(Int<64>{}, Int<32>{})));
  using ALayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Fp8>{},
                                         make_shape(Int<Tokens>{}, Int<32>{})));
  constexpr int Depth = 4;
  constexpr bool RegisterA = BothGroup32 || RegisterPacked;
  static_assert(!VectorScales ||
                (Tokens == 64 && !SplitK && BothGroup32 && RegisterPacked));
  __shared__ __align__(128) uint32_t raw[Depth][256];
  __shared__ __align__(128) Fp8 sw[RegisterA ? 1 : cosize_v<WLayout>];
  __shared__ __align__(128) Fp8 sa[Depth * cosize_v<ALayout>];
  __shared__ float ws[Depth][64];
  __shared__ float as[Depth][Tokens];
  __shared__ __align__(16) float vector_as[VectorScales ? 2 : 1][Tokens][4];
  auto mma = [] {
    if constexpr (RegisterA)
      return make_tiled_mma(typename RegisterMmaSelector<Tokens>::Op{});
    else
      return make_tiled_mma(typename MmaSelector<Tokens>::Op{});
  }();
  mma.accumulate_ = GMMA::ScaleOut::Zero;
  auto thread_mma = mma.get_slice(threadIdx.x);
  auto coordinates = thread_mma.partition_C(
      make_identity_tensor(make_shape(Int<64>{}, Int<Tokens>{})));
  auto partial = thread_mma.partition_fragment_C(
      make_identity_tensor(make_shape(Int<64>{}, Int<Tokens>{})));
  clear(partial);
  auto total = make_tensor<float>(shape(partial));
  const int n_tiles = (n + 63) / 64;
  const int groups = k / 32;
  const int weight_scale_groups = BothGroup32 ? groups : k / w_group;
  const int activation_scale_groups = BothGroup32 ? groups : k / a_group;
  constexpr int SplitGroups = 4;
  const int splits = SplitK ? (groups + SplitGroups - 1) / SplitGroups : 1;
  const int work = tile_prefix[experts] * n_tiles * splits;
  for (int task = blockIdx.x; task < work; task += gridDim.x) {
    const int row_channel_task = task / splits;
    const int token_tile = row_channel_task / n_tiles;
    const int channel_tile = row_channel_task % n_tiles;
    const int group_begin = SplitK ? task % splits * SplitGroups : 0;
    const int group_end =
        SplitK ? min(group_begin + SplitGroups, groups) : groups;
    const int channel_start = channel_tile * 64;
    int lo = 0, hi = experts;
    while (lo + 1 < hi) {
      int mid = (lo + hi) / 2;
      if (tile_prefix[mid] <= token_tile)
        lo = mid;
      else
        hi = mid;
    }
    const int expert = lo;
    const int begin = max(0, min(rows, offsets[expert]));
    const int end = max(begin, min(rows, offsets[expert + 1]));
    const int row_start = begin + (token_tile - tile_prefix[expert]) * Tokens;
    const int64_t weight_base =
        (int64_t(expert) * n_tiles + channel_tile) * groups * 256 +
        threadIdx.x * 2;
    const int activation_row = row_start + threadIdx.x / 2;
    const int64_t activation_base =
        int64_t(activation_row) * k + (threadIdx.x % 2) * 16;
    const int64_t weight_scale_base =
        (int64_t(expert) * n_tiles + channel_tile) * weight_scale_groups * 64 +
        threadIdx.x;
    const int scale_row = row_start + threadIdx.x;
    const int64_t activation_scale_base =
        int64_t(scale_row) * activation_scale_groups;
    auto prefetch = [&](int group, int slot) {
      const int i = threadIdx.x * 2;
      // The packed allocation includes all 64 channels, including padded tails.
      const bool valid = group < group_end;
      const int64_t wi = valid ? weight_base + int64_t(group) * 256 : 0;
      SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint64_t>::copy(
          *reinterpret_cast<const uint64_t*>(weight + wi),
          *reinterpret_cast<uint64_t*>(&raw[slot][i]), valid);
      if (threadIdx.x < Tokens * 2) {
        const bool av = group < group_end && activation_row < end;
        const int64_t ai = av ? activation_base + int64_t(group) * 32 : 0;
        auto a_smem = make_tensor(make_smem_ptr(sa + slot * cosize_v<ALayout>),
                                  ALayout{});
        SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
            *reinterpret_cast<const uint4*>(a + ai),
            *reinterpret_cast<uint4*>(
                &a_smem(threadIdx.x / 2, (threadIdx.x % 2) * 16)),
            av);
      }
      if (threadIdx.x < 64) {
        const bool sv = group < group_end;
        const int64_t si =
            sv ? weight_scale_base +
                     int64_t(BothGroup32 || w_group == 32 ? group : group / 4) *
                         64
               : 0;
        SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(w_scale[si],
                                                     ws[slot][threadIdx.x], sv);
      }
      if constexpr (VectorScales) {
        // Two banks hold the current and next batches of four K32 scales.
        if (threadIdx.x < Tokens && group % 4 == 0) {
          const int bank = (group / 4) % 2;
          const bool vectorized =
              groups % 4 == 0 && group + 3 < group_end &&
              (reinterpret_cast<uintptr_t>(a_scale) & 15) == 0;
          if (vectorized) {
            const bool sv = scale_row < end;
            const int64_t si = sv ? activation_scale_base + group : 0;
            SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
                *reinterpret_cast<const uint4*>(a_scale + si),
                *reinterpret_cast<uint4*>(&vector_as[bank][threadIdx.x][0]),
                sv);
          } else {
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j) {
              const bool sv = group + j < group_end && scale_row < end;
              const int64_t si = sv ? activation_scale_base + group + j : 0;
              SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(
                  a_scale[si], vector_as[bank][threadIdx.x][j], sv);
            }
          }
        }
      } else {
        if (threadIdx.x < Tokens) {
          const bool sv = group < group_end && scale_row < end;
          const int64_t si = sv ? activation_scale_base +
                                      (BothGroup32 || a_group == 32 ? group : 0)
                                : 0;
          SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(
              a_scale[si], as[slot][threadIdx.x], sv);
        }
      }
      cp_async_fence();
    };
    CUTE_UNROLL
    for (int p = 0; p < (SplitK ? Depth : Depth - 1); ++p)
      prefetch(group_begin + p, p);
    clear(total);
    for (int group = group_begin; group < group_end; ++group) {
      const int slot = (group - group_begin) % Depth;
      cp_async_wait<SplitK ? Depth - 1 : Depth - 2>();
      __syncthreads();
      if constexpr (!SplitK) {
        // The barrier also retires the preceding iteration's shared reads.
        // Refill that free slot while the current group computes.
        prefetch(group + Depth - 1, (slot + Depth - 1) % Depth);
      }
      auto w_smem = make_tensor(make_smem_ptr(sw), WLayout{});
      auto a_smem =
          make_tensor(make_smem_ptr(sa + slot * cosize_v<ALayout>), ALayout{});
      uint64_t bits;
      if constexpr (RegisterPacked) {
        bits = *reinterpret_cast<const uint64_t*>(&raw[slot][threadIdx.x * 2]);
      } else if constexpr (BothGroup32) {
        // SM90 ALayout_64x32: two M rows, each with four values per K half.
        const int m = (threadIdx.x / 4) % 8 + (threadIdx.x / 32) * 16;
        const int word = (threadIdx.x % 4) / 2;
        const int shift = (threadIdx.x % 2) * 16;
        const uint64_t q0 = (raw[slot][m * 4 + word] >> shift) & 0xffff;
        const uint64_t q1 = (raw[slot][(m + 8) * 4 + word] >> shift) & 0xffff;
        const uint64_t q2 = (raw[slot][m * 4 + word + 2] >> shift) & 0xffff;
        const uint64_t q3 =
            (raw[slot][(m + 8) * 4 + word + 2] >> shift) & 0xffff;
        bits = q0 | (q1 << 16) | (q2 << 32) | (q3 << 48);
      } else {
        bits = *reinterpret_cast<const uint64_t*>(&raw[slot][threadIdx.x * 2]);
      }
      cutlass::Array<cutlass::int4b_t, 16> quantized;
      memcpy(&quantized, &bits, sizeof(bits));
      auto converted =
          cutlass::NumericArrayConverter<Fp8, cutlass::int4b_t, 16>{}(
              quantized);
      auto fragment_w = thread_mma.partition_fragment_A(w_smem);
      if constexpr (RegisterA) {
        static_assert(size(fragment_w) == 16);
        CUTE_UNROLL
        for (int i = 0; i < 16; ++i) fragment_w(i) = converted[i];
      } else {
        uint4 packed;
        memcpy(&packed, &converted, sizeof(packed));
        *reinterpret_cast<uint4*>(
            &w_smem(threadIdx.x / 2, (threadIdx.x % 2) * 16)) = packed;
        __syncthreads();
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      warpgroup_arrive();
      auto fragment_a = thread_mma.partition_fragment_B(a_smem);
      gemm(mma, fragment_w, fragment_a, partial);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      if constexpr (!SplitK) {
        // The SM90 M64 accumulator repeats these two channels for each token
        // pair. Mask only the final stores; padded fragments cannot affect
        // valid ones.
        const float ws0 = ws[slot][get<0>(coordinates(0))];
        const float ws1 = ws[slot][get<0>(coordinates(2))];
        CUTE_UNROLL
        for (int i = 0; i < size(total); i += 4) {
          const float as0 =
              (VectorScales ? vector_as[(group / 4) % 2][get<1>(coordinates(i))]
                                       [group % 4]
                            : as[slot][get<1>(coordinates(i))]);
          const float as1 =
              (VectorScales ? vector_as[(group / 4) % 2]
                                       [get<1>(coordinates(i + 1))][group % 4]
                            : as[slot][get<1>(coordinates(i + 1))]);
          total(i) =
              __fadd_rn(total(i), __fmul_rn(__fmul_rn(partial(i), as0), ws0));
          total(i + 1) = __fadd_rn(
              total(i + 1), __fmul_rn(__fmul_rn(partial(i + 1), as1), ws0));
          total(i + 2) = __fadd_rn(
              total(i + 2), __fmul_rn(__fmul_rn(partial(i + 2), as0), ws1));
          total(i + 3) = __fadd_rn(
              total(i + 3), __fmul_rn(__fmul_rn(partial(i + 3), as1), ws1));
        }
      } else {
        CUTE_UNROLL
        for (int i = 0; i < size(total); ++i) {
          int channel = channel_start + get<0>(coordinates(i));
          int row = row_start + get<1>(coordinates(i));
          if (channel < n && row < end) {
            float contribution = __fmul_rn(
                __fmul_rn(partial(i),
                          (VectorScales ? vector_as[(group / 4) % 2][get<1>(
                                              coordinates(i))][group % 4]
                                        : as[slot][get<1>(coordinates(i))])),
                ws[slot][get<0>(coordinates(i))]);
            static_cast<float*>(
                output)[(int64_t(group) * rows + row) * n + channel] =
                contribution;
          }
        }
      }
      if constexpr (SplitK) {
        __syncthreads();
        prefetch(group + Depth, slot);
      }
    }
    cp_async_wait<0>();
    if constexpr (!SplitK) {
      CUTE_UNROLL
      for (int i = 0; i < size(total); ++i) {
        int channel = channel_start + get<0>(coordinates(i));
        int row = row_start + get<1>(coordinates(i));
        if (channel < n && row < end) {
          if (output_float)
            static_cast<float*>(output)[int64_t(row) * n + channel] = total(i);
          else
            static_cast<__nv_bfloat16*>(output)[int64_t(row) * n + channel] =
                __float2bfloat16_rn(total(i));
        }
      }
    }
    __syncthreads();
  }
}

template <int Tokens, bool SplitK = false, bool BothGroup32 = false,
          bool RegisterPacked = false>
__global__ void grouped_gemm_pipeline_kernel(
    const Fp8* a, const uint32_t* weight, const float* a_scale,
    const float* w_scale, const int* offsets, const int* tile_prefix,
    void* output, int rows, int n, int k, int experts, int a_group, int w_group,
    bool output_float) {
  grouped_gemm_pipeline_impl<Tokens, SplitK, BothGroup32, RegisterPacked,
                             false>(a, weight, a_scale, w_scale, offsets,
                                    tile_prefix, output, rows, n, k, experts,
                                    a_group, w_group, output_float);
}

// Keep the measured H20 register limit local to this long-K specialization.
template <int Tokens>
__global__ void __maxnreg__(96) grouped_gemm_pipeline_kernel_vector_scales(
    const Fp8* a, const uint32_t* weight, const float* a_scale,
    const float* w_scale, const int* offsets, const int* tile_prefix,
    void* output, int rows, int n, int k, int experts, int a_group, int w_group,
    bool output_float) {
  grouped_gemm_pipeline_impl<Tokens, false, true, true, true>(
      a, weight, a_scale, w_scale, offsets, tile_prefix, output, rows, n, k,
      experts, a_group, w_group, output_float);
}

__global__ void reduce_ordered_kernel(const float* partials, void* output,
                                      int64_t elements, int groups,
                                      bool output_float) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  float value = 0.f;
  for (int g = 0; g < groups; ++g)
    value = __fadd_rn(value, partials[int64_t(g) * elements + i]);
  if (output_float)
    static_cast<float*>(output)[i] = value;
  else
    static_cast<__nv_bfloat16*>(output)[i] = __float2bfloat16_rn(value);
}

void require_tensor(const Tensor& tensor, const Tensor& reference, Type type) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.device() == reference.device() &&
                  tensor.scalar_type() == type);
}

bool overlaps(const Tensor& a, const Tensor& b) {
  auto ab = reinterpret_cast<uintptr_t>(a.data_ptr());
  auto bb = reinterpret_cast<uintptr_t>(b.data_ptr());
  return ab < bb + b.numel() * b.element_size() &&
         bb < ab + a.numel() * a.element_size();
}

template <bool RegisterPacked = false>
__global__ void pack_weight_kernel(const uint32_t* weights, const float* scales,
                                   uint32_t* packed, float* packed_scales,
                                   int experts, int n, int k, int w_group) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int nt = (n + 63) / 64;
  const int groups = k / 32;
  const int scale_groups = k / w_group;
  if (index < int64_t(experts) * nt * groups * 256) {
    int word = index % 4;
    int channel = index / 4 % 64;
    int group = index / 256 % groups;
    int tile = index / (256LL * groups) % nt;
    int expert = index / (256LL * groups * nt);
    if constexpr (RegisterPacked) {
      const int thread = (index % 256) / 2;
      const int k_half = index % 2;
      const int m = tile * 64 + (thread / 4) % 8 + (thread / 32) * 16;
      const int source_word = group * 4 + (thread % 4) / 2 + k_half * 2;
      const int shift = (thread % 2) * 16;
      const uint32_t lo =
          m < n ? (weights[(int64_t(expert) * n + m) * (k / 8) + source_word] >>
                   shift) &
                      0xffff
                : 0;
      const uint32_t hi =
          m + 8 < n ? (weights[(int64_t(expert) * n + m + 8) * (k / 8) +
                               source_word] >>
                       shift) &
                          0xffff
                    : 0;
      packed[index] = lo | (hi << 16);
    } else {
      int c = tile * 64 + channel;
      packed[index] =
          c < n
              ? weights[(int64_t(expert) * n + c) * (k / 8) + group * 4 + word]
              : 0;
    }
  }
  if (index < int64_t(experts) * nt * scale_groups * 64) {
    int channel = index % 64;
    int group = index / 64 % scale_groups;
    int tile = index / (64LL * scale_groups) % nt;
    int expert = index / (64LL * scale_groups * nt);
    int c = tile * 64 + channel;
    packed_scales[index] =
        c < n ? scales[(int64_t(expert) * n + c) * scale_groups + group] : 0.f;
  }
}

void pack_weights(const Tensor& weights, const Tensor& scales, Tensor& packed,
                  Tensor& packed_scales, int64_t w_group) {
  STD_TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                  weights.dim() == 3 && weights.scalar_type() == Type::Int);
  require_tensor(scales, weights, Type::Float);
  require_tensor(packed, weights, Type::Int);
  require_tensor(packed_scales, weights, Type::Float);
  const int64_t e = weights.size(0), n = weights.size(1),
                k = weights.size(2) * 8;
  STD_TORCH_CHECK(e > 0 && e <= 256 && n > 0 && n < INT_MAX && k > 0 &&
                  k < INT_MAX);
  STD_TORCH_CHECK((w_group == 32 || w_group == 128) && k % w_group == 0);
  STD_TORCH_CHECK(scales.dim() == 3 && scales.size(0) == e &&
                  scales.size(1) == n && scales.size(2) == k / w_group);
  const int64_t nt = (n + 63) / 64;
  STD_TORCH_CHECK(packed.dim() == 5 && packed.size(0) == e &&
                  packed.size(1) == nt && packed.size(2) == k / 32 &&
                  ((packed.size(3) == 64 && packed.size(4) == 4) ||
                   (packed.size(3) == 128 && packed.size(4) == 2)));
  STD_TORCH_CHECK(packed_scales.dim() == 4 && packed_scales.size(0) == e &&
                  packed_scales.size(1) == nt &&
                  packed_scales.size(2) == k / w_group &&
                  packed_scales.size(3) == 64);
  for (const auto* input : {&weights, &scales})
    STD_TORCH_CHECK(
        !overlaps(*input, packed) && !overlaps(*input, packed_scales),
        "packing outputs must not overlap inputs");
  STD_TORCH_CHECK(!overlaps(packed, packed_scales),
                  "packing outputs must not overlap");
  const int64_t blocks = (packed.numel() + 255) / 256;
  STD_TORCH_CHECK(blocks < INT_MAX);
  const auto device = weights.device().index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto kernel = packed.size(3) == 128 ? pack_weight_kernel<true>
                                      : pack_weight_kernel<false>;
  kernel<<<blocks, 256, 0, get_current_cuda_stream(device)>>>(
      static_cast<const uint32_t*>(weights.data_ptr()),
      static_cast<const float*>(scales.data_ptr()),
      static_cast<uint32_t*>(packed.data_ptr()),
      static_cast<float*>(packed_scales.data_ptr()), e, n, k, w_group);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "INT4 weight packing launch failed");
}

template <bool Silu>
void quantize(const Tensor& input, Tensor& output, Tensor& scales,
              int64_t group_size, const std::optional<Tensor>& row_scale) {
  STD_TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                  input.dim() == 2 && input.scalar_type() == Type::BFloat16);
  const int64_t width = input.size(1) / (Silu ? 2 : 1);
  STD_TORCH_CHECK(width > 0 && width < INT_MAX &&
                  input.size(1) == width * (Silu ? 2 : 1));
  STD_TORCH_CHECK(group_size > 0 && group_size % 32 == 0 &&
                  width % group_size == 0);
  require_tensor(output, input, Type::Float8_e4m3fn);
  require_tensor(scales, input, Type::Float);
  STD_TORCH_CHECK(output.dim() == 2 && output.size(0) == input.size(0) &&
                  output.size(1) == width && scales.dim() == 2 &&
                  scales.size(0) == input.size(0) &&
                  scales.size(1) == width / group_size);
  STD_TORCH_CHECK(!overlaps(input, output) && !overlaps(input, scales) &&
                      !overlaps(output, scales),
                  "quantization buffers must not overlap");
  if (row_scale) {
    require_tensor(*row_scale, input, Type::Float);
    STD_TORCH_CHECK(row_scale->dim() == 1 &&
                    row_scale->numel() == input.size(0));
    STD_TORCH_CHECK(!overlaps(*row_scale, output) &&
                    !overlaps(*row_scale, scales));
  }
  const auto device = input.device().index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  const int64_t groups = output.numel() / group_size;
  if (!groups) return;
  if (group_size == 32 &&
      (reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(output.data_ptr()) & 7) == 0) {
    quantize_group32_kernel<Silu>
        <<<(groups + 31) / 32, 128, 0, get_current_cuda_stream(device)>>>(
            static_cast<const __nv_bfloat16*>(input.data_ptr()),
            static_cast<Fp8*>(output.data_ptr()),
            static_cast<float*>(scales.data_ptr()), groups, width,
            row_scale ? static_cast<const float*>(row_scale->data_ptr())
                      : nullptr);
    STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "vector group32 quantization launch failed");
    return;
  }
  quantize_kernel<Silu>
      <<<(groups + 3) / 4, 128, 0, get_current_cuda_stream(device)>>>(
          static_cast<const __nv_bfloat16*>(input.data_ptr()),
          static_cast<Fp8*>(output.data_ptr()),
          static_cast<float*>(scales.data_ptr()), groups, group_size, width,
          row_scale ? static_cast<const float*>(row_scale->data_ptr())
                    : nullptr);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "group FP8 quantization launch failed");
}

template <int Tokens, bool SplitK = false>
void launch(const Tensor& a, const Tensor& weights, const Tensor& a_scale,
            const Tensor& w_scale, const Tensor& offsets, Tensor& prefix,
            const Tensor& output, int n, int a_group, int w_group,
            cudaStream_t stream) {
  const int rows = a.size(0), k = a.size(1);
  const int e = weights.size(0);
  tile_prefix_kernel<Tokens>
      <<<1, 256, 0, stream>>>(static_cast<const int*>(offsets.data_ptr()),
                              static_cast<int*>(prefix.data_ptr()), e, rows);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         a.device().index());
  const int blocks =
      std::max(1, std::min(sms * 8, ((rows + Tokens - 1) / Tokens + e) *
                                        ((n + 63) / 64) *
                                        (SplitK ? (k + 127) / 128 : 1)));
  auto kernel = weights.dim() == 5 ? grouped_gemm_kernel<Tokens, true>
                                   : grouped_gemm_kernel<Tokens, false>;
  const bool register_packed = weights.dim() == 5 && weights.size(3) == 128;
  if (register_packed) kernel = grouped_gemm_kernel<Tokens, true, true>;
  if (weights.dim() == 5 &&
      (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(weights.data_ptr()) & 7) == 0) {
    kernel = a_group == 32 && w_group == 32
                 ? grouped_gemm_pipeline_kernel<Tokens, false, true>
                 : grouped_gemm_pipeline_kernel<Tokens>;
    if (register_packed)
      kernel = a_group == 32 && w_group == 32
                   ? grouped_gemm_pipeline_kernel<Tokens, false, true, true>
                   : grouped_gemm_pipeline_kernel<Tokens, false, false, true>;
    if constexpr (Tokens == 64 && !SplitK) {
      if (register_packed && a_group == 32 && w_group == 32 && k >= 512)
        kernel = grouped_gemm_pipeline_kernel_vector_scales<Tokens>;
    }
  }
  if constexpr (SplitK) {
    kernel = a_group == 32 && w_group == 32
                 ? grouped_gemm_pipeline_kernel<Tokens, true, true>
                 : grouped_gemm_pipeline_kernel<Tokens, true>;
    if (register_packed)
      kernel = a_group == 32 && w_group == 32
                   ? grouped_gemm_pipeline_kernel<Tokens, true, true, true>
                   : grouped_gemm_pipeline_kernel<Tokens, true, false, true>;
  }
  kernel<<<blocks, 128, 0, stream>>>(
      static_cast<const Fp8*>(a.data_ptr()),
      static_cast<const uint32_t*>(weights.data_ptr()),
      static_cast<const float*>(a_scale.data_ptr()),
      static_cast<const float*>(w_scale.data_ptr()),
      static_cast<const int*>(offsets.data_ptr()),
      static_cast<const int*>(prefix.data_ptr()), output.data_ptr(), rows, n, k,
      e, a_group, w_group, output.scalar_type() == Type::Float);
}

void grouped_mm(const Tensor& a, const Tensor& weights, const Tensor& a_scale,
                const Tensor& w_scale, const Tensor& offsets, Tensor& prefix,
                Tensor& output, int64_t a_group, int64_t w_group,
                int64_t tile_tokens, const std::optional<Tensor>& partials) {
  STD_TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 &&
                  a.scalar_type() == Type::Float8_e4m3fn);
  require_tensor(weights, a, Type::Int);
  require_tensor(a_scale, a, Type::Float);
  require_tensor(w_scale, a, Type::Float);
  require_tensor(offsets, a, Type::Int);
  require_tensor(prefix, a, Type::Int);
  STD_TORCH_CHECK(output.scalar_type() == Type::BFloat16 ||
                  output.scalar_type() == Type::Float);
  require_tensor(output, a, output.scalar_type());
  STD_TORCH_CHECK((weights.dim() == 3 || weights.dim() == 5) &&
                  output.dim() == 2);
  const int64_t r = a.size(0), k = a.size(1), e = weights.size(0),
                n = output.size(1);
  STD_TORCH_CHECK(k > 0 && k % 32 == 0 && n > 0 && e > 0 && e <= 256);
  STD_TORCH_CHECK(r < INT_MAX && k < INT_MAX && n < INT_MAX);
  STD_TORCH_CHECK(a_group == 32 || a_group == k);
  STD_TORCH_CHECK(w_group == 32 || w_group == 128);
  STD_TORCH_CHECK(k % w_group == 0);
  STD_TORCH_CHECK(a_scale.dim() == 2 && a_scale.size(0) == r &&
                  a_scale.size(1) == k / a_group);
  if (weights.dim() == 3) {
    STD_TORCH_CHECK(weights.size(1) == n && weights.size(2) == k / 8);
    STD_TORCH_CHECK(w_scale.dim() == 3 && w_scale.size(0) == e &&
                    w_scale.size(1) == n && w_scale.size(2) == k / w_group);
  } else {
    STD_TORCH_CHECK(weights.size(1) == (n + 63) / 64 &&
                    weights.size(2) == k / 32 &&
                    ((weights.size(3) == 64 && weights.size(4) == 4) ||
                     (weights.size(3) == 128 && weights.size(4) == 2)));
    STD_TORCH_CHECK(w_scale.dim() == 4 && w_scale.size(0) == e &&
                    w_scale.size(1) == (n + 63) / 64 &&
                    w_scale.size(2) == k / w_group && w_scale.size(3) == 64);
  }
  STD_TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == e + 1 &&
                  prefix.dim() == 1 && prefix.numel() == e + 1);
  STD_TORCH_CHECK(output.dim() == 2 && output.size(0) == r &&
                  output.size(1) == n);
  STD_TORCH_CHECK(tile_tokens == 8 || tile_tokens == 32 || tile_tokens == 64);
  STD_TORCH_CHECK(((r + 7) / 8 + e) * ((n + 63) / 64) < INT_MAX);
  for (const auto* tensor : {&a, &weights, &a_scale, &w_scale, &offsets}) {
    STD_TORCH_CHECK(!overlaps(output, *tensor) && !overlaps(prefix, *tensor),
                    "GEMM scratch/output must not overlap inputs");
  }
  STD_TORCH_CHECK(!overlaps(output, prefix),
                  "GEMM scratch/output must not overlap");
  if (partials) {
    require_tensor(*partials, a, Type::Float);
    STD_TORCH_CHECK(r <= 256 && k >= 512 && k <= 2048,
                    "ordered split-K is restricted to small-row FC1 shapes");
    STD_TORCH_CHECK(weights.dim() == 5 &&
                    (reinterpret_cast<uintptr_t>(a.data_ptr()) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(weights.data_ptr()) & 7) == 0);
    STD_TORCH_CHECK(partials->dim() == 3 && partials->size(0) == k / 32 &&
                    partials->size(1) == r && partials->size(2) == n);
    STD_TORCH_CHECK(partials->numel() <= 16LL * 1024 * 1024,
                    "ordered split-K workspace must not exceed 64 MiB");
    const Tensor* buffers[] = {&a,       &weights, &a_scale, &w_scale,
                               &offsets, &prefix,  &output};
    for (const auto* tensor : buffers)
      STD_TORCH_CHECK(!overlaps(*partials, *tensor),
                      "partial buffer must not overlap other buffers");
  }
  const auto device = a.device().index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  STD_TORCH_CHECK(major == 9, "SM90 groupwise GEMM requires Hopper");
  if (!r) return;
  const auto stream = get_current_cuda_stream(device);
  if (partials) {
    if (tile_tokens == 8)
      launch<8, true>(a, weights, a_scale, w_scale, offsets, prefix, *partials,
                      n, a_group, w_group, stream);
    else if (tile_tokens == 32)
      launch<32, true>(a, weights, a_scale, w_scale, offsets, prefix, *partials,
                       n, a_group, w_group, stream);
    else
      launch<64, true>(a, weights, a_scale, w_scale, offsets, prefix, *partials,
                       n, a_group, w_group, stream);
    reduce_ordered_kernel<<<(r * n + 255) / 256, 256, 0, stream>>>(
        static_cast<const float*>(partials->data_ptr()), output.data_ptr(),
        r * n, k / 32, output.scalar_type() == Type::Float);
    STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "ordered split-K launch failed");
    return;
  }
  if (tile_tokens == 8)
    launch<8>(a, weights, a_scale, w_scale, offsets, prefix, output, n, a_group,
              w_group, stream);
  else if (tile_tokens == 32)
    launch<32>(a, weights, a_scale, w_scale, offsets, prefix, output, n,
               a_group, w_group, stream);
  else
    launch<64>(a, weights, a_scale, w_scale, offsets, prefix, output, n,
               a_group, w_group, stream);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "groupwise FP8 GEMM launch failed");
}
}  // namespace h20_groupwise

STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  m.def(
      "h20_w4a8_groupwise_pack(Tensor weights, Tensor scales, Tensor(a!) "
      "packed, "
      "Tensor(b!) packed_scales, int w_group) -> ()");
  m.def(
      "h20_fp8_group_quant(Tensor input, Tensor(a!) output, Tensor(b!) scales, "
      "int group_size, Tensor? row_scale=None) -> ()");
  m.def(
      "h20_silu_fp8_group_quant(Tensor input, Tensor(a!) output, Tensor(b!) "
      "scales, int group_size, Tensor? row_scale=None) -> ()");
  m.def(
      "h20_w4a8_groupwise_mm(Tensor a, Tensor weights, Tensor a_scale, Tensor "
      "w_scale, Tensor offsets, Tensor(a!) prefix, Tensor(b!) output, int "
      "a_group, int w_group, int tile_tokens, Tensor(c!)? partials=None) -> "
      "()");
}
STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("h20_w4a8_groupwise_pack", TORCH_BOX(&h20_groupwise::pack_weights));
  m.impl("h20_fp8_group_quant", TORCH_BOX(&h20_groupwise::quantize<false>));
  m.impl("h20_silu_fp8_group_quant", TORCH_BOX(&h20_groupwise::quantize<true>));
  m.impl("h20_w4a8_groupwise_mm", TORCH_BOX(&h20_groupwise::grouped_mm));
}
