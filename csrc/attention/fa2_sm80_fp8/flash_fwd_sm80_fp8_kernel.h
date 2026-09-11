#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "flash_fwd_kernel.h"
#include "flash_sm80_fp8.h"

namespace FLASH_NAMESPACE {

using namespace cute;

__forceinline__ __device__ float decode_e4m3fn(uint8_t bits) {
  const uint32_t sign = uint32_t(bits & 0x80U) << 24U;
  const uint32_t magnitude = bits & 0x7FU;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 0x07U;
  if (exponent == 0U) {
    const float value = float(mantissa) * 0.001953125f;  // 2^-9
    return sign == 0U ? value : -value;
  }
  // E4M3FN normals map exactly into FP32.  The only NaN code is 0x7f
  // (and its signed counterpart); cache values produced from finite,
  // saturated QDQ never contain it, but preserve the format semantics.
  if (magnitude == 0x7FU) {
    return __uint_as_float(sign | 0x7FC00000U);
  }
  return __uint_as_float(sign | ((exponent + 120U) << 23U) | (mantissa << 20U));
}

__forceinline__ __device__ float scaled_e4m3fn_qdq(float value, float scale) {
  const float normalized =
      fminf(fmaxf(__fdiv_rn(value, scale), -448.0f), 448.0f);
  const float magnitude = fabsf(normalized);
  const uint32_t bits = __float_as_uint(magnitude);
  // Retain three fraction bits with ties-to-even, matching E4M3 conversion.
  const uint32_t rounded =
      (bits + 0x7FFFFU + ((bits >> 20U) & 1U)) & 0xFFF00000U;
  const float quantized = magnitude < 0.015625f
                              ? nearbyintf(magnitude * 512.0f) * 0.001953125f
                              : __uint_as_float(rounded);
  return copysignf(fminf(quantized, 448.0f), normalized) * scale;
}

template <typename KernelTraits, typename Element, typename SmemTensor>
__forceinline__ __device__ void load_q_qdq(
    const Flash_fwd_sm80_fp8_params& params, const BlockInfo<true>& binfo,
    int bidh, int m_block, SmemTensor& sQ) {
  constexpr int kHeadDim = 256;
  constexpr int kVec = 8;
  constexpr int kVecsPerRow = kHeadDim / kVec;
  constexpr int kNThreads = KernelTraits::kNThreads;
  constexpr int kBlockM = KernelTraits::kBlockM;

  const int valid_rows =
      min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
  const Element* q = reinterpret_cast<const Element*>(params.q_ptr);
  const int64_t q_base =
      binfo.q_offset(params.q_batch_stride, params.q_row_stride, blockIdx.y) +
      int64_t(m_block * kBlockM) * params.q_row_stride +
      int64_t(bidh) * params.q_head_stride;
  const float scale = __ldg(params.q_scale_ptr);

  for (int vec = threadIdx.x; vec < valid_rows * kVecsPerRow;
       vec += kNThreads) {
    const int row = vec / kVecsPerRow;
    const int col = (vec % kVecsPerRow) * kVec;
    const auto values = *reinterpret_cast<const cutlass::Array<Element, kVec>*>(
        q + q_base + int64_t(row) * params.q_row_stride + col);
    alignas(16) cutlass::Array<Element, kVec> q_qdq;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      q_qdq[i] =
          Element(scaled_e4m3fn_qdq(static_cast<float>(values[i]), scale));
    }
    // Swizzle<3, 3, 3> only permutes 16-byte vectors.  Values within an
    // eight-BF16 vector remain contiguous, so evaluate the CuTe layout once
    // and issue one 128-bit shared-memory store instead of eight scalar
    // stores (and eight dynamic swizzle address calculations).
    const auto smem_offset = sQ.layout()(make_coord(row, col));
    *reinterpret_cast<uint4*>(sQ.data().get() + smem_offset) =
        *reinterpret_cast<const uint4*>(&q_qdq);
  }
}

template <typename KernelTraits, typename Element, typename SmemTensorK,
          typename SmemTensorV>
__forceinline__ __device__ void load_paged_kv_fp8(
    const Flash_fwd_sm80_fp8_params& params, const BlockInfo<true>& binfo,
    const int* block_table, int kv_head, int n_block, SmemTensorK& sK,
    SmemTensorV& sV, const Element* k_lookup, const Element* v_lookup) {
  constexpr int kVec = 8;
  constexpr int kNThreads = KernelTraits::kNThreads;
  constexpr int kBlockN = KernelTraits::kBlockN;
  constexpr int kPageSize = 16;
  constexpr int kPagesPerTile = kBlockN / kPageSize;
  constexpr int kNWarps = kNThreads / 32;
  static_assert(kBlockN % kPageSize == 0);

  const uint8_t* k = reinterpret_cast<const uint8_t*>(params.k_ptr);
  const uint8_t* v = reinterpret_cast<const uint8_t*>(params.v_ptr);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int col = lane * kVec;
  const int tile_token = n_block * kBlockN;
  const int first_virtual_page = tile_token >> 4;

  for (int page = 0; page < kPagesPerTile; ++page) {
    const int page_token = tile_token + page * kPageSize;
    int physical_page = 0;
    if (page_token < binfo.actual_seqlen_k) {
      physical_page = block_table[first_virtual_page + page];
    }
    const int64_t k_page_base = int64_t(physical_page) * params.k_batch_stride +
                                int64_t(kv_head) * params.k_head_stride;
    const int64_t v_page_base = int64_t(physical_page) * params.v_batch_stride +
                                int64_t(kv_head) * params.v_head_stride;

    for (int page_offset = warp; page_offset < kPageSize;
         page_offset += kNWarps) {
      const int row = page * kPageSize + page_offset;
      const int token = page_token + page_offset;
      uint64_t k_bits = 0;
      uint64_t v_bits = 0;
      if (token < binfo.actual_seqlen_k) {
        const int64_t k_offset =
            k_page_base + int64_t(page_offset) * params.k_row_stride + col;
        const int64_t v_offset =
            v_page_base + int64_t(page_offset) * params.v_row_stride + col;
        k_bits = *reinterpret_cast<const uint64_t*>(k + k_offset);
        v_bits = *reinterpret_cast<const uint64_t*>(v + v_offset);
      }

      alignas(16) cutlass::Array<Element, kVec> k_values;
      alignas(16) cutlass::Array<Element, kVec> v_values;
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        const uint8_t kb = static_cast<uint8_t>(k_bits >> (i * 8));
        const uint8_t vb = static_cast<uint8_t>(v_bits >> (i * 8));
        k_values[i] = k_lookup[kb];
        v_values[i] = v_lookup[vb];
      }
      const auto k_smem_offset = sK.layout()(make_coord(row, col));
      const auto v_smem_offset = sV.layout()(make_coord(row, col));
      *reinterpret_cast<uint4*>(sK.data().get() + k_smem_offset) =
          *reinterpret_cast<const uint4*>(&k_values);
      *reinterpret_cast<uint4*>(sV.data().get() + v_smem_offset) =
          *reinterpret_cast<const uint4*>(&v_values);
    }
  }
}

template <typename KernelTraits, bool IsKey>
__forceinline__ __device__ void prefetch_paged_fp8(
    const Flash_fwd_sm80_fp8_params& params, const BlockInfo<true>& binfo,
    const int* block_table, int kv_head, int n_block, uint8_t* raw) {
  constexpr int kVectors = KernelTraits::kBlockN * 256 / 16;
  const uint8_t* source =
      reinterpret_cast<const uint8_t*>(IsKey ? params.k_ptr : params.v_ptr);
  const int64_t batch_stride =
      IsKey ? params.k_batch_stride : params.v_batch_stride;
  const int64_t row_stride = IsKey ? params.k_row_stride : params.v_row_stride;
  const int64_t head_stride =
      IsKey ? params.k_head_stride : params.v_head_stride;
  for (int vec = threadIdx.x; vec < kVectors; vec += KernelTraits::kNThreads) {
    const int row = vec / 16;
    const int col = (vec % 16) * 16;
    const int token = n_block * KernelTraits::kBlockN + row;
    const bool valid = token < binfo.actual_seqlen_k;
    const int physical_page = valid ? block_table[token / 16] : 0;
    const uint8_t* src = source + int64_t(physical_page) * batch_stride +
                         int64_t(token % 16) * row_stride +
                         int64_t(kv_head) * head_stride + col;
    const uint32_t dst =
        static_cast<uint32_t>(__cvta_generic_to_shared(raw + vec * 16));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst),
                 "l"(src), "r"(valid ? 16 : 0));
  }
}

template <typename KernelTraits, typename Element, typename SmemTensor>
__forceinline__ __device__ void decode_shared_fp8(const uint8_t* raw,
                                                  SmemTensor& destination,
                                                  const Element* lookup) {
  constexpr int kVectors = KernelTraits::kBlockN * 256 / 8;
  for (int vec = threadIdx.x; vec < kVectors; vec += KernelTraits::kNThreads) {
    const uint64_t bits = *reinterpret_cast<const uint64_t*>(raw + vec * 8);
    alignas(16) cutlass::Array<Element, 8> values;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = lookup[static_cast<uint8_t>(bits >> (i * 8))];
    }
    const auto offset =
        destination.layout()(make_coord(vec / 32, (vec % 32) * 8));
    *reinterpret_cast<uint4*>(destination.data().get() + offset) =
        *reinterpret_cast<const uint4*>(&values);
  }
}

template <typename KernelTraits, bool IsDecode = false, bool UseAsync = false,
          bool CompactAsync = false>
inline __device__ void compute_attn_sm80_fp8(
    const Flash_fwd_sm80_fp8_params& params) {
  using Element = typename KernelTraits::Element;
  using ElementAccum = typename KernelTraits::ElementAccum;
  using index_t = typename KernelTraits::index_t;
  constexpr int kBlockM = KernelTraits::kBlockM;
  constexpr int kBlockN = KernelTraits::kBlockN;
  constexpr int kHeadDim = KernelTraits::kHeadDim;
  constexpr int kNWarps = KernelTraits::kNWarps;
  constexpr int kOutputDim = IsDecode ? kHeadDim / kNWarps : kHeadDim;
  constexpr int kMmaWarps = IsDecode ? 1 : kNWarps;
  static_assert(!CompactAsync || (UseAsync && IsDecode));
  static_assert(
      (kBlockM == 64 || kBlockM == 128 || (IsDecode && kBlockM == 16)) &&
      kBlockN == 64 && kHeadDim == 256 &&
      (KernelTraits::kNThreads == 128 || KernelTraits::kNThreads == 256));

  extern __shared__ char smem_raw[];
  const int tidx = threadIdx.x;
  const int mma_thread = IsDecode ? tidx % 32 : tidx;
  const int output_partition = IsDecode ? tidx / 32 : 0;
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;
  const BlockInfo<true> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) {
    return;
  }

  const int n_block_min = 0;
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (params.is_causal) {
    n_block_max = min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM +
                                                      binfo.actual_seqlen_k -
                                                      binfo.actual_seqlen_q,
                                                  kBlockN));
  }

  if (n_block_max <= n_block_min) {
    const int valid_rows =
        min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
    Element* out = reinterpret_cast<Element*>(params.o_ptr);
    const index_t out_base =
        binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
        index_t(m_block * kBlockM) * params.o_row_stride +
        index_t(bidh) * params.o_head_stride;
    for (int linear = tidx; linear < valid_rows * kHeadDim;
         linear += KernelTraits::kNThreads) {
      out[out_base + index_t(linear / kHeadDim) * params.o_row_stride +
          linear % kHeadDim] = Element(0.0f);
    }
    if (params.return_softmax_lse && tidx < valid_rows) {
      const int logical_row = m_block * kBlockM + tidx;
      const index_t lse_offset =
          params.packed_decode_gqa
              ? index_t(bidh * params.seqlen_q + logical_row) * params.total_q +
                    bidb
              : index_t(bidh) * params.total_q +
                    binfo.q_offset(params.seqlen_q, 1, bidb) + logical_row;
      reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)[lse_offset] =
          INFINITY;
    }
    return;
  }

  const int* block_table =
      params.block_table + index_t(bidb) * params.block_table_batch_stride;
  const int kv_head = bidh / params.h_h_k_ratio;

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_raw)),
                          typename KernelTraits::SmemLayoutQ{});
  Tensor sK =
      make_tensor(sQ.data() + (KernelTraits::Share_Q_K_smem ? 0 : size(sQ)),
                  typename KernelTraits::SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + (UseAsync ? 0 : size(sK)),
                          typename KernelTraits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename KernelTraits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data().get(), typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sVtPart = local_tile(sVt, Shape<Int<kOutputDim>, Int<kBlockN>>{},
                              make_coord(output_partition, 0));
  Tensor sVtNoSwizzlePart =
      local_tile(sVtNoSwizzle, Shape<Int<kOutputDim>, Int<kBlockN>>{},
                 make_coord(output_partition, 0));

  // Static per-layer scales make the complete E4M3FN -> BF16 mapping a
  // 256-entry table.  Build it once per CTA and replace tens of thousands of
  // scalar decode/multiply/convert operations in every K/V tile with shared
  // memory lookups.
  constexpr int kRawSavings = CompactAsync ? kBlockN * kHeadDim : 0;
  // Decode alternates raw K and V in one buffer so three CTAs fit per SM.
  Element* k_lookup = reinterpret_cast<Element*>(
      smem_raw + KernelTraits::kSmemSize - kRawSavings);
  Element* v_lookup = k_lookup + 256;
  uint8_t* raw_k = reinterpret_cast<uint8_t*>(sK.data().get() + size(sK));
  uint8_t* raw_v = raw_k + (CompactAsync ? 0 : kBlockN * kHeadDim);
  if constexpr (CompactAsync) {
    prefetch_paged_fp8<KernelTraits, true>(params, binfo, block_table, kv_head,
                                           n_block_max - 1, raw_k);
    cute::cp_async_fence();
  }
  const float k_scale = __ldg(params.k_scale_ptr);
  const float v_scale = __ldg(params.v_scale_ptr);
  for (int bits = tidx; bits < 256; bits += KernelTraits::kNThreads) {
    k_lookup[bits] =
        Element(decode_e4m3fn(static_cast<uint8_t>(bits)) * k_scale);
    v_lookup[bits] =
        Element(decode_e4m3fn(static_cast<uint8_t>(bits)) * v_scale);
  }
  load_q_qdq<KernelTraits, Element>(params, binfo, bidh, m_block, sQ);
  __syncthreads();

  typename KernelTraits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(mma_thread);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzlePart);
  Tensor acc_o =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kOutputDim>>{});

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(mma_thread);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(mma_thread);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);
  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename KernelTraits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(mma_thread);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVtPart);

  if constexpr (KernelTraits::Is_Q_in_regs) {
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    // sK aliases sQ in this configuration.  All warps must finish reading
    // Q before the first FP8 K tile overwrites the shared-memory buffer.
    __syncthreads();
  }

  clear(acc_o);
  FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
  FLASH_NAMESPACE::Mask<true, false, false> causal_mask(
      binfo.actual_seqlen_k, binfo.actual_seqlen_q, -1, 0, 0.0f);
  FLASH_NAMESPACE::Mask<false, false, false> full_mask(
      binfo.actual_seqlen_k, binfo.actual_seqlen_q, -1, -1, 0.0f);

  if constexpr (UseAsync) {
    if constexpr (!CompactAsync) {
      prefetch_paged_fp8<KernelTraits, true>(params, binfo, block_table,
                                             kv_head, n_block_max - 1, raw_k);
      prefetch_paged_fp8<KernelTraits, false>(params, binfo, block_table,
                                              kv_head, n_block_max - 1, raw_v);
      cute::cp_async_fence();
    }
    cute::cp_async_wait<0>();
    __syncthreads();
  }
  bool first = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    if constexpr (UseAsync) {
      decode_shared_fp8<KernelTraits>(raw_k, sK, k_lookup);
    } else {
      load_paged_kv_fp8<KernelTraits, Element>(params, binfo, block_table,
                                               kv_head, n_block, sK, sV,
                                               k_lookup, v_lookup);
    }
    __syncthreads();
    if constexpr (UseAsync) {
      if constexpr (CompactAsync) {
        // K is decoded; overlap this tile's V transfer with its QK product.
        prefetch_paged_fp8<KernelTraits, false>(params, binfo, block_table,
                                                kv_head, n_block, raw_v);
        cute::cp_async_fence();
      } else if (n_block > n_block_min) {
        prefetch_paged_fp8<KernelTraits, true>(params, binfo, block_table,
                                               kv_head, n_block - 1, raw_k);
        cute::cp_async_fence();
      }
    }

    Tensor acc_s =
        partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    FLASH_NAMESPACE::gemm<KernelTraits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    if constexpr (UseAsync) {
      if constexpr (CompactAsync) {
        cute::cp_async_wait<0>();
      }
      __syncthreads();
      decode_shared_fp8<KernelTraits>(raw_v, sV, v_lookup);
      __syncthreads();
      if (n_block > n_block_min) {
        // Compact decode reuses the raw buffer for the next K during PV.
        prefetch_paged_fp8<KernelTraits, CompactAsync>(
            params, binfo, block_table, kv_head, n_block - 1, raw_v);
        cute::cp_async_fence();
      }
    }

    if (params.is_causal) {
      causal_mask.template apply_mask<true, false>(
          acc_s, n_block * kBlockN,
          m_block * kBlockM + (mma_thread / 32) * 16 + (tidx % 32) / 4,
          kMmaWarps * 16);
    } else {
      full_mask.template apply_mask<false, false>(
          acc_s, n_block * kBlockN,
          m_block * kBlockM + (mma_thread / 32) * 16 + (tidx % 32) / 4,
          kMmaWarps * 16);
    }

    if (first) {
      softmax.template softmax_rescale_o<true, true>(acc_s, acc_o,
                                                     params.scale_softmax_log2);
      first = false;
    } else {
      softmax.template softmax_rescale_o<false, true>(
          acc_s, acc_o, params.scale_softmax_log2);
    }

    Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(
        rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<
                       typename KernelTraits::TiledMma>(rP.layout()));
    FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
                             smem_tiled_copy_V, smem_thr_copy_V);
    if constexpr (UseAsync) {
      cute::cp_async_wait<0>();
    }
    __syncthreads();
  }

  Tensor lse = softmax.template normalize_softmax_lse<false, false>(
      acc_o, params.scale_softmax);
  Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_raw)),
                          typename KernelTraits::SmemLayoutO{});
  Tensor sOPart = local_tile(sO, Shape<Int<kBlockM>, Int<kOutputDim>>{},
                             make_coord(0, output_partition));
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename KernelTraits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(mma_thread);
  Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sOPart);
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  const index_t row_offset_o =
      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
      index_t(m_block * kBlockM) * params.o_row_stride +
      index_t(bidh) * params.o_head_stride;
  Tensor gO = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.o_row_stride, _1{}));
  typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  __syncthreads();
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kOutputDim>>{});
  Tensor taccOcO = thr_mma.partition_C(caccO);
  Tensor taccOcO_row =
      logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  if (params.return_softmax_lse && output_partition == 0 &&
      get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        const int logical_row = m_block * kBlockM + row;
        const index_t lse_offset =
            params.packed_decode_gqa
                ? index_t(bidh * params.seqlen_q + logical_row) *
                          params.total_q +
                      bidb
                : index_t(bidh) * params.total_q +
                      binfo.q_offset(params.seqlen_q, 1, bidb) + logical_row;
        reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)[lse_offset] =
            lse(mi);
      }
    }
  }

  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  FLASH_NAMESPACE::copy<false, true, false, false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
      binfo.actual_seqlen_q - m_block * kBlockM);
}

}  // namespace FLASH_NAMESPACE
