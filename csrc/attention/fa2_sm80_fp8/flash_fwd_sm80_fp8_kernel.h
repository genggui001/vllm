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
    return __uint_as_float(sign | ((exponent + 120U) << 23U) |
                           (mantissa << 20U));
}

__forceinline__ __device__ float scaled_e4m3fn_qdq(float value,
                                                    float scale) {
    const float normalized = fminf(fmaxf(__fdiv_rn(value, scale), -448.0f),
                                   448.0f);
    __nv_fp8_e4m3 quantized(normalized);
    return static_cast<float>(quantized) * scale;
}

template <typename KernelTraits, typename Element, typename SmemTensor>
__forceinline__ __device__ void load_q_qdq(
    const Flash_fwd_sm80_fp8_params &params, const BlockInfo<true> &binfo,
    int bidh, int m_block, SmemTensor &sQ) {
    constexpr int kHeadDim = 256;
    constexpr int kVec = 8;
    constexpr int kVecsPerRow = kHeadDim / kVec;
    constexpr int kNThreads = KernelTraits::kNThreads;
    constexpr int kBlockM = KernelTraits::kBlockM;

    const int valid_rows =
        min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
    const Element *q = reinterpret_cast<const Element *>(params.q_ptr);
    const int64_t q_base =
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, blockIdx.y) +
        int64_t(m_block * kBlockM) * params.q_row_stride +
        int64_t(bidh) * params.q_head_stride;
    const float scale = __ldg(params.q_scale_ptr);

    for (int vec = threadIdx.x; vec < valid_rows * kVecsPerRow;
         vec += kNThreads) {
        const int row = vec / kVecsPerRow;
        const int col = (vec % kVecsPerRow) * kVec;
        const auto values = *reinterpret_cast<const cutlass::Array<Element, kVec> *>(
            q + q_base + int64_t(row) * params.q_row_stride + col);
        alignas(16) cutlass::Array<Element, kVec> q_qdq;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            q_qdq[i] = Element(
                scaled_e4m3fn_qdq(static_cast<float>(values[i]), scale));
        }
        // Swizzle<3, 3, 3> only permutes 16-byte vectors.  Values within an
        // eight-BF16 vector remain contiguous, so evaluate the CuTe layout once
        // and issue one 128-bit shared-memory store instead of eight scalar
        // stores (and eight dynamic swizzle address calculations).
        const auto smem_offset =
            sQ.layout()(make_coord(row, col));
        *reinterpret_cast<uint4 *>(sQ.data().get() + smem_offset) =
            *reinterpret_cast<const uint4 *>(&q_qdq);
    }
}

template <typename KernelTraits, typename Element, typename SmemTensorK,
          typename SmemTensorV>
__forceinline__ __device__ void load_paged_kv_fp8(
    const Flash_fwd_sm80_fp8_params &params, const BlockInfo<true> &binfo,
    const int *block_table, int kv_head, int n_block, SmemTensorK &sK,
    SmemTensorV &sV, const Element *k_lookup, const Element *v_lookup) {
    constexpr int kVec = 8;
    constexpr int kNThreads = KernelTraits::kNThreads;
    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kPageSize = 16;
    constexpr int kPagesPerTile = kBlockN / kPageSize;
    constexpr int kNWarps = kNThreads / 32;
    static_assert(kBlockN % kPageSize == 0);

    const uint8_t *k = reinterpret_cast<const uint8_t *>(params.k_ptr);
    const uint8_t *v = reinterpret_cast<const uint8_t *>(params.v_ptr);
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
        const int64_t k_page_base =
            int64_t(physical_page) * params.k_batch_stride +
            int64_t(kv_head) * params.k_head_stride;
        const int64_t v_page_base =
            int64_t(physical_page) * params.v_batch_stride +
            int64_t(kv_head) * params.v_head_stride;

        for (int page_offset = warp; page_offset < kPageSize;
             page_offset += kNWarps) {
            const int row = page * kPageSize + page_offset;
            const int token = page_token + page_offset;
            uint64_t k_bits = 0;
            uint64_t v_bits = 0;
            if (token < binfo.actual_seqlen_k) {
                const int64_t k_offset =
                    k_page_base +
                    int64_t(page_offset) * params.k_row_stride + col;
                const int64_t v_offset =
                    v_page_base +
                    int64_t(page_offset) * params.v_row_stride + col;
                k_bits = *reinterpret_cast<const uint64_t *>(k + k_offset);
                v_bits = *reinterpret_cast<const uint64_t *>(v + v_offset);
            }

            alignas(16) cutlass::Array<Element, kVec> k_values;
            alignas(16) cutlass::Array<Element, kVec> v_values;
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const uint8_t kb =
                    static_cast<uint8_t>(k_bits >> (i * 8));
                const uint8_t vb =
                    static_cast<uint8_t>(v_bits >> (i * 8));
                k_values[i] = k_lookup[kb];
                v_values[i] = v_lookup[vb];
            }
            const auto k_smem_offset =
                sK.layout()(make_coord(row, col));
            const auto v_smem_offset =
                sV.layout()(make_coord(row, col));
            *reinterpret_cast<uint4 *>(sK.data().get() + k_smem_offset) =
                *reinterpret_cast<const uint4 *>(&k_values);
            *reinterpret_cast<uint4 *>(sV.data().get() + v_smem_offset) =
                *reinterpret_cast<const uint4 *>(&v_values);
        }
    }
}

template <typename KernelTraits>
inline __device__ void compute_attn_sm80_fp8(
    const Flash_fwd_sm80_fp8_params &params) {
    using Element = typename KernelTraits::Element;
    using ElementAccum = typename KernelTraits::ElementAccum;
    using index_t = typename KernelTraits::index_t;
    constexpr int kBlockM = KernelTraits::kBlockM;
    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kHeadDim = KernelTraits::kHeadDim;
    constexpr int kNWarps = KernelTraits::kNWarps;
    static_assert((kBlockM == 64 || kBlockM == 128) && kBlockN == 64 &&
                  kHeadDim == 256 &&
                  (KernelTraits::kNThreads == 128 ||
                   KernelTraits::kNThreads == 256));

    extern __shared__ char smem_raw[];
    const int tidx = threadIdx.x;
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
        n_block_max = min(
            n_block_max,
            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                               binfo.actual_seqlen_q,
                           kBlockN));
    }

    if (n_block_max <= n_block_min) {
        const int valid_rows =
            min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
        Element *out = reinterpret_cast<Element *>(params.o_ptr);
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
            const index_t lse_offset = params.packed_decode_gqa
                ? index_t(bidh * params.seqlen_q + logical_row) *
                          params.total_q +
                      bidb
                : index_t(bidh) * params.total_q +
                      binfo.q_offset(params.seqlen_q, 1, bidb) + logical_row;
            reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)
                [lse_offset] = INFINITY;
        }
        return;
    }

    const int *block_table =
        params.block_table + index_t(bidb) * params.block_table_batch_stride;
    const int kv_head = bidh / params.h_h_k_ratio;

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_raw)),
                            typename KernelTraits::SmemLayoutQ{});
    Tensor sK = make_tensor(
                            sQ.data() +
                                (KernelTraits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename KernelTraits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename KernelTraits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(),
                             typename KernelTraits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(
        sV.data().get(), typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});

    // Static per-layer scales make the complete E4M3FN -> BF16 mapping a
    // 256-entry table.  Build it once per CTA and replace tens of thousands of
    // scalar decode/multiply/convert operations in every K/V tile with shared
    // memory lookups.
    Element *k_lookup = sV.data().get() + size(sV);
    Element *v_lookup = k_lookup + 256;
    const float k_scale = __ldg(params.k_scale_ptr);
    const float v_scale = __ldg(params.v_scale_ptr);
    for (int bits = tidx; bits < 256; bits += KernelTraits::kNThreads) {
        k_lookup[bits] =
            Element(decode_e4m3fn(static_cast<uint8_t>(bits)) * k_scale);
        v_lookup[bits] =
            Element(decode_e4m3fn(static_cast<uint8_t>(bits)) * v_scale);
    }
    __syncthreads();

    load_q_qdq<KernelTraits, Element>(params, binfo, bidh, m_block, sQ);
    __syncthreads();

    typename KernelTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor acc_o =
        partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q =
        make_tiled_copy_A(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    auto smem_tiled_copy_K =
        make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(
        typename KernelTraits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

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

    bool first = true;
    for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
        load_paged_kv_fp8<KernelTraits, Element>(
            params, binfo, block_table, kv_head, n_block, sK, sV, k_lookup,
            v_lookup);
        __syncthreads();

        Tensor acc_s = partition_fragment_C(
            tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        FLASH_NAMESPACE::gemm<KernelTraits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
            smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (params.is_causal) {
            causal_mask.template apply_mask<true, false>(
                acc_s, n_block * kBlockN,
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                kNWarps * 16);
        } else {
            full_mask.template apply_mask<false, false>(
                acc_s, n_block * kBlockN,
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                kNWarps * 16);
        }

        if (first) {
            softmax.template softmax_rescale_o<true, true>(
                acc_s, acc_o, params.scale_softmax_log2);
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
        __syncthreads();
    }

    Tensor lse = softmax.template normalize_softmax_lse<false, false>(
        acc_o, params.scale_softmax);
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_raw)),
                            typename KernelTraits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(
        typename KernelTraits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o =
        binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) +
        index_t(m_block * kBlockM) * params.o_row_stride +
        index_t(bidh) * params.o_head_stride;
    Tensor gO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(params.o_row_stride, _1{}));
    typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    __syncthreads();
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO =
        make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row =
        logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    if (params.return_softmax_lse && get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                const int logical_row = m_block * kBlockM + row;
                const index_t lse_offset = params.packed_decode_gqa
                    ? index_t(bidh * params.seqlen_q + logical_row) *
                              params.total_q +
                          bidb
                    : index_t(bidh) * params.total_q +
                          binfo.q_offset(params.seqlen_q, 1, bidb) + logical_row;
                reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)
                    [lse_offset] = lse(mi);
            }
        }
    }

    Tensor cO = make_identity_tensor(
        make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    FLASH_NAMESPACE::copy<false, true, false, false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
}

}  // namespace FLASH_NAMESPACE
