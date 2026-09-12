#include <cutlass/numeric_types.h>

#include "flash_fwd_sm80_fp8_kernel.h"
#include "cuda_check.h"
#include "hardware_info.h"
#include "static_switch.h"

namespace FLASH_NAMESPACE {

using Sm80Fp8Traits =
    Flash_fwd_kernel_traits<256, 64, 64, 4, true, true, cutlass::bfloat16_t>;

// Each warp preserves FA2's complete score/softmax reduction, while owning
// a partition of the output columns. K/V loading uses the entire CTA.
template <int NWarps>
struct Sm80Fp8DecodeTraitsBase
    : Flash_fwd_kernel_traits<256, 16, 64, NWarps, true, true,
                              cutlass::bfloat16_t> {
  using Base = Flash_kernel_traits<256, 16, 64, NWarps, cutlass::bfloat16_t>;
  using TiledMma =
      cute::TiledMMA<typename Base::MMA_Atom_Arch,
                     cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
                     cute::Tile<cute::_16, cute::_16, cute::_16>>;
};

using Sm80Fp8DecodeTraits = Sm80Fp8DecodeTraitsBase<4>;
using Sm80Fp8CompactDecodeTraits = Sm80Fp8DecodeTraitsBase<2>;

__global__ void flash_fwd_sm80_fp8_kernel(
    const Flash_fwd_sm80_fp8_params params) {
  compute_attn_sm80_fp8<Sm80Fp8Traits, false, true>(params);
}

__global__ void flash_fwd_sm80_fp8_unaligned_kernel(
    const Flash_fwd_sm80_fp8_params params) {
  compute_attn_sm80_fp8<Sm80Fp8Traits>(params);
}

__global__ __launch_bounds__(
    Sm80Fp8CompactDecodeTraits::kNThreads,
    3) void flash_decode_sm80_fp8_kernel(const Flash_fwd_sm80_fp8_params
                                             params) {
  compute_attn_sm80_fp8<Sm80Fp8CompactDecodeTraits, true, true, true>(params);
}

__global__ void flash_decode_sm80_fp8_small_grid_kernel(
    const Flash_fwd_sm80_fp8_params params) {
  compute_attn_sm80_fp8<Sm80Fp8DecodeTraits, true, true>(params);
}

void run_mha_fwd_sm80_fp8(Flash_fwd_sm80_fp8_params& params,
                          cudaStream_t stream) {
  // Retain the original eight-byte loader for KV views that cannot use
  // sixteen-byte cp.async transfers. The backend's normal cache is aligned.
  const uint64_t kv_alignment = reinterpret_cast<uintptr_t>(params.k_ptr) |
                                reinterpret_cast<uintptr_t>(params.v_ptr) |
                                params.k_batch_stride | params.k_row_stride |
                                params.k_head_stride | params.v_batch_stride |
                                params.v_row_stride | params.v_head_stride;
  if ((kv_alignment & 15U) != 0) {
    constexpr size_t fallback_smem_size =
        Sm80Fp8Traits::kSmemSize + 2 * 256 * sizeof(cutlass::bfloat16_t);
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        flash_fwd_sm80_fp8_unaligned_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, fallback_smem_size));
    const dim3 fallback_grid(
        (params.seqlen_q + Sm80Fp8Traits::kBlockM - 1) / Sm80Fp8Traits::kBlockM,
        params.b, params.h);
    flash_fwd_sm80_fp8_unaligned_kernel<<<
        fallback_grid, Sm80Fp8Traits::kNThreads, fallback_smem_size, stream>>>(
        params);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (params.seqlen_q <= 16) {
    // For smaller grids, spills cost more than the extra CTA residency.
    // Retain separate raw K/V buffers and the unconstrained register budget.
    if (params.b * params.h < 256) {
      constexpr size_t small_grid_smem_size =
          Sm80Fp8DecodeTraits::kSmemSize +
          2 * 256 * sizeof(cutlass::bfloat16_t);
      FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
          flash_decode_sm80_fp8_small_grid_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize, small_grid_smem_size));
      flash_decode_sm80_fp8_small_grid_kernel<<<dim3(1, params.b, params.h),
                                                Sm80Fp8DecodeTraits::kNThreads,
                                                small_grid_smem_size, stream>>>(
          params);
      FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    constexpr size_t decode_smem_size =
        Sm80Fp8CompactDecodeTraits::kSmemSize -
        Sm80Fp8CompactDecodeTraits::kBlockN *
            Sm80Fp8CompactDecodeTraits::kHeadDim +
        2 * 256 * sizeof(cutlass::bfloat16_t);
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        flash_decode_sm80_fp8_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, decode_smem_size));
    flash_decode_sm80_fp8_kernel<<<dim3(1, params.b, params.h),
                                   Sm80Fp8CompactDecodeTraits::kNThreads,
                                   decode_smem_size, stream>>>(params);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  constexpr size_t smem_size =
      Sm80Fp8Traits::kSmemSize + 2 * 256 * sizeof(cutlass::bfloat16_t);
  const dim3 grid(
      (params.seqlen_q + Sm80Fp8Traits::kBlockM - 1) / Sm80Fp8Traits::kBlockM,
      params.b, params.h);
  if (smem_size >= 48 * 1024) {
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        flash_fwd_sm80_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
  }
  flash_fwd_sm80_fp8_kernel<<<grid, Sm80Fp8Traits::kNThreads, smem_size,
                              stream>>>(params);
  FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void flash_fwd_sm80_fp8_kernel_prequantized_q(
    const Flash_fwd_sm80_fp8_params params) {
  compute_attn_sm80_fp8<Sm80Fp8Traits, false, true, false, true>(params);
}

__global__ void flash_fwd_sm80_fp8_unaligned_kernel_prequantized_q(
    const Flash_fwd_sm80_fp8_params params) {
  compute_attn_sm80_fp8<Sm80Fp8Traits, false, false, false, true>(params);
}

__global__ __launch_bounds__(
    Sm80Fp8CompactDecodeTraits::kNThreads,
    3) void flash_decode_sm80_fp8_kernel_prequantized_q(const Flash_fwd_sm80_fp8_params
                                                            params) {
  compute_attn_sm80_fp8<Sm80Fp8CompactDecodeTraits, true, true, true, true>(
      params);
}

__global__ void flash_decode_sm80_fp8_small_grid_kernel_prequantized_q(
    const Flash_fwd_sm80_fp8_params params) {
  compute_attn_sm80_fp8<Sm80Fp8DecodeTraits, true, true, false, true>(params);
}

void run_mha_fwd_sm80_fp8_prequantized_q(Flash_fwd_sm80_fp8_params& params,
                                         cudaStream_t stream) {
  // Retain the original eight-byte loader for KV views that cannot use
  // sixteen-byte cp.async transfers. The backend's normal cache is aligned.
  const uint64_t kv_alignment = reinterpret_cast<uintptr_t>(params.k_ptr) |
                                reinterpret_cast<uintptr_t>(params.v_ptr) |
                                params.k_batch_stride | params.k_row_stride |
                                params.k_head_stride | params.v_batch_stride |
                                params.v_row_stride | params.v_head_stride;
  if ((kv_alignment & 15U) != 0) {
    constexpr size_t fallback_smem_size =
        Sm80Fp8Traits::kSmemSize + 2 * 256 * sizeof(cutlass::bfloat16_t);
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        flash_fwd_sm80_fp8_unaligned_kernel_prequantized_q,
        cudaFuncAttributeMaxDynamicSharedMemorySize, fallback_smem_size));
    const dim3 fallback_grid(
        (params.seqlen_q + Sm80Fp8Traits::kBlockM - 1) / Sm80Fp8Traits::kBlockM,
        params.b, params.h);
    flash_fwd_sm80_fp8_unaligned_kernel_prequantized_q<<<
        fallback_grid, Sm80Fp8Traits::kNThreads, fallback_smem_size, stream>>>(
        params);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (params.seqlen_q <= 16) {
    // For smaller grids, spills cost more than the extra CTA residency.
    // Retain separate raw K/V buffers and the unconstrained register budget.
    if (params.b * params.h < 256) {
      constexpr size_t small_grid_smem_size =
          Sm80Fp8DecodeTraits::kSmemSize +
          2 * 256 * sizeof(cutlass::bfloat16_t);
      FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
          flash_decode_sm80_fp8_small_grid_kernel_prequantized_q,
          cudaFuncAttributeMaxDynamicSharedMemorySize, small_grid_smem_size));
      flash_decode_sm80_fp8_small_grid_kernel_prequantized_q<<<
          dim3(1, params.b, params.h), Sm80Fp8DecodeTraits::kNThreads,
          small_grid_smem_size, stream>>>(params);
      FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    constexpr size_t decode_smem_size =
        Sm80Fp8CompactDecodeTraits::kSmemSize -
        Sm80Fp8CompactDecodeTraits::kBlockN *
            Sm80Fp8CompactDecodeTraits::kHeadDim +
        2 * 256 * sizeof(cutlass::bfloat16_t);
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        flash_decode_sm80_fp8_kernel_prequantized_q,
        cudaFuncAttributeMaxDynamicSharedMemorySize, decode_smem_size));
    flash_decode_sm80_fp8_kernel_prequantized_q<<<
        dim3(1, params.b, params.h), Sm80Fp8CompactDecodeTraits::kNThreads,
        decode_smem_size, stream>>>(params);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  constexpr size_t smem_size =
      Sm80Fp8Traits::kSmemSize + 2 * 256 * sizeof(cutlass::bfloat16_t);
  const dim3 grid(
      (params.seqlen_q + Sm80Fp8Traits::kBlockM - 1) / Sm80Fp8Traits::kBlockM,
      params.b, params.h);
  if (smem_size >= 48 * 1024) {
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        flash_fwd_sm80_fp8_kernel_prequantized_q,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  flash_fwd_sm80_fp8_kernel_prequantized_q<<<grid, Sm80Fp8Traits::kNThreads,
                                             smem_size, stream>>>(params);
  FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
}

// Materialize each Q value and each referenced KV page once per prefill.
// FP8 QDQ arithmetic is shared with the existing kernel, including signed zero.
__global__ void stage_sm80_fp8_prefill_q(const Flash_fwd_sm80_fp8_params params,
                                         cutlass::bfloat16_t* staged_q) {
  constexpr int kVec = 8;
  const int64_t vec = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t head_row = vec / (256 / kVec);
  if (head_row >= int64_t(params.total_q) * params.h) return;
  const int row = head_row / params.h;
  const int head = head_row % params.h;
  const int col = (vec % (256 / kVec)) * kVec;
  const auto* q = static_cast<const cutlass::bfloat16_t*>(params.q_ptr);
  const int64_t src = int64_t(row) * params.q_row_stride +
                      int64_t(head) * params.q_head_stride + col;
  using Vector = cutlass::Array<cutlass::bfloat16_t, kVec>;
  const auto values = *reinterpret_cast<const Vector*>(q + src);
  alignas(16) Vector converted;
  const float scale = __ldg(params.q_scale_ptr);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    converted[i] =
        cutlass::bfloat16_t(scaled_e4m3fn_qdq(float(values[i]), scale));
  }
  *reinterpret_cast<uint4*>(staged_q + vec * kVec) =
      *reinterpret_cast<const uint4*>(&converted);
}

__global__ void stage_sm80_fp8_prefill_kv(
    const Flash_fwd_sm80_fp8_params params, cutlass::bfloat16_t* staged_k,
    cutlass::bfloat16_t* staged_v, int* staged_table, int pages) {
  const int page = blockIdx.x;
  const int batch = blockIdx.y;
  const int head = blockIdx.z;
  const int compact_page = batch * pages + page;
  if (head == 0 && threadIdx.x == 0) staged_table[compact_page] = compact_page;
  const int length = params.seqused_k[batch];
  if (page * 16 >= length) return;
  const int physical_page =
      params
          .block_table[int64_t(batch) * params.block_table_batch_stride + page];
  const auto* k = static_cast<const uint8_t*>(params.k_ptr);
  const auto* v = static_cast<const uint8_t*>(params.v_ptr);
  const float k_scale = __ldg(params.k_scale_ptr);
  const float v_scale = __ldg(params.v_scale_ptr);
  constexpr int kVec = 8;
  for (int vec = threadIdx.x; vec < 16 * 256 / kVec; vec += blockDim.x) {
    const int row = vec / (256 / kVec);
    const int col = (vec % (256 / kVec)) * kVec;
    if (page * 16 + row < length) {
      const int64_t k_offset = int64_t(physical_page) * params.k_batch_stride +
                               int64_t(row) * params.k_row_stride +
                               int64_t(head) * params.k_head_stride + col;
      const int64_t v_offset = int64_t(physical_page) * params.v_batch_stride +
                               int64_t(row) * params.v_row_stride +
                               int64_t(head) * params.v_head_stride + col;
      const int64_t out =
          ((int64_t(compact_page) * 16 + row) * params.h_k + head) * 256 + col;
      const uint64_t k_bits = *reinterpret_cast<const uint64_t*>(k + k_offset);
      const uint64_t v_bits = *reinterpret_cast<const uint64_t*>(v + v_offset);
      alignas(16) cutlass::Array<cutlass::bfloat16_t, kVec> k_values, v_values;
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        k_values[i] = cutlass::bfloat16_t(
            decode_e4m3fn(static_cast<uint8_t>(k_bits >> (8 * i))) * k_scale);
        v_values[i] = cutlass::bfloat16_t(
            decode_e4m3fn(static_cast<uint8_t>(v_bits >> (8 * i))) * v_scale);
      }
      *reinterpret_cast<uint4*>(staged_k + out) =
          *reinterpret_cast<const uint4*>(&k_values);
      *reinterpret_cast<uint4*>(staged_v + out) =
          *reinterpret_cast<const uint4*>(&v_values);
    }
  }
}

using Sm80StagedPrefillTraits =
    Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template <bool IsCausal>
__global__ __launch_bounds__(
    Sm80StagedPrefillTraits::
        kNThreads) void flash_fwd_sm80_staged_prefill_kernel(const Flash_fwd_params
                                                                 params) {
  compute_attn_splitkv<Sm80StagedPrefillTraits, IsCausal, false, false, false,
                       true, false, false, false>(params);
}

void run_mha_fwd_sm80_fp8_staged(const Flash_fwd_sm80_fp8_params& source,
                                 void* q, void* k, void* v, int* block_table,
                                 int pages_per_sequence, cudaStream_t stream,
                                 bool prequantized_q) {
  if (!prequantized_q) {
    const int64_t q_vectors = int64_t(source.total_q) * source.h * (256 / 8);
    stage_sm80_fp8_prefill_q<<<(q_vectors + 127) / 128, 128, 0, stream>>>(
        source, static_cast<cutlass::bfloat16_t*>(q));
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
  }
  stage_sm80_fp8_prefill_kv<<<dim3(pages_per_sequence, source.b, source.h_k),
                              128, 0, stream>>>(
      source, static_cast<cutlass::bfloat16_t*>(k),
      static_cast<cutlass::bfloat16_t*>(v), block_table, pages_per_sequence);
  FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
  Flash_fwd_params params = source;
  params.q_ptr = prequantized_q ? source.q_ptr : q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.q_row_stride = prequantized_q ? source.q_row_stride : params.h * 256;
  params.q_head_stride = prequantized_q ? source.q_head_stride : 256;
  params.k_batch_stride = params.v_batch_stride = 16 * params.h_k * 256;
  params.k_row_stride = params.v_row_stride = params.h_k * 256;
  params.k_head_stride = params.v_head_stride = 256;
  params.block_table = block_table;
  params.block_table_batch_stride = pages_per_sequence;
  params.num_splits = 1;
  constexpr size_t smem_size = Sm80StagedPrefillTraits::kSmemSize;
  const dim3 grid((params.seqlen_q + 63) / 64, params.b, params.h);
  BOOL_SWITCH(params.is_causal, IsCausal, [&] {
    auto kernel = &flash_fwd_sm80_staged_prefill_kernel<IsCausal>;
    FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<grid, Sm80StagedPrefillTraits::kNThreads, smem_size, stream>>>(
        params);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace FLASH_NAMESPACE
