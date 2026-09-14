// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// H20 SiLU, FP8 quantization, and native WGMMA FC2 fusion.
// Read the original encoded INT4 weights directly; no extra weight allocation.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include <array>
#include <cub/block/block_reduce.cuh>
#include "libtorch_stable/cuda_vec_utils.cuh"
#include "quantization/w8a8/fp8/common.cuh"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"

namespace h20_fused_fc2_encoded {
using torch::stable::Tensor;
using ST = torch::headeronly::ScalarType;
using Atom = decltype(cutlass::compute_memory_reordering_atom<cutlass::float_e4m3_t>());

__device__ __forceinline__ uint32_t lookup4(uint32_t nibble, uint32_t lo, uint32_t hi) {
  // Identical LUT bytes to CUTLASS mixed_input_utils: the positive zero
  // entry is 0x80. Other positive entries flip the negative table sign bit.
  uint32_t poslo = (lo & 0xffffff00u) ^ 0x80808080u;
  uint32_t poshi = hi ^ 0x80808080u;
  uint32_t index = nibble & 0x7777u;
  uint32_t selector = ((nibble & 0x8888u) | 0x6420u) >> 1;
  uint32_t result;
  asm volatile(
      "{ .reg .b32 p, n;\n"
      "prmt.b32 n, %1, %2, %5;\n"
      "prmt.b32 p, %3, %4, %5;\n"
      "prmt.b32 %0, p, n, %6; }\n"
      : "=r"(result) : "r"(lo), "r"(hi), "r"(poslo), "r"(poshi),
        "r"(index), "r"(selector));
  return result;
}


using Fp8 = cutlass::float_e4m3_t;
using QuantFp8 = torch::headeronly::Float8_e4m3fn;
using Mma = decltype(cute::make_tiled_mma(
    cute::SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_RS_TN<>{}));
using BLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_SW128_Atom<Fp8>{}, cute::Shape<cute::_8, cute::_256>{}));

struct Maximum {
  __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

template<bool Fused>
__global__ void fused_fc2_kernel(const int64_t* ptr, int pitch,
                                const int* problems, const __nv_bfloat162* input,
                                const __nv_bfloat16* output_base) {
  using namespace cute;
  const int group = blockIdx.y;
  const int count = problems[group * 3 + 1];
  if (count == 0) return;
  const int tid = threadIdx.x;
  const int channel_start = blockIdx.x * 64;
  auto encoded_layout = cute::tile_to_shape(Atom{},
      cute::make_shape(cute::Int<2048>{}, cute::Int<256>{}, cute::Int<1>{}));
  const auto* weight = reinterpret_cast<const uint8_t*>(ptr[pitch + group]);
  auto* output = reinterpret_cast<__nv_bfloat16*>(ptr[2 * pitch + group]);
  const auto* channel = reinterpret_cast<const float*>(ptr[4 * pitch + group]);
  const auto* lookup = reinterpret_cast<const uint64_t*>(ptr[5 * pitch + group]);
  const int64_t first_row = (output - output_base) / 2048;
  __shared__ __align__(128) uint8_t activation[cosize_v<BLayout>];
  __shared__ float scales[8];
  using Reduce = cub::BlockReduce<float, 128>;
  __shared__ typename Reduce::TempStorage reduction;
  BLayout layout;
  if constexpr (Fused) {
    for (int row = 0; row < count; ++row) {
      const auto gate = input[(first_row + row) * 256 + tid];
      const auto up = input[(first_row + row) * 256 + 128 + tid];
      float2 x = vllm::cast_to_float2(gate);
      const float2 u = vllm::cast_to_float2(up);
      x.x = x.x / (1.0f + expf(-x.x * 1.0f));
      x.y = x.y / (1.0f + expf(-x.y * 1.0f));
      x = vllm::cast_to_float2(vllm::cast_to_packed<__nv_bfloat162>(x));
      x.x *= u.x + 0.0f;
      x.y *= u.y + 0.0f;
      x = vllm::cast_to_float2(vllm::cast_to_packed<__nv_bfloat162>(x));
      float amax = fmaxf(fmaxf(0.0f, fabsf(x.x)), fabsf(x.y));
      float maximum = Reduce(reduction).Reduce(amax, Maximum{});
      if (tid == 0) {
        scales[row] = fmaxf(maximum / quant_type_max_v<QuantFp8>,
                           min_scaling_factor<QuantFp8>::val());
      }
      __syncthreads();
      auto a = vllm::scaled_fp8_conversion<false, QuantFp8>(x.x, scales[row]);
      auto b = vllm::scaled_fp8_conversion<false, QuantFp8>(x.y, scales[row]);
      activation[layout(row, 2 * tid)] = *reinterpret_cast<uint8_t*>(&a);
      activation[layout(row, 2 * tid + 1)] = *reinterpret_cast<uint8_t*>(&b);
      __syncthreads();
    }
    for (int i = count * 256 + tid; i < 8 * 256; i += 128) {
      activation[layout(i / 256, i % 256)] = 0;
    }
  } else {
    const auto* a = reinterpret_cast<const uint8_t*>(ptr[group]);
    const auto* token = reinterpret_cast<const float*>(ptr[3 * pitch + group]);
    for (int i = tid; i < 8 * 256; i += 128) {
      activation[layout(i / 256, i % 256)] = i / 256 < count ? a[i] : 0;
    }
    if (tid < count) scales[tid] = token[tid];
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  auto sB = make_tensor(make_smem_ptr(reinterpret_cast<Fp8*>(activation)), layout);
  Mma mma;
  auto thr = mma.get_thread_slice(tid);
  auto b_desc = thr.make_fragment_B(thr.partition_B(sB));
  auto a_shape = make_layout(Shape<_64, _128>{}, Stride<_128, _1>{});
  auto a_storage = make_tensor(make_gmem_ptr(reinterpret_cast<const Fp8*>(weight)), a_shape);
  auto reg_a = thr.partition_fragment_A(a_storage);
  auto coord_a = thr.partition_A(make_identity_tensor(Shape<_64, _128>{}));
  auto coord_c = thr.partition_C(make_identity_tensor(Shape<_64, _8>{}));
  auto accum = partition_fragment_C(mma, Shape<_64, _8>{});
  clear(accum);
  mma.accumulate_ = GMMA::ScaleOut::Zero;
  for (int kb = 0; kb < 256; kb += 128) {
    auto reg_words = recast<uint32_t>(reg_a);
    const auto* weight_words = reinterpret_cast<const uint16_t*>(weight);
    CUTE_UNROLL
    for (int i = 0; i < size(reg_words); ++i) {
      const int row = channel_start + get<0>(coord_a(4 * i));
      const int column = kb + get<1>(coord_a(4 * i));
      const uint32_t nibbles = weight_words[encoded_layout(make_coord(row, column, 0)) / 4];
      const uint64_t table = lookup[(kb / 128) * 2048 + row];
      reg_words(i) = lookup4(nibbles, uint32_t(table), uint32_t(table >> 32));
    }
    CUTE_UNROLL
    for (int sub = 0; sub < 4; ++sub) {
      warpgroup_arrive();
      cute::gemm(mma, reg_a(_, _, sub), b_desc(_, _, kb / 32 + sub), accum);
      mma.accumulate_ = GMMA::ScaleOut::One;
      warpgroup_commit_batch();
    }
    warpgroup_wait<0>();
  }
  CUTE_UNROLL
  for (int i = 0; i < size(accum); ++i) {
    const int row = get<1>(coord_c(i));
    const int col = channel_start + get<0>(coord_c(i));
    if (row < count) {
      output[int64_t(row) * 2048 + col] = __float2bfloat16_rn(
          __fmul_rn(channel[col], __fmul_rn(scales[row], accum(i))));
    }
  }
}

bool validate_fragment_mapping() {
  using namespace cute;
  Mma mma;
  auto encoded = tile_to_shape(Atom{}, make_shape(Int<2048>{}, Int<256>{}, Int<1>{}));
  for (int tid = 0; tid < 128; ++tid) {
    auto c = mma.get_thread_slice(tid).partition_A(
        make_identity_tensor(Shape<_64, _128>{}));
    for (int i = 0; i < size(c); i += 4) {
      STD_TORCH_CHECK(get<1>(c(i)) % 4 == 0);
      for (int ch = 0; ch < 2048; ch += 64) {
        for (int kb : {0, 128}) {
          const int first = encoded(make_coord(ch + get<0>(c(i)), kb + get<1>(c(i)), 0));
          STD_TORCH_CHECK(first % 4 == 0);
          for (int j = 1; j < 4; ++j) {
            STD_TORCH_CHECK(encoded(make_coord(ch + get<0>(c(i+j)), kb + get<1>(c(i+j)), 0)) == first + j,
                            "Encoded INT4 quartet is not contiguous");
          }
        }
      }
      for (int j = 1; j < 4; ++j) {
        STD_TORCH_CHECK(get<0>(c(i + j)) == get<0>(c(i)) &&
                        get<1>(c(i + j)) == get<1>(c(i)) + j,
                        "WGMMA FP8 register fragment is not a contiguous four-byte group");
      }
    }
  }
  return true;
}

void run(Tensor& out, const Tensor& input, const Tensor& a, const Tensor& b,
         const Tensor& token, const Tensor& channel, const Tensor& scale,
         const Tensor& problems, const Tensor& ptrs, bool fused) {
  static const bool mapping_valid = validate_fragment_mapping();
  STD_TORCH_CHECK(mapping_valid);
  const auto device = out.device();
  const std::array<const Tensor*, 9> tensors{
      &out, &input, &a, &b, &token, &channel, &scale, &problems, &ptrs};
  for (auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->is_cuda() && tensor->device() == device && tensor->is_contiguous());
  }
  STD_TORCH_CHECK(out.scalar_type() == ST::BFloat16 && out.dim() == 2 && out.size(0) == 8 && out.size(1) == 2048);
  STD_TORCH_CHECK(input.scalar_type() == ST::BFloat16 && input.dim() == 2 && input.size(0) == 8 && input.size(1) == 512);
  STD_TORCH_CHECK(a.scalar_type() == ST::Float8_e4m3fn && a.dim() == 2 && a.size(0) == 8 && a.size(1) == 256);
  STD_TORCH_CHECK(b.scalar_type() == ST::Int && b.dim() == 3 && b.size(0) == 256 && b.size(1) == 2048 && b.size(2) == 32);
  STD_TORCH_CHECK(token.scalar_type() == ST::Float && token.numel() == 8);
  STD_TORCH_CHECK(channel.scalar_type() == ST::Float && channel.numel() == 256 * 2048);
  STD_TORCH_CHECK(scale.scalar_type() == ST::Float8_e4m3fn && scale.numel() == 256 * 2 * 2048 * 8);
  STD_TORCH_CHECK(problems.scalar_type() == ST::Int && problems.dim() == 2 && problems.size(0) == 8 && problems.size(1) == 3);
  STD_TORCH_CHECK(ptrs.scalar_type() == ST::Long && ptrs.dim() == 2 && ptrs.size(0) == 6 && ptrs.size(1) == 8);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(ptrs.data_ptr()) % 16 == 0);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 4 == 0);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(b.data_ptr()) % 2 == 0);
  auto output_ptr = reinterpret_cast<uintptr_t>(out.data_ptr());
  auto input_ptr = reinterpret_cast<uintptr_t>(input.data_ptr());
  STD_TORCH_CHECK(output_ptr + out.numel() * 2 <= input_ptr || input_ptr + input.numel() * 2 <= output_ptr);
  const torch::stable::accelerator::DeviceGuard guard(device.index());
  auto stream = get_current_cuda_stream(device.index());
  STD_TORCH_CHECK(fused, "H20 FC2 uses the selected fused kernel");
  auto kernel = fused_fc2_kernel<true>;
  kernel<<<dim3(32, 8), 128, 0, stream>>>(
      static_cast<const int64_t*>(ptrs.data_ptr()), 8,
      static_cast<const int*>(problems.data_ptr()),
      static_cast<const __nv_bfloat162*>(input.data_ptr()),
      static_cast<const __nv_bfloat16*>(out.data_ptr()));
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "native WGMMA FC2 launch failed");
}

STABLE_TORCH_LIBRARY(h20_fused_fc2_encoded, m) {
  m.def("run(Tensor! out, Tensor input, Tensor a, Tensor b, Tensor token, Tensor channel, Tensor scale, Tensor problems, Tensor ptrs, bool fused) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_fused_fc2_encoded, CUDA, m) {
  m.impl("run", TORCH_BOX(&run));
}
}  // namespace h20_fused_fc2_encoded
