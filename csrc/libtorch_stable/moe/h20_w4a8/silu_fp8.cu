// SPDX-License-Identifier: Apache-2.0
// Retain the existing SiLU BF16 rounding boundaries in fused FP8 quantization.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/cuda_vec_utils.cuh"
#include "quantization/w8a8/fp8/common.cuh"

namespace h20_silu_fp8 {
using Fp8 = torch::headeronly::Float8_e4m3fn;
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;

struct Maximum {
  __device__ __forceinline__ float operator()(float x, float y) const {
    return fmaxf(x, y);
  }
};

template <bool WriteIntermediate>
__global__ void silu_fp8_kernel(Fp8* __restrict__ out,
    const __nv_bfloat162* __restrict__ input, float* __restrict__ scale,
    __nv_bfloat162* __restrict__ intermediate, float alpha, float beta) {
  constexpr int Width = 256;
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const auto gate = input[row * Width + tid];
  const auto up = input[row * Width + Width / 2 + tid];
  float2 x = vllm::cast_to_float2(gate);
  const float2 u = vllm::cast_to_float2(up);
  x.x = x.x / (1.0f + expf(-x.x * alpha));
  x.y = x.y / (1.0f + expf(-x.y * alpha));
  // packed_silu_kernel rounds before packed_compute multiplies by up.
  x = vllm::cast_to_float2(vllm::cast_to_packed<__nv_bfloat162>(x));
  x.x *= u.x + beta;
  x.y *= u.y + beta;
  const auto activated = vllm::cast_to_packed<__nv_bfloat162>(x);
  x = vllm::cast_to_float2(activated);
  if constexpr (WriteIntermediate) intermediate[row * Width / 2 + tid] = activated;
  float amax = fmaxf(0.0f, fabsf(x.x));
  amax = fmaxf(amax, fabsf(x.y));
  using Reduce = cub::BlockReduce<float, 128>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float token_scale;
  const float maximum = Reduce(storage).Reduce(amax, Maximum{});
  if (tid == 0) {
    token_scale = fmaxf(maximum / quant_type_max_v<Fp8>,
                        min_scaling_factor<Fp8>::val());
    scale[row] = token_scale;
  }
  __syncthreads();
  out[row * Width + 2 * tid] =
      vllm::scaled_fp8_conversion<false, Fp8>(x.x, token_scale);
  out[row * Width + 2 * tid + 1] =
      vllm::scaled_fp8_conversion<false, Fp8>(x.y, token_scale);
}

bool overlaps(const Tensor& a, const Tensor& b) {
  const uintptr_t ab = reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t bb = reinterpret_cast<uintptr_t>(b.data_ptr());
  const uintptr_t ae = ab + a.numel() * a.element_size();
  const uintptr_t be = bb + b.numel() * b.element_size();
  return ab < be && bb < ae;
}

void run(Tensor& out, const Tensor& input, Tensor& scale,
         Tensor& intermediate, bool write_intermediate) {
  STD_TORCH_CHECK(input.is_cuda() && input.scalar_type() == Type::BFloat16);
  STD_TORCH_CHECK(input.is_contiguous() && input.dim() == 2 && input.size(1) == 512);
  STD_TORCH_CHECK(out.device() == input.device() && out.scalar_type() == Type::Float8_e4m3fn);
  STD_TORCH_CHECK(out.is_contiguous() && out.dim() == 2 &&
                  out.size(0) == input.size(0) && out.size(1) == 256);
  STD_TORCH_CHECK(scale.device() == input.device() && scale.scalar_type() == Type::Float);
  STD_TORCH_CHECK(scale.is_contiguous() && scale.numel() == input.size(0));
  STD_TORCH_CHECK(intermediate.device() == input.device() &&
                  intermediate.scalar_type() == Type::BFloat16);
  STD_TORCH_CHECK(intermediate.is_contiguous() && intermediate.sizes() == out.sizes());
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 4 == 0);
  STD_TORCH_CHECK(!overlaps(input, out) && !overlaps(input, scale) && !overlaps(out, scale),
                  "Fused SiLU+FP8 requires disjoint input/output/scale storage");
  if (write_intermediate) {
    STD_TORCH_CHECK(!overlaps(input, intermediate) && !overlaps(out, intermediate) &&
                    !overlaps(scale, intermediate));
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(intermediate.data_ptr()) % 4 == 0);
  }
  if (input.size(0) == 0) return;
  const auto device_id = input.device().index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_id);
  auto stream = get_current_cuda_stream(device_id);
  auto kernel = write_intermediate ? silu_fp8_kernel<true> : silu_fp8_kernel<false>;
  kernel<<<input.size(0), 128, 0, stream>>>(
      static_cast<Fp8*>(out.data_ptr()),
      static_cast<const __nv_bfloat162*>(input.data_ptr()),
      static_cast<float*>(scale.data_ptr()),
      static_cast<__nv_bfloat162*>(intermediate.data_ptr()), 1.0f, 0.0f);
}

STABLE_TORCH_LIBRARY(h20_silu_fp8, m) {
  m.def("run(Tensor! out, Tensor input, Tensor! scale, Tensor! intermediate, bool write_intermediate) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_silu_fp8, CUDA, m) {
  m.impl("run", TORCH_BOX(&run));
}
}  // namespace h20_silu_fp8
