// SPDX-License-Identifier: Apache-2.0
// Single-request native FP8 quantization, stable expert routing and GEMM metadata.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/cuda_vec_utils.cuh"
#include "quantization/w8a8/fp8/common.cuh"

namespace h20_compact_fp8 {
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;
using Fp8 = torch::headeronly::Float8_e4m3fn;
struct Maximum {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

struct Addresses {
  int64_t y, sy, q2, s2, o1, o2, w1, w2, c1, c2, g1, g2;
};

template <typename Index, bool Replicate>
__global__ void compact_fp8_prepare_kernel(
    const __nv_bfloat162* __restrict__ input, const Index* __restrict__ ids,
    Fp8* __restrict__ y, float* __restrict__ sy,
    int64_t* __restrict__ offsets, int32_t* __restrict__ inv,
    int32_t* __restrict__ perm, int64_t* __restrict__ ptr1,
    int64_t* __restrict__ ptr2, int32_t* __restrict__ prob1,
    int32_t* __restrict__ prob2, Addresses a) {
  constexpr int Width = 2048, Hidden = 256, TopK = 8;
  const int tid = threadIdx.x;
  if (blockIdx.x == 0) {
    for (int expert = tid; expert <= 256; expert += blockDim.x) {
      int start = 0;
#pragma unroll
      for (int i = 0; i < TopK; ++i) start += ids[i] < expert;
      offsets[expert] = start;
    }
    if (tid < TopK) {
      const int expert_id = static_cast<int>(ids[tid]);
      int destination = 0;
#pragma unroll
      for (int i = 0; i < TopK; ++i) {
        const int id = static_cast<int>(ids[i]);
        destination += id < expert_id || (id == expert_id && i < tid);
      }
      inv[tid] = destination;
      perm[destination] = tid;
      const int expert = min(expert_id, 255);
      const int count = expert_id < 256;
      const int64_t first[6] = {
          a.y + destination * Width,
          a.w1 + int64_t(expert) * Width * Hidden,
          a.o1 + destination * Hidden * 4,
          a.sy + destination * 4,
          a.c1 + int64_t(expert) * Hidden * 8,
          a.g1 + int64_t(expert) * (Width / 128) * Hidden * 16};
      const int64_t second[6] = {
          a.q2 + destination * Hidden,
          a.w2 + int64_t(expert) * Hidden * Width / 2,
          a.o2 + destination * Width * 2,
          a.s2 + destination * 4,
          a.c2 + int64_t(expert) * Width * 4,
          a.g2 + int64_t(expert) * (Hidden / 128) * Width * 8};
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        ptr1[j * TopK + destination] = first[j];
        ptr2[j * TopK + destination] = second[j];
      }
      prob1[3 * destination] = 2 * Hidden;
      prob1[3 * destination + 1] = count;
      prob1[3 * destination + 2] = Width;
      prob2[3 * destination] = Width;
      prob2[3 * destination + 1] = count;
      prob2[3 * destination + 2] = Hidden;
    }
  }
  float values[16];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 pair = vllm::cast_to_float2(input[tid + i * 128]);
    values[2 * i] = pair.x;
    values[2 * i + 1] = pair.y;
    maximum = fmaxf(maximum, fabsf(pair.x));
    maximum = fmaxf(maximum, fabsf(pair.y));
  }
  using Reduce = cub::BlockReduce<float, 128>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float token_scale;
  maximum = Reduce(storage).Reduce(maximum, Maximum{});
  if (tid == 0) {
    token_scale = fmaxf(maximum / quant_type_max_v<Fp8>,
                        min_scaling_factor<Fp8>::val());
  }
  __syncthreads();
  Fp8 quantized[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    quantized[i] = vllm::scaled_fp8_conversion<false, Fp8>(values[i], token_scale);
  }
  const int begin = Replicate ? blockIdx.x : 0;
  const int end = Replicate ? begin + 1 : TopK;
  // All eight expanded rows of one input token have identical FP8 bytes.
  for (int row = begin; row < end; ++row) {
    if (tid == 0) sy[row] = token_scale;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      y[row * Width + 2 * (tid + i * 128)] = quantized[2 * i];
      y[row * Width + 2 * (tid + i * 128) + 1] = quantized[2 * i + 1];
    }
  }
}

int64_t address(const Tensor& tensor) {
  return reinterpret_cast<int64_t>(tensor.data_ptr());
}

void check(const Tensor& tensor, const Tensor& input, Type type, int64_t elements) {
  STD_TORCH_CHECK(tensor.device() == input.device() && tensor.scalar_type() == type);
  STD_TORCH_CHECK(tensor.is_contiguous() && tensor.numel() == elements);
}

void run(const Tensor& input, const Tensor& ids, Tensor& y, Tensor& sy,
    Tensor& offsets, Tensor& inv, Tensor& perm,
    const Tensor& q2, const Tensor& s2, const Tensor& o1, const Tensor& o2,
    const Tensor& w1, const Tensor& w2, const Tensor& c1, const Tensor& c2,
    const Tensor& g1, const Tensor& g2, Tensor& ptr1, Tensor& ptr2,
    Tensor& prob1, Tensor& prob2, int64_t replicas) {
  STD_TORCH_CHECK(input.is_cuda() && input.dim() == 2 &&
                  input.size(0) == 1 && input.size(1) == 2048);
  check(input, input, Type::BFloat16, 2048);
  STD_TORCH_CHECK(address(input) % 4 == 0);
  STD_TORCH_CHECK(ids.scalar_type() == Type::Int || ids.scalar_type() == Type::Long);
  check(ids, input, ids.scalar_type(), 8);
  check(y, input, Type::Float8_e4m3fn, 8 * 2048);
  check(sy, input, Type::Float, 8);
  check(offsets, input, Type::Long, 257);
  check(inv, input, Type::Int, 8);
  check(perm, input, Type::Int, 8);
  check(q2, input, Type::Float8_e4m3fn, 8 * 256);
  check(s2, input, Type::Float, 8);
  check(o1, input, Type::BFloat16, 8 * 512);
  check(o2, input, Type::BFloat16, 8 * 2048);
  check(w1, input, Type::Int, 256 * 512 * 2048 / 8);
  check(w2, input, Type::Int, 256 * 2048 * 256 / 8);
  check(c1, input, Type::Float, 256 * 512);
  check(c2, input, Type::Float, 256 * 2048);
  check(g1, input, Type::Float8_e4m3fn, 256 * (2048 / 128) * 512 * 8);
  check(g2, input, Type::Float8_e4m3fn, 256 * (256 / 128) * 2048 * 8);
  check(ptr1, input, Type::Long, 6 * 8);
  check(ptr2, input, Type::Long, 6 * 8);
  check(prob1, input, Type::Int, 8 * 3);
  check(prob2, input, Type::Int, 8 * 3);
  STD_TORCH_CHECK(address(ptr1) % 16 == 0 && address(ptr2) % 16 == 0);
  STD_TORCH_CHECK(address(input) + 4096 <= address(y) ||
                  address(y) + 8 * 2048 <= address(input), "Input and FP8 output overlap");
  STD_TORCH_CHECK(replicas == 1 || replicas == 8);
  const Addresses a{address(y), address(sy), address(q2), address(s2),
      address(o1), address(o2), address(w1), address(w2), address(c1),
      address(c2), address(g1), address(g2)};
  const auto device_id = input.device().index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_id);
  auto stream = get_current_cuda_stream(device_id);
#define LAUNCH(Index, Replicate) \
  compact_fp8_prepare_kernel<Index, Replicate><<<replicas, 128, 0, stream>>>( \
      static_cast<const __nv_bfloat162*>(input.data_ptr()), \
      static_cast<const Index*>(ids.data_ptr()), static_cast<Fp8*>(y.data_ptr()), \
      static_cast<float*>(sy.data_ptr()), static_cast<int64_t*>(offsets.data_ptr()), \
      static_cast<int32_t*>(inv.data_ptr()), static_cast<int32_t*>(perm.data_ptr()), \
      static_cast<int64_t*>(ptr1.data_ptr()), static_cast<int64_t*>(ptr2.data_ptr()), \
      static_cast<int32_t*>(prob1.data_ptr()), static_cast<int32_t*>(prob2.data_ptr()), a)
  if (ids.scalar_type() == Type::Int) {
    if (replicas == 1) { LAUNCH(int32_t, false); }
    else { LAUNCH(int32_t, true); }
  } else {
    if (replicas == 1) { LAUNCH(int64_t, false); }
    else { LAUNCH(int64_t, true); }
  }
#undef LAUNCH
}

STABLE_TORCH_LIBRARY(h20_compact_fp8, m) {
  m.def("run(Tensor input, Tensor ids, Tensor! y, Tensor! sy, Tensor! offsets, "
        "Tensor! inv, Tensor! perm, Tensor q2, Tensor s2, Tensor o1, Tensor o2, "
        "Tensor w1, Tensor w2, Tensor c1, Tensor c2, Tensor g1, Tensor g2, "
        "Tensor! ptr1, Tensor! ptr2, Tensor! prob1, Tensor! prob2, int replicas) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_compact_fp8, CUDA, m) {
  m.impl("run", TORCH_BOX(&run));
}
}  // namespace h20_compact_fp8
