// SPDX-License-Identifier: Apache-2.0
// Native FP8 input quantization, stable routing and grouped-GEMM metadata.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/cuda_vec_utils.cuh"
#include "quantization/w8a8/fp8/common.cuh"

namespace h20_batch_fp8 {
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;
using Fp8 = torch::headeronly::Float8_e4m3fn;
struct Stats {
  float maximum;
  int start, count, destination;
};
struct Combine {
  __device__ __forceinline__ Stats operator()(Stats a, Stats b) const {
    return {fmaxf(a.maximum, b.maximum), a.start + b.start,
            a.count + b.count, a.destination + b.destination};
  }
};
struct Addresses {
  int64_t y, sy, q2, s2, o1, o2, w1, w2, c1, c2, g1, g2;
};

template <typename Index, int Threads>
__global__ void batch_fp8_prepare_kernel(
    const __nv_bfloat162* __restrict__ input, const Index* __restrict__ ids,
    Fp8* __restrict__ y, float* __restrict__ sy,
    int64_t* __restrict__ offsets, int32_t* __restrict__ inv,
    int32_t* __restrict__ perm, int64_t* __restrict__ ptr1,
    int64_t* __restrict__ ptr2, int32_t* __restrict__ prob1,
    int32_t* __restrict__ prob2, Addresses a, int rows) {
  constexpr int Width = 2048, Hidden = 256, Experts = 256;
  constexpr int Pairs = Width / (Threads * 2);
  const int p = blockIdx.x;
  const int tid = threadIdx.x;
  const bool has_row = p < rows;
  const int expert = has_row ? static_cast<int>(ids[p]) : 0;
  Stats local{0.f, 0, 0, 0};
  for (int i = tid; i < rows; i += Threads) {
    const int id = static_cast<int>(ids[i]);
    local.start += id < p;
    local.count += id == p;
    local.destination += id < expert || (id == expert && i < p);
  }
  float values[Pairs * 2];
  if (has_row) {
#pragma unroll
    for (int i = 0; i < Pairs; ++i) {
      const float2 pair = vllm::cast_to_float2(
          input[(p / 8) * (Width / 2) + tid + i * Threads]);
      values[2 * i] = pair.x;
      values[2 * i + 1] = pair.y;
      local.maximum = fmaxf(local.maximum, fabsf(pair.x));
      local.maximum = fmaxf(local.maximum, fabsf(pair.y));
    }
  }
  using Reduce = cub::BlockReduce<Stats, Threads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float token_scale;
  __shared__ int destination;
  const Stats aggregate = Reduce(storage).Reduce(local, Combine{});
  if (tid == 0) {
    offsets[p] = aggregate.start;
    if (p < Experts) {
      const int64_t start = aggregate.start;
      const int64_t first[6] = {
          a.y + start * Width, a.w1 + int64_t(p) * Width * Hidden,
          a.o1 + start * Hidden * 4, a.sy + start * 4,
          a.c1 + int64_t(p) * Hidden * 8,
          a.g1 + int64_t(p) * (Width / 128) * Hidden * 16};
      const int64_t second[6] = {
          a.q2 + start * Hidden, a.w2 + int64_t(p) * Hidden * Width / 2,
          a.o2 + start * Width * 2, a.s2 + start * 4,
          a.c2 + int64_t(p) * Width * 4,
          a.g2 + int64_t(p) * (Hidden / 128) * Width * 8};
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        ptr1[j * Experts + p] = first[j];
        ptr2[j * Experts + p] = second[j];
      }
      prob1[3 * p] = Hidden * 2;
      prob1[3 * p + 1] = aggregate.count;
      prob1[3 * p + 2] = Width;
      prob2[3 * p] = Width;
      prob2[3 * p + 1] = aggregate.count;
      prob2[3 * p + 2] = Hidden;
    }
    if (has_row) {
      destination = aggregate.destination;
      inv[p] = destination;
      perm[destination] = p;
      token_scale = fmaxf(aggregate.maximum / quant_type_max_v<Fp8>,
                          min_scaling_factor<Fp8>::val());
      sy[destination] = token_scale;
    }
  }
  __syncthreads();
  if (has_row) {
#pragma unroll
    for (int i = 0; i < Pairs; ++i) {
      const int column = 2 * (tid + i * Threads);
      y[destination * Width + column] =
          vllm::scaled_fp8_conversion<false, Fp8>(values[2 * i], token_scale);
      y[destination * Width + column + 1] =
          vllm::scaled_fp8_conversion<false, Fp8>(values[2 * i + 1], token_scale);
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
    Tensor& prob1, Tensor& prob2, int64_t threads) {
  STD_TORCH_CHECK(input.is_cuda() && input.dim() == 2 &&
                  input.size(0) >= 2 && input.size(0) <= 32 && input.size(1) == 2048);
  const int tokens = input.size(0), rows = tokens * 8;
  check(input, input, Type::BFloat16, int64_t(tokens) * 2048);
  STD_TORCH_CHECK(address(input) % 4 == 0);
  STD_TORCH_CHECK(ids.scalar_type() == Type::Int || ids.scalar_type() == Type::Long);
  check(ids, input, ids.scalar_type(), rows);
  check(y, input, Type::Float8_e4m3fn, rows * 2048);
  check(sy, input, Type::Float, rows);
  check(offsets, input, Type::Long, 257);
  check(inv, input, Type::Int, rows);
  check(perm, input, Type::Int, rows);
  check(q2, input, Type::Float8_e4m3fn, rows * 256);
  check(s2, input, Type::Float, rows);
  check(o1, input, Type::BFloat16, rows * 512);
  check(o2, input, Type::BFloat16, rows * 2048);
  check(w1, input, Type::Int, 256 * 512 * 2048 / 8);
  check(w2, input, Type::Int, 256 * 2048 * 256 / 8);
  check(c1, input, Type::Float, 256 * 512);
  check(c2, input, Type::Float, 256 * 2048);
  check(g1, input, Type::Float8_e4m3fn, 256 * 16 * 512 * 8);
  check(g2, input, Type::Float8_e4m3fn, 256 * 2 * 2048 * 8);
  check(ptr1, input, Type::Long, 6 * 256);
  check(ptr2, input, Type::Long, 6 * 256);
  check(prob1, input, Type::Int, 256 * 3);
  check(prob2, input, Type::Int, 256 * 3);
  STD_TORCH_CHECK(address(ptr1) % 16 == 0 && address(ptr2) % 16 == 0);
  STD_TORCH_CHECK(address(input) + int64_t(tokens) * 4096 <= address(y) ||
                  address(y) + int64_t(rows) * 2048 <= address(input),
                  "Input and FP8 output overlap");
  STD_TORCH_CHECK(threads == 128 || threads == 256);
  const Addresses a{address(y), address(sy), address(q2), address(s2),
      address(o1), address(o2), address(w1), address(w2), address(c1),
      address(c2), address(g1), address(g2)};
  const auto device_id = input.device().index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_id);
  auto stream = get_current_cuda_stream(device_id);
#define LAUNCH(Index, Threads) \
  batch_fp8_prepare_kernel<Index, Threads><<<257, Threads, 0, stream>>>( \
      static_cast<const __nv_bfloat162*>(input.data_ptr()), \
      static_cast<const Index*>(ids.data_ptr()), static_cast<Fp8*>(y.data_ptr()), \
      static_cast<float*>(sy.data_ptr()), static_cast<int64_t*>(offsets.data_ptr()), \
      static_cast<int32_t*>(inv.data_ptr()), static_cast<int32_t*>(perm.data_ptr()), \
      static_cast<int64_t*>(ptr1.data_ptr()), static_cast<int64_t*>(ptr2.data_ptr()), \
      static_cast<int32_t*>(prob1.data_ptr()), static_cast<int32_t*>(prob2.data_ptr()), a, rows)
  if (ids.scalar_type() == Type::Int) {
    if (threads == 128) { LAUNCH(int32_t, 128); }
    else { LAUNCH(int32_t, 256); }
  } else {
    if (threads == 128) { LAUNCH(int64_t, 128); }
    else { LAUNCH(int64_t, 256); }
  }
#undef LAUNCH
}
STABLE_TORCH_LIBRARY(h20_batch_fp8, m) {
  m.def("run(Tensor input, Tensor ids, Tensor! y, Tensor! sy, Tensor! offsets, "
        "Tensor! inv, Tensor! perm, Tensor q2, Tensor s2, Tensor o1, Tensor o2, "
        "Tensor w1, Tensor w2, Tensor c1, Tensor c2, Tensor g1, Tensor g2, "
        "Tensor! ptr1, Tensor! ptr2, Tensor! prob1, Tensor! prob2, int threads) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_batch_fp8, CUDA, m) {
  m.impl("run", TORCH_BOX(&run));
}
}  // namespace h20_batch_fp8
