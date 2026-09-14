#include <torch/csrc/stable/library.h>
/*
 * Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <type_traits>

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "cuda_compat.h"
#include "libtorch_stable/torch_utils.h"

#ifndef USE_ROCM
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
#else
    #include <hip/hip_bf16.h>
    #include <hip/hip_fp16.h>
    typedef __hip_bfloat16 __nv_bfloat16;
    typedef __hip_bfloat162 __nv_bfloat162;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace h20_topk {
namespace reference {

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

}  // namespace reference

template<int Warps, typename Index>
__launch_bounds__(Warps * 32) __global__ void register_topk(
    const __nv_bfloat16* input, float* weights, Index* indices, int* sources,
    int rows, bool renormalize) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * Warps + threadIdx.y;
  if (row >= rows) return;
  const int first = lane * 8;
  using Vec = reference::AlignedArray<__nv_bfloat16, 8>;
  const Vec data = reinterpret_cast<const Vec*>(input + row * 256)[lane];
  float values[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 pair = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(data.data + i * 2));
    values[2 * i] = pair.x;
    values[2 * i + 1] = pair.y;
  }
  // Preserve the reference's per-thread and butterfly reduction order.
  float maximum = values[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) maximum = max(maximum, values[i]);
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2)
    maximum = max(maximum, VLLM_SHFL_XOR_SYNC_WIDTH(maximum, mask, 32));
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = expf(values[i] - maximum);
    sum += values[i];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2)
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  const float reciprocal = 1.0f / sum;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] *= reciprocal;
    if (isnan(values[i]) || isinf(values[i])) values[i] = 0.0f;
  }
  float selected_sum = 0.0f;
  float saved_weight = 0.0f;
  int saved_expert = 0;
#pragma unroll
  for (int rank = 0; rank < 8; ++rank) {
    float value = values[0];
    int expert = first;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (values[i] > value) {
        value = values[i];
        expert = first + i;
      }
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
      const float other = VLLM_SHFL_XOR_SYNC_WIDTH(value, mask, 32);
      const int index = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, 32);
      if (other > value || (other == value && index < expert)) {
        value = other;
        expert = index;
      }
    }
    // Every lane sees the same winner. Keeping one winner per lane avoids
    // global weight writes followed by global normalization reads.
    if (lane == rank) {
      saved_weight = value;
      saved_expert = expert;
    }
    if (renormalize) selected_sum += value;
    if (rank != 7) {
      // Static indexing keeps the eight candidates in registers.
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (expert == first + i) values[i] = -10000.0f;
    }
  }
  float scale = 1.0f;
  if (renormalize) scale /= selected_sum > 0.0f ? selected_sum : 1.0f;
  if (lane < 8) {
    weights[row * 8 + lane] = saved_weight * scale;
    indices[row * 8 + lane] = static_cast<Index>(saved_expert);
    sources[row * 8 + lane] = lane * rows + row;
  }
}

using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;

template<int Warps, typename Index>
void launch(const Tensor& input, Tensor& weights, Tensor& indices,
            Tensor& sources, bool renormalize, bool registers,
            cudaStream_t stream) {
  const int rows = input.size(0);
  const dim3 block(32, Warps);
  const int grid = (rows + Warps - 1) / Warps;
  register_topk<Warps, Index><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<float*>(weights.data_ptr()),
      reinterpret_cast<Index*>(indices.data_ptr()),
      reinterpret_cast<int*>(sources.data_ptr()), rows, renormalize);
}

void run(const Tensor& input, Tensor& weights, Tensor& indices, Tensor& sources,
         bool renormalize, int64_t warps, bool registers) {
  STD_TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2);
  STD_TORCH_CHECK(input.size(1) == 256 && input.size(0) > 0 && input.size(0) <= 256);
  STD_TORCH_CHECK(input.scalar_type() == Type::BFloat16);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0);
  for (auto* t : {&weights, &indices, &sources}) {
    STD_TORCH_CHECK(t->device() == input.device() && t->is_contiguous());
    STD_TORCH_CHECK(t->dim() == 2 && t->size(0) == input.size(0) && t->size(1) == 8);
  }
  STD_TORCH_CHECK(weights.scalar_type() == Type::Float && sources.scalar_type() == Type::Int);
  STD_TORCH_CHECK(indices.scalar_type() == Type::Int || indices.scalar_type() == Type::Long);
  STD_TORCH_CHECK(warps == 4 && registers, "H20 Top-k supports the selected four-warp kernel");
  const auto device = input.device().index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto stream = get_current_cuda_stream(device);
  if (indices.scalar_type() == Type::Int) {
    launch<4, int32_t>(input, weights, indices, sources, renormalize, registers, stream);
  } else {
    launch<4, int64_t>(input, weights, indices, sources, renormalize, registers, stream);
  }
}

STABLE_TORCH_LIBRARY(h20_topk, m) {
  m.def("run(Tensor input, Tensor! weights, Tensor! indices, Tensor! sources, bool renormalize, int warps, bool registers) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_topk, CUDA, m) {
  m.impl("run", TORCH_BOX(&run));
}
}  // namespace h20_topk
