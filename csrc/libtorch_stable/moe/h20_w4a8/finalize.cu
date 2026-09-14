// SPDX-License-Identifier: Apache-2.0
// Arithmetic copied from the installed-reference source. Only column scheduling changes.
#include <algorithm>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/moe/permute_unpermute_kernels/dispatch.h"
#include "cutlass/array.h"
#include "cutlass/numeric_size.h"

namespace h20_unrolled_finalize {
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  using Type = typename U::Element;
  static_assert(T::kElements == U::kElements);
  U u;
#pragma unroll
  for (int i = 0; i < U::kElements; i++) {
    u[i] = static_cast<Type>(input[i]);
  }
  return u;
}

template <typename T, typename OutputType, bool CHECK_SKIPPED>
__global__ void unrolledFinalizeMoeRoutingKernel(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t const orig_cols, int64_t const k, int64_t const* num_valid_ptr) {
  assert(orig_cols % 4 == 0);
  int64_t const original_row = blockIdx.x;
  auto const offset = original_row * orig_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
  int64_t const num_valid = *num_valid_ptr;

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(cutlass::sizeof_bits<OutputType>::value,
                     cutlass::sizeof_bits<T>::value);

  int64_t const start_offset = blockIdx.y * blockDim.x + threadIdx.x;
  int64_t const stride = blockDim.x * gridDim.y;
  int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

  using InputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* expanded_permuted_rows_v =
      reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_col;
       elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    #pragma unroll
    for (int k_idx = 0; k_idx < 8; ++k_idx) {
      int64_t const expanded_original_row = original_row * k + k_idx;
      int64_t const expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      int64_t const k_offset = original_row * k + k_idx;
      float const row_scale = scales[k_offset];

      if (CHECK_SKIPPED && expanded_permuted_row >= num_valid) {
        continue;
      }

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(
          expanded_permuted_rows_row_ptr[elem_index]);
      thread_output = thread_output + row_scale * (expert_result);
    }

    OutputElem output_elem =
        arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
}

bool overlaps(const Tensor& a, const Tensor& b) {
  const uintptr_t ab = reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t bb = reinterpret_cast<uintptr_t>(b.data_ptr());
  return ab < bb + b.numel() * b.element_size() &&
         bb < ab + a.numel() * a.element_size();
}

void run(Tensor& out, const Tensor& input, const Tensor& weights,
         const Tensor& inverse, const Tensor& offsets, int64_t shards) {
  STD_TORCH_CHECK(out.is_cuda() && out.scalar_type() == Type::BFloat16);
  STD_TORCH_CHECK(out.is_contiguous() && out.dim() == 2 && out.size(1) == 2048);
  STD_TORCH_CHECK(out.size(0) > 0 && out.size(0) <= 256);
  const int64_t rows = out.size(0);
  STD_TORCH_CHECK(input.device() == out.device() && input.scalar_type() == Type::BFloat16);
  STD_TORCH_CHECK(input.is_contiguous() && input.dim() == 2 &&
                  input.size(0) == rows * 8 && input.size(1) == 2048);
  STD_TORCH_CHECK(weights.device() == out.device() && weights.scalar_type() == Type::Float);
  STD_TORCH_CHECK(weights.is_contiguous() && weights.dim() == 2 &&
                  weights.size(0) == rows && weights.size(1) == 8);
  STD_TORCH_CHECK(inverse.device() == out.device() && inverse.scalar_type() == Type::Int);
  STD_TORCH_CHECK(inverse.is_contiguous() && inverse.numel() == rows * 8);
  STD_TORCH_CHECK(offsets.device() == out.device() && offsets.scalar_type() == Type::Long);
  STD_TORCH_CHECK(offsets.is_contiguous() && offsets.dim() == 1 && offsets.numel() == 257);
  STD_TORCH_CHECK(shards == 1 || shards == 2 || shards == 4 || shards == 8);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
  STD_TORCH_CHECK(!overlaps(out, input) && !overlaps(out, weights) &&
                  !overlaps(out, inverse) && !overlaps(out, offsets));
  const auto device_id = out.device().index();
  const torch::stable::accelerator::DeviceGuard guard(device_id);
  auto stream = get_current_cuda_stream(device_id);
  unrolledFinalizeMoeRoutingKernel<__nv_bfloat16, __nv_bfloat16, true>
      <<<dim3(rows, shards), 256 / shards, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(input.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()),
          static_cast<const float*>(weights.data_ptr()),
          static_cast<const int*>(inverse.data_ptr()), 2048, 8,
          static_cast<const int64_t*>(offsets.data_ptr()) + 256);
}

STABLE_TORCH_LIBRARY(h20_unrolled_finalize, m) {
  m.def("run(Tensor! out, Tensor input, Tensor weights, Tensor inverse, Tensor offsets, int shards) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_unrolled_finalize, CUDA, m) {
  m.impl("run", TORCH_BOX(&run));
}
} // namespace h20_unrolled_finalize
