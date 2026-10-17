// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include "libtorch_stable/torch_utils.h"

#define MARLIN_NAMESPACE_NAME sm80_dense_fp8
#include "kernels.cuh"

namespace {
using torch::headeronly::ScalarType;
using torch::stable::Tensor;
namespace k = sm80_dense_fp8;

enum class Family {
  Generic,
  DecodeQKV,
  DecodeOut,
  SmallN,
  SmallNCompact,
  SharedA,
  Wide,
  FullAWide,
  FullASplit,
  LastSplit,
  Middle,
  MiddleFirst,
  AsyncFC,
  AsyncGate
};
struct Plan {
  Family family;
  int tile_m, tile_n, split, threads, shared;
};

int64_t padded_n(int64_t n) { return n <= 32 ? 64 : ((n + 127) / 128) * 128; }

Plan select_plan(int64_t m, int64_t n, int64_t width) {
  if (n == 6144 && width == 2048) {
    if (m <= 8) return {Family::DecodeQKV, 8, 256, 16, 128, 32768};
    if (m == 2048 || m == 8192)
      return {Family::FullAWide, 64, 512, 1, 256, 98304};
  }
  if (n == 2048 && width == 2048 && m <= 8)
    return {Family::DecodeOut, 8, 128, 16, 128, 32768};
  if (n == 2048 && (width == 2048 || width == 256)) {
    if (m == 2048) return {Family::SharedA, 64, 128, 1, 128, 49152};
    if (m == 8192) return {Family::Wide, 64, 512, 1, 256, 98304};
  }
  if (n == 512 && width == 2048) {
    if (m == 2048) return {Family::FullASplit, 64, 128, 2, 128, 49152};
    if (m == 8192) return {Family::AsyncGate, 64, 128, 1, 128, 49152};
  }
  if (n == 1024 && width == 4096) {
    if (m == 1024) return {Family::LastSplit, 80, 128, 2, 128, 53248};
    if (m == 2048) return {Family::MiddleFirst, 80, 128, 1, 128, 53248};
    if (m == 8192) return {Family::AsyncFC, 64, 256, 1, 128, 65536};
  }
  if (n <= 32) {
    if (width == 2048 && (m == 2048 || m == 4096))
      return {Family::SmallNCompact, 32, 64, 1, 128, 49152};
    const int split = width == 2048 ? (m == 512 ? 8 : (m == 1024 ? 4 : 1)) : 1;
    return {Family::SmallN, 16, 64, split, 128, 32768};
  }
  // The same FP8/G32 contract, including arbitrary M tails. No BF16 GEMM.
  return {Family::Generic, 16, 128, 1, 128, 32768};
}

auto kernel_for(Family f) {
  auto fn =
      k::Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
                vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 1, 8, 2,
                false, 2, 2, false, false, false, true>;
  switch (f) {
    case Family::DecodeQKV:
      return k::Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                       vllm::kBFloat16.id(),
                       vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 1, 16,
                       2, true, 2, 2, false, false>;
    case Family::DecodeOut:
      return k::Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                       vllm::kBFloat16.id(),
                       vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 1, 8,
                       4, true, 2, 2, false, false>;
    case Family::SmallN:
      return k::Marlin<vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                       vllm::kBFloat16.id(),
                       vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 1, 4,
                       8, false, 2, 2, false, false>;
    case Family::SmallNCompact:
      return k::MarlinBAPackedShared<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 2, 4, 8, false, 2,
          2, false, false, false, false, false, false, true>;
    case Family::SharedA:
      return k::Marlin<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 4, 8, 2, false, 4,
          2, false, false, false, true, false, true, true>;
    case Family::Wide:
      return k::Marlin<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 256, 4, 32, 2, false, 4,
          2, false, false, false, false, false, true, true>;
    case Family::FullAWide:
      return k::MarlinFullAN512<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 256, 4, 32, 2, false, 4,
          2, false, false, false, false, false, true, true, false, false, false,
          false, false, true>;
    case Family::FullASplit:
      return k::MarlinFullASplit2<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 4, 8, 2, false, 4,
          2, false, false, false, true, false, true, true, false, false, false,
          false, false, false, true>;
    case Family::LastSplit:
      return k::MarlinLastSplit2<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 5, 8, 2, false, 4,
          2, false, false, false, true, false, true, true, false, false, false,
          false, false, false, true>;
    case Family::Middle:
      return k::MarlinM80Native<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 5, 8, 2, false, 4,
          2, false, false, false, true, false, true, true>;
    case Family::MiddleFirst:
      return k::MarlinM80First32<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 5, 8, 2, false, 4,
          2, false, false, false, true, false, true, true>;
    case Family::AsyncFC:
      return k::MarlinAsyncFC<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 4, 16, 2, false, 4,
          2, false, false, false, false, false, true, true, false, false, false,
          false, false, false, false, false, 1>;
    case Family::AsyncGate:
      return k::MarlinAsyncGate<
          vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), vllm::kBFloat16.id(),
          vllm::ScalarType::float_IEEE754(8, 23).id(), 128, 4, 8, 2, false, 4,
          2, false, false, false, true, false, true, true, false, false, false,
          false, false, false, false, false, 1>;
    default:
      return fn;
  }
}

// Retained ordered FP32 split-K reducers.
#include "reduce.cuh"

Tensor dense_gemm(Tensor a, Tensor sa, Tensor w, Tensor sw, Tensor auxiliary,
                  int64_t n) {
  for (const auto& t : {a, sa, w, sw, auxiliary}) {
    STD_TORCH_CHECK(
        t.device().is_cuda() && t.is_contiguous() && t.device() == a.device(),
        "SM80 FP8 requires contiguous colocated CUDA tensors");
  }
  STD_TORCH_CHECK(a.dim() == 2 &&
                      a.scalar_type() == ScalarType::Float8_e4m3fn &&
                      sa.scalar_type() == ScalarType::Float &&
                      sw.scalar_type() == ScalarType::Float &&
                      w.scalar_type() == ScalarType::Int &&
                      auxiliary.scalar_type() == ScalarType::Int,
                  "SM80 FP8 requires E4M3 A, FP32 scales and INT32 packed W");
  const int64_t m = a.size(0), width = a.size(1), pn = padded_n(n);
  STD_TORCH_CHECK(n > 0 && width >= 128 && width <= 65536 && width % 128 == 0 &&
                      m <= 524280 && m * pn <= INT32_MAX &&
                      m * width <= INT32_MAX && pn * width <= INT32_MAX,
                  "SM80 FP8 shape exceeds startup-supported limits");
  STD_TORCH_CHECK(
      sa.dim() == 2 && sa.size(0) == m && sa.size(1) == width / 32 &&
          w.dim() == 2 && w.size(0) == width / 16 && w.size(1) == pn * 4 &&
          sw.dim() == 2 && sw.size(0) == width / 32 && sw.size(1) == pn,
      "SM80 FP8 G32 scale or packed weight shape mismatch");
  const auto device = a.device();
  const int index = a.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(index);
  cudaDeviceProp props;
  STD_TORCH_CHECK(cudaGetDeviceProperties(&props, index) == cudaSuccess &&
                      props.major == 8 && props.minor == 0,
                  "SM80 FP8 requires an A100/SM80 device");
  auto out =
      torch::stable::empty({m, pn}, ScalarType::BFloat16, std::nullopt, device);
  if (m == 0) return out;
  const auto plan = select_plan(m, n, width);
  const bool fragment = plan.family == Family::FullASplit;
  const bool last = plan.family == Family::LastSplit;
  const bool extra = plan.family == Family::SmallNCompact ||
                     plan.family == Family::AsyncFC ||
                     plan.family == Family::AsyncGate;
  const int64_t tiles = (m + plan.tile_m - 1) / plan.tile_m;
  const int64_t nt = pn / plan.tile_n;
  if (extra) {
    const int64_t expected =
        plan.family == Family::SmallNCompact
            ? width / 16 * pn * 2
            : width / 32 * (pn / plan.tile_n) * (plan.tile_n * 9);
    STD_TORCH_CHECK(auxiliary.numel() == expected,
                    "Missing prepared FP8 weight layout");
  }
  auto stream = get_current_cuda_stream(index);
  auto fn = kernel_for(plan.family);
  if (plan.shared > props.sharedMemPerBlock)
    STD_TORCH_CHECK(
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             plan.shared) == cudaSuccess,
        "Cannot configure SM80 FP8 shared memory");
  Tensor partials, locks;
  void* target = out.mutable_data_ptr();
  void* temporary = nullptr;
  int* lock_ptr = nullptr;
  if (plan.split > 1) {
    int64_t count = (fragment || last) ? plan.split * tiles * pn * plan.tile_m
                                       : plan.split * m * pn;
    STD_TORCH_CHECK(count <= INT32_MAX, "SM80 FP8 split workspace overflow");
    partials =
        torch::stable::empty({count}, ScalarType::Float, std::nullopt, device);
    if (!last) target = partials.mutable_data_ptr();
    if (fragment || last) temporary = partials.mutable_data_ptr();
    if (last) {
      locks = torch::stable::empty({tiles * nt}, ScalarType::Int, std::nullopt,
                                   device);
      lock_ptr = static_cast<int*>(locks.mutable_data_ptr());
      STD_TORCH_CHECK(cudaMemsetAsync(lock_ptr, 0, tiles * nt * sizeof(int),
                                      stream) == cudaSuccess,
                      "Cannot initialize exclusive FP8 split workspace");
    }
  }
  fn<<<dim3(nt, tiles, plan.split), plan.threads, plan.shared, stream>>>(
      static_cast<const int4*>(a.const_data_ptr()),
      static_cast<const int4*>(extra ? auxiliary.const_data_ptr()
                                     : w.const_data_ptr()),
      static_cast<int4*>(target), static_cast<int4*>(temporary), nullptr,
      static_cast<const float*>(sa.const_data_ptr()),
      static_cast<const int4*>(sw.const_data_ptr()), nullptr, nullptr, nullptr,
      width / 32, m, pn, width, width, lock_ptr, false, false, true,
      plan.shared, plan.split > 1 && !last, nullptr, nullptr);
  if (fragment) {
    reduce_fragment_partials<4, 2><<<dim3(nt, tiles), 128, 0, stream>>>(
        static_cast<const float4*>(partials.const_data_ptr()),
        static_cast<nv_bfloat16*>(out.mutable_data_ptr()), m, pn, plan.split);
  } else if (plan.split > 1 && !last) {
    if (plan.family == Family::DecodeQKV && m == 8)
      reduce_partials_fixed16<<<(m * pn + 255) / 256, 256, 0, stream>>>(
          static_cast<const float*>(partials.const_data_ptr()),
          out.mutable_data_ptr(), m * pn, plan.split, false);
    else
      reduce_partials<<<(m * pn + 255) / 256, 256, 0, stream>>>(
          static_cast<const float*>(partials.const_data_ptr()),
          out.mutable_data_ptr(), m * pn, plan.split, false);
  }
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "SM80 FP8 launch failed");
  return out;
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm80_fp8_g32_gemm(Tensor a, Tensor sa, Tensor w, Tensor sw, Tensor "
      "auxiliary, int n) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm80_fp8_g32_gemm", TORCH_BOX(&dense_gemm));
}
