// SPDX-License-Identifier: Apache-2.0
// Isolated exact FP32-scale G32 activation quantizer. No GEMM changes.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <climits>
#include <cmath>

namespace h20_quant_g32 {
using Tensor = torch::stable::Tensor;
using Type = torch::headeronly::ScalarType;

template<bool Aligned>
__global__ void quant_register(const __nv_bfloat16* __restrict__ x,
                               uint8_t* __restrict__ q,
                               float* __restrict__ scales,
                               int64_t groups, int rows, int groups_per_row,
                               float eps) {
  constexpr int ThreadsPerGroup = 4;
  constexpr int GroupsPerCTA = 64;
  const int lane = threadIdx.x % ThreadsPerGroup;
  const int64_t g = int64_t(blockIdx.x) * GroupsPerCTA + threadIdx.x / ThreadsPerGroup;
  // A complete four-lane subgroup exits together; no CTA synchronization.
  if (g >= groups) return;
  const int64_t offset = g * 32 + lane * 8;
  float values[8];
  if constexpr (Aligned) {
    union Pack { uint4 vector; __nv_bfloat16 bf16[8]; } pack;
    pack.vector = *reinterpret_cast<const uint4*>(x + offset);
    #pragma unroll
    for (int i=0; i<8; ++i) values[i] = __bfloat162float(pack.bf16[i]);
  } else {
    #pragma unroll
    for (int i=0; i<8; ++i) values[i] = __bfloat162float(x[offset+i]);
  }
  float amax = eps;
  #pragma unroll
  for (int i=0; i<8; ++i) amax = fmaxf(amax, fabsf(values[i]));
  const unsigned mask = 0xfu << ((threadIdx.x % 32) & ~3);
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2, 4));
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1, 4));
  const float scale = __fdiv_rn(amax, 448.0f);
  if (lane == 0) {
    const int64_t row = g / groups_per_row;
    const int64_t group_in_row = g % groups_per_row;
    scales[group_in_row * rows + row] = scale;
  }
  uint64_t packed = 0;
  #pragma unroll
  for (int i=0; i<8; ++i) {
    const float v = fminf(fmaxf(__fdiv_rn(values[i], scale), -448.0f), 448.0f);
    const uint8_t byte = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    if constexpr (Aligned) packed |= uint64_t(byte) << (8*i);
    else q[offset+i] = byte;
  }
  if constexpr (Aligned) *reinterpret_cast<uint64_t*>(q+offset) = packed;
}

void quant(Tensor const& x, Tensor q, Tensor scales, double eps) {
  STD_TORCH_CHECK(x.is_cuda() && q.is_cuda() && scales.is_cuda());
  STD_TORCH_CHECK(x.device()==q.device() && x.device()==scales.device());
  STD_TORCH_CHECK(x.dim()==2 && q.dim()==2 && scales.dim()==2);
  STD_TORCH_CHECK(x.scalar_type()==Type::BFloat16);
  STD_TORCH_CHECK(q.scalar_type()==Type::Float8_e4m3fn && scales.scalar_type()==Type::Float);
  STD_TORCH_CHECK(x.is_contiguous() && q.is_contiguous());
  const auto m=x.size(0), k=x.size(1);
  STD_TORCH_CHECK(m>0 && m<=INT_MAX && k>0 && k<=INT_MAX && k%32==0);
  STD_TORCH_CHECK(q.size(0)==m && q.size(1)==k);
  STD_TORCH_CHECK(scales.size(0)==m && scales.size(1)==k/32);
  STD_TORCH_CHECK(scales.stride(0)==1 && scales.stride(1)==m);
  STD_TORCH_CHECK(std::isfinite(eps) && eps>0 && float(eps)>0);
  const int64_t groups=m*(k/32), blocks=(groups+63)/64;
  STD_TORCH_CHECK(blocks<=INT_MAX);
  const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());
  auto stream=get_current_cuda_stream();
  auto xp=static_cast<const __nv_bfloat16*>(x.data_ptr());
  auto qp=static_cast<uint8_t*>(q.data_ptr());
  auto sp=static_cast<float*>(scales.data_ptr());
  const bool aligned=(reinterpret_cast<uintptr_t>(xp)%16==0 && reinterpret_cast<uintptr_t>(qp)%8==0);
  if (aligned) quant_register<true><<<blocks,256,0,stream>>>(xp,qp,sp,groups,m,k/32,float(eps));
  else quant_register<false><<<blocks,256,0,stream>>>(xp,qp,sp,groups,m,k/32,float(eps));
  STD_TORCH_CHECK(cudaGetLastError()==cudaSuccess, "G32 quantization launch failed");
}
}

STABLE_TORCH_LIBRARY(h20_quant_g32,m) {
  m.def("quant(Tensor x, Tensor(a!) q, Tensor(b!) scales, float eps) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(h20_quant_g32,CUDA,m) {
  m.impl("quant",TORCH_BOX(&h20_quant_g32::quant));
}
