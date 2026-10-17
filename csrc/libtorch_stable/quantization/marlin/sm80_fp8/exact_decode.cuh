// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

__device__ inline float scaled_decode(uint8_t code, float scale) {
  // Finite codes are a precondition of this fast core.
  // Restore the exponent in FP32 before multiplying the unchanged FP32 scale.
  unsigned magnitude = code & 127;
  unsigned bits = (magnitude << 20) | (static_cast<unsigned>(code & 128) << 24);
  float value = __fmul_rn(__uint_as_float(bits), 0x1p120f);
  return __fmul_rn(value, scale);
}

__device__ inline nv_bfloat16 exact_decode(uint8_t code, float scale) {
  return __float2bfloat16_rn(scaled_decode(code, scale));
}

__device__ inline nv_bfloat162 exact_decode_pair(uint8_t a, uint8_t b,
                                                 float scale) {
  return __floats2bfloat162_rn(scaled_decode(a, scale),
                               scaled_decode(b, scale));
}

template <bool use_fp16>
__device__ inline nv_bfloat162 exact_decode_word_pair(unsigned raw,
                                                      float scale) {
  float2 values;
  if constexpr (use_fp16) {
    __half2 raw_pair = *reinterpret_cast<__half2*>(&raw);
    values = __half22float2(__hmul2(raw_pair, __float2half2_rn(0x1p8f)));
  } else {
    nv_bfloat162 raw_pair = *reinterpret_cast<nv_bfloat162*>(&raw);
    values =
        __bfloat1622float2(__hmul2(raw_pair, __float2bfloat162_rn(0x1p120f)));
  }
  return __floats2bfloat162_rn(__fmul_rn(values.x, scale),
                               __fmul_rn(values.y, scale));
}

template <bool use_fp16>
__device__ inline void exact_decode_fragment(int packed, float scale,
                                             nv_bfloat16* output) {
  // Both native paired mappings restore the finite code exactly.
  // Saved group32 scales stay FP32 and BF16 rounding occurs after
  // multiplication.
  unsigned q = static_cast<unsigned>(packed);
  unsigned odd = (q & 0x80008000u) | ((q & 0x7f007f00u) >> (use_fp16 ? 1 : 4));
  q <<= 8;
  unsigned even = (q & 0x80008000u) | ((q & 0x7f007f00u) >> (use_fp16 ? 1 : 4));
  auto* pairs = reinterpret_cast<nv_bfloat162*>(output);
  pairs[0] = exact_decode_word_pair<use_fp16>(even, scale);
  pairs[1] = exact_decode_word_pair<use_fp16>(odd, scale);
}

// scale is the exact, finite FP32 product of the saved scale and 2^120.
// raw BF16 codes are exactly code*2^-120, including subnormals/signed zero.
// Their FP32 product therefore has the same one RN step as code*saved_scale.
__device__ inline nv_bfloat162 exact_decode_prebiased_pair(unsigned raw,
                                                           float scale) {
  nv_bfloat162 raw_pair = *reinterpret_cast<nv_bfloat162*>(&raw);
  float2 values = __bfloat1622float2(raw_pair);
  return __floats2bfloat162_rn(__fmul_rn(values.x, scale),
                               __fmul_rn(values.y, scale));
}

__device__ inline void exact_decode_prebiased_fragment(int packed, float scale,
                                                       nv_bfloat16* output) {
  unsigned q = static_cast<unsigned>(packed);
  unsigned odd = (q & 0x80008000u) | ((q & 0x7f007f00u) >> 4);
  q <<= 8;
  unsigned even = (q & 0x80008000u) | ((q & 0x7f007f00u) >> 4);
  auto* pairs = reinterpret_cast<nv_bfloat162*>(output);
  pairs[0] = exact_decode_prebiased_pair(even, scale);
  pairs[1] = exact_decode_prebiased_pair(odd, scale);
}
