// SPDX-License-Identifier: Apache-2.0
template <int m_blocks, int fixed_split = 0>
__global__ void reduce_fragment_partials(const float4* parts, nv_bfloat16* out,
                                         int m, int n, int split_k) {
  static_assert(fixed_split == 0 || fixed_split == 2);
  constexpr int tile_m = 16 * m_blocks;
  constexpr int vectors_per_tile = tile_m * 128 / 4;
  constexpr int shared_stride = 128 + 8;
  __shared__ __align__(16) nv_bfloat16 reordered[tile_m * shared_stride];
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp = tid / 32;
  int64_t tile = static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  int64_t part_stride =
      static_cast<int64_t>(gridDim.x) * gridDim.y * vectors_per_tile;
#pragma unroll
  for (int i = 0; i < m_blocks; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        int64_t offset =
            tile * vectors_per_tile + ((i * 2 + j) * 2 + b) * 128 + tid;
        float4 value = parts[offset];
        for (int p = 1; p < (fixed_split ? fixed_split : split_k); ++p) {
          float4 other = parts[offset + p * part_stride];
          value.x = __fadd_rn(value.x, other.x);
          value.y = __fadd_rn(value.y, other.y);
          value.z = __fadd_rn(value.z, other.z);
          value.w = __fadd_rn(value.w, other.w);
        }
        int row = i * 16 + lane / 4;
        int col = warp * 32 + j * 16 + b * 8 + (lane % 4) * 2;
        *reinterpret_cast<nv_bfloat162*>(reordered + row * shared_stride +
                                         col) =
            __floats2bfloat162_rn(value.x, value.y);
        *reinterpret_cast<nv_bfloat162*>(reordered + (row + 8) * shared_stride +
                                         col) =
            __floats2bfloat162_rn(value.z, value.w);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < tile_m * 16 / 128; ++i) {
    int vector = tid + i * 128;
    int row = vector / 16;
    int col = vector % 16;
    int global_row = blockIdx.y * tile_m + row;
    if (global_row < m) {
      reinterpret_cast<int4*>(out)[static_cast<int64_t>(global_row) * (n / 8) +
                                   blockIdx.x * 16 + col] =
          reinterpret_cast<int4*>(reordered)[row * 17 + col];
    }
  }
}

__global__ void reduce_partials(const float* parts, void* out, int64_t size,
                                int split_k, bool output_float) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    float value = parts[index];
    for (int part = 1; part < split_k; ++part)
      value = __fadd_rn(value, parts[part * size + index]);
    if (output_float)
      static_cast<float*>(out)[index] = value;
    else
      static_cast<nv_bfloat16*>(out)[index] = __float2bfloat16_rn(value);
  }
}

// Explicit v58 experiment: same scalar layout/order, constant split16 only.
__global__ void reduce_partials_fixed16(const float* parts, void* out,
                                        int64_t size, int split_k,
                                        bool output_float) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    float value = parts[index];
#pragma unroll
    for (int part = 1; part < 16; ++part)
      value = __fadd_rn(value, parts[part * size + index]);
    if (output_float)
      static_cast<float*>(out)[index] = value;
    else
      static_cast<nv_bfloat16*>(out)[index] = __float2bfloat16_rn(value);
  }
}
