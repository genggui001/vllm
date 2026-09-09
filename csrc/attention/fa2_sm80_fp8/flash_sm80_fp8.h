#pragma once

#include <cuda_runtime.h>

#include "flash.h"

namespace FLASH_NAMESPACE {

struct Flash_fwd_sm80_fp8_params : public Flash_fwd_params {
    const float *__restrict__ q_scale_ptr;
    const float *__restrict__ k_scale_ptr;
    const float *__restrict__ v_scale_ptr;
};

void run_mha_fwd_sm80_fp8(Flash_fwd_sm80_fp8_params &params,
                          cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
