#include <cutlass/numeric_types.h>

#include "flash_fwd_sm80_fp8_kernel.h"
#include "cuda_check.h"
#include "hardware_info.h"
#include "static_switch.h"

namespace FLASH_NAMESPACE {

using Sm80Fp8Traits = Flash_fwd_kernel_traits<
    256, 64, 64, 4, true, true, cutlass::bfloat16_t>;

__global__ void flash_fwd_sm80_fp8_kernel(
    const Flash_fwd_sm80_fp8_params params) {
    compute_attn_sm80_fp8<Sm80Fp8Traits>(params);
}

void run_mha_fwd_sm80_fp8(Flash_fwd_sm80_fp8_params &params,
                          cudaStream_t stream) {
    constexpr size_t smem_size =
        Sm80Fp8Traits::kSmemSize + 2 * 256 * sizeof(cutlass::bfloat16_t);
    const dim3 grid((params.seqlen_q + Sm80Fp8Traits::kBlockM - 1) /
                        Sm80Fp8Traits::kBlockM,
                    params.b, params.h);
    if (smem_size >= 48 * 1024) {
        FLASHATTENTION_CUDA_CHECK(cudaFuncSetAttribute(
            flash_fwd_sm80_fp8_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    flash_fwd_sm80_fp8_kernel<<<grid, Sm80Fp8Traits::kNThreads, smem_size,
                                stream>>>(params);
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace FLASH_NAMESPACE
