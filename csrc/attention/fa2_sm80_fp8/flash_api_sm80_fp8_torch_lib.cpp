#include "namespace_config.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <vector>

#include "registration.h"

using torch::stable::Tensor;

namespace FLASH_NAMESPACE {

std::vector<Tensor> mha_varlen_fwd_sm80_fp8(
    const Tensor &q, const Tensor &k, const Tensor &v, Tensor out,
    const Tensor &cu_seqlens_q, const Tensor &seqused_k,
    const Tensor &block_table, const Tensor &q_scale, const Tensor &k_scale,
    const Tensor &v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal);

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_lse(
    const Tensor &q, const Tensor &k, const Tensor &v, Tensor out,
    const Tensor &cu_seqlens_q, const Tensor &seqused_k,
    const Tensor &block_table, const Tensor &q_scale, const Tensor &k_scale,
    const Tensor &v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal);

}  // namespace FLASH_NAMESPACE

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "varlen_fwd(Tensor q, Tensor k, Tensor v, Tensor! out, "
        "Tensor cu_seqlens_q, Tensor seqused_k, Tensor block_table, "
        "Tensor q_scale, Tensor k_scale, Tensor v_scale, int max_seqlen_q, "
        "int max_seqlen_k, float softmax_scale, bool is_causal) -> Tensor[]");
    ops.def(
        "varlen_fwd_lse(Tensor q, Tensor k, Tensor v, Tensor! out, "
        "Tensor cu_seqlens_q, Tensor seqused_k, Tensor block_table, "
        "Tensor q_scale, Tensor k_scale, Tensor v_scale, int max_seqlen_q, "
        "int max_seqlen_k, float softmax_scale, bool is_causal) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("varlen_fwd",
             TORCH_BOX(&FLASH_NAMESPACE::mha_varlen_fwd_sm80_fp8));
    ops.impl("varlen_fwd_lse",
             TORCH_BOX(&FLASH_NAMESPACE::mha_varlen_fwd_sm80_fp8_lse));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
