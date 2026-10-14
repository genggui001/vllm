#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "hardware_info.h"
#include "namespace_config.h"
#include "flash_sm80_fp8.h"

#define CHECK_DEVICE(x) STD_TORCH_CHECK((x).is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_SHAPE(x, ...)                          \
  STD_TORCH_CHECK((x).sizes().equals({__VA_ARGS__}), \
                  #x " must have shape (" #__VA_ARGS__ ")")

namespace FLASH_NAMESPACE {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// Match the pinned native FA2 split-count heuristic for group32 decode.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                                int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 || ceildiv(num_n_blocks, num_splits) !=
                                  ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

static inline cudaStream_t get_current_cuda_stream_sm80_fp8(const Tensor& t) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

static void check_scale(const Tensor& scale, const Tensor& reference,
                        const char* name) {
  CHECK_DEVICE(scale);
  STD_TORCH_CHECK(scale.scalar_type() == ScalarType::Float, name,
                  " must have dtype float32");
  STD_TORCH_CHECK(scale.numel() == 1, name, " must contain one value");
  STD_TORCH_CHECK(scale.get_device_index() == reference.get_device_index(),
                  name, " must be on the same device as q");
}

static void check_stage_workspace(const Tensor& workspace, const Tensor& q,
                                  int64_t budget,
                                  std::initializer_list<const Tensor*> inputs) {
  CHECK_DEVICE(workspace);
  CHECK_CONTIGUOUS(workspace);
  STD_TORCH_CHECK(workspace.get_device_index() == q.get_device_index(),
                  "workspace must be on the same device as q");
  STD_TORCH_CHECK(
      workspace.scalar_type() == ScalarType::Byte && workspace.dim() == 1,
      "workspace must be a one-dimensional uint8 tensor");
  STD_TORCH_CHECK(workspace.numel() >= budget,
                  "workspace is smaller than stage_budget_bytes");
  const auto start = reinterpret_cast<uintptr_t>(workspace.data_ptr());
  STD_TORCH_CHECK((start & 255U) == 0, "workspace must be 256-byte aligned");
  if (workspace.numel() == 0) return;
  const auto end = start + workspace.numel();
  // Include stride gaps: K/V and block tables can be noncontiguous views.
  for (const auto* tensor : inputs) {
    if (tensor->numel() == 0) continue;
    const auto input_start = reinterpret_cast<uintptr_t>(tensor->data_ptr());
    uintptr_t last = 0;
    for (int64_t i = 0; i < tensor->dim(); ++i) {
      STD_TORCH_CHECK(tensor->stride(i) >= 0, "negative strides unsupported");
      last += uintptr_t(tensor->size(i) - 1) * tensor->stride(i);
    }
    const auto input_end = input_start + (last + 1) * tensor->element_size();
    STD_TORCH_CHECK(end <= input_start || input_end <= start,
                    "workspace must not overlap attention inputs or output");
  }
}

static std::vector<Tensor> mha_varlen_fwd_sm80_fp8_impl(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal, bool return_softmax_lse,
    bool prequantized_q = false, bool group32_decode = false,
    int64_t num_splits = 0, int64_t stage_budget = 256LL * 1024 * 1024,
    const Tensor* stage_workspace = nullptr) {
  CHECK_DEVICE(q);
  torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
  const auto [cc_major, cc_minor] =
      get_compute_capability(get_current_device());
  STD_TORCH_CHECK(cc_major == 8 && cc_minor == 0,
                  "FA2 SM80 FP8 attention requires compute capability 8.0");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(seqused_k);
  CHECK_DEVICE(block_table);
  for (const auto* tensor : std::initializer_list<const Tensor*>{
           &k, &v, &out, &cu_seqlens_q, &seqused_k, &block_table}) {
    STD_TORCH_CHECK(tensor->get_device_index() == q.get_device_index(),
                    "all attention tensors must be on the same device as q");
  }
  STD_TORCH_CHECK(q.dim() == 3, "q must have shape [tokens, heads, 256]");
  STD_TORCH_CHECK(k.dim() == 4 && v.dim() == 4,
                  "k and v must have shape [blocks, 16, kv_heads, 256]");
  STD_TORCH_CHECK(out.dim() == 3, "out must have shape [tokens, heads, 256]");
  STD_TORCH_CHECK(
      cu_seqlens_q.dim() == 1 && seqused_k.dim() == 1 && block_table.dim() == 2,
      "sequence lengths must be 1D and block_table must be 2D");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16,
                  "q must have dtype bfloat16");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16,
                  "out must have dtype bfloat16");
  STD_TORCH_CHECK(k.scalar_type() == ScalarType::Byte,
                  "k must contain raw E4M3FN bytes as uint8");
  STD_TORCH_CHECK(v.scalar_type() == ScalarType::Byte,
                  "v must contain raw E4M3FN bytes as uint8");
  STD_TORCH_CHECK(cu_seqlens_q.scalar_type() == ScalarType::Int,
                  "cu_seqlens_q must have dtype int32");
  STD_TORCH_CHECK(seqused_k.scalar_type() == ScalarType::Int,
                  "seqused_k must have dtype int32");
  STD_TORCH_CHECK(block_table.scalar_type() == ScalarType::Int,
                  "block_table must have dtype int32");
  STD_TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 &&
                      out.stride(-1) == 1,
                  "q, k, v, and out must have contiguous last dimensions");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(seqused_k);
  STD_TORCH_CHECK(block_table.stride(-1) == 1,
                  "block_table must have a contiguous last dimension");
  check_scale(q_scale, q, "q_scale");
  check_scale(k_scale, q, "k_scale");
  check_scale(v_scale, q, "v_scale");

  constexpr int64_t max_int = std::numeric_limits<int>::max();
  for (const auto* tensor : std::initializer_list<const Tensor*>{
           &q, &k, &v, &out, &cu_seqlens_q, &seqused_k, &block_table}) {
    for (const auto size : tensor->sizes()) {
      STD_TORCH_CHECK(size <= max_int, "tensor dimensions exceed int32");
    }
  }
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int num_blocks = k.size(0);
  const int page_block_size = k.size(1);
  const int num_heads_k = k.size(2);
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int max_num_blocks_per_seq = block_table.size(1);

  STD_TORCH_CHECK(batch_size > 0 && batch_size <= 65535,
                  "batch size must be in [1, 65535]");
  STD_TORCH_CHECK(num_heads > 0 && num_heads <= 65535 && num_heads_k > 0,
                  "query heads must be in [1, 65535] and KV heads positive");
  STD_TORCH_CHECK(head_size == 256,
                  "the first FA2 SM80 FP8 kernel only supports head_dim=256");
  STD_TORCH_CHECK(page_block_size == 16,
                  "the first FA2 SM80 FP8 kernel only supports block_size=16");
  STD_TORCH_CHECK(num_heads % num_heads_k == 0,
                  "the number of query heads must be divisible by KV heads");
  CHECK_SHAPE(q, total_q, num_heads, head_size);
  CHECK_SHAPE(out, total_q, num_heads, head_size);
  CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
  CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(seqused_k, batch_size);
  CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
  STD_TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_q <= max_int - 63 &&
                      max_seqlen_k > 0 && max_seqlen_k <= max_int - 63,
                  "max sequence lengths must be positive and fit int32 tiles");
  STD_TORCH_CHECK((max_seqlen_k + 15) / 16 <= max_num_blocks_per_seq,
                  "block_table is too narrow for max_seqlen_k");
  STD_TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");
  for (const auto* tensor : std::initializer_list<const Tensor*>{&q, &out}) {
    STD_TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(tensor->data_ptr()) & 15U) == 0 &&
            tensor->stride(0) % 8 == 0 && tensor->stride(1) % 8 == 0,
        "q and out must have 16-byte-aligned rows and heads");
  }
  for (const auto* tensor : {&k, &v}) {
    STD_TORCH_CHECK(
        (reinterpret_cast<uintptr_t>(tensor->data_ptr()) & 7U) == 0 &&
            tensor->stride(0) % 8 == 0 && tensor->stride(1) % 8 == 0 &&
            tensor->stride(2) % 8 == 0,
        "k and v must have 8-byte-aligned pages, rows and heads");
  }
  if (stage_workspace != nullptr) {
    check_stage_workspace(*stage_workspace, q, stage_budget,
                          {&q, &k, &v, &out, &cu_seqlens_q, &seqused_k,
                           &block_table, &q_scale, &k_scale, &v_scale});
  }
  if (total_q == 0) {
    auto lse = torch::stable::new_empty(q, {num_heads, 0}, ScalarType::Float);
    return return_softmax_lse ? std::vector<Tensor>{out, lse}
                              : std::vector<Tensor>{out};
  }

  // Match FA2's decode-time GQA packing without materializing the
  // transpose.  For one query token per sequence, reinterpret the query-head
  // group as the logical sequence dimension and launch one CTA per KV head.
  // K/V is then loaded and dequantized once for all query heads sharing that
  // KV head.  Strides below express the logical [batch, group, kv_head, dim]
  // view directly over the physical [batch, kv_head, group, dim] tensor.
  const int q_head_groups = num_heads / num_heads_k;
  const bool pack_decode_gqa =
      max_seqlen_q == 1 && total_q == batch_size && q_head_groups > 1;
  const int kernel_num_heads = pack_decode_gqa ? num_heads_k : num_heads;
  const int kernel_seqlen_q = pack_decode_gqa ? q_head_groups : max_seqlen_q;
  int splits = 1;
  if (group32_decode) {
    STD_TORCH_CHECK(pack_decode_gqa && q_head_groups <= 16 && !prequantized_q,
                    "Group32 split-KV requires raw Q, one query per sequence, "
                    "and 2..16 query heads per KV head");
    STD_TORCH_CHECK(num_splits >= 0 && num_splits <= 128,
                    "Group32 num_splits must be in [0, 128]");
    splits = num_splits > 0
                 ? int(num_splits)
                 : num_splits_heuristic(batch_size * kernel_num_heads,
                                        get_num_sm(get_current_device()) * 2,
                                        (max_seqlen_k + 63) / 64, 128);
  }
  // Bound temporary BF16 storage independently of the persistent FP8 cache.
  // Small queries retain the current decode kernel and allocate no workspace.
  const int64_t stage_pages =
      (max_seqlen_k + page_block_size - 1) / page_block_size;
  const int64_t q_and_lse_bytes =
      int64_t(total_q) * num_heads * (head_size * 2 + 4);
  const int64_t bytes_per_page =
      4LL * page_block_size * num_heads_k * head_size + 4;
  const bool stage_prefill =
      max_seqlen_q > 16 && q_and_lse_bytes <= stage_budget &&
      int64_t(batch_size) * stage_pages <=
          (stage_budget - q_and_lse_bytes) / bytes_per_page;
  auto softmax_lse = torch::stable::new_empty(
      q,
      {num_heads,
       (return_softmax_lse || stage_prefill || splits > 1) ? total_q : 0},
      ScalarType::Float);
  Flash_fwd_sm80_fp8_params params{};
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();
  params.q_batch_stride = q.stride(0);
  params.q_row_stride = pack_decode_gqa ? q.stride(1) : q.stride(0);
  params.q_head_stride =
      pack_decode_gqa ? q_head_groups * q.stride(1) : q.stride(1);
  params.k_batch_stride = k.stride(0);
  params.k_row_stride = k.stride(1);
  params.k_head_stride = k.stride(2);
  params.v_batch_stride = v.stride(0);
  params.v_row_stride = v.stride(1);
  params.v_head_stride = v.stride(2);
  params.o_batch_stride = out.stride(0);
  params.o_row_stride = pack_decode_gqa ? out.stride(1) : out.stride(0);
  params.o_head_stride =
      pack_decode_gqa ? q_head_groups * out.stride(1) : out.stride(1);
  params.softmax_lse_ptr = (return_softmax_lse || stage_prefill || splits > 1)
                               ? softmax_lse.data_ptr()
                               : nullptr;
  params.cu_seqlens_q =
      pack_decode_gqa ? nullptr : static_cast<int*>(cu_seqlens_q.data_ptr());
  // Paged K/V addresses do not use sum_s_k. seqused_k is authoritative.
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_q.data_ptr());
  params.seqused_k = static_cast<int*>(seqused_k.data_ptr());
  params.block_table = static_cast<int*>(block_table.data_ptr());
  params.block_table_batch_stride = block_table.stride(0);
  params.page_block_size = page_block_size;
  params.b = batch_size;
  params.h = kernel_num_heads;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = pack_decode_gqa ? 1 : q_head_groups;
  params.seqlen_q = kernel_seqlen_q;
  params.seqlen_k = max_seqlen_k;
  // total_q remains the physical token count because softmax_lse is exposed
  // as [num_query_heads, total_q].  The packed-GQA kernel has a larger
  // logical row count, but stores each row back into that public layout.
  params.total_q = total_q;
  params.d = head_size;
  params.d_rounded = head_size;
  params.scale_softmax = static_cast<float>(softmax_scale);
  params.scale_softmax_log2 =
      static_cast<float>(softmax_scale) * static_cast<float>(M_LOG2E);
  params.p_dropout = 1.0f;
  params.rp_dropout = 1.0f;
  params.is_bf16 = true;
  // With one real query token, causal attention is identical to full
  // attention over its already-bounded KV cache.  Logical group rows all
  // represent that same token and therefore must not mask one another.
  params.is_causal = pack_decode_gqa ? false : is_causal;
  params.is_seqlens_k_cumulative = true;
  params.unpadded_lse = true;
  params.window_size_left = -1;
  params.window_size_right = params.is_causal ? 0 : -1;
  params.q_scale_ptr = static_cast<const float*>(q_scale.data_ptr());
  params.k_scale_ptr = static_cast<const float*>(k_scale.data_ptr());
  params.v_scale_ptr = static_cast<const float*>(v_scale.data_ptr());
  params.packed_decode_gqa = pack_decode_gqa;
  params.return_softmax_lse = return_softmax_lse;

  const auto stream = get_current_cuda_stream_sm80_fp8(q);
  if (splits > 1) {
    auto partial = torch::stable::new_empty(
        q, {splits, batch_size, kernel_num_heads, kernel_seqlen_q, head_size},
        ScalarType::Float);
    auto partial_lse = torch::stable::new_empty(
        q, {splits, batch_size, kernel_num_heads, kernel_seqlen_q},
        ScalarType::Float);
    params.num_splits = splits;
    params.oaccum_ptr = partial.data_ptr();
    params.softmax_lseaccum_ptr = partial_lse.data_ptr();
    params.seqlenq_ngroups_swapped = true;
    run_mha_fwd_sm80_fp8_split(params, stream);
  } else if (stage_prefill && stage_workspace != nullptr) {
    const int64_t q_bytes = int64_t(total_q) * num_heads * head_size * 2;
    const int64_t kv_bytes = int64_t(batch_size) * stage_pages *
                             page_block_size * num_heads_k * head_size * 2;
    auto* q_ptr = static_cast<char*>(stage_workspace->data_ptr());
    auto* k_ptr = q_ptr + q_bytes;
    auto* v_ptr = k_ptr + kv_bytes;
    auto* table_ptr = reinterpret_cast<int*>(v_ptr + kv_bytes);
    // Outputs and LSE own their storage and survive reuse by the next call.
    run_mha_fwd_sm80_fp8_staged(params, q_ptr, k_ptr, v_ptr, table_ptr,
                                stage_pages, stream, false);
  } else if (stage_prefill) {
    auto staged_q = torch::stable::new_empty(
        q, {prequantized_q ? 0 : total_q, num_heads, head_size},
        ScalarType::BFloat16);
    auto staged_kv = torch::stable::new_empty(
        q,
        {2, batch_size * stage_pages, page_block_size, num_heads_k, head_size},
        ScalarType::BFloat16);
    auto staged_table =
        torch::stable::new_empty(q, {batch_size, stage_pages}, ScalarType::Int);
    const int64_t kv_bytes = int64_t(batch_size) * stage_pages *
                             page_block_size * num_heads_k * head_size * 2;
    auto* kv_ptr = static_cast<char*>(staged_kv.data_ptr());
    run_mha_fwd_sm80_fp8_staged(params, staged_q.data_ptr(), kv_ptr,
                                kv_ptr + kv_bytes,
                                static_cast<int*>(staged_table.data_ptr()),
                                stage_pages, stream, prequantized_q);
  } else if (prequantized_q) {
    run_mha_fwd_sm80_fp8_prequantized_q(params, stream);
  } else {
    run_mha_fwd_sm80_fp8(params, stream);
  }
  return return_softmax_lse ? std::vector<Tensor>{out, softmax_lse}
                            : std::vector<Tensor>{out};
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal) {
  return mha_varlen_fwd_sm80_fp8_impl(
      q, k, v, out, cu_seqlens_q, seqused_k, block_table, q_scale, k_scale,
      v_scale, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, false);
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_lse(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal) {
  return mha_varlen_fwd_sm80_fp8_impl(
      q, k, v, out, cu_seqlens_q, seqused_k, block_table, q_scale, k_scale,
      v_scale, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, true);
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_prequantized_q(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal) {
  return mha_varlen_fwd_sm80_fp8_impl(q, k, v, out, cu_seqlens_q, seqused_k,
                                      block_table, q_scale, k_scale, v_scale,
                                      max_seqlen_q, max_seqlen_k, softmax_scale,
                                      is_causal, false, true);
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_lse_prequantized_q(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal) {
  return mha_varlen_fwd_sm80_fp8_impl(q, k, v, out, cu_seqlens_q, seqused_k,
                                      block_table, q_scale, k_scale, v_scale,
                                      max_seqlen_q, max_seqlen_k, softmax_scale,
                                      is_causal, true, true);
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_group32(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal, int64_t num_splits) {
  return mha_varlen_fwd_sm80_fp8_impl(
      q, k, v, out, cu_seqlens_q, seqused_k, block_table, q_scale, k_scale,
      v_scale, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, false,
      false, true, num_splits);
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_group32_lse(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal, int64_t num_splits) {
  return mha_varlen_fwd_sm80_fp8_impl(q, k, v, out, cu_seqlens_q, seqused_k,
                                      block_table, q_scale, k_scale, v_scale,
                                      max_seqlen_q, max_seqlen_k, softmax_scale,
                                      is_causal, true, false, true, num_splits);
}

std::vector<Tensor> mha_varlen_fwd_sm80_fp8_group32_prefill_workspace_lse(
    const Tensor& q, const Tensor& k, const Tensor& v, Tensor out,
    const Tensor& cu_seqlens_q, const Tensor& seqused_k,
    const Tensor& block_table, const Tensor& q_scale, const Tensor& k_scale,
    const Tensor& v_scale, int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal, int64_t stage_budget_bytes,
    Tensor workspace) {
  STD_TORCH_CHECK(
      stage_budget_bytes >= 0 && stage_budget_bytes <= 2LL * 1024 * 1024 * 1024,
      "Group32 prefill staging budget must be in [0, 2 GiB]");
  return mha_varlen_fwd_sm80_fp8_impl(
      q, k, v, out, cu_seqlens_q, seqused_k, block_table, q_scale, k_scale,
      v_scale, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, true,
      false, false, 0, stage_budget_bytes, &workspace);
}

}  // namespace FLASH_NAMESPACE
