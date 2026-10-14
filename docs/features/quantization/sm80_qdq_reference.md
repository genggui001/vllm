# SM80 group32 QDQ accuracy reference

This opt-in reference loads compressed-tensors Qwen3.5 MoE checkpoints with
INT4 group32 expert weights, FP8 group32 dense weights and activations, and
saved per-tensor Q/K/V FP32 scales. It uses BF16 matrix multiplication after
dequantization. It is intended as a precision baseline before optimizing kernels.

## Enable

```bash
vllm serve /path/to/exported-model \
    --quantization sm80_qdq_reference \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --attention-backend FLASH_ATTN \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --language-model-only
```

The checkpoint configuration remains unchanged. The explicit quantization
option selects the reference instead of the default compressed-tensors path.
`OMP_NUM_THREADS` is inherited from the caller.

For reproducible precision comparisons, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before starting the process, and add:

```bash
--additional-config '{"sm80_qdq_reference_deterministic":true}'
```

This explicitly enables PyTorch deterministic algorithms in the model worker,
including training's deterministic BF16 `index_add` unpermute branch. Without
this option, the normal training-style BF16 atomic reduction can produce
run-to-run numerical changes even when every QDQ operation is bit-exact.

For Qwen3.5 MTP, use the model's matching speculative method:

```bash
--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

Use MTP disabled for the initial reproducibility baseline. The MTP3 smoke test
passes loading, saved-scale, and runtime QDQ bit checks, but repeated requests
still produce different logprobs even with the deterministic option. That option
controls PyTorch algorithms; it does not guarantee determinism of every vLLM
kernel or speculative scheduling decision.

## Numerical contract

* Dense FP8 weights use the original FP8 codes and FP32 group32 scales.
  Multiplication occurs in FP32 and the dequantized result is rounded to BF16
  once during model loading. Scales are not rounded to BF16 or E8M0.
* MoE INT4 weights use compressed-tensors unsigned nibbles (offset 8) and
  the stored BF16 group32 scales. There is no online weight requantization.
* Activations use group32 absolute maximum divided by 448, with a FP32-tiny
  floor, FP32 scaling, saturated E4M3 conversion, and BF16 dequantization.
* Expert FC1 input and post-SwiGLU FC2 input each receive activation QDQ.
  The router probability multiplies the already-QDQ FC2 input, then the product
  is rounded to BF16 before FC2. It is not multiplied after FC2.
* SwiGLU preserves the separate BF16 SiLU, `(up + 0)`, and product boundaries,
  including in the shared expert.
* Expert outputs follow training's expert-major ordering and non-fused BF16
  `scatter_add` unpermute (or `index_add` when deterministic algorithms are
  enabled). This differs from a single FP32 top-k reduction. Atomic addition
  order can vary, as in the corresponding training implementation.
* Q/K QDQ occurs after normalization and RoPE. V is also QDQ before attention.
  All Q/K/V scales must be present, finite, positive FP32 checkpoint scalars.
  Missing scales fail loading; the reference never recalibrates them.
* The BF16 KV cache stores the already-QDQ K/V values. BF16 here specifies the
  storage format of the reference, while the values obey FP8 QDQ semantics.

The saved scale checks also cover MTP attention. MTP dense projections and
experts use their checkpoint quantization targets, including `mtp.fc`.

## Scope and validation

Start with eager execution, BF16 cache, FlashAttention, and tensor parallelism.
EP, sequence-parallel MoE, PP > 1, LoRA, other activation schemes, and missing
QKV scales are rejected. Dequantized expert weights consume substantially more
memory than the production packed-weight backend. Expert routing and activation
QDQ are batched; a TorchScript loop dispatches the original per-expert cuBLAS
GEMMs without a Python loop. FC1 and FC2 share one device-to-host transfer of
expert boundaries per MoE layer. This preserves the original GEMM shapes and
rounding behavior; it is not a fully asynchronous GPU grouped-GEMM backend.

The regression test contains training-generated golden hashes for every finite
BF16 encoding, signed zero/subnormals, INT4 packing, dtype preservation, and
invalid checkpoint-scale rejection:

```bash
.venv/bin/python -m pytest tests/quantization/test_sm80_qdq_reference.py -q
```

Quantizer bit equality does not imply an entire Megatron training forward is
bit-identical: GEMM implementations, attention, TP layout/collectives, GDN,
router selection, and expert addition order also affect results. Full-model
comparisons must fix requests, token history, batching, and sampling settings;
dataset accuracy must be reported separately from the operator checks.
