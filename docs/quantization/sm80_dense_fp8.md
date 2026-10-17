# SM80 Dense FP8 group32

The `sm80_group32` backend uses a dedicated Dense FP8 core. Its API takes
E4M3 activations and weights with independent FP32 scales for each group of
32 K elements and produces BF16 output. The checkpoint codes and FP32
weight scales are packed at loading time without requantization. There is
no BF16 GEMM fallback in this backend. `sm80_qdq_reference` remains an
explicit, separate precision reference for testing.

## Dispatch and numeric contract

`_C::sm80_fp8_g32_gemm` selects the implementation from logical M, N and K.
Model layers do not select kernel versions. Tuned families cover small-M
QKV/out, large-M shared-A projections, split-K FC/gate, compact BA and
asynchronously loaded weight/scale records. Other supported shapes use a
generic FP8 group32 kernel, including M tails and padded N. Packing is
performed once; no runtime autotuning or full-weight BF16 expansion occurs.

The scalar reference reconstructs each operand as
`BF16_RNE(FP32(code) * FP32(scale))`, then applies BF16 `F.linear`.
Split-K and alternative accumulation orders can differ from that reference.
Report absolute RMSE and normalized RMSE = RMSE / max(RMS(reference), 1e-30),
plus max absolute error and nonfinite counts. For empty results, both RMSE
and normalized RMSE are zero. Bit equality is diagnostic rather than an
acceptance requirement. Kernel error tests do not establish model accuracy.

## Fused activation producers

- Decoder residual RMSNorm emits FP8 codes/scales for Dense consumers while
  retaining the BF16 residual and router input. The routed MoE path keeps its
  previous quantization and router-probability order.
- Shared experts consume an explicit `Group32FP8Input`; the carrier is passed
  through the shared-expert custom-op boundary and existing stream events.
  It is never stored as mutable state on the layer.
- Large-M shared SwiGLU writes FP8 codes/scales directly. Small-M execution
  preserves the existing overlap schedule and quantizes the training SwiGLU
  result in a separate producer.
- Ordinary GDN gated RMSNorm writes FP8 directly for the output projection.
  Existing fused speculative GDN decode is retained; that path quantizes its
  normalized BF16 result before the FP8 GEMM.
- MTP's two independent pre-FC norms write directly into separate slices of
  the final FP8 concatenation buffer.

Intermediate BF16 values required by residuals, routing and the SM80 tensor
core arithmetic are part of the computation. They are not an alternative
BF16 Dense backend.

## Startup support checks

Before the engine is ready, model loading validates the compiled operator,
SM80 device, bias-free Dense projections, E4M3 weights, positive finite FP32
G32 scales, K alignment and scheduler/graph indexing bounds. K must be a
multiple of 128 in [128, 65536]. N must be positive; padded shapes and the
maximum scheduled/captured token count must fit signed 32-bit indexing.
M is limited to 524280, with stricter limits where M*K or M*padded_N requires
it. Missing kernels or unsupported static configurations raise an error.

The existing backend's text-only, BF16, model-family and parallelism checks
still apply. The fused Gemma norms support widths 256/2048; GDN gated norms
support head widths 128/256. Dense retains runtime argument checks for
malformed direct operator calls.

## Verification

Run the existing numerical suite against an extension containing the new
stable operator:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/quantization/test_sm80_qdq_reference.py -q
```

Run the resident-core benchmark on a free SM80 GPU:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
  benchmarks/kernels/benchmark_sm80_dense_fp8.py \
  --output /tmp/dense-fp8-speed.json --seed 20261023 --rounds 5
```

It measures Graph/eager ABBA rounds against native W8A16 Marlin G128 and
resident BF16 `F.linear`. A is converted to BF16 before the native timing.
Weight packing and activation conversion are excluded. Synthetic G128
weight scales are repeated four times for the G32 candidate so the logical
reference data match; the G32-specific correctness suite remains separate.
The JSON preserves raw timing samples, chronology and numerical errors.

The kernel families derive from the retained standalone Marlin experiments;
original template bodies are retained. The integration evidence manifest
records their source hashes and the mapping of selected shapes to profiles.
