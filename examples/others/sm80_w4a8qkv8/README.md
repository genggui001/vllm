# SM80 W4A8 QKV8 serving

This branch adds `FLASH_ATTN_QKV_FP8_SM80_FUSED` attention and
`marlin_fp8_qdq_fused` experts to vLLM 0.28.0. It retains the measured Prefill
staging implementation, adaptive Decode kernels, incremental block-table
uploads, and optional exact 2048-token CUDA graphs. The native-prepared release
also fuses Gemma residual RMSNorm with expert input QDQ and QK norm/MRoPE with
query QDQ for the supported Qwen3.5 layout. Already prepared expert inputs use
native CUDA routing for 513–2047 tokens; the exact 2048-token path retains the
measured prewarmed routing kernels. Other shapes keep the general fallback.
The shared-expert stream is additionally used at exactly 2048 tokens only by
this custom MoE backend. SM80 GDN chunk configurations are pinned for the
validated BF16 layout to avoid autotuning-dependent accumulation differences.
The staged FA2 empty-KV LSE indexing fix is applied only to the custom target.

Neither the backend nor the launch script sets `OMP_NUM_THREADS`. For example,
`OMP_NUM_THREADS=1 bash serve.sh /path/to/model` explicitly requests one thread;
leaving it unset retains upstream vLLM runtime and weight-loading policies.

On A100, W4A8 means W4 weights with QAT-compatible E4M3 activation
quantize/dequantize before each expert GEMM. The GEMMs still execute through
Marlin's BF16/FP16 tensor-core path. A100 does not execute native FP8 GEMMs.
Attention stores persistent K/V as E4M3 bytes, applies the checkpoint's static
Q/K/V scales, and computes FA2 attention in BF16.

## Supported configuration

- NVIDIA compute capability 8.0, BF16 decoder attention, head dimension 256.
- Paged K/V with 16 tokens per page and scalar FP32 Q/K/V scales.
- W4 compressed-tensors MoE weights and SILU/SwiGLU activation, with router
  weights applied after FC2. Leave `VLLM_MARLIN_INPUT_DTYPE` unset.
- Packed symmetric INT4 MoE weights, group size 128, without activation ordering;
  symmetric dynamic token E4M3 activations from the checkpoint configuration.
- The checkpoint must contain calibrated static per-tensor FP8 Q/K/V scales.
  These settings do not quantize an arbitrary BF16 checkpoint into this format.
- ALiBi, sliding windows, softcap, DCP, and output quantization are rejected.

`compilation.json` retains the original graph capture order and adds exactly
2048 tokens. Batches above 512 that lack an exact graph keep their unpadded
shape. The extra graph is opt-in; generic vLLM graph defaults are unchanged.
Prefill stages Q/K/V only when the longest query exceeds 16 tokens and
the calculated temporary storage fits 256 MiB. Other shapes use the existing
Decode/fallback kernels. The persistent cache remains FP8 in both paths.

## Launch

Install the supplied wheel in an environment with its required PyTorch/CUDA
dependencies. Activate that environment so `vllm` resolves to its console
script, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/others/sm80_w4a8qkv8/serve.sh /path/to/model
```

The FP8 profile keeps the checkpoint's original `input_activations` configuration
and passes `--kv-cache-dtype fp8_e4m3` (`fp8` is also accepted). On SM80, vLLM
automatically selects `FLASH_ATTN_QKV_FP8_SM80_FUSED` for a matching W4A8/QKV8
checkpoint and `marlin_fp8_qdq_fused` for its quantized MoE layers. No
`--hf-overrides`, `--attention-backend`, or `--moe-backend` option is needed.
The log reports the selected backends. Explicit attention backend settings take
precedence; an explicitly incompatible MoE backend raises an error instead of
discarding token FP8 activation quantization. Unsupported shapes and quantization
schemes remain subject to backend validation.

This automatic selection requires the `autoselect1` revision or its source
changes. The original `0.28.0+sm80w4a8qkv8.cu132` wheel predates this interface.
Its explicit-backend launch configuration remains accepted by this revision.
Externally selecting FP8 describes the actual E4M3 byte cache; the custom kernel
dequantizes it for BF16 attention math and applies query QDQ exactly once.

The example defaults to TP equal to the number of visible GPUs, maximum model
length 262144, batch budget 2048, maximum 256 sequences, and memory utilization
0.8. Prefix caching and cascade attention are disabled in the measured profile.
Use `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`, and `PORT` to override those values.
Additional command-line arguments are forwarded to `vllm serve`.

For a matched W4A16 comparison using the same checkpoint, graph policy and
serving limits:

```bash
CUDA_VISIBLE_DEVICES=0,1 PROFILE=w4a16 \
  bash examples/others/sm80_w4a8qkv8/serve.sh /path/to/model
```

That profile selects native `FLASH_ATTN` plus `marlin` and BF16 K/V. It disables
the expert activation QDQ and attention QKV QDQ. The synthetic HTTP benchmark
is `benchmarks/benchmark_sm80_w4a8qkv8.py`. Check GPU occupancy before every
timing batch and exclude concurrent compilation/profiling from timed runs.

## Build from source

Use a CUDA-enabled PyTorch build matching the CUDA toolkit and a compiler
supported by that toolkit. Install the build dependencies into a venv using
`uv`; do not use system Python or system package installation.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
source .venv/bin/activate
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=12 NVCC_THREADS=1 \
  bash examples/others/sm80_w4a8qkv8/build-wheel.sh
```

This is a source build. `setup.py` includes the custom SM80 FA2 target and
installs its shared library into `vllm/vllm_flash_attn` inside the wheel.
Upstream stock precompiled wheels do not contain this custom extension.
The release report records the exact toolkit, PyTorch, Python, binary hashes,
platform requirements and installation checks for the supplied wheel.

## Operator checks

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m pytest \
  tests/v1/attention/test_flash_attn_qkv_fp8_sm80_fused.py \
  tests/kernels/moe/test_marlin_fp8_qdq_fused.py
```

The tests compare against independent materialized QDQ/FA2 references and cover
empty queries, long KV, large batches, staging-budget boundaries, mixed query
lengths, CUDA Graph replay, and rejected invalid input layouts/devices. Device
metadata supplied by the engine must contain valid page IDs and sequence
offsets. These GPU-resident values are not copied to the host on each call.
Malformed dimensions and incompatible devices/layouts are rejected before
kernel launch. This validation does not add a device-to-host synchronization.

## Chat reasoning compatibility

When `reasoning` is present, `/v1/chat/completions` also serializes an identical
`reasoning_content` field. This applies to both the normal response message and
each streamed delta. The original `reasoning` field remains available. An empty
string is copied; a `None` value does not add the alias. Configure the model's
reasoning parser as usual, for example `--reasoning-parser qwen3`.

The serializer uses `setdefault`, preserving an explicitly supplied
`reasoning_content` extra field. Requests that disable reasoning output do not
gain reasoning text through this alias.
