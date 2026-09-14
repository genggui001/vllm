# Native QKV FP8 and INT4/FP8 MoE on SM89

This backend runs QKV FP8 attention and INT4-weight, FP8-activation MoE on
SM89 GPUs. The attention implementation uses FA2 online tiling in Triton,
with FP8 numerical conventions based on FA3. Both attention matrix products
and both Marlin expert matrix products use native FP8 Tensor Core instructions.

## Release wheel

The `v0.28.0-sm89w4a8qkv8` branch provides the
`0.28.0+sm89w4a8qkv8.cu132` wheel for Linux x86-64, CPython 3.12,
and PyTorch 2.13.0 with CUDA 13.2. Native extensions are precompiled;
Triton and PyTorch can populate their compilation caches on first startup.
The validation hardware is a pair of RTX 4090 D GPUs (SM89, 114 SMs each).

```bash
uv pip install --python /path/to/python3.12 \
    ./vllm-0.28.0+sm89w4a8qkv8.cu132-cp312-cp312-linux_x86_64.whl
```

Retain the checkpoint configuration below when moving to another installation.

## Checkpoint configuration

Keep the checkpoint's original `compressed-tensors` quantization configuration
and scales. The two backends are selected independently:

| Component | Automatic selection conditions |
| --- | --- |
| Attention | SM89, E4M3 KV cache, static symmetric tensor FP8 `kv_cache_scheme`, and supported attention features |
| MoE | SM89, symmetric INT4 weights with group size 128 and no activation ordering, and dynamic symmetric token FP8 `input_activations` |

The MoE activation configuration includes:

```json
{
  "num_bits": 8,
  "type": "float",
  "strategy": "token",
  "dynamic": true,
  "symmetric": true
}
```

This is a fragment of an existing checkpoint recipe. Retain its weight
configuration, ignored layers, and other fields. Setting `input_activations`
to `null` selects a different numerical path. A backend flag alone does not
add a token-FP8 recipe to a W4A16 checkpoint.

## Running

For automatic selection with a matching checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5 vllm serve /path/to/checkpoint \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --max-cudagraph-capture-size 512 \
    --gpu-memory-utilization 0.85
```

Use GPU indices and memory/concurrency settings appropriate for the deployment.
The release performance checks use `max_num_seqs=128`, a 2048-token prefill
budget, and the following graph configuration in place of the graph-size flag:

```bash
--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512,1024,2048],"max_cudagraph_capture_size":2048}'
```

To select the backends explicitly, add:

```bash
--attention-backend FLASH_ATTN_FP8_SM89 \
--moe-backend marlin
```

With this checkpoint recipe, `marlin` consumes FP8 activations. With a W4A16
recipe, the same backend name consumes 16-bit activations. An explicitly
selected attention backend takes precedence over automatic selection.

The KV cache stores E4M3 bytes. Query quantization and KV updates use the
checkpoint's Q/K/V scales. This feature does not set `OMP_NUM_THREADS`; the
application may set that environment variable using normal vLLM behavior.

For BF16 GDN layers with 8 key heads, 16 value heads and 128-dimensional heads
per TP rank, the SM89 FP8 recipe also selects `triton_graph` for GDN prefill.
It prewarms single-sequence buckets up to 512 tokens before KV allocation.
The cache has a 256 MiB budget per worker for graph private pools and static
inputs; other shapes use the existing Triton/FLA path without runtime capture.
The budget excludes CUDA driver metadata and global allocator slack.
`--gdn-prefill-backend triton` disables these short-prefill graphs, while
`--gdn-prefill-backend triton_graph` selects them explicitly on supported
configurations. Eager and sleep modes use the existing path.

For this GDN head configuration, small BF16 decode batches of 1–8 sequences
use a smaller value tile on SM89. This reduces register pressure while retaining
the original single-warp reduction order and FP32/BF16 state behavior.

## Numerical behavior

Attention performs QK with E4M3 operands, combining 32-element partial products
in FP32. Score scaling, online softmax maxima, exponentials, and denominators
are FP32. The unnormalized exponentials are multiplied by 256 before conversion
to E4M3 for PV; the denominator is computed before that conversion. PV history,
split merging, and final normalization are FP32. Outputs are BF16 or FP16.
Ada's native MMA rounding still differs from a full FP32 reference GEMM. Loop
unrolling depends on the query tile shape while retaining the same 32-element
partial-product boundaries.

Decode can divide the KV sequence into segments. Different segment boundaries
can change FP8 probability rounding. Query heads that share a KV head are
grouped into one block for supported decode shapes while retaining the same
segment boundaries. Prefill and larger head groups use the generic path.

MoE quantizes inputs independently per token for FC1 and per expert-token for
FC2. The fused SiLU-and-quantization path preserves the CUDA activation's dtype
rounding before producing FP8 bytes and scales. Layers excluded by the original
quantization recipe retain their original precision.

On SM89, W4A8 Marlin routing with 256 experts and no expert mapping preserves
flattened token order within each expert for supported contiguous INT32/INT64
routing tensors. Up to 512 token/expert pairs use one CTA with block sizes 8
or 16. From 513 through 16384 pairs, two kernels count and sort each tile,
then compute offsets and scatter, supporting block sizes 8, 16, 32, 48, and 64.
Other shapes use the existing CUDA alignment. Stable expert ordering prevents
one source of changes in Marlin's floating-point reduction partitions; it does
not promise batch-invariant outputs for arbitrary workloads.

On SM89 devices with 114 SMs, the BF16 single-token path for 256 experts,
top-k 8, hidden size 2048, and per-rank intermediate size 256 uses native FP8
Marlin tiles with eight rows. It requires group-128 symmetric INT4 weights,
token FP8 activations, the fused SiLU/token-FP8 stage, and no bias, activation
ordering, expert mapping, or input router-weight multiplication. The FC2 tile
preserves the validated FP32 partial-sum scaling order. Other configurations
retain the existing Marlin dispatch. Both paths remain behind the same
optional backend and checkpoint-based selection.

## Supported attention features

- Compute capability 8.9, head sizes 64, 128, and 256.
- Dense causal decoder attention, ragged batches, GQA, paged E4M3 KV caches,
  chunked prefill, and CUDA Graph execution.
- BF16 or FP16 attention outputs.

Sliding windows, ALiBi, sinks, soft caps, noncausal attention, MLA, sparse
attention, multimodal prefix attention, context parallelism, cascade attention,
and per-head quantization scales are unsupported by this backend. It does not
advertise batch-invariant output. Features outside this list require separate
integration validation.
