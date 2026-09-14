# H20 W4A8/QKV8 release

Branch: `v0.28.0-sm90w4a8qkv8`.
Wheel version: `0.28.0+sm90w4a8qkv8.cu132`.

This release integrates the selected H20 native FP8 MoE kernels and the GDN
host argument preparation improvement into the vLLM package. It retains the
upstream FlashAttention 3 FP8 implementation. MoE activation processing and
matrix computation use native FP8; there is no FP8-to-BF16 QDQ/Marlin path.
GDN retains its existing BF16/FP32 computation.

## Selection and supported scope

The optimized schedule is part of the existing `cutlass` MoE backend. Automatic
backend selection uses the model's compressed-tensors INT4 weights and dynamic
per-token FP8 input activation configuration. Keep that activation configuration
in the model; do not replace it with `null`.

The H20 specialization checks device, dtype, shapes, and available native
operators. The validated TP=2 expert layout has 256 experts, hidden size 2048,
intermediate size 256 per rank, top-8 routing, and weight group size 128. Other
configurations retain upstream dispatch. H100 and other SM90 devices are not
claimed to have the same performance gains.

The selected kernels vary with batch size: narrow and prepared GEMMs for small
batches, resource and pipeline schedules for intermediate batches, and ping-pong
GEMMs for larger decode batches. Single-token decode also fuses SiLU,
quantization, and FC2. The release retains these useful schedules and necessary
fallbacks; unused experimental implementations and runtime source loaders are
excluded.

For the tested model:

```bash
vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 262144 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9
```

`--moe-backend cutlass --attention-backend FLASH_ATTN` can select the existing
backends explicitly. On the tested H20 configuration, FlashAttention selects
FA3. `OMP_NUM_THREADS` remains under the caller's control. No `H20_USE_SOURCE`,
`H20_SOURCE_ROOT`, or custom `sitecustomize.py` is needed for normal serving.

## Reproducible component build and complete wheel

The release wheel targets Linux x86-64 and CPython 3.12, with the tested
Torch 2.13.0+cu132, CUDA 13.2, and FlashInfer 0.6.16.post3 stack. The build uses
CUTLASS 4.5.0 headers bundled with that FlashInfer package and GCC 11.4.

The recipe recompiles the **complete MoE extension**, including the selected
SM90a kernels. Unchanged upstream libraries, including FA3, and vendored package
assets are reused from the unmodified vLLM 0.28.0 installation. This is a complete
installable wheel, not a claim that every upstream extension was recompiled.

Use an existing environment with these dependencies and commit the release
checkout first. Choose new output directories outside the checkout:

```bash
RELEASE_PYTHON=/path/to/validated/environment/bin/python
RELEASE_OUTPUT=/path/to/new/release-output

"$RELEASE_PYTHON" tools/sm90_release/build_moe.py \
    --output "$RELEASE_OUTPUT/moe-build"

"$RELEASE_PYTHON" tools/sm90_release/package_wheel.py \
    --build "$RELEASE_OUTPUT/moe-build" \
    --output "$RELEASE_OUTPUT/package"
```

The scripts neither install packages nor overwrite the existing environment.
They capture source hashes and compile commands, preserve the caller's OMP
setting, validate the upstream installation against its `RECORD`, regenerate
wheel metadata and hashes, and unpack the result for acceptance testing.
The rebuilt library uses paths relative to its installation for Torch and CUDA
runtime dependencies. Build outputs and testing data stay outside the checkout.

The wheel is written to `package/dist/`; its package files are unpacked into
`package/wheel-test/site/`. `package-status.json` records the wheel SHA256 and
requires a separate runtime acceptance result. `vllm/sm90_release.json` inside
the wheel records the source commit and the provenance of rebuilt and reused
libraries.

To install into another environment with the matching dependencies already
prepared:

```bash
python -m pip install --no-deps \
    /path/to/vllm-0.28.0+sm90w4a8qkv8.cu132-cp312-cp312-linux_x86_64.whl
```

## Acceptance requirements

Test the unpacked wheel directly with its package directory on `PYTHONPATH`,
without an experimental source overlay. Check operator outputs and CUDA Graph
replay before throughput and stability tests. The release acceptance covers:

- Full MoE, Top-k, output restoration, and GDN comparisons, including extreme
  routing, disabled experts, zero scales, large values, and changing graph inputs.
- Compute Sanitizer checks for the complete MoE operation.
- Fixed requests, batch, ordering, and sampling for full-vocabulary logits and
  generated token comparisons against the previously validated implementation.
- Long-context boundaries, large and queued batches, mixed traffic, long
  generation, sustained load, request cancellation, and recovery after rejecting
  an overlength request.
- Actual kernel dispatch and package provenance, with GPU ownership monitored.

The preceding medical evaluation used the fixed A100-origin subset on H20:
CHIP-CDEE 100 cases, CMeEE 100 cases, and MedSafety 242 requests in 50 qid groups.
Three versions, base/r/enr, five seeds, and an additional same-seed repeat totaled
48 rounds and 21,216 requests. Optimization-minus-original differences ranged
from -1.51 to +1.60 percentage points; all nine paired t and bootstrap 95%
intervals included zero. These are fixed-subset results, not a proof of strict
equivalence on every input. The release's separate acceptance record must also
identify the exact wheel that was tested.
