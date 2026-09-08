# A100 SM80 QAT FP8 backend: implementation and evaluation

Date: 2026-09-09

## Outcome

The A100 implementation now has independent, opt-in backends for both parts of
the QAT inference path:

- `TRITON_ATTN_QKV_FP8_SM80`: stores K/V as real E4M3FN bytes and applies the
  checkpoint's static Q scale as an in-place BF16 -> E4M3FN -> BF16 QDQ pass.
  Attention math remains BF16 on SM80.
- `marlin_fp8_qdq_fused`: keeps the W4A16 Marlin GEMMs, fuses the FC1 input QDQ
  and the SwiGLU + FC2-input QDQ preprocessing, and keeps `router_prob` after
  FC2.

These are separate from the original attention and MoE implementations. They
are selected explicitly through `--attention-backend` and `--moe-backend`.

Branch and final implementation commit:

```text
a100-qkv-fp8-sm80-v028
30cf6dfde4 feat(attention): add SM80 QKV FP8 QAT backend
```

## Numerical semantics

The Q path uses the checkpoint/QAT static scale and performs:

```text
BF16 query
  -> FP32 / scale
  -> clamp to E4M3FN finite range (+/-448)
  -> software E4M3FN round-to-nearest-even
  -> multiply by scale
  -> BF16 query consumed by attention
```

Q is quantized once in-place before attention. An earlier experiment put QDQ
inside the attention tile, but that increased register pressure and made the
attention kernel 25-27% slower, so it was not retained. The selected side
kernel costs about 0.023 ms/layer and allocates no replacement query tensor.

K/V are stored as `uint8` E4M3FN cache bytes and decoded to BF16 while loading
an attention tile. This is a real FP8-capacity cache rather than a BF16 cache
with a cosmetic quantization step.

## Correctness tests

The final attention backend passed:

- all 256 E4M3 byte-pattern decode cases;
- exact in-place Q QDQ comparison with a materialized PyTorch reference;
- preservation of query rows beyond `num_actual_tokens`;
- exact cache-scatter byte checks;
- whole-attention comparison against materialized BF16 QDQ reference;
- backend registry and SM80 guard tests.

The fused MoE backend passed 12 GPU numerical equivalence cases plus 12 backend
mapping cases for FP16/BF16 and representative shapes.

## Throughput and cache capacity

Measurement: TP=2 on physical A100 GPUs 0,1, 64 concurrent requests, 256 output
tokens/request, prefix caching disabled. The final value is the median of six
runs; the first post-start run was cold but does not affect the median.

| Configuration | Throughput (tok/s) | Delta vs final |
|---|---:|---:|
| Native Marlin, BF16 KV + FA2 | 4935.2 | final is -6.62% |
| Fused MoE QDQ, BF16 KV + FA2 | 4826.9 | final is -4.53% |
| Fused MoE QDQ, FP8 KV only | 4631.0 | final is -0.49% |
| **Fused MoE QDQ, QKV FP8 (final)** | **4608.3** | **baseline** |
| Unfused MoE QDQ reference | 3923.6 | final is +17.45% |

At `--gpu-memory-utilization 0.8`, the final service allocated a 2,491,416-token
KV cache. The corresponding BF16-cache service allocated 1,307,195 tokens, so
usable KV capacity increased by about 1.91x (+90.6%).

## Full standalone MedBench evaluation

All runs used 64-way concurrency, `max_tokens=32768`, and seed 0. `base` used
temperature 0.7/top-p 0.8/top-k 20; `r` and `enr` used temperature 1.0/top-p
0.95. Dataset sizes were 100 CHIP-CDEE, 100 CMeEE, and 242 MedSafety samples.

### Final QKV FP8 scores

| Profile | CHIP-CDEE | CMeEE | MedSafety |
|---|---:|---:|---:|
| base | 43.9183 | 54.9476 | 72.0000 |
| r | 40.4476 | 55.8789 | 76.0000 |
| enr | 40.0966 | 57.6151 | 80.0000 |

Range across the three profiles is 3.8217 points on CHIP-CDEE, 2.6675 on
CMeEE, and 8.0000 on MedSafety.

### Change from the previous A100 FP8-KV-only candidate

| Profile | CHIP-CDEE | CMeEE | MedSafety |
|---|---:|---:|---:|
| base | -0.9261 | +0.5514 | -2.0000 |
| r | -1.4388 | -2.2007 | +2.0000 |
| enr | -5.1472 | -3.0194 | 0.0000 |

The full final run produced 1,326/1,326 valid responses: zero HTTP/model
errors, zero empty responses, zero U+FFFD replacement characters, and zero NUL
characters. All 884 `r`/`enr` responses contained exactly one `</ggthink>`.
The earlier repeated/corrupt reasoning-output failure did not reproduce.

### Deterministic Q-only A/B for the non-reasoning profile

To isolate the effect of Q QDQ, two services were run simultaneously with the
same model, MoE backend, FP8 K/V backend, cache capacity, prompts, seed, and
sampling parameters. The control used BF16 Q and the experiment used FP8-QDQ
Q. Both used temperature 0. The only intended numerical difference was Q QDQ.

| Attention path | CHIP-CDEE | CMeEE | MedSafety |
|---|---:|---:|---:|
| BF16 Q + FP8 K/V | 47.7272 | 57.6388 | 76.0000 |
| Q QDQ + FP8 K/V | 49.7165 | 57.5057 | 78.0000 |
| **Q QDQ delta** | **+1.9893** | **-0.1331** | **+2.0000** |

Both sides produced 442/442 valid responses with no corruption. Exact output
agreement was 42/100 on CHIP-CDEE, 72/100 on CMeEE, and 234/242 on MedSafety,
which confirms that Q QDQ materially changes the numerical path rather than
being optimized away.

A temperature-0 paired run was also attempted for the reasoning profile, but
30 control requests and 15 QDQ requests entered deterministic loops that were
still generating toward the 32K-token limit. The diagnostic was stopped and is
not reported as a score. This behavior occurred on both attention paths;
temperature 0 is useful for the short non-reasoning A/B, but is not a suitable
production setting for this model's reasoning profiles.

## Interpretation and next accuracy experiment

The engineering result is positive: Q QDQ adds only about 0.49% throughput
cost on top of the real FP8-KV backend, the FP8 cache capacity is retained, and
long concurrent generation is stable. The paired temperature-0 `base` result
also establishes that Q QDQ is an effective non-reasoning accuracy change on
this A100 setup: two datasets improve by about two points and the third changes
by only -0.13 point.

This run does **not** establish that A100 accuracy has caught H20. The full-run
comparison above is against the previous A100 FP8-KV-only candidate, and its
profiles use non-zero temperature. Their mixed score deltas therefore include
sampling-path divergence after the intended Q perturbation.

The remaining discriminating test is to run the same standalone non-reasoning
`base` evaluation at temperature 0 on the H20 CUTLASS W4A8 + QKV FP8 baseline
and compare it with the two saved A100 results above. Use identical prompts,
limits, tokenizer/chat template, checkpoint scales, and post-FC2 `router_prob`
order. Compare per-sample exact answer agreement and score deltas in addition
to aggregate dataset scores. For `r` and `enr`, retain the production
temperature of 1.0 and use multiple fixed seeds rather than temperature 0.

## Reproduction

Start the final backend from the repository root:

```bash
GPU_SET=0,1 \
ATTENTION_BACKEND=TRITON_ATTN_QKV_FP8_SM80 \
MOE_BACKEND=marlin_fp8_qdq_fused \
ENABLE_PREFIX_CACHING=false \
SERVED_MODEL_NAME=pulse-v20-qkv-fp8-moe-qdq-nopc \
benchmarks/qat_accuracy/launch_v20_marlin_fp8_qdq.sh
```

Run one profile without OpenCompass:

```bash
python benchmarks/qat_accuracy/eval_medbench_subset.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model pulse-v20-qkv-fp8-moe-qdq-nopc \
  --profile base \
  --output benchmarks/qat_accuracy/results/v20_qkv_fp8_moe_qdq/base.json \
  --concurrency 64 \
  --max-tokens 32768 \
  --seed 0
```

Raw JSON results remain local under
`benchmarks/qat_accuracy/results/v20_qkv_fp8_moe_qdq/` and are intentionally
excluded from Git.
