#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 MODEL_PATH [additional vllm serve arguments]" >&2
    exit 2
fi
model_path="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
for device in "${devices[@]}"; do
    case "$device" in
        0 | 1 | 2 | 3) ;;
        *) echo "This SM80 example is restricted to physical GPUs 0–3." >&2; exit 2 ;;
    esac
done
case "${PROFILE:-fp8}" in
    fp8)
        attention_backend=FLASH_ATTN_QKV_FP8_SM80_FUSED
        moe_backend=marlin_fp8_qdq_fused
        ;;
    w4a16)
        attention_backend=FLASH_ATTN
        moe_backend=marlin
        ;;
    *) echo "PROFILE must be fp8 or w4a16" >&2; exit 2 ;;
esac

exec "${VLLM_BIN:-vllm}" serve "$model_path" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-${#devices[@]}}" \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --max-num-seqs 256 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype bfloat16 \
    --attention-backend "$attention_backend" \
    --moe-backend "$moe_backend" \
    --hf-overrides "$(cat "$script_dir/qwen3_next_quantization.json")" \
    --compilation-config "$(cat "$script_dir/compilation.json")" \
    --no-enable-prefix-caching \
    --disable-cascade-attn \
    --host 127.0.0.1 \
    --port "${PORT:-18080}" \
    "$@"
