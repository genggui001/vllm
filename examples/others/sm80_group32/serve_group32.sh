#!/usr/bin/env bash
# TP2 text-only deployment example; OMP is inherited.
set -euo pipefail

: "${MODEL:?Set MODEL to the original v20.1 group32 checkpoint directory}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#gpu_ids[@]} != 2 || ${gpu_ids[0]} == "${gpu_ids[1]}" ]]; then
    echo "This TP2 example requires two distinct physical GPUs from 0,1,2,3." >&2
    exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! $gpu_id =~ ^[0-3]$ ]]; then
        echo "Only physical GPUs 0,1,2,3 are allowed in this environment." >&2
        exit 2
    fi
done

cmd=("${VLLM_BIN:-vllm}" serve "$MODEL"
    --dtype bfloat16
    --kv-cache-dtype fp8_e4m3
    --served-model-name "${SERVED_MODEL_NAME:-Dayi-v20.1-35b-a3b-group32}"
    --tensor-parallel-size 2
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}"
    --max-num-seqs "${MAX_NUM_SEQS:-256}"
    --enable-chunked-prefill
    --enable-prefix-caching
    --language-model-only
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser-plugin "$script_dir/ggthink_parser.py"
    --reasoning-parser group32_ggthink
    --default-chat-template-kwargs '{"enable_thinking":true}'
    --host "${HOST:-0.0.0.0}"
    --port "${PORT:-8000}")

if [[ -n ${MAX_NUM_BATCHED_TOKENS:-} ]]; then
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
case "${MTP_TOKENS:-0}" in
    0) ;;
    1|3)
        cmd+=(--speculative-config
            "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
        ;;
    *) echo "MTP_TOKENS must be 0, 1, or 3." >&2; exit 2 ;;
esac
case "${CASCADE_ATTN:-0}" in
    0) ;;
    1) cmd+=(--no-disable-cascade-attn) ;;
    *) echo "CASCADE_ATTN must be 0 or 1." >&2; exit 2 ;;
esac

if [[ ${1:-} == --print-argv ]]; then
    shift
    printf '%s\0' "${cmd[@]}" "$@"
    exit 0
fi
if [[ ${1:-} == --dry-run ]]; then
    shift
    printf '%q ' "${cmd[@]}" "$@"
    printf '\n'
    exit 0
fi

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,name,memory.used,utilization.gpu --format=csv
active_pids=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ $active_pids =~ [0-9] ]]; then
    echo "Selected GPUs have active compute processes; check occupancy first." >&2
    printf '%s\n' "$active_pids" >&2
    exit 1
fi
exec "${cmd[@]}" "$@"
