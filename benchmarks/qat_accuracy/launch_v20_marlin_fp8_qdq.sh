#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

source /mnt/nas/home/genggui/miniconda3/etc/profile.d/conda.sh
conda activate vllm_dev
set -u

# This development run is intentionally restricted to the user-approved GPUs.
gpu_set="${GPU_SET:-0,1}"
case "$gpu_set" in
    0,1 | 2,3 | 0,1,2,3) ;;
    *)
        echo "GPU_SET must be one of: 0,1; 2,3; 0,1,2,3" >&2
        exit 2
        ;;
esac
export CUDA_VISIBLE_DEVICES="$gpu_set"
cuda_target="$CONDA_PREFIX/targets/x86_64-linux"
export CUDA_INC_PATH="$cuda_target"
export CUDA_LIB_PATH="$cuda_target"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$cuda_target/lib/stubs:${LIBRARY_PATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MKL_SERVICE_FORCE_INTEL=TRUE
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset VLLM_TARGET_DEVICE

model_path="${MODEL_PATH:-/mnt/nas/home/genggui/code/Megatron-Next/model_dir/pulse_v20_35b_a3b_next_gemini_bf16/outputs_kl_fake_wint4afp8qkvfp8/checkpoint/finetune-kl-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-kl-0.05-bs-1-gbs-8-seqlen-65536-pr-bf16-tp-8-pp-3-cp-1-ac-full-do-true-sp-true-ti-20480-wi-8-best-hf-moe-int4-fp8}"
served_model_name="${SERVED_MODEL_NAME:-pulse-v20-marlin-fp8-qdq}"
port="${PORT:-18080}"
moe_backend="${MOE_BACKEND:-marlin_fp8_qdq}"
case "$moe_backend" in
    marlin | marlin_fp8_qdq | marlin_fp8_qdq_fused) ;;
    *)
        echo "MOE_BACKEND must be marlin, marlin_fp8_qdq, or marlin_fp8_qdq_fused" >&2
        exit 2
        ;;
esac
attention_backend="${ATTENTION_BACKEND:-FLASH_ATTN}"
case "$attention_backend" in
    FLASH_ATTN | FLASH_ATTN_FP8_QDQ_SM80 | FLASH_ATTN_KV_FP8_QDQ_SM80 | FLASH_ATTN_QKV_FP8_SM80_FUSED | TRITON_ATTN_FP8_SM80 | TRITON_ATTN_QKV_FP8_SM80) ;;
    *)
        echo "ATTENTION_BACKEND must be FLASH_ATTN, FLASH_ATTN_FP8_QDQ_SM80, FLASH_ATTN_KV_FP8_QDQ_SM80, FLASH_ATTN_QKV_FP8_SM80_FUSED, TRITON_ATTN_FP8_SM80, or TRITON_ATTN_QKV_FP8_SM80" >&2
        exit 2
        ;;
esac
enable_prefix_caching="${ENABLE_PREFIX_CACHING:-true}"
prefix_cache_args=()
case "$enable_prefix_caching" in
    true) ;;
    false) prefix_cache_args+=(--no-enable-prefix-caching) ;;
    *)
        echo "ENABLE_PREFIX_CACHING must be true or false" >&2
        exit 2
        ;;
esac
enable_cascade_attention="${ENABLE_CASCADE_ATTENTION:-false}"
cascade_args=()
case "$enable_cascade_attention" in
    true) cascade_args+=(--no-disable-cascade-attn) ;;
    false) ;;
    *)
        echo "ENABLE_CASCADE_ATTENTION must be true or false" >&2
        exit 2
        ;;
esac

hf_overrides='{"quantization_config":{"config_groups":{"group_0":{"format":"pack-quantized","input_activations":null,"output_activations":null,"targets":["Linear"],"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":128,"num_bits":4,"observer":"minmax","observer_kwargs":{},"strategy":"group","symmetric":true,"type":"int"}}},"format":"pack-quantized","ignore":["re:.*self_attn.*","re:.*linear_attn.*","re:.*shared_expert.*","re:.*mlp[.](gate|up|gate_up|down)_proj.*","re:.*lm_head.*","re:.*mtp.*","re:.*visual.*"],"kv_cache_scheme":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"observer":"minmax","observer_kwargs":{},"scale_dtype":null,"strategy":"tensor","symmetric":true,"type":"float","zp_dtype":null},"quant_method":"compressed-tensors","quantization_status":"compressed"}}'

exec vllm serve "$model_path" \
    --served-model-name "$served_model_name" \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 256 \
    --kv-cache-dtype bfloat16 \
    --attention-backend "$attention_backend" \
    --moe-backend "$moe_backend" \
    --hf-overrides "$hf_overrides" \
    "${prefix_cache_args[@]}" \
    "${cascade_args[@]}" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$port"
