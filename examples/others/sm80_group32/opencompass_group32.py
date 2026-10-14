# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast-backend configuration for the locally extended OpenCompass."""

from opencompass.models import VLLMwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

model_path = (
    "/mnt/nas/home/genggui/code/Megatron-Next/model_dir/"
    "pulse_v20_1_35b_a3b_next_gemini_bf16/"
    "outputs_kl_fake_wint4agfp8qkvfp8/checkpoint/"
    "finetune-kl-mcore-qwen3-next-A3B-lr-7e-6-minlr-7e-7-kl-0.05-"
    "bs-1-gbs-8-seqlen-65536-pr-bf16-tp-8-pp-3-cp-1-ac-full-"
    "do-true-sp-true-ti-20480-wi-8-best-hf-int4fp8"
)
seed = 20260926
models = []

for mode, thinking, english in (
    ("base", False, False),
    ("r", True, False),
    ("enr", True, True),
):
    generation = dict(
        temperature=1.0,
        max_tokens=32768,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=seed,
    )
    if mode == "base":
        generation.update(temperature=0.7, top_p=0.8, top_k=20)
    models.append(
        dict(
            type=VLLMwithChatTemplate,
            abbr=f"pulse_{mode}_v20_1_35b_a3b_group32_fast",
            path=model_path,
            model_kwargs=dict(
                tensor_parallel_size=2,
                gpu_memory_utilization=0.8,
                dtype="bfloat16",
                kv_cache_dtype="fp8_e4m3",
                language_model_only=True,
                trust_remote_code=True,
                max_num_seqs=256,
                max_num_batched_tokens=8192,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                seed=seed,
            ),
            max_seq_len=262144,
            max_out_len=32768,
            batch_size=8192,
            per_request_seed=True,
            generation_kwargs=generation,
            apply_chat_template_kwargs=dict(
                enable_thinking=thinking, use_en_thinking=english
            ),
            run_cfg=dict(num_gpus=2),
            pred_postprocessor=(
                dict(
                    type=extract_non_reasoning_content,
                    think_start_token="<ggthink>",
                    think_end_token="</ggthink>",
                )
                if thinking
                else None
            ),
        )
    )
