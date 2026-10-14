# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recognize the validated Qwen3.5 MoE group32 training export contract."""

from typing import Any

MOE_TARGETS = {r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"}
DENSE_TARGETS = {
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
    r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    r"re:.*mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$",
    r"re:.*mtp\.fc$",
}
IGNORE = {
    r"re:.*lm_head.*",
    r"re:.*visual.*",
    r"re:.*shared_expert_gate.*",
    r"re:.*mlp\.gate$",
}


def _matches_args(
    args: Any,
    bits: int,
    kind: str,
    strategy: str,
    group_size: int | None,
    dynamic: bool,
) -> bool:
    return isinstance(args, dict) and all(
        args.get(k) == v
        for k, v in {
            "num_bits": bits,
            "type": kind,
            "strategy": strategy,
            "group_size": group_size,
            "dynamic": dynamic,
            "symmetric": True,
            "actorder": None,
            "block_structure": None,
            "zp_dtype": None,
        }.items()
    )


def _matches_export(hf: dict[str, Any]) -> bool:
    """Only the validated export layout opts in; explicit choices always win."""
    if not isinstance(hf, dict) or hf.get("model_type") != "qwen3_5_moe":
        return False
    text = hf.get("text_config", {})
    if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_moe_text":
        return False
    if text.get("hidden_act") != "silu" or text.get("head_dim") != 256:
        return False
    positive = (
        "hidden_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
    )
    if any(type(text.get(k)) is not int or text[k] <= 0 for k in positive):
        return False
    if (
        text["hidden_size"] % 128
        or text["moe_intermediate_size"] % 128
        or text["shared_expert_intermediate_size"] % 32
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or text["num_experts_per_tok"] > text["num_experts"]
    ):
        return False
    layers = text.get("layer_types", [])
    if (
        not isinstance(layers, list)
        or len(layers) != text["num_hidden_layers"]
        or set(layers) != {"linear_attention", "full_attention"}
        or text.get("mlp_only_layers", [])
    ):
        return False
    cfg = hf.get("quantization_config", {})
    if not isinstance(cfg, dict) or not all(
        cfg.get(k) == v
        for k, v in {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "quantization_status": "compressed",
        }.items()
    ):
        return False
    if cfg.get("transform_config") or cfg.get("sparsity_config"):
        return False
    ignore = cfg.get("ignore", [])
    if (
        not isinstance(ignore, list)
        or not all(isinstance(x, str) for x in ignore)
        or set(ignore) != IGNORE
    ):
        return False
    if not _matches_args(cfg.get("kv_cache_scheme"), 8, "float", "tensor", None, False):
        return False
    groups = cfg.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 2:
        return False
    found = set()
    for group in groups.values():
        if not isinstance(group, dict):
            return False
        targets = group.get("targets")
        if not isinstance(targets, list) or not all(
            isinstance(x, str) for x in targets
        ):
            return False
        targets = set(targets)
        if targets == MOE_TARGETS:
            role, bits, kind, fmt = "moe", 4, "int", "pack-quantized"
        elif targets == DENSE_TARGETS:
            role, bits, kind, fmt = "dense", 8, "float", "float-quantized"
        else:
            return False
        if role in found:
            return False
        found.add(role)
        if (
            group.get("format") != fmt
            or group.get("output_activations") is not None
            or not _matches_args(group.get("weights"), bits, kind, "group", 32, False)
            or not _matches_args(
                group.get("input_activations"), 8, "float", "group", 32, True
            )
        ):
            return False
        if role == "dense" and group["weights"].get("scale_dtype") != "torch.float32":
            return False
        if role == "moe" and group["weights"].get("scale_dtype") not in (
            None,
            "torch.bfloat16",
        ):
            return False
        if group["input_activations"].get("scale_dtype") not in (None, "torch.float32"):
            return False
    return found == {"moe", "dense"}


def is_group32_export(hf_quant_cfg: dict[str, Any], hf_config: Any) -> bool:
    if hf_config is None:
        return False
    config = hf_config.to_dict()
    config["quantization_config"] = hf_quant_cfg
    return _matches_export(config)


def has_static_fp8_qkv(scheme: Any) -> bool:
    return _matches_args(scheme, 8, "float", "tensor", None, False)
