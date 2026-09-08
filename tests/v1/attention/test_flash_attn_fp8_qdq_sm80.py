# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn_fp8_qdq_sm80 import (
    FlashAttentionFp8QdqSm80Backend,
    FlashAttentionKvFp8QdqSm80Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.fp8_qdq import scaled_fp8_e4m3_qdq


def test_backend_registration_and_sm80_guard() -> None:
    assert (
        AttentionBackendEnum.FLASH_ATTN_FP8_QDQ_SM80.get_class()
        is FlashAttentionFp8QdqSm80Backend
    )
    assert (
        AttentionBackendEnum.FLASH_ATTN_KV_FP8_QDQ_SM80.get_class()
        is FlashAttentionKvFp8QdqSm80Backend
    )
    assert FlashAttentionFp8QdqSm80Backend.supports_compute_capability(
        DeviceCapability(8, 0)
    )
    assert not FlashAttentionFp8QdqSm80Backend.supports_compute_capability(
        DeviceCapability(8, 9)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaled_fp8_e4m3_qdq_matches_qat_reference() -> None:
    x = torch.tensor(
        [[-16.0, -1.25, -0.03, 0.0], [0.02, 0.75, 3.5, 32.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    scale = torch.tensor([0.037109375], dtype=torch.float32, device="cuda")

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    reference = (
        (x / scale.reshape(1))
        .clamp(min=-fp8_max, max=fp8_max)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scale.reshape(1))
        .to(x.dtype)
    )

    actual = scaled_fp8_e4m3_qdq(x, scale)
    assert torch.equal(actual, reference)
