# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, prewarmed CUDA Graphs for short SM89 GDN prefills."""

import threading
import weakref
from typing import Any

import torch

from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

_TOKEN_KEYS = ("q", "k", "v", "g", "beta")
_TENSOR_KEYS = (
    *_TOKEN_KEYS,
    "initial_state",
    "cu_seqlens",
    "chunk_indices",
    "chunk_offsets",
)
_SCALAR_KEYS = ("scale", "output_final_state", "use_qk_l2norm_in_kernel")
_MAX_TOKENS = 512
_CHUNK_SIZE = 64


def _signature(arguments: dict[str, Any]) -> tuple:
    total = arguments["q"].shape[1]
    padded = (total + _CHUNK_SIZE - 1) // _CHUNK_SIZE * _CHUNK_SIZE
    shapes = []
    for key in _TENSOR_KEYS:
        value = arguments[key]
        shape = list(value.shape)
        if key in _TOKEN_KEYS:
            shape[1] = padded
        shapes.append((key, tuple(shape), value.dtype, value.device))
    return tuple(shapes), tuple(arguments[key] for key in _SCALAR_KEYS)


class _GraphEntry:
    def __init__(self, arguments: dict[str, Any]):
        validate_cudagraph_capturing_enabled()
        self.device = arguments["q"].device
        self.static = {
            key: torch.empty_like(arguments[key], memory_format=torch.contiguous_format)
            for key in _TENSOR_KEYS
        }
        self.static.update({key: arguments[key] for key in _SCALAR_KEYS})
        torch._foreach_copy_(
            [self.static[key] for key in _TENSOR_KEYS],
            [arguments[key] for key in _TENSOR_KEYS],
        )
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                chunk_gated_delta_rule(**self.static)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.accelerator.synchronize(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = chunk_gated_delta_rule(**self.static)
        segments = torch.cuda.memory_snapshot(self.graph.pool(), include_traces=False)
        assert segments and all(
            tuple(segment["segment_pool_id"]) == tuple(self.graph.pool())
            for segment in segments
        )
        self.charged_bytes = sum(segment["total_size"] for segment in segments) + sum(
            self.static[key].untyped_storage().nbytes() for key in _TENSOR_KEYS
        )
        self.runtime_stream: int | None = None

    def replay(self, arguments: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        total = arguments["q"].shape[1]
        destinations = [
            self.static[key][:, :total] if key in _TOKEN_KEYS else self.static[key]
            for key in _TENSOR_KEYS
        ]
        torch._foreach_copy_(destinations, [arguments[key] for key in _TENSOR_KEYS])
        self.graph.replay()
        output, final_state = self.outputs
        # Later layers and requests reuse these graphs. Returned tensors must
        # remain valid after the next replay, including its state-cache update.
        return output[:, :total].clone(), final_state.clone()


class ShortPrefillGraphCache:
    def __init__(self, owner: Any, max_bytes: int = 256 * 1024**2):
        self.owner = weakref.ref(owner)
        self.max_bytes = max_bytes
        self.charged_bytes = 0
        self.entries: dict[tuple, _GraphEntry] = {}
        self._warmups: set[tuple] = set()
        self._lock = threading.RLock()

    def warmup(self, device: torch.device, state_dtype: torch.dtype) -> None:
        """Capture the eight single-sequence buckets before KV allocation."""
        if (
            device.type != "cuda"
            or state_dtype not in (torch.bfloat16, torch.float32)
            or torch.is_grad_enabled()
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.get_device_capability(device) != (8, 9)
        ):
            return
        scope = (device, state_dtype)
        with self._lock:
            if scope in self._warmups:
                return
            self._warmups.add(scope)
            for total in (128, 512, 64, 256, 192, 320, 384, 448):
                free, _ = torch.accelerator.get_memory_info(device)
                if free < 128 * 1024**2 or self.charged_bytes >= self.max_bytes:
                    break
                chunks = total // _CHUNK_SIZE
                arguments = {
                    key: torch.zeros(
                        (1, total, heads, 128), device=device, dtype=torch.bfloat16
                    )
                    for key, heads in (("q", 8), ("k", 8), ("v", 16))
                }
                arguments.update(
                    g=torch.zeros((1, total, 16), device=device, dtype=torch.float32),
                    beta=torch.zeros(
                        (1, total, 16), device=device, dtype=torch.float32
                    ),
                    initial_state=torch.zeros(
                        (1, 16, 128, 128), device=device, dtype=state_dtype
                    ),
                    cu_seqlens=torch.tensor(
                        [0, total], device=device, dtype=torch.int32
                    ),
                    chunk_indices=torch.tensor(
                        [[0, i] for i in range(chunks)],
                        device=device,
                        dtype=torch.int32,
                    ),
                    chunk_offsets=torch.tensor(
                        [0, chunks], device=device, dtype=torch.int64
                    ),
                    scale=128**-0.5,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                )
                entry = _GraphEntry(arguments)
                if self.charged_bytes + entry.charged_bytes > self.max_bytes:
                    del entry
                    break
                self.entries[_signature(arguments)] = entry
                self.charged_bytes += entry.charged_bytes

    def __call__(self, **arguments):
        q = arguments["q"]
        state = arguments["initial_state"]
        eligible = (
            q.is_cuda
            and not torch.is_grad_enabled()
            and not torch.cuda.is_current_stream_capturing()
            and q.dtype == torch.bfloat16
            and q.shape[0] == 1
            and 0 < q.shape[1] <= _MAX_TOKENS
            and state is not None
            and state.shape == (1, 16, 128, 128)
            and arguments.get("core_attn_out") is None
            and arguments["output_final_state"]
            and not arguments["use_qk_l2norm_in_kernel"]
            and all(
                isinstance(arguments.get(key), torch.Tensor) for key in _TENSOR_KEYS
            )
        )
        if eligible:
            if arguments.get("scale") is None:
                arguments["scale"] = arguments["k"].shape[-1] ** -0.5
            with self._lock:
                entry = self.entries.get(_signature(arguments))
                if entry is not None:
                    stream = torch.cuda.current_stream(q.device).cuda_stream
                    if entry.runtime_stream is None:
                        entry.runtime_stream = stream
                    if entry.runtime_stream == stream:
                        return entry.replay(arguments)
        return chunk_gated_delta_rule(**arguments)


_CACHES: weakref.WeakValueDictionary[int, ShortPrefillGraphCache] = (
    weakref.WeakValueDictionary()
)
_CACHES_LOCK = threading.Lock()


def get_short_prefill_graph_cache(owner: Any) -> ShortPrefillGraphCache:
    with _CACHES_LOCK:
        cache = _CACHES.get(id(owner))
        if cache is None or cache.owner() is not owner:
            cache = ShortPrefillGraphCache(owner)
            _CACHES[id(owner)] = cache
        return cache
