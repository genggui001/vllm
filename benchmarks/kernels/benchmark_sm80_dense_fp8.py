# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident SM80 Dense G32 core vs native W8A16 Marlin G128 and BF16.

All activation conversion and weight packing are outside timed regions.
G128 synthetic weight scales are repeated four times for the G32 candidate;
this is a same-shape speed control, not a replacement for G32 accuracy tests.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mf
from vllm.model_executor.layers.quantization.utils import sm80_dense_fp8 as dense
from vllm.scalar_type import scalar_types

SHAPES = (
    [("qkv", m, 6144, 2048) for m in (1, 4, 8, 2048, 8192)]
    + [("out", m, 2048, 2048) for m in (1, 4, 8, 2048, 8192)]
    + [("gate", m, 512, 2048) for m in (2048, 8192)]
    + [("down", m, 2048, 256) for m in (2048, 8192)]
    + [("fc", m, 1024, 4096) for m in (1024, 2048, 8192)]
    + [("ba", m, 32, 2048) for m in (512, 1024, 2048, 4096)]
)


def dq(q, s):
    return (q.float().reshape(*s.shape, 32) * s[..., None]).reshape(q.shape).bfloat16()


def stats(y, ref):
    d = y.float() - ref.float()
    rmse = d.square().mean().sqrt().item() if d.numel() else 0.0
    rms = ref.float().square().mean().sqrt().item() if d.numel() else 0.0
    return dict(
        rmse=rmse,
        nrmse=rmse / max(rms, 1e-30),
        max_abs=d.abs().max().item() if d.numel() else 0.0,
        nonfinite=int((~torch.isfinite(y)).sum()),
        bit_mismatch=int((y.view(torch.int16) != ref.view(torch.int16)).sum()),
    )


def describe(samples):
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "p10_us": ordered[int(0.1 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0,
    }


def measurer(fn, m, mode, robust=False):
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    calls = (512 if m < 512 else 16) if robust else (32 if m < 512 else 4)
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        nodes = (128 if m < 512 else 4) if robust else (8 if m < 512 else 2)
        with torch.cuda.graph(graph):
            for _ in range(nodes):
                fn()
        repeats = calls // nodes

        def run():
            for _ in range(repeats):
                graph.replay()
    else:

        def run():
            for _ in range(calls):
                fn()

    def measure():
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin = time.perf_counter()
        start.record()
        run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / calls, (
            time.perf_counter() - begin
        ) * 1e6 / calls

    if robust:
        until = time.perf_counter() + 0.2
        while time.perf_counter() < until:
            measure()
    return measure


def paired(baseline, candidate, m, mode, rounds, robust=False):
    a = measurer(baseline, m, mode, robust)
    b = measurer(candidate, m, mode, robust)
    raw_a, raw_b, wall_a, wall_b, chronology = [], [], [], [], []
    for repeat in range(rounds):
        for label, fn, target, wall in [
            ("A", a, raw_a, wall_a),
            ("B", b, raw_b, wall_b),
            ("B", b, raw_b, wall_b),
            ("A", a, raw_a, wall_a),
        ]:
            gpu, host = fn()
            target.append(gpu)
            wall.append(host)
            chronology.append(
                {"repeat": repeat, "which": label, "gpu_us": gpu, "wall_us": host}
            )
    return {
        "baseline": describe(raw_a),
        "candidate": describe(raw_b),
        "baseline_wall": describe(wall_a),
        "candidate_wall": describe(wall_b),
        "speedup": statistics.median(raw_a) / statistics.median(raw_b),
        "chronology": chronology,
    }


def run_case(args, idx, case, m, n, k):
    gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
    a = (
        (torch.randn(m, k, device="cuda", generator=gen) * 32)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    w = (
        (torch.randn(n, k, device="cuda", generator=gen) * 32)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    sa = torch.rand(m, k // 32, device="cuda", generator=gen) * 0.02 + 0.002
    sw128 = torch.rand(n, k // 128, device="cuda", generator=gen) * 0.0015 + 0.0005
    sw = sw128.repeat_interleave(4, 1).contiguous()
    ab, wb = dq(a, sa), dq(w, sw)
    pw, ps, aux = dense.prepare_weight(w, sw)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sw128, requires_grad=False)
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [1, 128]
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    mf.prepare_fp8_layer_for_marlin(layer, size_k_first=False, input_dtype=None)
    pn, pk = mu.marlin_repacked_nk(layer.weight, num_bits=8)

    def native():
        return ops.marlin_gemm(
            ab,
            None,
            layer.weight,
            None,
            layer.weight_scale,
            None,
            None,
            None,
            None,
            None,
            layer.workspace,
            scalar_types.float8_e4m3fn,
            m,
            pn,
            pk,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[:, :n]

    def custom():
        return dense.dense_fp8(dense.Group32FP8Input(a, sa), pw, ps, aux, n)

    def bf16():
        return F.linear(ab, wb)

    ref = bf16()
    out = custom()
    nat = native()
    torch.accelerator.synchronize()
    error = stats(out, ref)
    assert error["nonfinite"] == 0
    row = dict(
        case=case,
        shape=[m, n, k],
        error=error,
        native_error=stats(nat, ref),
        timing={},
    )
    for mode in args.modes:
        row["timing"][mode] = dict(
            vs_native=paired(native, custom, m, mode, args.rounds, robust=True),
            vs_bf16=paired(bf16, custom, m, mode, args.rounds, robust=True),
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--modes", nargs="+", choices=("graph", "eager"), default=["graph", "eager"]
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing benchmark evidence")
    if args.rounds < 1 or torch.cuda.get_device_capability() != (8, 0):
        parser.error("Requires SM80 and at least one ABBA round")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, (case, m, n, k) in enumerate(SHAPES):
        row = run_case(args, idx, case, m, n, k)
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2))
        print(
            json.dumps({"case": case, "shape": [m, n, k], "error": row["error"]}),
            flush=True,
        )
        gc.collect()
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
