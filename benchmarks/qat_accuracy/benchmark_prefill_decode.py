#!/usr/bin/env python3
"""Measure vLLM prefill and decode separately through the OpenAI API."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def stream_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, float | int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content_at = None
    last_content_at = None
    prompt_tokens = 0
    completion_tokens = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(
                usage.get("completion_tokens") or completion_tokens
            )
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                now = time.perf_counter()
                if first_content_at is None:
                    first_content_at = now
                last_content_at = now
    finished = time.perf_counter()
    if first_content_at is None:
        first_content_at = finished
    if last_content_at is None:
        last_content_at = first_content_at
    decode_time = max(0.0, last_content_at - first_content_at)
    tpot = (
        decode_time / (completion_tokens - 1)
        if completion_tokens > 1
        else math.nan
    )
    return {
        "latency_s": finished - started,
        "ttft_s": first_content_at - started,
        "decode_s": decode_time,
        "tpot_s": tpot,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_batch(
    *,
    phase: str,
    base_url: str,
    model: str,
    concurrency: int,
    prompt_repeat: int,
    max_tokens: int,
    timeout: int,
    round_index: int,
) -> dict[str, float | int | str]:
    stem = "请逐条分析临床表现、诊断依据、鉴别诊断和治疗原则。"
    prompts = [
        stem * prompt_repeat + f"\n独立样本编号：{i}-{round_index}"
        for i in range(concurrency)
    ]
    started = time.perf_counter()
    rows = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                stream_one, base_url, model, prompt, max_tokens, timeout
            )
            for prompt in prompts
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report all client failures
                errors.append(repr(exc))
    wall_s = time.perf_counter() - started
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    ttfts = [float(row["ttft_s"]) for row in rows]
    tpots = [
        float(row["tpot_s"])
        for row in rows
        if not math.isnan(float(row["tpot_s"]))
    ]
    result: dict[str, float | int | str] = {
        "phase": phase,
        "round": round_index,
        "concurrency": concurrency,
        "wall_s": round(wall_s, 4),
        "requests_ok": len(rows),
        "errors": len(errors),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_tokens_per_s": round(prompt_tokens / wall_s, 1),
        "output_tokens_per_s": round(completion_tokens / wall_s, 1),
        "ttft_p50_s": round(statistics.median(ttfts), 4) if ttfts else math.nan,
        "ttft_p95_s": round(percentile(ttfts, 0.95), 4),
        "tpot_p50_ms": round(statistics.median(tpots) * 1000, 3)
        if tpots
        else math.nan,
        "tpot_p95_ms": round(percentile(tpots, 0.95) * 1000, 3)
        if tpots
        else math.nan,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if errors:
        print(json.dumps({"sample_errors": errors[:3]}, ensure_ascii=False), flush=True)
    return result


def summarize(phase: str, rows: list[dict[str, float | int | str]]) -> None:
    key = "input_tokens_per_s" if phase == "prefill" else "output_tokens_per_s"
    values = [float(row[key]) for row in rows]
    print(
        json.dumps(
            {
                "phase": phase,
                "rounds": len(rows),
                f"{key}_median": round(statistics.median(values), 1),
                f"{key}_min": round(min(values), 1),
                f"{key}_max": round(max(values), 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--phase", choices=("prefill", "decode", "both"), default="both"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefill-concurrency", type=int, default=8)
    parser.add_argument("--prefill-prompt-repeat", type=int, default=2048)
    parser.add_argument("--decode-concurrency", type=int, default=64)
    parser.add_argument("--decode-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    # Warm up scheduler, HTTP streaming, and Triton JIT outside measurements.
    run_batch(
        phase="warmup",
        base_url=args.base_url,
        model=args.model,
        concurrency=2,
        prompt_repeat=1,
        max_tokens=8,
        timeout=args.timeout,
        round_index=0,
    )

    if args.phase in ("prefill", "both"):
        rows = [
            run_batch(
                phase="prefill",
                base_url=args.base_url,
                model=args.model,
                concurrency=args.prefill_concurrency,
                prompt_repeat=args.prefill_prompt_repeat,
                max_tokens=1,
                timeout=args.timeout,
                round_index=index + 1,
            )
            for index in range(args.rounds)
        ]
        summarize("prefill", rows)

    if args.phase in ("decode", "both"):
        rows = [
            run_batch(
                phase="decode",
                base_url=args.base_url,
                model=args.model,
                concurrency=args.decode_concurrency,
                prompt_repeat=1,
                max_tokens=args.decode_max_tokens,
                timeout=args.timeout,
                round_index=index + 1,
            )
            for index in range(args.rounds)
        ]
        summarize("decode", rows)


if __name__ == "__main__":
    main()
