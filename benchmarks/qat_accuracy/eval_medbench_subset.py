# SPDX-License-Identifier: Apache-2.0
"""Small MedBench evaluator for QAT accuracy experiments.

The script reuses prompts and references saved in an OpenCompass prediction
JSON, but has no OpenCompass dependency. It supports the three representative
sets used while debugging the A100 QAT path: CHIP-CDEE, CMeEE, and MedSafety.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_PREDICTIONS_ROOT = Path(
    "/mnt/nas/home/genggui/code/opencompass_src/outputs/medbench_chat_vllm/"
    "chat/20250514_093817/predictions/pulse_enr_v20_35b_a3b_int4_fp8"
)


@dataclass(frozen=True)
class Profile:
    temperature: float
    top_p: float
    top_k: int | None
    chat_template_kwargs: dict[str, Any]
    strip_reasoning: bool


PROFILES = {
    "enr": Profile(
        1.0, 0.95, None, {"enable_thinking": True, "use_en_thinking": True}, True
    ),
    "r": Profile(1.0, 0.95, None, {"enable_thinking": True}, True),
    "base": Profile(0.7, 0.8, 20, {"enable_thinking": False}, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="pulse-v20-marlin-fp8-qdq")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument(
        "--dataset",
        choices=("CHIP-CDEE", "CMeEE", "MedSafety"),
        action="append",
        dest="datasets",
        help="May be repeated; defaults to all supported datasets.",
    )
    parser.add_argument(
        "--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "Override the selected profile temperature. This is intended for "
            "deterministic kernel diagnostics; omitted preserves profile behavior."
        ),
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be non-negative")
    return args


def ordered_items(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def sort_key(key: str) -> tuple[int, int | str]:
        return (0, int(key)) if key.isdigit() else (1, key)

    return sorted(data.items(), key=lambda item: sort_key(item[0]))


def load_cases(root: Path, dataset: str, limit: int | None) -> list[dict[str, Any]]:
    path = root / f"MedBench_v3-{dataset}.json"
    with path.open(encoding="utf-8") as file:
        source = json.load(file)
    if not isinstance(source, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, got {type(source).__name__}"
        )

    cases = []
    for key, item in ordered_items(source):
        prompts = item["origin_prompt"]
        messages = []
        for prompt in prompts:
            role = str(prompt["role"]).upper()
            role = {"HUMAN": "user", "BOT": "assistant"}.get(role, role.lower())
            messages.append({"role": role, "content": prompt["prompt"]})
        cases.append(
            {
                "case_id": key,
                "dataset": dataset,
                "messages": messages,
                "gold": item["gold"],
            }
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def request_case(
    case: dict[str, Any],
    index: int,
    args: argparse.Namespace,
    profile: Profile,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": case["messages"],
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": args.max_tokens,
        "seed": args.seed + index,
        "chat_template_kwargs": profile.chat_template_kwargs,
    }
    if profile.top_k is not None:
        payload["top_k"] = profile.top_k

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                parsed = json.load(response)
            message = parsed["choices"][0]["message"]
            text = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if reasoning and not text:
                text = reasoning
            return {
                **case,
                "prediction": text,
                "reasoning_content": reasoning,
                "elapsed_seconds": time.perf_counter() - started,
                "usage": parsed.get("usage", {}),
                "error": None,
            }
        except (OSError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, urllib.error.HTTPError):
                try:
                    last_error = f"{error}: {error.read().decode('utf-8', 'replace')}"
                except OSError:
                    last_error = repr(error)
            else:
                last_error = repr(error)
            if attempt < args.retries:
                time.sleep(2**attempt)
    return {
        **case,
        "prediction": "",
        "reasoning_content": "",
        "elapsed_seconds": time.perf_counter() - started,
        "usage": {},
        "error": last_error,
    }


def strip_reasoning(text: str) -> str:
    start_token = "<ggthink>"
    end_token = "</ggthink>"
    if start_token not in text and end_token in text:
        return text.split(end_token)[-1].strip()
    return re.sub(
        rf"{re.escape(start_token)}.*?{re.escape(end_token)}",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def parse_json_prediction(text: Any) -> Any:
    if isinstance(text, (dict, list)) or text is None:
        return text
    start_idx, end_idx = text.rfind("```json"), text.rfind("```")
    if start_idx == -1 or end_idx == -1:
        matches = re.findall(r"[\[{][\s\S]+[}\]]", text)
        if matches:
            text = matches[0]
    else:
        text = text[start_idx + 7 : end_idx]
    text = text.replace("\n", "").replace(" ", "").replace(",]", "]").replace(",}", "}")
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}


def to_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, to_hashable(value[key])) for key in sorted(value))
    if isinstance(value, list):
        converted = [to_hashable(item) for item in value]
        try:
            return tuple(sorted(converted))
        except TypeError:
            return tuple(converted)
    return value if isinstance(value, str) else str(value)


def prf_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 100.0 * (2 * precision * recall / (precision + recall + 1e-6))


def score_chip_cdee(
    results: list[dict[str, Any]],
) -> tuple[float, list[dict[str, int]]]:
    tp = fp = fn = 0
    details = []
    for result in results:
        pred = parse_json_prediction(result["processed_prediction"])
        answer = json.loads(result["gold"]["answer"])
        if not isinstance(pred, (list, dict)):
            pred = [pred]
        pred_set = set(to_hashable(pred))
        answer_set = set(to_hashable(answer))
        current = {
            "tp": len(pred_set & answer_set),
            "fp": len(pred_set - answer_set),
            "fn": len(answer_set - pred_set),
        }
        tp += current["tp"]
        fp += current["fp"]
        fn += current["fn"]
        details.append(current)
    return prf_score(tp, fp, fn), details


def score_cmeee(results: list[dict[str, Any]]) -> tuple[float, list[dict[str, int]]]:
    tp = fp = fn = 0
    details = []
    for result in results:
        pred = parse_json_prediction(result["processed_prediction"])
        answer = json.loads(result["gold"]["answer"])
        if not isinstance(pred, dict):
            pred = {"unknown": pred}
        if not isinstance(answer, dict):
            answer = {"unknown": answer}
        current = {"tp": 0, "fp": 0, "fn": 0}
        for key in set(pred) | set(answer):
            pred_set = set(to_hashable(pred.get(key, [])))
            answer_set = set(to_hashable(answer.get(key, [])))
            current["tp"] += len(pred_set & answer_set)
            current["fp"] += len(pred_set - answer_set)
            current["fn"] += len(answer_set - pred_set)
        tp += current["tp"]
        fp += current["fp"]
        fn += current["fn"]
        details.append(current)
    return prf_score(tp, fp, fn), details


def extract_choice(text: str) -> list[str]:
    matches = re.findall(r"<\s*([A-E])\s*>", text.upper())
    if matches:
        return [matches[-1]]
    matches = re.findall(r"(?:答案|答|ANSWER)\s*[:：]?\s*([A-E])\b", text.upper())
    return [matches[-1]] if matches else []


def score_medsafety(
    results: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    by_qid: dict[Any, list[bool]] = defaultdict(list)
    details = []
    for result in results:
        pred = extract_choice(result["processed_prediction"])
        answer_match = re.search(r"([A-Z]+)", result["gold"]["answer"])
        answer = [answer_match.group(1)] if answer_match else []
        correct = pred == answer
        qid = result["gold"].get("qid", result["case_id"])
        by_qid[qid].append(correct)
        details.append({"pred": pred, "answer": answer, "correct": correct})
    correct_qids = sum(all(values) for values in by_qid.values())
    return 100.0 * correct_qids / (len(by_qid) + 1e-6), details


def corruption_stats(
    results: list[dict[str, Any]], profile_name: str
) -> dict[str, int]:
    stats = defaultdict(int)
    for result in results:
        text = result["prediction"]
        stats["errors"] += result["error"] is not None
        stats["empty"] += not text.strip()
        stats["replacement_chars"] += text.count("\ufffd")
        stats["nul_chars"] += text.count("\x00")
        end_count = text.count("</ggthink>")
        stats["think_end_total"] += end_count
        if profile_name in ("enr", "r"):
            stats["think_end_not_exactly_one"] += end_count != 1
    return dict(stats)


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    if args.temperature is not None:
        profile = replace(profile, temperature=args.temperature)
    datasets = args.datasets or ["CHIP-CDEE", "CMeEE", "MedSafety"]
    cases = []
    for dataset in datasets:
        cases.extend(load_cases(args.predictions_root, dataset, args.limit))

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(request_case, case, index, args, profile)
            for index, case in enumerate(cases)
        ]
        results = []
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            result["processed_prediction"] = (
                strip_reasoning(result["prediction"])
                if profile.strip_reasoning
                else result["prediction"]
            )
            results.append(result)
            print(f"completed {completed}/{len(futures)}", flush=True)
    results.sort(
        key=lambda item: (datasets.index(item["dataset"]), int(item["case_id"]))
    )

    scorers = {
        "CHIP-CDEE": score_chip_cdee,
        "CMeEE": score_cmeee,
        "MedSafety": score_medsafety,
    }
    scores = {}
    for dataset in datasets:
        subset = [result for result in results if result["dataset"] == dataset]
        score, details = scorers[dataset](subset)
        scores[dataset] = {"score": score, "num_samples": len(subset)}
        for result, detail in zip(subset, details):
            result["score_detail"] = detail

    elapsed = time.perf_counter() - started
    total_output_tokens = sum(
        result["usage"].get("completion_tokens", 0) for result in results
    )
    report = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "profile": args.profile,
            "datasets": datasets,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
        },
        "scores": scores,
        "corruption": corruption_stats(results, args.profile),
        "timing": {
            "elapsed_seconds": elapsed,
            "requests_per_second": len(results) / elapsed,
            "output_tokens_per_second": total_output_tokens / elapsed,
            "max_request_seconds": max(
                (item["elapsed_seconds"] for item in results), default=math.nan
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scores": scores,
                "corruption": report["corruption"],
                "timing": report["timing"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
