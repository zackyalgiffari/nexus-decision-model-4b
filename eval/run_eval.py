"""Evaluate a model on calib/test items and store raw letter logits.

Raw (uncalibrated) logits are stored per item in canonical option order, so temperature fitting
and metric computation (eval/report.py) never need another forward pass.

Reference label per item: the dataset gold label when available (public data), otherwise the
teacher's argmax (synthetic data). The synthetic reference is therefore model-generated.

Usage:
    uv run python -m eval.run_eval --name base  --split test --mode fast
    uv run python -m eval.run_eval --name tuned --adapters artifacts/adapters/full --split calib
    uv run python -m eval.run_eval --name tuned --adapters ... --split test --mode think --limit 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nexus_decision.runtime import NexusDecision
from nexus_decision.schema import parse_question

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "data"
OUT = ROOT / "artifacts" / "eval"


def to_canonical(rq_labels: list[str], labels: list[str], z: np.ndarray) -> list[float]:
    out = [0.0] * len(labels)
    for i, lbl in enumerate(rq_labels):
        out[labels.index(lbl)] = float(z[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. base / tuned")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--split", default="test", choices=["calib", "test"])
    ap.add_argument("--mode", default="fast", choices=["fast", "think"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-reverse", action="store_true", help="skip the reversed-order pass")
    ap.add_argument("--max-think-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(x) for x in (DATA / f"{args.split}.jsonl").read_text().splitlines()]
    if args.limit:
        rng = np.random.default_rng(args.seed)
        items = [items[i] for i in sorted(rng.choice(len(items), min(args.limit, len(items)), replace=False))]

    if args.backend == "mlx":
        from nexus_decision.runtime_mlx import MLXBackend

        backend = MLXBackend(args.model, adapter_path=args.adapters)
    else:
        from nexus_decision.runtime_hf import HFBackend

        backend = HFBackend(args.model)
    nd = NexusDecision(backend)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}_{args.split}_{args.mode}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(x)["key"] for x in path.read_text().splitlines()}
    reverse = not args.no_reverse and args.mode == "fast"

    todo = [it for it in items if f"{it['sid']}/{it['question']}" not in done]
    print(f"{path.name}: {len(items) - len(todo)} done, {len(todo)} to go", flush=True)
    t_start = time.perf_counter()
    with path.open("a") as f:
        for i, it in enumerate(todo, 1):
            key = f"{it['sid']}/{it['question']}"
            q = parse_question(it["q"])
            labels = it["labels"]
            n = len(labels)
            ref = it["gold"] if it["gold"] is not None else labels[int(np.argmax(it["teacher"]))]
            t0 = time.perf_counter()
            rq, z, reasoning = nd.question_logits(
                it["state"], it["question"], q, list(range(n)), args.mode,
                max_think_tokens=args.max_think_tokens, seed=args.seed,
            )
            dt = time.perf_counter() - t0
            rec = {
                "key": key, "source": it["source"], "domain": it["domain"], "labels": labels,
                "ref": ref, "gold": it["gold"], "teacher": it["teacher"],
                "logits": to_canonical(rq.labels, labels, z), "latency_s": dt,
            }
            if reasoning is not None:
                rec["reasoning_chars"] = len(reasoning)
            if reverse:
                rq2, z2, _ = nd.question_logits(it["state"], it["question"], q, list(range(n))[::-1], "fast")
                rec["logits_reversed"] = to_canonical(rq2.labels, labels, z2)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                print(f"  {i}/{len(todo)}  {el / i:.2f}s/item  eta {(len(todo) - i) * el / i / 60:.1f} min", flush=True)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
