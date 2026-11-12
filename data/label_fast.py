"""Soft-label every question with the teacher in non-thinking mode.

For each question the exact student prompt is sent with ``max_tokens=1`` and ``top_logprobs=20``.
The first-token distribution is mapped onto option letters, then back to option labels. This is
repeated for several option orders (canonical, reversed, then seeded random) and averaged, which
removes most of the teacher's own position bias.

Output: artifacts/data/labels_fast.jsonl
    {"sid", "question", "labels": [canonical], "probs": [canonical], "per_order": [[...], ...],
     "order_agree": bool}

Usage:
    uv run python -m data.label_fast --limit 100          # pilot
    uv run python -m data.label_fast --orders 3 --offpeak-only
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

from data.teacher import ROOT, Teacher, letter_distribution, run_concurrent
from nexus_decision.prompt import build_messages, render_question
from nexus_decision.schema import parse_question

DATA = ROOT / "artifacts" / "data"
OUT = DATA / "labels_fast.jsonl"


def load_scenarios() -> list[dict]:
    out = []
    for name in ("public.jsonl", "scenarios.jsonl"):
        p = DATA / name
        if p.exists():
            out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def question_items(scenarios: list[dict]) -> list[tuple[dict, str, dict]]:
    return [(s, qn, q) for s in scenarios for qn, q in s["questions"].items()]


def orders_for(n: int, k: int, key: str) -> list[list[int]]:
    base = [list(range(n)), list(range(n))[::-1]]
    rng = random.Random(key)
    while len(base) < k:
        o = list(range(n))
        rng.shuffle(o)
        base.append(o)
    return base[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", default="all", choices=["all", "synthetic", "public"])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--offpeak-only", action="store_true")
    args = ap.parse_args()

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sid"], r["question"]))
    items = [it for it in question_items(load_scenarios()) if (it[0]["id"], it[1]) not in done
             and args.source in ("all", "public" if it[0]["source"] != "synthetic" else "synthetic")]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} questions to label ({len(done)} already done)")

    teacher = Teacher(budget_usd=args.budget_usd, offpeak_only=args.offpeak_only)

    def call(item):
        scn, qname, qraw = item
        q = parse_question(qraw)
        labels = [lbl for lbl, _ in q.option_items()]
        n = len(labels)
        per_order = []
        for order in orders_for(n, args.orders, f"{scn['id']}/{qname}"):
            rq = render_question(scn["state"], qname, q, order)
            r = teacher.chat(
                build_messages(rq.user_text), step="label_fast", thinking=False, max_tokens=1,
                temperature=1.0, logprobs=True, top_logprobs=20,
            )
            if not r.top_logprobs:
                return None
            p_disp = letter_distribution(r.top_logprobs[0], n)
            p_canon = [0.0] * n
            for i, lbl in enumerate(rq.labels):
                p_canon[labels.index(lbl)] = p_disp[i]
            per_order.append(p_canon)
        probs = np.mean(per_order, axis=0)
        argmaxes = {int(np.argmax(p)) for p in per_order}
        return {
            "sid": scn["id"], "question": qname, "labels": labels,
            "probs": [round(float(x), 6) for x in probs / probs.sum()],
            "per_order": [[round(x, 6) for x in p] for p in per_order],
            "order_agree": len(argmaxes) == 1,
        }

    results = run_concurrent(call, items, workers=args.workers, desc="label_fast")
    ok = [r for r in results if r]
    with OUT.open("a") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agree = np.mean([r["order_agree"] for r in ok]) if ok else float("nan")
    print(f"labeled {len(ok)}/{len(items)}; order agreement {agree:.3f}")
    print(f"spent this step: ${teacher.spent('label_fast'):.4f}  total: ${teacher.spent():.4f}")


if __name__ == "__main__":
    main()
