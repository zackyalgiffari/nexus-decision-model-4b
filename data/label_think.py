"""Collect System Two reasoning traces from the teacher (thinking mode).

A stratified subset of questions is rendered with a seeded random option order and sent in
thinking mode. We keep the ``reasoning_content`` and the final answer letter. Filtering against
the fast soft labels happens later in build_splits.py.

Output: artifacts/data/labels_think.jsonl
    {"sid", "question", "order": [...], "labels_display": [...], "reasoning", "letter", "label"}

Usage:
    uv run python -m data.label_think --n 20                 # pilot
    uv run python -m data.label_think --n 4000 --offpeak-only
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict

from data.label_fast import load_scenarios, question_items
from data.teacher import ROOT, Teacher, run_concurrent
from nexus_decision.prompt import build_messages, render_question
from nexus_decision.schema import LETTERS, parse_question

OUT = ROOT / "artifacts" / "data" / "labels_think.jsonl"


def parse_letter(content: str, n: int) -> str | None:
    m = re.search(r"\b([A-Z])\b", content.strip()[:20])
    if m and m.group(1) in LETTERS[:n]:
        return m.group(1)
    return None


def stratified(items, n: int, seed: int):
    """Round-robin over domains so every domain is represented. Test splits are excluded."""
    rng = random.Random(seed)
    by_dom = defaultdict(list)
    for it in items:
        scn = it[0]
        if scn.get("split") == "test":
            continue
        by_dom[scn["domain"]].append(it)
    for v in by_dom.values():
        rng.shuffle(v)
    out, doms = [], sorted(by_dom)
    while len(out) < n and any(by_dom.values()):
        for d in doms:
            if by_dom[d] and len(out) < n:
                out.append(by_dom[d].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--effort", default="low", choices=["low", "high", "max"])
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--source", default="all", choices=["all", "synthetic", "public"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--offpeak-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    done = set()
    if OUT.exists():
        done = {(json.loads(x)["sid"], json.loads(x)["question"]) for x in OUT.read_text().splitlines()}
    pool = [it for it in question_items(load_scenarios()) if (it[0]["id"], it[1]) not in done
            and args.source in ("all", "public" if it[0]["source"] != "synthetic" else "synthetic")]
    items = stratified(pool, args.n, args.seed)
    print(f"{len(items)} questions for think traces ({len(done)} already done)")

    teacher = Teacher(budget_usd=args.budget_usd, offpeak_only=args.offpeak_only)

    def call(item):
        scn, qname, qraw = item
        q = parse_question(qraw)
        n = len(q.option_items())
        order = list(range(n))
        random.Random(f"think/{scn['id']}/{qname}").shuffle(order)
        rq = render_question(scn["state"], qname, q, order)
        r = teacher.chat(
            build_messages(rq.user_text), step="label_think", thinking=True,
            max_tokens=args.max_tokens, reasoning_effort=args.effort,
        )
        letter = parse_letter(r.content, n)
        if not letter or not r.reasoning:
            return None
        return {
            "sid": scn["id"], "question": qname, "order": order, "labels_display": rq.labels,
            "reasoning": r.reasoning.strip(), "letter": letter,
            "label": rq.labels[LETTERS.index(letter)],
        }

    results = run_concurrent(call, items, workers=args.workers, desc="label_think")
    ok = [r for r in results if r]
    with OUT.open("a") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if ok:
        avg = sum(len(r["reasoning"]) for r in ok) / len(ok)
        print(f"kept {len(ok)}/{len(items)} traces; avg reasoning {avg:.0f} chars")
    print(f"spent this step: ${teacher.spent('label_think'):.4f}  total: ${teacher.spent():.4f}")


if __name__ == "__main__":
    main()
