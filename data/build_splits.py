"""Join scenarios with teacher labels and write train / val / calibration / test files.

Split rules:
  * public data: the dataset's own test split -> test; its train split -> train (90%) / calib (10%)
  * synthetic data: by scenario-id hash -> train 90% / calib 5% / test 5%
  All questions of one scenario stay in the same split.

Training targets:
  * synthetic: teacher soft distribution (averaged over option orders)
  * public: 0.5 * teacher + 0.5 * one-hot(gold)  (gold-only with 0.05 smoothing if no teacher)
  * think traces are kept only when the trace's answer matches the target argmax and the
    reasoning is at most --max-reason-chars long.

Outputs (artifacts/data/):
  train.jsonl  - training records (fast + think), consumed by train/train_mlx.py
  val.jsonl    - fast records from the calibration split, for loss monitoring during training
  calib.jsonl  - eval items for temperature fitting
  test.jsonl   - eval items for the final evaluation
  stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter

import numpy as np

from data.label_fast import load_scenarios
from data.teacher import ROOT
from nexus_decision.prompt import render_question
from nexus_decision.schema import parse_question

DATA = ROOT / "artifacts" / "data"


def bucket(sid: str) -> float:
    return int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(scn: dict) -> str:
    b = bucket(scn["id"])
    if scn["source"] != "synthetic":
        if scn.get("split") == "test":
            return "test"
        return "calib" if b < 0.10 else "train"
    return "calib" if b < 0.05 else ("test" if b < 0.10 else "train")


def read_jsonl(name: str) -> list[dict]:
    p = DATA / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def target_for(labels, teacher, gold) -> list[float] | None:
    n = len(labels)
    if gold is not None:
        onehot = np.full(n, 0.05 / max(1, n - 1))
        onehot[labels.index(gold)] = 0.95
        if teacher is None:
            return onehot.tolist()
        hard = np.zeros(n)
        hard[labels.index(gold)] = 1.0
        t = 0.5 * np.asarray(teacher) + 0.5 * hard
        return (t / t.sum()).tolist()
    return None if teacher is None else list(teacher)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-views", type=int, default=1, help="option orders per train question")
    ap.add_argument("--max-reason-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    scenarios = {s["id"]: s for s in load_scenarios()}
    fast = {(r["sid"], r["question"]): r for r in read_jsonl("labels_fast.jsonl")}
    think = read_jsonl("labels_think.jsonl")

    out = {"train": [], "val": [], "calib": [], "test": []}
    stats = Counter()

    for sid, scn in scenarios.items():
        sp = split_of(scn)
        for qname, qraw in scn["questions"].items():
            q = parse_question(qraw)
            labels = [lbl for lbl, _ in q.option_items()]
            fl = fast.get((sid, qname))
            teacher = fl["probs"] if fl else None
            gold = (scn.get("gold") or {}).get(qname)
            if sp in ("calib", "test"):
                if teacher is None and gold is None:
                    continue
                out[sp].append({
                    "sid": sid, "source": scn["source"], "domain": scn["domain"],
                    "question": qname, "state": scn["state"], "q": qraw, "labels": labels,
                    "teacher": teacher, "gold": gold,
                })
                stats[f"{sp}_items"] += 1
            target = target_for(labels, teacher, gold)
            if target is None or sp == "test":
                continue
            dest = "train" if sp == "train" else "val"
            views = args.fast_views if dest == "train" else 1
            for v in range(views):
                order = list(range(len(labels)))
                random.Random(f"{sid}/{qname}/{v}/{args.seed}").shuffle(order)
                rq = render_question(scn["state"], qname, q, order)
                out[dest].append({
                    "id": f"{sid}/{qname}/fast{v}", "kind": "fast", "user_text": rq.user_text,
                    "labels": rq.labels, "target": [target[labels.index(lbl)] for lbl in rq.labels],
                    "reasoning": None,
                })
                stats[f"{dest}_fast"] += 1

    for t in think:
        scn = scenarios.get(t["sid"])
        if not scn or split_of(scn) != "train":
            continue
        q = parse_question(scn["questions"][t["question"]])
        labels = [lbl for lbl, _ in q.option_items()]
        fl = fast.get((t["sid"], t["question"]))
        gold = (scn.get("gold") or {}).get(t["question"])
        target = target_for(labels, fl["probs"] if fl else None, gold)
        if target is None:
            stats["think_dropped_nolabel"] += 1
            continue
        if labels[int(np.argmax(target))] != t["label"]:
            stats["think_dropped_disagree"] += 1
            continue
        if len(t["reasoning"]) > args.max_reason_chars:
            stats["think_dropped_long"] += 1
            continue
        rq = render_question(scn["state"], t["question"], q, t["order"])
        out["train"].append({
            "id": f"{t['sid']}/{t['question']}/think", "kind": "think", "user_text": rq.user_text,
            "labels": rq.labels, "target": [target[labels.index(lbl)] for lbl in rq.labels],
            "reasoning": t["reasoning"],
        })
        stats["train_think"] += 1

    rng.shuffle(out["train"])
    for name, rows in out.items():
        with (DATA / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = dict(sorted(stats.items()))
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
