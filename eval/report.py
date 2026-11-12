"""Fit temperatures on the calibration split and compute test metrics.

Usage:
    uv run python -m eval.report --name tuned            # fits T on calib, scores test
    uv run python -m eval.report --name base --no-fit    # T = 1
    uv run python -m eval.report --teacher               # teacher baseline from labels_fast

Writes artifacts/eval/results.json (merged across runs) and, for fitted runs,
artifacts/eval/nexus_config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nexus_decision import metrics
from nexus_decision.calibrate import fit_temperature, save_config, softmax

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
GROUPS = {"all": lambda r: True, "public": lambda r: r["gold"] is not None,
          "synthetic": lambda r: r["gold"] is None}


def load(name: str, split: str, mode: str) -> list[dict]:
    p = EVAL / f"{name}_{split}_{mode}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def score(rows: list[dict], temp: float, key: str = "logits") -> dict:
    out = {}
    for g, pred in GROUPS.items():
        rs = [r for r in rows if pred(r)]
        if not rs:
            continue
        probs = [softmax(np.asarray(r[key]), temp) for r in rs]
        labels = [r["labels"].index(r["ref"]) for r in rs]
        out[g] = metrics.summary(probs, labels)
        if "logits_reversed" in rs[0]:
            a = [r["labels"][int(np.argmax(r["logits"]))] for r in rs]
            b = [r["labels"][int(np.argmax(r["logits_reversed"]))] for r in rs]
            out[g]["order_flip_rate"] = metrics.flip_rate(a, b)
            # debiased: average of canonical + reversed probabilities
            pdb = [(softmax(np.asarray(r["logits"]), temp) + softmax(np.asarray(r["logits_reversed"]), temp)) / 2 for r in rs]
            out[g]["debiased_accuracy"] = metrics.accuracy(pdb, labels)
            out[g]["debiased_ece"] = metrics.ece(pdb, labels)
        lat = [r["latency_s"] for r in rs if "latency_s" in r]
        if lat:
            out[g]["latency_ms_p50"] = float(np.median(lat) * 1000)
    return out


def teacher_rows() -> list[dict]:
    rows = []
    for x in (ROOT / "artifacts" / "data" / "test.jsonl").read_text().splitlines():
        it = json.loads(x)
        if it["teacher"] is None:
            continue
        ref = it["gold"] if it["gold"] is not None else it["labels"][int(np.argmax(it["teacher"]))]
        rows.append({"labels": it["labels"], "ref": ref, "gold": it["gold"],
                     "logits": np.log(np.clip(it["teacher"], 1e-9, 1)).tolist()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--teacher", action="store_true")
    args = ap.parse_args()

    res_path = EVAL / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    if args.teacher:
        results["teacher"] = {"fast": {"temperature": 1.0, "test": score(teacher_rows(), 1.0)}}
    else:
        cfg = {"temperature": {}}
        entry = {}
        for mode in ("fast", "think"):
            test = load(args.name, "test", mode)
            if not test:
                continue
            calib = load(args.name, "calib", mode)
            temp = 1.0
            if calib and not args.no_fit:
                logits = [np.asarray(r["logits"]) for r in calib]
                targets = [r["labels"].index(r["ref"]) for r in calib]
                temp = fit_temperature(logits, targets)
            cfg["temperature"][mode] = temp
            entry[mode] = {"temperature": temp, "test": score(test, temp),
                           "test_uncalibrated": score(test, 1.0)}
        results[args.name] = entry
        if not args.no_fit and cfg["temperature"]:
            cfg["temperature"].setdefault("think", cfg["temperature"].get("fast", 1.0))
            save_config(cfg, EVAL / "nexus_config.json")

    res_path.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k in (args.name, "teacher")}, indent=2))


if __name__ == "__main__":
    main()
