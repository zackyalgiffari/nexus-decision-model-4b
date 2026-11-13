"""Derived evaluation numbers for the model card and docs.

Reads the per-item eval files written by ``eval.run_eval`` and the temperatures written by
``eval.report``. It writes ``artifacts/eval/analysis.json``, which holds:

- bootstrap confidence intervals;
- breakdowns by source, question type, option count, abstain items, domain, language and
  difficulty;
- confidence-threshold coverage tables;
- think mode against fast mode on the same items;
- latency per machine and runtime, and quantised-MLX agreement.

Usage:
    uv run python -m eval.analysis
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nexus_decision.calibrate import softmax

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
DATA = ROOT / "artifacts" / "data"
N_BOOT = 2000
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def load(name: str) -> list[dict]:
    p = EVAL / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def item_meta() -> dict[str, dict]:
    """key -> question type, language and difficulty, joined from the test split and scenarios."""
    scen = {}
    for x in (DATA / "scenarios.jsonl").read_text().splitlines():
        s = json.loads(x)
        scen[s["id"]] = s
    meta = {}
    for x in (DATA / "test.jsonl").read_text().splitlines():
        it = json.loads(x)
        s = scen.get(it["sid"], {})
        meta[f"{it['sid']}/{it['question']}"] = {
            "qtype": it["q"]["type"],
            "language": s.get("language", "English"),
            "difficulty": s.get("difficulty"),
        }
    return meta


def probs_of(r: dict, temp: float, key: str = "logits") -> np.ndarray:
    return softmax(np.asarray(r[key]), temp)


def correct(r: dict) -> bool:
    """Top-1 is unchanged by temperature scaling, so no temperature is needed here."""
    return r["labels"][int(np.argmax(r["logits"]))] == r["ref"]


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def option_bucket(n: int) -> str:
    return "2" if n == 2 else "3-5" if n <= 5 else "6-8" if n <= 8 else "9+"


def breakdown(rows_by_model: dict[str, list[dict]], keyf, meta) -> list[dict]:
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for model, rows in rows_by_model.items():
        for r in rows:
            g = keyf(r, meta.get(r["key"], {}))
            if g is not None:
                groups[g][model].append(correct(r))
    out = []
    for g, by in groups.items():
        row = {"group": g, "n": len(next(iter(by.values())))}
        for model, v in by.items():
            row[model] = float(np.mean(v))
        out.append(row)
    return sorted(out, key=lambda r: -r["n"])


def thresholds(rows: list[dict], temp: float) -> list[dict]:
    conf = np.array([probs_of(r, temp).max() for r in rows])
    ok = np.array([correct(r) for r in rows])
    out = []
    for t in THRESHOLDS:
        m = conf >= t
        out.append({"threshold": t, "coverage": float(m.mean()),
                    "accuracy": float(ok[m].mean()) if m.any() else None,
                    "error": float(1 - ok[m].mean()) if m.any() else None})
    return out


def latency(rows: list[dict]) -> dict | None:
    lat = np.array([r["latency_s"] for r in rows if "latency_s" in r]) * 1000
    if not len(lat):
        return None
    return {"n": len(lat), "p50_ms": float(np.median(lat)), "p90_ms": float(np.percentile(lat, 90))}


def main() -> None:
    rng = np.random.default_rng(0)
    cfg = json.loads((EVAL / "nexus_config.json").read_text())["temperature"]
    base_cfg_path = EVAL / "nexus_config_base.json"
    base_t = json.loads(base_cfg_path.read_text())["temperature"]["fast"] if base_cfg_path.exists() else 1.0
    temps = {"tuned": cfg["fast"], "base": base_t}
    meta = item_meta()
    rows = {"tuned": load("tuned_test_fast"), "base": load("base_test_fast")}
    out: dict = {"temperatures": {"tuned": cfg, "base_fast": base_t}}

    # Confidence intervals on accuracy, and on the tuned-minus-base difference (paired bootstrap).
    base_by_key = {r["key"]: r for r in rows["base"]}
    ci = {}
    for g, pred in {"all": lambda r: True, "public": lambda r: r["gold"] is not None,
                    "synthetic": lambda r: r["gold"] is None}.items():
        tr = [r for r in rows["tuned"] if pred(r)]
        a = np.array([correct(r) for r in tr], float)
        b = np.array([correct(base_by_key[r["key"]]) for r in tr], float)
        idx = rng.integers(0, len(a), (N_BOOT, len(a)))
        d = a[idx].mean(1) - b[idx].mean(1)
        ci[g] = {"n": len(a), "tuned": float(a.mean()), "tuned_ci": boot_ci(a, rng),
                 "base": float(b.mean()), "base_ci": boot_ci(b, rng), "diff": float(a.mean() - b.mean()),
                 "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    out["accuracy_ci"] = ci

    out["breakdown"] = {
        "source": breakdown(rows, lambda r, m: r["source"], meta),
        "question_type": breakdown(rows, lambda r, m: m.get("qtype"), meta),
        "options": breakdown(rows, lambda r, m: option_bucket(len(r["labels"])), meta),
        "answer_kind": breakdown(rows, lambda r, m: "abstain (none_of_these)"
                                 if r["ref"] == "none_of_these" else "a named option", meta),
        "domain": breakdown(rows, lambda r, m: r["domain"] if r["source"] == "synthetic" else None, meta),
        "language": breakdown(rows, lambda r, m: m.get("language") if r["source"] == "synthetic" else None, meta),
        "difficulty": breakdown(rows, lambda r, m: m.get("difficulty") if r["source"] == "synthetic" else None, meta),
    }
    out["thresholds"] = {"all": thresholds(rows["tuned"], temps["tuned"]),
                         "public": thresholds([r for r in rows["tuned"] if r["gold"] is not None], temps["tuned"])}

    # Think mode against fast mode on exactly the same items.
    think = load("tuned_test_think")
    if think:
        fast = {r["key"]: r for r in rows["tuned"]}
        ks = [r["key"] for r in think]
        tk = {r["key"]: r for r in think}
        f_ok = np.array([correct(fast[k]) for k in ks])
        t_ok = np.array([correct(tk[k]) for k in ks])
        out["think_vs_fast"] = {
            "n": len(ks), "fast_accuracy": float(f_ok.mean()), "think_accuracy": float(t_ok.mean()),
            "agreement": float(np.mean([np.argmax(fast[k]["logits"]) == np.argmax(tk[k]["logits"]) for k in ks])),
            "think_fixes": int(np.sum(~f_ok & t_ok)), "think_breaks": int(np.sum(f_ok & ~t_ok)),
            "median_reasoning_chars": float(np.median([tk[k].get("reasoning_chars", 0) for k in ks])),
        }

    # Latency per machine and runtime; MLX quantisations against MLX bf16 on the same items.
    lat = {"l4_fast": latency(rows["tuned"]), "l4_think": latency(think)}
    m4 = {r["key"]: r for r in load("tuned_m4_test_fast")}
    if m4:
        lat["m4_mlx_bf16"] = latency(list(m4.values()))
        quant = {}
        for bits in (8, 4):
            q = load(f"m4_mlx{bits}bit_test_fast")
            if not q:
                continue
            lat[f"m4_mlx_{bits}bit"] = latency(q)
            quant[f"{bits}bit"] = {
                "n": len(q),
                "agreement_with_bf16": float(np.mean([np.argmax(r["logits"]) == np.argmax(m4[r["key"]]["logits"])
                                                      for r in q])),
                "accuracy": float(np.mean([correct(r) for r in q])),
            }
        quant["bf16_accuracy_same_items"] = float(np.mean([correct(r) for r in m4.values()]))
        out["mlx_quantisation"] = quant
    out["latency"] = {k: v for k, v in lat.items() if v}

    path = EVAL / "analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
