"""Build the model card for nexus-decision-4b and its MLX variants.

Every number in the card is read from the evaluation outputs, so the card cannot drift from the
results:

- ``artifacts/eval/results.json``, written by ``eval.report``;
- ``artifacts/eval/analysis.json``, written by ``eval.analysis``;
- ``artifacts/data/stats.json``, written by ``data.build_splits``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
DATA = ROOT / "artifacts" / "data"
TEMPLATE = ROOT / "release" / "model_card.md.j2"
COLLECTION = "nexus-decision-4b-6af7fc5003be12bf9aa5b4c9"  # slug of the HF collection, after the user name

MODELS = [("tuned", "**nexus-decision-4b**"), ("base", "Qwen3.5-4B (base, same prompt)"),
          ("teacher", "DeepSeek-V4.1-Flash (teacher)")]
LANG_MIN_N = 20  # languages with fewer synthetic test items are pooled, not reported on their own


def f3(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def table(head: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] + ["--:"] * (len(head) - 1)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def quality_table(results: dict, ci: dict) -> str:
    rows = []
    for group in ("public", "synthetic"):
        for run, label in MODELS:
            m = results.get(run, {}).get("fast", {}).get("test", {}).get(group)
            if not m:
                continue
            if run == "teacher" and group == "synthetic":
                rows.append([f"{group} (n={m['n']:,})", label] + ["–"] * 6 + ["–"])
                continue
            acc = f3(m["accuracy"])
            if run in ("tuned", "base"):
                lo, hi = ci[group][f"{run}_ci"]
                acc = f"{acc} ({lo:.3f}–{hi:.3f})"
            if run == "tuned":
                acc = f"**{acc}**"
            rows.append([f"{group} (n={m['n']:,})", label, acc, f3(m["macro_f1"]), f3(m["ece"]), f3(m["brier"]),
                         f3(m["nll"]), f3(m["coverage_at_5pct_error"]), f3(m.get("order_flip_rate"))])
    return table(["Test group", "Model", "Accuracy (95% CI)", "Macro-F1", "ECE ↓", "Brier ↓", "NLL ↓",
                  "Coverage @5% error", "Order flip ↓"], rows, ["---", "---"] + ["--:"] * 7)


def threshold_table(rows: list[dict]) -> str:
    return table(["Act automatically when confidence ≥", "Share of questions decided", "Accuracy on those",
                  "Error on those"],
                 [[f"{r['threshold']:.2f}", pct(r["coverage"]), pct(r["accuracy"]), pct(r["error"])] for r in rows])


def breakdown_table(rows: list[dict], label: str, order: list[str] | None = None) -> str:
    if order:
        rows = sorted(rows, key=lambda r: order.index(r["group"]) if r["group"] in order else len(order))
    return table([label, "n", "nexus-decision-4b", "Base", "Δ"],
                 [[r["group"].replace("_", " "), f"{r['n']:,}", f3(r["tuned"]), f3(r["base"]),
                   f"{r['tuned'] - r['base']:+.3f}"] for r in rows])


def language_table(rows: list[dict]) -> str:
    big = [r for r in rows if r["n"] >= LANG_MIN_N]
    small = [r for r in rows if r["n"] < LANG_MIN_N]
    if small:
        n = sum(r["n"] for r in small)
        big.append({"group": "Other (" + ", ".join(r["group"] for r in small) + ")", "n": n,
                    "tuned": sum(r["tuned"] * r["n"] for r in small) / n,
                    "base": sum(r["base"] * r["n"] for r in small) / n})
    return breakdown_table(big, "Language")


def latency_table(lat: dict) -> str:
    spec = [("l4_fast", "NVIDIA L4", "Transformers, bf16", "fast"),
            ("l4_think", "NVIDIA L4", "Transformers, bf16", "think"),
            ("m4_mlx_bf16", "Apple M4, 24 GB", "MLX, bf16", "fast"),
            ("m4_mlx_8bit", "Apple M4, 24 GB", "MLX, 8-bit", "fast"),
            ("m4_mlx_4bit", "Apple M4, 24 GB", "MLX, 4-bit", "fast")]

    def ms(x: float) -> str:
        return f"{x:,.0f} ms" if x < 1000 else f"{x / 1000:.1f} s"

    rows = [[hw, rt, mode, f"{lat[k]['n']:,}", ms(lat[k]["p50_ms"]), ms(lat[k]["p90_ms"])]
            for k, hw, rt, mode in spec if k in lat]
    return table(["Hardware", "Runtime", "Mode", "Questions", "p50", "p90"], rows,
                 ["---", "---", "---", "--:", "--:", "--:"])


def model_index(name: str, user: str, results: dict) -> list[dict]:
    """Structured results for the Hub's evaluation widget."""
    out = []
    for group, desc in (("public", "BANKING77 + CLINC150 test, as typed questions"),
                        ("synthetic", "synthetic decision scenarios, teacher labels")):
        m = results["tuned"]["fast"]["test"][group]
        out.append({
            "task": {"type": "text-classification", "name": "Typed decision questions"},
            "dataset": {"name": f"nexus-decision-data ({desc})", "type": f"{user}/nexus-decision-data",
                        "split": "test"},
            "metrics": [{"type": "accuracy", "value": round(m["accuracy"], 4), "name": "Accuracy"},
                        {"type": "f1", "value": round(m["macro_f1"], 4), "name": "Macro-F1"},
                        {"type": "ece", "value": round(m["ece"], 4), "name": "Expected calibration error"},
                        {"type": "brier", "value": round(m["brier"], 4), "name": "Brier score"}],
        })
    return [{"name": name, "results": out}]


def context(user: str, github: str, hardware: str, train_info: dict | None) -> dict:
    results = json.loads((EVAL / "results.json").read_text())
    an = json.loads((EVAL / "analysis.json").read_text())
    stats = json.loads((DATA / "stats.json").read_text())
    ci = an["accuracy_ci"]
    tuned, base, teacher = (results[k]["fast"]["test"] for k in ("tuned", "base", "teacher"))
    bd = an["breakdown"]
    weak = sorted(bd["domain"], key=lambda r: r["tuned"])[:2]
    thr = {r["threshold"]: r for r in an["thresholds"]["all"]}
    return {
        "hf_user": user, "gh_repo": github, "hardware": hardware,
        "collection_url": f"https://huggingface.co/collections/{user}/{COLLECTION}",
        "results": results, "an": an, "stats": stats, "ci": ci,
        "tuned": tuned, "base": base, "teacher": teacher, "temps": an["temperatures"]["tuned"],
        "thr": thr, "weak_domains": weak, "think": an.get("think_vs_fast"), "lat": an["latency"],
        "quant": an.get("mlx_quantisation", {}), "train": train_info or {},
        "quality_table": quality_table(results, ci),
        "threshold_table": threshold_table(an["thresholds"]["all"]),
        "qtype_table": breakdown_table(bd["question_type"], "Question type", ["choice", "binary", "score"]),
        "options_table": breakdown_table(bd["options"], "Options shown", ["2", "3-5", "6-8", "9+"]),
        "answer_table": breakdown_table(bd["answer_kind"], "Correct answer"),
        "source_table": breakdown_table(bd["source"], "Source", ["banking77", "clinc150", "synthetic"]),
        "domain_table": breakdown_table(sorted(bd["domain"], key=lambda r: -r["tuned"]), "Domain (synthetic)"),
        "language_table": language_table(bd["language"]),
        "difficulty_table": breakdown_table(bd["difficulty"], "Difficulty (synthetic)",
                                            ["easy", "medium", "hard", "ambiguous"]),
        "latency_table": latency_table(an["latency"]),
        "f3": f3, "pct": pct,
    }


def render(ctx: dict, repo_name: str, mlx_bits: int | None = None) -> str:
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(TEMPLATE.read_text())
    meta = {
        "license": "apache-2.0",
        **({"base_model": f"{ctx['hf_user']}/nexus-decision-4b", "base_model_relation": "quantized",
            "library_name": "mlx"} if mlx_bits else
           {"base_model": "Qwen/Qwen3.5-4B", "library_name": "transformers"}),
        "pipeline_tag": "text-generation",
        "language": ["en", "id", "es", "de", "fr", "pt", "ja"],
        "tags": ["decision-making", "classification", "routing", "triage", "calibration", "distillation",
                 "qwen3.5", "deepseek"] + (["mlx"] if mlx_bits else []),
        "datasets": [f"{ctx['hf_user']}/nexus-decision-data", "PolyAI/banking77", "clinc/clinc_oos"],
        "model-index": model_index(repo_name, ctx["hf_user"], ctx["results"]),
    }
    front = "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n\n"
    return front + tpl.render(**ctx, repo_name=repo_name, mlx_bits=mlx_bits)
