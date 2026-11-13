"""Render the model card and upload nexus-decision-4b (and optional variants) to the Hugging Face Hub.

Repos are created **private** by default. Pass --public only after reviewing them.

Usage:
    uv run python -m release.publish --user <hf-username> --model-dir artifacts/fused/nexus-decision-4b \
        [--mlx 4 8] [--public] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
METRICS = [("accuracy", "acc", "{:.3f}"), ("macro_f1", "F1", "{:.3f}"), ("ece", "ECE", "{:.3f}"),
           ("brier", "Brier", "{:.3f}"), ("coverage_at_5pct_error", "cov@5%", "{:.2f}"),
           ("order_flip_rate", "flip", "{:.3f}"), ("latency_ms_p50", "p50 ms", "{:.0f}")]
RUNS = [("tuned", "**nexus-decision-4b**"), ("base", "Qwen3.5-4B (base, same prompt)"),
        ("teacher", "DeepSeek-V4.1-Flash (teacher)")]


def results_table(results: dict) -> str:
    head = "| Split | Model | " + " | ".join(h for _, h, _ in METRICS) + " |"
    sep = "|---|---|" + "---|" * len(METRICS)
    rows = [head, sep]
    for group in ("public", "synthetic"):
        for run, label in RUNS:
            m = results.get(run, {}).get("fast", {}).get("test", {}).get(group)
            if not m:
                continue
            cells = [fmt.format(m[k]) if k in m else "-" for k, _, fmt in METRICS]
            rows.append(f"| {group} (n={m['n']}) | {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", default="nexus-decision-4b")
    ap.add_argument("--github", default="zackyalgiffari/nexus-decision-model-4b", help="GitHub repo with the code")
    ap.add_argument("--mlx", nargs="*", type=int, default=[], help="MLX quantisations, e.g. 4 8")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hardware", default="1x NVIDIA L4 24GB (Google Colab), PyTorch + PEFT")
    ap.add_argument("--latency-note", default="on an NVIDIA L4 (Transformers, bf16), batch size 1, canonical order only; "
                    "on an Apple M4 (24 GB) with MLX bf16 the fast-mode p50 is about 0.8 s")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    results = json.loads((EVAL / "results.json").read_text())
    stats = json.loads((ROOT / "artifacts" / "data" / "stats.json").read_text())
    data_summary = (f"{stats.get('train_fast', 0):,} fast records and {stats.get('train_think', 0):,} "
                    "think records (synthetic scenarios in 12 domains plus BANKING77 and CLINC150)")
    template = Template((ROOT / "release" / "model_card.md.j2").read_text(), keep_trailing_newline=True)

    def render(repo_name: str, mlx_bits: int | None = None) -> str:
        return template.render(
            hf_user=args.user, repo_name=repo_name, mlx_bits=mlx_bits, gh_repo=args.github, results_table=results_table(results),
            data_summary=data_summary, hardware=args.hardware, latency_note=args.latency_note,
        )

    card = render(args.name)
    (model_dir / "README.md").write_text(card)
    cfg = EVAL / "nexus_config.json"
    if cfg.exists():
        shutil.copyfile(cfg, model_dir / "nexus_config.json")
    if not (model_dir / "nexus_config.json").exists():
        sys.exit("nexus_config.json missing: run eval.report first")
    print(f"model card written to {model_dir / 'README.md'}")

    variants = []
    for bits in args.mlx:
        out = ROOT / "artifacts" / "release" / f"{args.name}-MLX-{bits}bit"
        if not out.exists():
            subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(model_dir),
                            "--mlx-path", str(out), "-q", "--q-bits", str(bits)], check=True)
        shutil.copyfile(model_dir / "nexus_config.json", out / "nexus_config.json")
        (out / "README.md").write_text(render(f"{args.name}-MLX-{bits}bit", bits))
        variants.append((out, f"{args.user}/{args.name}-MLX-{bits}bit"))

    if args.dry_run:
        print("dry run: nothing uploaded")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    for folder, repo in [(model_dir, f"{args.user}/{args.name}"), *variants]:
        api.create_repo(repo, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message="Upload nexus-decision-4b")
        print(f"uploaded https://huggingface.co/{repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
