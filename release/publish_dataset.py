"""Upload the nexus-decision-4b training data to the Hugging Face Hub as a dataset repo.

Private by default; pass --public after review.

Usage:
    uv run python -m release.publish_dataset --user <hf-username> [--public] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "data"
FILES = ["scenarios.jsonl", "public.jsonl", "labels_fast.jsonl", "labels_think.jsonl",
         "train.jsonl", "val.jsonl", "calib.jsonl", "test.jsonl", "stats.json"]

CARD = """---
license: cc-by-4.0
language: [en, id, es, de, fr, pt, ja]
task_categories: [text-classification]
tags: [decision-making, routing, triage, calibration, distillation, synthetic]
size_categories: [10K<n<100K]
configs:
  - config_name: train
    data_files: train.jsonl
  - config_name: test
    data_files: test.jsonl
---

# nexus-decision-data

This is the training and evaluation data for
[{user}/nexus-decision-4b](https://huggingface.co/{user}/nexus-decision-4b), an open-weight
typed decision model. It holds typed decision questions (`choice`, `score`, `binary`) about an
application state, with soft labels from a teacher model.

| File | Contents |
|---|---|
| `scenarios.jsonl` | Synthetic scenarios: `state` plus 1-4 typed `questions`, in 12 domains. Written by DeepSeek-V4.1-Flash. |
| `public.jsonl` | BANKING77 and CLINC150 texts converted to typed questions, with `gold` labels |
| `labels_fast.jsonl` | Teacher soft distributions (top-20 logprobs, averaged over 3 option orders) |
| `labels_think.jsonl` | Teacher reasoning traces (thinking mode) with the displayed option order and the answer |
| `train.jsonl` / `val.jsonl` | Rendered training records (`user_text`, `labels`, `target`, optional `reasoning`) |
| `calib.jsonl` / `test.jsonl` | Evaluation items (`state`, `q`, `labels`, `teacher`, `gold`) |
| `stats.json` | Split statistics |

{stats}

## Provenance and licensing

- **Synthetic scenarios and labels.** They were generated with the DeepSeek API (`deepseek-flash`)
  and are released under CC-BY-4.0. They are model-generated: they may contain mistakes, and any
  names and IDs in them are fictional.
- **BANKING77-derived items** (`source: banking77`). The texts come from
  [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), CC-BY-4.0.
- **CLINC150-derived items** (`source: clinc150`). The texts come from
  [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos), CC-BY-3.0 (Larson et al., 2019).

The test split of the public data is the original datasets' test split. It was never used for
training.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--name", default="nexus-decision-data")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = ROOT / "artifacts" / "release" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (DATA / f).exists():
            shutil.copyfile(DATA / f, out / f)
    stats = json.loads((DATA / "stats.json").read_text())
    table = "| Statistic | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v:,} |" for k, v in stats.items())
    (out / "README.md").write_text(CARD.format(user=args.user, stats=table))
    print(f"prepared {out}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    repo = f"{args.user}/{args.name}"
    api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(out),
                      commit_message="Upload nexus-decision-data")
    print(f"uploaded https://huggingface.co/datasets/{repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
