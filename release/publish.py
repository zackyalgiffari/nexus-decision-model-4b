"""Render the model cards and upload nexus-decision-4b and its MLX variants to the Hugging Face Hub.

Repos are created **private** by default. Pass --public only after reviewing them.

Usage:
    uv run python -m eval.analysis                       # refresh the numbers the card uses
    uv run python -m release.publish --user <hf-user> --model-dir artifacts/fused/nexus-decision-4b \
        [--mlx 4 8] [--public] [--dry-run]
    # write the cards somewhere else, without touching the model folders:
    uv run python -m release.publish --user <hf-user> --model-dir ... --mlx 4 8 --preview /tmp/cards
    # re-upload only README.md to repos that already exist:
    uv run python -m release.publish --user <hf-user> --model-dir ... --mlx 4 8 --card-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from release.card import context, render

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", default="nexus-decision-4b")
    ap.add_argument("--github", default="zackyalgiffari/nexus-decision-model-4b", help="GitHub repo with the code")
    ap.add_argument("--train-info", default="artifacts/adapters/colab/final/nexus_train.json",
                    help="nexus_train.json written by the trainer next to the adapter")
    ap.add_argument("--mlx", nargs="*", type=int, default=[], help="MLX quantisations, e.g. 4 8")
    ap.add_argument("--hardware", default="1× NVIDIA L4 24 GB (Google Colab), PyTorch + PEFT, about 5.5 h")
    ap.add_argument("--public", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="prepare folders, upload nothing")
    mode.add_argument("--preview", metavar="DIR", help="write the cards to DIR only")
    mode.add_argument("--card-only", action="store_true", help="upload README.md to existing repos")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    train_path = ROOT / args.train_info
    train_info = json.loads(train_path.read_text()) if train_path.exists() else None
    if train_info is None:
        sys.exit(f"{train_path} missing: pass --train-info")
    ctx = context(args.user, args.github, args.hardware, train_info)

    repos = [(args.name, None, model_dir)]
    repos += [(f"{args.name}-MLX-{b}bit", b, ROOT / "artifacts" / "release" / f"{args.name}-MLX-{b}bit")
              for b in args.mlx]
    cards = {name: render(ctx, name, bits) for name, bits, _ in repos}

    if args.preview:
        out = Path(args.preview)
        out.mkdir(parents=True, exist_ok=True)
        for name, card in cards.items():
            (out / f"{name}.md").write_text(card)
        print(f"cards written to {out}")
        return

    if args.card_only:
        from huggingface_hub import HfApi

        api = HfApi()
        for name, _, folder in repos:
            (folder / "README.md").write_text(cards[name])
            api.upload_file(path_or_fileobj=str(folder / "README.md"), path_in_repo="README.md",
                            repo_id=f"{args.user}/{name}", commit_message="Update model card")
            print(f"card updated: https://huggingface.co/{args.user}/{name}")
        return

    cfg = EVAL / "nexus_config.json"
    if not cfg.exists():
        sys.exit("artifacts/eval/nexus_config.json missing: run eval.report first")
    for name, bits, folder in repos:
        if bits and not folder.exists():
            subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(model_dir),
                            "--mlx-path", str(folder), "-q", "--q-bits", str(bits)], check=True)
        shutil.copyfile(cfg, folder / "nexus_config.json")
        (folder / "README.md").write_text(cards[name])
        for junk in folder.rglob(".DS_Store"):
            junk.unlink()
    print("cards written: " + ", ".join(str(f / "README.md") for _, _, f in repos))

    if args.dry_run:
        print("dry run: nothing uploaded")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    for name, _, folder in repos:
        repo = f"{args.user}/{name}"
        api.create_repo(repo, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message="Upload nexus-decision-4b")
        print(f"uploaded https://huggingface.co/{repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
