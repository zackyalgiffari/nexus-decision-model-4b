#!/usr/bin/env bash
# Bundle the code and evaluation data that notebooks/eval_colab.ipynb needs into one zip.
# Upload the result to Google Drive at  MyDrive/nexus-decision-4b/nexus-eval.zip
set -euo pipefail
cd "$(dirname "$0")/.."
out=artifacts/colab/nexus-eval.zip
mkdir -p artifacts/colab
rm -f "$out"
for f in artifacts/data/calib.jsonl artifacts/data/test.jsonl; do
  [[ -s "$f" ]] || { echo "missing $f (run data/build_splits.py first)"; exit 1; }
done
zip -q -r "$out" pyproject.toml src eval train/__init__.py train/fuse_export.py \
  artifacts/data/calib.jsonl artifacts/data/test.jsonl artifacts/data/stats.json \
  -x '*/__pycache__/*'
ls -lh "$out"
echo "Upload $out to Google Drive: MyDrive/nexus-decision-4b/nexus-eval.zip"
