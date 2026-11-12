#!/usr/bin/env bash
# Bundle the code and training data that the Colab notebook needs into one zip.
# Upload the result to Google Drive at  MyDrive/nexus-decision-4b/nexus-colab.zip
set -euo pipefail
cd "$(dirname "$0")/.."
out=artifacts/colab/nexus-colab.zip
mkdir -p artifacts/colab
rm -f "$out"
for f in artifacts/data/train.jsonl artifacts/data/val.jsonl; do
  [[ -s "$f" ]] || { echo "missing $f (run data/build_splits.py first)"; exit 1; }
done
zip -q -r "$out" pyproject.toml src train/__init__.py train/train_torch.py train/fuse_export.py \
  train/configs artifacts/data/train.jsonl artifacts/data/val.jsonl artifacts/data/stats.json \
  -x '*/__pycache__/*'
ls -lh "$out"
echo "Upload $out to Google Drive: MyDrive/nexus-decision-4b/nexus-colab.zip"
