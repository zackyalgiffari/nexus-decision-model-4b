# Next steps (handoff, 2026-11-13)

This is a working note for resuming the project. Delete it before the repo goes public.

## Where things stand

| Item | Status |
|---|---|
| Data pipeline | Done. Teacher spend $4.12. Splits are in `artifacts/data/`. |
| Training | Done on Colab L4: run `run1`, 3,544 steps, about 5.5 h. Final val_agree 0.917, val_soft_ce 0.264. |
| Adapter | `artifacts/adapters/colab/final/` (PEFT), plus a copy on Drive at `MyDrive/nexus-decision-4b/adapters/run1/final`. The training log is `artifacts/adapters/colab/console.log`. |
| Merged model | `artifacts/fused/nexus-decision-4b/` (8.7 GB). Rebuild it in about 25 s with `uv run --no-sync python train/fuse_export.py --adapters artifacts/adapters/colab/final --out artifacts/fused/nexus-decision-4b`. |
| Sanity check | 60 test items, fast mode, MLX on M4. Tuned acc 0.950 vs base 0.883. p50 latency 0.82 s, which includes the reversed-order pass. |
| Full evaluation | **Not run yet.** Prepared to run on Colab: `notebooks/eval_colab.ipynb` and `artifacts/colab/nexus-eval.zip`; rebuild the zip with `bash scripts/pack_colab_eval.sh`. |
| Old base eval | The MLX base test run was renamed to `artifacts/eval/base_mlx_test_fast.jsonl`. The `base` entry in `artifacts/eval/results.json` is still from that MLX run. It will be overwritten. |
| Git | Local commits only. There is no remote. |

## 0. Move the repo out of iCloud first (recommended)

`~/Documents` is synced by iCloud ("Desktop & Documents Folders" is on). This causes four problems:

- **Upload failure.** iCloud is trying to upload the 20 GB repo, and the upload fails with
  `Quota exceeded` (about 0.8 GB left).
- **Evicted files.** iCloud has evicted files to "dataless" placeholders: 65k of them under
  `~/Documents`.
- **Conflict copies** such as `* 2.pth` and `WHEEL 2` in `.venv`.
- **Hidden flag on `.pth` files**, which breaks the editable install.

Steps:

1. Make sure no process is using the repo, then move it:
   `mkdir -p ~/Developer && mv ~/Documents/Programming/nexus-decision-model-4b ~/Developer/`
2. Recreate the virtualenv: `cd ~/Developer/nexus-decision-model-4b && rm -rf .venv && uv sync --all-extras`.
   Then run `uv run pytest -q`. Expect 13 tests to pass.
3. Copy the Claude Code memory, which is keyed by the project path:
   `cp -R ~/.claude/projects/-Users-zacky-Documents-Programming-nexus-decision-model-4b/memory ~/.claude/projects/-Users-zacky-Developer-nexus-decision-model-4b/`
   Create the target folder first if it does not exist.
4. Start Claude Code from the new folder.

## 1. Full evaluation on Colab (about 40 min on L4)

1. Upload `artifacts/colab/nexus-eval.zip` to `MyDrive/nexus-decision-4b/`.
2. Open `notebooks/eval_colab.ipynb` in Colab and select an L4 GPU.
3. Run the cells in order up to **2. Fast mode** and the quick-look cell. This does three things:
   - merges the adapter;
   - evaluates the tuned model on calib (804) and test (1,786);
   - evaluates the base model on calib and test, with the reversed order included.

   Results go to `MyDrive/nexus-decision-4b/eval/`. The cells are resumable.
4. Optional: **3. Think mode**. Run the calib subset (150) first, check the `s/item` figure, and only
   then run the test subset (300).
5. Download every `*.jsonl` file from `MyDrive/nexus-decision-4b/eval/` into `artifacts/eval/`.

## 2. Report and calibration (local, seconds)

```bash
uv run --no-sync python -m eval.report --teacher
uv run --no-sync python -m eval.report --name base    # writes nexus_config_base.json
uv run --no-sync python -m eval.report --name tuned   # writes nexus_config.json (ships with the model)
```

Check that tuned beats base on accuracy and ECE for the public and the synthetic test groups.
Check the flip rate as well.

## 3. Latency on the M4 (a few minutes)

The latency in the Colab results is L4 GPU latency. For the model card, also measure on the Mac:

```bash
uv run --no-sync python -m eval.run_eval --name tuned_m4 --model artifacts/fused/nexus-decision-4b \
    --split test --limit 50 --no-reverse
```

Then decide what the card reports. Either:

- pass `--latency-note "on an NVIDIA L4 (Transformers, bf16), batch size 1"` to `release/publish.py`, or
- add the M4 number to the card text.

## 4. Docs

- Fill `docs/evaluation.md` and `docs/training.md` with the real numbers. Training ran on an **L4**,
  not an A100. Use these values:
  - batch size 4 (think batch size 1), grad accumulation 4;
  - 2 epochs, 3,544 steps;
  - about 870 tok/s, peak 13.2 GB.
- `release/publish.py` defaults `--hardware` to A100. Pass
  `--hardware "1x NVIDIA L4 24GB (Google Colab), PyTorch + PEFT"`.

## 5. Release (only after the final numbers are confirmed)

1. Log in with `! hf auth login` and note the HF username.
2. Do a dry run and review `artifacts/fused/nexus-decision-4b/README.md`:
   `uv run --no-sync python -m release.publish --user <hf-user> --model-dir artifacts/fused/nexus-decision-4b --mlx 4 8 --hardware "..." --dry-run`
3. Run the README quick-start against the local folder, with Transformers and with MLX.
4. Publish **private** first by running the same command without `--dry-run`. Then publish the dataset:
   `uv run --no-sync python -m release.publish_dataset --user <hf-user>`
5. Load the uploaded repo fresh and recheck. Then switch to public with `--public`, or in the HF UI.
6. Delete this file before the repo goes public.
