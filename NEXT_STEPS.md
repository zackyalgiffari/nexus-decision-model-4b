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
| Full evaluation | **Done 2026-11-14** (Colab L4). Test, calibrated, fast mode: tuned acc 0.913 / ECE 0.017 / flip 0.036 vs base 0.847 / 0.020 / 0.111; teacher acc 0.927. Think mode (300 items) acc 0.880 vs fast 0.887 on the same items, p50 4.3 s. Results are in `artifacts/eval/results.json`; `artifacts/eval/nexus_config.json` (T fast 0.975, think 1.683) ships with the model. The pre-eval results file is kept as `results_before_full_eval.json`. |
| Old base eval | The MLX base test run was renamed to `artifacts/eval/base_mlx_test_fast.jsonl`. The `base` entry in `artifacts/eval/results.json` is still from that MLX run. It will be overwritten. |
| Git | Private GitHub repo `zackyalgiffari/nexus-decision-model-4b` (created 2026-11-14). Make it public together with the model, after deleting this file. `artifacts/` is not in git. |

## 0. Move the repo out of iCloud (done 2026-11-13)

The repo now lives at `~/Programming/nexus-decision-model-4b`, outside the iCloud-synced `~/Documents`.
`.venv` was rebuilt, and `uv run pytest -q` passes (13 tests). The Claude Code memory was copied to the new project
path. `~/Documents/Programming` is now a symlink to `~/Programming`, so old paths still resolve. It is the same folder,
not a second copy: never delete through that path.

## 1. Full evaluation on Colab (done 2026-11-14)

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

## 2. Report and calibration (done 2026-11-14)

```bash
uv run --no-sync python -m eval.report --teacher
uv run --no-sync python -m eval.report --name base    # writes nexus_config_base.json
uv run --no-sync python -m eval.report --name tuned   # writes nexus_config.json (ships with the model)
```

Check that tuned beats base on accuracy and ECE for the public and the synthetic test groups.
Check the flip rate as well.

## 3. Latency on the M4 (measured 2026-11-14)

Result: 50 test items, fast mode, MLX bf16, no reversed pass: p50 795 ms, p90 1.3 s (first item 1.9 s with warm-up).
On the same items the L4 p50 is 91 ms. MLX and L4 predictions agree on 50/50. Output: `artifacts/eval/tuned_m4_test_fast.jsonl`.

Original instructions:

The latency in the Colab results is L4 GPU latency. For the model card, also measure on the Mac:

```bash
uv run --no-sync python -m eval.run_eval --name tuned_m4 --model artifacts/fused/nexus-decision-4b \
    --split test --limit 50 --no-reverse
```

Then decide what the card reports. Either:

- pass `--latency-note "on an NVIDIA L4 (Transformers, bf16), batch size 1"` to `release/publish.py`, or
- add the M4 number to the card text.

## 4. Docs (done 2026-11-14)

`docs/evaluation.md` has a Results section, and `docs/training.md` has the released L4 run. Think mode is
documented as a reasoning/explanation feature, not an accuracy gain. The `--hardware` note below still applies
at release time.

Original instructions:

- Fill `docs/evaluation.md` and `docs/training.md` with the real numbers. Training ran on an **L4**,
  not an A100. Use these values:
  - batch size 4 (think batch size 1), grad accumulation 4;
  - 2 epochs, 3,544 steps;
  - about 870 tok/s, peak 13.2 GB.
- `release/publish.py` defaults `--hardware` to A100. Pass
  `--hardware "1x NVIDIA L4 24GB (Google Colab), PyTorch + PEFT"`.

## 5. Release (private upload verified, 2026-11-14)

All four repos are on the Hub as **private**, uploaded with `HF_XET_HIGH_PERFORMANCE=1` (about 6x faster than
the default): `zackyalgiffari/nexus-decision-4b`, `-MLX-4bit`, `-MLX-8bit`, and the dataset
`zackyalgiffari/nexus-decision-data`.

Verified: every weight file matches the local SHA-256; the README on the Hub matches the generated card; a
fresh download of the MLX 4-bit repo runs the card's quick start and the package installed from GitHub.

Remaining, after the user approves:

1. Delete this file and push.
2. Make the three model repos, the dataset and the GitHub repo public together.
