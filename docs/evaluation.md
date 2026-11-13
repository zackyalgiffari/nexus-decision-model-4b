# Evaluation and calibration

## Procedure

1. `eval/run_eval.py` runs a model over the calibration and test items and stores the
   **uncalibrated** letter logits per item. In fast mode it does this for the canonical and the
   reversed option order.
2. `eval/report.py` does two things:
   - it fits one temperature per mode on the calibration split, by minimising NLL against the
     reference labels, and writes `nexus_config.json`;
   - it scores the test split.

## Metrics

| Metric | Meaning |
|---|---|
| accuracy, macro-F1 | Top-1 against the reference label |
| ECE (15 bins) | Gap between confidence and accuracy (lower is better) |
| Brier, NLL | Proper scoring rules (lower is better) |
| coverage@5% error | Fraction of items that can be auto-decided while keeping the error rate at or below 5%, taking the most confident items first |
| AURC | Area under the risk-coverage curve (lower is better) |
| order flip rate | Fraction of items whose predicted label changes when the option order is reversed |
| debiased accuracy / ECE | The same metrics after averaging canonical and reversed orders |
| latency p50 | Median wall-clock time per question on the evaluation machine |

Results are reported separately for **public** items (human gold labels) and **synthetic** items
(teacher reference labels).

## Baselines

| Baseline | What it is |
|---|---|
| `base` | Untuned `Qwen/Qwen3.5-4B`, with the same prompt and letter readout |
| `teacher` | `deepseek-flash` soft labels, averaged over 3 option orders |

The teacher sees the same prompts. On synthetic items it is the reference by construction, so
compare it on public items only.

## Results (released model)

### Setup

| | |
|---|---|
| Test split | 1,786 items: 600 banking77, 600 clinc150 (public, gold labels), 586 synthetic (teacher labels) |
| Calibration split | 804 items, used only to fit the temperatures |
| Hardware | 1× NVIDIA L4 (Google Colab), Transformers 5.22.2, bf16, batch size 1 (`--backend hf`) |
| Temperatures | Fast mode 0.975, think mode 1.683; they ship in `nexus_config.json` |

All figures below are on the test split after calibration, unless marked otherwise.

### Fast mode

| Model | Group | n | Accuracy | Macro-F1 | ECE | Brier | NLL | Coverage@5% err | AURC |
|---|---|---|---|---|---|---|---|---|---|
| **nexus-decision-4b** | all | 1,786 | **0.913** | **0.883** | **0.017** | **0.126** | **0.240** | **0.923** | **0.014** |
| | public | 1,200 | 0.915 | 0.889 | 0.020 | 0.118 | 0.228 | 0.934 | 0.012 |
| | synthetic | 586 | 0.908 | 0.885 | 0.021 | 0.143 | 0.264 | 0.860 | 0.017 |
| Qwen3.5-4B (base) | all | 1,786 | 0.847 | 0.805 | 0.020 | 0.232 | 0.463 | 0.608 | 0.048 |
| | public | 1,200 | 0.845 | 0.811 | 0.029 | 0.229 | 0.476 | 0.573 | 0.048 |
| | synthetic | 586 | 0.850 | 0.815 | 0.054 | 0.239 | 0.435 | 0.604 | 0.046 |
| Teacher (deepseek-flash) | public | 1,200 | 0.892 | 0.869 | 0.070 | 0.180 | 0.544 | 0.826 | 0.030 |

- **Against the base model**, the tuned model is better on every metric, in both the public and
  the synthetic group.
- **Against the teacher**, on the public items with human gold labels, the student is more
  accurate (0.915 vs 0.892) and much better calibrated (ECE 0.020 vs 0.070). The teacher's
  synthetic score is 1.000 by construction, so it is left out.
- **Calibration** barely moves the tuned model: its fitted temperature is 0.975, and its
  uncalibrated ECE is already 0.017. The base model needs T = 1.209 to get from ECE 0.045 to
  0.020.

### Robustness to option order

| Model | Order flip rate | Debiased accuracy | Debiased ECE |
|---|---|---|---|
| nexus-decision-4b | **0.036** | **0.918** | **0.013** |
| Qwen3.5-4B (base) | 0.111 | 0.851 | 0.019 |

Reversing the option order changes the base model's answer on 11.1% of items. For the tuned model
the figure is 3.6%. Averaging the canonical and the reversed order adds about half a point of
accuracy. The runtime's `debias=True` does the same kind of averaging, over cyclic rotations of
the options (see [protocol.md](protocol.md)).

### Think mode

Think mode was run on a random subset of 300 test items (seed 0, `max_think_tokens=1024`), with
its temperature fitted on 150 calibration items.

| Mode, same 300 items | Accuracy | ECE | Latency p50 (L4) |
|---|---|---|---|
| fast | 0.887 | – | 0.09 s |
| think | 0.880 | 0.041 | 4.3 s |

Think mode does **not** improve accuracy. It fixes 11 of the fast-mode errors but introduces 13
new ones, and the two modes agree on 91.3% of items. The median trace is short, about 180
characters. Use fast mode for decisions. Use think mode when a human-readable reason is worth
roughly 45× the latency.

### Latency

| Machine | Runtime | Mode | p50 | p90 |
|---|---|---|---|---|
| NVIDIA L4 | Transformers, bf16 | fast | 93 ms | – |
| NVIDIA L4 | Transformers, bf16 | think | 4.3 s | – |
| Apple M4, 24 GB | MLX, bf16 | fast | 795 ms | 1.3 s |

Latency is wall-clock time for one question, batch size 1, for the canonical order only. The
reversed-order pass is excluded. The M4 figure comes from 50 test items. On those items the MLX
and the L4 predictions agree 50 out of 50 times.

### Reproduce

```bash
# on a CUDA GPU (Colab: notebooks/eval_colab.ipynb)
python -m eval.run_eval --backend hf --mode fast --name tuned --model <fused-model> --split calib
python -m eval.run_eval --backend hf --mode fast --name tuned --model <fused-model> --split test
python -m eval.run_eval --backend hf --mode think --name tuned --model <fused-model> --split test --limit 300
# locally
uv run python -m eval.report --teacher
uv run python -m eval.report --name base
uv run python -m eval.report --name tuned      # writes artifacts/eval/nexus_config.json
```
