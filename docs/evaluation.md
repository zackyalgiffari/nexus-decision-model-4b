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

All figures below are on the test split after calibration, unless marked otherwise. The 95%
confidence intervals are percentile bootstrap intervals over items (2,000 resamples, seed 0). The
tuned-minus-base difference uses a paired bootstrap.

> **Not comparable to BANKING77 or CLINC150 leaderboards.** Those benchmarks are 77-way and
> 150-way intent classification. Here each message becomes a typed question: the correct intent
> among 3 to 11 distractors plus `none_of_these`, with the correct intent withheld in 15% of
> in-scope cases, or a binary yes/no question (20%). See [data.md](data.md).

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
  the synthetic group. Accuracy gains, with 95% CIs: public +7.0 points (+5.3 to +8.7), synthetic
  +5.8 (+3.1 to +8.5). Tuned accuracy on public items is 0.915 (0.898 to 0.931).
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

### Confidence thresholds

Coverage and error when only answers at or above a confidence threshold are acted on, over the
full test set:

| Confidence ≥ | Coverage | Accuracy | Error |
|---|---|---|---|
| 0.50 | 97.4% | 92.5% | 7.5% |
| 0.70 | 88.9% | 95.8% | 4.2% |
| 0.80 | 83.7% | 96.8% | 3.2% |
| 0.90 | 73.9% | 98.2% | 1.8% |
| 0.95 | 68.3% | 98.6% | 1.4% |
| 0.99 | 52.5% | 99.7% | 0.3% |

### Breakdown

Accuracy of the tuned model and of the base model:

| Slice | n | Tuned | Base |
|---|---|---|---|
| Question type: choice | 1,265 | 0.911 | 0.834 |
| Question type: binary | 404 | 0.936 | 0.899 |
| Question type: score | 117 | 0.855 | 0.803 |
| Options: 2 | 404 | 0.936 | 0.899 |
| Options: 3–5 | 595 | 0.919 | 0.859 |
| Options: 6–8 | 374 | 0.917 | 0.834 |
| Options: 9+ | 413 | 0.877 | 0.789 |
| Answer is a named option | 1,532 | 0.926 | 0.862 |
| Answer is `none_of_these` (abstain) | 254 | 0.835 | 0.752 |
| Source: banking77 | 600 | 0.877 | 0.793 |
| Source: clinc150 | 600 | 0.953 | 0.897 |
| Source: synthetic | 586 | 0.908 | 0.850 |

The weakest areas are abstain cases, `score` questions, lists of 9 or more options, and two
synthetic domains: moderation triage (0.818, n=55) and policy compliance (0.837, n=49). The
per-domain, per-language and per-difficulty tables are in the model card and in
`artifacts/eval/analysis.json`. Each non-English language has fewer than 30 test items.

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
| NVIDIA L4 | Transformers, bf16 | fast | 93 ms | 157 ms |
| NVIDIA L4 | Transformers, bf16 | think | 4.3 s | 8.3 s |
| Apple M4, 24 GB | MLX, bf16 | fast | 795 ms | 1.3 s |
| Apple M4, 24 GB | MLX, 8-bit | fast | 829 ms | 1.4 s |
| Apple M4, 24 GB | MLX, 4-bit | fast | 785 ms | 1.3 s |

Latency is wall-clock time for one question, batch size 1, for the canonical order only. The
reversed-order pass is excluded. The M4 figures come from the same 50 test items. On those items
the MLX bf16 answers match the L4 answers 50 times out of 50, and the 8-bit and 4-bit answers match
MLX bf16 50 times out of 50. On a Mac, quantisation saves memory (4-bit is 2.2 GB, against 8.7 GB
for bf16) but not time.

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
uv run python -m eval.analysis                 # CIs, breakdowns, thresholds -> artifacts/eval/analysis.json
```
