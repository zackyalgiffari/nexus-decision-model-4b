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
