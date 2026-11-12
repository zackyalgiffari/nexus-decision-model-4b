# Data

All training data is produced by `data/`. Each step is resumable, and every teacher response is
cached in `artifacts/cache/teacher.sqlite`, so re-running a step costs nothing.

## Sources

| Source | License | Use |
|---|---|---|
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 | Gold-labelled intent questions (train + held-out test) |
| [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) (`plus`) | CC-BY-3.0 | Gold-labelled intent questions, including out-of-scope |
| Synthetic scenarios written by the teacher | Released with the dataset | 12 decision domains (see `data/seeds/domains.yaml`) |

The teacher is DeepSeek-V4.1-Flash (`deepseek-flash` on the DeepSeek API; the model weights are
MIT-licensed).

## Pipeline

```
convert_public.py   public intents -> choice/binary questions with gold labels
gen_scenarios.py    teacher writes state + 1-4 typed questions per scenario
label_fast.py       teacher soft labels (non-thinking, top-20 logprobs, 3 option orders averaged)
label_think.py      teacher reasoning traces (thinking mode, effort=low) on a stratified subset
build_splits.py     join, split, render training records
```

### Public conversion

Each public question works as follows:

- **Choice questions** offer the gold intent plus 3-11 distractors. Half of the distractors are
  "hard" ones that share a word with the gold label. A `none_of_these` option is always present.
- **Abstain cases.** In 15% of in-scope cases the gold intent is withheld, so the answer is
  `none_of_these`. CLINC out-of-scope texts also answer `none_of_these`.
- **Binary questions** make up 20% and ask "the message is about X", with 50% positives.

### Synthetic scenarios

Each generation call samples:

- a domain, weighted;
- a difficulty: easy / medium / hard / ambiguous;
- a state format: JSON, free text, JSON with a long text field, or log lines;
- a language: 85% English, the rest Indonesian, Spanish, German, French, Portuguese or Japanese.

Scenarios are validated against the protocol schema. Near-duplicate states are removed by a
normalised hash, with digits and punctuation collapsed.

### Soft labels

For every question the teacher sees exactly the student prompt, with `max_tokens=1` and
`top_logprobs=20`. The first-token distribution is mapped onto option letters: `B`, ` B` and `B)`
all count as B. Mass outside the top 20 is spread over the unseen letters. This is repeated with
the canonical, reversed and one random option order, then averaged by option label.

### Think traces

These come from thinking mode with `reasoning_effort=low` and a seeded random option order. A trace
is kept only if both of these hold:

1. its final answer matches the argmax of the training target;
2. the reasoning is at most 6,000 characters.

### Training targets

| Data | Target |
|---|---|
| Synthetic | The teacher's soft distribution |
| Public | `0.5 * teacher + 0.5 * one_hot(gold)` |

### Splits

| Data | Rule |
|---|---|
| Public | The dataset's own test split becomes our test split, and is never used for training. Its train split goes 90% train / 10% calibration. |
| Synthetic | Split by scenario-ID hash: 90% train / 5% calibration / 5% test. |

All questions of a scenario stay in one split. The reference label for synthetic test items is the
teacher's argmax, so it is model-generated. Results on public data use human gold labels.

## Budget control

| Setting | Effect |
|---|---|
| `NEXUS_BUDGET_USD` / `--budget-usd` (default 18) | Hard cap. Calls stop once the ledger reaches it. |
| `--offpeak-only` | Waits for DeepSeek's off-peak window, when prices are half (see `data/teacher.py`). |

Spend per step: `sqlite3 artifacts/cache/teacher.sqlite "select step, round(sum(cost),4) from ledger group by step"`.
