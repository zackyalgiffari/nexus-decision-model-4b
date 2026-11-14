# nexus-decision-4b

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-nexus--decision--4b-yellow)](https://huggingface.co/zackyalgiffari/nexus-decision-4b)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-nexus--decision--data-yellow)](https://huggingface.co/datasets/zackyalgiffari/nexus-decision-data)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20collection-all%20variants-yellow)](https://huggingface.co/collections/zackyalgiffari/nexus-decision-4b-6af7fc5003be12bf9aa5b4c9)

An open-weight **typed decision model** with calibrated probabilities. You give it a
state and a set of typed questions (`choice`, `score`, `binary`). For each question it returns
one of the options you declared, a probability distribution over all of them, and a confidence
value. It is built for routing, triage, scoring and gating inside software, not for chat.

This repository holds the runtime package, the data and training pipeline, and the evaluation
code. The weights and the data are on the Hugging Face Hub.

- **Base model:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0)
- **Teacher:** DeepSeek-V4.1-Flash through the DeepSeek API, as `deepseek-flash` (MIT-licensed
  weights)
- **Modes:**
  - **fast** (System One): one prefill per question, with no generation.
  - **think** (System Two): a short `<think>` trace first, then the same typed readout.
- **License:** Apache-2.0 (code and weights), CC-BY-4.0 (data)

> nexus-decision-4b follows the "System One decision model" interface popularised by TypeSafe AI's
> Jev. It is an independent project. It is not affiliated with TypeSafe AI and contains none of
> their weights, data or code.

## Models and data

| Repo | Format | Size | Use it for |
|---|---|--:|---|
| [nexus-decision-4b](https://huggingface.co/zackyalgiffari/nexus-decision-4b) | Transformers, bf16 | 9.3 GB | CUDA, CPU, MPS; the reference weights |
| [nexus-decision-4b-MLX-8bit](https://huggingface.co/zackyalgiffari/nexus-decision-4b-MLX-8bit) | MLX, 8-bit | 4.5 GB | Apple silicon |
| [nexus-decision-4b-MLX-4bit](https://huggingface.co/zackyalgiffari/nexus-decision-4b-MLX-4bit) | MLX, 4-bit | 2.4 GB | Apple silicon with less memory |
| [nexus-decision-data](https://huggingface.co/datasets/zackyalgiffari/nexus-decision-data) | JSONL | 53 MB | Training and evaluation data |

Each model repo ships `nexus_config.json` with the fitted calibration temperatures, so the runtime
returns calibrated probabilities without extra setup. On 50 test questions the MLX 8-bit and 4-bit
variants gave the same answers as bf16.

## Results

Held-out test set (1,786 questions), fast mode, calibrated. "Public" is BANKING77 and CLINC150
rewritten as typed questions with their gold labels; "synthetic" is generated scenarios labelled
by the teacher.

| Model | Accuracy, all (95% CI) | Accuracy, public | ECE ↓ | Brier ↓ | Coverage @5% error | Order flip ↓ |
|---|--:|--:|--:|--:|--:|--:|
| **nexus-decision-4b** | **0.913** (0.899–0.926) | **0.915** | **0.017** | **0.126** | **0.923** | **0.036** |
| Qwen3.5-4B, same prompt | 0.847 (0.830–0.862) | 0.845 | 0.020 | 0.232 | 0.608 | 0.111 |
| DeepSeek-V4.1-Flash (teacher) | – | 0.892 | – | – | – | – |

Columns other than "Accuracy, public" cover all 1,786 questions. The teacher has no score there
because it wrote the labels for the synthetic half.

- Fine-tuning adds 6.6 accuracy points over the base model (paired 95% CI 5.2–8.1).
- On the public items, which have human gold labels, the student beats its teacher.
- The model card on the Hub has the rest: the full metric table, confidence thresholds,
  breakdowns by question type and domain, think mode, latency, and how to read these numbers.
  See [the Evaluation section](https://huggingface.co/zackyalgiffari/nexus-decision-4b#evaluation)
  and [docs/evaluation.md](docs/evaluation.md).

These results come from a single training run. They are not comparable to BANKING77 or CLINC150
leaderboards: each question shows the correct intent among 3 to 11 distractors plus
`none_of_these`, not all 77 or 150 intents.

## Quick start

Install the package straight from GitHub:

```bash
pip install "nexus-decision[mlx] @ git+https://github.com/zackyalgiffari/nexus-decision-model-4b"   # Apple silicon
pip install "nexus-decision[hf] @ git+https://github.com/zackyalgiffari/nexus-decision-model-4b"    # PyTorch
```

```python
from nexus_decision import NexusDecision

nd = NexusDecision.from_pretrained("zackyalgiffari/nexus-decision-4b", backend="mlx")  # or backend="hf"

request = {
    "state": {"channel": "email", "body": "I was billed twice for order #4471, please fix it today."},
    "questions": {
        "route": {"type": "choice", "instructions": "Which queue should handle this?",
                  "options": {"billing": "Payments, refunds, double charges",
                              "shipping": "Delivery problems",
                              "needs_review": "Anything else or unclear"}},
        "urgent": {"type": "binary", "instructions": "The customer needs a same-day response."},
    },
}
res = nd.decide(request)
for name, a in res.answers.items():
    print(name, a.choice, round(a.confidence, 3), a.probabilities)

# System Two for hard cases, and order-debiasing for fast mode:
nd.decide(request, mode="think", max_think_tokens=768)
nd.decide(request, mode="fast", debias=True)
```

Read [docs/protocol.md](docs/protocol.md) before using this in production. It covers the request
and response schema, the raw prompt format for other runtimes, and the limitations.

## Repository layout

| Path | What it holds |
|---|---|
| `src/nexus_decision/` | Runtime: schema, prompt rendering, MLX and Transformers backends, calibration, metrics |
| `data/` | Teacher pipeline: public conversion, synthetic scenarios, soft labels, think traces, splits |
| `train/` | LoRA distillation: PyTorch/PEFT for CUDA (`train_torch.py`) and MLX (`train_mlx.py`). Export to a full Hugging Face checkpoint (`fuse_export.py`). |
| `notebooks/` | Google Colab notebooks for the GPU training and evaluation runs |
| `eval/` | Evaluation runs, temperature fitting, the metrics report and the analysis behind the model card |
| `release/` | Model card generator and the Hugging Face upload scripts |
| `docs/` | [protocol](docs/protocol.md), [data](docs/data.md), [training](docs/training.md), [evaluation](docs/evaluation.md) |

## Reproducing

```bash
git clone https://github.com/zackyalgiffari/nexus-decision-model-4b && cd nexus-decision-model-4b
cp .env.example .env                                   # add DEEPSEEK_API_KEY
uv sync --all-extras

# 1. data (or download it: huggingface.co/datasets/zackyalgiffari/nexus-decision-data)
uv run python -m data.convert_public
uv run python -m data.gen_scenarios --calls 900 --offpeak-only
uv run python -m data.label_fast --orders 3 --offpeak-only
uv run python -m data.label_think --n 4000 --offpeak-only
uv run python -m data.build_splits

# 2. training on a CUDA GPU (Colab L4 or A100: notebooks/train_colab.ipynb)
scripts/pack_colab.sh                                   # upload the zip to Google Drive
#    ...run the notebook, then download adapters/run1/final -> artifacts/adapters/colab/final

# 3. export, calibration and evaluation
uv run python train/fuse_export.py --adapters artifacts/adapters/colab/final \
    --out artifacts/fused/nexus-decision-4b
uv run python -m eval.run_eval --name tuned --model artifacts/fused/nexus-decision-4b --split calib
uv run python -m eval.run_eval --name tuned --model artifacts/fused/nexus-decision-4b --split test
uv run python -m eval.report --name tuned              # writes artifacts/eval/nexus_config.json
uv run python -m eval.analysis                         # CIs, thresholds and breakdowns for the card
```

The data pipeline cost about $4 in DeepSeek API calls. Training took about 5.5 hours on one
NVIDIA L4. [docs/training.md](docs/training.md) and [docs/evaluation.md](docs/evaluation.md) have
the details.

Development: `uv run ruff check .` and `uv run pytest`.

## Citation

```bibtex
@misc{nexus-decision-4b,
  title  = {nexus-decision-4b: a calibrated typed decision model distilled from DeepSeek-V4.1-Flash},
  author = {zackyalgiffari},
  year   = {2026},
  url    = {https://huggingface.co/zackyalgiffari/nexus-decision-4b}
}
```

GitHub also offers this through "Cite this repository", from [CITATION.cff](CITATION.cff).

## Attribution

- The base model is Qwen3.5-4B by the Qwen team (Apache-2.0).
- The teacher is DeepSeek-V4.1-Flash by DeepSeek (MIT).
- The public data is BANKING77 (CC-BY-4.0, PolyAI) and CLINC150 (CC-BY-3.0, Larson et al., 2019).
