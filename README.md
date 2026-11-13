# nexus-decision-4b

An open-weight **typed decision model** with calibrated probabilities. You give it a
state and a set of typed questions (`choice`, `score`, `binary`). For each question it returns
one of the options you declared, a probability distribution over all of them, and a confidence
value. It is built for routing, triage, scoring and gating inside software, not for chat.

- **Base model:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0)
- **Teacher:** DeepSeek-V4.1-Flash through the DeepSeek API, as `deepseek-flash` (MIT-licensed
  weights)
- **Modes:**
  - **fast** (System One): one prefill per question, with no generation.
  - **think** (System Two): a short `<think>` trace first, then the same typed readout.
- **License:** Apache-2.0

> nexus-decision-4b follows the "System One decision model" interface popularised by TypeSafe AI's
> Jev. It is an independent project. It is not affiliated with TypeSafe AI and contains none of
> their weights, data or code.

## Quick start

```bash
uv sync --extra mlx          # Apple silicon
# or: uv sync --extra hf     # PyTorch (CUDA / MPS / CPU)
```

```python
from nexus_decision import NexusDecision

nd = NexusDecision.from_pretrained("<user>/nexus-decision-4b", backend="mlx")  # or backend="hf"

res = nd.decide({
    "state": {"channel": "email", "body": "I was billed twice for order #4471, please fix it today."},
    "questions": {
        "route": {"type": "choice", "instructions": "Which queue should handle this?",
                  "options": {"billing": "Payments, refunds, double charges",
                              "shipping": "Delivery problems",
                              "needs_review": "Anything else or unclear"}},
        "urgent": {"type": "binary", "instructions": "The customer needs a same-day response."},
    },
})
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
| `notebooks/` | Google Colab notebook for the GPU training run |
| `eval/` | Evaluation runs, temperature fitting and the metrics report |
| `docs/` | [protocol](docs/protocol.md), [data](docs/data.md), [training](docs/training.md), [evaluation](docs/evaluation.md) |

## Reproducing

```bash
cp .env.example .env                                   # add DEEPSEEK_API_KEY
uv sync --all-extras

# 1. data
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
```

Tests: `uv run pytest`.

## Attribution

- The base model is Qwen3.5-4B by the Qwen team (Apache-2.0).
- The teacher is DeepSeek-V4.1-Flash by DeepSeek (MIT).
- The public data is BANKING77 (CC-BY-4.0, PolyAI) and CLINC150 (CC-BY-3.0, Larson et al., 2019).
