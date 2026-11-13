# Training

The student is `Qwen/Qwen3.5-4B` (Apache-2.0), fine-tuned with LoRA. There are two trainers with
the same objective and the same tokenisation (`src/nexus_decision/encode.py`):

| Trainer | Hardware | Notes |
|---|---|---|
| `train/train_torch.py` (PyTorch + PEFT) | CUDA GPU, e.g. a Colab A100 or L4 | Used for the released model. Needs `flash-linear-attention` for fast Gated DeltaNet kernels. See `notebooks/train_colab.ipynb`. |
| `train/train_mlx.py` (MLX) | Apple silicon | Good for small runs and experiments. |

### Why the released run used a GPU

Qwen3.5 is a hybrid model. Three of every four decoder blocks use Gated DeltaNet linear
attention.

- **The problem on Apple silicon.** mlx-lm 0.32's fast Metal kernel for that recurrence has no
  backward pass, so training falls back to a per-token loop. On an M4 (24 GB) this gave about
  12 tok/s.
- **The partial fix.** `train/mlx_patches.py` vendors the chunk-parallel form from the open
  upstream PR ml-explore/mlx-lm#2021. That raises it to about 45 tok/s with LoRA on 24 layers,
  or about 70 tok/s on 12 layers. That is still too slow for a dataset of about 13M tokens.
- **The GPU path.** On CUDA, `flash-linear-attention` provides fused kernels. The released run
  took about 5.5 hours on a single Colab L4.

## Objective

Every training record is one rendered question. Its loss is:

```
loss = kl_weight * KL(target || softmax(letter logits at the answer position))
     + lm_weight * CE(reasoning + "</think>\n\n" + letter + "<|im_end|>")      # think records only
```

Only the answer position (and, for think records, the completion) is projected through the LM
head. The 248k-token vocabulary never gets materialised for the prompt tokens.

## Hyper-parameters

The released run used `train/configs/colab_l4.yaml`:

| | |
|---|---|
| LoRA layers | All 32 decoder blocks, all linear projections (attention, Gated DeltaNet, MLP) |
| Rank / alpha / dropout | 16 / 32 / 0 |
| Trainable parameters | 32.5M (0.77% of 4.24B) |
| Optimiser | AdamW, lr 1e-4, 50 warm-up steps, cosine decay, grad-norm clip 1.0 |
| Batch | 4 fast records (or 1 think record) × 4 accumulation steps, length-grouped |
| Epochs | 2 |
| Max length | 2,048 tokens |
| Loss weights | KL 1.0, LM 0.5 |
| Precision / memory | bf16, gradient checkpointing on the decoder blocks |

`train/configs/colab_a100.yaml` has the same settings and the same effective batch, sized for
40–80 GB: 8 fast records (or 2 think records) × 2 accumulation steps.

## Released run

| | |
|---|---|
| Hardware | 1× NVIDIA L4 24 GB (Google Colab), PyTorch + PEFT |
| Data | 16,340 training records (4,004 with think traces), 300 validation items |
| Steps | 3,544 optimiser steps over 2 epochs |
| Time | About 5.5 h |
| Throughput | About 870 tok/s (median) |
| Peak memory | 13.2 GB |
| Final validation | `val_soft_ce` 0.264, `val_agree` 0.917 |

Validation was measured every 200 steps on 300 items:

| Step | 200 | 600 | 1,000 | 1,400 | 2,000 | 2,400 | 2,800 | 3,200 | 3,544 |
|---|---|---|---|---|---|---|---|---|---|
| `val_soft_ce` | 0.360 | 0.375 | 0.323 | 0.308 | 0.276 | 0.279 | 0.267 | 0.264 | 0.264 |
| `val_agree` | 0.850 | 0.877 | 0.887 | 0.890 | 0.903 | 0.910 | 0.907 | 0.917 | 0.917 |

Both curves were nearly flat over the last 600 steps, while the cosine schedule took the learning
rate to zero. Colab had no
`causal_conv1d` installed, so the Gated DeltaNet short convolution ran on its reference PyTorch
path. Installing it should make the run faster.

For test-set results see [evaluation.md](evaluation.md).

## Running

On Colab, follow `notebooks/train_colab.ipynb`. It needs `scripts/pack_colab.sh` to build the
upload. On any CUDA machine:

```bash
python train/train_torch.py --config train/configs/colab_l4.yaml     # or colab_a100.yaml; add --resume to continue
```

On Apple silicon:

```bash
caffeinate -i uv run python train/train_mlx.py --config train/configs/full.yaml
# resume after an interruption:
uv run python train/train_mlx.py --config train/configs/full.yaml \
    --resume artifacts/adapters/full/adapters.safetensors --resume-step <step>
```

`train_log.jsonl` in the output directory records the following:

| Field | Meaning |
|---|---|
| `loss`, `kl`, `ce` | Training losses |
| `tok_per_s` | Training throughput |
| `peak_mem_gb` | Peak memory |
| `eta_h` | Estimated hours remaining |
| `val_soft_ce`, `val_agree` | Validation metrics |

## Export

`fuse_export.py` accepts both adapter formats: PEFT (`adapter_model.safetensors`) and MLX
(`adapters.safetensors`).

```bash
uv run python train/fuse_export.py --adapters artifacts/adapters/colab/final \
    --out artifacts/fused/nexus-decision-4b --config artifacts/eval/nexus_config.json
```

The LoRA deltas are patched straight into the original Qwen3.5 safetensors shards, so the vision
tower, the MTP head and the config stay intact. The result loads with stock `transformers`
(`Qwen3_5ForConditionalGeneration`). MLX quantisations are made from the fused checkpoint with
`mlx_lm.convert`.
