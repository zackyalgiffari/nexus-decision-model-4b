# Training

The student is `Qwen/Qwen3.5-4B` (Apache-2.0), fine-tuned with LoRA in MLX on Apple silicon
(`train/train_mlx.py`).

## Objective

Every training record is one rendered question. Its loss is:

```
loss = kl_weight * KL(target || softmax(letter logits at the answer position))
     + lm_weight * CE(reasoning + "</think>\n\n" + letter + "<|im_end|>")      # think records only
```

Only the answer position (and, for think records, the completion) is projected through the LM
head. The 248k-token vocabulary never gets materialised for the prompt tokens.

## Default hyper-parameters (`train/configs/full.yaml`)

| | |
|---|---|
| LoRA layers | Last 24 of 32 decoder blocks, all linear projections (attention, Gated DeltaNet, MLP) |
| Rank / scale | 16 / 2.0 (equivalent to alpha = 32) |
| Optimiser | AdamW, lr 1e-4, 50 warm-up steps, cosine decay to 1e-5 |
| Batch | 1 sequence × 8 gradient-accumulation steps |
| Epochs | 2 |
| Max length | 2,048 tokens |
| Loss weights | KL 1.0, LM 0.5 |
| Memory | Gradient checkpointing on the decoder blocks |

## Running

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

```bash
uv run python train/fuse_export.py --adapters artifacts/adapters/full \
    --out artifacts/fused/nexus-decision-4b --config artifacts/eval/nexus_config.json
```

The LoRA deltas are patched straight into the original Qwen3.5 safetensors shards, so the vision
tower, the MTP head and the config stay intact. The result loads with stock `transformers`
(`Qwen3_5ForConditionalGeneration`). MLX quantisations are made from the fused checkpoint with
`mlx_lm.convert`.
