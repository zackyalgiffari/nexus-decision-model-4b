"""LoRA distillation of nexus-decision-4b with MLX.

Each training record (JSONL, produced by data/build_splits.py) is one rendered question:

    {"id": str, "kind": "fast" | "think", "user_text": str, "labels": [str, ...],
     "target": [float, ...],             # teacher distribution over the displayed letters
     "reasoning": str | null}            # think traces only

Loss per record:
    kl_weight * KL(target || softmax(letter logits at the answer position))
  + lm_weight * token cross-entropy over "<reasoning></think>\\n\\n<letter><|im_end|>" (think only)

Only the answer position (and the think completion) goes through the LM head, so memory stays
flat in sequence length apart from the backbone activations.

Usage:
    uv run python train/train_mlx.py --config train/configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nexus_decision.prompt import close_think, generation_prefix
from nexus_decision.runtime import letter_token_ids
from nexus_decision.schema import LETTERS
from train import mlx_patches

DEFAULTS = {
    "model": "Qwen/Qwen3.5-4B",
    "train_file": "artifacts/data/train.jsonl",
    "val_file": "artifacts/data/val.jsonl",
    "out_dir": "artifacts/adapters/run",
    "max_examples": None,
    "num_layers": 24,
    "rank": 16,
    "scale": 2.0,
    "dropout": 0.0,
    "lr": 1.0e-4,
    "warmup_steps": 50,
    "epochs": 2,
    "batch_size": 4,
    "think_batch_size": 1,
    "grad_accum": 2,
    "max_seq_len": 2048,
    "kl_weight": 1.0,
    "lm_weight": 0.5,
    "grad_checkpoint": True,
    "save_every": 250,
    "eval_every": 250,
    "val_max_examples": 200,
    "report_every": 10,
    "seed": 0,
}


# --------------------------------------------------------------------------- data
class Encoder:
    def __init__(self, tokenizer, max_seq_len: int):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.letter_ids = letter_token_ids(tokenizer)
        self.eot = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def encode(self, rec: dict) -> dict | None:
        n = len(rec["labels"])
        target = np.asarray(rec["target"], dtype=np.float32)
        target = target / target.sum()
        if rec["kind"] == "fast" or not rec.get("reasoning"):
            prefix = generation_prefix(self.tok, rec["user_text"], "fast")
            toks = self.tok.encode(prefix, add_special_tokens=False)
            ans_pos = len(toks) - 1  # logits at the last prompt token predict the letter
            lm_start = None
        else:
            prefix = generation_prefix(self.tok, rec["user_text"], "think")
            p_toks = self.tok.encode(prefix, add_special_tokens=False)
            body = close_think(rec["reasoning"])
            b_toks = self.tok.encode(body, add_special_tokens=False)
            letter = LETTERS[int(np.argmax(target))]
            toks = p_toks + b_toks + [self.letter_ids[LETTERS.index(letter)]] + self.eot
            ans_pos = len(p_toks) + len(b_toks) - 1
            lm_start = len(p_toks)
        if len(toks) > self.max_seq_len:
            return None
        return {
            "tokens": np.asarray(toks, dtype=np.int32),
            "ans_pos": ans_pos,
            "letters": np.asarray(self.letter_ids[:n], dtype=np.int32),
            "target": target,
            "lm_start": lm_start,
        }


def load_records(path: str, limit: int | None) -> list[dict]:
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return recs[:limit] if limit else recs


# --------------------------------------------------------------------------- batching
def make_batches(train: list[dict], batch_size: int, think_batch_size: int, rng: random.Random):
    """Homogeneous batches (fast-only or think-only), grouped by length to limit padding."""
    batches = []
    for kind, bs in (("fast", batch_size), ("think", think_batch_size)):
        idx = [i for i, ex in enumerate(train) if (ex["lm_start"] is None) == (kind == "fast")]
        rng.shuffle(idx)
        mega = bs * 32
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s : s + mega], key=lambda i: len(train[i]["tokens"]))
            batches += [chunk[j : j + bs] for j in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def collate(exs: list[dict]) -> dict:
    """Right-pad a batch. Causal attention means padding never affects earlier positions, so no
    mask is needed (and the chunked Gated DeltaNet path requires mask=None)."""
    B = len(exs)
    L = max(len(e["tokens"]) for e in exs)
    tokens = np.zeros((B, L), dtype=np.int32)
    letters = np.zeros((B, len(LETTERS)), dtype=np.int32)
    valid = np.zeros((B, len(LETTERS)), dtype=bool)
    target = np.zeros((B, len(LETTERS)), dtype=np.float32)
    ans_pos = np.zeros(B, dtype=np.int32)
    rows, pos, tgt = [], [], []
    for b, e in enumerate(exs):
        n = len(e["letters"])
        tokens[b, : len(e["tokens"])] = e["tokens"]
        letters[b, :n] = e["letters"]
        valid[b, :n] = True
        target[b, :n] = e["target"]
        ans_pos[b] = e["ans_pos"]
        if e["lm_start"] is not None:
            p = np.arange(e["lm_start"] - 1, len(e["tokens"]) - 1)
            rows += [b] * len(p)
            pos += p.tolist()
            tgt += e["tokens"][p + 1].tolist()
    out = {
        "tokens": mx.array(tokens), "letters": mx.array(letters), "valid": mx.array(valid),
        "target": mx.array(target), "ans_pos": mx.array(ans_pos), "n_tokens": int(sum(len(e["tokens"]) for e in exs)),
        "lm": None,
    }
    if rows:
        out["lm"] = (mx.array(np.array(rows, np.int32)), mx.array(np.array(pos, np.int32)),
                     mx.array(np.array(tgt, np.int32)))
    return out


# --------------------------------------------------------------------------- model helpers
def hidden_and_head(model):
    """Return (backbone, head_fn) so we can project only selected positions."""
    lm = model.language_model if hasattr(model, "language_model") else model
    backbone = lm.model
    if lm.args.tie_word_embeddings:
        head = backbone.embed_tokens.as_linear
    else:
        head = lm.lm_head
    return backbone, head


def letter_logp(backbone, head, tokens, ans_pos, letters, valid):
    h = backbone(tokens)  # (B, L, H)
    B = tokens.shape[0]
    h_ans = h[mx.arange(B), ans_pos]  # (B, H)
    z = mx.take_along_axis(head(h_ans), letters, axis=-1).astype(mx.float32)  # (B, 26)
    z = mx.where(valid, z, -1e9)
    return h, z - mx.logsumexp(z, axis=-1, keepdims=True)


def make_loss(model, kl_w: float, lm_w: float):
    backbone, head = hidden_and_head(model)

    def loss_fn(model, tokens, ans_pos, letters, valid, target, lm):
        h, logp = letter_logp(backbone, head, tokens, ans_pos, letters, valid)
        t = target
        kl_rows = mx.sum(mx.where(t > 0, t * (mx.log(mx.maximum(t, 1e-12)) - logp), 0.0), axis=-1)
        kl = kl_rows.mean()
        loss = kl_w * kl
        ce = mx.array(0.0)
        if lm is not None:
            rows, pos, tgt = lm
            logits = head(h[rows, pos]).astype(mx.float32)
            ce = nn.losses.cross_entropy(logits, tgt).mean()
            loss = loss + lm_w * ce
        return loss, (kl, ce)

    return loss_fn


def evaluate(model, encoded: list[dict], batch_size: int = 8) -> dict:
    model.eval()
    backbone, head = hidden_and_head(model)
    nll, correct = 0.0, 0
    for s in range(0, len(encoded), batch_size):
        exs = encoded[s : s + batch_size]
        b = collate(exs)
        _, logp = letter_logp(backbone, head, b["tokens"], b["ans_pos"], b["letters"], b["valid"])
        logp = np.array(logp)
        for i, ex in enumerate(exs):
            n = len(ex["letters"])
            nll += float(-(ex["target"] * logp[i, :n]).sum())
            correct += int(np.argmax(logp[i, :n]) == np.argmax(ex["target"]))
    model.train()
    n = max(1, len(encoded))
    return {"val_soft_ce": nll / n, "val_agree": correct / n}


def save_adapters(model, out_dir: Path, cfg: dict, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    mx.save_safetensors(str(out_dir / f"{step:07d}_adapters.safetensors"), weights)
    adapter_cfg = {
        "fine_tune_type": "lora",
        "num_layers": cfg["num_layers"],
        "lora_parameters": {"rank": cfg["rank"], "scale": cfg["scale"], "dropout": cfg["dropout"]},
        "base_model": cfg["model"],
        "step": step,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="adapters.safetensors to resume from")
    ap.add_argument("--resume-step", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (spike/bench)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (YAML values)")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    mx.random.seed(cfg["seed"])
    out_dir = ROOT / cfg["out_dir"]

    mlx_patches.apply()
    print(f"loading {cfg['model']}", flush=True)
    model, tok = load(cfg["model"])
    model.freeze()
    linear_to_lora_layers(
        model, cfg["num_layers"],
        {"rank": cfg["rank"], "scale": cfg["scale"], "dropout": cfg["dropout"]},
    )
    if args.resume:
        model.load_weights(args.resume, strict=False)
    model.train()  # Gated DeltaNet: use the differentiable ops path, not the Metal kernel
    if cfg["grad_checkpoint"]:
        grad_checkpoint(model.layers[0])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.2f}M", flush=True)

    enc = Encoder(tok, cfg["max_seq_len"])
    train = [e for e in map(enc.encode, load_records(cfg["train_file"], cfg["max_examples"])) if e]
    val = []
    if cfg.get("val_file") and Path(ROOT / cfg["val_file"]).exists():
        val = [e for e in map(enc.encode, load_records(ROOT / cfg["val_file"], cfg["val_max_examples"])) if e]
    print(f"train examples: {len(train)}  val examples: {len(val)}", flush=True)

    rng = random.Random(cfg["seed"])
    epoch_batches = [make_batches(train, cfg["batch_size"], cfg["think_batch_size"], rng)
                     for _ in range(cfg["epochs"])]
    all_batches = [bt for ep in epoch_batches for bt in ep]
    total_steps = math.ceil(len(all_batches) / cfg["grad_accum"])
    warm = max(1, min(cfg["warmup_steps"], total_steps - 1))
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-7, cfg["lr"], warm),
         optim.cosine_decay(cfg["lr"], max(1, total_steps - warm), cfg["lr"] * 0.1)],
        [warm],
    )
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, make_loss(model, cfg["kl_weight"], cfg["lm_weight"]))

    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))

    step = args.resume_step
    micro = 0
    acc_grads = None
    t0, toks_seen = time.time(), 0
    run_loss, run_kl, run_ce, run_n = 0.0, 0.0, 0.0, 0
    print(f"batches: {len(all_batches)}  optimizer steps: {total_steps}", flush=True)

    for bidx in all_batches[args.resume_step * cfg["grad_accum"]:]:
        bt = collate([train[i] for i in bidx])
        (loss, (kl, ce)), grads = loss_and_grad(
            model, bt["tokens"], bt["ans_pos"], bt["letters"], bt["valid"], bt["target"], bt["lm"],
        )
        acc_grads = grads if acc_grads is None else nn.utils.tree_map(mx.add, acc_grads, grads)
        mx.eval(acc_grads, loss)
        micro += 1
        toks_seen += bt["n_tokens"]
        run_loss += loss.item()
        run_kl += kl.item()
        run_ce += ce.item()
        run_n += 1
        if micro % cfg["grad_accum"]:
            continue
        acc_grads = nn.utils.tree_map(lambda g: g / cfg["grad_accum"], acc_grads)
        opt.update(model, acc_grads)
        mx.eval(model.trainable_parameters(), opt.state)
        acc_grads = None
        step += 1

        if step % cfg["report_every"] == 0:
            dt = time.time() - t0
            rec = {
                "step": step, "total_steps": total_steps, "loss": run_loss / run_n,
                "kl": run_kl / run_n, "ce": run_ce / run_n, "lr": float(opt.learning_rate),
                "tok_per_s": toks_seen / dt, "peak_mem_gb": mx.get_peak_memory() / 1e9,
                "eta_h": (total_steps - step) * (dt / cfg["report_every"]) / 3600,
            }
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            t0, toks_seen = time.time(), 0
            run_loss, run_kl, run_ce, run_n = 0.0, 0.0, 0.0, 0
        if val and step % cfg["eval_every"] == 0:
            ev = {"step": step, **evaluate(model, val)}
            print(json.dumps(ev), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(ev) + "\n")
        if step % cfg["save_every"] == 0:
            save_adapters(model, out_dir, cfg, step)
        if args.max_steps and step >= args.max_steps:
            break

    save_adapters(model, out_dir, cfg, step)
    if val:
        print(json.dumps({"step": step, "final": True, **evaluate(model, val)}), flush=True)
    print(f"done: {step} steps -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
