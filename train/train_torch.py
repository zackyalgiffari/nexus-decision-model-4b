"""LoRA distillation of nexus-decision-4b with PyTorch + PEFT (CUDA, e.g. Google Colab A100/L4).

Same objective as train/train_mlx.py:

    kl_weight * KL(target || softmax(letter logits at the answer position))
  + lm_weight * CE(reasoning + "</think>\\n\\n" + letter + "<|im_end|>")      (think records only)

Only the answer positions and think completions go through the LM head, so the 248k-token
vocabulary is never materialised for prompt tokens.

Install ``flash-linear-attention`` on CUDA. Without it, transformers runs Qwen3.5's Gated
DeltaNet through a slow reference path.

Usage:
    python train/train_torch.py --config train/configs/colab_a100.yaml
    python train/train_torch.py --config ... --resume          # continue from out_dir/last
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nexus_decision.encode import Encoder, collate_numpy, load_records, make_batches

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",                              # full attention
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",    # Gated DeltaNet
    "gate_proj", "up_proj", "down_proj",                                 # MLP
]

DEFAULTS = {
    "model": "Qwen/Qwen3.5-4B",
    "train_file": "artifacts/data/train.jsonl",
    "val_file": "artifacts/data/val.jsonl",
    "out_dir": "artifacts/adapters/torch",
    "max_examples": None,
    "num_layers": 32,          # LoRA on the last N decoder layers
    "rank": 16,
    "alpha": 32,
    "dropout": 0.0,
    "lr": 1.0e-4,
    "warmup_steps": 50,
    "epochs": 2,
    "batch_size": 8,
    "think_batch_size": 2,
    "grad_accum": 2,
    "max_seq_len": 2048,
    "kl_weight": 1.0,
    "lm_weight": 0.5,
    "grad_checkpoint": True,
    "save_every": 200,
    "eval_every": 200,
    "val_max_examples": 300,
    "report_every": 10,
    "seed": 0,
    "dtype": "bfloat16",
}


def to_torch(c: dict, device) -> dict:
    out = {k: torch.from_numpy(c[k]).to(device) for k in
           ("tokens", "attention_mask", "letters", "valid", "target", "ans_pos")}
    out["lm"] = None if c["lm"] is None else tuple(torch.from_numpy(x).to(device) for x in c["lm"])
    out["n_tokens"] = c["n_tokens"]
    return out


def backbone_and_head(model):
    """Return (decoder, lm_head) of a (PEFT-wrapped) Qwen3_5ForCausalLM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def letter_logp(decoder, head, b):
    h = decoder(input_ids=b["tokens"], attention_mask=b["attention_mask"]).last_hidden_state
    B = h.shape[0]
    h_ans = h[torch.arange(B, device=h.device), b["ans_pos"]]
    z = head(h_ans).float().gather(-1, b["letters"])
    z = z.masked_fill(~b["valid"], -1e9)
    return h, z.log_softmax(-1)


def chunked_lm_ce(head, hs, tgt, chunk: int = 256):
    """Mean token cross-entropy without materialising all [N, vocab] logits at once.

    Each chunk's logits are recomputed in backward (activation checkpointing), so peak memory is
    one chunk of logits (~254 MB for 256 tokens x 248k vocab in fp32) instead of the whole trace.
    """
    from torch.utils.checkpoint import checkpoint

    def chunk_sum(h_, t_):
        return F.cross_entropy(head(h_).float(), t_, reduction="sum")

    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        total = total + checkpoint(chunk_sum, hs[i : i + chunk], tgt[i : i + chunk], use_reentrant=False)
    return total / hs.shape[0]


def compute_loss(decoder, head, b, kl_w: float, lm_w: float):
    h, logp = letter_logp(decoder, head, b)
    t = b["target"]
    kl = torch.where(t > 0, t * (t.clamp_min(1e-12).log() - logp), torch.zeros_like(t)).sum(-1).mean()
    loss = kl_w * kl
    ce = torch.zeros((), device=h.device)
    if b["lm"] is not None:
        rows, pos, tgt = b["lm"]
        ce = chunked_lm_ce(head, h[rows, pos], tgt)
        loss = loss + lm_w * ce
    return loss, kl.detach(), ce.detach()


@torch.no_grad()
def evaluate(model, decoder, head, val: list[dict], device, bs: int = 16) -> dict:
    model.eval()
    nll, correct = 0.0, 0
    for s in range(0, len(val), bs):
        exs = val[s : s + bs]
        b = to_torch(collate_numpy(exs), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, logp = letter_logp(decoder, head, b)
        logp = logp.float().cpu().numpy()
        for i, ex in enumerate(exs):
            n = len(ex["letters"])
            nll += float(-(ex["target"] * logp[i, :n]).sum())
            correct += int(np.argmax(logp[i, :n]) == np.argmax(ex["target"]))
    model.train()
    n = max(1, len(val))
    return {"val_soft_ce": nll / n, "val_agree": correct / n}


def save(model, opt, sched, out_dir: Path, cfg: dict, step: int, tag: str) -> None:
    d = out_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, d / "trainer_state.pt")
    (d / "nexus_train.json").write_text(json.dumps({"step": step, "config": cfg}, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/last")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (YAML values)")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(args.config).read_text()) or {})
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {cfg['model']} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]))
    model.to(device)
    model.config.use_cache = False
    n_layers = model.config.num_hidden_layers
    layers = list(range(max(0, n_layers - cfg["num_layers"]), n_layers))

    last = out_dir / "last"
    start_step = 0
    if args.resume and (last / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, last, is_trainable=True)
        start_step = torch.load(last / "trainer_state.pt", weights_only=False)["step"]
        print(f"resuming from step {start_step}", flush=True)
    else:
        lcfg = LoraConfig(
            r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
            target_modules=LORA_TARGETS, layers_to_transform=layers, layers_pattern="layers",
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
    if cfg["grad_checkpoint"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()
    decoder, head = backbone_and_head(model)

    enc = Encoder(tok, cfg["max_seq_len"])

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    train = [e for e in map(enc.encode, load_records(resolve(cfg["train_file"]), cfg["max_examples"])) if e]
    val = []
    if cfg.get("val_file") and resolve(cfg["val_file"]).exists():
        val = [e for e in map(enc.encode, load_records(resolve(cfg["val_file"]), cfg["val_max_examples"])) if e]
    n_think = sum(e["lm_start"] is not None for e in train)
    print(f"train: {len(train)} ({n_think} think)  val: {len(val)}", flush=True)

    rng = random.Random(cfg["seed"])
    all_batches = [bt for _ in range(cfg["epochs"])
                   for bt in make_batches(train, cfg["batch_size"], cfg["think_batch_size"], rng)]
    total_steps = math.ceil(len(all_batches) / cfg["grad_accum"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, min(cfg["warmup_steps"], total_steps // 10 + 1), total_steps)
    if start_step:
        st = torch.load(last / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
    print(f"batches: {len(all_batches)}  optimizer steps: {total_steps}", flush=True)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log_path = out_dir / "train_log.jsonl"

    step = start_step
    micro = 0
    t0, toks = time.time(), 0
    run = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}
    opt.zero_grad(set_to_none=True)
    for bidx in all_batches[start_step * cfg["grad_accum"]:]:
        b = to_torch(collate_numpy([train[i] for i in bidx]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, kl, ce = compute_loss(decoder, head, b, cfg["kl_weight"], cfg["lm_weight"])
        (loss / cfg["grad_accum"]).backward()
        micro += 1
        toks += b["n_tokens"]
        run["loss"] += loss.item()
        run["kl"] += kl.item()
        run["ce"] += ce.item()
        run["n"] += 1
        if micro % cfg["grad_accum"]:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg["report_every"] == 0:
            dt = time.time() - t0
            rec = {
                "step": step, "total_steps": total_steps,
                **{k: run[k] / run["n"] for k in ("loss", "kl", "ce")},
                "lr": sched.get_last_lr()[0], "tok_per_s": toks / dt,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
                "eta_h": (total_steps - step) * dt / cfg["report_every"] / 3600,
            }
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            t0, toks = time.time(), 0
            run = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}
        if val and step % cfg["eval_every"] == 0:
            ev = {"step": step, **evaluate(model, decoder, head, val, device)}
            print(json.dumps(ev), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(ev) + "\n")
        if step % cfg["save_every"] == 0:
            save(model, opt, sched, out_dir, cfg, step, "last")
        if args.max_steps and step >= args.max_steps:
            break

    save(model, opt, sched, out_dir, cfg, step, "last")
    save(model, opt, sched, out_dir, cfg, step, "final")
    if val:
        ev = {"step": step, "final": True, **evaluate(model, decoder, head, val, device)}
        print(json.dumps(ev), flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(ev) + "\n")
    print(f"done: {step} steps -> {out_dir / 'final'}", flush=True)


if __name__ == "__main__":
    main()
