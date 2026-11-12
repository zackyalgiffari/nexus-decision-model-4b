"""Merge MLX LoRA adapters into the *original* Hugging Face checkpoint.

mlx-lm drops the vision tower and MTP head of Qwen3.5 when it loads the model, so `mlx_lm.fuse`
would produce a text-only checkpoint with MLX-specific weight layouts. Instead we patch the LoRA
deltas straight into the original safetensors shards, which keeps every other tensor (vision
tower, MTP head, norms, conv kernels) byte-identical and keeps the original config, so the result
loads with the stock `transformers` Qwen3.5 classes.

    delta W = scale * lora_b^T @ lora_a^T        (same formula as mlx_lm LoRALinear.fuse)

MLX key  language_model.model.layers.N.<path>.lora_a
HF key   model.language_model.layers.N.<path>.weight

Usage:
    uv run python train/fuse_export.py --adapters artifacts/adapters/full \
        --out artifacts/fused/nexus-decision-4b [--config artifacts/eval/nexus_config.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Files copied verbatim from the base repo (weights are rewritten separately).
COPY = [
    "config.json", "chat_template.jinja", "merges.txt", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "preprocessor_config.json", "video_preprocessor_config.json",
    "model.safetensors.index.json", "generation_config.json",
]


def mlx_to_hf(key: str) -> str:
    prefix = "language_model.model."
    if not key.startswith(prefix):
        raise KeyError(f"unexpected adapter key {key}")
    return "model.language_model." + key[len(prefix):]


def peft_to_hf(module: str) -> str:
    """base_model.model.model.layers.N.<path>  ->  model.language_model.layers.N.<path>"""
    prefix = "base_model.model.model."
    if not module.startswith(prefix):
        raise KeyError(f"unexpected PEFT key {module}")
    return "model.language_model." + module[len(prefix):]


def load_peft_deltas(adapter_dir: Path, cfg: dict) -> dict[str, torch.Tensor]:
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        raise ValueError("only plain LoRA adapters are supported")
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open has no __iter__)
            module, leaf = k.rsplit(".", 2)[0], k.rsplit(".", 2)[1]  # ...lora_A.weight
            pairs[module][leaf] = f.get_tensor(k).float()
    deltas = {}
    for module, ab in pairs.items():
        if set(ab) != {"lora_A", "lora_B"}:
            raise ValueError(f"{module}: expected lora_A/lora_B, got {sorted(ab)}")
        deltas[peft_to_hf(module) + ".weight"] = scale * ab["lora_B"] @ ab["lora_A"]
    return deltas


def load_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Load LoRA deltas from an MLX (train_mlx.py) or PEFT (train_torch.py) adapter directory."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if (adapter_dir / "adapter_model.safetensors").exists():
        return load_peft_deltas(adapter_dir, cfg)
    scale = float(cfg["lora_parameters"]["scale"])
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(str(adapter_dir / "adapters.safetensors"), framework="pt") as f:
        for k in f.keys():  # noqa: SIM118
            module, leaf = k.rsplit(".", 1)
            pairs[module][leaf] = f.get_tensor(k).float()
    deltas = {}
    for module, ab in pairs.items():
        if set(ab) != {"lora_a", "lora_b"}:
            raise ValueError(f"{module}: expected lora_a/lora_b, got {sorted(ab)}")
        deltas[mlx_to_hf(module) + ".weight"] = scale * ab["lora_b"].T @ ab["lora_a"].T
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--config", default=None, help="nexus_config.json to ship with the model")
    args = ap.parse_args()

    base = Path(snapshot_download(args.base))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deltas = load_deltas(Path(args.adapters))
    print(f"{len(deltas)} LoRA deltas to merge")

    index = json.loads((base / "model.safetensors.index.json").read_text())
    missing = [k for k in deltas if k not in index["weight_map"]]
    if missing:
        raise KeyError(f"{len(missing)} adapter targets not in base checkpoint, e.g. {missing[:3]}")

    merged = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(str(base / shard))
        for k in list(tensors):
            if k in deltas:
                w = tensors[k]
                d = deltas[k]
                if d.shape != w.shape:
                    raise ValueError(f"{k}: delta {tuple(d.shape)} vs weight {tuple(w.shape)}")
                tensors[k] = (w.float() + d).to(w.dtype)
                merged += 1
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"wrote {shard}")
    if merged != len(deltas):
        raise RuntimeError(f"merged {merged} of {len(deltas)} deltas")

    for name in COPY:
        if (base / name).exists():
            shutil.copyfile(base / name, out / name)
    if args.config:
        shutil.copyfile(args.config, out / "nexus_config.json")
    print(f"merged {merged} weights -> {out}")


if __name__ == "__main__":
    main()
