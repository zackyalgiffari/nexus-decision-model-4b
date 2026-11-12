"""Tokenisation of training records, shared by the MLX and PyTorch trainers.

A record (JSONL, produced by data/build_splits.py) is one rendered question:

    {"id": str, "kind": "fast" | "think", "user_text": str, "labels": [str, ...],
     "target": [float, ...],             # teacher distribution over the displayed letters
     "reasoning": str | null}            # think traces only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .prompt import close_think, generation_prefix
from .runtime import letter_token_ids
from .schema import LETTERS


class Encoder:
    def __init__(self, tokenizer: Any, max_seq_len: int):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.letter_ids = letter_token_ids(tokenizer)
        self.eot = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def encode(self, rec: dict) -> dict | None:
        """Return token ids, the answer position, letter ids and the target, or None if too long.

        ``ans_pos`` is the position whose next-token logits are the decision distribution.
        ``lm_start`` (think records only) is the first completion token; the LM loss covers
        ``tokens[lm_start:]`` predicted from positions ``lm_start - 1 .. len - 2``.
        """
        n = len(rec["labels"])
        target = np.asarray(rec["target"], dtype=np.float32)
        target = target / target.sum()
        if rec["kind"] == "fast" or not rec.get("reasoning"):
            prefix = generation_prefix(self.tok, rec["user_text"], "fast")
            toks = self.tok.encode(prefix, add_special_tokens=False)
            ans_pos = len(toks) - 1
            lm_start = None
        else:
            prefix = generation_prefix(self.tok, rec["user_text"], "think")
            p_toks = self.tok.encode(prefix, add_special_tokens=False)
            b_toks = self.tok.encode(close_think(rec["reasoning"]), add_special_tokens=False)
            letter_idx = int(np.argmax(target))
            toks = p_toks + b_toks + [self.letter_ids[letter_idx]] + self.eot
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


def load_records(path: str | Path, limit: int | None = None) -> list[dict]:
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return recs[:limit] if limit else recs


def collate_numpy(exs: list[dict]) -> dict:
    """Right-pad a batch into numpy arrays (framework-agnostic)."""
    B = len(exs)
    L = max(len(e["tokens"]) for e in exs)
    K = len(LETTERS)
    out = {
        "tokens": np.zeros((B, L), dtype=np.int64),
        "attention_mask": np.zeros((B, L), dtype=np.int64),
        "letters": np.zeros((B, K), dtype=np.int64),
        "valid": np.zeros((B, K), dtype=bool),
        "target": np.zeros((B, K), dtype=np.float32),
        "ans_pos": np.zeros(B, dtype=np.int64),
        "n_tokens": int(sum(len(e["tokens"]) for e in exs)),
    }
    rows, pos, tgt = [], [], []
    for b, e in enumerate(exs):
        n, t = len(e["letters"]), len(e["tokens"])
        out["tokens"][b, :t] = e["tokens"]
        out["attention_mask"][b, :t] = 1
        out["letters"][b, :n] = e["letters"]
        out["valid"][b, :n] = True
        out["target"][b, :n] = e["target"]
        out["ans_pos"][b] = e["ans_pos"]
        if e["lm_start"] is not None:
            p = np.arange(e["lm_start"] - 1, t - 1)
            rows += [b] * len(p)
            pos += p.tolist()
            tgt += e["tokens"][p + 1].tolist()
    out["lm"] = (
        (np.asarray(rows, np.int64), np.asarray(pos, np.int64), np.asarray(tgt, np.int64))
        if rows else None
    )
    return out


def make_batches(examples: list[dict], batch_size: int, think_batch_size: int, rng) -> list[list[int]]:
    """Homogeneous batches (fast-only or think-only), grouped by length to limit padding."""
    batches = []
    for kind, bs in (("fast", batch_size), ("think", think_batch_size)):
        idx = [i for i, ex in enumerate(examples) if (ex["lm_start"] is None) == (kind == "fast")]
        rng.shuffle(idx)
        mega = bs * 32
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s : s + mega], key=lambda i: len(examples[i]["tokens"]))
            batches += [chunk[j : j + bs] for j in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches
