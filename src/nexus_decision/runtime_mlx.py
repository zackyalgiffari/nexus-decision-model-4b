"""MLX backend (Apple silicon)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _resolve_local_dir(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path))


class MLXBackend:
    def __init__(self, path: str, adapter_path: str | None = None, **_: Any):
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        self.local_dir = _resolve_local_dir(path)
        self.model, self.tokenizer = load(str(self.local_dir), adapter_path=adapter_path)
        self.model_id = path

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def next_token_logits(self, text: str) -> np.ndarray:
        mx = self._mx
        tokens = mx.array(self._encode(text))[None]
        logits = self.model(tokens)[0, -1].astype(mx.float32)
        mx.eval(logits)
        return np.array(logits)

    def generate_until(
        self, text: str, stop: str, max_tokens: int, temperature: float, top_p: float, seed: int
    ) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self._mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        out = ""
        for resp in stream_generate(
            self.model, self.tokenizer, prompt=self._encode(text), max_tokens=max_tokens,
            sampler=sampler,
        ):
            out += resp.text
            if stop in out:
                break
        return out
