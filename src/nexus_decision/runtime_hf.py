"""Transformers (PyTorch) backend: CUDA, MPS or CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend:
    def __init__(self, path: str, device: str | None = None, dtype: str = "bfloat16", **_: Any):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        p = Path(path)
        self.local_dir = p if p.exists() else Path(snapshot_download(path))
        self.device = device or _default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_dir, dtype=getattr(torch, dtype)
        ).to(self.device)
        self.model.eval()
        self.model_id = path

    def _ids(self, text: str):
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        return ids.to(self.device)

    def next_token_logits(self, text: str) -> np.ndarray:
        with self._torch.inference_mode():
            out = self.model(input_ids=self._ids(text))
        return out.logits[0, -1].float().cpu().numpy()

    def generate_until(
        self, text: str, stop: str, max_tokens: int, temperature: float, top_p: float, seed: int
    ) -> str:
        self._torch.manual_seed(seed)
        ids = self._ids(text)
        with self._torch.inference_mode():
            out = self.model.generate(
                input_ids=ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                stop_strings=[stop],
                tokenizer=self.tokenizer,
            )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
