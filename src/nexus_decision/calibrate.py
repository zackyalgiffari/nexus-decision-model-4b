"""Temperature scaling over option-letter logits.

The runtime turns the restricted letter logits ``z`` into probabilities with ``softmax(z / T)``.
``T`` is fitted per mode by minimising the cross-entropy against targets on a held-out set.
Targets can be hard labels (int) or soft teacher distributions (arrays).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _targets(target, k: int) -> np.ndarray:
    if np.isscalar(target):
        t = np.zeros(k)
        t[int(target)] = 1.0
        return t
    t = np.asarray(target, dtype=np.float64)
    return t / t.sum()


def cross_entropy(logits: Sequence[np.ndarray], targets: Sequence, temperature: float) -> float:
    total = 0.0
    for z, y in zip(logits, targets):
        p = softmax(z, temperature)
        t = _targets(y, len(z))
        total += float(-(t * np.log(np.clip(p, 1e-12, 1.0))).sum())
    return total / max(1, len(logits))


def fit_temperature(
    logits: Sequence[np.ndarray],
    targets: Sequence,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 80,
) -> float:
    """Golden-section search over log(T); the NLL is unimodal in T for temperature scaling."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc = cross_entropy(logits, targets, float(np.exp(c)))
    fd = cross_entropy(logits, targets, float(np.exp(d)))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = cross_entropy(logits, targets, float(np.exp(c)))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = cross_entropy(logits, targets, float(np.exp(d)))
    return float(np.exp((a + b) / 2))


DEFAULT_CONFIG = {"temperature": {"fast": 1.0, "think": 1.0}}


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    p = Path(path)
    if not p.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    cfg = json.loads(p.read_text())
    cfg.setdefault("temperature", {}).setdefault("fast", 1.0)
    cfg["temperature"].setdefault("think", 1.0)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2) + "\n")
