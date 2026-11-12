"""Decision-quality and calibration metrics.

All functions take ``probs`` of shape (N, K) (rows sum to 1, K may vary per row when passed as a
list of arrays) and integer ``labels`` of shape (N,). Rows with different K are supported via the
list form; metrics only use the top probability and the probability of the true label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-12


def _top_and_true(probs: Sequence[np.ndarray], labels: Sequence[int]) -> tuple[np.ndarray, ...]:
    conf = np.array([float(np.max(p)) for p in probs])
    pred = np.array([int(np.argmax(p)) for p in probs])
    labels = np.asarray(labels, dtype=int)
    p_true = np.array([float(p[y]) for p, y in zip(probs, labels)])
    return conf, pred, labels, p_true


def accuracy(probs, labels) -> float:
    _, pred, labels, _ = _top_and_true(probs, labels)
    return float(np.mean(pred == labels))


def macro_f1(probs, labels) -> float:
    _, pred, labels, _ = _top_and_true(probs, labels)
    classes = np.unique(np.concatenate([labels, pred]))
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def nll(probs, labels) -> float:
    _, _, _, p_true = _top_and_true(probs, labels)
    return float(np.mean(-np.log(np.clip(p_true, EPS, 1.0))))


def brier(probs, labels) -> float:
    """Multi-class Brier score (sum over classes, averaged over examples)."""
    out = []
    for p, y in zip(probs, labels):
        onehot = np.zeros_like(p)
        onehot[int(y)] = 1.0
        out.append(float(np.sum((p - onehot) ** 2)))
    return float(np.mean(out))


def ece(probs, labels, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins (top-label)."""
    conf, pred, labels, _ = _top_and_true(probs, labels)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            err += mask.sum() / total * abs(correct[mask].mean() - conf[mask].mean())
    return float(err)


def risk_coverage(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk when keeping the most-confident predictions first."""
    conf, pred, labels, _ = _top_and_true(probs, labels)
    order = np.argsort(-conf, kind="stable")
    wrong = (pred[order] != labels[order]).astype(float)
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), np.cumsum(wrong) / n


def coverage_at_risk(probs, labels, max_risk: float = 0.05) -> float:
    """Largest fraction of examples that can be auto-decided with error rate <= ``max_risk``."""
    coverage, risk = risk_coverage(probs, labels)
    ok = np.where(risk <= max_risk)[0]
    return float(coverage[ok.max()]) if ok.size else 0.0


def aurc(probs, labels) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(probs, labels)
    return float(np.mean(risk))


def flip_rate(pred_a: Sequence[str], pred_b: Sequence[str]) -> float:
    """Fraction of questions whose predicted *label* differs between two option orders."""
    a, b = np.asarray(pred_a), np.asarray(pred_b)
    return float(np.mean(a != b))


def summary(probs, labels, n_bins: int = 15) -> dict[str, float]:
    return {
        "n": len(labels),
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "ece": ece(probs, labels, n_bins),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "coverage_at_5pct_error": coverage_at_risk(probs, labels, 0.05),
        "aurc": aurc(probs, labels),
    }
