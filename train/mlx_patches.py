"""Chunk-parallel, differentiable Gated DeltaNet recurrence for MLX training.

Why: mlx-lm 0.32.x runs Qwen3.5's Gated DeltaNet through a Metal kernel that has no VJP, so in
training mode it falls back to ``gated_delta_ops``, a per-token Python loop. That path trained at
about 12 tokens/s for Qwen3.5-4B on an M4, which is far too slow for distillation.

``gated_delta_chunked`` / ``_tri_inv`` are vendored from the open upstream pull request
ml-explore/mlx-lm#2021 (commit 5f408542; mlx-lm is MIT-licensed, Copyright © 2023 Apple Inc.). The math is the chunkwise UT/WY form of the gated delta rule
(Yang et al., arXiv:2406.06484 / 2412.06464). It cuts the sequential depth from T to T/chunk.
``tests/test_mlx_patches.py`` checks outputs and gradients against ``gated_delta_ops``.

Call ``apply()`` before building the training graph. Inference is unchanged: the patch only
affects calls with ``use_kernel=False`` (training mode), no padding mask, and a scalar gate.
"""

from __future__ import annotations

import mlx.core as mx


def _tri_inv(A: mx.array) -> mx.array:
    """Inverse of a unit lower-triangular matrix via block-Schur recursion + Newton polish."""

    def _rec(M: mx.array) -> mx.array:
        n = M.shape[-1]
        if n == 1:
            return mx.ones_like(M)
        h = n // 2
        A11 = _rec(M[..., :h, :h])
        A22 = _rec(M[..., h:, h:])
        A21 = -A22 @ (M[..., h:, :h] @ A11)
        top = mx.concatenate([A11, mx.zeros_like(A21).swapaxes(-1, -2)], axis=-1)
        bot = mx.concatenate([A21, A22], axis=-1)
        return mx.concatenate([top, bot], axis=-2)

    C = A.shape[-1]
    X = _rec(A)
    eye = mx.eye(C, dtype=A.dtype)
    for _ in range(2):
        X = X @ (2.0 * eye - A @ X)
    return X


def gated_delta_chunked(q, k, v, g, beta, state=None, chunk_size: int = 64):
    """Chunk-parallel gated delta rule. q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv]."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        qc = q[:, start:end].swapaxes(1, 2).astype(mx.float32)
        kc = k[:, start:end].swapaxes(1, 2).astype(mx.float32)
        vc = v[:, start:end].swapaxes(1, 2).astype(mx.float32)
        gc = mx.maximum(g[:, start:end].swapaxes(1, 2).astype(mx.float32), 1e-8)
        bc = beta[:, start:end].swapaxes(1, 2).astype(mx.float32)
        C = qc.shape[2]

        idx = mx.arange(C)
        strict_lower = idx[:, None] > idx[None, :]
        incl_lower = idx[:, None] >= idx[None, :]

        log_gamma = mx.cumsum(mx.log(gc), axis=-1)
        gamma = mx.exp(log_gamma)
        decay_ratio = mx.exp(mx.minimum(log_gamma[..., :, None] - log_gamma[..., None, :], 0.0))

        kkt = kc @ kc.swapaxes(-1, -2)
        A = mx.eye(C) + bc[..., :, None] * mx.where(strict_lower, decay_ratio * kkt, 0.0)
        A_inv = _tri_inv(A)

        v_prime = vc - gamma[..., :, None] * (kc @ state.swapaxes(-1, -2))
        u = A_inv @ (bc[..., :, None] * v_prime)

        qkt = qc @ kc.swapaxes(-1, -2)
        p = mx.where(incl_lower, decay_ratio * qkt, 0.0)
        yc = gamma[..., :, None] * (qc @ state.swapaxes(-1, -2)) + p @ u

        gamma_to_end = mx.exp(log_gamma[..., -1:] - log_gamma)
        w = gamma_to_end[..., :, None] * u
        state = gamma[..., -1][..., None, None] * state + w.swapaxes(-1, -2) @ kc
        ys.append(yc.swapaxes(1, 2))

    return mx.concatenate(ys, axis=1).astype(q.dtype), state


_APPLIED = False


def apply() -> None:
    """Route training-mode Gated DeltaNet calls in qwen3_5 through ``gated_delta_chunked``."""
    global _APPLIED
    if _APPLIED:
        return
    import mlx_lm.models.gated_delta as gd
    import mlx_lm.models.qwen3_5 as q35

    original = gd.gated_delta_update

    def patched(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
                lower_bound=None, allow_neg_eigval=False):
        if not use_kernel and mask is None and q.shape[1] > 1:
            beta = mx.sigmoid(b)
            if allow_neg_eigval:
                beta = beta * 2.0
            if lower_bound is None:
                g = gd.compute_g(A_log, a, dt_bias)
            else:
                g = gd.compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
            if g.ndim == 3:
                return gated_delta_chunked(q, k, v, g, beta, state)
        return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                        lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)

    q35.gated_delta_update = patched
    _APPLIED = True
