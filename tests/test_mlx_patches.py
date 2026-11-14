import pytest

mx = pytest.importorskip("mlx.core")
gd = pytest.importorskip("mlx_lm.models.gated_delta")

from train.mlx_patches import gated_delta_chunked


def _inputs(T=150, B=1, Hk=4, Hv=8, D=32, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, D))
    k = mx.random.normal((B, T, Hk, D))
    q, k = gd.normalize_qk(q, k, inv_scale=D**-0.5, eps=1e-6)
    v = mx.random.normal((B, T, Hv, D))
    g = mx.exp(-mx.abs(mx.random.normal((B, T, Hv))) * 0.3)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    return q, k, v, g, beta


def test_forward_matches_sequential():
    q, k, v, g, beta = _inputs()
    y0, s0 = gd.gated_delta_ops(q, k, v, g, beta)
    y1, s1 = gated_delta_chunked(q, k, v, g, beta)
    assert mx.max(mx.abs(y0 - y1)).item() < 1e-3
    assert mx.max(mx.abs(s0 - s1)).item() < 1e-3


def test_gradients_match_sequential():
    q, k, v, g, beta = _inputs(T=90)

    def loss(fn):
        return lambda v, beta: mx.sum(fn(q, k, v, g, beta)[0] ** 2)

    gv0, gb0 = mx.grad(loss(gd.gated_delta_ops), argnums=(0, 1))(v, beta)
    gv1, gb1 = mx.grad(loss(gated_delta_chunked), argnums=(0, 1))(v, beta)
    assert mx.max(mx.abs(gv0 - gv1)).item() < 1e-2
    assert mx.max(mx.abs(gb0 - gb1)).item() < 1e-2
