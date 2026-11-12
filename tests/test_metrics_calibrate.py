import numpy as np

from nexus_decision import metrics
from nexus_decision.calibrate import fit_temperature, softmax


def test_ece_hand_computed():
    # Two bins hit: conf 0.9 (2 examples, 1 correct) and conf 0.6 (2 examples, 2 correct).
    probs = [np.array([0.9, 0.1]), np.array([0.9, 0.1]), np.array([0.6, 0.4]), np.array([0.4, 0.6])]
    labels = [0, 1, 0, 1]
    # |0.5-0.9|*0.5 + |1.0-0.6|*0.5 = 0.4
    assert abs(metrics.ece(probs, labels) - 0.4) < 1e-9
    assert metrics.accuracy(probs, labels) == 0.75


def test_coverage_at_risk():
    probs = [np.array([0.99, 0.01]), np.array([0.9, 0.1]), np.array([0.6, 0.4])]
    labels = [0, 0, 1]
    assert metrics.coverage_at_risk(probs, labels, 0.05) == 2 / 3


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    logits, labels = [], []
    for _ in range(2000):
        true = rng.normal(size=4) * 1.5
        y = rng.choice(4, p=softmax(true))
        logits.append(true * 3.0)  # model is 3x overconfident
        labels.append(y)
    t = fit_temperature(logits, labels)
    assert 2.5 < t < 3.6
