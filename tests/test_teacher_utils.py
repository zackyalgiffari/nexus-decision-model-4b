import math
from datetime import UTC, datetime

from data.teacher import is_peak, letter_distribution


def test_letter_distribution_merges_variants_and_spreads_residual():
    lp = {"B": math.log(0.6), " B": math.log(0.1), "A": math.log(0.2), "Because": math.log(0.05)}
    d = letter_distribution(lp, 3)
    # residual = 1 - 0.95 = 0.05 goes to C (the only unseen letter)
    assert abs(sum(d) - 1) < 1e-9
    assert d[1] > d[0] > d[2] > 0
    assert abs(d[2] - 0.05 / 0.95) < 1e-9


def test_lowercase_a_is_not_letter_a():
    d = letter_distribution({"a": math.log(0.9), "B": math.log(0.1)}, 2)
    assert d[1] > 0.99


def test_peak_hours():
    assert is_peak(datetime(2026, 11, 12, 7, 0, tzinfo=UTC))      # Thursday 07 UTC
    assert not is_peak(datetime(2026, 11, 12, 12, 0, tzinfo=UTC))
    assert not is_peak(datetime(2026, 11, 14, 7, 0, tzinfo=UTC))  # Saturday
