"""M4 statistics (protocol 8 and 11) on synthetic data with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.gravity.m4.config import Statistics
from phasespace.gravity.m4.stats import (
    bootstrap_ci,
    holm,
    paired_differences,
    primary_contrasts,
    seed_count,
    sign_flip_pvalue,
    spearman,
    stratified_permutation_pvalue,
    summary,
)


def test_sign_flip_exact_values() -> None:
    # All ten differences positive: only the identity and the all-flipped vector reach
    # |mean| -> p = 2 / 2^10 (protocol 8: the smallest attainable p with 10 seeds).
    assert sign_flip_pvalue([1.0] * 10) == pytest.approx(2 / 1024)
    assert sign_flip_pvalue([3.0, 1.0, 2.0, 5.0, 4.0, 1.5, 2.5, 0.5, 6.0, 7.0]) == pytest.approx(
        2 / 1024
    )
    # n = 3, d = (1, 2, 3): signed sums 6, 4, 2, 0, 0, -2, -4, -6 -> |sum| >= 6 twice.
    assert sign_flip_pvalue([1.0, 2.0, 3.0]) == pytest.approx(2 / 8)
    assert sign_flip_pvalue([1.0, -1.0] * 5) == 1.0
    assert math.isnan(sign_flip_pvalue([]))


def test_holm_known_example() -> None:
    # Sorted: 0.005 * 4 = 0.02, 0.01 * 3 = 0.03, 0.03 * 2 = 0.06, 0.04 * 1 -> max(0.06, 0.04).
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_bootstrap_ci() -> None:
    assert bootstrap_ci([2.0] * 10) == (2.0, 2.0)
    d = np.linspace(-1.0, 3.0, 10)
    lo, hi = bootstrap_ci(d)
    assert lo < float(d.mean()) < hi
    assert (lo, hi) == bootstrap_ci(d)  # seeded (4242)
    narrow = bootstrap_ci(d, Statistics(confidence=0.5))
    assert lo < narrow[0] < narrow[1] < hi


def test_paired_differences_require_matching_seeds() -> None:
    seeds, d = paired_differences({2: 5.0, 1: 3.0}, {1: 1.0, 2: 1.0})
    assert seeds == [1, 2] and list(d) == [2.0, 4.0]
    with pytest.raises(ValueError, match="same seeds"):
        paired_differences({1: 1.0}, {2: 1.0})


def test_primary_contrasts_detect_a_shift_and_not_a_null() -> None:
    rng = np.random.default_rng(1)
    seeds = range(1001, 1011)
    base = {s: float(v) for s, v in zip(seeds, rng.normal(10.0, 1.0, 10), strict=True)}
    shifted = {s: base[s] + 3.0 + 0.1 * k for k, s in enumerate(seeds)}
    same = dict(base)
    out = primary_contrasts({"T-X": (shifted, base), "T-Y": (same, base)})
    by = {c.name: c for c in out}
    assert by["T-X"].p_value == pytest.approx(2 / 1024)
    assert by["T-X"].p_holm == pytest.approx(2 * 2 / 1024)
    assert by["T-X"].significant and by["T-X"].ci[0] > 0.0
    assert by["T-Y"].mean_difference == 0.0 and by["T-Y"].p_value == 1.0
    assert not by["T-Y"].significant


def test_seed_count_rule() -> None:
    # (2.807 + 0.842)^2 = 13.3152...
    assert seed_count([1.0]) == (14, 14)
    assert seed_count([0.5, 0.2]) == (10, 4)
    assert seed_count([2.0]) == (16, 54)


def test_summary_and_spearman() -> None:
    s = summary([1.0, 2.0, 3.0, 10.0])
    assert s == {
        "mean": 4.0,
        "median": 2.5,
        "sd": pytest.approx(4.0825, abs=1e-4),
        "min": 1.0,
        "max": 10.0,
    }
    assert spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_stratified_permutation() -> None:
    x = [float(v) for v in range(8)] * 3
    strata = [s for s in (1, 2, 3) for _ in range(8)]
    rho, p = stratified_permutation_pvalue(x, [-v for v in x], strata, resamples=999)
    assert rho == pytest.approx(-1.0) and p == pytest.approx(1 / 1000)
    rng = np.random.default_rng(5)
    noise = rng.normal(size=24).tolist()
    _, p_null = stratified_permutation_pvalue(x, noise, strata, resamples=999)
    assert p_null > 0.05
