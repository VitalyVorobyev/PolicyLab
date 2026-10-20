"""Mechanisms against hand-computed offers and values recorded in the published CSV."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.citizens import mechanisms as mech


def test_round_one_is_equal_split() -> None:
    np.testing.assert_allclose(mech.paper_offers(200.0, None, mech.W_PROPORTIONAL), [50.0] * 4)


def test_mixed_offers_by_hand() -> None:
    # R = 100, c = (10, 20, 30, 40), w = 0.5: equal part 12.5 each, proportional part
    # 50 * c / 100 = (5, 10, 15, 20).
    offers = mech.paper_offers(100.0, [10.0, 20.0, 30.0, 40.0], mech.W_MIXED)
    np.testing.assert_allclose(offers, [17.5, 22.5, 27.5, 32.5], rtol=0, atol=1e-12)


def test_proportional_and_equal_by_hand() -> None:
    np.testing.assert_allclose(
        mech.paper_offers(197.4, [50.0, 70.0, 21.0, 0.0], mech.W_PROPORTIONAL),
        [197.4 * 50 / 141, 197.4 * 70 / 141, 197.4 * 21 / 141, 0.0],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mech.paper_offers(140.0, [1.0, 0.0, 0.0, 9.0], mech.W_EQUAL), [35.0] * 4
    )


def test_zero_previous_sum_gives_equal_split() -> None:
    for w in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(mech.paper_offers(0.01, [0.0] * 4, w), [0.0025] * 4)


def test_shares_sum_to_one_and_broadcast() -> None:
    c = np.array([[[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]])
    shares = mech.baseline_shares(c, np.array([[0.3, 0.7]]))
    assert shares.shape == (1, 2, 4)
    np.testing.assert_allclose(shares.sum(axis=-1), 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        mech.baseline_shares([-1.0, 1.0, 1.0, 1.0], 0.5)


def test_interpolating_weight_by_hand() -> None:
    # (R/200)^22: 1 at R = 200; 0.95^22 = exp(22 ln 0.95) = 0.3235; 0.9^22 = 0.0985;
    # 0.5^22 = 2.4e-7. The deployed weight is rounded to one decimal.
    raw = mech.interpolating_weight([200.0, 190.0, 180.0, 100.0], quantize=False)
    np.testing.assert_allclose(
        raw, [1.0, math.exp(22 * math.log(0.95)), math.exp(22 * math.log(0.9)), 0.5**22]
    )
    assert raw[1] == pytest.approx(0.32353, abs=1e-5)
    np.testing.assert_allclose(
        mech.interpolating_weight([200.0, 190.0, 180.0, 100.0]), [1, 0.3, 0.1, 0]
    )


def test_deployed_offers_match_values_printed_in_the_csv() -> None:
    # Round 0 of every game: 4 x 49.9997 (CSV mechanism_action "[49.9997 49.9997 ...]").
    np.testing.assert_allclose(
        mech.deployed_offers(200.0, None, 1.0), [49.9997] * 4, rtol=0, atol=4e-6
    )
    # Equal Baseline Exp 1, round 1: pool 112.0 -> offers 27.9998 each.
    np.testing.assert_allclose(
        mech.deployed_offers(112.0, [0.0, 25.0, 25.0, 30.0], mech.W_EQUAL),
        [27.9998] * 4,
        rtol=0,
        atol=4e-6,
    )
    # Proportional Baseline Game 2 Exp 4, round 1: pool 200, previous (49, 0, 49, 49)
    # -> (66.6663, 0, 66.6663, 66.6663) = round(199.999 / 3, 4).
    np.testing.assert_allclose(
        mech.deployed_offers(200.0, [49.0, 0.0, 49.0, 49.0], mech.W_PROPORTIONAL),
        [66.6663, 0.0, 66.6663, 66.6663],
        rtol=0,
        atol=4e-6,
    )
    # At the 0.01 floor with nothing returned: (0.01 - 0.001) / 4 = 0.00225 -> 0.0022.
    np.testing.assert_allclose(
        mech.deployed_offers(0.01, [0.0] * 4, mech.W_PROPORTIONAL), [0.0022] * 4, atol=1e-9
    )


def test_deployed_offers_withhold_about_a_thousandth() -> None:
    offers = mech.deployed_offers(100.0, [10.0, 20.0, 30.0, 40.0], mech.W_MIXED)
    # 99.999 * (0.175, 0.225, 0.275, 0.325) rounded to 4 decimals.
    np.testing.assert_allclose(offers, [17.4998, 22.4998, 27.4997, 32.4997], atol=4e-6)
    assert 100.0 - offers.sum() == pytest.approx(0.001, abs=2e-4)


def test_random_offers_are_seeded_and_feasible() -> None:
    a = mech.random_offers(200.0, np.random.default_rng(7))
    b = mech.random_offers(200.0, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert np.all(a >= 0.0) and a.sum() <= 200.0
    rng = np.random.default_rng(20261020)
    draws = np.array([mech.random_offers(1.0, rng) for _ in range(20000)])
    # Dirichlet(1) over 5 parts: each share has mean 1/5 and variance (1/5)(4/5)/6 = 0.0267.
    se = math.sqrt(0.2 * 0.8 / 6 / 20000)
    np.testing.assert_allclose(draws.mean(axis=0), 0.2, atol=5 * se)


def test_baseline_offers_default_to_the_deployed_rule() -> None:
    # D33: simulations face the offers humans saw (49.9997 each in round 0), and the
    # paper's whole-pool split (50 each) is an explicit alternative.
    np.testing.assert_allclose(mech.baseline_offers(200.0, None, 1.0), [49.9997] * 4, atol=4e-6)
    np.testing.assert_allclose(
        mech.baseline_offers(200.0, None, 1.0, rule="paper"), [50.0] * 4, rtol=0, atol=1e-12
    )
    with pytest.raises(ValueError, match="unknown offer rule"):
        mech.baseline_offers(200.0, None, 1.0, rule="other")  # pyright: ignore[reportArgumentType]


def test_deployed_offers_need_the_withheld_amount_in_the_pool() -> None:
    with pytest.raises(ValueError, match="at least"):
        mech.deployed_offers(0.0005, None, 1.0)
