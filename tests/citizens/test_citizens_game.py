"""Game rules against hand-computed cases (Koster et al. 2025, Fig. S13 and the CSV)."""

from __future__ import annotations

import numpy as np
import pytest

from phasespace.citizens import game
from phasespace.citizens import mechanisms as mech


def test_fig_s13_example() -> None:
    # Supplementary Fig. S13: the whole pool of 200 is offered, 42 coins come back.
    assert float(game.next_pool(200.0, [50.0, 50.0, 50.0, 50.0], [10.0, 12.0, 20.0, 0.0])) == (
        pytest.approx(58.8, abs=1e-12)
    )


def test_data_example_first_row() -> None:
    # CSV row 0 (Proportional BC 1, round 0): 200 - 4 * 49.9997 + 1.4 * 114.417175 = 160.18525.
    offers = [49.9997] * 4
    actions = [23.055292, 16.577978, 37.082737, 37.701168]
    expected = 200.0 - 199.9988 + 1.4 * 114.417175
    assert float(game.next_pool(200.0, offers, actions)) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(160.18525, abs=1e-5)


def test_pool_is_capped_at_carrying_capacity() -> None:
    # 200 - 100 + 1.4 * 100 = 240 > 200.
    assert float(game.next_pool(200.0, [25.0] * 4, [25.0] * 4)) == 200.0


def test_undistributed_pool_does_not_grow() -> None:
    # 100 offered of 150, nothing returned: the 50 kept back stays 50.
    assert float(game.next_pool(150.0, [25.0] * 4, [0.0] * 4)) == pytest.approx(50.0)


def test_surplus_is_offer_minus_reciprocation() -> None:
    np.testing.assert_allclose(
        game.surplus([10.0, 5.0, 0.5, 0.0], [3.0, 5.0, 0.0, 0.0]), [7, 0, 0.5, 0]
    )


def test_human_reciprocation_bounds() -> None:
    assert list(game.max_human_reciprocation([29.4, 0.99, 1.0, 66.6663])) == [29.0, 0.0, 1.0, 66.0]
    assert game.reciprocations_valid([29.4, 0.5, 2, 2], [29, 0, 2, 0], integer=True)
    assert not game.reciprocations_valid([29.4, 0.5, 2, 2], [29.4, 0, 2, 0], integer=True)
    assert not game.reciprocations_valid([29.4, 0.5, 2, 2], [30, 0, 2, 0], integer=False)
    assert game.reciprocations_valid([29.4, 0.5, 2, 2], [29.4, 0.25, 2, 0], integer=False)


def test_depletion_and_exclusion_thresholds() -> None:
    assert list(game.is_depleted([0.99, 1.0, 0.01])) == [True, False, True]
    assert list(game.is_excluded([0.0295, 1.0, 0.9999])) == [True, False, True]


def test_step_updates_state_and_rejects_invalid_moves() -> None:
    state = game.GameState()
    result = game.step(state, [50, 50, 50, 50], [10, 12, 20, 0], integer=True)
    assert result.state.pool == pytest.approx(58.8)
    assert result.state.round_index == 1
    assert result.state.prev_reciprocations == (10.0, 12.0, 20.0, 0.0)
    assert result.surplus == (40.0, 38.0, 30.0, 50.0)
    with pytest.raises(ValueError, match="sum to at most the pool"):
        game.step(result.state, [20, 20, 20, 0], [0, 0, 0, 0])
    with pytest.raises(ValueError, match="0 <= c <= e"):
        game.step(state, [50, 50, 50, 50], [51, 0, 0, 0])
    with pytest.raises(ValueError, match="integral"):
        game.step(state, [50, 50, 50, 50], [0.5, 0, 0, 0], integer=True)
    with pytest.raises(ValueError, match="non-finite"):
        game.step(state, [np.nan, 50, 50, 50], [0, 0, 0, 0])


def test_depleted_state_is_absorbing_for_humans() -> None:
    # Below 1 every offer is below 1, so a human must return 0 and nothing grows.
    state = game.GameState(pool=0.8)
    result = game.step(state, [0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0], integer=True)
    assert result.state.depleted
    assert result.state.pool == game.POOL_FLOOR  # the recorded software's 0.01 floor
    assert game.step(result.state, [0.0022] * 4, [0] * 4).state.pool == game.POOL_FLOOR


def test_floor_can_be_disabled_for_the_paper_equation() -> None:
    assert float(game.next_pool(0.8, [0.2] * 4, [0] * 4, floor=-np.inf)) == pytest.approx(0.0)


def test_step_accepts_every_baseline_offer_despite_float_rounding() -> None:
    # Regression: R * share_i with shares summing to 1 can sum to R + 1 ulp; the budget
    # check must accept that (relative slack 1e-12) yet the cases below must include such
    # sums, or the test would not exercise the bug. Synthetic R (log-uniform over the
    # floor to the cap) and previous reciprocations (integers, zeros and continuous).
    rng = np.random.default_rng(20261020)
    n = 4000
    pools = np.exp(rng.uniform(np.log(game.POOL_FLOOR), np.log(game.MAX_POOL), n))
    prev = rng.integers(0, 60, size=(n, 4)).astype(np.float64)
    prev[rng.random((n, 4)) < 0.3] = 0.0
    prev[n // 2 :] = rng.uniform(0.0, 60.0, size=(n - n // 2, 4))
    weights = rng.choice([mech.W_PROPORTIONAL, mech.W_MIXED, mech.W_EQUAL], size=n)
    interpolating = mech.interpolating_weight(pools, quantize=False)
    over_by_rounding = 0
    for rule in ("paper", "deployed"):
        for w in (weights, interpolating):
            offers = mech.baseline_offers(pools, prev, w, rule=rule)
            over_by_rounding += int(np.count_nonzero(offers.sum(axis=-1) > pools))
            for r, e in zip(pools, offers, strict=True):
                state = game.GameState(pool=float(r), prev_reciprocations=(0.0,) * 4)
                game.step(state, e, np.zeros(4))  # raises if the offer is rejected
    assert over_by_rounding > 0


def test_offer_budget_check_still_rejects_real_overspending() -> None:
    # 1e-9 coins over a pool of 200 is ~5e-12 relative: above the 1e-12 slack.
    assert game.offers_valid(200.0, [50.0, 50.0, 50.0, 50.0])
    assert not game.offers_valid(200.0, [50.0, 50.0, 50.0, 50.0 + 1e-9])
    assert not game.offers_valid(0.01, [0.0025, 0.0025, 0.0025, 0.0026])
    assert not game.offers_valid(1.0, [-0.1, 0.0, 0.0, 0.0])
