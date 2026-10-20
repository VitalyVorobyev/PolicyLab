"""Notebook metrics against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.citizens import metrics


def test_gini_of_known_vectors() -> None:
    # 16 ordered pairs including i = j. (0,0,0,1): 6 pairs differ by 1 -> mad 6/16,
    # mean 1/4 -> 0.5 * 0.375 / 0.25 = 0.75. (1,2,3,4): sum over i<j of |diff| = 10,
    # mad 20/16 = 1.25, mean 2.5 -> 0.25.
    assert float(metrics.gini([1.0, 1.0, 1.0, 1.0])) == 0.0
    assert float(metrics.gini([0.0, 0.0, 0.0, 1.0])) == pytest.approx(0.75)
    assert float(metrics.gini([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.25)
    np.testing.assert_allclose(metrics.gini([[2.0, 2.0, 2.0, 2.0], [0, 0, 0, 5]]), [0.0, 0.75])


def test_gini_of_all_zero_rewards_is_perfect_equality() -> None:
    # Convention: everyone received nothing -> 0.0 (not the notebook's 0/0 = NaN).
    assert float(metrics.gini([0.0, 0.0, 0.0, 0.0])) == 0.0
    np.testing.assert_array_equal(metrics.gini([[0.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]]), [0.0, 0.75])
    with pytest.raises(ValueError, match="non-negative"):
        metrics.gini([1.0, -1.0, 0.0, 0.0])


def test_surplus_and_gini_over_recorded_rounds_only() -> None:
    nan = math.nan
    rewards = np.array(
        [
            [[4, 2, 0, 2], [2, 2, 0, 2], [nan, nan, nan, nan]],  # 2 rounds
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 5]],  # 3 rounds
        ]
    )
    n_rounds = np.array([2, 3])
    np.testing.assert_allclose(
        metrics.mean_rewards(rewards, n_rounds), [[3, 2, 0, 2], [1, 1, 1, 7 / 3]]
    )
    np.testing.assert_allclose(metrics.surplus(rewards, n_rounds), [7 / 4, (3 + 7 / 3) / 4])
    # Game 1: x = (3, 2, 0, 2), mean 7/4; sum_{i<j}|diff| = 1+3+1+2+0+2 = 9 -> mad 18/16.
    assert metrics.reward_gini(rewards, n_rounds)[0] == pytest.approx(0.5 * (18 / 16) / (7 / 4))


def test_depletion_round_is_one_based_with_default_40() -> None:
    pools = np.array(
        [
            [200.0, 50.0, 0.5, 0.01, 0.01],
            [200.0, 150.0, 120.0, 100.0, 90.0],
            [200.0, 0.99, 5.0, np.nan, np.nan],
        ]
    )
    np.testing.assert_array_equal(metrics.depletion_round(pools, [5, 5, 3]), [3, 40, 2])


def test_players_active() -> None:
    active = np.array([[4, 4, 3, 2], [4, 3, -1, -1]])
    np.testing.assert_allclose(metrics.mean_players_active(active, [4, 2]), [13 / 4, 3.5])
    np.testing.assert_array_equal(metrics.last_players_active(active, [4, 2]), [2, 3])


def test_rank_sum_z_by_hand() -> None:
    # a = (1, 2, 3), b = (4, 5, 6): rank sum of a = 6, mean 3 * 7 / 2 = 10.5,
    # variance 3 * 3 * 7 / 12 = 5.25 -> z = -4.5 / sqrt(5.25).
    assert metrics.rank_sum_z([1, 2, 3], [4, 5, 6]) == pytest.approx(-4.5 / math.sqrt(5.25))
    assert metrics.rank_sum_z([4, 5, 6], [1, 2, 3]) == pytest.approx(4.5 / math.sqrt(5.25))
