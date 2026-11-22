"""Analytical and invariant tests for phasespace.library.belief."""

from __future__ import annotations

import math

import numpy as np

from phasespace.library.belief import (
    bayes_update,
    bayes_update_batch,
    belief_entropy_bits,
    kl_divergence_nats,
    map_levels,
    terrain_entropy_bits,
    uniform_belief,
)
from phasespace.library.world import observation_matrix


def test_bayes_update_matches_hand_calculation_and_is_commutative() -> None:
    obs = observation_matrix(n_levels=4, beta=1.4)
    prior = uniform_belief(n_terrains=8, n_levels=4)

    # Hand calculate posterior for terrain 3 after observing y=1 once from uniform prior:
    unnorm = np.array([obs[lvl, 1] * 0.25 for lvl in range(4)], dtype=np.float64)
    expected_row = unnorm / unnorm.sum()
    post = bayes_update(prior, terrain_id=3, observed_y=1, obs_matrix=obs)
    np.testing.assert_allclose(post[3], expected_row, atol=1e-14)
    # Other rows unchanged:
    np.testing.assert_allclose(post[0], prior[0], atol=1e-15)

    # Order of observations must commute:
    p_ab = bayes_update_batch(prior, [(2, 0), (2, 3), (2, 1), (5, 2)], obs)
    p_ba = bayes_update_batch(prior, [(5, 2), (2, 1), (2, 3), (2, 0)], obs)
    np.testing.assert_allclose(p_ab, p_ba, atol=1e-14)


def test_entropy_and_kl_invariants() -> None:
    u = uniform_belief(n_terrains=8, n_levels=4)
    np.testing.assert_allclose(terrain_entropy_bits(u), np.full(8, 2.0), atol=1e-14)
    assert math.isclose(belief_entropy_bits(u), 16.0, abs_tol=1e-14)

    det = np.zeros((8, 4), dtype=np.float64)
    det[:, 0] = 1.0
    assert math.isclose(belief_entropy_bits(det), 0.0, abs_tol=1e-14)

    assert math.isclose(kl_divergence_nats(u, u), 0.0, abs_tol=1e-14)
    # KL(delta || uniform) = sum_{t=0..7} ln(4) = 8 * ln(4)
    assert math.isclose(kl_divergence_nats(det, u), 8.0 * math.log(4.0), rel_tol=1e-12)
    assert map_levels(det) == (0, 0, 0, 0, 0, 0, 0, 0)
