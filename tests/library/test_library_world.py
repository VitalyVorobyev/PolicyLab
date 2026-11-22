"""Analytical and invariant tests for phasespace.library.world."""

from __future__ import annotations

import math

import numpy as np

from phasespace.library.config import DEFAULT_CONFIG
from phasespace.library.world import (
    GridLayout,
    LatentRule,
    evaluate_path_expected_cost,
    expected_level_costs,
    generate_layout,
    mutate_rule,
    observation_matrix,
    oracle_cost,
    sample_rule,
    shortest_path,
    true_terrain_expected_costs,
)


def test_observation_matrix_analytical_normalization() -> None:
    obs = observation_matrix(n_levels=4, beta=1.4)
    assert obs.shape == (4, 4)
    np.testing.assert_allclose(obs.sum(axis=1), np.ones(4), atol=1e-15)
    for lvl in range(4):
        assert obs[lvl, lvl] == np.max(obs[lvl])
    assert math.isclose(float(obs[0, 1] / obs[0, 0]), math.exp(-1.4), rel_tol=1e-14)
    assert math.isclose(float(obs[0, 2] / obs[0, 0]), math.exp(-2.8), rel_tol=1e-14)


def test_expected_level_costs_hand_calculated() -> None:
    obs = observation_matrix(n_levels=4, beta=1.4)
    exp_costs = expected_level_costs((1.0, 3.0, 7.0, 16.0), obs)
    weights = [math.exp(-1.4 * abs(0 - y)) for y in range(4)]
    expected_l0 = sum(w * c for w, c in zip(weights, (1.0, 3.0, 7.0, 16.0), strict=True)) / sum(
        weights
    )
    assert math.isclose(float(exp_costs[0]), expected_l0, rel_tol=1e-14)
    assert float(exp_costs[0]) < float(exp_costs[1]) < float(exp_costs[2]) < float(exp_costs[3])


def test_mutate_rule_strict_derangement() -> None:
    rng = np.random.default_rng(101)
    for _ in range(50):
        rule = sample_rule(rng, DEFAULT_CONFIG)
        mutated = mutate_rule(rule, rng, DEFAULT_CONFIG)
        assert len(mutated.levels) == DEFAULT_CONFIG.n_terrains
        assert all(0 <= lv < DEFAULT_CONFIG.n_levels for lv in mutated.levels)
        assert all(a != b for a, b in zip(rule.levels, mutated.levels, strict=True))


def test_shortest_path_matches_hand_solved_grid() -> None:
    # 3x3 layout where direct top-right/bottom corner has high-cost barrier on row 1 col 0 & 1
    # forcing the agent along row 0 then down col 2.
    # Start=(0,0) [-1], Goal=(2,2) [-1].
    # Costs per terrain: T0=1.0, T1=20.0
    terrains = np.array(
        [
            [-1, 0, 0],
            [1, 1, 0],
            [0, 0, -1],
        ],
        dtype=np.int64,
    )
    layout = GridLayout(size=3, terrains=terrains)
    costs = np.array([2.0, 20.0], dtype=np.float64)
    path, total = shortest_path(layout, costs)
    # Minimal path steps: (0,0)->(0,1)[2.0]->(0,2)[2.0]->(1,2)[2.0]->(2,2)[1.0] = 7.0
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert math.isclose(total, 7.0, abs_tol=1e-12)
    assert math.isclose(evaluate_path_expected_cost(layout, path, costs), 7.0, abs_tol=1e-12)


def test_generate_layout_determinism_and_regret_non_negative() -> None:
    rng1 = np.random.default_rng(777)
    rng2 = np.random.default_rng(777)
    layout1 = generate_layout(rng1, DEFAULT_CONFIG)
    layout2 = generate_layout(rng2, DEFAULT_CONFIG)
    np.testing.assert_array_equal(layout1.terrains, layout2.terrains)

    assert int(layout1.terrains[0, 0]) == -1
    assert int(layout1.terrains[8, 8]) == -1
    non_neutral = layout1.terrains[layout1.terrains >= 0]
    distinct = set(int(x) for x in non_neutral)
    assert 1 <= len(distinct) <= DEFAULT_CONFIG.active_terrains_per_layout
    assert distinct.issubset(set(range(DEFAULT_CONFIG.n_terrains)))

    rule = LatentRule(levels=(0, 1, 2, 3, 0, 1, 2, 3))
    exp_costs = expected_level_costs(
        DEFAULT_CONFIG.level_costs,
        observation_matrix(DEFAULT_CONFIG.n_levels, DEFAULT_CONFIG.noise_beta),
    )
    true_costs = true_terrain_expected_costs(rule, exp_costs)
    wrong_costs = np.ones(DEFAULT_CONFIG.n_terrains, dtype=np.float64)
    subopt_path, _ = shortest_path(layout1, wrong_costs)
    assert evaluate_path_expected_cost(layout1, subopt_path, true_costs) >= oracle_cost(
        layout1, rule, DEFAULT_CONFIG, exp_costs
    )
