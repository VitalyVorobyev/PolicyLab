"""C3 stage helpers: the k grid, the paired bootstrap and the configuration hash."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl.stage import paired_bootstrap, ppo_config


def test_ln_k_grid_is_the_paper_grid() -> None:
    grid = c3config.ln_k_grid()
    assert len(grid) == 101
    assert grid[0] == -5.0 and grid[-1] == 5.0 and 3.1 in grid
    assert np.allclose(np.diff(grid), 0.1)
    assert math.exp(3.1) == pytest.approx(22.2, abs=0.01)  # the published k = 22 point


def test_paired_bootstrap_of_a_constant_shift_is_exact() -> None:
    baseline = np.random.default_rng(1).normal(1000.0, 50.0, 200)
    planner = np.stack([baseline + 7.0, baseline + 7.0, baseline + 7.0])
    mean, lo, hi = paired_bootstrap(planner, baseline, 500, 3)
    assert mean == pytest.approx(7.0) and lo == pytest.approx(7.0) and hi == pytest.approx(7.0)


def test_paired_bootstrap_covers_the_true_difference() -> None:
    """Independent check: a known mean difference of 2 with unit noise and 400 games per
    seed; the 95 % CI must contain 2 and have about the normal-theory width."""
    rng = np.random.default_rng(2)
    baseline = rng.normal(0.0, 5.0, 400)
    planner = baseline[None, :] + 2.0 + rng.normal(0.0, 1.0, (3, 400))
    mean, lo, hi = paired_bootstrap(planner, baseline, 2000, 4)
    assert lo < 2.0 < hi
    se = 1.0 / math.sqrt(3 * 400)
    assert 2.0 * 1.96 * se * 0.5 < hi - lo < 2.0 * 1.96 * se * 4.0
    assert mean == pytest.approx(float((planner - baseline).mean()))


def test_ppo_defaults_are_the_recorded_configuration() -> None:
    cfg = ppo_config()
    assert cfg.n_games == c3config.PPO_N_GAMES
    assert cfg.learning_rate == c3config.PPO_LEARNING_RATE
    assert ppo_config(learning_rate=0.1).learning_rate == 0.1
    assert len(c3config.config_sha256()) == 64
