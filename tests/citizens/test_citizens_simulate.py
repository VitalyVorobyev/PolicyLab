"""Closed-loop simulator: hand-computed games, feasibility stop, recorded-game replay.

The replay tests need the parsed dataset (`just citizens-data`) and use TRAINING games
only (human Equal / Mixed / Proportional); they are skipped when it is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phasespace.citizens import data, validation
from phasespace.citizens.models.base import (
    K_MAX,
    CitizenModel,
    FloatArray,
    Observation,
    n_max,
)
from phasespace.citizens.models.baselines import B0Rational, Replay
from phasespace.citizens.simulate import (
    EXACT_MECHANISMS,
    BaselineMechanism,
    InfeasibleGameError,
    game_metrics,
    simulate,
)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
HAS_DATA = data.processed_path(PROCESSED, data.TRAINING_FILE).exists()


class ReturnAll(CitizenModel[None]):
    """Always returns floor(e) (deterministic)."""

    name = "return-all"

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> None:
        return None

    def step(self, state: None, obs: Observation) -> tuple[FloatArray, None]:
        pmf = np.zeros((obs.n_streams, K_MAX))
        pmf[np.arange(obs.n_streams), n_max(obs.own_offer)] = 1.0
        return pmf, None

    def describe(self) -> dict[str, Any]:
        return {}


class Cheater(ReturnAll):
    """Returns floor(e) + 1: infeasible (tests the K2 stop)."""

    def sample(self, pmf: FloatArray, obs: Observation, rng: np.random.Generator) -> Any:
        return n_max(obs.own_offer) + 1


def test_full_return_under_equal_by_hand() -> None:
    games = simulate(ReturnAll(), EXACT_MECHANISMS["Equal"], 3, np.random.default_rng(0))
    # Deployed equal split of 200: 0.25 * (200 - 0.001) = 49.99975, rounded to 4 decimals
    # (49.9997 in float32, as recorded); each player returns 49, so
    # R' = min(200, 200 - 4 * 49.9997 + 1.4 * 196) = 200 in every round.
    assert np.allclose(games.offers, 49.99975, atol=1e-4)
    assert np.all(games.actions == 49.0)
    assert np.all(games.pool == 200.0)
    m = games.game_metrics()
    assert m["surplus"] == pytest.approx(np.full(3, 0.99975), abs=1e-4)
    assert m["gini"] == pytest.approx(np.zeros(3), abs=1e-12)
    assert m["depletion_round"].tolist() == [40.0, 40.0, 40.0]
    assert m["players_active"].tolist() == [4.0, 4.0, 4.0]


def test_rational_control_depletes_in_round_two() -> None:
    games = simulate(B0Rational(), EXACT_MECHANISMS["Proportional"], 2, np.random.default_rng(0))
    # Round 1 keeps everything: R' = max(0.01, 200 - 199.999...) = 0.01 (the floor);
    # offers then are (0.01 - 0.001) / 4 ~ 0.0022 < 1, so everybody is excluded.
    assert games.pool[:, 1] == pytest.approx([0.01, 0.01], abs=1e-12)
    m = games.game_metrics()
    assert m["depletion_round"].tolist() == [2.0, 2.0]
    assert m["players_active"].tolist() == [0.0, 0.0]
    expected = (games.offers[0, 0, 0] + 39 * games.offers[0, 1, 0]) / 40
    assert m["surplus"][0] == pytest.approx(expected, rel=1e-12)
    assert m["surplus"][0] == pytest.approx((49.99975 + 39 * 0.00225) / 40, abs=2e-4)


def test_infeasible_reciprocation_stops_the_run() -> None:
    with pytest.raises(InfeasibleGameError):
        simulate(Cheater(), EXACT_MECHANISMS["Equal"], 1, np.random.default_rng(0))


def test_arbitrary_lengths_are_truncated_per_game() -> None:
    games = simulate(
        ReturnAll(),
        BaselineMechanism("Equal", 1.0),
        3,
        np.random.default_rng(0),
        n_rounds=[25, 31, 49],
    )
    assert games.n_rounds.tolist() == [25, 31, 49]
    assert np.isnan(games.pool[0, 25:]).all() and not np.isnan(games.pool[2]).any()
    assert games.game_metrics()["depletion_round"].tolist() == [40.0, 40.0, 40.0]


def test_interpolating_uses_the_quantised_weight() -> None:
    mech = EXACT_MECHANISMS["Interpolating"]
    pool = np.array([180.0, 200.0])
    prev = np.array([[10.0, 0.0, 0.0, 10.0], [10.0, 0.0, 0.0, 10.0]])
    offers, _ = mech.offers(None, pool, None, prev, 3)
    # w(180) = round(0.9^22, 1) = round(0.0985, 1) = 0.1; w(200) = 1 (equal split).
    share_180 = np.array([0.1 / 4 + 0.9 * 0.5, 0.1 / 4, 0.1 / 4, 0.1 / 4 + 0.9 * 0.5])
    assert offers[0] == pytest.approx(share_180 * 179.999, abs=2e-4)
    assert offers[1] == pytest.approx(np.full(4, 199.999 / 4), abs=2e-4)


REPLAY_TOLERANCE = 10 * 1e-4
"""Coins. The deployed software computed offers in float32 and rounded them to 4
decimals; a float64 simulation whose pool differs from the recorded float32 pool by
float32 resolution can flip a 4th-decimal rounding (1e-4 per offer). The replayed paths
must agree to within 10 such flips, far below the 1-coin action resolution."""


@pytest.mark.skipif(not HAS_DATA, reason="parsed dataset absent (run `just citizens-data`)")
@pytest.mark.parametrize(
    ("mechanism", "key"),
    [
        ("Equal Baseline", "Equal"),
        ("Mixed Baseline", "Mixed"),
        ("Proportional Baseline", "Proportional"),
    ],
)
def test_replay_reproduces_recorded_training_games(mechanism: str, key: str) -> None:
    """Every non-legacy Exp 1/2 training game of the mechanism, replayed closed-loop."""
    training = data.load_training(PROCESSED)
    human = training.select(training.is_human & np.isin(training.cohort, ["Exp 1", "Exp 2"]))
    candidates = human.where(mechanism=mechanism)
    legacy = np.isin(candidates.launch_id, sorted(validation.LEGACY_REMAINDER_DROPPED))
    games = candidates.select(~legacy)
    assert games.n_games >= 36
    recorded = game_metrics(games.rewards, games.pool, games.players_kept_active, games.n_rounds)
    for g in range(games.n_games):
        n = int(games.n_rounds[g])
        # Replay stream b = i (one game): player i's recorded actions over the rounds.
        actions = games.actions[g, :n].T.astype(np.int64)
        sim = simulate(
            Replay(actions), EXACT_MECHANISMS[key], 1, np.random.default_rng(0), n_rounds=n
        )
        assert np.abs(sim.offers[0] - games.offers[g, :n]).max() <= REPLAY_TOLERANCE
        assert np.abs(sim.pool[0] - games.pool[g, :n]).max() <= REPLAY_TOLERANCE
        assert np.array_equal(sim.actions[0], games.actions[g, :n])
        m = sim.game_metrics()
        assert m["surplus"][0] == pytest.approx(recorded["surplus"][g], abs=REPLAY_TOLERANCE)
        assert m["gini"][0] == pytest.approx(recorded["gini"][g], abs=1e-4)
        assert m["depletion_round"][0] == recorded["depletion_round"][g]
        assert m["players_active"][0] == recorded["players_active"][g]
