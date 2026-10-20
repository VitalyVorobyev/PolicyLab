"""Planner versus B4: LLM citizens playing games whose offers come from a trained planner.

`PlannerGameMechanism` is a `llm.runner.GameMechanism`: each round it runs the planner's
deterministic policy (shares = softmax(mu), the evaluation policy) on the game so far,
one game at a time, so the LLM runner's game loop, prompts (`b4-prompt-v2`), CLI flags,
retries and archive are exactly those of the frozen B4 runs; only the source of the
offers changes. B4 is an evaluation-only population (D35): nothing here trains on it.

Call budget: every player-round with an offer >= 1 is one call (plus retries). The
calibration games (citizens-c2-b4-calibration-v2, Proportional) measured 106.1 calls per
40-round game; a planner that keeps all four players active needs up to 160
(`calls_estimate`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import equinox as eqx
import numpy as np
from jax import Array

from phasespace.citizens.game import N_PLAYERS
from phasespace.citizens.llm import runner
from phasespace.citizens.llm.backend import Backend
from phasespace.citizens.rl import c4_config, twin
from phasespace.citizens.rl.citizens import MechObs
from phasespace.citizens.rl.planner import Planner, player_features

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

MAX_CALLS_PER_GAME: Final = N_PLAYERS * 40


@eqx.filter_jit
def _planner_offers(planner: Planner, h: Array, obs: MechObs) -> tuple[Array, Array]:
    h_new, mu = planner.policy_step(h, player_features(obs))
    return planner.offers(obs.pool, mu), h_new


@dataclass(frozen=True)
class PlannerGameMechanism:
    """A trained planner's deterministic policy, one game at a time (`GameMechanism`).

    The state is (round index, the policy's hidden state (1, 4, H)).
    """

    name: str
    planner: Planner = field(repr=False)

    def start(self) -> object:
        with twin.x64():
            return (0, self.planner.initial_state(1)[0])

    def offers(
        self,
        state: object,
        pool: float,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
    ) -> tuple[FloatArray, object]:
        if not isinstance(state, tuple):
            raise TypeError("state must come from start()")
        t_raw, h_raw = cast(tuple[object, object], state)
        t, h = int(cast(int, t_raw)), cast(Array, h_raw)
        with twin.x64():
            zeros = np.zeros(N_PLAYERS)
            obs = MechObs(
                pool=twin.f64([pool]),
                prev_offers=twin.f64([zeros if prev_offers is None else prev_offers]),
                prev_reciprocations=twin.f64(
                    [zeros if prev_reciprocations is None else prev_reciprocations]
                ),
                t=twin.asarray(t),
            )
            e, h_new = _planner_offers(self.planner, h, obs)
            out = np.asarray(e, dtype=np.float64)[0]
        return out, (t + 1, h_new)


def calls_estimate(
    decisions_per_game: float | None = None, games: int = c4_config.B4_PLANNER_GAMES
) -> dict[str, float]:
    """Expected B4 calls for `games` planner games: decisions per game (offers >= 1, e.g.
    measured on the planner's own model; default: the calibration's 106.1 calls per game,
    which already include retries) times (1 + the calibration's retry fraction)."""
    retry = c4_config.B4_RETRY_FRACTION_CALIBRATION
    if decisions_per_game is None:
        per_game = c4_config.B4_CALLS_PER_GAME_CALIBRATION
    else:
        per_game = decisions_per_game * (1.0 + retry)
    return {
        "calls_per_game": per_game,
        "games": float(games),
        "calls": per_game * games,
        "calls_worst_case": MAX_CALLS_PER_GAME * (1.0 + retry) * games,
    }


def run_planner_games(
    mechanism: PlannerGameMechanism,
    backend: Backend,
    runs_dir: Path,
    run_id: str,
    games: int,
    spec: Callable[[int], runner.GameSpec],
    *,
    variant: runner.Variant,
    workers: int = runner.WORKERS,
    log: Callable[[str], None] | None = None,
    infra_backoff: tuple[float, ...] = runner.INFRA_BACKOFF_S,
) -> tuple[runner.RunConfig, list[runner.GameOutcome]]:
    """Play `games` resumable B4 games against the planner (the frozen runner and prompts)."""
    config = runner.RunConfig(
        run_id=run_id,
        label="c4-planner",
        variant=variant,
        mechanism=mechanism.name,
        target_games=games,
        n_rounds=40,
        workers=workers,
        max_retries=runner.MAX_RETRIES,
        discard_limit=runner.discard_limit(games),
        seed=runner.B4_SEED,
        backend=backend.describe(),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )
    store = runner.prepare_work_dir(runs_dir / runner.WORK_DIR / run_id, config)
    outcomes = asyncio.run(
        runner.run_games(
            config, spec, mechanism, backend, store, log=log, infra_backoff=infra_backoff
        )
    )
    return config, outcomes


def describe_planner(run_id: str, manifest_sha256: str, retention: bool) -> dict[str, Any]:
    return {
        "planner_run": run_id,
        "planner_manifest_sha256": manifest_sha256,
        "policy": "deterministic (shares = softmax(mu))",
        "retention": retention,
    }
