"""Closed-loop simulation of citizen models under a mechanism (pure; protocol 5, 7, 8).

Each simulated game starts at R_0 = 200. In round t the mechanism offers e_t from what it
observes (pool, previous offers and reciprocations), every player (one model stream per
game and player, own-first view) draws an integer reciprocation from its model, and the
pool follows R' = max(0.01, min(200, R - sum e + 1.4 sum c)) (`game.next_pool`). Every
round is checked for feasibility (protocol 5): finite values, e >= 0, sum e <= R + 1e-9,
integer 0 <= c <= floor(e), c = 0 when e < 1. A violation raises `InfeasibleGameError`
(stop rule K2): an invalid game is a bug and is never dropped.

Mechanisms implement `Mechanism`; the exact baselines and the Interpolating rule use the
deployed offer rule (D33). A learned surrogate (protocol 6.4) plugs in through the same
interface. The game metrics of protocol 8 are computed per game with the notebook
definitions of `phasespace.citizens.metrics`; players active at the end is
#{i : e_{i,T} >= 1} in the last round, which equals the CSV's `players_kept_active` on
every training row (protocol 13 pre-freeze check, `tests/citizens`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from phasespace.citizens import config, metrics
from phasespace.citizens.game import (
    EXCLUSION_THRESHOLD,
    MAX_POOL,
    N_PLAYERS,
    next_pool,
)
from phasespace.citizens.mechanisms import (
    BASELINE_WEIGHTS,
    DEFAULT_OFFER_RULE,
    OfferRule,
    baseline_offers,
    interpolating_weight,
)
from phasespace.citizens.models.base import CitizenModel, Observation, rotate

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


class InfeasibleGameError(RuntimeError):
    """A simulated round violated protocol 5 (stop rule K2)."""


class Mechanism[S](Protocol):
    """An allocation rule for G games at once (offers (G, 4) from the pool and history).

    `name` is a read-only property, so frozen dataclasses (fields are read-only) and
    classes with a computed name both satisfy the protocol.
    """

    @property
    def name(self) -> str: ...

    def initial_state(self, n_games: int) -> S: ...

    def offers(
        self,
        state: S,
        pool: FloatArray,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
        round_index: int,
    ) -> tuple[FloatArray, S]: ...


@dataclass(frozen=True, slots=True)
class BaselineMechanism:
    """Equal (w = 1), Mixed (0.5) or Proportional (0), deployed offer rule by default."""

    name: str
    w: float
    rule: OfferRule = DEFAULT_OFFER_RULE

    def initial_state(self, n_games: int) -> None:
        return None

    def offers(
        self,
        state: None,
        pool: FloatArray,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
        round_index: int,
    ) -> tuple[FloatArray, None]:
        return baseline_offers(pool, prev_reciprocations, self.w, rule=self.rule), None


@dataclass(frozen=True, slots=True)
class InterpolatingMechanism:
    """w = round((R/200)^22, 1) (deployed, quantised) with the deployed offer rule."""

    name: str = "Interpolating"
    rule: OfferRule = DEFAULT_OFFER_RULE

    def initial_state(self, n_games: int) -> None:
        return None

    def offers(
        self,
        state: None,
        pool: FloatArray,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
        round_index: int,
    ) -> tuple[FloatArray, None]:
        w = interpolating_weight(pool)
        return baseline_offers(pool, prev_reciprocations, w, rule=self.rule), None


EXACT_MECHANISMS: Final = {
    "Equal": BaselineMechanism("Equal", BASELINE_WEIGHTS["Equal Baseline"]),
    "Mixed": BaselineMechanism("Mixed", BASELINE_WEIGHTS["Mixed Baseline"]),
    "Proportional": BaselineMechanism("Proportional", BASELINE_WEIGHTS["Proportional Baseline"]),
    "Interpolating": InterpolatingMechanism(),
}
"""The mechanisms simulated exactly (protocol 3); M1/M2 need a surrogate."""


@dataclass(frozen=True, slots=True)
class SimulatedGames:
    offers: FloatArray  # (S, T, 4)
    actions: FloatArray  # (S, T, 4) integer-valued
    pool: FloatArray  # (S, T) before allocation
    n_rounds: IntArray  # (S,)

    @property
    def n_games(self) -> int:
        return int(self.n_rounds.shape[0])

    def round_mask(self) -> npt.NDArray[np.bool_]:
        return np.arange(self.pool.shape[1], dtype=np.int64)[None, :] < self.n_rounds[:, None]

    def players_active(self) -> IntArray:
        """(S, T) #{i : e_{i,t} >= 1} (the simulated `players_kept_active`)."""
        return metrics.players_offered_at_least_one(self.offers)

    def game_metrics(self) -> dict[str, FloatArray]:
        return game_metrics(
            self.offers - self.actions, self.pool, self.players_active(), self.n_rounds
        )


def game_metrics(
    rewards: npt.ArrayLike,
    pool: npt.ArrayLike,
    players_active: npt.ArrayLike,
    n_rounds: npt.ArrayLike,
) -> dict[str, FloatArray]:
    """The 4 protocol-8 metrics per game (notebook definitions, `metrics`)."""
    return {
        "surplus": metrics.surplus(rewards, n_rounds),
        "gini": metrics.reward_gini(rewards, n_rounds),
        "depletion_round": metrics.depletion_round(pool, n_rounds).astype(np.float64),
        "players_active": metrics.last_players_active(players_active, n_rounds).astype(np.float64),
    }


def _check_round(pool: FloatArray, e: FloatArray, c: IntArray, t: int) -> None:
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(pool))):
        raise InfeasibleGameError(f"round {t}: non-finite offer or pool")
    if np.any(e < 0.0) or np.any(e.sum(axis=1) > pool + config.SIM_FEASIBILITY_ATOL):
        raise InfeasibleGameError(f"round {t}: offers negative or above the pool")
    if np.any(c < 0) or np.any(c > np.floor(e)):
        raise InfeasibleGameError(f"round {t}: reciprocation outside 0..floor(e)")
    if np.any(c[e < EXCLUSION_THRESHOLD] != 0):
        raise InfeasibleGameError(f"round {t}: excluded player reciprocated")


def simulate(
    model: CitizenModel[Any],
    mechanism: Mechanism[Any],
    n_games: int,
    rng: np.random.Generator,
    *,
    n_rounds: int | npt.ArrayLike = config.SIM_ROUNDS,
) -> SimulatedGames:
    """S closed-loop games; `n_rounds` is one length or one length per game.

    Streams are stream b = 4 g + i (game g, player i). Games shorter than the longest are
    simulated on (their streams are independent of the other games) and truncated.
    """
    lengths = np.broadcast_to(np.asarray(n_rounds, dtype=np.int64), (n_games,)).copy()
    if np.any(lengths < 1):
        raise ValueError("every game needs at least one round")
    t_max = int(lengths.max())
    offers = np.zeros((n_games, t_max, N_PLAYERS))
    actions = np.zeros((n_games, t_max, N_PLAYERS))
    pools = np.zeros((n_games, t_max))
    pool = np.full(n_games, MAX_POOL)
    prev_e: FloatArray | None = None
    prev_c: FloatArray | None = None
    m_state = mechanism.initial_state(n_games)
    c_state = model.initial_state(n_games * N_PLAYERS, rng)
    for t in range(t_max):
        e, m_state = mechanism.offers(m_state, pool, prev_e, prev_c, t)
        e = np.asarray(e, dtype=np.float64)
        prev_view = np.zeros((n_games, N_PLAYERS)) if prev_c is None else prev_c
        obs = Observation(
            offers=rotate(e).reshape(n_games * N_PLAYERS, N_PLAYERS),
            prev_reciprocations=rotate(prev_view).reshape(n_games * N_PLAYERS, N_PLAYERS),
            pool=np.repeat(pool, N_PLAYERS),
            round_index=t,
        )
        pmf, c_state = model.step(c_state, obs)
        c = model.sample(pmf, obs, rng).reshape(n_games, N_PLAYERS)
        _check_round(pool, e, c, t)
        offers[:, t], actions[:, t], pools[:, t] = e, c, pool
        c_float = c.astype(np.float64)
        pool = next_pool(pool, e, c_float)
        prev_e, prev_c = e, c_float
    keep = np.arange(t_max)[None, :] < lengths[:, None]
    return SimulatedGames(
        offers=np.where(keep[..., None], offers, np.nan),
        actions=np.where(keep[..., None], actions, np.nan),
        pool=np.where(keep, pools, np.nan),
        n_rounds=lengths,
    )
