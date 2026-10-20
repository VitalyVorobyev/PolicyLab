"""The common-pool game of Koster et al. (2025), Nature Communications 16:2824 (pure).

Four players share a pool R (initially and at most 200 coins). Each round the mechanism
offers e_i >= 0 with sum(e) <= R; player i returns a reciprocation 0 <= c_i <= e_i and
keeps the surplus e_i - c_i. Only what is reciprocated grows:

    R' = min(200, R - sum(e) + 1.4 * sum(c)),

and, as the recorded data show (C1 audit; not stated in the paper), the experiment
software never lets the pool fall below 0.01: `next_pool` applies that floor.

Human players choose whole coins (a slider from 0 to the offer), so a human c_i is an
integer no larger than floor(e_i); an offer below 1 is an exclusion and forces c_i = 0.
The pool counts as depleted once R < 1 (the notebook's `find_first_depletion_trial`).
Exps 1-3 last 40 rounds; Exp 4 games last at least 25 rounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

N_PLAYERS: Final = 4
MAX_POOL: Final = 200.0
"""Initial pool R_0 and carrying capacity."""
GROWTH_FACTOR: Final = 1.4
"""Each reciprocated coin returns 1 + r = 1.4 coins to the pool (r = 0.4)."""
POOL_FLOOR: Final = 0.01
"""Smallest recorded pool: R' = max(0.01, ...) in every human and BC game (C1 audit)."""
DEPLETION_THRESHOLD: Final = 1.0
"""The pool is depleted when R < 1."""
EXCLUSION_THRESHOLD: Final = 1.0
"""An offer below 1 is an exclusion: the player cannot reciprocate."""
N_ROUNDS: Final = 40
"""Rounds in Exps 1-3 (not disclosed to players)."""
OFFER_SUM_RTOL: Final = 1e-12
"""Relative slack of the budget check sum(e) <= R.

An offer vector computed as R * share_i with shares summing to 1 in exact arithmetic
can exceed R by a few float64 ulps: the shares, the 4 products and the 3 additions each
round once, so sum(e) - R is a small multiple of eps R (eps = 2.2e-16; observed
<= 4.6e-16 R over 2e5 random cases). 1e-12 R covers that with a wide margin and is
eight orders of magnitude below the 1e-4 coin resolution of recorded offers, so no
offer that really over-spends the pool is accepted.
"""
OFFER_SUM_ATOL: Final = 1e-12
"""Absolute slack (coins) of the budget check, for pools near zero (floor disabled)."""


def next_pool(
    pool: npt.ArrayLike,
    offers: npt.ArrayLike,
    reciprocations: npt.ArrayLike,
    *,
    floor: float = POOL_FLOOR,
) -> FloatArray:
    """R' = max(floor, min(200, R - sum_i e_i + 1.4 sum_i c_i)); players on the last axis."""
    r = np.asarray(pool, dtype=np.float64)
    e = np.asarray(offers, dtype=np.float64)
    c = np.asarray(reciprocations, dtype=np.float64)
    raw = r - e.sum(axis=-1) + GROWTH_FACTOR * c.sum(axis=-1)
    return np.maximum(floor, np.minimum(MAX_POOL, raw))


def surplus(offers: npt.ArrayLike, reciprocations: npt.ArrayLike) -> FloatArray:
    """Per-player surplus (reward) e_i - c_i."""
    return np.asarray(offers, dtype=np.float64) - np.asarray(reciprocations, dtype=np.float64)


def is_depleted(pool: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    return np.asarray(pool, dtype=np.float64) < DEPLETION_THRESHOLD


def is_excluded(offers: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    return np.asarray(offers, dtype=np.float64) < EXCLUSION_THRESHOLD


def max_human_reciprocation(offers: npt.ArrayLike) -> FloatArray:
    """Largest whole-coin reciprocation: floor(e), hence 0 for an exclusion (e < 1)."""
    return np.floor(np.asarray(offers, dtype=np.float64))


def offers_valid(
    pool: float,
    offers: npt.ArrayLike,
    *,
    rtol: float = OFFER_SUM_RTOL,
    atol: float = OFFER_SUM_ATOL,
) -> bool:
    """e_i >= 0 and sum(e) <= R (1 + rtol) + atol (float rounding slack, see OFFER_SUM_RTOL)."""
    e = np.asarray(offers, dtype=np.float64)
    return bool(np.all(e >= 0.0)) and float(e.sum()) <= pool * (1.0 + rtol) + atol


def reciprocations_valid(
    offers: npt.ArrayLike, reciprocations: npt.ArrayLike, *, integer: bool
) -> bool:
    """0 <= c_i <= e_i; for humans also c_i integral (so c_i <= floor(e_i))."""
    e = np.asarray(offers, dtype=np.float64)
    c = np.asarray(reciprocations, dtype=np.float64)
    if not (np.all(c >= 0.0) and np.all(c <= e)):
        return False
    return not integer or bool(np.all(c == np.floor(c)))


@dataclass(frozen=True, slots=True)
class GameState:
    """Pool before allocation, the previous round's reciprocations and the round index."""

    pool: float = MAX_POOL
    prev_reciprocations: tuple[float, ...] | None = None  # None before round 0
    round_index: int = 0

    @property
    def depleted(self) -> bool:
        return self.pool < DEPLETION_THRESHOLD


@dataclass(frozen=True, slots=True)
class StepResult:
    state: GameState
    surplus: tuple[float, ...]


def step(
    state: GameState,
    offers: npt.ArrayLike,
    reciprocations: npt.ArrayLike,
    *,
    integer: bool = False,
) -> StepResult:
    """Apply one round; raises ValueError on an invalid offer or reciprocation."""
    e = np.asarray(offers, dtype=np.float64)
    c = np.asarray(reciprocations, dtype=np.float64)
    if e.shape != (N_PLAYERS,) or c.shape != (N_PLAYERS,):
        raise ValueError(f"offers and reciprocations need shape ({N_PLAYERS},)")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(c))):
        raise ValueError("non-finite offer or reciprocation")
    if not offers_valid(state.pool, e):
        raise ValueError("offers must be >= 0 and sum to at most the pool")
    if not reciprocations_valid(e, c, integer=integer):
        raise ValueError("reciprocations must satisfy 0 <= c <= e (integral for humans)")
    new_pool = float(next_pool(state.pool, e, c))
    if not math.isfinite(new_pool):
        raise ValueError("non-finite pool")
    return StepResult(
        state=GameState(
            pool=new_pool,
            prev_reciprocations=tuple(float(v) for v in c),
            round_index=state.round_index + 1,
        ),
        surplus=tuple(float(v) for v in surplus(e, c)),
    )
