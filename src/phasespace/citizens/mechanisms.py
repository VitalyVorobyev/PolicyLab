"""Allocation mechanisms of Koster et al. (2025) (pure, deterministic given a seed).

**Paper rules.** Round 1 is an equal split. Afterwards the baseline family offers

    e_i = w R / 4 + (1 - w) R c_{i,t-1} / sum_j c_{j,t-1},

with Equal w = 1, Mixed w = 0.5, Proportional w = 0, and the Interpolating baseline
w = (R / 200)^k, k = 22. The Random baseline draws 5 shares (4 players and the pool)
from Dirichlet(1).

**Rules established from the data (C1 audit, `phasespace.citizens.validation`).**

- *Sum of previous reciprocations zero.* The paper does not say. In the data
  sum(c_{t-1}) = 0 occurs only when the pool sits at its 0.01 floor (everyone kept
  everything, so R' = max(0.01, remainder)); every such row is an equal split.
  `baseline_shares` therefore uses the equal split when sum(c_{t-1}) = 0.
- *Interpolating weight is quantised.* The recorded offers use
  w = round((R / 200)^22, 1), i.e. w in {0, 0.1, ..., 1}: no other rounding of the
  weight, and no other k near 22, reproduces them. `interpolating_weight` quantises
  by default.
- *The deployed software withheld 0.001 and rounded.* Recorded baseline offers equal
  round(share_i * (R - 0.001), 4) computed in float32: about 0.001 coins (0.0008 to
  0.0012 after the per-player rounding) stays in the pool every round, independent
  of R. `distribute_pool_ratio` is 1.0 in every human baseline row and empty in BC
  rows, so it never acts as a ratio in this dataset.

**Which rule is simulated (D33).** `baseline_offers` defaults to the *deployed* rule
(`deployed_offers`: 0.001 withheld, 4-decimal rounding, float32 shares), so simulated
games (C2-C4) face exactly the offers the human players experienced. The paper's
idealised formula (whole pool, float64, no rounding) stays available as
`paper_offers` / ``rule="paper"``; the two differ by at most about 0.0011 coins per
player.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from phasespace.citizens.game import MAX_POOL, N_PLAYERS

type FloatArray = npt.NDArray[np.float64]
type OfferRule = Literal["deployed", "paper"]

W_EQUAL: Final = 1.0
W_MIXED: Final = 0.5
W_PROPORTIONAL: Final = 0.0
K_INTERPOLATING: Final = 22.0
INTERPOLATING_DECIMALS: Final = 1
"""Decimals of the deployed Interpolating weight (data: w in {0, 0.1, ..., 1})."""
RECORDED_WITHHELD: Final = 0.001
"""Coins the deployed baseline software kept back from the pool before splitting it."""
RECORDED_DECIMALS: Final = 4
"""Decimals of the recorded (deployed) offers."""
DEFAULT_OFFER_RULE: Final[OfferRule] = "deployed"
"""Simulations use the deployed rule (D33); "paper" is the idealised alternative."""

BASELINE_WEIGHTS: Final = {
    "Equal Baseline": W_EQUAL,
    "Mixed Baseline": W_MIXED,
    "Proportional Baseline": W_PROPORTIONAL,
}


def baseline_shares(prev_reciprocations: npt.ArrayLike | None, w: npt.ArrayLike) -> FloatArray:
    """Shares w/4 + (1 - w) c_i / sum(c); equal split in round 1 (None) or if sum(c) = 0.

    `prev_reciprocations` has players on the last axis; `w` broadcasts against the
    leading axes. The shares sum to 1.
    """
    weight = np.asarray(w, dtype=np.float64)
    if prev_reciprocations is None:
        return np.broadcast_to(
            np.full(N_PLAYERS, 1.0 / N_PLAYERS), (*weight.shape, N_PLAYERS)
        ).copy()
    c = np.asarray(prev_reciprocations, dtype=np.float64)
    if c.shape[-1] != N_PLAYERS:
        raise ValueError(f"reciprocations need {N_PLAYERS} players on the last axis")
    if np.any(c < 0.0):
        raise ValueError("reciprocations must be non-negative")
    total = c.sum(axis=-1, keepdims=True)
    positive = total > 0.0
    proportional = np.divide(c, total, out=np.full_like(c, 1.0 / N_PLAYERS), where=positive)
    wk = weight[..., None]
    return wk / N_PLAYERS + (1.0 - wk) * proportional


def paper_offers(
    pool: npt.ArrayLike, prev_reciprocations: npt.ArrayLike | None, w: npt.ArrayLike
) -> FloatArray:
    """The paper's idealised baseline offers R * share_i (the whole pool is distributed)."""
    r = np.asarray(pool, dtype=np.float64)
    return r[..., None] * baseline_shares(prev_reciprocations, np.broadcast_to(w, r.shape))


def baseline_offers(
    pool: npt.ArrayLike,
    prev_reciprocations: npt.ArrayLike | None,
    w: npt.ArrayLike,
    *,
    rule: OfferRule = DEFAULT_OFFER_RULE,
) -> FloatArray:
    """Baseline offers under the deployed rule (default) or the paper's idealised rule."""
    if rule == "deployed":
        return deployed_offers(pool, prev_reciprocations, w)
    if rule == "paper":
        return paper_offers(pool, prev_reciprocations, w)
    raise ValueError(f"unknown offer rule {rule!r}")


def interpolating_weight(
    pool: npt.ArrayLike, k: float = K_INTERPOLATING, *, quantize: bool = True
) -> FloatArray:
    """w = (R / 200)^k, rounded to one decimal as deployed unless `quantize` is False."""
    w = (np.asarray(pool, dtype=np.float64) / MAX_POOL) ** k
    return np.round(w, INTERPOLATING_DECIMALS) if quantize else w


def deployed_offers(
    pool: npt.ArrayLike,
    prev_reciprocations: npt.ArrayLike | None,
    w: npt.ArrayLike,
    *,
    withheld: float = RECORDED_WITHHELD,
) -> FloatArray:
    """Offers as the experiment software made and recorded them (float32 arithmetic).

    round(share_i * (R - withheld), 4) with the shares of `baseline_shares` computed in
    float32 and the rounding applied to the float32 product; returned as float64. The
    deployed `withheld` is 0.001 (other values exist only to test that the data pin it
    down). Requires R >= withheld, which the 0.01 pool floor guarantees.
    """
    f32 = np.float32
    r = np.asarray(pool, dtype=np.float64)
    if np.any(r < withheld):
        raise ValueError(f"deployed offers need a pool of at least {withheld} coins")
    weight = np.broadcast_to(np.asarray(w, dtype=np.float64), r.shape).astype(f32)
    if prev_reciprocations is None:
        share = np.full((*r.shape, N_PLAYERS), 1.0 / N_PLAYERS, dtype=f32)
    else:
        c = np.asarray(prev_reciprocations, dtype=np.float64).astype(f32)
        total = c.sum(axis=-1, keepdims=True, dtype=f32)
        positive = total > 0
        proportional = np.divide(
            c, total, out=np.full_like(c, f32(1.0 / N_PLAYERS)), where=positive
        )
        wk = weight[..., None]
        share = wk / f32(N_PLAYERS) + (f32(1.0) - wk) * proportional
    distributed = (r.astype(f32) - f32(withheld))[..., None]
    return np.round(share * distributed, RECORDED_DECIMALS).astype(np.float64)


def random_offers(pool: float, rng: np.random.Generator) -> FloatArray:
    """Random baseline: Dirichlet(1) over 4 players and the pool, times R (pass a seeded rng)."""
    shares = rng.dirichlet(np.ones(N_PLAYERS + 1))
    return float(pool) * shares[:N_PLAYERS]
