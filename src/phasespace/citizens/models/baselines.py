"""Reference models: B0 rational control, M0 marginal, P0 persistence, and a replay model.

**B0** (protocol 6.3, no parameters). Closed loop: c = 0 always (the myopic payoff
maximiser keeps every coin). Likelihood (so that the control has a finite score):
P(c) = (1 - eps) 1[c = 0] + eps / (n + 1), eps = 0.05, n = floor(e).

**M0** (protocol 6.2). The fit-split frequencies of the 10 floor bins over non-forced
decisions, plus 0.5 per bin, normalised: p_b = (k_b + 0.5) / (K + 5); mapped to integers
by the protocol 4 map. It is also simulated closed-loop as the no-dynamics control
M0-sim (non-gating).

**P0** (protocol 6.2, diagnostic). With probability q the player repeats its previous
floor bin b' = b(c_{t-1}; e_{t-1}) when t > 0 and e_{t-1} >= 1, otherwise M0:
p_b = q 1[b = b'] + (1 - q) M0_b (p = M0 when there is no previous bin); q by MLE.

**Replay** (tests only): the point mass at recorded actions, so the simulator can be
checked against a recorded game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize

from phasespace.citizens import config
from phasespace.citizens.game import EXCLUSION_THRESHOLD
from phasespace.citizens.models.base import (
    K_MAX,
    CitizenModel,
    FloatArray,
    IntArray,
    Observation,
    Streams,
    bin_grid,
    bin_index,
    bins_to_integer_pmf,
    forced_pmf,
    n_max,
)


class B0Rational(CitizenModel[None]):
    name = "B0"

    def __init__(self, eps: float = config.B0_EPS) -> None:
        self.eps = eps

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> None:
        return None

    def step(self, state: None, obs: Observation) -> tuple[FloatArray, None]:
        e = obs.own_offer
        n = n_max(e)
        c = np.arange(K_MAX)[None, :]
        pmf = np.where(c <= n[:, None], self.eps / (n[:, None] + 1.0), 0.0)
        pmf[:, 0] += 1.0 - self.eps
        forced = e < EXCLUSION_THRESHOLD
        pmf[forced] = forced_pmf(int(forced.sum()))
        return pmf, None

    def sample(self, pmf: FloatArray, obs: Observation, rng: np.random.Generator) -> IntArray:
        """The closed-loop control keeps everything: c = 0 (no randomness)."""
        return np.zeros(obs.n_streams, dtype=np.int64)

    def describe(self) -> dict[str, Any]:
        return {"eps": self.eps, "closed_loop": "c = 0"}


def floor_bin_counts(streams: Streams) -> FloatArray:
    """(10,) counts of the observed floor bins over the scored decisions of `streams`."""
    scored = streams.scored
    b = bin_index(streams.actions[scored], streams.own_offer[scored], "floor10")
    return np.bincount(b, minlength=config.N_FLOOR_BINS).astype(np.float64)


def fit_m0(streams: Streams, smoothing: float = config.M0_SMOOTHING) -> FloatArray:
    counts = floor_bin_counts(streams) + smoothing
    return counts / counts.sum()


class M0Marginal(CitizenModel[None]):
    name = "M0"

    def __init__(self, p_bins: npt.ArrayLike) -> None:
        p = np.asarray(p_bins, dtype=np.float64)
        if p.shape != (config.N_FLOOR_BINS,) or np.any(p <= 0.0) or abs(p.sum() - 1.0) > 1e-12:
            raise ValueError("M0 needs 10 positive bin probabilities summing to 1")
        self.p_bins = p

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> None:
        return None

    def step(self, state: None, obs: Observation) -> tuple[FloatArray, None]:
        p = np.broadcast_to(self.p_bins, (obs.n_streams, config.N_FLOOR_BINS))
        return bins_to_integer_pmf(p, obs.own_offer, "floor10"), None

    def describe(self) -> dict[str, Any]:
        return {"p_bins": self.p_bins.tolist()}


@dataclass(frozen=True, slots=True)
class P0State:
    prev_offer: FloatArray  # (B,) own offer of the previous round (NaN before round 0)


def p0_bin_probs(p_m0: FloatArray, q: float, prev_bin: IntArray) -> FloatArray:
    """(B, 10) q 1[b = prev_bin] + (1 - q) M0, and M0 where prev_bin = -1."""
    p = np.broadcast_to((1.0 - q) * p_m0, (prev_bin.shape[0], p_m0.shape[0])).copy()
    has = prev_bin >= 0
    p[has, prev_bin[has]] += q
    p[~has] = p_m0
    return p


def previous_floor_bin(prev_action: FloatArray, prev_offer: FloatArray) -> IntArray:
    """b(c_{t-1}; e_{t-1}) if e_{t-1} >= 1, else -1 (also -1 before round 0: NaN offer)."""
    ok = prev_offer >= EXCLUSION_THRESHOLD
    b = bin_index(prev_action, np.where(ok, prev_offer, 1.0), "floor10")
    return np.where(ok, b, -1)


class P0Persistence(CitizenModel[P0State]):
    name = "P0"

    def __init__(self, p_bins: npt.ArrayLike, q: float) -> None:
        self.p_bins = np.asarray(p_bins, dtype=np.float64)
        self.q = float(q)

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> P0State:
        return P0State(prev_offer=np.full(n_streams, np.nan))

    def step(self, state: P0State, obs: Observation) -> tuple[FloatArray, P0State]:
        prev_bin = previous_floor_bin(obs.prev_reciprocations[:, 0], state.prev_offer)
        p = p0_bin_probs(self.p_bins, self.q, prev_bin)
        return bins_to_integer_pmf(p, obs.own_offer, "floor10"), P0State(obs.own_offer.copy())

    def describe(self) -> dict[str, Any]:
        return {"p_bins": self.p_bins.tolist(), "q": self.q}


def p0_mean_nll(q: float, p_m0: FloatArray, streams: Streams) -> float:
    """Mean NLL per scored decision of P0 with persistence q (vectorised, no step loop)."""
    scored = streams.scored
    prev_offer = np.full_like(streams.own_offer, np.nan)
    prev_offer[:, 1:] = streams.own_offer[:, :-1]
    prev_bin = previous_floor_bin(streams.prev_reciprocations[..., 0], prev_offer)[scored]
    e = streams.own_offer[scored]
    c = streams.actions[scored]
    grid = bin_grid(e, "floor10")
    rows = np.arange(e.shape[0], dtype=np.int64)
    b = bin_index(c, e, "floor10")
    p = p0_bin_probs(p_m0, q, prev_bin)
    z = np.where(grid.counts > 0, p, 0.0).sum(axis=1)
    log_p = np.log(p[rows, b]) - np.log(grid.counts[rows, b]) - np.log(z)
    return float(-log_p.mean())


def fit_p0(p_m0: FloatArray, streams: Streams) -> tuple[float, float]:
    """MLE of q in [0, 1] (bounded Brent); returns (q, mean NLL)."""
    result = optimize.minimize_scalar(
        lambda q: p0_mean_nll(float(q), p_m0, streams),
        bounds=(0.0, 1.0 - 1e-12),
        method="bounded",
        options={"xatol": 1e-10},
    )
    q = float(result.x)
    return q, p0_mean_nll(q, p_m0, streams)


@dataclass(frozen=True, slots=True)
class ReplayState:
    t: int


class Replay(CitizenModel[ReplayState]):
    """Point mass at recorded actions (B, T) (stream-major); for simulator tests."""

    name = "replay"

    def __init__(self, actions: npt.ArrayLike) -> None:
        self.actions = np.asarray(actions, dtype=np.int64)

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> ReplayState:
        if n_streams != self.actions.shape[0]:
            raise ValueError("replay needs one recorded stream per simulated stream")
        return ReplayState(t=0)

    def step(self, state: ReplayState, obs: Observation) -> tuple[FloatArray, ReplayState]:
        pmf = np.zeros((obs.n_streams, K_MAX))
        pmf[np.arange(obs.n_streams), self.actions[:, state.t]] = 1.0
        return pmf, ReplayState(t=state.t + 1)

    def describe(self) -> dict[str, Any]:
        return {"replay": True}
