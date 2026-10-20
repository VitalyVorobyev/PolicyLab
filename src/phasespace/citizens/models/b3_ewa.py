"""B3: EWA learning with realised forward reinforcement, and B23: EWA + conditional pull.

Protocol 6.3; experience-weighted attraction after Camerer & Ho (1999), Econometrica
67(4):827-874. Actions are the 11 levels a_j = j / 10 (fraction of the offer returned).

**Initial state.** A_j(0) = log(p0_j) / lambda, where p0 is the fit-split round-1 level
distribution (+0.5 per level); N(0) = N0. Hence the round-1 choice is exactly p0.

**Update.** After a non-forced round t (e_{i,t} >= 1) with chosen level
j* = l(c_{i,t}; e_{i,t}), at the start of round t + 1 (when e_{i,t+1} is seen):

    N(t+1)   = rho N(t) + 1,
    A_j(t+1) = [phi N(t) A_j(t)
                + (delta + (1 - delta) 1[j = j*]) e_{i,t} (1 - a_j) / 50
                + 1[j = j*] theta e_{i,t+1} / 50] / N(t+1).

The first payoff term is the myopic surplus of level j (foregone levels weighted by
delta); the theta term reinforces the chosen level by the next offer received, the
institution's realised response. A forced round (e < 1) leaves A and N unchanged.

**Choice.** B3: p_j = softmax_j(lambda A_j(t)). B23 adds a conditional-cooperation pull
towards others' previous fraction x3 (the B2 feature):
p_j proportional to exp(lambda A_j(t) - kappa (a_j - x3)^2 1[t > 0]).
The integer pmf follows from the 11-level map of protocol 4.

Parameters (phi, rho, delta, lambda, N0, theta[, kappa]) with bounds phi, rho, delta in
[0, 1], lambda in [0.01, 50], N0 in [0, 20], theta in [0, 5], kappa in [0, 50]; fitted by
L-BFGS-B from 10 starts uniform in the bounds (seeds 301-310 for B3, 311-320 for B23).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from phasespace.citizens import config
from phasespace.citizens.game import EXCLUSION_THRESHOLD
from phasespace.citizens.models.b2_condcoop import stream_features
from phasespace.citizens.models.base import (
    LEVELS,
    BoolArray,
    CitizenModel,
    FloatArray,
    IntArray,
    Observation,
    Streams,
    bin_grid,
    bin_index,
    bins_to_integer_pmf,
    log_sum_exp_masked,
)
from phasespace.citizens.models.optim import lbfgsb

B3_NAMES: Final = ("phi", "rho", "delta", "lambda", "n0", "theta")
B23_NAMES: Final = (*B3_NAMES, "kappa")


def bounds(hybrid: bool) -> list[tuple[float, float]]:
    out = [config.B3_BOUNDS[name] for name in B3_NAMES]
    return [*out, config.B23_KAPPA_BOUNDS] if hybrid else out


def round1_levels(streams: Streams, smoothing: float = config.EWA_ROUND1_SMOOTHING) -> FloatArray:
    """p0: level distribution of the scored round-1 (t = 0) decisions, + smoothing, normalised."""
    scored = streams.scored[:, 0]
    lv = bin_index(streams.actions[scored, 0], streams.own_offer[scored, 0], "level11")
    counts = np.bincount(lv, minlength=config.N_LEVELS).astype(np.float64) + smoothing
    return counts / counts.sum()


def ewa_update(
    attractions: FloatArray,
    experience: FloatArray,
    prev_offer: FloatArray,
    chosen: IntArray,
    offer: FloatArray,
    active: BoolArray,
    *,
    phi: float,
    rho: float,
    delta: float,
    theta: float,
) -> tuple[FloatArray, FloatArray]:
    """One EWA update of (A (B, 11), N (B,)) where `active` (the previous round non-forced)."""
    n_new = rho * experience + 1.0
    one_hot = np.zeros_like(attractions)
    rows = np.arange(attractions.shape[0], dtype=np.int64)
    one_hot[rows, np.clip(chosen, 0, config.N_LEVELS - 1)] = 1.0
    scale = config.EWA_PAYOFF_SCALE
    payoff = (delta + (1.0 - delta) * one_hot) * prev_offer[:, None] * (1.0 - LEVELS) / scale
    payoff = payoff + one_hot * theta * offer[:, None] / scale
    a_new = (phi * experience[:, None] * attractions + payoff) / n_new[:, None]
    return (
        np.where(active[:, None], a_new, attractions),
        np.where(active, n_new, experience),
    )


def choice_logits(
    attractions: FloatArray, lam: float, kappa: float, x3: FloatArray, t: int
) -> FloatArray:
    """lambda A_j - kappa (a_j - x3)^2 1[t > 0] (kappa = 0 for B3)."""
    logits = lam * attractions
    if t > 0 and kappa != 0.0:
        logits = logits - kappa * (LEVELS[None, :] - x3[:, None]) ** 2
    return logits


def _softmax(logits: FloatArray) -> FloatArray:
    z = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class EWAState:
    attractions: FloatArray  # (B, 11)
    experience: FloatArray  # (B,)
    prev_offers: FloatArray | None  # (B, 4) own first; None before round 0


def _others_fraction(prev_offers: FloatArray, prev_reciprocations: FloatArray) -> FloatArray:
    others_e = prev_offers[:, 1:].sum(axis=1)
    ok = others_e >= EXCLUSION_THRESHOLD
    return np.where(ok, prev_reciprocations[:, 1:].sum(axis=1) / np.where(ok, others_e, 1.0), 0.0)


class EWAModel(CitizenModel[EWAState]):
    """B3 (`hybrid=False`, 6 parameters) or B23 (`hybrid=True`, 7 parameters)."""

    def __init__(self, params: npt.ArrayLike, p0: npt.ArrayLike, *, hybrid: bool) -> None:
        p = np.asarray(params, dtype=np.float64)
        expected = config.B23_PARAMS if hybrid else config.B3_PARAMS
        if p.shape != (expected,):
            raise ValueError(f"{'B23' if hybrid else 'B3'} needs {expected} parameters")
        self.params = p
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.hybrid = hybrid
        self.name = "B23" if hybrid else "B3"

    @property
    def kappa(self) -> float:
        return float(self.params[6]) if self.hybrid else 0.0

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> EWAState:
        lam, n0 = float(self.params[3]), float(self.params[4])
        a0 = np.log(self.p0) / lam
        return EWAState(
            attractions=np.broadcast_to(a0, (n_streams, config.N_LEVELS)).copy(),
            experience=np.full(n_streams, n0),
            prev_offers=None,
        )

    def step(self, state: EWAState, obs: Observation) -> tuple[FloatArray, EWAState]:
        phi, rho, delta, lam, _, theta = (float(v) for v in self.params[:6])
        a, n = state.attractions, state.experience
        x3 = np.zeros(obs.n_streams)
        t = obs.round_index
        if t > 0 and state.prev_offers is not None:
            prev_own = state.prev_offers[:, 0]
            active = prev_own >= EXCLUSION_THRESHOLD
            chosen = bin_index(
                obs.prev_reciprocations[:, 0], np.where(active, prev_own, 1.0), "level11"
            )
            a, n = ewa_update(
                a,
                n,
                prev_own,
                chosen,
                obs.own_offer,
                active,
                phi=phi,
                rho=rho,
                delta=delta,
                theta=theta,
            )
            x3 = _others_fraction(state.prev_offers, obs.prev_reciprocations)
        p = _softmax(choice_logits(a, lam, self.kappa, x3, t))
        pmf = bins_to_integer_pmf(p, obs.own_offer, "level11")
        return pmf, EWAState(attractions=a, experience=n, prev_offers=obs.offers.copy())

    def describe(self) -> dict[str, Any]:
        names = B23_NAMES if self.hybrid else B3_NAMES
        return {
            "params": dict(zip(names, self.params.tolist(), strict=True)),
            "p0_levels": self.p0.tolist(),
        }


@dataclass(frozen=True, slots=True)
class FitData:
    """Per-stream sequences of the fit split, precomputed for the vectorised NLL."""

    offer: FloatArray  # (B, T) own offer
    scored: BoolArray  # (B, T)
    level: IntArray  # (B, T) observed level (0 where not scored)
    log_counts: FloatArray  # (B, T, 11) log m_j (0 where not scored)
    available: BoolArray  # (B, T, 11) m_j > 0 (all True where not scored)
    x3: FloatArray  # (B, T)
    p0: FloatArray  # (11,)

    @classmethod
    def from_streams(cls, streams: Streams, p0: FloatArray) -> FitData:
        scored = streams.scored
        b, t = scored.shape
        e = streams.own_offer
        level = np.zeros((b, t), dtype=np.int64)
        level[scored] = bin_index(streams.actions[scored], e[scored], "level11")
        counts = np.ones((b, t, config.N_LEVELS), dtype=np.int64)
        counts[scored] = bin_grid(e[scored], "level11").counts
        return cls(
            offer=e,
            scored=scored,
            level=level,
            log_counts=np.log(np.maximum(counts, 1)),
            available=counts > 0,
            x3=stream_features(streams)[..., 2],
            p0=p0,
        )


def log_likelihood_terms(params: npt.ArrayLike, data: FitData, *, hybrid: bool) -> FloatArray:
    """(B, T) log P(c_observed) of B3/B23 on the scored decisions (0 elsewhere)."""
    p = np.asarray(params, dtype=np.float64)
    phi, rho, delta, lam, n0, theta = (float(v) for v in p[:6])
    kappa = float(p[6]) if hybrid else 0.0
    b, t_max = data.scored.shape
    a = np.broadcast_to(np.log(data.p0) / lam, (b, config.N_LEVELS)).copy()
    n = np.full(b, n0)
    rows = np.arange(b, dtype=np.int64)
    out = np.zeros((b, t_max))
    for t in range(t_max):
        if t > 0:
            a, n = ewa_update(
                a,
                n,
                data.offer[:, t - 1],
                data.level[:, t - 1],
                data.offer[:, t],
                data.scored[:, t - 1],
                phi=phi,
                rho=rho,
                delta=delta,
                theta=theta,
            )
        logits = choice_logits(a, lam, kappa, data.x3[:, t], t)
        lp = (
            logits[rows, data.level[:, t]]
            - data.log_counts[rows, t, data.level[:, t]]
            - log_sum_exp_masked(logits, data.available[:, t])
        )
        out[:, t] = np.where(data.scored[:, t], lp, 0.0)
    return out


def mean_nll(params: npt.ArrayLike, data: FitData, hybrid: bool) -> float:
    total = log_likelihood_terms(params, data, hybrid=hybrid).sum()
    return float(-total / data.scored.sum())


@dataclass(frozen=True, slots=True)
class StartResult:
    seed: int
    x0: list[float]
    params: list[float]
    mean_nll: float
    converged: bool
    iterations: int
    message: str


def fit_start(data: FitData, seed: int, *, hybrid: bool) -> StartResult:
    rng = np.random.default_rng(seed)
    bnds = bounds(hybrid)
    x0 = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
    result = lbfgsb(lambda p: mean_nll(p, data, hybrid), x0, bnds)
    params = result.x
    return StartResult(
        seed=seed,
        x0=x0.tolist(),
        params=params.tolist(),
        mean_nll=mean_nll(params, data, hybrid),
        converged=result.success,
        iterations=result.iterations,
        message=result.message,
    )


def fit(data: FitData, *, hybrid: bool, seeds: tuple[int, ...] | None = None) -> list[StartResult]:
    chosen = seeds if seeds is not None else (config.B23_SEEDS if hybrid else config.B3_SEEDS)
    return [fit_start(data, seed, hybrid=hybrid) for seed in chosen]
