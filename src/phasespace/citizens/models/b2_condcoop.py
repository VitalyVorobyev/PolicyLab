"""B2: parametric conditional cooperation (12 parameters, maximum likelihood; protocol 6.3).

**Features** of decision (i, t), from player i's observation and its previous one:

- x1 = min(4, 4 e_{i,t} / R_t): own offer relative to the equal share R/4;
- x2 = c_{i,t-1} / e_{i,t-1}: own previous fraction (0 if t = 0 or e_{i,t-1} < 1);
- x3 = sum_{j != i} c_{j,t-1} / sum_{j != i} e_{j,t-1}: others' previous fraction
  (0 if t = 0 or the denominator < 1);
- x4 = R_t / 200; x5 = 1[t = 0]; x6 = 1[t > 0 and e_{i,t-1} < 1].

**Likelihood** for n = floor(e) >= 1, with sigma the logistic function:

- zero hurdle pi0 = sigma(g0 + g1 x2 + g2 x3);
- full-return spike piF = sigma(d0);
- interior mean mu = sigma(b0 + sum_k b_k x_k), precision phi = exp(eta);
- P(0) = pi0; if n = 1, P(1) = 1 - pi0; if n >= 2, P(n) = (1 - pi0) piF and, for
  1 <= c <= n - 1, P(c) = (1 - pi0)(1 - piF) BetaBinomial(c - 1; n - 2, mu phi, (1 - mu) phi),

  BetaBinomial(k; N, a, b) = C(N, k) B(k + a, N - k + b) / B(a, b).

b3 is the conditional-cooperation slope on others' previous fraction (Fischbacher &
Gaechter 2010: under-matching means b3 > 0 with an imperfect match). The parameter vector
is (g0, g1, g2, d0, b0, b1, ..., b6, eta).

**Fit:** L-BFGS-B on the fit-split mean NLL from 10 starts N(0, 0.5^2) (seeds 201-210),
keeping the best; no bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy import special

from phasespace.citizens import config
from phasespace.citizens.game import EXCLUSION_THRESHOLD, MAX_POOL
from phasespace.citizens.models.base import (
    K_MAX,
    CitizenModel,
    FloatArray,
    Observation,
    Streams,
    forced_pmf,
    n_max,
)
from phasespace.citizens.models.optim import lbfgsb

PARAM_NAMES: Final = (
    "g0",
    "g1",
    "g2",
    "d0",
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "eta",
)


def features(
    own_offer: FloatArray,
    pool: FloatArray,
    t: int,
    prev_offers: FloatArray | None,
    prev_reciprocations: FloatArray,
) -> FloatArray:
    """(B, 6) features x1..x6; `prev_offers` (B, 4) own first, None in round 0."""
    b = own_offer.shape[0]
    x = np.zeros((b, 6))
    x[:, 0] = np.minimum(config.B2_MAX_X1, 4.0 * own_offer / pool)
    x[:, 3] = pool / MAX_POOL
    if t == 0 or prev_offers is None:
        x[:, 4] = 1.0
        return x
    own_prev = prev_offers[:, 0]
    own_ok = own_prev >= EXCLUSION_THRESHOLD
    x[:, 1] = np.where(own_ok, prev_reciprocations[:, 0] / np.where(own_ok, own_prev, 1.0), 0.0)
    others_e = prev_offers[:, 1:].sum(axis=1)
    others_ok = others_e >= EXCLUSION_THRESHOLD
    others_c = prev_reciprocations[:, 1:].sum(axis=1)
    x[:, 2] = np.where(others_ok, others_c / np.where(others_ok, others_e, 1.0), 0.0)
    x[:, 5] = (~own_ok).astype(np.float64)
    return x


def stream_features(streams: Streams) -> FloatArray:
    """(B, T, 6) features of every stream and round (padding included)."""
    out = np.zeros((streams.n_streams, streams.n_rounds_max, 6))
    for t in range(streams.n_rounds_max):
        prev = streams.offers[:, t - 1] if t > 0 else None
        out[:, t] = features(
            streams.own_offer[:, t],
            streams.pool[:, t],
            t,
            prev,
            streams.prev_reciprocations[:, t],
        )
    return out


def _log_sigmoid(z: FloatArray) -> FloatArray:
    return -np.logaddexp(0.0, -z)


@dataclass(frozen=True, slots=True)
class Components:
    log_pi0: FloatArray
    log_1m_pi0: FloatArray
    log_pif: FloatArray
    log_1m_pif: FloatArray
    a: FloatArray  # mu phi
    b: FloatArray  # (1 - mu) phi


def components(theta: npt.ArrayLike, x: FloatArray) -> Components:
    th = np.asarray(theta, dtype=np.float64)
    z0 = th[0] + th[1] * x[:, 1] + th[2] * x[:, 2]
    zf = np.full(x.shape[0], th[3])
    zm = th[4] + x @ th[5:11]
    phi = np.exp(th[11])
    mu = 0.5 * (1.0 + np.tanh(0.5 * zm))  # logistic sigmoid
    return Components(
        log_pi0=_log_sigmoid(z0),
        log_1m_pi0=_log_sigmoid(-z0),
        log_pif=_log_sigmoid(zf),
        log_1m_pif=_log_sigmoid(-zf),
        a=mu * phi,
        b=(1.0 - mu) * phi,
    )


def log_beta_binomial(k: FloatArray, n: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    return (
        special.gammaln(n + 1.0)
        - special.gammaln(k + 1.0)
        - special.gammaln(n - k + 1.0)
        + special.betaln(k + a, n - k + b)
        - special.betaln(a, b)
    )


def log_prob(theta: npt.ArrayLike, x: FloatArray, c: FloatArray, n: FloatArray) -> FloatArray:
    """log P(c) for non-forced decisions (n >= 1, 0 <= c <= n integers)."""
    comp = components(theta, x)
    interior = (c >= 1) & (c <= n - 1)
    k = np.where(interior, c - 1.0, 0.0)
    nn = np.where(interior, n - 2.0, 0.0)
    bb = log_beta_binomial(k, nn, comp.a, comp.b)
    return np.where(
        c == 0,
        comp.log_pi0,
        np.where(
            n == 1,
            comp.log_1m_pi0,
            np.where(
                c == n,
                comp.log_1m_pi0 + comp.log_pif,
                comp.log_1m_pi0 + comp.log_1m_pif + bb,
            ),
        ),
    )


def pmf(theta: npt.ArrayLike, x: FloatArray, own_offer: FloatArray) -> FloatArray:
    """(B, K_MAX) integer pmf (point mass at 0 for e < 1)."""
    n = n_max(own_offer).astype(np.float64)
    rows = x.shape[0]
    width = int(n.max()) + 1 if rows else 1  # columns above every n are zero
    c = np.arange(width, dtype=np.float64)[None, :]
    valid = c <= n[:, None]
    xx = np.repeat(x, width, axis=0)
    lp = log_prob(theta, xx, np.tile(c[0], rows), np.repeat(np.maximum(n, 1.0), width)).reshape(
        rows, width
    )
    out = np.zeros((rows, K_MAX))
    out[:, :width] = np.where(valid, np.exp(lp), 0.0)
    forced = own_offer < EXCLUSION_THRESHOLD
    out[forced] = forced_pmf(int(forced.sum()))
    return out


@dataclass(frozen=True, slots=True)
class B2State:
    prev_offers: FloatArray | None  # (B, 4) own first; None before round 0


class B2ConditionalCooperation(CitizenModel[B2State]):
    name = "B2"

    def __init__(self, theta: npt.ArrayLike) -> None:
        th = np.asarray(theta, dtype=np.float64)
        if th.shape != (config.B2_PARAMS,):
            raise ValueError(f"B2 needs {config.B2_PARAMS} parameters")
        self.theta = th

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> B2State:
        return B2State(prev_offers=None)

    def step(self, state: B2State, obs: Observation) -> tuple[FloatArray, B2State]:
        x = features(
            obs.own_offer, obs.pool, obs.round_index, state.prev_offers, obs.prev_reciprocations
        )
        return pmf(self.theta, x, obs.own_offer), B2State(prev_offers=obs.offers.copy())

    def describe(self) -> dict[str, Any]:
        return {"theta": dict(zip(PARAM_NAMES, self.theta.tolist(), strict=True))}


@dataclass(frozen=True, slots=True)
class FitData:
    """Scored decisions of the fit split, flattened: features, actions, n."""

    x: FloatArray  # (D, 6)
    c: FloatArray  # (D,)
    n: FloatArray  # (D,)

    @classmethod
    def from_streams(cls, streams: Streams) -> FitData:
        scored = streams.scored
        return cls(
            x=stream_features(streams)[scored],
            c=streams.actions[scored],
            n=np.floor(streams.own_offer[scored]),
        )


def mean_nll(theta: npt.ArrayLike, data: FitData) -> float:
    return float(-log_prob(theta, data.x, data.c, data.n).mean())


@dataclass(frozen=True, slots=True)
class StartResult:
    seed: int
    x0: list[float]
    theta: list[float]
    mean_nll: float
    converged: bool
    iterations: int
    message: str


def fit_start(data: FitData, seed: int) -> StartResult:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, config.B2_START_SD, config.B2_PARAMS)
    result = lbfgsb(lambda th: mean_nll(th, data), x0)
    theta = result.x
    return StartResult(
        seed=seed,
        x0=x0.tolist(),
        theta=theta.tolist(),
        mean_nll=mean_nll(theta, data),
        converged=result.success,
        iterations=result.iterations,
        message=result.message,
    )


def fit(data: FitData, seeds: tuple[int, ...] = config.B2_SEEDS) -> list[StartResult]:
    """Every start's result; the caller keeps the best (lowest fit-split NLL)."""
    return [fit_start(data, seed) for seed in seeds]
