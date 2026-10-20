"""Common interface of a citizen model and the protocol's integer likelihood (pure NumPy).

**Streams and observations.** A model sees one *stream* per (game, player). In round t
player i observes (protocol 4; Koster et al. 2025 clone inputs)

- the four current offers e_t rotated so that its own comes first,
  ``roll(e_t, -i) = (e_i, e_{i+1}, e_{i+2}, e_{i+3})`` (indices mod 4),
- the four previous reciprocations c_{t-1} rotated the same way (zeros in round 0, as
  recorded in the CSV's `mechanism_observation.prev_reciprocations`),
- the current pool R_t and the round index t (0-based).

Earlier rounds enter only through the model's recurrent state, which a model updates
from each observation it is shown (its own previous action is ``prev_reciprocations[0]``
of the next observation). So teacher forcing and closed-loop play run the same `step`.

**Integer pmf.** `step` returns, per stream, a pmf over c in {0, ..., K_MAX - 1}
(K_MAX = 201, since floor(e) <= 200) that is zero above n = floor(e) and is the point
mass at 0 when e < 1 (a forced decision, excluded from every likelihood).

**Bin-to-integer map (protocol 4).** A model with probabilities p_b over fraction bins
assigns integer c <= n to bin b(c) and sets

    P(c) = p_{b(c)} / m_{b(c)} / sum_{b : m_b > 0} p_b,   m_b = |{c' <= n : b(c') = b}|,

i.e. each bin's mass is shared equally by its integers and the bins holding no integer
are renormalised away. Two binnings, both computed from q = (10 c) / e evaluated as one
correctly rounded division (so that an exact boundary such as c = 3, e = 10 gives q = 3
exactly):

- 10 *floor bins* (B1, M0, P0): b(c) = min(9, floor(q));
- 11 *levels* (B3, B23): l(c) = floor(q + 1/2) (round half up), level j means fraction j/10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from phasespace.citizens.game import EXCLUSION_THRESHOLD, MAX_POOL, N_PLAYERS
from phasespace.citizens.records import Games

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]
type Binning = Literal["floor10", "level11"]

K_MAX: Final = int(MAX_POOL) + 1
"""Integer actions 0..200: an offer never exceeds the pool cap of 200."""
N_BINS: Final = {"floor10": 10, "level11": 11}
LEVELS: Final = np.arange(11, dtype=np.float64) / 10.0
"""Fractions a_j = j / 10 of the 11 levels."""

_ROTATION: Final = (np.arange(N_PLAYERS)[:, None] + np.arange(N_PLAYERS)[None, :]) % N_PLAYERS
"""_ROTATION[i] = (i, i+1, i+2, i+3) mod 4: player i's view of a per-player vector."""


def rotate(x: npt.ArrayLike) -> FloatArray:
    """(..., 4) -> (..., 4, 4): out[..., i, :] = roll(x, -i), player i's own-first view."""
    a = np.asarray(x, dtype=np.float64)
    return a[..., _ROTATION]


def n_max(offers: npt.ArrayLike) -> IntArray:
    """Largest integer reciprocation floor(e) (0 for an exclusion e < 1)."""
    e = np.asarray(offers, dtype=np.float64)
    return np.where(e >= EXCLUSION_THRESHOLD, np.floor(e), 0.0).astype(np.int64)


def bin_index(c: npt.ArrayLike, e: npt.ArrayLike, binning: Binning) -> IntArray:
    """Bin (floor10) or level (level11) of integer c for offer e >= 1 (broadcasting)."""
    cc = np.asarray(c, dtype=np.float64)
    ee = np.asarray(e, dtype=np.float64)
    q = (10.0 * cc) / np.where(ee >= EXCLUSION_THRESHOLD, ee, 1.0)
    if binning == "floor10":
        return np.minimum(9, np.floor(q)).astype(np.int64)
    return np.clip(np.floor(q + 0.5), 0, 10).astype(np.int64)


@dataclass(frozen=True, slots=True)
class BinGrid:
    """Per-stream integer grid of a binning: bin of every c <= n and the bin counts m_b."""

    bins: IntArray  # (B, K_MAX) bin of c, -1 for c > n or a forced decision
    counts: IntArray  # (B, n_bins) m_b
    forced: BoolArray  # (B,) e < 1


def bin_grid(offers: npt.ArrayLike, binning: Binning) -> BinGrid:
    e = np.asarray(offers, dtype=np.float64).reshape(-1)
    nb = N_BINS[binning]
    forced = e < EXCLUSION_THRESHOLD
    c = np.arange(K_MAX, dtype=np.float64)[None, :]
    valid = (c <= n_max(e)[:, None]) & ~forced[:, None]
    bins = np.where(valid, bin_index(c, e[:, None], binning), -1)
    rows = np.broadcast_to(np.arange(e.shape[0])[:, None], bins.shape)
    flat = (rows * nb + bins)[valid]
    counts = np.bincount(flat, minlength=e.shape[0] * nb).reshape(e.shape[0], nb)
    return BinGrid(bins=bins, counts=counts.astype(np.int64), forced=forced)


def forced_pmf(n_streams: int) -> FloatArray:
    pmf = np.zeros((n_streams, K_MAX))
    pmf[:, 0] = 1.0
    return pmf


def bins_to_integer_pmf(p: npt.ArrayLike, offers: npt.ArrayLike, binning: Binning) -> FloatArray:
    """(B, n_bins) bin probabilities and (B,) offers -> (B, K_MAX) integer pmf (protocol 4)."""
    probs = np.asarray(p, dtype=np.float64)
    grid = bin_grid(offers, binning)
    n = grid.bins.shape[0]
    if probs.shape != (n, N_BINS[binning]):
        raise ValueError(f"bin probabilities need shape ({n}, {N_BINS[binning]})")
    available = grid.counts > 0
    z = np.where(available, probs, 0.0).sum(axis=1)
    safe_bins = np.maximum(grid.bins, 0)
    rows = np.arange(n, dtype=np.int64)[:, None]
    per_integer = probs[rows, safe_bins] / np.maximum(grid.counts[rows, safe_bins], 1)
    pmf = np.where(grid.bins >= 0, per_integer / np.where(z > 0.0, z, 1.0)[:, None], 0.0)
    pmf[grid.forced] = forced_pmf(int(grid.forced.sum()))
    return pmf


def integer_log_prob(
    log_p: npt.ArrayLike, c: npt.ArrayLike, offers: npt.ArrayLike, binning: Binning
) -> FloatArray:
    """log P(c) of the protocol 4 map at observed c, from log bin probabilities (B, n_bins).

    Only for non-forced decisions (e >= 1, integer 0 <= c <= floor(e)).
    """
    lp = np.asarray(log_p, dtype=np.float64)
    e = np.asarray(offers, dtype=np.float64)
    cc = np.asarray(c, dtype=np.float64)
    if np.any(e < EXCLUSION_THRESHOLD) or np.any(cc > np.floor(e)) or np.any(cc < 0):
        raise ValueError("integer_log_prob needs non-forced decisions with 0 <= c <= floor(e)")
    grid = bin_grid(e, binning)
    rows = np.arange(e.shape[0], dtype=np.int64)
    b = bin_index(cc, e, binning)
    log_z = log_sum_exp_masked(lp, grid.counts > 0)
    return lp[rows, b] - np.log(grid.counts[rows, b]) - log_z


def log_sum_exp_masked(x: FloatArray, mask: BoolArray) -> FloatArray:
    """log sum_{mask} exp(x) over the last axis (requires at least one True per row)."""
    masked = np.where(mask, x, -np.inf)
    top = masked.max(axis=-1)
    return top + np.log(np.where(mask, np.exp(masked - top[..., None]), 0.0).sum(axis=-1))


def sample_pmf(pmf: FloatArray, rng: np.random.Generator) -> IntArray:
    """One draw per row by inversion of the cumulative pmf (uses B uniforms of `rng`)."""
    cdf = np.cumsum(pmf, axis=1)
    u = rng.random(int(pmf.shape[0])) * cdf[:, -1]
    draw = (cdf < u[:, None]).sum(axis=1)
    support_top = K_MAX - 1 - np.argmax(pmf[:, ::-1] > 0.0, axis=1)
    return np.minimum(draw, support_top).astype(np.int64)


@dataclass(frozen=True, slots=True)
class Observation:
    """What every stream sees in one round (own first)."""

    offers: FloatArray  # (B, 4)
    prev_reciprocations: FloatArray  # (B, 4), zeros in round 0
    pool: FloatArray  # (B,)
    round_index: int

    @property
    def own_offer(self) -> FloatArray:
        return self.offers[:, 0]

    @property
    def n_streams(self) -> int:
        return int(self.offers.shape[0])


class CitizenModel[S](ABC):
    """A behavioural model of one player, vectorised over streams.

    `initial_state(n, rng)`: with `rng` (closed-loop play) the model may draw per-stream
    latent choices (B1: one ensemble member per player); with ``rng=None`` it returns the
    state of its predictive (likelihood) form. `step` consumes one observation per
    stream and returns the integer pmf of this round's reciprocation and the new state.
    `sample` draws the closed-loop action (default: from the pmf; B0 overrides it).
    """

    name: str = "model"

    @abstractmethod
    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> S: ...

    @abstractmethod
    def step(self, state: S, obs: Observation) -> tuple[FloatArray, S]: ...

    def sample(self, pmf: FloatArray, obs: Observation, rng: np.random.Generator) -> IntArray:
        del obs
        return sample_pmf(pmf, rng)

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-serialisable parameters (or a reference to them) for manifests."""


# --- recorded games as streams -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Streams:
    """Recorded games as (game, player) streams; stream b = 4 g + i.

    Padding (rounds past a game's end) holds offers 0 (forced), pool 200 and actions 0;
    `recorded` marks the real rounds. `excluded` marks recorded decisions that cannot be
    scored as recorded (protocol-validation A29: non-integer c, c > floor(e), c < 0, or
    c > 0 with e < 1); they stay in every model's observed history as recorded but are
    never scored.
    """

    offers: FloatArray  # (B, T, 4) own first
    prev_reciprocations: FloatArray  # (B, T, 4) own first, zeros at t = 0
    pool: FloatArray  # (B, T)
    actions: FloatArray  # (B, T) own reciprocation c_{i,t}
    recorded: BoolArray  # (B, T)
    game: IntArray  # (B,) index of the game in the source Games
    player: IntArray  # (B,)
    excluded: BoolArray  # (B, T) anomalous recorded decisions (A29), never scored

    @property
    def n_streams(self) -> int:
        return int(self.offers.shape[0])

    @property
    def n_rounds_max(self) -> int:
        return int(self.offers.shape[1])

    @property
    def own_offer(self) -> FloatArray:
        return self.offers[..., 0]

    @property
    def scored(self) -> BoolArray:
        """(B, T) non-forced recorded decisions (e >= 1, not excluded): the likelihood's
        support."""
        return self.recorded & (self.own_offer >= EXCLUSION_THRESHOLD) & ~self.excluded

    def observation(self, t: int) -> Observation:
        return Observation(
            offers=self.offers[:, t],
            prev_reciprocations=self.prev_reciprocations[:, t],
            pool=self.pool[:, t],
            round_index=t,
        )


def streams_from_arrays(
    offers: npt.ArrayLike,
    actions: npt.ArrayLike,
    pool: npt.ArrayLike,
    n_rounds: npt.ArrayLike,
) -> Streams:
    """Streams of games given (G, T, 4) offers and actions, (G, T) pools and (G,) lengths."""
    e = np.asarray(offers, dtype=np.float64)
    c = np.asarray(actions, dtype=np.float64)
    r = np.asarray(pool, dtype=np.float64)
    n = np.asarray(n_rounds, dtype=np.int64)
    g, t_max = int(e.shape[0]), int(n.max()) if n.size else 0
    recorded = np.arange(t_max)[None, :] < n[:, None]
    e = np.where(recorded[..., None], e[:, :t_max], 0.0)
    c = np.where(recorded[..., None], c[:, :t_max], 0.0)
    r = np.where(recorded, r[:, :t_max], MAX_POOL)
    prev = np.zeros_like(c)
    prev[:, 1:] = c[:, :-1]
    # (G, T, 4) -> (G, T, 4 players, 4 own-first) -> (G, 4, T, 4) -> (4G, T, 4)
    e_rot = rotate(e).transpose(0, 2, 1, 3).reshape(g * N_PLAYERS, t_max, N_PLAYERS)
    prev_rot = rotate(prev).transpose(0, 2, 1, 3).reshape(g * N_PLAYERS, t_max, N_PLAYERS)
    return Streams(
        offers=e_rot,
        prev_reciprocations=prev_rot,
        pool=np.repeat(r, N_PLAYERS, axis=0),
        actions=c.transpose(0, 2, 1).reshape(g * N_PLAYERS, t_max),
        recorded=np.repeat(recorded, N_PLAYERS, axis=0),
        game=np.repeat(np.arange(g, dtype=np.int64), N_PLAYERS),
        player=np.tile(np.arange(N_PLAYERS, dtype=np.int64), g),
        excluded=np.zeros((g * N_PLAYERS, t_max), dtype=np.bool_),
    )


def anomalous_decisions(streams: Streams) -> BoolArray:
    """(B, T) recorded decisions that are not an integer 0 <= c <= floor(e) (A29).

    Covers non-integer c, c > floor(e) for e >= 1, c < 0, and c > 0 for e < 1 (floor(e)
    is taken as 0 for an exclusion, `n_max`).
    """
    a, e = streams.actions, streams.own_offer
    bad = (a != np.floor(a)) | (a < 0.0) | (a > n_max(e))
    return streams.recorded & bad


def streams_from_games(games: Games) -> Streams:
    """Streams of recorded games; anomalous decisions (A29) are excluded from scoring.

    Training human games have none (C1 `action_validity`); for held-out games the count
    is reported by every evaluation (`Streams.excluded`).
    """
    s = streams_from_arrays(games.offers, games.actions, games.pool, games.n_rounds)
    return replace(s, excluded=anomalous_decisions(s))


def teacher_forced_log_prob(model: CitizenModel[Any], streams: Streams) -> FloatArray:
    """(B, T) log P_model(c_observed) under the recorded histories; NaN where not scored."""
    state = model.initial_state(streams.n_streams, None)
    out = np.full((streams.n_streams, streams.n_rounds_max), np.nan)
    rows = np.arange(streams.n_streams)
    for t in range(streams.n_rounds_max):
        pmf, state = model.step(state, streams.observation(t))
        # Unscored entries (forced, padding, excluded) may hold any recorded value.
        c = np.clip(streams.actions[:, t], 0, K_MAX - 1).astype(np.int64)
        with np.errstate(divide="ignore"):
            lp = np.log(pmf[rows, c])
        out[:, t] = np.where(streams.scored[:, t], lp, np.nan)
    return out
