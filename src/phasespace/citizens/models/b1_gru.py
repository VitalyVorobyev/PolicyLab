"""B1: GRU behavioural clone, NumPy inference (training lives in `b1_train`, JAX).

Architecture (Koster et al. 2025, Table S1, BC1 scale; protocol 6.3): 9 inputs, all
divided by 200 -- the four current offers (own first), the four previous reciprocations
(own first; zeros in round 0) and the pool -- then

    x -> tanh(L16) -> tanh(L32) -> GRU(64) -> tanh(L32) -> tanh(L16) -> L10 (logits),

one sequence per (game, player), hidden state zero at the start of a game. The 10
logits are the floor bins of protocol 4 (bin k = fraction [k/10, (k+1)/10)).

The GRU cell is Equinox's `GRUCell` (equinox 0.13):

    r = sigmoid(W_ir x + b_r + W_hr h),   z = sigmoid(W_iz x + b_z + W_hz h),
    n = tanh(W_in x + b_n + r (W_hn h + b_hn)),   h' = n + z (h - n),

with W_ih = [W_ir; W_iz; W_in] (3H x D), W_hh likewise, `bias` = [b_r; b_z; b_n] and
`bias_n` = b_hn. Inference here is float64 on the (float32-trained) weights.

**Ensemble** of N = 4 members (seeds 101-104). Likelihood: the mean of the members'
integer pmfs. Closed loop: every simulated player draws one member uniformly with
replacement at game start.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from phasespace.citizens import config
from phasespace.citizens.models.base import (
    CitizenModel,
    FloatArray,
    IntArray,
    Observation,
    bins_to_integer_pmf,
)

WEIGHT_NAMES: Final = (
    "enc1.weight",
    "enc1.bias",
    "enc2.weight",
    "enc2.bias",
    "cell.weight_ih",
    "cell.weight_hh",
    "cell.bias",
    "cell.bias_n",
    "proj1.weight",
    "proj1.bias",
    "proj2.weight",
    "proj2.bias",
    "head.weight",
    "head.bias",
)
_ZIP_EPOCH: Final = (1980, 1, 1, 0, 0, 0)


def weight_shapes() -> dict[str, tuple[int, ...]]:
    d, (e1, e2), h = config.B1_INPUTS, config.B1_ENCODER, config.B1_GRU
    (p1, p2), k = config.B1_PROJECTION, config.B1_BINS
    return {
        "enc1.weight": (e1, d),
        "enc1.bias": (e1,),
        "enc2.weight": (e2, e1),
        "enc2.bias": (e2,),
        "cell.weight_ih": (3 * h, e2),
        "cell.weight_hh": (3 * h, h),
        "cell.bias": (3 * h,),
        "cell.bias_n": (h,),
        "proj1.weight": (p1, h),
        "proj1.bias": (p1,),
        "proj2.weight": (p2, p1),
        "proj2.bias": (p2,),
        "head.weight": (k, p2),
        "head.bias": (k,),
    }


@dataclass(frozen=True, slots=True)
class GRUWeights:
    arrays: Mapping[str, FloatArray]

    def __post_init__(self) -> None:
        shapes = weight_shapes()
        if set(self.arrays) != set(shapes):
            raise ValueError("GRU weights need exactly the names of WEIGHT_NAMES")
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"{name} has shape {self.arrays[name].shape}, expected {shape}")

    def __getitem__(self, name: str) -> FloatArray:
        return self.arrays[name]

    def to_npz_bytes(self) -> bytes:
        """Deterministic .npz bytes (fixed entry order and timestamps; float32 as trained)."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in WEIGHT_NAMES:
                entry = io.BytesIO()
                np.lib.format.write_array(
                    entry, self.arrays[name].astype(np.float32), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, entry.getvalue())
        return buffer.getvalue()

    @classmethod
    def from_npz_bytes(cls, payload: bytes) -> GRUWeights:
        with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
            arrays = {name: np.asarray(npz[name], dtype=np.float64) for name in WEIGHT_NAMES}
        return cls(arrays=arrays)


def inputs(obs: Observation) -> FloatArray:
    """(B, 9) network inputs of an observation, all divided by 200."""
    scale = config.B1_INPUT_SCALE
    return np.concatenate(
        [obs.offers / scale, obs.prev_reciprocations / scale, obs.pool[:, None] / scale], axis=1
    )


def _sigmoid(x: FloatArray) -> FloatArray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _linear(w: GRUWeights, name: str, x: FloatArray) -> FloatArray:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def forward_step(w: GRUWeights, hidden: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """One time step for a batch: (h (B, 64), x (B, 9)) -> (h', logits (B, 10))."""
    z = np.tanh(_linear(w, "enc2", np.tanh(_linear(w, "enc1", x))))
    h = config.B1_GRU
    ig = z @ w["cell.weight_ih"].T + w["cell.bias"]
    hg = hidden @ w["cell.weight_hh"].T
    reset = _sigmoid(ig[:, :h] + hg[:, :h])
    update = _sigmoid(ig[:, h : 2 * h] + hg[:, h : 2 * h])
    new = np.tanh(ig[:, 2 * h :] + reset * (hg[:, 2 * h :] + w["cell.bias_n"]))
    h_next = new + update * (hidden - new)
    y = np.tanh(_linear(w, "proj2", np.tanh(_linear(w, "proj1", h_next))))
    return h_next, _linear(w, "head", y)


def softmax(logits: FloatArray) -> FloatArray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True, slots=True)
class B1State:
    hidden: FloatArray  # (M, B, 64) likelihood form, or (B, 64) closed loop
    member: IntArray | None  # (B,) closed loop; None for the ensemble-mean form


class B1GRUEnsemble(CitizenModel[B1State]):
    name = "B1"

    def __init__(self, members: Sequence[GRUWeights], seeds: Sequence[int] = ()) -> None:
        if not members:
            raise ValueError("B1 needs at least one member")
        self.members = tuple(members)
        self.seeds = tuple(seeds)

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> B1State:
        h = config.B1_GRU
        if rng is None:
            return B1State(hidden=np.zeros((len(self.members), n_streams, h)), member=None)
        member = rng.integers(0, len(self.members), size=n_streams).astype(np.int64)
        return B1State(hidden=np.zeros((n_streams, h)), member=member)

    def step(self, state: B1State, obs: Observation) -> tuple[FloatArray, B1State]:
        x = inputs(obs)
        e = obs.own_offer
        if state.member is None:
            hidden = np.empty_like(state.hidden)
            pmf = np.zeros((obs.n_streams, 0))
            for m, w in enumerate(self.members):
                hidden[m], logits = forward_step(w, state.hidden[m], x)
                member_pmf = bins_to_integer_pmf(softmax(logits), e, "floor10")
                pmf = member_pmf if m == 0 else pmf + member_pmf
            return pmf / len(self.members), B1State(hidden=hidden, member=None)
        hidden = np.empty_like(state.hidden)
        probs = np.empty((obs.n_streams, config.B1_BINS))
        for m, w in enumerate(self.members):
            rows = state.member == m
            if rows.any():
                hidden[rows], logits = forward_step(w, state.hidden[rows], x[rows])
                probs[rows] = softmax(logits)
        return bins_to_integer_pmf(probs, e, "floor10"), B1State(hidden, state.member)

    def describe(self) -> dict[str, Any]:
        return {"members": len(self.members), "seeds": list(self.seeds)}


def random_weights(rng: np.random.Generator, scale: float = 0.3) -> GRUWeights:
    """Random weights of the right shapes (tests only)."""
    arrays: dict[str, npt.NDArray[np.float64]] = {
        name: rng.normal(0.0, scale, shape) for name, shape in weight_shapes().items()
    }
    return GRUWeights(arrays=arrays)
