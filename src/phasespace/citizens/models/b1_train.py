"""B1 training in JAX / Equinox / Optax (protocol 6.3; inference is `b1_gru`, NumPy).

Loss: cross-entropy of the observed floor bin on the non-forced decisions (forced steps
are fed as inputs but not scored), back-propagated through the whole game; Adam with a
constant learning rate 5e-4, batches of 64 (game, player) sequences, no regularisation.
Every 250 updates the validation score is computed -- the per-decision integer-pmf NLL of
protocol 4 on the validation split (config.B1_EARLY_STOPPING_SCORE) -- and training stops
after 8 evaluations without improvement or at 20,000 updates; the best-validation
checkpoint is kept. Batches: each epoch is a permutation of the fit sequences drawn with
numpy default_rng(seed), cut into consecutive batches of 64 (a last partial batch is
dropped). Initialisation: Equinox defaults with jax.random.PRNGKey(seed).

JAX and Equinox are only partly typed; the narrow `pyright: ignore` comments below mark
exactly the calls whose types they leave unknown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]

from phasespace.citizens import config
from phasespace.citizens.models.b1_gru import GRUWeights, inputs
from phasespace.citizens.models.base import BoolArray, FloatArray, IntArray, Streams, bin_grid
from phasespace.citizens.models.base import bin_index as _bin_index


def _device(a: np.ndarray[Any, Any]) -> jax.Array:
    """A NumPy array as a JAX array (same dtype)."""
    return jnp.asarray(a)  # pyright: ignore[reportUnknownMemberType]


class GRUClone(eqx.Module):
    enc1: eqx.nn.Linear
    enc2: eqx.nn.Linear
    cell: eqx.nn.GRUCell
    proj1: eqx.nn.Linear
    proj2: eqx.nn.Linear
    head: eqx.nn.Linear

    def __init__(self, key: jax.Array) -> None:
        k = jax.random.split(key, 6)
        (e1, e2), h = config.B1_ENCODER, config.B1_GRU
        p1, p2 = config.B1_PROJECTION
        self.enc1 = eqx.nn.Linear(config.B1_INPUTS, e1, key=k[0])
        self.enc2 = eqx.nn.Linear(e1, e2, key=k[1])
        self.cell = eqx.nn.GRUCell(e2, h, key=k[2])
        self.proj1 = eqx.nn.Linear(h, p1, key=k[3])
        self.proj2 = eqx.nn.Linear(p1, p2, key=k[4])
        self.head = eqx.nn.Linear(p2, config.B1_BINS, key=k[5])

    def __call__(self, xs: jax.Array) -> jax.Array:
        """(T, 9) inputs of one sequence -> (T, 10) logits."""

        def body(hidden: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
            z = jnp.tanh(self.enc2(jnp.tanh(self.enc1(x))))
            hidden = self.cell(z, hidden)
            y = jnp.tanh(self.proj2(jnp.tanh(self.proj1(hidden))))
            return hidden, self.head(y)

        h0: jax.Array = jnp.zeros(config.B1_GRU)  # pyright: ignore[reportUnknownMemberType]
        _, logits = jax.lax.scan(body, h0, xs)
        return logits


@dataclass(frozen=True, slots=True)
class SequenceData:
    """Padded sequences as arrays for training and scoring."""

    x: FloatArray  # (B, T, 9)
    target: IntArray  # (B, T) observed floor bin (0 where not scored)
    scored: BoolArray  # (B, T)
    log_count: FloatArray  # (B, T) log m_{b(c)} (0 where not scored)
    available: BoolArray  # (B, T, 10) bins holding an integer (all True where not scored)

    @classmethod
    def from_streams(cls, streams: Streams) -> SequenceData:
        b, t = streams.n_streams, streams.n_rounds_max
        x = np.stack([inputs(streams.observation(k)) for k in range(t)], axis=1)
        scored = streams.scored
        e = streams.own_offer
        target = np.zeros((b, t), dtype=np.int64)
        target[scored] = _bin_index(streams.actions[scored], e[scored], "floor10")
        counts = np.ones((b, t, config.B1_BINS), dtype=np.int64)
        counts[scored] = bin_grid(e[scored], "floor10").counts
        log_count = np.zeros((b, t))
        rows, cols = np.nonzero(scored)
        log_count[rows, cols] = np.log(counts[rows, cols, target[rows, cols]])
        return cls(x=x, target=target, scored=scored, log_count=log_count, available=counts > 0)

    def device(self, index: IntArray | None = None) -> tuple[jax.Array, ...]:
        sel: Any = slice(None) if index is None else index
        return (
            _device(self.x[sel].astype(np.float32)),
            _device(self.target[sel].astype(np.int32)),
            _device(self.scored[sel]),
            _device(self.log_count[sel].astype(np.float32)),
            _device(self.available[sel]),
        )


def _log_softmax_batch(model: GRUClone, x: jax.Array) -> jax.Array:
    logits = jax.vmap(model)(x)
    return jax.nn.log_softmax(logits, axis=-1)


def bin_cross_entropy(model: GRUClone, batch: tuple[jax.Array, ...]) -> jax.Array:
    x, target, scored, _, _ = batch
    lsm = _log_softmax_batch(model, x)
    picked = jnp.take_along_axis(lsm, target[..., None], axis=-1)[..., 0]
    return -jnp.sum(jnp.where(scored, picked, 0.0)) / jnp.maximum(jnp.sum(scored), 1)


def integer_nll_sum(model: GRUClone, batch: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
    """(sum of -log P(c), number of scored decisions) under the protocol 4 integer map."""
    x, target, scored, log_count, available = batch
    lsm = _log_softmax_batch(model, x)
    picked = jnp.take_along_axis(lsm, target[..., None], axis=-1)[..., 0]
    log_z = jax.nn.logsumexp(jnp.where(available, lsm, -jnp.inf), axis=-1)
    log_p = picked - log_count - log_z
    return -jnp.sum(jnp.where(scored, log_p, 0.0)), jnp.sum(scored)


def to_weights(model: GRUClone) -> GRUWeights:
    def arr(a: Any) -> FloatArray:
        return np.asarray(a, dtype=np.float64)

    m = model
    return GRUWeights(
        arrays={
            "enc1.weight": arr(m.enc1.weight),
            "enc1.bias": arr(m.enc1.bias),
            "enc2.weight": arr(m.enc2.weight),
            "enc2.bias": arr(m.enc2.bias),
            "cell.weight_ih": arr(m.cell.weight_ih),
            "cell.weight_hh": arr(m.cell.weight_hh),
            "cell.bias": arr(m.cell.bias),
            "cell.bias_n": arr(m.cell.bias_n),
            "proj1.weight": arr(m.proj1.weight),
            "proj1.bias": arr(m.proj1.bias),
            "proj2.weight": arr(m.proj2.weight),
            "proj2.bias": arr(m.proj2.bias),
            "head.weight": arr(m.head.weight),
            "head.bias": arr(m.head.bias),
        }
    )


@dataclass(frozen=True, slots=True)
class TrainResult:
    seed: int
    weights: GRUWeights
    best_update: int
    best_val_nll: float
    updates: int
    stopped_early: bool
    history: list[dict[str, float]]  # every evaluation: update, val_nll, train_ce
    seconds: float


def train_member(
    fit: SequenceData,
    val: SequenceData,
    seed: int,
    *,
    max_updates: int = config.B1_MAX_UPDATES,
    eval_every: int = config.B1_EVAL_EVERY,
    patience: int = config.B1_PATIENCE,
    batch: int = config.B1_BATCH,
    learning_rate: float = config.B1_LEARNING_RATE,
) -> TrainResult:
    """Train one ensemble member with early stopping (see the module docstring)."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    model = GRUClone(jax.random.PRNGKey(seed))
    optimiser = optax.adam(learning_rate)  # pyright: ignore[reportUnknownMemberType]
    params: Any = eqx.filter(model, eqx.is_array)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    opt_state = optimiser.init(params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    val_batch = val.device()

    fit_all = fit.device()

    @eqx.filter_jit
    def train_step(
        model: GRUClone, opt_state: Any, index: jax.Array
    ) -> tuple[GRUClone, Any, jax.Array]:
        data = tuple(a[index] for a in fit_all)
        value_and_grad: Any = eqx.filter_value_and_grad(bin_cross_entropy)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        loss: jax.Array
        loss, grads = value_and_grad(model, data)
        updates, opt_state = optimiser.update(grads, opt_state, model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]
        new_model = cast(GRUClone, eqx.apply_updates(model, updates))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        return new_model, opt_state, loss

    @eqx.filter_jit
    def val_nll(model: GRUClone) -> jax.Array:
        total, count = integer_nll_sum(model, val_batch)
        return total / count

    n = int(fit.x.shape[0])
    if n < batch:
        raise ValueError(f"need at least {batch} fit sequences, got {n}")
    best_model, best_val, best_update = model, float("inf"), 0
    history: list[dict[str, float]] = []
    updates, bad, stopped = 0, 0, False
    losses: list[float] = []
    while updates < max_updates and not stopped:
        perm = rng.permutation(n)
        for k in range(n // batch):
            index = _device(perm[k * batch : (k + 1) * batch].astype(np.int32))
            model, opt_state, loss = train_step(model, opt_state, index)
            losses.append(float(loss))
            updates += 1
            if updates % eval_every == 0:
                v = float(val_nll(model))
                history.append(
                    {"update": updates, "val_nll": v, "train_ce": float(np.mean(losses))}
                )
                losses = []
                if v < best_val:
                    best_model, best_val, best_update, bad = model, v, updates, 0
                else:
                    bad += 1
                    if bad >= patience:
                        stopped = True
                        break
            if updates >= max_updates:
                break
    if updates % eval_every != 0:  # a final evaluation when max_updates is off the grid
        v = float(val_nll(model))
        history.append({"update": updates, "val_nll": v, "train_ce": float(np.mean(losses))})
        if v < best_val:
            best_model, best_val, best_update = model, v, updates
    return TrainResult(
        seed=seed,
        weights=to_weights(best_model),
        best_update=best_update,
        best_val_nll=best_val,
        updates=updates,
        stopped_early=stopped,
        history=history,
        seconds=time.perf_counter() - start,
    )
