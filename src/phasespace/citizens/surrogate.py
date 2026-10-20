"""Recurrent surrogates of the unreleased RL mechanisms M1 and M2 (protocol-validation §6.4).

The M1/M2 policies of Koster et al. (2025) are not released, so closed-loop simulation
under them needs a surrogate: a recurrent map from the mechanism's own observation
sequence to its 5-way share (4 players + the pool). It is fitted only on the
mechanism rows of the simulated clone cohorts, never on human rows:

- M1 from "RL Agent (M1)" games of cohort BC 1, M2 from "RL Agent (M2)" games of BC 2
  (in the CSV, BC 1 played Equal/Mixed/Proportional/Interpolating/M1 and BC 2
  Interpolating/M2; 512 games each under M1 and M2).

**Architecture (permutation-equivariant, recurrent).** Per player i and round t the
inputs are (e_{i,t-1}, c_{i,t-1}, R_t, mean_j e_{j,t-1}, mean_j c_{j,t-1}) / 200 (zeros
for the previous round in round 1, as recorded), taken from the recorded mechanism
observation (`prev_offers`, `prev_reciprocations`, `pool`). A shared FC layer (tanh)
feeds a shared GRU cell per player; a shared FC layer (tanh) and a linear unit give one
logit per player; the mean of the four hidden states goes through its own FC (tanh)
and linear unit to the pool logit. Offers are e = R softmax(5 logits)[:4]. Every hidden
width is `width` (32; the pre-declared capacity step is 64).

**Fit.** Cross-entropy between the recorded share vector (e_i / R, 1 - sum e / R) and
the predicted shares over recorded rounds with R >= 1; Adam 1e-3, batches of 64 games,
offer MAE on the BC validation split every 250 updates, patience 8 evaluations, at most
20,000 updates, the best-validation checkpoint kept; three seeds per mechanism, the
seed with the best validation MAE kept. The BC split is 410/102 games per mechanism,
drawn by `numpy.random.default_rng(20261021)` (a fresh generator per mechanism)
permuting the games sorted by `launch_id`.

**Fidelity** (§6.4 acceptance metrics, teacher-forced from the recorded observation
sequence, player-rounds of rounds with R >= 1): median and 90th percentile of
|e_hat - e| in coins, and the agreement of 1[e_hat < 1] with 1[e < 1].

Human mechanism rows are used only by `human_fidelity`, after the validation protocol
freeze, through `data.load_heldout`.

Commands::

    uv run python -m phasespace.citizens.surrogate fit --runs-dir runs --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import Array

from phasespace.citizens import config as cfg
from phasespace.citizens import data, validation
from phasespace.citizens import export as c1
from phasespace.citizens.game import (
    DEPLETION_THRESHOLD,
    EXCLUSION_THRESHOLD,
    GROWTH_FACTOR,
    MAX_POOL,
    N_PLAYERS,
    N_ROUNDS,
    POOL_FLOOR,
)
from phasespace.citizens.mechanisms import (
    DEFAULT_OFFER_RULE,
    INTERPOLATING_DECIMALS,
    K_INTERPOLATING,
    RECORDED_DECIMALS,
    RECORDED_WITHHELD,
    W_EQUAL,
    W_MIXED,
    W_PROPORTIONAL,
)
from phasespace.citizens.records import Games
from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    run_status,
    validate_run_id,
    write_run_directory,
)

type FloatArray = npt.NDArray[np.float64]
type F32Array = npt.NDArray[np.float32]
type BoolArray = npt.NDArray[np.bool_]
type MechanismKey = Literal["M1", "M2"]

# --- configuration (protocol-validation Appendix A, via phasespace.citizens.config) ----

N_FEATURES: Final = 5
N_OUTPUTS: Final = N_PLAYERS + 1
if cfg.SURROGATE_FC != cfg.SURROGATE_GRU:
    raise ImportError("the surrogate uses one hidden width (fc = gru)")
BASE_WIDTH: Final = cfg.SURROGATE_FC
"""fc = gru = 32 (Appendix A); every hidden layer of the surrogate has this width."""
STEP2_WIDTH: Final = cfg.SURROGATE_STEP2_WIDTH
"""The pre-declared capacity step if the base width fails the BC check."""
LEARNING_RATE: Final = cfg.SURROGATE_LEARNING_RATE
BATCH_GAMES: Final = cfg.SURROGATE_BATCH
EVAL_EVERY: Final = cfg.SURROGATE_EVAL_EVERY
PATIENCE: Final = cfg.SURROGATE_PATIENCE
MAX_UPDATES: Final = cfg.SURROGATE_MAX_UPDATES
BC_SPLIT_SEED: Final = cfg.BC_SPLIT_SEED
BC_FIT_GAMES: Final = cfg.BC_FIT_GAMES
BC_VALIDATION_GAMES: Final = cfg.BC_VAL_GAMES
ACCEPT_MEDIAN_ABS: Final = cfg.SURROGATE_ACCEPT_MEDIAN_ABS
"""Acceptance: median |e_hat - e| <= 1 coin."""
ACCEPT_P90_ABS: Final = cfg.SURROGATE_ACCEPT_P90_ABS
"""Acceptance: 90th percentile |e_hat - e| <= 5 coins."""
ACCEPT_EXCLUSION_AGREEMENT: Final = cfg.SURROGATE_ACCEPT_EXCLUSION
"""Acceptance: 1[e_hat < 1] = 1[e < 1] on >= 90 % of player-rounds."""


@dataclass(frozen=True, slots=True)
class MechanismSpec:
    key: MechanismKey
    mechanism: str  # the CSV's base mechanism name
    cohort: str  # the BC cohort that played it
    seeds: tuple[int, ...]  # base width
    step2_seeds: tuple[int, ...]  # capacity step


MECHANISM_SPECS: Final[dict[MechanismKey, MechanismSpec]] = {
    "M1": MechanismSpec(
        "M1", "RL Agent (M1)", "BC 1", cfg.SURROGATE_SEEDS_M1, cfg.SURROGATE_STEP2_SEEDS_M1
    ),
    "M2": MechanismSpec(
        "M2", "RL Agent (M2)", "BC 2", cfg.SURROGATE_SEEDS_M2, cfg.SURROGATE_STEP2_SEEDS_M2
    ),
}


# --- data ------------------------------------------------------------------------------


def features(
    pool: npt.ArrayLike, prev_offers: npt.ArrayLike, prev_reciprocations: npt.ArrayLike
) -> F32Array:
    """(..., 4, 5) per-player inputs (e_prev, c_prev, R, mean e_prev, mean c_prev) / 200."""
    r = np.asarray(pool, dtype=np.float64)
    e = np.asarray(prev_offers, dtype=np.float64)
    c = np.asarray(prev_reciprocations, dtype=np.float64)
    if e.shape[-1] != N_PLAYERS or c.shape != e.shape or r.shape != e.shape[:-1]:
        raise ValueError("need pool (...,), prev_offers and prev_reciprocations (..., 4)")
    out = np.stack(
        [
            e,
            c,
            np.broadcast_to(r[..., None], e.shape),
            np.broadcast_to(e.mean(axis=-1, keepdims=True), e.shape),
            np.broadcast_to(c.mean(axis=-1, keepdims=True), e.shape),
        ],
        axis=-1,
    )
    return (out / MAX_POOL).astype(np.float32)


def target_shares(pool: npt.ArrayLike, offers: npt.ArrayLike) -> F32Array:
    """(..., 5) recorded shares (e_i / R, 1 - sum e / R), clipped to >= 0 and renormalised.

    Recorded M1/M2 offers sum to R within ~2e-7 R in either direction (float32 software),
    so the pool share can be a tiny negative number; it is clipped to 0.
    """
    r = np.asarray(pool, dtype=np.float64)
    e = np.asarray(offers, dtype=np.float64)
    shares = e / r[..., None]
    full = np.concatenate([shares, 1.0 - shares.sum(axis=-1, keepdims=True)], axis=-1)
    full = np.clip(full, 0.0, None)
    return (full / full.sum(axis=-1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class SequenceData:
    """Games as fixed-length sequences: what the mechanism saw and what it offered."""

    inputs: F32Array  # (G, T, 4, 5)
    pool: FloatArray  # (G, T) R_t before allocation
    offers: FloatArray  # (G, T, 4) recorded offers
    targets: F32Array  # (G, T, 5) recorded shares
    mask: BoolArray  # (G, T) recorded rounds with R >= 1 (scored)
    launch_id: npt.NDArray[np.str_]  # (G,)

    @property
    def n_games(self) -> int:
        return int(self.pool.shape[0])

    def take(self, index: npt.NDArray[np.int64]) -> SequenceData:
        return SequenceData(
            inputs=self.inputs[index],
            pool=self.pool[index],
            offers=self.offers[index],
            targets=self.targets[index],
            mask=self.mask[index],
            launch_id=self.launch_id[index],
        )


def sequences(games: Games) -> SequenceData:
    """Mechanism observation -> offer sequences of `games` (trimmed to the longest game)."""
    t = int(games.n_rounds.max())
    recorded = games.round_mask()[:, :t]
    pool = np.where(recorded, games.pool[:, :t], MAX_POOL)
    prev_e = np.where(recorded[..., None], games.prev_offers[:, :t], 0.0)
    prev_c = np.where(recorded[..., None], games.prev_reciprocations[:, :t], 0.0)
    offers = np.where(recorded[..., None], games.offers[:, :t], MAX_POOL / N_PLAYERS)
    if not (np.isfinite(pool).all() and np.isfinite(prev_e).all() and np.isfinite(prev_c).all()):
        raise ValueError("non-finite mechanism observation in a recorded round")
    return SequenceData(
        inputs=features(pool, prev_e, prev_c),
        pool=pool,
        offers=offers,
        targets=target_shares(pool, offers),
        mask=recorded & (pool >= DEPLETION_THRESHOLD),
        launch_id=games.launch_id,
    )


def mechanism_games(training: Games, key: MechanismKey) -> Games:
    """The BC games of mechanism `key` (M1: BC 1, M2: BC 2); refuses human games."""
    spec = MECHANISM_SPECS[key]
    games = training.where(mechanism=spec.mechanism, cohort=spec.cohort)
    if games.is_human.any():
        raise ValueError("surrogates are fitted on BC (clone) games only")
    return games


def bc_split(games: Games, seed: int = BC_SPLIT_SEED) -> tuple[Games, Games]:
    """(fit, validation) = 410/102 games: a fresh default_rng(seed) permutes games sorted by
    launch_id; the first 410 of the permutation are the fit split."""
    if games.n_games != BC_FIT_GAMES + BC_VALIDATION_GAMES:
        raise ValueError(
            f"expected {BC_FIT_GAMES + BC_VALIDATION_GAMES} BC games, got {games.n_games}"
        )
    order = np.argsort(games.launch_id, kind="stable")
    perm = order[np.random.default_rng(seed).permutation(games.n_games)]
    fit = np.zeros(games.n_games, dtype=np.bool_)
    fit[perm[:BC_FIT_GAMES]] = True
    return games.select(fit), games.select(~fit)


# --- network ---------------------------------------------------------------------------


class SurrogateNet(eqx.Module):
    """Shared per-player encoder and GRU; per-player and pooled heads (see module doc)."""

    width: int = eqx.field(static=True)
    encoder: eqx.nn.Linear
    gru: eqx.nn.GRUCell
    player_hidden: eqx.nn.Linear
    player_out: eqx.nn.Linear
    pool_hidden: eqx.nn.Linear
    pool_out: eqx.nn.Linear

    def __init__(self, width: int, key: Array) -> None:
        k = jax.random.split(key, 6)
        self.width = width
        self.encoder = eqx.nn.Linear(N_FEATURES, width, key=k[0])
        self.gru = eqx.nn.GRUCell(width, width, key=k[1])
        self.player_hidden = eqx.nn.Linear(width, width, key=k[2])
        self.player_out = eqx.nn.Linear(width, 1, key=k[3])
        self.pool_hidden = eqx.nn.Linear(width, width, key=k[4])
        self.pool_out = eqx.nn.Linear(width, 1, key=k[5])

    def _encode(self, v: Array) -> Array:
        return jnp.tanh(self.encoder(v))

    def _player_logit(self, h: Array) -> Array:
        return self.player_out(jnp.tanh(self.player_hidden(h)))

    def step(self, hidden: Array, x: Array) -> tuple[Array, Array]:
        """One round of one game: new hidden states and the 5 logits."""
        encoded = jax.vmap(self._encode)(x)
        new_hidden = jax.vmap(self.gru)(encoded, hidden)
        players = jax.vmap(self._player_logit)(new_hidden)
        pool = self.pool_out(jnp.tanh(self.pool_hidden(new_hidden.mean(axis=0))))
        return new_hidden, jnp.concatenate([players[:, 0], pool])

    def initial_hidden(self) -> Array:
        return to_jax(np.zeros((N_PLAYERS, self.width), dtype=np.float32))

    def sequence(self, inputs: Array) -> Array:
        """Logits of every round of one game (teacher-forced inputs)."""

        def body(h: Array, x: Array) -> tuple[Array, Array]:
            return self.step(h, x)

        _, logits = jax.lax.scan(body, self.initial_hidden(), inputs)
        return logits


def to_jax(x: npt.ArrayLike) -> Array:
    """NumPy -> JAX array (jnp.asarray's `device` parameter is untyped in jax 0.12)."""
    return jnp.asarray(x)  # pyright: ignore[reportUnknownMemberType]


def trainable(model: SurrogateNet) -> Any:
    """The model's array leaves (equinox's `filter` is untyped)."""
    return cast(Any, eqx.filter(model, eqx.is_array))  # pyright: ignore[reportUnknownMemberType]


def _batch_logits(model: SurrogateNet, inputs: Array) -> Array:
    return jax.vmap(model.sequence)(inputs)


@eqx.filter_jit
def _logits_jit(model: SurrogateNet, inputs: Array) -> Array:
    return _batch_logits(model, inputs)


@eqx.filter_jit
def _step_jit(model: SurrogateNet, hidden: Array, x: Array) -> tuple[Array, Array]:
    return jax.vmap(model.step)(hidden, x)


def shares_from_logits(logits: npt.ArrayLike) -> FloatArray:
    """Float64 softmax over the last axis (5 shares that sum to 1 to ~1e-16)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=-1, keepdims=True)


def offers_from_logits(pool: npt.ArrayLike, logits: npt.ArrayLike) -> FloatArray:
    """(..., 4) offers R * softmax(logits)[:4] in float64 (sum <= R up to rounding)."""
    r = np.asarray(pool, dtype=np.float64)
    return r[..., None] * shares_from_logits(logits)[..., :N_PLAYERS]


def predict_offers(model: SurrogateNet, seqs: SequenceData) -> FloatArray:
    """(G, T, 4) teacher-forced offers for recorded observation sequences."""
    logits = np.asarray(_logits_jit(model, to_jax(seqs.inputs)))
    return offers_from_logits(seqs.pool, logits)


# --- fidelity --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fidelity:
    """§6.4 acceptance metrics over the player-rounds of rounds with R >= 1."""

    player_rounds: int
    median_abs: float  # coins
    p90_abs: float  # coins
    mean_abs: float  # coins
    exclusion_agreement: float  # fraction of player-rounds
    recorded_exclusion_rate: float  # fraction with e < 1 among the scored player-rounds

    @property
    def accepted(self) -> bool:
        return (
            self.median_abs <= ACCEPT_MEDIAN_ABS
            and self.p90_abs <= ACCEPT_P90_ABS
            and self.exclusion_agreement >= ACCEPT_EXCLUSION_AGREEMENT
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "player_rounds": self.player_rounds,
            "median_abs_coin": self.median_abs,
            "p90_abs_coin": self.p90_abs,
            "mean_abs_coin": self.mean_abs,
            "exclusion_agreement": self.exclusion_agreement,
            "recorded_exclusion_rate": self.recorded_exclusion_rate,
            "accepted": self.accepted,
        }


def fidelity(predicted: npt.ArrayLike, recorded: npt.ArrayLike, mask: npt.ArrayLike) -> Fidelity:
    """Metrics over the player-rounds of rounds where `mask` (G, T) is True."""
    p = np.asarray(predicted, dtype=np.float64)
    e = np.asarray(recorded, dtype=np.float64)
    m = np.asarray(mask, dtype=np.bool_)
    if p.shape != e.shape or p.shape[:-1] != m.shape:
        raise ValueError("predicted/recorded (G, T, 4) and mask (G, T) shapes disagree")
    p_rows, e_rows = p[m].ravel(), e[m].ravel()
    if p_rows.size == 0:
        raise ValueError("no player-rounds to score")
    err = np.abs(p_rows - e_rows)
    excluded = e_rows < EXCLUSION_THRESHOLD
    return Fidelity(
        player_rounds=int(err.size),
        median_abs=float(np.median(err)),
        p90_abs=float(np.percentile(err, 90.0)),
        mean_abs=float(err.mean()),
        exclusion_agreement=float(np.mean(np.equal(p_rows < EXCLUSION_THRESHOLD, excluded))),
        recorded_exclusion_rate=float(excluded.mean()),
    )


def offer_mae(model: SurrogateNet, seqs: SequenceData) -> float:
    """Validation criterion: mean |e_hat - e| over scored player-rounds (coins)."""
    return fidelity(predict_offers(model, seqs), seqs.offers, seqs.mask).mean_abs


# --- training --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainConfig:
    width: int = BASE_WIDTH
    learning_rate: float = LEARNING_RATE
    batch_games: int = BATCH_GAMES
    eval_every: int = EVAL_EVERY
    patience: int = PATIENCE
    max_updates: int = MAX_UPDATES

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "learning_rate": self.learning_rate,
            "batch_games": self.batch_games,
            "eval_every": self.eval_every,
            "patience": self.patience,
            "max_updates": self.max_updates,
            "optimizer": "adam (optax defaults b1=0.9, b2=0.999, eps=1e-8)",
            "loss": "cross-entropy of recorded 5-way shares, recorded rounds with R >= 1",
        }


@dataclass(frozen=True, slots=True)
class FitResult:
    seed: int
    config: TrainConfig
    model: SurrogateNet
    best_update: int
    best_validation_mae: float
    updates_run: int
    stopped: str  # "patience" or "max_updates"
    seconds: float
    curve: list[tuple[int, float, float]] = field(default_factory=list[tuple[int, float, float]])
    # (update, mean training loss since last evaluation, validation MAE)


def _loss(
    model: SurrogateNet,
    inputs: Array,
    targets: Array,
    mask: Array,
) -> Array:
    logp = jax.nn.log_softmax(_batch_logits(model, inputs), axis=-1)
    ce = -(targets * logp).sum(axis=-1)
    return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def make_update(
    optimizer: optax.GradientTransformation,
) -> Callable[..., tuple[SurrogateNet, Any, Array]]:
    @eqx.filter_jit
    def update(
        model: SurrogateNet,
        opt_state: Any,
        inputs: Array,
        targets: Array,
        mask: Array,
    ) -> tuple[SurrogateNet, Any, Array]:
        # equinox's filter_value_and_grad / apply_updates return untyped pytrees.
        loss, grads = eqx.filter_value_and_grad(_loss)(model, inputs, targets, mask)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        updates, opt_state = optimizer.update(grads, opt_state, trainable(model))  # pyright: ignore[reportUnknownArgumentType]
        new_model = cast(SurrogateNet, eqx.apply_updates(model, updates))  # pyright: ignore[reportUnknownMemberType]
        return new_model, opt_state, loss

    return update


def fit_seed(
    fit_data: SequenceData,
    validation_data: SequenceData,
    seed: int,
    config: TrainConfig | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> FitResult:
    """Train one surrogate; early stopping on validation offer MAE; best checkpoint kept.

    Deterministic given `seed`: the initial weights come from jax.random.key(seed) and
    the batch order from numpy default_rng(seed) (a permutation of the fit games per
    epoch, consecutive batches of `batch_games`, the last partial batch of an epoch
    dropped).
    """
    config = config or TrainConfig()
    start = time.perf_counter()
    model = SurrogateNet(config.width, jax.random.key(seed))
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(trainable(model))
    update = make_update(optimizer)
    rng = np.random.default_rng(seed)
    inputs = to_jax(fit_data.inputs)
    targets = to_jax(fit_data.targets)
    mask = to_jax(fit_data.mask.astype(np.float32))
    n = fit_data.n_games
    batch = min(config.batch_games, n)
    per_epoch = n // batch
    best_model, best_mae, best_update = model, float("inf"), 0
    curve: list[tuple[int, float, float]] = []
    bad, losses, stopped = 0, list[float](), "max_updates"
    order = rng.permutation(n)
    step_in_epoch = 0
    updates = 0
    for updates in range(1, config.max_updates + 1):
        if step_in_epoch == per_epoch:
            order, step_in_epoch = rng.permutation(n), 0
        idx = to_jax(order[step_in_epoch * batch : (step_in_epoch + 1) * batch])
        step_in_epoch += 1
        model, opt_state, loss = update(model, opt_state, inputs[idx], targets[idx], mask[idx])
        losses.append(float(loss))
        if updates % config.eval_every == 0:
            mae = offer_mae(model, validation_data)
            curve.append((updates, float(np.mean(losses)), mae))
            losses = []
            if log is not None:
                log(f"  seed {seed} width {config.width} update {updates}: val MAE {mae:.4f}")
            if mae < best_mae:
                best_model, best_mae, best_update, bad = model, mae, updates, 0
            else:
                bad += 1
                if bad >= config.patience:
                    stopped = "patience"
                    break
    return FitResult(
        seed=seed,
        config=config,
        model=best_model,
        best_update=best_update,
        best_validation_mae=best_mae,
        updates_run=updates,
        stopped=stopped,
        seconds=time.perf_counter() - start,
        curve=curve,
    )


# --- the mechanism adapter --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurrogateState:
    """Hidden state of a batch of games: (G, 4, width) float32."""

    hidden: F32Array


@dataclass(frozen=True, slots=True)
class SurrogateMechanism:
    """A fitted surrogate as a deterministic, batched allocation rule.

    Frozen: `name` is a read-only field, as `simulate.Mechanism` declares it.
    Satisfies `simulate.Mechanism`: pool and the previous round's offers and
    reciprocations in, offers out, with an explicit recurrent state (the RL mechanisms
    observed previous offers and carry memory)::

        state = mech.initial_state(n_games)
        offers, state = mech.offers(state, pool, None, None, 0)            # round 1
        offers, state = mech.offers(state, pool, prev_offers, prev_c, t)   # later rounds

    `round_index` is accepted for interface compatibility and not used: the network sees
    only the observation sequence.

    `pool` is (G,), previous offers/reciprocations (G, 4) or None in round 1; offers are
    float64 (G, 4), non-negative with sum <= R up to float rounding (well inside
    `game.OFFER_SUM_RTOL`). Pure: the state is never mutated.
    """

    name: str
    model: SurrogateNet

    def initial_state(self, n_games: int) -> SurrogateState:
        return SurrogateState(np.zeros((n_games, N_PLAYERS, self.model.width), dtype=np.float32))

    def offers(
        self,
        state: SurrogateState,
        pool: npt.ArrayLike,
        prev_offers: npt.ArrayLike | None,
        prev_reciprocations: npt.ArrayLike | None,
        round_index: int = 0,
    ) -> tuple[FloatArray, SurrogateState]:
        del round_index
        r = np.atleast_1d(np.asarray(pool, dtype=np.float64))
        g = int(r.shape[0])
        if state.hidden.shape[0] != g:
            raise ValueError("state and pool disagree on the number of games")
        if (prev_offers is None) != (prev_reciprocations is None):
            raise ValueError("pass both previous offers and reciprocations, or neither")
        zeros = np.zeros((g, N_PLAYERS))
        e = zeros if prev_offers is None else np.asarray(prev_offers, dtype=np.float64)
        c = zeros if prev_reciprocations is None else np.asarray(prev_reciprocations)
        hidden, logits = _step_jit(self.model, to_jax(state.hidden), to_jax(features(r, e, c)))
        return offers_from_logits(r, np.asarray(logits)), SurrogateState(np.asarray(hidden))

    def teacher_forced(self, seqs: SequenceData) -> FloatArray:
        return predict_offers(self.model, seqs)


# --- (de)serialisation -----------------------------------------------------------------


def serialise(model: SurrogateNet) -> bytes:
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, model)  # pyright: ignore[reportUnknownMemberType]
    return buffer.getvalue()


def deserialise(payload: bytes, width: int) -> SurrogateNet:
    skeleton = SurrogateNet(width, jax.random.key(0))
    loaded = eqx.tree_deserialise_leaves(io.BytesIO(payload), skeleton)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast(SurrogateNet, loaded)


def load_mechanism(run_dir: Path, key: MechanismKey) -> SurrogateMechanism:
    """The kept surrogate of `key` from a fit run, verified against the manifest sha256."""
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    entry: dict[str, Any] = manifest["citizens"]["surrogates"][key]
    path: str = entry["parameters"]
    payload = (run_dir / path).read_bytes()
    expected = next(a["sha256"] for a in manifest["artifacts"] if a["path"] == path)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{run_dir / path}: sha256 differs from the manifest")
    return SurrogateMechanism(
        name=f"{MECHANISM_SPECS[key].mechanism} surrogate",
        model=deserialise(payload, int(entry["width"])),
    )


# --- fitting both mechanisms -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MechanismFit:
    key: MechanismKey
    fit_games: int
    validation_games: int
    validation_launch_ids: list[str]
    attempts: list[FitResult]  # every seed trained, in order
    kept: FitResult
    validation_fidelity: Fidelity
    fit_fidelity: Fidelity
    step2_used: bool


def _best(results: Sequence[FitResult]) -> FitResult:
    return min(results, key=lambda r: (r.best_validation_mae, r.seed))


def fit_mechanism(
    training: Games,
    key: MechanismKey,
    *,
    max_updates: int = MAX_UPDATES,
    seeds: Sequence[int] | None = None,
    log: Callable[[str], None] | None = None,
) -> MechanismFit:
    """Fit the base-width seeds; if the kept one fails the BC check, the width-64 step."""
    spec = MECHANISM_SPECS[key]
    fit_games, val_games = bc_split(mechanism_games(training, key))
    fit_seqs, val_seqs = sequences(fit_games), sequences(val_games)
    attempts: list[FitResult] = []
    base = TrainConfig(width=BASE_WIDTH, max_updates=max_updates)
    for seed in seeds if seeds is not None else spec.seeds:
        attempts.append(fit_seed(fit_seqs, val_seqs, seed, base, log=log))
    kept = _best(attempts)
    val_fid = fidelity(predict_offers(kept.model, val_seqs), val_seqs.offers, val_seqs.mask)
    step2 = False
    if not val_fid.accepted and seeds is None:
        step2 = True
        wide = TrainConfig(width=STEP2_WIDTH, max_updates=max_updates)
        step2_results = [fit_seed(fit_seqs, val_seqs, s, wide, log=log) for s in spec.step2_seeds]
        attempts.extend(step2_results)
        kept = _best(step2_results)
        val_fid = fidelity(predict_offers(kept.model, val_seqs), val_seqs.offers, val_seqs.mask)
    fit_fid = fidelity(predict_offers(kept.model, fit_seqs), fit_seqs.offers, fit_seqs.mask)
    return MechanismFit(
        key=key,
        fit_games=fit_games.n_games,
        validation_games=val_games.n_games,
        validation_launch_ids=sorted(val_games.launch_id.tolist()),
        attempts=attempts,
        kept=kept,
        validation_fidelity=val_fid,
        fit_fidelity=fit_fid,
        step2_used=step2,
    )


# --- human-row fidelity (after the freeze only) ------------------------------------------


@dataclass(frozen=True, slots=True)
class HumanFidelity:
    protocol_sha256: str
    by_experiment: dict[tuple[MechanismKey, str], Fidelity]


def human_fidelity(
    mechanisms: dict[MechanismKey, SurrogateMechanism],
    *,
    confirm_frozen: bool,
    processed_dir: Path = data.DEFAULT_PROCESSED_DIR,
) -> HumanFidelity:
    """§6.4 acceptance on held-out human M1/M2 rows, per (mechanism, experiment).

    Runs the post-freeze guard first (`heldout.require_post_freeze`: the validation
    protocol frozen and committed, `confirm_frozen` True, a clean working tree), then
    reads held-out games only through `data.load_heldout`. Offers are teacher-forced
    from each human game's recorded observation sequence.
    """
    from phasespace.citizens.heldout import require_post_freeze  # imports this module lazily

    require_post_freeze(confirm_frozen)
    heldout = data.load_heldout(confirm_frozen=confirm_frozen, processed_dir=processed_dir)
    out: dict[tuple[MechanismKey, str], Fidelity] = {}
    for key, mech in sorted(mechanisms.items()):
        games = heldout.games.where(mechanism=MECHANISM_SPECS[key].mechanism)
        for cohort in sorted(set(games.cohort.tolist())):
            seqs = sequences(games.where(cohort=cohort))
            out[(key, cohort)] = fidelity(mech.teacher_forced(seqs), seqs.offers, seqs.mask)
    return HumanFidelity(protocol_sha256=heldout.protocol.sha256, by_experiment=out)


# --- the fit run -----------------------------------------------------------------------

STAGE: Final = "c2_surrogate_fit"
EXPERIMENT_ID: Final = "citizens-c2-surrogates"


def _checks(fits: Sequence[MechanismFit]) -> list[Check]:
    out: list[Check] = []
    for f in fits:
        v = f.validation_fidelity
        n = f"{f.validation_games} BC validation games, {v.player_rounds} player-rounds R >= 1"
        out.append(
            Check(
                f"surrogate_{f.key.lower()}_bc_val_median_abs",
                v.median_abs,
                ACCEPT_MEDIAN_ABS,
                f"median |e_hat - e| (coins), teacher-forced, {n}",
            )
        )
        out.append(
            Check(
                f"surrogate_{f.key.lower()}_bc_val_p90_abs",
                v.p90_abs,
                ACCEPT_P90_ABS,
                f"90th percentile |e_hat - e| (coins), {n}",
            )
        )
        out.append(
            Check(
                f"surrogate_{f.key.lower()}_bc_val_exclusion_disagreement",
                1.0 - v.exclusion_agreement,
                round(1.0 - ACCEPT_EXCLUSION_AGREEMENT, 10),
                f"fraction of player-rounds with 1[e_hat < 1] != 1[e < 1], {n}",
            )
        )
    return out


def _curve_table(run_id: str, fits: Sequence[MechanismFit]) -> bytes:
    rows: list[list[Any]] = []
    for f in fits:
        for a in f.attempts:
            for update, loss, mae in a.curve:
                rows.append([f.key, a.seed, a.config.width, update, loss, mae])
    return dumps_json(
        {
            "schema_version": 1,
            "run_id": run_id,
            "columns": [
                {"name": "mechanism", "unit": None, "definition": "M1 or M2"},
                {"name": "seed", "unit": "count", "definition": "training seed"},
                {"name": "width", "unit": "count", "definition": "hidden width"},
                {"name": "update", "unit": "count", "definition": "Adam updates so far"},
                {
                    "name": "train_loss",
                    "unit": "dimensionless",
                    "definition": "mean batch cross-entropy (nats) since the last evaluation",
                },
                {
                    "name": "validation_mae",
                    "unit": "coin",
                    "definition": "mean |e_hat - e| on the BC validation split",
                },
            ],
            "rows": rows,
        }
    )


def build_fit_run(
    fits: Sequence[MechanismFit],
    run_id: str,
    *,
    training_sidecar: dict[str, Any],
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
    pre_freeze: bool,
    validation_protocol: dict[str, Any] | None = None,
    reference: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str, bytes]]]:
    """The fit run. `reference` maps "M1"/"M2" to the SHA-256 of a previous run's
    parameter file (the pre-freeze fit); each must be reproduced bit for bit."""
    timestamp = created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    checks = _checks(fits)
    if reference is not None:
        for f in fits:
            digest = hashlib.sha256(serialise(f.kept.model)).hexdigest()
            checks.append(
                Check(
                    f"surrogate_{f.key.lower()}_differs_from_reference",
                    0.0 if digest == reference.get(f.key) else 1.0,
                    0.0,
                    "1 if the kept surrogate's parameter bytes differ from the reference "
                    "(pre-freeze) run's, else 0 (the fit is deterministic given its seeds)",
                )
            )
    files: list[tuple[str, str, bytes]] = []
    surrogates: dict[str, Any] = {}
    for f in fits:
        name = f"surrogate-{f.key.lower()}.eqx"
        files.append(("model_parameters", name, serialise(f.kept.model)))
        surrogates[f.key] = {
            "mechanism": MECHANISM_SPECS[f.key].mechanism,
            "bc_cohort": MECHANISM_SPECS[f.key].cohort,
            "parameters": name,
            "format": "equinox.tree_serialise_leaves of SurrogateNet(width)",
            "width": f.kept.config.width,
            "kept_seed": f.kept.seed,
            "step2_used": f.step2_used,
            "fit_games": f.fit_games,
            "validation_games": f.validation_games,
            "validation_launch_ids": f.validation_launch_ids,
            "attempts": [
                {
                    "seed": a.seed,
                    "width": a.config.width,
                    "best_update": a.best_update,
                    "best_validation_mae_coin": a.best_validation_mae,
                    "updates_run": a.updates_run,
                    "stopped": a.stopped,
                    "seconds": a.seconds,
                }
                for a in f.attempts
            ],
            "bc_validation_fidelity": f.validation_fidelity.to_json(),
            "bc_fit_fidelity": f.fit_fidelity.to_json(),
        }
    files.append(("sample_table", "training_curves.json", _curve_table(run_id, fits)))
    status = run_status(checks)
    limitations = [
        "Fitted on simulated clone (BC) games only; BC fidelity says how well the "
        "surrogate imitates M1/M2 against clones, not against humans. Acceptance on held-"
        "out human mechanism rows (§6.4) is evaluated only after the protocol freeze.",
        "The surrogate observes per-player summaries (own previous offer and "
        "reciprocation, pool, the means over players), not the paper's full GNN "
        "observation graph.",
    ]
    if pre_freeze:
        status = "computed_unvalidated"
        limitations.insert(
            0,
            "Pre-freeze development fit (protocol-validation.md is a draft): the "
            "thresholds are not frozen and the fit is redone after the freeze (§10.2). "
            "Status is computed_unvalidated whatever the checks say.",
        )
    if git_dirty:
        limitations.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "study_id": c1.STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "created_at": timestamp,
        "source": {
            "repository": c1.REPOSITORY,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generator": "phasespace.citizens.surrogate",
            "generator_version": version("phasespace"),
        },
        "environment": {
            **environment_info(),
            "jax": jax.__version__,
            "equinox": version("equinox"),
            "optax": version("optax"),
            "jax_backend": jax.default_backend(),
        },
        "model": citizens_model(),
        "simulator": None,
        "random_seed": BC_SPLIT_SEED,
        "artifacts": [],
        "metrics": {
            f"{f.key.lower()}_bc_val_{k}": {
                "value": v,
                "unit": "coin" if k.endswith("abs") else "dimensionless",
                "definition": f"{f.key} surrogate, BC validation split, {k}",
            }
            for f in fits
            for k, v in (
                ("median_abs", f.validation_fidelity.median_abs),
                ("p90_abs", f.validation_fidelity.p90_abs),
                ("exclusion_agreement", f.validation_fidelity.exclusion_agreement),
            )
        },
        "limitations": limitations,
        "validation": {
            "checks": [c.to_json() for c in checks],
            "validated_by": "phasespace.citizens.surrogate automated checks "
            "(pass: observed <= threshold; protocol-validation §6.4 BC check)",
            "validated_at": timestamp,
        },
        "citizens": {
            "stage": STAGE,
            "pre_freeze": pre_freeze,
            "training_data": {
                "parser_version": training_sidecar["parser_version"],
                "training_sha256": training_sidecar["file_sha256"],
                "rows_used": "BC 1 'RL Agent (M1)' and BC 2 'RL Agent (M2)' games only; "
                "no human row",
            },
            "split": {
                "seed": BC_SPLIT_SEED,
                "fit_games": BC_FIT_GAMES,
                "validation_games": BC_VALIDATION_GAMES,
                "rule": "fresh numpy default_rng(seed) per mechanism permutes the games "
                "sorted by launch_id; the first 410 are the fit split",
            },
            "train_config": TrainConfig().to_json(),
            "acceptance": {
                "median_abs_coin": ACCEPT_MEDIAN_ABS,
                "p90_abs_coin": ACCEPT_P90_ABS,
                "exclusion_agreement": ACCEPT_EXCLUSION_AGREEMENT,
            },
            "config_sha256": cfg.config_sha256(),
            "validation_protocol": validation_protocol
            or {
                "path": data.VALIDATION_PROTOCOL_PATH.as_posix(),
                "held_out_access": "none",
            },
            "reference_parameters_sha256": reference,
            "surrogates": surrogates,
        },
    }
    return manifest, files


def protocol_record() -> dict[str, Any]:
    """Path, version, header status and SHA-256 of the validation protocol at run time."""
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    content = path.read_bytes()
    return {
        "path": data.VALIDATION_PROTOCOL_PATH.as_posix(),
        "version": cfg.PROTOCOL_VERSION,
        "status": data.header_status(content.decode("utf-8")),
        "sha256": hashlib.sha256(content).hexdigest(),
        "held_out_access": "none (BC rows only)",
    }


def citizens_model() -> dict[str, Any]:
    return {
        "name": "koster2025-common-pool",
        "units": c1.COIN,
        "players": N_PLAYERS,
        "max_pool": MAX_POOL,
        "growth_factor": GROWTH_FACTOR,
        "pool_floor": POOL_FLOOR,
        "depletion_threshold": DEPLETION_THRESHOLD,
        "exclusion_threshold": EXCLUSION_THRESHOLD,
        "rounds_exps_1_3": N_ROUNDS,
        "baseline_weights": {"equal": W_EQUAL, "mixed": W_MIXED, "proportional": W_PROPORTIONAL},
        "interpolating_k": K_INTERPOLATING,
        "interpolating_weight_decimals": INTERPOLATING_DECIMALS,
        "recorded_offer_withheld": RECORDED_WITHHELD,
        "recorded_offer_decimals": RECORDED_DECIMALS,
        "offer_rule": DEFAULT_OFFER_RULE,
        "legacy_remainder_dropped_games": len(validation.LEGACY_REMAINDER_DROPPED),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.surrogate")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit the M1 and M2 surrogates on BC games (new run)")
    fit.add_argument("--run-id", required=True)
    fit.add_argument(
        "--reproduce",
        type=Path,
        default=None,
        help="a previous surrogate run whose parameter files must be reproduced bit for bit",
    )
    args = parser.parse_args(argv)
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    protocol = protocol_record()
    reference: dict[str, str] | None = None
    if args.reproduce is not None:
        ref = json.loads((args.reproduce / "manifest.json").read_text("utf-8"))
        shas = {a["path"]: a["sha256"] for a in ref["artifacts"]}
        reference = {k: shas[v["parameters"]] for k, v in ref["citizens"]["surrogates"].items()}
    training = data.load_training(args.processed_dir)
    fits: list[MechanismFit] = []
    for key in ("M1", "M2"):
        print(f"fitting the {key} surrogate", flush=True)
        f = fit_mechanism(training, key, log=lambda s: print(s, flush=True))
        print(f"{key}: kept seed {f.kept.seed} width {f.kept.config.width}", flush=True)
        print(json.dumps(f.validation_fidelity.to_json(), indent=2), flush=True)
        fits.append(f)
    commit, dirty = git_source()
    manifest, files = build_fit_run(
        fits,
        run_id,
        training_sidecar=data.processed_sidecar(args.processed_dir, data.TRAINING_FILE),
        git_commit=commit,
        git_dirty=dirty,
        created_at=datetime.now(UTC),
        pre_freeze=protocol["status"] != data.FROZEN,
        validation_protocol=protocol,
        reference=reference,
    )
    write_run_directory(out_dir, manifest, files)
    print(f"{out_dir}: status {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
