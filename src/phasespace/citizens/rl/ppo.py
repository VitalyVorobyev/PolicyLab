"""PPO for the planner (pure JAX; deterministic given its seed).

Proximal policy optimisation (Schulman et al. 2017, arXiv:1707.06347) with generalised
advantage estimation (Schulman et al. 2016, arXiv:1506.02438), specialised to this game:
every episode has the same fixed horizon (40 rounds; no early termination), so a batch
is G complete games played in parallel and the recurrent policy is re-evaluated over
whole sequences (back-propagation through its GRU over the episode).

- **Reward**: the round's total surplus sum_i (e_i - c_i) (Koster et al. 2025: maximise
  the total surplus over the episode), times `reward_scale` for the critic; return
  undiscounted (gamma = 1) by default. Gini is never part of the objective.
- **Update**: `epochs` passes over the batch in `minibatches` random splits of the
  games; clipped surrogate (`clip`), squared-error critic loss (`value_coef`), entropy
  bonus (`entropy_coef`; the logistic-normal head's entropy depends on log sigma only),
  advantages normalised per minibatch, global gradient-norm clip, Adam.
- **Randomness**: one key per iteration splits into the citizens' key, the planner's
  sampling key and the minibatch permutation key.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import Array

from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl import twin
from phasespace.citizens.rl.citizens import JaxCitizen, Trajectory, rollout
from phasespace.citizens.rl.planner import (
    Planner,
    PlannerExtras,
    deterministic_step,
    gaussian_entropy,
    gaussian_log_prob,
    sampling_step,
)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Every PPO hyperparameter; defaults are `rl.config` (recorded in each training run)."""

    n_games: int = c3config.PPO_N_GAMES
    n_rounds: int = c3config.PPO_N_ROUNDS
    epochs: int = c3config.PPO_EPOCHS
    minibatches: int = c3config.PPO_MINIBATCHES
    learning_rate: float = c3config.PPO_LEARNING_RATE
    clip: float = c3config.PPO_CLIP
    gamma: float = c3config.PPO_GAMMA
    gae_lambda: float = c3config.PPO_GAE_LAMBDA
    value_coef: float = c3config.PPO_VALUE_COEF
    entropy_coef: float = c3config.PPO_ENTROPY_COEF
    max_grad_norm: float = c3config.PPO_MAX_GRAD_NORM
    reward_scale: float = c3config.PPO_REWARD_SCALE
    hidden: int = c3config.PPO_HIDDEN
    init_log_std: float = c3config.PPO_INIT_LOG_STD
    retention: bool = True
    """False: the no-retention ablation planner (C4; `planner.Planner`)."""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def steps_per_iteration(self) -> int:
        """Environment steps (planner decisions = game rounds) per iteration."""
        return self.n_games * self.n_rounds


class Batch(NamedTuple):
    features: Array  # (T, G, 4, 3)
    z: Array  # (T, G, 5)
    log_prob: Array  # (T, G)
    advantage: Array  # (T, G)
    returns: Array  # (T, G)


def gae(rewards: Array, values: Array, gamma: float, lam: float) -> tuple[Array, Array]:
    """(T, G) rewards and values -> advantages and returns; V = 0 after the last round."""

    def body(carry: Array, inp: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        r, v, v_next = inp
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * carry
        return adv, adv

    v_next = jnp.concatenate([values[1:], twin.zeros_like(values[:1])], axis=0)
    _, adv = jax.lax.scan(body, twin.zeros_like(values[0]), (rewards, values, v_next), reverse=True)
    return adv, adv + values


def round_surplus(traj: Trajectory) -> Array:
    """(G, T) total surplus sum_i (e_i - c_i) of every round."""
    return (traj.offers - traj.actions).sum(axis=-1)


def _sequence_outputs(planner: Planner, features: Array) -> tuple[Array, Array]:
    """Re-run actor and critic over (T, G, 4, 3) features -> mu (T, G, 5), value (T, G)."""
    g = features.shape[1]
    h0 = planner.initial_state(g)

    def body(
        carry: tuple[Array, Array], inp: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        h_pi, h_v = carry
        x, t = inp
        h_pi, mu = planner.policy_step(h_pi, x)
        h_v, v = planner.value_step(h_v, x, t)
        return (h_pi, h_v), (mu, v)

    _, (mu, v) = jax.lax.scan(body, h0, (features, twin.arange(features.shape[0])))
    return mu, v


def replay(planner: Planner, batch: Batch) -> tuple[Array, Array]:
    """Re-run the planner over a batch's games: log-prob of the stored z (T, G) and the
    critic's values (T, G)."""
    mu, v = _sequence_outputs(planner, batch.features)
    return gaussian_log_prob(batch.z, mu, planner.log_std), v


def ppo_loss(planner: Planner, batch: Batch, cfg: PPOConfig) -> tuple[Array, dict[str, Array]]:
    logp, v = replay(planner, batch)
    ratio = jnp.exp(logp - batch.log_prob)
    adv = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + 1e-8)
    surrogate = jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv)
    policy_loss = -surrogate.mean()
    value_loss = 0.5 * ((v - batch.returns) ** 2).mean()
    entropy = gaussian_entropy(planner.log_std)
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
    stats = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
        "clip_fraction": (jnp.abs(ratio - 1.0) > cfg.clip).mean(),
    }
    return loss, stats


def make_optimizer(cfg: PPOConfig) -> Any:
    return optax.chain(  # pyright: ignore[reportUnknownMemberType]
        optax.clip_by_global_norm(cfg.max_grad_norm),  # pyright: ignore[reportUnknownMemberType]
        optax.adam(cfg.learning_rate),  # pyright: ignore[reportUnknownMemberType]
    )


class TrainState(NamedTuple):
    params: Any  # the planner's array leaves
    opt_state: Any


def array_part(planner: Planner) -> Any:
    """The planner's array leaves (equinox's `filter` is untyped)."""
    return cast(Any, eqx.filter(planner, eqx.is_array))  # pyright: ignore[reportUnknownMemberType]


def static_part(planner: Planner) -> Any:
    return cast(Any, eqx.filter(planner, eqx.is_array, inverse=True))  # pyright: ignore[reportUnknownMemberType]


def combine(params: Any, static: Any) -> Planner:
    return cast(Planner, eqx.combine(params, static))  # pyright: ignore[reportUnknownMemberType]


def init_train_state(planner: Planner, cfg: PPOConfig) -> TrainState:
    params = array_part(planner)
    return TrainState(params, make_optimizer(cfg).init(params))  # pyright: ignore[reportUnknownMemberType]


def collect(
    planner: Planner, citizen: JaxCitizen, cfg: PPOConfig, k_citizen: Array, k_mech: Array
) -> tuple[Batch, Array]:
    """Play `cfg.n_games` games with the sampling policy: the training batch (advantages
    and returns by GAE) and the (T, G) round surpluses."""
    traj, extras = rollout(
        citizen,
        sampling_step(planner),
        planner.initial_state(cfg.n_games),
        k_citizen,
        k_mech,
        cfg.n_games,
        cfg.n_rounds,
    )
    ex: PlannerExtras = extras
    rewards = round_surplus(traj).T  # (T, G)
    adv, ret = gae(rewards * cfg.reward_scale, ex.value, cfg.gamma, cfg.gae_lambda)
    return Batch(ex.features, ex.z, ex.log_prob, adv, ret), rewards


def minibatch_indices(cfg: PPOConfig, key: Array) -> Array:
    """(minibatches, G / minibatches) game indices of one epoch (a random permutation)."""
    mb = cfg.n_games // cfg.minibatches
    return jax.random.permutation(key, cfg.n_games).reshape(cfg.minibatches, mb)


def make_iteration(
    static: Planner, citizen: JaxCitizen, cfg: PPOConfig
) -> Callable[[TrainState, Array], tuple[TrainState, dict[str, Array]]]:
    """One PPO iteration: collect G games with the sampling policy, then update."""
    optimizer = make_optimizer(cfg)
    frozen = static_part(static)

    def loss_fn(params: Any, batch: Batch) -> tuple[Array, dict[str, Array]]:
        return ppo_loss(combine(params, frozen), batch, cfg)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def iteration(state: TrainState, key: Array) -> tuple[TrainState, dict[str, Array]]:
        planner = combine(state.params, frozen)
        k_citizen, k_mech, k_perm = jax.random.split(key, 3)
        full, rewards = collect(planner, citizen, cfg, k_citizen, k_mech)

        def epoch(carry: tuple[Any, Any], k: Array) -> tuple[tuple[Any, Any], dict[str, Array]]:
            perm = minibatch_indices(cfg, k)

            def minibatch(
                c: tuple[Any, Any], idx: Array
            ) -> tuple[tuple[Any, Any], dict[str, Array]]:
                params, opt_state = c
                batch = Batch(*(a[:, idx] for a in full))
                (_, stats), grads = grad_fn(params, batch)
                updates, opt_state = optimizer.update(grads, opt_state, params)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                params = optax.apply_updates(params, updates)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                return (params, opt_state), stats

            return jax.lax.scan(minibatch, carry, perm)

        (params, opt_state), stats = jax.lax.scan(
            epoch, (state.params, state.opt_state), jax.random.split(k_perm, cfg.epochs)
        )
        out = {k: v.mean() for k, v in stats.items()}
        out["episode_surplus"] = rewards.sum(axis=0).mean()
        out["log_std_mean"] = planner.log_std.mean()
        return TrainState(params, opt_state), out

    return iteration


def make_chunk(
    static: Planner, citizen: JaxCitizen, cfg: PPOConfig, n_iterations: int
) -> Callable[[TrainState, Array], tuple[TrainState, dict[str, Array]]]:
    """`n_iterations` PPO iterations as one jitted scan (keys split from `key`)."""
    iteration = make_iteration(static, citizen, cfg)

    @eqx.filter_jit
    def chunk(state: TrainState, key: Array) -> tuple[TrainState, dict[str, Array]]:
        return jax.lax.scan(iteration, state, jax.random.split(key, n_iterations))

    return chunk


@dataclass(frozen=True, slots=True)
class CurvePoint:
    iteration: int
    env_steps: int
    seconds: float
    episode_surplus: float  # sampled policy, total over the episode, mean over the batch
    eval_surplus: float | None  # deterministic policy on the fixed monitoring games
    approx_kl: float
    clip_fraction: float
    value_loss: float
    log_std_mean: float


def train(
    citizen: JaxCitizen,
    cfg: PPOConfig,
    seed: int,
    n_iterations: int,
    *,
    chunk: int = 10,
    monitor: Callable[[Planner], float] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Planner, list[CurvePoint]]:
    """Train a planner from `seed` for `n_iterations` PPO iterations.

    `monitor` (optional) scores the deterministic planner after every chunk (the
    learning curve's evaluation column); it must not use evaluation games.
    """
    twin.require_x64()
    if n_iterations % chunk:
        raise ValueError("n_iterations must be a multiple of chunk")
    k_init, k_train = jax.random.split(jax.random.key(seed))
    planner = Planner(cfg.hidden, cfg.init_log_std, k_init, retention=cfg.retention)
    state = init_train_state(planner, cfg)
    run_chunk = make_chunk(planner, citizen, cfg, chunk)
    frozen = static_part(planner)
    curve: list[CurvePoint] = []
    start = time.perf_counter()
    keys = jax.random.split(k_train, n_iterations // chunk)
    for c in range(n_iterations // chunk):
        state, stats = run_chunk(state, keys[c])
        last = {k: float(np.asarray(v)[-1]) for k, v in stats.items()}
        current = combine(state.params, frozen)
        score = monitor(current) if monitor is not None else None
        it = (c + 1) * chunk
        point = CurvePoint(
            iteration=it,
            env_steps=it * cfg.steps_per_iteration,
            seconds=time.perf_counter() - start,
            episode_surplus=float(np.asarray(stats["episode_surplus"]).mean()),
            eval_surplus=score,
            approx_kl=last["approx_kl"],
            clip_fraction=last["clip_fraction"],
            value_loss=last["value_loss"],
            log_std_mean=last["log_std_mean"],
        )
        curve.append(point)
        if log is not None:
            log(
                f"it {it:6d}  steps {point.env_steps / 1e6:8.2f}M  {point.seconds:8.1f}s  "
                f"sampled {point.episode_surplus:8.2f}  eval {score if score is not None else float('nan'):8.2f}  "
                f"kl {point.approx_kl:.4f}  logstd {point.log_std_mean:+.3f}"
            )
    return combine(state.params, frozen), curve


def evaluate_planner(
    planner: Planner, citizen: JaxCitizen, citizen_key: Array, n_games: int, n_rounds: int = 40
) -> tuple[Trajectory, Array]:
    """Deterministic planner (shares = softmax(mu)) on `n_games` games of `citizen_key`;
    returns the trajectory and the (T, G, 5) shares."""
    twin.require_x64()
    return rollout(
        citizen,
        deterministic_step(planner),
        planner.initial_state(n_games),
        citizen_key,
        jax.random.key(0),
        n_games,
        n_rounds,
    )
