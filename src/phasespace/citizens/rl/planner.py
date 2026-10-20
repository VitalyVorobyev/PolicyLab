"""The PPO planner: a permutation-equivariant set encoder over players plus a GRU (pure).

**Observation** (Koster et al. 2025 Fig S16; README "Planners"): in round t the planner
sees, per player i, the previous offer e_{i,t-1}, the previous reciprocation c_{i,t-1}
(both 0 in round 1) and the current pool R_t, each divided by 200.

**Network** (the analogue of the paper's GNN over a fully connected player graph), with
H = `hidden` units and players treated as a set:

    u_i = tanh(W_in x_i + b),                                   per-player encoder
    v_i = tanh(A u_i + B mean_j u_j + b'),                      equivariant mixing
    h_i' = GRU(v_i, h_i),                                       one shared cell per player
    y_i = tanh(C h_i' + D mean_j h_j' + b''),                   equivariant mixing
    mu_i = w . y_i + b_w  (i = 1..4),   mu_pool = MLP(mean_j h_j'),

so permuting the players permutes mu_1..4 and leaves mu_pool unchanged.

**Action** (logistic-normal): z ~ N(mu, diag(sigma^2)) in R^5 with a learned,
state-independent log sigma; shares = softmax(z) (4 players, then the amount kept in
the pool); offers = R * shares[:4]. PPO's ratio uses the Gaussian density of z. At
evaluation the planner is deterministic: shares = softmax(mu).

**No-retention ablation** (`retention=False`; C4 review item 1): z in R^4 (the pooled
logit is not used), offers = (R - 0.001) * softmax(z), so sum e = R - 0.001 exactly as
under the deployed rules (D33), and the planner cannot keep coins in the pool. The
default (`retention=True`) is unchanged, parameter for parameter.

**Critic** (training only): the same architecture with its own weights, given also the
round index t / ROUNDS (the planner itself is not told the horizon; the critic may use
privileged information, it is never used to act).
"""

from __future__ import annotations

import io
import math
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phasespace.citizens.game import MAX_POOL, N_PLAYERS
from phasespace.citizens.mechanisms import RECORDED_WITHHELD
from phasespace.citizens.rl import twin
from phasespace.citizens.rl.citizens import MechObs
from phasespace.citizens.rl.config import ROUNDS

N_SHARES = N_PLAYERS + 1
OBS_SCALE = MAX_POOL
LOG_2PI = math.log(2.0 * math.pi)


def player_features(obs: MechObs) -> Array:
    """(G, 4, 3): previous offer, previous reciprocation, pool, all / 200."""
    pool = jnp.broadcast_to(obs.pool[:, None], obs.prev_offers.shape)
    return jnp.stack([obs.prev_offers, obs.prev_reciprocations, pool], axis=-1) / OBS_SCALE


class SetGRU(eqx.Module):
    """One game: (4, F) player features and (4, H) hidden -> (4, H) hidden, 4 player
    outputs and 1 pooled output."""

    encoder: eqx.nn.Linear
    mix_self: eqx.nn.Linear
    mix_mean: eqx.nn.Linear
    cell: eqx.nn.GRUCell
    out_self: eqx.nn.Linear
    out_mean: eqx.nn.Linear
    player_head: eqx.nn.Linear
    pooled_hidden: eqx.nn.Linear
    pooled_head: eqx.nn.Linear
    hidden: int = eqx.field(static=True)

    def __init__(self, n_features: int, hidden: int, key: Array) -> None:
        k = jax.random.split(key, 9)
        self.hidden = hidden
        self.encoder = eqx.nn.Linear(n_features, hidden, key=k[0])
        self.mix_self = eqx.nn.Linear(hidden, hidden, key=k[1])
        self.mix_mean = eqx.nn.Linear(hidden, hidden, use_bias=False, key=k[2])
        self.cell = eqx.nn.GRUCell(hidden, hidden, key=k[3])
        self.out_self = eqx.nn.Linear(hidden, hidden, key=k[4])
        self.out_mean = eqx.nn.Linear(hidden, hidden, use_bias=False, key=k[5])
        self.player_head = eqx.nn.Linear(hidden, 1, key=k[6])
        self.pooled_hidden = eqx.nn.Linear(hidden, hidden, key=k[7])
        self.pooled_head = eqx.nn.Linear(hidden, 1, key=k[8])

    def __call__(self, h: Array, x: Array) -> tuple[Array, Array, Array]:
        u = jnp.tanh(jax.vmap(self.encoder)(x))
        v = jnp.tanh(jax.vmap(self.mix_self)(u) + self.mix_mean(u.mean(axis=0)))
        h_new = jax.vmap(self.cell)(v, h)
        y = jnp.tanh(jax.vmap(self.out_self)(h_new) + self.out_mean(h_new.mean(axis=0)))
        players = jax.vmap(self.player_head)(y)[:, 0]
        pooled = self.pooled_head(jnp.tanh(self.pooled_hidden(h_new.mean(axis=0))))[0]
        return h_new, players, pooled


class Planner(eqx.Module):
    """Actor (`policy`), critic (`value`) and the action log standard deviation."""

    policy: SetGRU
    value: SetGRU
    log_std: Array  # (5,), or (4,) without retention
    retention: bool = eqx.field(static=True)

    def __init__(
        self, hidden: int, init_log_std: float, key: Array, retention: bool = True
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.policy = SetGRU(3, hidden, k1)
        self.value = SetGRU(4, hidden, k2)
        self.retention = retention
        self.log_std = twin.full((N_SHARES if retention else N_PLAYERS,), init_log_std)

    @property
    def hidden(self) -> int:
        return self.policy.hidden

    def initial_state(self, n_games: int) -> tuple[Array, Array]:
        zeros = twin.zeros((n_games, N_PLAYERS, self.hidden))
        return zeros, zeros

    def policy_step(self, h: Array, x: Array) -> tuple[Array, Array]:
        """(G, 4, H), (G, 4, 3) -> new hidden, mean logits mu (G, 5) or, without
        retention, (G, 4)."""
        h_new, players, pooled = jax.vmap(self.policy)(h, x)
        if not self.retention:
            return h_new, players
        return h_new, jnp.concatenate([players, pooled[:, None]], axis=1)

    def offers(self, pool: Array, z: Array) -> Array:
        """(G, 4) offers from logits z: R softmax(z)[:4], or (R - 0.001) softmax(z)."""
        if self.retention:
            return pool[:, None] * shares(z)[:, :N_PLAYERS]
        return (pool - RECORDED_WITHHELD)[:, None] * shares(z)

    def full_shares(self, pool: Array, z: Array) -> Array:
        """(G, 5) shares of the pool (4 players, then the amount kept) of logits z."""
        if self.retention:
            return shares(z)
        players = self.offers(pool, z) / pool[:, None]
        return jnp.concatenate([players, (RECORDED_WITHHELD / pool)[:, None]], axis=1)

    def value_step(self, h: Array, x: Array, t: Array) -> tuple[Array, Array]:
        """Critic: (G, 4, H), (G, 4, 3), round index -> new hidden, value (G,)."""
        tt = jnp.broadcast_to(twin.asarray(t, dtype=x.dtype) / ROUNDS, x.shape[:-1])[..., None]
        h_new, _, pooled = jax.vmap(self.value)(h, jnp.concatenate([x, tt], axis=-1))
        return h_new, pooled


def gaussian_log_prob(z: Array, mu: Array, log_std: Array) -> Array:
    """log N(z; mu, diag(exp(2 log_std))) summed over the last axis."""
    return (-0.5 * ((z - mu) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * LOG_2PI).sum(axis=-1)


def gaussian_entropy(log_std: Array) -> Array:
    return (log_std + 0.5 * (1.0 + LOG_2PI)).sum()


def shares(z: Array) -> Array:
    """softmax over the 5 logits (4 players, then the pool)."""
    return jax.nn.softmax(z, axis=-1)


class PlannerExtras(NamedTuple):
    """Per-round training record of a sampled rollout."""

    features: Array  # (G, 4, 3)
    z: Array  # (G, 5)
    log_prob: Array  # (G,)
    value: Array  # (G,)


def sampling_step(planner: Planner) -> Any:
    """MechStep that samples z (training): extras are `PlannerExtras`."""

    def step(state: tuple[Array, Array], obs: MechObs, key: Array) -> tuple[Array, Any, Any]:
        h_pi, h_v = state
        x = player_features(obs)
        h_pi, mu = planner.policy_step(h_pi, x)
        h_v, v = planner.value_step(h_v, x, obs.t)
        z = mu + jnp.exp(planner.log_std) * jax.random.normal(key, mu.shape, dtype=mu.dtype)
        logp = gaussian_log_prob(z, mu, planner.log_std)
        return planner.offers(obs.pool, z), (h_pi, h_v), PlannerExtras(x, z, logp, v)

    return step


def deterministic_step(planner: Planner) -> Any:
    """MechStep of the evaluated planner: shares = softmax(mu); extras are the (G, 5)
    shares of the pool (4 players, then the amount kept)."""

    def step(state: tuple[Array, Array], obs: MechObs, key: Array) -> tuple[Array, Any, Any]:
        del key
        h_pi, h_v = state
        h_pi, mu = planner.policy_step(h_pi, player_features(obs))
        return planner.offers(obs.pool, mu), (h_pi, h_v), planner.full_shares(obs.pool, mu)

    return step


def serialise(planner: Planner) -> bytes:
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, planner)  # pyright: ignore[reportUnknownMemberType]
    return buffer.getvalue()


def deserialise(payload: bytes, hidden: int, retention: bool = True) -> Planner:
    skeleton = Planner(hidden, 0.0, jax.random.key(0), retention=retention)
    loaded = eqx.tree_deserialise_leaves(io.BytesIO(payload), skeleton)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast(Planner, loaded)
