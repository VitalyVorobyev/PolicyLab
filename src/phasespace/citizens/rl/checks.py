"""Agreement of the JAX ports with the NumPy reference models (used by tests and runs).

- `pmf_difference`: max |P_jax(c) - P_ref(c)| over every stream, round and c on shared
  random histories (teacher forcing: both see the same recorded offers and actions).
- `closed_loop_z`: per metric, |mean_jax - mean_ref| / SE of the difference between S
  closed-loop games of the JAX rollout and S games of `phasespace.citizens.simulate`
  (independent random numbers, so agreement is within Monte Carlo error).

Synthetic histories only; no human data.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phasespace.citizens import config as c2config
from phasespace.citizens import simulate
from phasespace.citizens.models.b1_gru import B1GRUEnsemble, B1State
from phasespace.citizens.models.base import CitizenModel, Streams, streams_from_arrays
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import twin

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def random_histories(seed: int, n_games: int = 24, n_rounds: int = 40) -> Streams:
    """Random games: Dirichlet offers with exclusions (e < 1), integer offers and exact
    bin boundaries (e = 10), and integer actions 0..floor(e)."""
    rng = np.random.default_rng(seed)
    pool = rng.uniform(0.01, 200.0, (n_games, n_rounds))
    shares = rng.dirichlet(np.ones(5), (n_games, n_rounds))[..., :4]
    offers = pool[..., None] * shares
    pick = rng.random(offers.shape)
    offers = np.where(pick < 0.08, rng.uniform(0.0, 0.999, offers.shape), offers)
    offers = np.where((pick > 0.9) & (offers >= 1.0), np.floor(offers), offers)
    offers = np.where((pick > 0.97) & (offers >= 10.0), 10.0, offers)
    n = np.where(offers >= 1.0, np.floor(offers), 0.0)
    actions = np.floor(rng.random(offers.shape) * (n + 1.0))
    return streams_from_arrays(offers, actions, pool, np.full(n_games, n_rounds))


def pmf_difference(
    reference: CitizenModel[Any], port: rc.JaxCitizen, streams: Streams, seed: int = 0
) -> tuple[float, float]:
    """(max |pmf difference|, max |sum(pmf_jax) - 1|) over the histories.

    B1 runs in its closed-loop form with one member per stream drawn from `seed` (the
    same members for both); the other models have no latent draw.
    """
    b = streams.n_streams
    if isinstance(reference, B1GRUEnsemble):
        member = np.random.default_rng(seed).integers(0, len(reference.members), b)
        ref_state: Any = B1State(hidden=np.zeros((b, c2config.B1_GRU)), member=member)
        state: Any = rc.B1State(
            hidden=twin.zeros((b, c2config.B1_GRU), dtype=jnp.float64), member=twin.asarray(member)
        )
    else:
        ref_state = reference.initial_state(b, None)
        state = port.init(jax.random.key(seed), b)

    @eqx.filter_jit
    def step(s: Any, o: rc.Obs) -> tuple[jax.Array, Any]:
        return port.step(s, o)

    worst, mass = 0.0, 0.0
    for t in range(streams.n_rounds_max):
        obs = streams.observation(t)
        ref_pmf, ref_state = reference.step(ref_state, obs)
        jobs = rc.Obs(
            offers=twin.f64(obs.offers),
            prev_reciprocations=twin.f64(obs.prev_reciprocations),
            pool=twin.f64(obs.pool),
            t=twin.asarray(t),
        )
        pmf, state = step(state, jobs)
        p = np.asarray(pmf)
        worst = max(worst, float(np.abs(p - ref_pmf).max()))
        mass = max(mass, float(np.abs(p.sum(axis=1) - 1.0).max()))
    return worst, mass


REFERENCE_MECHANISMS = {
    "Equal": rc.weight_rule_step(rc.constant_weight(1.0)),
    "Mixed": rc.weight_rule_step(rc.constant_weight(0.5)),
    "Proportional": rc.weight_rule_step(rc.constant_weight(0.0)),
    "Interpolating": rc.weight_rule_step(rc.interpolating(22.0)),
}
"""JAX counterparts of `simulate.EXACT_MECHANISMS` (deployed offer rule)."""


def trajectory_metrics(traj: rc.Trajectory) -> dict[str, FloatArray]:
    """Per-game metrics (notebook definitions via `simulate.game_metrics`) plus the
    planner objective `total_surplus` = sum over rounds and players of e - c."""
    offers, actions = np.asarray(traj.offers), np.asarray(traj.actions)
    games = simulate.SimulatedGames(
        offers=offers,
        actions=actions,
        pool=np.asarray(traj.pool),
        n_rounds=np.full(offers.shape[0], offers.shape[1]),
    )
    out = games.game_metrics()
    out["total_surplus"] = (offers - actions).sum(axis=(1, 2))
    return out


def closed_loop_z(
    reference: CitizenModel[Any], port: rc.JaxCitizen, mechanism: str, n_games: int, seed: int
) -> dict[str, float]:
    """|z| per metric between the JAX rollout and the reference simulator."""
    ref = simulate.simulate(
        reference, simulate.EXACT_MECHANISMS[mechanism], n_games, np.random.default_rng(seed)
    ).game_metrics()
    mech = REFERENCE_MECHANISMS[mechanism]

    @eqx.filter_jit
    def run(key: jax.Array) -> rc.Trajectory:
        k1, k2 = jax.random.split(key)
        return rc.rollout(port, mech, None, k1, k2, n_games)[0]

    got = trajectory_metrics(run(jax.random.key(seed)))
    out: dict[str, float] = {}
    for metric in c2config.METRICS:
        a, b = got[metric], ref[metric]
        se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
        gap = abs(float(a.mean() - b.mean()))
        out[metric] = 0.0 if gap == 0.0 else (gap / se if se > 0.0 else float("inf"))
    return out
