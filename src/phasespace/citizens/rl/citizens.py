"""JAX ports of the fitted citizen models B1, B2, B3, B23 and the vectorised game (pure).

Each port reproduces the NumPy reference model of `phasespace.citizens.models` step for
step: the same observation (own-first offers, own-first previous reciprocations, pool,
round index), the same recurrent state, and the same integer pmf over c in 0..200 (the
protocol 4 bin-to-integer map for B1, B3 and B23; B2's hurdle/spike/beta-binomial
likelihood). `tests/citizens/test_citizens_rl_citizens.py` checks the pmfs against the
reference on shared histories (to 1e-6) and the closed-loop metric means against
`phasespace.citizens.simulate` (within Monte Carlo error).

**Closed loop (`rollout`).** One citizen stream per (game, player), stream b = 4 g + i as
in the reference simulator. B1 draws one ensemble member per stream uniformly with
replacement at game start (the reference draws `rng.integers(0, 4)` per stream); B2, B3
and B23 use their fitted parameters. A reciprocation is drawn by inverting the pmf's
cumulative sum at a uniform (`models.base.sample_pmf`). The citizens' randomness (member
draws and one uniform per round and player) comes from `citizen_key` only, so two
mechanisms evaluated with the same key face common random numbers. Nothing here reads
human data: the parameters come from the frozen C2 fit run (`load_citizens`).

Requires JAX's 64-bit mode (`twin.x64()`), as the reference models run in float64.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jax import Array

from phasespace.citizens import config
from phasespace.citizens.fit import load_fitted
from phasespace.citizens.game import EXCLUSION_THRESHOLD, MAX_POOL, N_PLAYERS
from phasespace.citizens.models import b2_condcoop as b2
from phasespace.citizens.models import b3_ewa as b3
from phasespace.citizens.models.b1_gru import WEIGHT_NAMES, B1GRUEnsemble
from phasespace.citizens.models.base import K_MAX, N_BINS, Binning
from phasespace.citizens.rl import twin

TRAINABLE_MODELS: Final = ("B1", "B2", "B3", "B23")
"""The citizen models planners are trained against (D35: all four, each with its own
validation label; only B1 is closed-loop validated)."""
LEVELS: Final = tuple(j / 10.0 for j in range(config.N_LEVELS))


class Obs(NamedTuple):
    """What every citizen stream sees in one round (own first), as `models.base.Observation`."""

    offers: Array  # (B, 4)
    prev_reciprocations: Array  # (B, 4), zeros in round 0
    pool: Array  # (B,)
    t: Array  # () int, 0-based round index


class JaxCitizen(Protocol):
    """A vectorised citizen model: `init` (per-stream latent draws) and `step` (pmf)."""

    @property
    def name(self) -> str: ...

    def init(self, key: Array, n_streams: int) -> Any: ...

    def step(self, state: Any, obs: Obs) -> tuple[Array, Any]:
        """(B, 201) integer pmf of this round's reciprocation and the new state."""
        ...


# --- the protocol 4 bin-to-integer map ------------------------------------------------------


def bins_to_integer_pmf(p: Array, offers: Array, binning: Binning) -> Array:
    """(B, n_bins) bin probabilities and (B,) offers -> (B, 201) integer pmf (protocol 4).

    Port of `models.base.bins_to_integer_pmf`: q = (10 c) / e as one division, floor bins
    min(9, floor(q)) or levels floor(q + 1/2); each bin's mass is shared by its integers
    and empty bins are renormalised away; point mass at 0 when e < 1.
    """
    nb = N_BINS[binning]
    forced = offers < EXCLUSION_THRESHOLD
    n = twin.max_reciprocation(offers)
    c = twin.arange(K_MAX, dtype=jnp.float64)[None, :]
    valid = (c <= n[:, None]) & ~forced[:, None]
    q = twin.divide(10.0 * c, jnp.where(forced, 1.0, offers)[:, None])
    raw = (
        jnp.minimum(9.0, jnp.floor(q))
        if binning == "floor10"
        else jnp.clip(jnp.floor(q + 0.5), 0.0, 10.0)
    )
    bins = jnp.where(valid, raw.astype(jnp.int32), 0)
    rows = twin.arange(offers.shape[0])[:, None]
    counts = (
        twin.zeros((offers.shape[0], nb), dtype=jnp.float64)
        .at[rows, bins]
        .add(valid.astype(jnp.float64))
    )
    z = jnp.where(counts > 0.0, p, 0.0).sum(axis=1)
    per_integer = jnp.take_along_axis(p, bins, axis=1) / jnp.maximum(
        jnp.take_along_axis(counts, bins, axis=1), 1.0
    )
    pmf = jnp.where(valid, per_integer / jnp.where(z > 0.0, z, 1.0)[:, None], 0.0)
    return jnp.where(forced[:, None], _point_mass_at_zero(offers.shape[0]), pmf)


def _point_mass_at_zero(n: int) -> Array:
    return twin.zeros((n, K_MAX), dtype=jnp.float64).at[:, 0].set(1.0)


def sample_pmf(pmf: Array, u: Array) -> Array:
    """One integer draw per row by inverting the cumulative pmf at u * total (u in [0, 1)).

    Port of `models.base.sample_pmf` (same rule, including the clip to the support's top).
    """
    cdf = jnp.cumsum(pmf, axis=1)
    target = u * cdf[:, -1]
    draw = (cdf < target[:, None]).sum(axis=1)
    support_top = K_MAX - 1 - jnp.argmax(pmf[:, ::-1] > 0.0, axis=1)
    return jnp.minimum(draw, support_top).astype(jnp.float64)


def _softmax(logits: Array) -> Array:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = jnp.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: Array) -> Array:
    return 0.5 * (1.0 + jnp.tanh(0.5 * x))


# --- B1: GRU clone ensemble -------------------------------------------------------------------


class B1State(NamedTuple):
    hidden: Array  # (B, 64)
    member: Array  # (B,) int


def _linear(w: Mapping[str, Array], name: str, x: Array) -> Array:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def b1_forward(w: Mapping[str, Array], hidden: Array, x: Array) -> tuple[Array, Array]:
    """Port of `b1_gru.forward_step`: (h (B, 64), x (B, 9)) -> (h', logits (B, 10))."""
    z = jnp.tanh(_linear(w, "enc2", jnp.tanh(_linear(w, "enc1", x))))
    h = config.B1_GRU
    ig = z @ w["cell.weight_ih"].T + w["cell.bias"]
    hg = hidden @ w["cell.weight_hh"].T
    reset = _sigmoid(ig[:, :h] + hg[:, :h])
    update = _sigmoid(ig[:, h : 2 * h] + hg[:, h : 2 * h])
    new = jnp.tanh(ig[:, 2 * h :] + reset * (hg[:, 2 * h :] + w["cell.bias_n"]))
    h_next = new + update * (hidden - new)
    y = jnp.tanh(_linear(w, "proj2", jnp.tanh(_linear(w, "proj1", h_next))))
    return h_next, _linear(w, "head", y)


class B1Port(eqx.Module):
    """B1 closed-loop form: one member per stream, drawn uniformly at game start."""

    weights: dict[str, Array]  # each (M, ...) stacked over members
    name: str = eqx.field(static=True, default="B1")

    @property
    def n_members(self) -> int:
        return int(self.weights["head.bias"].shape[0])

    @classmethod
    def from_reference(cls, model: B1GRUEnsemble) -> B1Port:
        stacked = {
            name: twin.f64(np.stack([m[name] for m in model.members])) for name in WEIGHT_NAMES
        }
        return cls(weights=stacked)

    def init(self, key: Array, n_streams: int) -> B1State:
        member = jax.random.randint(key, (n_streams,), 0, self.n_members)
        return B1State(
            hidden=twin.zeros((n_streams, config.B1_GRU), dtype=jnp.float64), member=member
        )

    def step(self, state: B1State, obs: Obs) -> tuple[Array, B1State]:
        scale = config.B1_INPUT_SCALE
        x = jnp.concatenate(
            [obs.offers / scale, obs.prev_reciprocations / scale, obs.pool[:, None] / scale], axis=1
        )
        hidden_all, logits_all = jax.vmap(b1_forward, in_axes=(0, None, None))(
            self.weights, state.hidden, x
        )
        rows = twin.arange(x.shape[0])
        hidden = hidden_all[state.member, rows]
        probs = _softmax(logits_all[state.member, rows])
        return bins_to_integer_pmf(probs, obs.offers[:, 0], "floor10"), B1State(
            hidden, state.member
        )


def b1_member_disagreement(port: B1Port, traj: Trajectory) -> Array:
    """(G, T, 4) mean pairwise total-variation distance between the B1 members' pmfs.

    Every member is run, teacher-forced, on the same recorded history (the trajectory's
    offers, pools and actions), so the members differ only in their weights; TV(p, q) =
    0.5 sum_c |p(c) - q(c)|, averaged over the M(M - 1)/2 member pairs. NaN for forced
    decisions (e < 1). A diagnostic of extrapolation: the members agree where the fit
    data constrain them and can disagree off their training distribution.
    """
    g, t_max, _ = traj.offers.shape
    b = g * N_PLAYERS
    m = port.n_members
    scale = config.B1_INPUT_SCALE
    offers = jnp.swapaxes(traj.offers, 0, 1)
    actions = jnp.swapaxes(traj.actions, 0, 1)
    prev = jnp.concatenate([twin.zeros((1, g, N_PLAYERS)), actions[:-1]], axis=0)
    pools = jnp.swapaxes(traj.pool, 0, 1)

    def body(hidden: Array, inp: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        e, c_prev, pool = inp
        e_rot = twin.rotate(e).reshape(b, N_PLAYERS)
        c_rot = twin.rotate(c_prev).reshape(b, N_PLAYERS)
        x = jnp.concatenate(
            [e_rot / scale, c_rot / scale, jnp.repeat(pool, N_PLAYERS)[:, None] / scale], axis=1
        )
        new_hidden, logits = jax.vmap(b1_forward, in_axes=(0, 0, None))(port.weights, hidden, x)
        own = e_rot[:, 0]

        def member_pmf(probs: Array) -> Array:
            return bins_to_integer_pmf(probs, own, "floor10")

        pmfs = jax.vmap(member_pmf)(_softmax(logits))
        tv = 0.5 * jnp.abs(pmfs[:, None] - pmfs[None, :]).sum(axis=-1)  # (M, M, B)
        mean_tv = tv.sum(axis=(0, 1)) / (m * (m - 1))
        return new_hidden, jnp.where(own >= EXCLUSION_THRESHOLD, mean_tv, jnp.nan)

    h0 = twin.zeros((m, b, config.B1_GRU))
    _, tv = jax.lax.scan(body, h0, (offers, prev, pools))
    return jnp.swapaxes(tv.reshape(t_max, g, N_PLAYERS), 0, 1)


# --- B2: conditional cooperation ------------------------------------------------------------


class PrevOffersState(NamedTuple):
    prev_offers: Array  # (B, 4) own first; zeros before round 0 (unused there)


def b2_features(
    own_offer: Array, pool: Array, t: Array, prev_offers: Array, prev_c: Array
) -> Array:
    """Port of `b2_condcoop.features`: (B, 6) x1..x6 (x5 = 1[t = 0], others 0 there)."""
    x1 = jnp.minimum(config.B2_MAX_X1, 4.0 * own_offer / pool)
    x4 = pool / MAX_POOL
    own_prev = prev_offers[:, 0]
    own_ok = own_prev >= EXCLUSION_THRESHOLD
    x2 = jnp.where(own_ok, prev_c[:, 0] / jnp.where(own_ok, own_prev, 1.0), 0.0)
    others_e = prev_offers[:, 1:].sum(axis=1)
    others_ok = others_e >= EXCLUSION_THRESHOLD
    x3 = jnp.where(others_ok, prev_c[:, 1:].sum(axis=1) / jnp.where(others_ok, others_e, 1.0), 0.0)
    x6 = (~own_ok).astype(jnp.float64)
    first = t == 0
    zero = twin.zeros_like(x1)
    return jnp.stack(
        [
            x1,
            jnp.where(first, zero, x2),
            jnp.where(first, zero, x3),
            x4,
            jnp.where(first, 1.0, zero),
            jnp.where(first, zero, x6),
        ],
        axis=1,
    )


def _log_sigmoid(z: Array) -> Array:
    return -jnp.logaddexp(0.0, -z)


def _betaln(x: Array, y: Array) -> Array:
    """log B(x, y) from log-gamma values. `jax.scipy.special.betaln` (jax 0.12) differs
    from SciPy's by up to ~7e-7 here; the log-gamma form agrees to ~1e-12 for the
    arguments of B2 (x, y <= 200 + phi)."""
    return jsp.gammaln(x) + jsp.gammaln(y) - jsp.gammaln(x + y)


def _log_beta_binomial(k: Array, n: Array, a: Array, b: Array) -> Array:
    return (
        jsp.gammaln(n + 1.0)
        - jsp.gammaln(k + 1.0)
        - jsp.gammaln(n - k + 1.0)
        + _betaln(k + a, n - k + b)
        - _betaln(a, b)
    )


def b2_pmf(theta: Array, x: Array, own_offer: Array) -> Array:
    """Port of `b2_condcoop.pmf`: (B, 201) integer pmf (point mass at 0 for e < 1)."""
    z0 = theta[0] + theta[1] * x[:, 1] + theta[2] * x[:, 2]
    zm = theta[4] + x @ theta[5:11]
    phi = jnp.exp(theta[11])
    mu = 0.5 * (1.0 + jnp.tanh(0.5 * zm))
    log_pi0, log_1m_pi0 = _log_sigmoid(z0)[:, None], _log_sigmoid(-z0)[:, None]
    log_pif, log_1m_pif = _log_sigmoid(theta[3]), _log_sigmoid(-theta[3])
    a, b = (mu * phi)[:, None], ((1.0 - mu) * phi)[:, None]
    n_true = twin.max_reciprocation(own_offer)[:, None]
    n = jnp.maximum(n_true, 1.0)
    c = twin.arange(K_MAX, dtype=jnp.float64)[None, :]
    interior = (c >= 1.0) & (c <= n - 1.0)
    bb = _log_beta_binomial(
        jnp.where(interior, c - 1.0, 0.0), jnp.where(interior, n - 2.0, 0.0), a, b
    )
    lp = jnp.where(
        c == 0.0,
        log_pi0,
        jnp.where(
            n == 1.0,
            log_1m_pi0,
            jnp.where(c == n, log_1m_pi0 + log_pif, log_1m_pi0 + log_1m_pif + bb),
        ),
    )
    pmf = jnp.where(c <= n_true, jnp.exp(lp), 0.0)
    forced = own_offer < EXCLUSION_THRESHOLD
    return jnp.where(forced[:, None], _point_mass_at_zero(own_offer.shape[0]), pmf)


class B2Port(eqx.Module):
    theta: Array  # (12,)
    name: str = eqx.field(static=True, default="B2")

    def init(self, key: Array, n_streams: int) -> PrevOffersState:
        del key
        return PrevOffersState(twin.zeros((n_streams, N_PLAYERS), dtype=jnp.float64))

    def step(self, state: PrevOffersState, obs: Obs) -> tuple[Array, PrevOffersState]:
        x = b2_features(
            obs.offers[:, 0], obs.pool, obs.t, state.prev_offers, obs.prev_reciprocations
        )
        return b2_pmf(self.theta, x, obs.offers[:, 0]), PrevOffersState(obs.offers)


# --- B3 / B23: EWA -------------------------------------------------------------------------


class EWAState(NamedTuple):
    attractions: Array  # (B, 11)
    experience: Array  # (B,)
    prev_offers: Array  # (B, 4)


def _level_index(c: Array, e: Array) -> Array:
    """Port of `bin_index(c, e, "level11")` for e >= 1 (callers pass e = 1 otherwise)."""
    q = twin.divide(10.0 * c, jnp.where(e >= EXCLUSION_THRESHOLD, e, 1.0))
    return jnp.clip(jnp.floor(q + 0.5), 0.0, 10.0).astype(jnp.int32)


class EWAPort(eqx.Module):
    """B3 (`hybrid=False`) or B23 (`hybrid=True`), port of `b3_ewa.EWAModel`."""

    params: Array  # (6,) or (7,)
    p0: Array  # (11,)
    hybrid: bool = eqx.field(static=True)
    name: str = eqx.field(static=True)

    @classmethod
    def from_reference(cls, model: b3.EWAModel) -> EWAPort:
        return cls(
            params=twin.f64(model.params),
            p0=twin.f64(model.p0),
            hybrid=model.hybrid,
            name=model.name,
        )

    def init(self, key: Array, n_streams: int) -> EWAState:
        del key
        lam, n0 = self.params[3], self.params[4]
        a0 = jnp.log(self.p0) / lam
        return EWAState(
            attractions=jnp.broadcast_to(a0, (n_streams, config.N_LEVELS)),
            experience=twin.full((n_streams,), n0),
            prev_offers=twin.zeros((n_streams, N_PLAYERS), dtype=jnp.float64),
        )

    def step(self, state: EWAState, obs: Obs) -> tuple[Array, EWAState]:
        phi, rho, delta, lam, _, theta = (self.params[i] for i in range(6))
        kappa = self.params[6] if self.hybrid else twin.asarray(0.0, dtype=jnp.float64)
        levels = twin.asarray(LEVELS, dtype=jnp.float64)
        later = obs.t > 0
        a, n = state.attractions, state.experience
        prev_own = state.prev_offers[:, 0]
        active = later & (prev_own >= EXCLUSION_THRESHOLD)
        chosen = _level_index(obs.prev_reciprocations[:, 0], jnp.where(active, prev_own, 1.0))
        one_hot = jax.nn.one_hot(chosen, config.N_LEVELS, dtype=jnp.float64)
        scale = config.EWA_PAYOFF_SCALE
        n_new = rho * n + 1.0
        payoff = (delta + (1.0 - delta) * one_hot) * prev_own[:, None] * (1.0 - levels) / scale
        payoff = payoff + one_hot * theta * obs.offers[:, 0][:, None] / scale
        a_new = (phi * n[:, None] * a + payoff) / n_new[:, None]
        a = jnp.where(active[:, None], a_new, a)
        n = jnp.where(active, n_new, n)
        others_e = state.prev_offers[:, 1:].sum(axis=1)
        ok = others_e >= EXCLUSION_THRESHOLD
        x3 = jnp.where(
            ok, obs.prev_reciprocations[:, 1:].sum(axis=1) / jnp.where(ok, others_e, 1.0), 0.0
        )
        logits = lam * a
        if self.hybrid:
            pull = kappa * (levels[None, :] - x3[:, None]) ** 2
            logits = jnp.where(later, logits - pull, logits)
        pmf = bins_to_integer_pmf(_softmax(logits), obs.offers[:, 0], "level11")
        return pmf, EWAState(attractions=a, experience=n, prev_offers=obs.offers)


# --- analytic citizens (the C3 correctness checks; not fitted models) ---------------------------


class DeterministicCitizen(eqx.Module):
    """A citizen with a closed-form, continuous reaction c = react(e) (no integer rounding).

    Used only by the C3 correctness checks, whose optimal planner value is known.
    """

    def init(self, key: Array, n_streams: int) -> None:
        del key, n_streams

    def step(self, state: None, obs: Obs) -> tuple[Array, None]:
        raise NotImplementedError("a deterministic citizen acts through `react`")

    def react(self, own_offer: Array) -> Array:
        raise NotImplementedError


class FixedFraction(DeterministicCitizen):
    """Returns the fixed fraction f of every offer: c = f e."""

    fraction: float = eqx.field(static=True)
    name: str = eqx.field(static=True, default="fixed-fraction")

    def react(self, own_offer: Array) -> Array:
        return self.fraction * own_offer


class QuadraticToy(DeterministicCitizen):
    """Returns c = min(e, e^2 / (2 a)): own surplus e - e^2 / (2a) peaks at e = a.

    A toy with an interior optimum: with a <= 50 and one round from R = 200, the best
    planner offers a to every player and keeps 200 - 4a, for a total surplus of 2a.
    """

    a: float = eqx.field(static=True)
    name: str = eqx.field(static=True, default="quadratic-toy")

    def react(self, own_offer: Array) -> Array:
        return jnp.minimum(own_offer, own_offer**2 / (2.0 * self.a))


def act(citizen: JaxCitizen, state: Any, obs: Obs, u: Array) -> tuple[Array, Any]:
    """This round's reciprocations (B,) and the new state."""
    if isinstance(citizen, DeterministicCitizen):
        return citizen.react(obs.offers[:, 0]), state
    pmf, new_state = citizen.step(state, obs)
    return sample_pmf(pmf, u), new_state


# --- loading the frozen fit --------------------------------------------------------------------


def load_citizens(fit_run: Path) -> dict[str, JaxCitizen]:
    """JAX ports of B1, B2, B3 and B23 from a C2 fitted-model run (checksums verified).

    Reads only the run's artifacts (`fit.load_fitted`): no human data.
    """
    models = load_fitted(fit_run).sim_models()
    b1_model, b2_model = models["B1"], models["B2"]
    b3_model, b23_model = models["B3"], models["B23"]
    if not isinstance(b1_model, B1GRUEnsemble) or not isinstance(
        b2_model, b2.B2ConditionalCooperation
    ):
        raise TypeError("unexpected model classes in the fit run")
    if not isinstance(b3_model, b3.EWAModel) or not isinstance(b23_model, b3.EWAModel):
        raise TypeError("unexpected model classes in the fit run")
    return {
        "B1": B1Port.from_reference(b1_model),
        "B2": B2Port(theta=twin.f64(b2_model.theta)),
        "B3": EWAPort.from_reference(b3_model),
        "B23": EWAPort.from_reference(b23_model),
    }


# --- the vectorised closed loop -----------------------------------------------------------------


class MechObs(NamedTuple):
    """What a mechanism sees before offering in round t (G games)."""

    pool: Array  # (G,)
    prev_offers: Array  # (G, 4), zeros in round 0
    prev_reciprocations: Array  # (G, 4), zeros in round 0
    t: Array  # () int


type MechStep = Callable[[Any, MechObs, Array], tuple[Array, Any, Any]]
"""(state, observation, key) -> (offers (G, 4), new state, per-round extras)."""


class Trajectory(NamedTuple):
    offers: Array  # (G, T, 4)
    actions: Array  # (G, T, 4)
    pool: Array  # (G, T) before allocation


def citizen_draws(citizen_key: Array, n_games: int, n_rounds: int) -> tuple[Array, Array]:
    """(init key, uniforms (T, G * 4)): every random number the citizens use."""
    k_init, k_u = jax.random.split(citizen_key)
    return k_init, jax.random.uniform(k_u, (n_rounds, n_games * N_PLAYERS), dtype=jnp.float64)


def rollout(
    citizen: JaxCitizen,
    mech_step: MechStep,
    mech_state: Any,
    citizen_key: Array,
    mech_key: Array,
    n_games: int,
    n_rounds: int = config.SIM_ROUNDS,
) -> tuple[Trajectory, Any]:
    """G closed-loop games from R_0 = 200; returns the trajectory and the stacked extras."""
    twin.require_x64()
    k_init, uniforms = citizen_draws(citizen_key, n_games, n_rounds)
    b = n_games * N_PLAYERS
    c_state = citizen.init(k_init, b)
    mech_keys = jax.random.split(mech_key, n_rounds)
    zeros = twin.zeros((n_games, N_PLAYERS), dtype=jnp.float64)
    pool0 = twin.full((n_games,), MAX_POOL, dtype=jnp.float64)

    def body(
        carry: tuple[Any, ...], inputs: tuple[Array, Array, Array]
    ) -> tuple[tuple[Any, ...], Any]:
        pool, prev_e, prev_c, m_state, cz_state = carry
        t, u, key = inputs
        e, m_state, extras = mech_step(m_state, MechObs(pool, prev_e, prev_c, t), key)
        obs = Obs(
            offers=twin.rotate(e).reshape(b, N_PLAYERS),
            prev_reciprocations=twin.rotate(prev_c).reshape(b, N_PLAYERS),
            pool=jnp.repeat(pool, N_PLAYERS),
            t=t,
        )
        c, cz_state = act(citizen, cz_state, obs, u)
        c = c.reshape(n_games, N_PLAYERS)
        new_pool = twin.next_pool(pool, e, c)
        return (new_pool, e, c, m_state, cz_state), (e, c, pool, extras)

    ts = twin.arange(n_rounds)
    _, (e, c, pool, extras) = jax.lax.scan(
        body, (pool0, zeros, zeros, mech_state, c_state), (ts, uniforms, mech_keys)
    )
    traj = Trajectory(
        offers=jnp.swapaxes(e, 0, 1), actions=jnp.swapaxes(c, 0, 1), pool=jnp.swapaxes(pool, 0, 1)
    )
    return traj, extras


# --- baseline mechanisms as MechStep ------------------------------------------------------------


def weight_rule_step(weight: Callable[[Array], Array]) -> MechStep:
    """A deployed-rule baseline offering with weight w(R) (Equal, Mixed, Proportional, k)."""

    def step(state: Any, obs: MechObs, key: Array) -> tuple[Array, Any, Any]:
        del key
        e = twin.deployed_offers(obs.pool, obs.prev_reciprocations, weight(obs.pool), obs.t == 0)
        return e, state, None

    return step


def constant_weight(w: float) -> Callable[[Array], Array]:
    return lambda pool: twin.full_like(pool, w)


def interpolating(k: Array | float) -> Callable[[Array], Array]:
    return lambda pool: twin.interpolating_weight(pool, k)
