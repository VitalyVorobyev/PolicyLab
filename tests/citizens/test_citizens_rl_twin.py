"""JAX game twin against the NumPy reference (`game`, `mechanisms`), D33 rules included.

Independent reference: the C1-audited NumPy game and mechanisms, which reproduce the
recorded pools and offers. Random trajectories are played by both implementations from
the same pre-drawn uniforms; pools must agree to 1e-9 coins and deployed offers exactly.
"""

from __future__ import annotations

from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasespace.citizens import game, mechanisms
from phasespace.citizens.models.base import n_max
from phasespace.citizens.rl import twin

POOL_ATOL = 1e-9


@pytest.fixture(autouse=True)
def x64_mode() -> Iterator[None]:
    with twin.x64():
        yield


def test_requires_x64() -> None:
    with jax.enable_x64(False), pytest.raises(RuntimeError):  # pyright: ignore[reportUnknownMemberType]
        twin.f64([1.0])


def test_next_pool_matches_reference_including_cap_and_floor() -> None:
    rng = np.random.default_rng(1)
    pool = rng.uniform(0.01, 200.0, 20_000)
    offers = pool[:, None] * rng.dirichlet(np.ones(5), 20_000)[:, :4]
    frac = rng.choice([0.0, 0.3, 1.0], size=offers.shape)
    recip = np.floor(frac * offers)
    ref = game.next_pool(pool, offers, recip)
    got = np.asarray(twin.next_pool(twin.f64(pool), twin.f64(offers), twin.f64(recip)))
    assert np.abs(got - ref).max() <= POOL_ATOL
    assert (ref == game.MAX_POOL).any() and (ref == game.POOL_FLOOR).any()  # both clamps hit


def test_max_reciprocation_forces_zero_below_one() -> None:
    e = np.array([0.0, 0.5, 0.999999, 1.0, 1.5, 7.0, 199.99])
    got = np.asarray(twin.max_reciprocation(twin.f64(e)))
    np.testing.assert_array_equal(got, n_max(e).astype(np.float64))


def test_interpolating_weight_matches_reference() -> None:
    pool = np.random.default_rng(2).uniform(0.01, 200.0, 50_000)
    for k in (22.0, float(np.exp(-5.0)), float(np.exp(1.3)), float(np.exp(5.0))):
        ref = mechanisms.interpolating_weight(pool, k)
        got = np.asarray(twin.interpolating_weight(twin.f64(pool), k))
        assert np.array_equal(got, ref), k


@pytest.mark.parametrize("w", [1.0, 0.5, 0.0, 0.3])
def test_deployed_offers_equal_reference_exactly(w: float) -> None:
    rng = np.random.default_rng(3)
    pool = rng.uniform(0.01, 200.0, 20_000)
    c = np.floor(rng.uniform(0.0, 60.0, (20_000, 4)) * (rng.random((20_000, 4)) > 0.2))
    c[:50] = 0.0  # sum(c) = 0: equal split
    ws = np.full(pool.shape, w)
    for first in (False, True):
        ref = mechanisms.deployed_offers(pool, None if first else c, ws)
        got = twin.deployed_offers(twin.f64(pool), twin.f64(c), twin.f64(ws), twin.asarray(first))
        np.testing.assert_array_equal(np.asarray(got), ref)


def test_paper_offers_match_reference() -> None:
    rng = np.random.default_rng(4)
    pool = rng.uniform(0.01, 200.0, 5000)
    c = np.floor(rng.uniform(0.0, 60.0, (5000, 4)))
    w = rng.random(5000)
    ref = mechanisms.paper_offers(pool, c, w)
    got = twin.paper_offers(twin.f64(pool), twin.f64(c), twin.f64(w), twin.asarray(False))
    np.testing.assert_allclose(np.asarray(got), ref, rtol=0.0, atol=1e-12)


def _numpy_game(
    rule: str, u: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], ...]:
    """Reference trajectory: deployed-rule offers, c = floor(u (floor(e) + 1)) <= floor(e)."""
    g, t_max, _ = u.shape
    pool = np.full(g, game.MAX_POOL)
    prev: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None
    pools, offers = np.zeros((g, t_max)), np.zeros((g, t_max, 4))
    for t in range(t_max):
        w = mechanisms.interpolating_weight(pool) if rule == "interp" else np.full(g, 0.5)
        e = mechanisms.deployed_offers(pool, prev, w)
        c = np.floor(u[:, t] * (n_max(e) + 1.0))
        pools[:, t], offers[:, t] = pool, e
        pool = game.next_pool(pool, e, c)
        prev = c
    return pools, offers


def _jax_game(
    rule: str, u: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
) -> tuple[jax.Array, jax.Array]:
    uu = twin.f64(np.swapaxes(u, 0, 1))

    def body(
        carry: tuple[jax.Array, jax.Array], inp: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        pool, prev = carry
        t, ut = inp
        w = twin.interpolating_weight(pool) if rule == "interp" else twin.full_like(pool, 0.5)
        e = twin.deployed_offers(pool, prev, w, t == 0)
        c = jnp.floor(ut * (twin.max_reciprocation(e) + 1.0))
        return (twin.next_pool(pool, e, c), c), (pool, e)

    g = u.shape[0]
    init = (
        twin.full((g,), game.MAX_POOL, dtype=jnp.float64),
        twin.zeros((g, 4), dtype=jnp.float64),
    )
    _, (pools, offers) = jax.lax.scan(eqx.filter_jit(body), init, (twin.arange(u.shape[1]), uu))
    return jnp.swapaxes(pools, 0, 1), jnp.swapaxes(offers, 0, 1)


@pytest.mark.parametrize("rule", ["interp", "mixed"])
def test_random_trajectories_match_reference(rule: str) -> None:
    """2,000 random 40-round games: the same pre-drawn uniforms drive both twins."""
    rng = np.random.default_rng(5)
    # Low reciprocation in some games so that pools deplete and hit the 0.01 floor.
    scale = rng.choice([0.3, 1.0, 1.3], size=(2000, 1, 1))
    u = np.minimum(rng.random((2000, 40, 4)) * scale, 0.999999)
    ref_pool, ref_offers = _numpy_game(rule, u)
    pool, offers = _jax_game(rule, u)
    assert np.abs(np.asarray(pool) - ref_pool).max() <= POOL_ATOL
    np.testing.assert_array_equal(np.asarray(offers), ref_offers)
    assert (ref_pool < game.DEPLETION_THRESHOLD).any()
    np.testing.assert_array_equal(np.asarray(twin.is_depleted(pool)), game.is_depleted(ref_pool))


def test_share_offers_are_feasible_steps() -> None:
    """R * shares[:4] passes the reference `game.step` validity checks."""
    rng = np.random.default_rng(6)
    pool = rng.uniform(0.01, 200.0, 2000)
    shares = rng.dirichlet(np.full(5, 0.3), 2000)
    offers = np.asarray(twin.offers_from_shares(twin.f64(pool), twin.f64(shares)))
    for r, e in zip(pool, offers, strict=True):
        state = game.GameState(pool=float(r))
        game.step(state, e, np.floor(0.5 * n_max(e)), integer=True)
