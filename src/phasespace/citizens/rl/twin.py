"""JAX twin of the common-pool game and the baseline mechanisms (pure, jit-able).

Every function here is the JAX counterpart of a NumPy reference and must agree with it
(`tests/citizens/test_citizens_rl_twin.py`):

- `next_pool`, `surplus` -- `phasespace.citizens.game` (float64): R' = max(0.01,
  min(200, R - sum e + 1.4 sum c));
- `deployed_offers`, `paper_offers`, `interpolating_weight` --
  `phasespace.citizens.mechanisms` (the deployed rule withholds 0.001 coins and rounds
  float32 offers to 4 decimals, D33; the quantised Interpolating weight
  round((R/200)^k, 1) in float64);
- `max_reciprocation` -- `game.max_human_reciprocation` / `models.base.n_max`: integer
  actions 0..floor(e), forced to 0 when e < 1;
- `is_depleted` -- R < 1.

**XLA versus NumPy arithmetic.** XLA (CPU) rewrites a division by a broadcast or
constant divisor as a multiplication by the reciprocal and contracts a * b + c into a
fused multiply-add; NumPy does neither. Where a discrete outcome depends on the last
bit (floor bins, the float32 offer rounding) the twin forces NumPy's rounding
(`divide`, an optimization barrier before the add); elsewhere the difference is a few
ulps.

**Precision.** The game is float64, as in the reference: every function requires JAX's
64-bit mode (`x64()`; `require_x64` raises otherwise), so that pools and offers agree
with NumPy to rounding (about 1e-13 coins; the sum over 4 players may associate
differently). The deployed offers are float32 as deployed and agree exactly whenever
the previous reciprocations are integers (their float32 sum is then exact), which they
always are in play.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phasespace.citizens.game import (
    DEPLETION_THRESHOLD,
    EXCLUSION_THRESHOLD,
    GROWTH_FACTOR,
    MAX_POOL,
    N_PLAYERS,
    POOL_FLOOR,
)
from phasespace.citizens.mechanisms import (
    INTERPOLATING_DECIMALS,
    K_INTERPOLATING,
    RECORDED_DECIMALS,
    RECORDED_WITHHELD,
)

ROTATION: Final = tuple(
    tuple((i + j) % N_PLAYERS for j in range(N_PLAYERS)) for i in range(N_PLAYERS)
)
"""ROTATION[i] = (i, i+1, i+2, i+3) mod 4, as `models.base._ROTATION`."""


# --- typed array constructors (jax.numpy's are only partly typed in jax 0.12) -------------


def zeros(shape: int | tuple[int, ...], dtype: Any = jnp.float64) -> Array:
    return jnp.zeros(shape, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]


def zeros_like(x: Array) -> Array:
    return jnp.zeros_like(x)  # pyright: ignore[reportUnknownMemberType]


def full(shape: int | tuple[int, ...], value: Array | float, dtype: Any = jnp.float64) -> Array:
    return jnp.full(shape, value, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]


def full_like(x: Array, value: Array | float) -> Array:
    return jnp.full_like(x, value)  # pyright: ignore[reportUnknownMemberType]


def arange(n: int, dtype: Any = None) -> Array:
    return jnp.arange(n, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]


def asarray(x: object, dtype: Any = None) -> Array:
    return jnp.asarray(x, dtype=dtype)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]


def barrier(x: Array) -> Array:
    """`jax.lax.optimization_barrier` (stops XLA fusing across it)."""
    out: Array = jax.lax.optimization_barrier(x)  # pyright: ignore[reportUnknownMemberType]
    return out


_WEIGHT_TABLE: Final = tuple(
    float(v) for v in np.arange(10**INTERPOLATING_DECIMALS + 1) / 10.0**INTERPOLATING_DECIMALS
)
"""The quantised weights j / 10, j = 0..10, each a correctly rounded division."""


def x64() -> AbstractContextManager[None]:
    """Context in which JAX defaults to 64-bit types (the twin needs it)."""
    return jax.enable_x64(True)  # pyright: ignore[reportUnknownVariableType, reportReturnType]


def require_x64() -> None:
    """Raise unless JAX's 64-bit mode is on (float64 game arithmetic)."""
    if not jax.config.read("jax_enable_x64"):  # pyright: ignore[reportUnknownMemberType]
        raise RuntimeError("the JAX twin needs 64-bit mode: run inside `twin.x64()`")


def f64(x: object) -> Array:
    """An array as float64 (requires 64-bit mode)."""
    require_x64()
    return jnp.asarray(x, dtype=jnp.float64)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]


def divide(a: Array, b: Array | float) -> Array:
    """a / b correctly rounded, element by element (finite a).

    XLA's CPU simplifier turns a division by a broadcast or constant divisor into a
    multiplication by the divisor's reciprocal, which is not correctly rounded (NumPy's
    division is): it moved floor bins at exact boundaries and float32 offers by 1e-4.
    Adding 0 * a makes the divisor a full elementwise array, which keeps the division.
    `tests/citizens/test_citizens_rl_twin.py` checks the twin bit for bit against NumPy.
    """
    return a / (b + 0.0 * a)


def rotate(x: Array) -> Array:
    """(..., 4) -> (..., 4, 4): out[..., i, :] = roll(x, -i), player i's own-first view."""
    return x[..., asarray(ROTATION)]


def next_pool(
    pool: Array, offers: Array, reciprocations: Array, floor: float = POOL_FLOOR
) -> Array:
    """R' = max(floor, min(200, R - sum_i e_i + 1.4 sum_i c_i)); players on the last axis."""
    raw = pool - offers.sum(axis=-1) + GROWTH_FACTOR * reciprocations.sum(axis=-1)
    return jnp.maximum(floor, jnp.minimum(MAX_POOL, raw))


def surplus(offers: Array, reciprocations: Array) -> Array:
    """Per-player surplus e_i - c_i."""
    return offers - reciprocations


def is_depleted(pool: Array) -> Array:
    return pool < DEPLETION_THRESHOLD


def max_reciprocation(offers: Array) -> Array:
    """floor(e) as float64, 0 for an exclusion (e < 1): the largest integer action."""
    return jnp.where(offers >= EXCLUSION_THRESHOLD, jnp.floor(offers), 0.0)


def interpolating_weight(pool: Array, k: Array | float = K_INTERPOLATING) -> Array:
    """w = round((R / 200)^k, 1) (the deployed, quantised weight), float64.

    The quantised value is looked up in the table j / 10 computed by NumPy (equal to
    NumPy's rint(10 x) / 10).
    """
    scale = 10.0**INTERPOLATING_DECIMALS
    index = jnp.rint(divide(pool, MAX_POOL) ** k * scale).astype(jnp.int32)
    return asarray(_WEIGHT_TABLE, dtype=jnp.float64)[index]


def paper_offers(pool: Array, prev_reciprocations: Array, w: Array, first: Array) -> Array:
    """Idealised baseline offers R * (w/4 + (1-w) c_i / sum c); equal split if `first` or
    sum c = 0 (float64). `pool` (G,), `prev_reciprocations` (G, 4), `w` (G,), `first` ()."""
    total = prev_reciprocations.sum(axis=-1, keepdims=True)
    positive = total > 0.0
    proportional = jnp.where(
        positive, divide(prev_reciprocations, jnp.where(positive, total, 1.0)), 0.25
    )
    wk = w[..., None]
    share = jnp.where(first, 1.0 / N_PLAYERS, wk / N_PLAYERS + (1.0 - wk) * proportional)
    return pool[..., None] * share


def deployed_offers(pool: Array, prev_reciprocations: Array, w: Array, first: Array) -> Array:
    """Deployed baseline offers round(share_i * (R - 0.001), 4) in float32, as float64.

    The float32 port of `mechanisms.deployed_offers` (same operations in the same order);
    `first` (round 1) gives the equal split, as the reference does for
    ``prev_reciprocations=None``.
    """
    f32 = jnp.float32
    weight = w.astype(f32)
    c = prev_reciprocations.astype(f32)
    total = c.sum(axis=-1, keepdims=True)
    positive = total > 0
    quarter = asarray(1.0 / N_PLAYERS, dtype=f32)
    proportional = jnp.where(positive, divide(c, jnp.where(positive, total, f32(1.0))), quarter)
    wk = weight[..., None]
    # The barrier stops XLA from contracting the product and the sum into one fused
    # multiply-add (NumPy rounds the product first), which moved offers by 1e-4.
    share = divide(wk, f32(N_PLAYERS)) + barrier((f32(1.0) - wk) * proportional)
    share = jnp.where(first, quarter, share)
    distributed = (pool.astype(f32) - f32(RECORDED_WITHHELD))[..., None]
    return round_float32(share * distributed, RECORDED_DECIMALS)


def round_float32(x: Array, decimals: int) -> Array:
    """NumPy's float32 round(x, d) = rint(x * 10^d) / 10^d in float32, returned as float64."""
    scale = jnp.float32(10.0**decimals)
    return divide(jnp.rint(x * scale), scale).astype(jnp.float64)


def offers_from_shares(pool: Array, shares: Array) -> Array:
    """(G, 4) offers R * s_i from (G, 5) shares (4 players, then the pool) summing to 1."""
    return pool[..., None] * shares[..., :N_PLAYERS]
