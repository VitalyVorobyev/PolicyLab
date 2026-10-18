"""Planar circular restricted three-body problem (CR3BP) in normalized units.

Frame: barycentric, uniformly rotating (synodic) with unit angular rate about +z.
Earth (mass 1 - mu) sits at (-mu, 0) and the Moon (mass mu) at (1 - mu, 0).
State vector order: (x, y, vx, vy), velocities relative to the rotating frame.

Equations of motion, with r1 = |r - r_earth| and r2 = |r - r_moon|::

    vx' =  2 vy + x - (1 - mu)(x + mu)/r1^3 - mu (x - 1 + mu)/r2^3
    vy' = -2 vx + y - (1 - mu) y/r1^3       - mu y/r2^3

Jacobi constant: C = x^2 + y^2 + 2(1 - mu)/r1 + 2 mu/r2 - vx^2 - vy^2.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

type FloatArray = npt.NDArray[np.float64]

STATE_DIM: Final = 4


def _as_states(state: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape[-1:] != (STATE_DIM,):
        raise ValueError(f"state must have trailing dimension {STATE_DIM}, got {arr.shape}")
    return arr


def primary_distances(state: npt.ArrayLike, mu: float) -> tuple[FloatArray, FloatArray]:
    """Return (r1, r2): distances to Earth and Moon for states of shape (..., 4)."""
    s = _as_states(state)
    x, y = s[..., 0], s[..., 1]
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - 1.0 + mu, y)
    return r1, r2


def jacobi_constant(state: npt.ArrayLike, mu: float) -> FloatArray:
    """Jacobi constant for states of shape (..., 4); returns an array of shape (...)."""
    s = _as_states(state)
    x, y, vx, vy = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    r1, r2 = primary_distances(s, mu)
    return x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - vx * vx - vy * vy


def acceleration(state: npt.ArrayLike, mu: float) -> FloatArray:
    """Rotating-frame acceleration (ax, ay) for states of shape (..., 4)."""
    s = _as_states(state)
    x, y, vx, vy = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    r1, r2 = primary_distances(s, mu)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        k1 = (1.0 - mu) / r1**3
        k2 = mu / r2**3
        ax = 2.0 * vy + x - k1 * (x + mu) - k2 * (x - 1.0 + mu)
        ay = -2.0 * vx + y - k1 * y - k2 * y
    return np.stack([ax, ay], axis=-1)


def state_derivative(state: npt.ArrayLike, mu: float) -> FloatArray:
    """Time derivative (vx, vy, ax, ay) for states of shape (..., 4).

    At a primary's position the result is non-finite; callers must check.
    """
    s = _as_states(state)
    acc = acceleration(s, mu)
    return np.concatenate([s[..., 2:4], acc], axis=-1)


def potential_hessian(x: float, y: float, mu: float) -> tuple[float, float, float]:
    """Second derivatives (Uxx, Uxy, Uyy) of the pseudo-potential.

    U = (x^2 + y^2)/2 + (1 - mu)/r1 + mu/r2.
    """
    dx1, dx2 = x + mu, x - 1.0 + mu
    r1sq = dx1 * dx1 + y * y
    r2sq = dx2 * dx2 + y * y
    r1_3 = r1sq**1.5
    r2_3 = r2sq**1.5
    r1_5 = r1sq * r1_3
    r2_5 = r2sq * r2_3
    m1 = 1.0 - mu
    uxx = 1.0 - m1 / r1_3 - mu / r2_3 + 3.0 * m1 * dx1 * dx1 / r1_5 + 3.0 * mu * dx2 * dx2 / r2_5
    uyy = 1.0 - m1 / r1_3 - mu / r2_3 + 3.0 * m1 * y * y / r1_5 + 3.0 * mu * y * y / r2_5
    uxy = 3.0 * m1 * dx1 * y / r1_5 + 3.0 * mu * dx2 * y / r2_5
    return uxx, uxy, uyy


def state_jacobian(state: npt.ArrayLike, mu: float) -> FloatArray:
    """4x4 Jacobian of the equations of motion with respect to the state."""
    s = _as_states(state)
    if s.shape != (STATE_DIM,):
        raise ValueError("state_jacobian expects a single state of shape (4,)")
    uxx, uxy, uyy = potential_hessian(float(s[0]), float(s[1]), mu)
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [uxx, uxy, 0.0, 2.0],
            [uxy, uyy, -2.0, 0.0],
        ],
        dtype=np.float64,
    )


def variational_derivative(augmented: FloatArray, mu: float) -> FloatArray:
    """Derivative of the state plus row-major 4x4 state transition matrix (20 values)."""
    state = augmented[:STATE_DIM]
    phi = augmented[STATE_DIM:].reshape(STATE_DIM, STATE_DIM)
    dphi = state_jacobian(state, mu) @ phi
    return np.concatenate([state_derivative(state, mu), dphi.reshape(-1)])


def _collinear_force(x: float, mu: float) -> float:
    """dU/dx on the x-axis; zero at the collinear Lagrange points."""
    d1, d2 = x + mu, x - 1.0 + mu
    return x - (1.0 - mu) * d1 / abs(d1) ** 3 - mu * d2 / abs(d2) ** 3


def collinear_lagrange_points(mu: float) -> tuple[float, float, float]:
    """x-coordinates of L1 (between the primaries), L2 (beyond the Moon), L3 (beyond Earth).

    Each root of dU/dx = 0 on the x-axis is bracketed between the primaries' singularities
    (where the sign of dU/dx is known) and solved with Brent's method (absolute tolerance 1e-15).
    """
    if not (0.0 < mu <= 0.5):
        raise ValueError(f"mu must be in (0, 0.5], got {mu}")
    eps = 1e-9
    earth, moon = -mu, 1.0 - mu
    brackets = ((earth + eps, moon - eps), (moon + eps, 2.0), (-2.0, earth - eps))
    roots: list[float] = []
    for lo, hi in brackets:
        root = brentq(_collinear_force, lo, hi, args=(mu,), xtol=1e-15)
        roots.append(float(root))
    return roots[0], roots[1], roots[2]
