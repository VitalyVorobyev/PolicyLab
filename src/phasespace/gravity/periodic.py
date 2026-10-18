"""Symmetric planar Lyapunov orbits by single-shooting differential correction.

Planar CR3BP orbits that cross the x-axis perpendicularly twice are periodic, by the
symmetry (x, y, vx, vy, t) -> (x, -y, -vx, vy, -t). Starting from
s0 = (x0, 0, 0, vy0) we integrate the state and its 4x4 state transition matrix (STM)
to the next x-axis crossing at t_half and adjust vy0 (x0 fixed) until vx(t_half) = 0.

Linearizing about the crossing with the half-period free (dy must stay 0)::

    d vx = (Phi[2,3] - ax * Phi[1,3] / vy) d vy0

where ax is the x-acceleration and vy the y-velocity at the crossing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.integrate import solve_ivp

from phasespace.gravity.dynamics import (
    STATE_DIM,
    FloatArray,
    collinear_lagrange_points,
    jacobi_constant,
    state_derivative,
    variational_derivative,
)
from phasespace.gravity.propagation import SolverSettings

CORRECTOR_SOLVER: Final = SolverSettings(method="DOP853", rtol=1e-13, atol=1e-13)


class CorrectionError(RuntimeError):
    """Differential correction did not converge."""


@dataclass(frozen=True, slots=True)
class SymmetricOrbit:
    """A corrected symmetric periodic orbit with its convergence record."""

    mu: float
    x0: float
    vy0: float
    period: float
    jacobi: float
    iterations: int
    half_period_abs_vx: float
    solver: SolverSettings
    tolerance: float

    @property
    def state0(self) -> FloatArray:
        return np.array([self.x0, 0.0, 0.0, self.vy0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LinearGuess:
    """Linearized (Lyapunov) initial guess around a collinear point."""

    x0: float
    vy0: float
    period: float


def linear_lyapunov_guess(mu: float, point: int, x_offset: float) -> LinearGuess:
    """First-order planar Lyapunov guess about L1 or L2.

    With c2 = (1-mu)/r1^3 + mu/r2^3 at the libration point, the linearized in-plane
    motion has an oscillatory mode of frequency nu, nu^2 = (2 - c2 + sqrt(9 c2^2 - 8 c2))/2,
    with xi = A cos(nu t), eta = -k A sin(nu t), k = (nu^2 + 1 + 2 c2) / (2 nu).
    `x_offset` is A (signed displacement of the x-axis crossing from the libration point).
    """
    if point not in (1, 2):
        raise ValueError("point must be 1 or 2")
    lagrange_x = collinear_lagrange_points(mu)[point - 1]
    r1 = abs(lagrange_x + mu)
    r2 = abs(lagrange_x - 1.0 + mu)
    c2 = (1.0 - mu) / r1**3 + mu / r2**3
    nu = math.sqrt((2.0 - c2 + math.sqrt(9.0 * c2 * c2 - 8.0 * c2)) / 2.0)
    k = (nu * nu + 1.0 + 2.0 * c2) / (2.0 * nu)
    return LinearGuess(x0=lagrange_x + x_offset, vy0=-k * nu * x_offset, period=2.0 * math.pi / nu)


class _AxisCrossing:
    """Event y = 0 in the direction opposite to the initial vy (the half-period crossing)."""

    def __init__(self, direction: float) -> None:
        self.terminal = True
        self.direction = direction

    def __call__(self, t: float, y: FloatArray) -> float:
        return float(y[1])


def _half_period_crossing(
    mu: float, x0: float, vy0: float, solver: SolverSettings, max_time: float
) -> tuple[float, FloatArray, FloatArray]:
    """Integrate state + STM to the first opposite-direction x-axis crossing."""
    augmented0 = np.concatenate(
        [np.array([x0, 0.0, 0.0, vy0]), np.eye(STATE_DIM).reshape(-1)]
    ).astype(np.float64)

    def rhs(t: float, y: FloatArray) -> FloatArray:
        return variational_derivative(y, mu)

    sol = solve_ivp(
        rhs,
        (0.0, max_time),
        augmented0,
        method=solver.method,
        events=_AxisCrossing(-math.copysign(1.0, vy0)),
        rtol=solver.rtol,
        atol=solver.atol,
    )
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise CorrectionError(f"no x-axis crossing before t={max_time}: {sol.message}")
    t_half = float(sol.t_events[0][0])
    y_half = np.asarray(sol.y_events[0][0], dtype=np.float64)
    state = y_half[:STATE_DIM]
    phi = y_half[STATE_DIM:].reshape(STATE_DIM, STATE_DIM)
    return t_half, state, phi


def correct_symmetric_orbit(
    mu: float,
    x0: float,
    vy0_guess: float,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 25,
    solver: SolverSettings = CORRECTOR_SOLVER,
    max_time: float = 20.0,
) -> SymmetricOrbit:
    """Correct vy0 (x0 fixed) until |vx| < `tolerance` at the half-period x-axis crossing.

    Raises CorrectionError if it does not converge within `max_iterations`.
    """
    if vy0_guess == 0.0 or not math.isfinite(vy0_guess):
        raise ValueError("vy0_guess must be finite and non-zero")
    vy0 = vy0_guess
    for iteration in range(1, max_iterations + 1):
        t_half, state, phi = _half_period_crossing(mu, x0, vy0, solver, max_time)
        x, y, vx, vy = (float(v) for v in state)
        ax = float(state_derivative(state, mu)[2])
        if abs(vx) < tolerance:
            return SymmetricOrbit(
                mu=mu,
                x0=x0,
                vy0=vy0,
                period=2.0 * t_half,
                jacobi=float(jacobi_constant(np.array([x0, 0.0, 0.0, vy0]), mu)),
                iterations=iteration,
                half_period_abs_vx=abs(vx),
                solver=solver,
                tolerance=tolerance,
            )
        slope = float(phi[2, 3]) - ax * float(phi[1, 3]) / vy
        if slope == 0.0 or not math.isfinite(slope):
            raise CorrectionError(f"singular correction at iteration {iteration} (x={x}, y={y})")
        updated = vy0 - vx / slope
        if math.copysign(1.0, updated) != math.copysign(1.0, vy0_guess):
            raise CorrectionError(
                f"vy0 changed sign at iteration {iteration}; the guess is outside the basin "
                "of the intended orbit"
            )
        vy0 = updated
    raise CorrectionError(f"no convergence after {max_iterations} iterations")


def continue_lyapunov(
    mu: float,
    point: int,
    x_offset: float,
    *,
    steps: int = 10,
    tolerance: float = 1e-11,
    solver: SolverSettings = CORRECTOR_SOLVER,
) -> SymmetricOrbit:
    """Planar Lyapunov orbit about L1/L2 whose x-axis crossing is at L_point + x_offset.

    Natural-parameter continuation in x0: the first member (offset x_offset/steps) is
    corrected from `linear_lyapunov_guess`; each following member is corrected from a
    linear extrapolation of the previous two vy0 values. Uses no catalog data.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    offsets = [x_offset * (k + 1) / steps for k in range(steps)]
    first = linear_lyapunov_guess(mu, point, offsets[0])
    orbit = correct_symmetric_orbit(mu, first.x0, first.vy0, tolerance=tolerance, solver=solver)
    previous: SymmetricOrbit | None = None
    lagrange_x = first.x0 - offsets[0]
    for offset in offsets[1:]:
        if previous is None:
            guess = orbit.vy0 * offset / offsets[0]
        else:
            slope = (orbit.vy0 - previous.vy0) / (orbit.x0 - previous.x0)
            guess = orbit.vy0 + slope * (lagrange_x + offset - orbit.x0)
        previous, orbit = (
            orbit,
            correct_symmetric_orbit(
                mu, lagrange_x + offset, guess, tolerance=tolerance, solver=solver
            ),
        )
    return orbit
