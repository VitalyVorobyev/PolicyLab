"""Dynamics checks against hand computations, independent formulas and JPL catalog values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from jpl_catalog import load_catalog
from scipy.integrate import solve_ivp

from phasespace.gravity.dynamics import (
    FloatArray,
    collinear_lagrange_points,
    jacobi_constant,
    primary_distances,
    state_derivative,
    state_jacobian,
    variational_derivative,
)
from phasespace.gravity.propagation import propagate
from phasespace.gravity.system import EARTH_MOON
from phasespace.gravity.units import rotating_to_inertial

CATALOG = load_catalog()
MU = CATALOG.mu
RNG_SEED = 20261018


def test_primaries_placement_and_barycenter() -> None:
    system = CATALOG.system()
    assert system.earth_position == (-MU, 0.0)
    assert system.moon_position == (1.0 - MU, 0.0)
    assert system.moon_position[0] - system.earth_position[0] == pytest.approx(1.0, abs=1e-15)
    # Mass-weighted mean of the primaries is the origin.
    barycenter = (1.0 - MU) * system.earth_position[0] + MU * system.moon_position[0]
    assert barycenter == pytest.approx(0.0, abs=1e-16)


def test_distances_at_barycenter() -> None:
    r1, r2 = primary_distances(np.array([0.0, 0.0, 0.0, 0.0]), MU)
    assert float(r1) == pytest.approx(MU, abs=1e-16)
    assert float(r2) == pytest.approx(1.0 - MU, abs=1e-16)


def test_jacobi_constant_hand_computed_equilateral_point() -> None:
    # mu = 1/4: the point (1/2 - mu, sqrt(3)/2) is at unit distance from both primaries.
    # C = x^2 + y^2 + 2(1 - mu) + 2 mu - v^2 = 1/16 + 3/4 + 2 - (0.01 + 0.04) = 2.7625.
    mu = 0.25
    state = np.array([0.25, math.sqrt(3.0) / 2.0, 0.1, -0.2])
    assert float(jacobi_constant(state, mu)) == pytest.approx(2.7625, abs=1e-15)


def test_equilateral_points_are_equilibria() -> None:
    for y in (math.sqrt(3.0) / 2.0, -math.sqrt(3.0) / 2.0):
        derivative = state_derivative(np.array([0.5 - MU, y, 0.0, 0.0]), MU)
        assert np.max(np.abs(derivative)) < 1e-15


def _quintic_roots(mu: float) -> tuple[float, float, float]:
    """Collinear points from the classical quintics in gamma (Szebehely 1967, ch. 4)."""

    def real_positive(coefficients: list[float]) -> float:
        roots = np.roots(coefficients)
        real = [float(r.real) for r in roots if abs(r.imag) < 1e-12 and r.real > 0.0]
        assert len(real) == 1
        return real[0]

    g1 = real_positive([1.0, -(3.0 - mu), 3.0 - 2.0 * mu, -mu, 2.0 * mu, -mu])
    g2 = real_positive([1.0, 3.0 - mu, 3.0 - 2.0 * mu, -mu, -2.0 * mu, -mu])
    g3 = real_positive([1.0, 2.0 + mu, 1.0 + 2.0 * mu, -(1.0 - mu), -2.0 * (1.0 - mu), -(1.0 - mu)])
    return 1.0 - mu - g1, 1.0 - mu + g2, -mu - g3


@pytest.mark.parametrize("mu", [MU, EARTH_MOON.mu, 0.1, 0.5])
def test_collinear_points_match_independent_quintic(mu: float) -> None:
    ours = collinear_lagrange_points(mu)
    reference = _quintic_roots(mu)
    np.testing.assert_allclose(ours, reference, rtol=0.0, atol=1e-12)
    for x in ours:
        assert np.max(np.abs(state_derivative(np.array([x, 0.0, 0.0, 0.0]), mu))) < 1e-12


def test_collinear_points_match_jpl_catalog() -> None:
    l1, l2, l3 = collinear_lagrange_points(MU)
    # JPL lists 15 significant digits.
    assert l1 == pytest.approx(CATALOG.l1, abs=1e-14)
    assert l2 == pytest.approx(CATALOG.l2, abs=1e-14)
    assert l3 == pytest.approx(CATALOG.l3, abs=1e-14)


def test_jacobi_constant_matches_jpl_for_catalog_states() -> None:
    for orbit in CATALOG.orbits:
        state = np.array([orbit.x0, orbit.y0, orbit.vx0, orbit.vy0])
        assert float(jacobi_constant(state, MU)) == pytest.approx(orbit.jacobi, abs=1e-12)


def test_vector_field_conserves_jacobi_constant() -> None:
    # dC/dt = grad(C) . f must vanish; grad(C) by central differences.
    rng = np.random.default_rng(RNG_SEED)
    states = np.column_stack(
        [rng.uniform(-1.5, 1.5, 20), rng.uniform(0.2, 1.5, 20), rng.uniform(-1, 1, (20, 2))]
    )
    h = 1e-6
    for state in states:
        grad = np.array(
            [
                (
                    float(jacobi_constant(state + h * e, MU))
                    - float(jacobi_constant(state - h * e, MU))
                )
                / (2 * h)
                for e in np.eye(4)
            ]
        )
        assert abs(float(grad @ state_derivative(state, MU))) < 1e-7


def test_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    h = 1e-6
    for _ in range(10):
        state = np.array(
            [*rng.uniform(-1.2, 1.2, 1), *rng.uniform(0.2, 1.0, 1), *rng.uniform(-1, 1, 2)]
        )
        numeric = np.column_stack(
            [
                (state_derivative(state + h * e, MU) - state_derivative(state - h * e, MU))
                / (2 * h)
                for e in np.eye(4)
            ]
        )
        np.testing.assert_allclose(state_jacobian(state, MU), numeric, rtol=0.0, atol=1e-7)


def test_state_transition_matrix_matches_finite_differences() -> None:
    orbit = CATALOG.orbits[1]
    state0 = np.array([orbit.x0, orbit.y0, orbit.vx0, orbit.vy0])
    duration = 1.0
    augmented0 = np.concatenate([state0, np.eye(4).reshape(-1)])

    def rhs(t: float, y: FloatArray) -> FloatArray:
        return variational_derivative(y, MU)

    sol = solve_ivp(rhs, (0.0, duration), augmented0, method="DOP853", rtol=1e-12, atol=1e-12)
    phi = np.asarray(sol.y[4:, -1]).reshape(4, 4)
    system = CATALOG.system()
    h = 1e-6
    numeric = np.column_stack(
        [
            (
                propagate(state0 + h * e, duration, system).final_state
                - propagate(state0 - h * e, duration, system).final_state
            )
            / (2 * h)
            for e in np.eye(4)
        ]
    )
    np.testing.assert_allclose(phi, numeric, rtol=0.0, atol=1e-6)


def test_rotating_dynamics_reproduce_inertial_newtonian_gravity() -> None:
    # Propagate in the rotating frame, map samples to the barycentric inertial frame, and
    # compare their second time derivative with Newton's law for the moving primaries.
    system = CATALOG.system()
    orbit = CATALOG.orbits[1]
    state0 = np.array([orbit.x0, orbit.y0, orbit.vx0, orbit.vy0])
    h = 1e-2
    weights = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)  # O(h^4) stencil
    for t in (0.3, 0.5, 0.7):
        stencil = t + h * np.arange(-2.0, 3.0)
        # Integrator end states (not the dense interpolant, whose error the stencil amplifies).
        rotating = np.array([propagate(state0, float(ti), system).final_state for ti in stencil])
        inertial = rotating_to_inertial(rotating, stencil, MU)
        accel = weights @ inertial[:, :2]
        position = inertial[2, :2]
        earth = -MU * np.array([math.cos(t), math.sin(t)])
        moon = (1 - MU) * np.array([math.cos(t), math.sin(t)])
        newton = (
            -(1 - MU) * (position - earth) / np.linalg.norm(position - earth) ** 3
            - MU * (position - moon) / np.linalg.norm(position - moon) ** 3
        )
        np.testing.assert_allclose(accel, newton, rtol=0.0, atol=1e-6)
