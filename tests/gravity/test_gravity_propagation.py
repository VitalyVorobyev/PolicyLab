"""Propagation checks: JPL periodicity, Jacobi drift, tolerance agreement, terminations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from jpl_catalog import JplOrbit, load_catalog

from phasespace.gravity.dynamics import jacobi_constant, primary_distances
from phasespace.gravity.propagation import (
    CollisionRadii,
    SolverSettings,
    Termination,
    propagate,
)

CATALOG = load_catalog()
SYSTEM = CATALOG.system()
MU = CATALOG.mu

# Declared thresholds (normalized units); see docs/studies/gravity-atlas/README.md.
PERIODICITY_TOLERANCE = 1e-9
JACOBI_DRIFT_TOLERANCE = 1e-10
TOLERANCE_SET_AGREEMENT = 1e-7
DRIFT_PERIODS = 3
LOOSE = SolverSettings(method="DOP853", rtol=1e-10, atol=1e-10)


def _state(orbit: JplOrbit) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.array([orbit.x0, orbit.y0, orbit.vx0, orbit.vy0], dtype=np.float64)


@pytest.mark.parametrize("orbit", CATALOG.orbits, ids=lambda o: f"row{o.row_index}")
def test_jpl_orbit_returns_after_one_jpl_period(orbit: JplOrbit) -> None:
    state0 = _state(orbit)
    result = propagate(state0, orbit.period, SYSTEM)
    assert result.termination is Termination.HORIZON
    assert result.t_final == orbit.period
    assert np.max(np.abs(result.final_state - state0)) < PERIODICITY_TOLERANCE


@pytest.mark.parametrize("orbit", CATALOG.orbits, ids=lambda o: f"row{o.row_index}")
def test_jacobi_drift_over_several_periods(orbit: JplOrbit) -> None:
    state0 = _state(orbit)
    result = propagate(state0, DRIFT_PERIODS * orbit.period, SYSTEM)
    assert result.termination is Termination.HORIZON
    times = np.linspace(0.0, result.t_final, 2001)
    drift = jacobi_constant(result.sample(times), MU) - orbit.jacobi
    assert np.max(np.abs(drift)) < JACOBI_DRIFT_TOLERANCE
    assert (
        np.max(np.abs(jacobi_constant(result.states, MU) - orbit.jacobi)) < JACOBI_DRIFT_TOLERANCE
    )


@pytest.mark.parametrize("orbit", CATALOG.orbits, ids=lambda o: f"row{o.row_index}")
def test_two_tolerance_sets_agree(orbit: JplOrbit) -> None:
    state0 = _state(orbit)
    tight = propagate(state0, orbit.period, SYSTEM)
    loose = propagate(state0, orbit.period, SYSTEM, solver=LOOSE)
    assert np.max(np.abs(tight.final_state - loose.final_state)) < TOLERANCE_SET_AGREEMENT
    assert loose.nfev < tight.nfev


def test_propagation_is_repeatable() -> None:
    state0 = _state(CATALOG.orbits[0])
    a = propagate(state0, 2.0, SYSTEM)
    b = propagate(state0, 2.0, SYSTEM)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.states, b.states)


@pytest.mark.parametrize(
    ("body", "start_x", "radius_km", "termination"),
    [
        ("moon", 1.0 - MU + 0.01, CATALOG.radius_secondary_km, Termination.MOON_COLLISION),
        ("earth", -MU + 0.03, 6371.0, Termination.EARTH_COLLISION),
    ],
)
def test_radial_plunge_fires_inward_collision_event(
    body: str, start_x: float, radius_km: float, termination: Termination
) -> None:
    # Released at rest (rotating frame) on the x-axis near the body: falls in nearly radially.
    result = propagate(np.array([start_x, 0.0, 0.0, 0.0]), 2.0, SYSTEM)
    assert result.termination is termination
    assert len(result.events) == 1
    event = result.events[0]
    assert event.name == termination.value
    assert event.direction == -1.0
    assert event.t == result.t_final < 2.0
    np.testing.assert_array_equal(np.array(event.state), result.final_state)
    r1, r2 = primary_distances(np.array(event.state), MU)
    distance = float(r1) if body == "earth" else float(r2)
    assert distance * SYSTEM.length_scale_km == pytest.approx(radius_km, abs=1e-6)
    # Radial velocity at the event is negative (entering the body).
    center_x = -MU if body == "earth" else 1.0 - MU
    x, y, vx, vy = event.state
    assert ((x - center_x) * vx + y * vy) / distance < 0.0


def test_outward_crossing_does_not_fire() -> None:
    radius = SYSTEM.moon_radius_km / SYSTEM.length_scale_km
    # Starts inside the collision radius moving outward fast; crosses r2 = R increasing.
    result = propagate(np.array([1.0 - MU + 0.5 * radius, 0.0, 5.0, 0.0]), 0.01, SYSTEM)
    assert result.termination is Termination.HORIZON
    assert result.events == ()
    _, r2 = primary_distances(result.final_state, MU)
    assert float(r2) > radius


def test_horizon_termination() -> None:
    result = propagate(_state(CATALOG.orbits[1]), 0.5, SYSTEM, t0=1.0)
    assert result.termination is Termination.HORIZON
    assert result.t[0] == 1.0
    assert result.t_final == 1.5
    assert result.events == ()
    assert np.all(np.diff(result.t) > 0.0)


def test_disabled_collision_radii_let_trajectory_continue() -> None:
    result = propagate(
        np.array([1.0 - MU + 0.01, 0.0, 0.0, 0.0]),
        0.0087,  # just past the 0.00858 impact time of the plunge test
        SYSTEM,
        collision=CollisionRadii(earth_km=None, moon_km=None),
    )
    assert result.termination is Termination.HORIZON
    _, r2 = primary_distances(result.final_state, MU)
    assert float(r2) < SYSTEM.moon_radius_km / SYSTEM.length_scale_km


def test_non_finite_derivative_at_singularity() -> None:
    # Exactly at the Earth's center (x + mu == 0 exactly): derivative is non-finite.
    result = propagate(
        np.array([-MU, 0.0, 0.0, 0.0]), 1.0, SYSTEM, collision=CollisionRadii(None, None)
    )
    assert result.termination is Termination.NON_FINITE
    assert result.dense is None
    assert result.t_final == 0.0


def test_near_singular_start_hits_evaluation_limit() -> None:
    # Next to the Moon's center with collisions disabled: tiny steps exhaust the budget.
    result = propagate(
        np.array([1.0 - MU + 1e-12, 0.0, 0.0, 0.0]),
        1.0,
        SYSTEM,
        collision=CollisionRadii(None, None),
        max_evaluations=5000,
    )
    assert result.termination is Termination.EVALUATION_LIMIT


@pytest.mark.parametrize(
    "state", [[math.nan, 0.0, 0.0, 0.0], [0.5, math.inf, 0.0, 0.0], [0.5, 0.0, 0.0]]
)
def test_invalid_initial_state_rejected(state: list[float]) -> None:
    with pytest.raises(ValueError):
        propagate(np.array(state), 1.0, SYSTEM)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.inf, math.nan])
def test_invalid_duration_rejected(duration: float) -> None:
    with pytest.raises(ValueError):
        propagate(np.array([0.5, 0.0, 0.0, 0.0]), duration, SYSTEM)
