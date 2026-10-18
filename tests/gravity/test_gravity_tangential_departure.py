"""Regression: tangential departure from a circular Earth parking orbit (an exact apsis).

The departure state has h = (r - r_earth) . v == 0.0 exactly at t0. LSODA's dense output
at the start of its first step is not bit-identical to the initial state, so solve_ivp's
brentq bracket for the Earth periapsis event had no sign change and raised ValueError
("f(a) and f(b) must have different signs"); DOP853 happened not to. Both methods must
now propagate, record the t0 apsis identically, and agree on the first perilune.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import phasespace.gravity.propagation as propagation
from phasespace.gravity.dynamics import FloatArray, primary_distances
from phasespace.gravity.propagation import (
    CollisionRadii,
    PropagationResult,
    SolverSettings,
    Termination,
    propagate,
)
from phasespace.gravity.system import EARTH_MOON
from phasespace.gravity.units import (
    inertial_to_rotating,
    length_from_km,
    time_from_days,
    velocity_from_km_s,
)

PARKING_ALTITUDE_KM = 185.0
RETURN_SHELL_KM = EARTH_MOON.earth_radius_km + 121.92
DEPARTURE_ANGLE_DEG = 227.63
HORIZON_DAYS = 10.0
COLLISION = CollisionRadii(earth_km=RETURN_SHELL_KM, moon_km=EARTH_MOON.moon_radius_km)
LSODA = SolverSettings(method="LSODA", rtol=1e-12, atol=1e-12)
DOP853 = SolverSettings(method="DOP853", rtol=1e-12, atol=1e-12)

# Independent-integrator agreement on the first perilune, normalized units (1e-8 is
# about 3.8 m in distance and 3.8 ms in time). Observed LSODA - DOP853 differences at
# 1e-12 are <= 2e-10 in both, dominated by LSODA (DOP853 at 1e-12 vs 1e-13 differs by
# ~1e-11), so the bound leaves a margin of ~50 while staying far below anything that
# matters for a lunar flyby classification.
PERILUNE_AGREEMENT = 1e-8


def _tangential_departure(dv_km_s: float) -> FloatArray:
    """Rotating-frame state just after a posigrade tangential burn at t0 = 0."""
    mu = EARTH_MOON.mu
    r = length_from_km(EARTH_MOON.earth_radius_km + PARKING_ALTITUDE_KM, EARTH_MOON)
    circular = math.sqrt((1.0 - mu) / r)
    dv = velocity_from_km_s(dv_km_s, EARTH_MOON)
    theta = math.radians(DEPARTURE_ANGLE_DEG)
    radial = np.array([math.cos(theta), math.sin(theta)])
    tangential = np.array([-math.sin(theta), math.cos(theta)])
    velocity = circular * tangential + dv * tangential
    inertial = np.array([r * radial[0], r * radial[1], velocity[0], velocity[1]])
    return inertial_to_rotating(inertial, 0.0, mu, origin="earth")


def _first_perilune(result: PropagationResult) -> tuple[float, float]:
    passes = [p for p in result.periapses if p.name == "moon_periapsis"]
    assert passes, "no lunar periapsis recorded"
    _, r2 = primary_distances(np.array(passes[0].state), EARTH_MOON.mu)
    return passes[0].t, float(r2)


@pytest.mark.parametrize(
    ("dv_km_s", "exact_apsis"),
    # 3.1628 km/s is the reported crash: h(t0) rounds to exactly 0.0. The others leave
    # h(t0) at rounding level (~1e-17), the same degeneracy without the exact zero.
    [(3.1, False), (3.1628, True), (3.2, False)],
)
def test_lsoda_and_dop853_agree_on_tangential_departure(dv_km_s: float, exact_apsis: bool) -> None:
    state0 = _tangential_departure(dv_km_s)
    x, y, vx, vy = (float(v) for v in state0)
    h0 = (x + EARTH_MOON.mu) * vx + y * vy
    assert (h0 == 0.0) if exact_apsis else (0.0 < abs(h0) < 1e-15)
    duration = time_from_days(HORIZON_DAYS, EARTH_MOON)

    results = [
        propagate(state0, duration, EARTH_MOON, solver=solver, collision=COLLISION)
        for solver in (LSODA, DOP853)
    ]
    for result in results:
        assert result.termination in (Termination.HORIZON, Termination.EARTH_COLLISION)
        if exact_apsis:
            # h(t0) == 0 and h increases: both methods record the apsis at exactly t0.
            assert result.periapses[0].name == "earth_periapsis"
            assert result.periapses[0].t == 0.0

    (t_lsoda, r_lsoda), (t_dop, r_dop) = (_first_perilune(r) for r in results)
    assert abs(t_lsoda - t_dop) < PERILUNE_AGREEMENT
    assert abs(r_lsoda - r_dop) < PERILUNE_AGREEMENT
    assert results[0].termination is results[1].termination


def test_solver_value_error_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_solve_ivp(*args: object, **kwargs: object) -> None:
        raise ValueError("f(a) and f(b) must have different signs")

    monkeypatch.setattr(propagation, "solve_ivp", failing_solve_ivp)
    state0 = _tangential_departure(3.1628)
    result = propagate(state0, 1.0, EARTH_MOON, solver=LSODA, collision=COLLISION, t0=0.5)
    assert result.termination is Termination.SOLVER_FAILURE
    assert "must have different signs" in result.message
    assert result.dense is None
    assert result.events == () and result.periapses == ()
    np.testing.assert_array_equal(result.t, [0.5])
    np.testing.assert_array_equal(result.final_state, state0)
