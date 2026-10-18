"""Grazing collisions: periapsis below the surface, entered and left within one solver step.

Each case is built from a periapsis placed `depth_km` below the body's surface (radial
velocity exactly zero, tangential speed `speed`), integrated backward to a start state
outside the body. The reference crossing time comes from a fixed-step RK4 written here
with its own equations of motion (no code under test), refined with brentq on the length
of a single RK4 step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.propagation import Termination, propagate
from phasespace.gravity.reference import JPL_EARTH_MOON

SYSTEM = JPL_EARTH_MOON
MU = SYSTEM.mu
RK4_STEP = 1e-6
"""Fixed RK4 step (normalized time); local dynamics rate v/R <= 1.2e3, so v h/R <= 1.2e-3."""
START_DISTANCE = 1.25
"""The start state is where the backward integration first reaches 1.25 body radii."""


def _rhs(s: FloatArray) -> FloatArray:
    x, y, vx, vy = (float(v) for v in s)
    r1 = math.sqrt((x + MU) ** 2 + y**2)
    r2 = math.sqrt((x - 1.0 + MU) ** 2 + y**2)
    ax = 2.0 * vy + x - (1.0 - MU) * (x + MU) / r1**3 - MU * (x - 1.0 + MU) / r2**3
    ay = -2.0 * vx + y - (1.0 - MU) * y / r1**3 - MU * y / r2**3
    return np.array([vx, vy, ax, ay])


def _rk4(s: FloatArray, h: float) -> FloatArray:
    k1 = _rhs(s)
    k2 = _rhs(s + 0.5 * h * k1)
    k3 = _rhs(s + 0.5 * h * k2)
    k4 = _rhs(s + h * k3)
    return s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _body(name: str) -> tuple[float, float]:
    """(center x, radius) in normalized units."""
    if name == "moon":
        return 1.0 - MU, SYSTEM.moon_radius_km / SYSTEM.length_scale_km
    return -MU, SYSTEM.earth_radius_km / SYSTEM.length_scale_km


def _distance(s: FloatArray, center_x: float) -> float:
    return math.hypot(float(s[0]) - center_x, float(s[1]))


def _grazing_start(name: str, speed: float, depth_km: float) -> tuple[FloatArray, float]:
    """Start state outside the body and the time from it to the constructed periapsis.

    The periapsis is on the -x side of the body center at distance R - depth, with
    prograde (counterclockwise) rotating-frame velocity of magnitude `speed`.
    """
    center_x, radius = _body(name)
    rp = radius - depth_km / SYSTEM.length_scale_km
    s = np.array([center_x - rp, 0.0, 0.0, -speed])
    t = 0.0
    while _distance(s, center_x) < START_DISTANCE * radius:
        s = _rk4(s, -RK4_STEP)
        t += RK4_STEP
        assert t < 0.05, "backward integration did not leave the body vicinity"
    return s, t


def _reference_first_crossing(
    s0: FloatArray, center_x: float, radius: float, t_max: float, step: float = RK4_STEP
) -> tuple[float, FloatArray] | None:
    """First inward crossing of the surface by fixed-step RK4, or None before t_max."""
    s, t = s0.copy(), 0.0
    while t < t_max:
        s_next = _rk4(s, step)
        if _distance(s_next, center_x) < radius:
            tau = float(
                brentq(_gap_after, 0.0, step, args=(s, center_x, radius), xtol=1e-18, rtol=1e-15)
            )
            return t + tau, _rk4(s, tau)
        s, t = s_next, t + step
    return None


def _gap_after(tau: float, s: FloatArray, center_x: float, radius: float) -> float:
    """distance - radius after one RK4 step of length tau from s."""
    return _distance(_rk4(s, tau), center_x) - radius


def _min_distance(s0: FloatArray, center_x: float, t_max: float) -> float:
    s, t, best = s0.copy(), 0.0, math.inf
    while t < t_max:
        s = _rk4(s, RK4_STEP)
        t += RK4_STEP
        best = min(best, _distance(s, center_x))
    return best


GRAZING_CASES = [
    # (body, periapsis speed, depth below the surface in km). Before periapsis events were
    # added, all of these except moon/1.8/1.0 and moon/2.5/1.0 were reported as 'horizon'.
    ("moon", 1.8, 0.01),
    ("moon", 1.8, 0.1),
    ("moon", 1.8, 1.0),
    ("moon", 2.5, 0.01),
    ("moon", 2.5, 0.1),
    ("moon", 2.5, 1.0),
    ("moon", 5.0, 0.01),
    ("moon", 5.0, 0.1),
    ("earth", 10.0, 0.01),
    ("earth", 12.0, 1.0),
]


def _case_id(case: tuple[str, float, float]) -> str:
    return f"{case[0]}-v{case[1]}-{case[2]}km"


@pytest.mark.parametrize("case", GRAZING_CASES, ids=_case_id)
def test_grazing_periapsis_below_surface_is_a_collision(case: tuple[str, float, float]) -> None:
    name, speed, depth_km = case
    center_x, radius = _body(name)
    start, t_to_periapsis = _grazing_start(name, speed, depth_km)
    reference = _reference_first_crossing(start, center_x, radius, 2.0 * t_to_periapsis)
    assert reference is not None
    t_ref, s_ref = reference

    result = propagate(start, 2.0 * t_to_periapsis, SYSTEM)
    expected = Termination.MOON_COLLISION if name == "moon" else Termination.EARTH_COLLISION
    assert result.termination is expected
    (event,) = result.events
    assert event.name == expected.value
    assert event.direction == -1.0
    assert event.t == result.t_final
    np.testing.assert_array_equal(np.array(event.state), result.final_state)
    assert result.t_final == pytest.approx(t_ref, abs=1e-8)
    np.testing.assert_allclose(result.final_state, s_ref, rtol=0.0, atol=1e-8)
    final = result.final_state
    assert _distance(final, center_x) == pytest.approx(radius, abs=1e-13)
    radial_velocity = (final[0] - center_x) * final[2] + final[1] * final[3]
    assert radial_velocity < 0.0
    # Nothing after the crossing is exposed.
    assert np.all(np.diff(result.t) > 0.0)
    assert all(p.t <= result.t_final for p in result.periapses)
    with pytest.raises(ValueError):
        result.sample([result.t_final + 1e-6])
    assert result.dense is not None
    np.testing.assert_allclose(result.sample([result.t_final])[0], final, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(("name", "speed"), [("moon", 2.5), ("moon", 5.0), ("earth", 12.0)])
def test_periapsis_just_above_surface_is_not_a_collision(name: str, speed: float) -> None:
    # Periapsis 10 m above the surface (negative depth): a near miss must not be flagged.
    center_x, radius = _body(name)
    start, t_to_periapsis = _grazing_start(name, speed, -0.01)
    assert _min_distance(start, center_x, 2.0 * t_to_periapsis) > radius
    result = propagate(start, 2.0 * t_to_periapsis, SYSTEM)
    assert result.termination is Termination.HORIZON
    assert result.events == ()
    passes = [p for p in result.periapses if p.name == f"{name}_periapsis"]
    assert len(passes) == 1
    (periapsis,) = passes
    assert periapsis.direction == 1.0
    assert periapsis.t == pytest.approx(t_to_periapsis, abs=1e-8)
    rp = radius + 0.01 / SYSTEM.length_scale_km
    assert _distance(np.array(periapsis.state), center_x) == pytest.approx(rp, abs=1e-11)


def test_fixed_step_reference_is_converged() -> None:
    # Halving the RK4 step changes the reference crossing time by far less than 1e-8.
    center_x, radius = _body("moon")
    start, t_to_periapsis = _grazing_start("moon", 5.0, 0.01)
    coarse = _reference_first_crossing(start, center_x, radius, 2.0 * t_to_periapsis)
    fine = _reference_first_crossing(
        start, center_x, radius, 2.0 * t_to_periapsis, step=RK4_STEP / 2.0
    )
    assert coarse is not None and fine is not None
    assert abs(coarse[0] - fine[0]) < 1e-10
