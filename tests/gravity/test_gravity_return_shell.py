"""Terminal Earth return shell in `propagate` (design section 9.2).

Grazing cases follow test_gravity_grazing.py: an Earth periapsis is placed below (or just
above) the shell and integrated backward with an independent fixed-step RK4 written here
(no code under test) to a start outside the shell; the reference crossing time comes from
the same RK4, refined with brentq on the length of a single step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.propagation import (
    CollisionRadii,
    PropagationResult,
    Termination,
    propagate,
)
from phasespace.gravity.system import EARTH_MOON

SYSTEM = EARTH_MOON
MU = SYSTEM.mu
L = SYSTEM.length_scale_km
SHELL_KM = SYSTEM.earth_radius_km + 121.92
SHELL = SHELL_KM / L
RK4_STEP = 1e-6


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


def _r1(s: FloatArray) -> float:
    return math.hypot(float(s[0]) + MU, float(s[1]))


def _start_from_periapsis(depth_km: float, speed: float) -> tuple[FloatArray, float]:
    """Start at 1.25 R_shell, integrated backward from a periapsis depth_km below the shell."""
    rp = SHELL - depth_km / L
    s = np.array([-MU - rp, 0.0, 0.0, -speed])
    t = 0.0
    while _r1(s) < 1.25 * SHELL:
        s = _rk4(s, -RK4_STEP)
        t += RK4_STEP
    return s, t


def _reference_crossing(s0: FloatArray, t_max: float) -> float | None:
    s, t = s0.copy(), 0.0
    while t < t_max:
        s_next = _rk4(s, RK4_STEP)
        if _r1(s_next) < SHELL:
            tau = float(brentq(_gap_after, 0.0, RK4_STEP, args=(s,), xtol=1e-18))
            return t + tau
        s, t = s_next, t + RK4_STEP
    return None


def _gap_after(h: float, s: FloatArray) -> float:
    """r1 - R_shell after one RK4 step of length h from s."""
    return _r1(_rk4(s, h)) - SHELL


def _radial_state(r: float, radial_speed: float) -> FloatArray:
    """Earth-relative radial motion along -x at rotating speed `radial_speed` (>0: outward)."""
    return np.array([-MU - r, 0.0, -radial_speed, 0.0])


def test_inbound_crossing_terminates_on_the_shell() -> None:
    result = propagate(
        _radial_state(SHELL + 2000.0 / L, -5.0), 0.01, SYSTEM, return_shell_km=SHELL_KM
    )
    assert result.termination is Termination.EARTH_RETURN_SHELL
    (event,) = result.events
    assert event.name == "earth_return_shell"
    assert event.t == result.t_final
    final = result.final_state
    assert _r1(final) == pytest.approx(SHELL, abs=1e-13)
    assert (final[0] + MU) * final[2] + final[1] * final[3] < 0.0
    # The same motion without the shell passes the shell radius at the same time.
    free = propagate(_radial_state(SHELL + 2000.0 / L, -5.0), result.t_final * 1.01, SYSTEM)
    t_cross = brentq(lambda t: _r1(free.sample([t])[0]) - SHELL, 0.0, free.t_final, xtol=1e-15)
    assert result.t_final == pytest.approx(t_cross, abs=1e-12)


def test_outbound_crossing_does_not_fire() -> None:
    # Start exactly on the shell moving outward (a departure through the shell).
    start = _radial_state(SHELL, 5.0)
    result = propagate(start, 0.002, SYSTEM, return_shell_km=SHELL_KM)
    assert result.termination is Termination.HORIZON
    assert result.events == ()
    assert _r1(result.final_state) > SHELL


def test_start_inside_the_shell_is_not_propagated() -> None:
    result = propagate(_radial_state(SHELL - 10.0 / L, 5.0), 0.01, SYSTEM, return_shell_km=SHELL_KM)
    assert result.termination is Termination.START_INSIDE_RETURN_SHELL
    assert result.nfev == 0 and result.dense is None


@pytest.mark.parametrize("earth_km", [SYSTEM.earth_radius_km, None])
@pytest.mark.parametrize(("depth_km", "speed"), [(0.01, 10.7), (1.0, 10.7), (0.1, 12.0)])
def test_grazing_dip_below_the_shell_is_a_return(
    depth_km: float, speed: float, earth_km: float | None
) -> None:
    start, t_peri = _start_from_periapsis(depth_km, speed)
    t_ref = _reference_crossing(start, 2.0 * t_peri)
    assert t_ref is not None
    # The dip lies between accepted steps: no step state is inside the shell, so the
    # inward distance event alone would miss it.
    plain = propagate(start, 2.0 * t_peri, SYSTEM)
    assert min(_r1(s) for s in plain.states) > SHELL
    result = propagate(
        start,
        2.0 * t_peri,
        SYSTEM,
        collision=CollisionRadii(earth_km=earth_km, moon_km=SYSTEM.moon_radius_km),
        return_shell_km=SHELL_KM,
    )
    assert result.termination is Termination.EARTH_RETURN_SHELL
    assert result.t_final == pytest.approx(t_ref, abs=1e-8)
    assert _r1(result.final_state) == pytest.approx(SHELL, abs=1e-13)

    # Earth periapses are recorded once even with two Earth-centred surfaces enabled: the
    # plain propagation (Earth surface only) records the same Earth periapses up to t_final.
    def earth_times(r: PropagationResult) -> list[float]:
        return [p.t for p in r.periapses if p.name == "earth_periapsis" and p.t <= result.t_final]

    assert earth_times(result) == pytest.approx(earth_times(plain), abs=1e-12)


def test_periapsis_just_above_the_shell_is_not_a_return() -> None:
    start, t_peri = _start_from_periapsis(-0.01, 10.7)
    result = propagate(start, 2.0 * t_peri, SYSTEM, return_shell_km=SHELL_KM)
    assert result.termination is Termination.HORIZON
    earth = [p for p in result.periapses if p.name == "earth_periapsis"]
    assert len(earth) == 1
    assert _r1(np.array(earth[0].state)) == pytest.approx(SHELL + 0.01 / L, abs=1e-11)


@pytest.mark.parametrize("shell_km", [SYSTEM.earth_radius_km, 6000.0, -1.0, math.nan])
def test_invalid_shell_radius_raises(shell_km: float) -> None:
    with pytest.raises(ValueError):
        propagate(_radial_state(0.1, 0.0), 0.1, SYSTEM, return_shell_km=shell_km)
