"""TLI departure state (design section 2), checked against hand-written formulas in km."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.gravity.departure import PARKING_ALTITUDE_KM, parking_radius, tli_state
from phasespace.gravity.free_return import ENTRY_INTERFACE_ALTITUDE_KM
from phasespace.gravity.system import EARTH_MOON, GM_EARTH_KM3_S2
from phasespace.gravity.units import rotating_to_inertial

SYSTEM = EARTH_MOON
L = SYSTEM.length_scale_km
V = SYSTEM.velocity_scale_km_s
R0_KM = 6371.0 + 185.0


def _hand_rotating_state(theta: float, dv: float, alpha: float) -> np.ndarray:
    """Rotating-frame state written out by hand, in km and km/s, then normalized.

    Earth-relative inertial velocity: v_c t_hat + dv (cos a t_hat + sin a r_hat) with
    v_c = sqrt(GM_earth / r0); rotating velocity v_rot = v_in - w x r_rel with
    w x r_rel = (-y, x) * (rate) and rate = V / L per second.
    """
    vc = math.sqrt(GM_EARTH_KM3_S2 / R0_KM)
    rx, ry = R0_KM * math.cos(theta), R0_KM * math.sin(theta)
    tx, ty = -math.sin(theta), math.cos(theta)
    vx = (vc + dv * math.cos(alpha)) * tx + dv * math.sin(alpha) * math.cos(theta)
    vy = (vc + dv * math.cos(alpha)) * ty + dv * math.sin(alpha) * math.sin(theta)
    rate = V / L  # rad/s
    vx_rot, vy_rot = vx + rate * ry, vy - rate * rx
    return np.array([rx / L - SYSTEM.mu, ry / L, vx_rot / V, vy_rot / V])


@pytest.mark.parametrize(
    ("theta_deg", "dv", "alpha_deg"),
    [(0.0, 3.0, 0.0), (227.63, 3.1628, 0.0), (90.0, 3.25, 7.5), (300.0, 3.1, -10.0)],
)
def test_tli_state_matches_hand_formula(theta_deg: float, dv: float, alpha_deg: float) -> None:
    theta, alpha = math.radians(theta_deg), math.radians(alpha_deg)
    state = tli_state(theta, dv, alpha, SYSTEM)
    np.testing.assert_allclose(state, _hand_rotating_state(theta, dv, alpha), rtol=0, atol=1e-13)


@pytest.mark.parametrize("theta_deg", [0.0, 45.0, 227.63, 359.0])
def test_tangential_burn_adds_to_circular_speed(theta_deg: float) -> None:
    theta = math.radians(theta_deg)
    dv = 3.15
    inertial = rotating_to_inertial(tli_state(theta, dv, 0.0, SYSTEM), 0.0, SYSTEM.mu, "earth")
    speed_km_s = math.hypot(float(inertial[2]), float(inertial[3])) * V
    assert speed_km_s == pytest.approx(math.sqrt(GM_EARTH_KM3_S2 / R0_KM) + dv, abs=1e-12)
    # Position at r0 along theta; velocity along t_hat (zero radial component).
    np.testing.assert_allclose(
        inertial[:2] * L, [R0_KM * math.cos(theta), R0_KM * math.sin(theta)], atol=1e-8
    )
    assert float(inertial[0] * inertial[2] + inertial[1] * inertial[3]) == pytest.approx(
        0.0, abs=1e-15
    )


def test_tangential_departure_is_an_earth_apsis_in_the_rotating_frame() -> None:
    # (r - r_E) . v_rot = (r - r_E) . v_in because (r - r_E) . (w x (r - r_E)) = 0.
    x, y, vx, vy = tli_state(math.radians(123.0), 3.2, 0.0, SYSTEM)
    assert (x + SYSTEM.mu) * vx + y * vy == pytest.approx(0.0, abs=1e-15)


def test_parking_orbit_lies_outside_the_return_shell() -> None:
    # The departure can never be counted as a return: it starts outside the shell.
    assert PARKING_ALTITUDE_KM > ENTRY_INTERFACE_ALTITUDE_KM
    assert parking_radius(SYSTEM) * L == pytest.approx(R0_KM, abs=1e-9)
    x, y, _, _ = tli_state(0.3, 3.1, -0.17, SYSTEM)
    assert math.hypot(x + SYSTEM.mu, y) * L > SYSTEM.earth_radius_km + ENTRY_INTERFACE_ALTITUDE_KM


@pytest.mark.parametrize(
    ("phase", "dv", "alpha"),
    [(math.nan, 3.0, 0.0), (0.0, -0.1, 0.0), (0.0, math.inf, 0.0), (0.0, 3.0, math.inf)],
)
def test_invalid_inputs_raise(phase: float, dv: float, alpha: float) -> None:
    with pytest.raises(ValueError):
        tli_state(phase, dv, alpha, SYSTEM)
