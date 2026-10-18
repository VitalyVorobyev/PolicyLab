"""Trans-lunar injection (TLI) states from an idealized circular Earth parking orbit.

M3 departure model (docs/studies/gravity-atlas/m3-feasibility-design.md, section 2):

1. Earth-relative position r = r0 (cos theta, sin theta), theta measured
   counterclockwise from +x (the Earth -> Moon direction) at t0 = 0, when the rotating
   axes coincide with the inertial axes. Since the CR3BP is autonomous, theta is the
   departure epoch relative to the Earth-Moon line.
2. Earth-relative inertial velocity before the burn: posigrade circular,
   v_c = sqrt((1 - mu) / r0) along t_hat = (-sin theta, cos theta).
3. Impulse dv (cos alpha t_hat + sin alpha r_hat), alpha measured from local horizontal
   (positive outward).
4. Rotating-frame state via `units.inertial_to_rotating(state, t=0, mu, origin="earth")`.

The parking orbit only generates the state at t0; it is never propagated, so the
CR3BP non-closure of a "circular" Earth orbit plays no role.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.system import EarthMoonSystem
from phasespace.gravity.units import inertial_to_rotating, velocity_from_km_s

PARKING_ALTITUDE_KM: Final = 185.0
"""Circular parking-orbit altitude (Pavlak 2010, section 3.3.3; Apollo 11 ~186 x 184 km)."""


def parking_radius(
    system: EarthMoonSystem, parking_altitude_km: float = PARKING_ALTITUDE_KM
) -> float:
    """Normalized Earth-centre distance of the parking orbit (mean Earth radius + altitude)."""
    if not (math.isfinite(parking_altitude_km) and parking_altitude_km > 0.0):
        raise ValueError(
            f"parking_altitude_km must be finite and positive, got {parking_altitude_km}"
        )
    return (system.earth_radius_km + parking_altitude_km) / system.length_scale_km


def tli_state(
    phase: float,
    delta_v_km_s: float,
    impulse_angle: float,
    system: EarthMoonSystem,
    *,
    parking_altitude_km: float = PARKING_ALTITUDE_KM,
) -> FloatArray:
    """Rotating-frame state (x, y, vx, vy) at t0 = 0 just after a single impulse.

    Args:
        phase: Departure phase theta in radians, counterclockwise from the +x axis
            (Earth -> Moon) about the Earth.
        delta_v_km_s: Impulse magnitude in km/s (finite, >= 0).
        impulse_angle: Impulse angle alpha in radians from the local horizontal (the
            posigrade tangential direction), positive toward the outward radial.
        system: Earth-Moon constants.
        parking_altitude_km: Circular parking-orbit altitude above the mean Earth radius.
    """
    for label, value in (("phase", phase), ("impulse_angle", impulse_angle)):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value}")
    if not (math.isfinite(delta_v_km_s) and delta_v_km_s >= 0.0):
        raise ValueError(f"delta_v_km_s must be finite and >= 0, got {delta_v_km_s}")
    mu = system.mu
    r0 = parking_radius(system, parking_altitude_km)
    r_hat = np.array([math.cos(phase), math.sin(phase)])
    t_hat = np.array([-math.sin(phase), math.cos(phase)])
    v_circular = math.sqrt((1.0 - mu) / r0)
    dv = velocity_from_km_s(delta_v_km_s, system)
    velocity = v_circular * t_hat + dv * (
        math.cos(impulse_angle) * t_hat + math.sin(impulse_angle) * r_hat
    )
    inertial = np.array([r0 * r_hat[0], r0 * r_hat[1], velocity[0], velocity[1]])
    return inertial_to_rotating(inertial, 0.0, mu, origin="earth")
