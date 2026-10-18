"""Unit conversions and planar rotating <-> inertial state transforms.

Inertial frames used here share their axes with the rotating frame at normalized time
t = 0; the rotating frame then turns by angle t (unit angular rate, counterclockwise
about +z). Rotating-frame velocities are relative to the rotating frame.

For a position r and rotating-frame velocity v (both in rotating axes), the inertial
velocity expressed in inertial axes is R(t) (v + w x r) with w x r = (-y, x), where
R(t) is the rotation by angle t and r is measured from the chosen origin.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from phasespace.gravity.dynamics import STATE_DIM, FloatArray
from phasespace.gravity.system import EarthMoonSystem

SECONDS_PER_DAY = 86400.0

type Origin = Literal["barycenter", "earth"]


@overload
def length_to_km(value: float, system: EarthMoonSystem) -> float: ...
@overload
def length_to_km(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def length_to_km(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """Normalized length -> km."""
    return value * system.length_scale_km


@overload
def length_from_km(value: float, system: EarthMoonSystem) -> float: ...
@overload
def length_from_km(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def length_from_km(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """km -> normalized length."""
    return value / system.length_scale_km


@overload
def time_to_seconds(value: float, system: EarthMoonSystem) -> float: ...
@overload
def time_to_seconds(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def time_to_seconds(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """Normalized time -> seconds."""
    return value * system.time_scale_s


@overload
def time_from_seconds(value: float, system: EarthMoonSystem) -> float: ...
@overload
def time_from_seconds(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def time_from_seconds(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """Seconds -> normalized time."""
    return value / system.time_scale_s


@overload
def time_to_days(value: float, system: EarthMoonSystem) -> float: ...
@overload
def time_to_days(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def time_to_days(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """Normalized time -> days of 86 400 s."""
    return value * (system.time_scale_s / SECONDS_PER_DAY)


@overload
def time_from_days(value: float, system: EarthMoonSystem) -> float: ...
@overload
def time_from_days(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def time_from_days(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """Days of 86 400 s -> normalized time."""
    return value * (SECONDS_PER_DAY / system.time_scale_s)


@overload
def velocity_to_km_s(value: float, system: EarthMoonSystem) -> float: ...
@overload
def velocity_to_km_s(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def velocity_to_km_s(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """Normalized velocity -> km/s."""
    return value * system.velocity_scale_km_s


@overload
def velocity_from_km_s(value: float, system: EarthMoonSystem) -> float: ...
@overload
def velocity_from_km_s(value: FloatArray, system: EarthMoonSystem) -> FloatArray: ...
def velocity_from_km_s(value: float | FloatArray, system: EarthMoonSystem) -> float | FloatArray:
    """km/s -> normalized velocity."""
    return value / system.velocity_scale_km_s


def _origin_x(origin: Origin, mu: float) -> float:
    if origin == "barycenter":
        return 0.0
    if origin == "earth":
        return -mu
    raise ValueError(f"unknown origin {origin!r}")


def _rotate(vx: FloatArray, vy: FloatArray, angle: FloatArray) -> tuple[FloatArray, FloatArray]:
    c, s = np.cos(angle), np.sin(angle)
    return c * vx - s * vy, s * vx + c * vy


def rotating_to_inertial(
    states: npt.ArrayLike, t: npt.ArrayLike, mu: float, origin: Origin = "barycenter"
) -> FloatArray:
    """Convert rotating-frame states (..., 4) at normalized times t to inertial states.

    `origin` selects the origin of the non-rotating frame: the barycenter (a true
    inertial frame in the CR3BP) or the Earth's center. The Earth-centered frame is
    non-rotating but translates with the Earth, which circles the barycenter, so it is
    only pseudo-inertial; positions and velocities are then relative to the Earth, as
    used for Earth-centered parking-orbit initializations. All values stay normalized.
    """
    s = np.asarray(states, dtype=np.float64)
    if s.shape[-1:] != (STATE_DIM,):
        raise ValueError(f"states must have trailing dimension {STATE_DIM}, got {s.shape}")
    angle = np.asarray(t, dtype=np.float64)
    x = s[..., 0] - _origin_x(origin, mu)
    y = s[..., 1]
    vx_in = s[..., 2] - y
    vy_in = s[..., 3] + x
    px, py = _rotate(x, y, angle)
    qx, qy = _rotate(vx_in, vy_in, angle)
    return np.stack(np.broadcast_arrays(px, py, qx, qy), axis=-1)


def inertial_to_rotating(
    states: npt.ArrayLike, t: npt.ArrayLike, mu: float, origin: Origin = "barycenter"
) -> FloatArray:
    """Inverse of `rotating_to_inertial`."""
    s = np.asarray(states, dtype=np.float64)
    if s.shape[-1:] != (STATE_DIM,):
        raise ValueError(f"states must have trailing dimension {STATE_DIM}, got {s.shape}")
    angle = -np.asarray(t, dtype=np.float64)
    x, y = _rotate(s[..., 0], s[..., 1], angle)
    vx_in, vy_in = _rotate(s[..., 2], s[..., 3], angle)
    vx = vx_in + y
    vy = vy_in - x
    return np.stack(np.broadcast_arrays(x + _origin_x(origin, mu), y, vx, vy), axis=-1)
