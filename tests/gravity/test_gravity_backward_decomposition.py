"""Decomposition of a backward-integration error into a time shift and a transverse rest.

The reference is constructed independently of the code under test: a pure time shift is
produced by a fixed-step RK4 written here (own equations of motion), and a known
transverse offset (position perpendicular to v, velocity perpendicular to the
acceleration, hence orthogonal to f = ds/dt) is added by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.free_return_validation import decompose_state_error
from phasespace.gravity.system import EARTH_MOON

SYSTEM = EARTH_MOON
MU = SYSTEM.mu
L, T, V = SYSTEM.length_scale_km, SYSTEM.time_scale_s, SYSTEM.velocity_scale_km_s


def _rhs(s: FloatArray) -> FloatArray:
    x, y, vx, vy = (float(v) for v in s)
    r1 = math.sqrt((x + MU) ** 2 + y**2)
    r2 = math.sqrt((x - 1.0 + MU) ** 2 + y**2)
    ax = 2.0 * vy + x - (1.0 - MU) * (x + MU) / r1**3 - MU * (x - 1.0 + MU) / r2**3
    ay = -2.0 * vx + y - (1.0 - MU) * y / r1**3 - MU * y / r2**3
    return np.array([vx, vy, ax, ay])


def _shifted(s: FloatArray, delta: float, steps: int = 10) -> FloatArray:
    h = delta / steps
    for _ in range(steps):
        k1 = _rhs(s)
        k2 = _rhs(s + 0.5 * h * k1)
        k3 = _rhs(s + 0.5 * h * k2)
        k4 = _rhs(s + h * k3)
        s = s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return s


# A departure-like state: ~155 km altitude on the -y side of the Earth, ~10.6 normalized
# rotating speed (an ~11 km/s TLI), where |ds/dt| is dominated by the acceleration.
S0 = np.array([-MU + 0.0123, -0.0117, 7.6, 7.4])
DELTA_S = 1e-3  # a 1 ms shift
DELTA = DELTA_S / T


def test_pure_time_shift_is_recovered() -> None:
    ds = _shifted(S0, DELTA) - S0
    diag = decompose_state_error(ds, S0, SYSTEM)
    # First-order exact; the O(delta^2) remainder is ~1e-9 relative here.
    assert diag.along_track_time_s == pytest.approx(DELTA_S, rel=1e-6)
    assert diag.transverse_position_km * 1000.0 < 1e-3  # < 1 mm
    assert diag.transverse_velocity_m_s * 1000.0 < 1e-3  # < 1 um/s


@pytest.mark.parametrize(("pos_m", "vel_mm_s"), [(0.5, 0.3), (2.0, 0.0), (0.0, 5.0)])
def test_known_transverse_offset_is_separated_from_the_time_shift(
    pos_m: float, vel_mm_s: float
) -> None:
    f = _rhs(S0)
    v, a = f[:2], f[2:]
    p_dir = np.array([-v[1], v[0]]) / math.hypot(*v)  # perpendicular to v
    q_dir = np.array([-a[1], a[0]]) / math.hypot(*a)  # perpendicular to a
    offset = np.concatenate([p_dir * pos_m / 1000.0 / L, q_dir * vel_mm_s / 1e6 / V])
    assert abs(float(offset @ f)) < 1e-12 * float(np.linalg.norm(offset) * np.linalg.norm(f))
    ds = _shifted(S0, DELTA) - S0 + offset
    diag = decompose_state_error(ds, S0, SYSTEM)
    assert diag.along_track_time_s == pytest.approx(DELTA_S, rel=1e-6)
    assert diag.transverse_position_km * 1000.0 == pytest.approx(pos_m, abs=1e-3)
    assert diag.transverse_velocity_m_s * 1000.0 == pytest.approx(vel_mm_s, abs=1e-3)
    assert diag.max_abs == pytest.approx(float(np.max(np.abs(ds))), rel=0, abs=0)


def test_max_norm_of_a_1_ms_shift_exceeds_the_superseded_threshold() -> None:
    # Why the max-norm criterion was replaced: a harmless 1 ms shift gives max |ds0| ~ 1e-5.
    ds = _shifted(S0, DELTA) - S0
    assert float(np.max(np.abs(ds))) > 1e-7
