"""Unit conversions, system constants and rotating <-> inertial transforms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from jpl_catalog import load_catalog

from phasespace.gravity.reference import JPL_EARTH_MOON
from phasespace.gravity.system import EARTH_MOON, LENGTH_SCALE_CONVENTIONS, EarthMoonSystem
from phasespace.gravity.units import (
    inertial_to_rotating,
    length_from_km,
    length_to_km,
    rotating_to_inertial,
    time_from_days,
    time_from_seconds,
    time_to_days,
    time_to_seconds,
    velocity_from_km_s,
    velocity_to_km_s,
)

CATALOG = load_catalog()
JPL = CATALOG.system()
RNG_SEED = 20261018


def test_default_system_constants_from_de440_and_fact_sheet() -> None:
    # Hand-entered DE440 GMs (km^3/s^2) and the 384 400 km fact-sheet distance.
    gm_earth, gm_moon, length = 398600.435507, 4902.800118, 384400.0
    assert EARTH_MOON.mu == pytest.approx(gm_moon / (gm_earth + gm_moon), rel=1e-15)
    assert EARTH_MOON.length_scale_km == length
    # Circular relative orbit speed sqrt(GM/L) is the velocity unit.
    assert EARTH_MOON.velocity_scale_km_s == pytest.approx(
        math.sqrt((gm_earth + gm_moon) / length), rel=1e-14
    )
    # Default mu agrees with the JPL catalog mass ratio to ~1e-7 relative.
    assert EARTH_MOON.mu == pytest.approx(CATALOG.mu, rel=2e-7)


def test_jpl_catalog_scales_consistent_with_de440_gm() -> None:
    # lunit^3 / tunit^2 = G (m_earth + m_moon); compare with the DE440 GM sum.
    implied_gm = CATALOG.lunit_km**3 / CATALOG.tunit_s**2
    assert implied_gm == pytest.approx(398600.435507 + 4902.800118, rel=1e-8)


def test_system_rejects_invalid_constants() -> None:
    convention = "jpl_periodic_orbits_catalog"
    with pytest.raises(ValueError):
        EarthMoonSystem("bad", 0.0, 1.0, 1.0, 1.0, 1.0, "test", convention)
    with pytest.raises(ValueError):
        EarthMoonSystem("bad", 0.01, -1.0, 1.0, 1.0, 1.0, "test", convention)
    unknown: Any = "mean_earth_moon_distance"
    with pytest.raises(ValueError):
        EarthMoonSystem("bad", 0.01, 1.0, 1.0, 1.0, 1.0, "test", unknown)


def test_length_scale_conventions() -> None:
    assert EARTH_MOON.length_scale_convention == "nssdca_lunar_semimajor_axis"
    assert JPL_EARTH_MOON.length_scale_convention == "jpl_periodic_orbits_catalog"
    assert set(LENGTH_SCALE_CONVENTIONS) == {
        "nssdca_lunar_semimajor_axis",
        "jpl_periodic_orbits_catalog",
    }


def test_scalar_conversions_hand_computed() -> None:
    assert length_to_km(1.0, JPL) == 389703.264829278
    assert length_to_km(0.5, JPL) == pytest.approx(194851.632414639, rel=1e-15)
    assert time_to_seconds(1.0, JPL) == 382981.289129055
    # One full revolution of the primaries (2 pi time units) in days.
    assert time_to_days(2.0 * math.pi, JPL) == pytest.approx(
        2.0 * math.pi * 382981.289129055 / 86400.0, rel=1e-15
    )
    assert time_to_days(2.0 * math.pi, JPL) == pytest.approx(27.851, abs=1e-3)
    assert velocity_to_km_s(1.0, JPL) == pytest.approx(1.0175517078536906, rel=1e-15)


def test_conversions_round_trip_arrays() -> None:
    values = np.random.default_rng(RNG_SEED).uniform(-3.0, 3.0, 50)
    for system in (JPL, EARTH_MOON):
        np.testing.assert_allclose(length_from_km(length_to_km(values, system), system), values)
        np.testing.assert_allclose(
            time_from_seconds(time_to_seconds(values, system), system), values
        )
        np.testing.assert_allclose(time_from_days(time_to_days(values, system), system), values)
        np.testing.assert_allclose(
            velocity_from_km_s(velocity_to_km_s(values, system), system), values
        )


def test_rotating_inertial_round_trip() -> None:
    rng = np.random.default_rng(RNG_SEED)
    states = rng.uniform(-1.5, 1.5, (100, 4))
    times = rng.uniform(-10.0, 10.0, 100)
    for origin in ("barycenter", "earth"):
        inertial = rotating_to_inertial(states, times, CATALOG.mu, origin)
        np.testing.assert_allclose(
            inertial_to_rotating(inertial, times, CATALOG.mu, origin), states, atol=1e-13
        )
        # Rotation preserves distance from the origin.
        offset = 0.0 if origin == "barycenter" else -CATALOG.mu
        np.testing.assert_allclose(
            np.hypot(inertial[:, 0], inertial[:, 1]),
            np.hypot(states[:, 0] - offset, states[:, 1]),
            atol=1e-13,
        )


def test_moon_inertial_velocity_hand_computed() -> None:
    mu = CATALOG.mu
    moon_at_rest = np.array([1.0 - mu, 0.0, 0.0, 0.0])
    # Barycentric inertial: the Moon moves on a circle of radius 1 - mu at unit rate.
    np.testing.assert_allclose(
        rotating_to_inertial(moon_at_rest, 0.0, mu), [1.0 - mu, 0.0, 0.0, 1.0 - mu], atol=1e-15
    )
    # Earth-centered: unit radius and unit speed; after a quarter turn it sits on +y
    # moving in -x.
    np.testing.assert_allclose(
        rotating_to_inertial(moon_at_rest, math.pi / 2.0, mu, "earth"),
        [0.0, 1.0, -1.0, 0.0],
        atol=1e-15,
    )
    speed_km_s = velocity_to_km_s(1.0, JPL)
    assert speed_km_s == pytest.approx(389703.264829278 / 382981.289129055, rel=1e-15)


def test_inertially_fixed_point_has_rotating_velocity() -> None:
    # A point at rest in the inertial frame at (0.5, 0) moves with velocity (0, -0.5) in
    # the rotating frame at t = 0.
    np.testing.assert_allclose(
        inertial_to_rotating(np.array([0.5, 0.0, 0.0, 0.0]), 0.0, CATALOG.mu),
        [0.5, 0.0, 0.0, -0.5],
        atol=1e-16,
    )


def test_circular_parking_orbit_injection_velocity() -> None:
    # Earth-centered circular orbit of radius r (normalized) at t = 0 on the +x side:
    # inertial speed sqrt((1 - mu)/r); rotating-frame vy = sqrt((1 - mu)/r) - r.
    mu = CATALOG.mu
    r = length_from_km(6371.0 + 200.0, JPL)
    v = math.sqrt((1.0 - mu) / r)
    rotating = inertial_to_rotating(np.array([r, 0.0, 0.0, v]), 0.0, mu, "earth")
    np.testing.assert_allclose(rotating, [-mu + r, 0.0, 0.0, v - r], atol=1e-15)
    # sqrt(398600.4 / 6571) = 7.788 km/s for a 200 km altitude two-body circular orbit.
    assert velocity_to_km_s(v, JPL) == pytest.approx(7.788, abs=2e-3)


def test_earth_centered_equals_barycentric_minus_earth_motion() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    states = rng.uniform(-1.0, 1.0, (20, 4))
    t = 1.234
    mu = CATALOG.mu
    bary = rotating_to_inertial(states, t, mu, "barycenter")
    earth = rotating_to_inertial(np.array([-mu, 0.0, 0.0, 0.0]), t, mu, "barycenter")
    np.testing.assert_allclose(
        rotating_to_inertial(states, t, mu, "earth"), bary - earth, atol=1e-14
    )
