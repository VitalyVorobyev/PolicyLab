"""Differential correction and continuation checked against the JPL catalog."""

from __future__ import annotations

import math

import numpy as np
import pytest
from jpl_catalog import JplOrbit, load_catalog

from phasespace.gravity.dynamics import collinear_lagrange_points, state_jacobian
from phasespace.gravity.periodic import (
    CorrectionError,
    continue_lyapunov,
    correct_symmetric_orbit,
    linear_lyapunov_guess,
    monodromy_matrix,
    stability_index,
)
from phasespace.gravity.propagation import propagate
from phasespace.gravity.reference import JPL_EARTH_MOON, JPL_L1_LYAPUNOV_REFERENCE

CATALOG = load_catalog()
MU = CATALOG.mu
L1 = collinear_lagrange_points(MU)[0]

# Agreement with the catalog (normalized units). Same values as the export checks
# (phasespace.gravity.export); rationale there and in docs/studies/gravity-atlas/README.md.
CATALOG_STATE_TOLERANCE = 1e-10
CATALOG_PERIOD_TOLERANCE = 1e-10
CATALOG_JACOBI_TOLERANCE = 1e-10
STABILITY_INDEX_REL_TOLERANCE = 1e-9


def _assert_matches_catalog(
    x0: float, vy0: float, period: float, jacobi: float, o: JplOrbit
) -> None:
    assert x0 == o.x0
    assert vy0 == pytest.approx(o.vy0, abs=CATALOG_STATE_TOLERANCE)
    assert period == pytest.approx(o.period, abs=CATALOG_PERIOD_TOLERANCE)
    assert jacobi == pytest.approx(o.jacobi, abs=CATALOG_JACOBI_TOLERANCE)


@pytest.mark.parametrize("orbit", CATALOG.orbits, ids=lambda o: f"row{o.row_index}")
@pytest.mark.parametrize("perturbation", [0.01, -0.03])
def test_correction_from_perturbed_guess_converges_to_jpl(
    orbit: JplOrbit, perturbation: float
) -> None:
    corrected = correct_symmetric_orbit(MU, orbit.x0, orbit.vy0 * (1.0 + perturbation))
    _assert_matches_catalog(corrected.x0, corrected.vy0, corrected.period, corrected.jacobi, orbit)
    assert corrected.half_period_abs_vx < 1e-11
    assert corrected.iterations > 1


@pytest.mark.parametrize("orbit", CATALOG.orbits, ids=lambda o: f"row{o.row_index}")
def test_continuation_without_catalog_velocity_reaches_jpl(orbit: JplOrbit) -> None:
    corrected = continue_lyapunov(MU, 1, orbit.x0 - L1, steps=10)
    assert corrected.x0 == pytest.approx(orbit.x0, abs=1e-15)
    assert corrected.vy0 == pytest.approx(orbit.vy0, abs=CATALOG_STATE_TOLERANCE)
    assert corrected.period == pytest.approx(orbit.period, abs=CATALOG_PERIOD_TOLERANCE)
    assert corrected.jacobi == pytest.approx(orbit.jacobi, abs=CATALOG_JACOBI_TOLERANCE)


def test_linear_frequency_matches_jacobian_eigenvalues() -> None:
    # The in-plane center eigenvalues at L1 are +-i nu; T_linear = 2 pi / nu.
    eigenvalues = np.linalg.eigvals(state_jacobian(np.array([L1, 0.0, 0.0, 0.0]), MU))
    nu = max(abs(float(e.imag)) for e in eigenvalues if abs(e.real) < 1e-9)
    guess = linear_lyapunov_guess(MU, 1, -1e-4)
    assert guess.period == pytest.approx(2.0 * math.pi / nu, rel=1e-12)


def test_small_orbit_period_approaches_linear_period() -> None:
    guess = linear_lyapunov_guess(MU, 1, -1e-4)
    corrected = correct_symmetric_orbit(MU, guess.x0, guess.vy0)
    # Amplitude corrections to the period are second order in the amplitude.
    assert corrected.period == pytest.approx(guess.period, abs=1e-5)
    assert corrected.vy0 == pytest.approx(guess.vy0, rel=1e-2)


def test_l2_lyapunov_orbit_is_periodic() -> None:
    corrected = continue_lyapunov(MU, 2, 0.01, steps=4)
    result = propagate(corrected.state0, corrected.period, JPL_EARTH_MOON)
    assert np.max(np.abs(result.final_state - corrected.state0)) < 1e-9


def test_guess_outside_basin_is_reported() -> None:
    # The linear guess is far from the row-2400 orbit; plain correction from it flips
    # the sign of vy0 and must fail loudly rather than return a different orbit.
    orbit = CATALOG.orbits[0]
    guess = linear_lyapunov_guess(MU, 1, orbit.x0 - L1)
    with pytest.raises(CorrectionError):
        correct_symmetric_orbit(MU, orbit.x0, guess.vy0)


def test_reference_module_matches_fixture() -> None:
    ref = JPL_L1_LYAPUNOV_REFERENCE
    matches = [o for o in CATALOG.orbits if o.row_index == ref.row_index]
    assert len(matches) == 1
    orbit = matches[0]
    assert (ref.x0, ref.vy0, ref.period, ref.jacobi, ref.stability_index) == (
        orbit.x0,
        orbit.vy0,
        orbit.period,
        orbit.jacobi,
        orbit.stability,
    )
    assert JPL_EARTH_MOON.mu == CATALOG.mu
    assert JPL_EARTH_MOON.length_scale_km == CATALOG.lunit_km
    assert JPL_EARTH_MOON.time_scale_s == CATALOG.tunit_s
    assert JPL_EARTH_MOON.moon_radius_km == CATALOG.radius_secondary_km


@pytest.mark.parametrize("orbit", CATALOG.orbits, ids=lambda o: f"row{o.row_index}")
def test_monodromy_stability_index_matches_jpl(orbit: JplOrbit) -> None:
    # Catalog state and period only (no corrector): nu = (lambda + 1/lambda) / 2.
    state0 = np.array([orbit.x0, orbit.y0, orbit.vx0, orbit.vy0])
    phi = monodromy_matrix(MU, state0, orbit.period)
    lambda_max, nu = stability_index(phi)
    assert nu == pytest.approx(orbit.stability, rel=STABILITY_INDEX_REL_TOLERANCE)
    assert lambda_max == pytest.approx(
        orbit.stability + math.sqrt(orbit.stability**2 - 1.0), rel=STABILITY_INDEX_REL_TOLERANCE
    )
    # The flow is divergence-free (trace of the Jacobian is 0), so det(phi) = 1.
    assert np.linalg.det(phi) == pytest.approx(1.0, abs=1e-8)
    eigenvalues = sorted(np.linalg.eigvals(phi), key=abs)
    # Reciprocal pair and a double unit eigenvalue.
    assert abs(eigenvalues[0]) * abs(eigenvalues[-1]) == pytest.approx(1.0, abs=1e-8)
    assert abs(eigenvalues[1] - 1.0) < 1e-5 and abs(eigenvalues[2] - 1.0) < 1e-5
