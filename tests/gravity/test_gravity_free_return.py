"""Free-return predicates (design section 3): constructed results and propagated cases."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from phasespace.gravity.dynamics import primary_distances
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    Outcome,
    classify,
    earth_angular_momentum_sign,
    evaluate,
    flight_path_angle_deg,
    osculating_perigee_radius,
    propagate_departure,
    targeting_residual,
)
from phasespace.gravity.propagation import (
    CollisionRadii,
    EventRecord,
    PropagationResult,
    Termination,
)
from phasespace.gravity.system import EARTH_MOON, GM_EARTH_KM3_S2

SYSTEM = EARTH_MOON
MU = SYSTEM.mu
L = SYSTEM.length_scale_km
DEF = DEFAULT_DEFINITION
SHELL = DEF.return_shell_km / L
MOON_X = 1.0 - MU

# --- constructed PropagationResults ------------------------------------------------------


def _moon_periapsis(t: float, altitude_km: float, sense: float = 1.0) -> EventRecord:
    r = (SYSTEM.moon_radius_km + altitude_km) / L
    return EventRecord("moon_periapsis", t, (MOON_X - r, 0.0, 0.0, -sense * 1.0), 1.0)


def _earth_periapsis(t: float, altitude_km: float, speed: float) -> EventRecord:
    r = (SYSTEM.earth_radius_km + altitude_km) / L
    return EventRecord("earth_periapsis", t, (-MU - r, 0.0, 0.0, -speed), 1.0)


def _shell_state(inbound: bool = True) -> np.ndarray:
    # On the shell, on the -x side of the Earth, moving with radial speed -/+1 (inbound/out).
    return np.array([-MU - SHELL, 0.0, 1.0 if inbound else -1.0, -10.0])


def _result(
    termination: Termination,
    t_final: float,
    final: np.ndarray,
    periapses: tuple[EventRecord, ...] = (),
) -> PropagationResult:
    start = np.array([-MU - 0.0171, 0.0, 0.0, -10.5])
    return PropagationResult(
        system=SYSTEM,
        solver=DEF.solver,
        collision=CollisionRadii.of(SYSTEM),
        return_shell_km=DEF.return_shell_km,
        duration=DEF.duration,
        t=np.array([0.0, t_final]),
        states=np.vstack([start, final]),
        termination=termination,
        events=(),
        periapses=periapses,
        nfev=10,
        message="constructed",
        dense=None,
    )


def test_constructed_feasible_return() -> None:
    peri = _moon_periapsis(0.8, 1000.0)
    rec = classify(_result(Termination.EARTH_RETURN_SHELL, 1.5, _shell_state(), (peri,)))
    assert rec.outcome is Outcome.FEASIBLE and rec.feasible
    assert rec.perilune_altitude_km == pytest.approx(1000.0, abs=1e-9)
    assert rec.return_time_days == pytest.approx(1.5 * SYSTEM.time_scale_s / 86400.0)
    assert rec.perigee_source == "return_shell"
    assert rec.return_flight_path_angle_deg < 0.0


@pytest.mark.parametrize(
    ("altitude_km", "outcome"),
    [
        # Just inside / outside the edges (the km round trip is not exact at the edge).
        (100.0 + 1e-6, Outcome.FEASIBLE),
        (30_000.0 - 1e-6, Outcome.FEASIBLE),
        (100.0 - 1e-6, Outcome.PERILUNE_OUT_OF_BAND),
        (30_000.0 + 1e-6, Outcome.PERILUNE_OUT_OF_BAND),
        (50_000.0, Outcome.PERILUNE_OUT_OF_BAND),
    ],
)
def test_perilune_band_edges(altitude_km: float, outcome: Outcome) -> None:
    peri = _moon_periapsis(0.8, altitude_km)
    rec = classify(_result(Termination.EARTH_RETURN_SHELL, 1.5, _shell_state(), (peri,)))
    assert rec.outcome is outcome


def test_closest_lunar_periapsis_before_return_decides() -> None:
    # A far periapsis then a close in-band one: the closest counts and both are recorded.
    peris = (_moon_periapsis(0.2, 200_000.0), _moon_periapsis(0.8, 500.0, sense=-1.0))
    rec = classify(_result(Termination.EARTH_RETURN_SHELL, 1.5, _shell_state(), peris))
    assert rec.outcome is Outcome.FEASIBLE
    assert rec.perilune_altitude_km == pytest.approx(500.0, abs=1e-9)
    assert rec.lunar_encounter_time_days == pytest.approx(0.8 * SYSTEM.time_scale_s / 86400.0)
    assert rec.moon_periapses_in_band == 1
    assert rec.lunar_passage_sense == -1


def test_return_before_lunar_encounter() -> None:
    # The only lunar periapsis is after the return time: it must not count.
    late = _moon_periapsis(2.0, 1000.0)
    rec = classify(_result(Termination.EARTH_RETURN_SHELL, 1.5, _shell_state(), (late,)))
    assert rec.outcome is Outcome.RETURN_BEFORE_LUNAR_ENCOUNTER
    assert math.isnan(rec.perilune_altitude_km)
    assert targeting_residual(rec) is None


def test_outbound_state_on_shell_is_not_a_return() -> None:
    peri = _moon_periapsis(0.8, 1000.0)
    rec = classify(
        _result(Termination.EARTH_RETURN_SHELL, 1.5, _shell_state(inbound=False), (peri,))
    )
    assert rec.outcome is Outcome.NUMERICAL_FAILURE


@pytest.mark.parametrize(
    ("termination", "outcome"),
    [
        (Termination.MOON_COLLISION, Outcome.MOON_COLLISION),
        (Termination.EARTH_COLLISION, Outcome.EARTH_COLLISION),
        (Termination.HORIZON, Outcome.NO_RETURN_WITHIN_T_MAX),
        (Termination.NON_FINITE, Outcome.NUMERICAL_FAILURE),
        (Termination.EVALUATION_LIMIT, Outcome.NUMERICAL_FAILURE),
        (Termination.SOLVER_FAILURE, Outcome.NUMERICAL_FAILURE),
        (Termination.START_INSIDE_RETURN_SHELL, Outcome.INVALID_START),
    ],
)
def test_non_return_terminations(termination: Termination, outcome: Outcome) -> None:
    peri = _moon_periapsis(0.8, 1000.0)
    rec = classify(_result(termination, 1.5, _shell_state(), (peri,)))
    assert rec.outcome is outcome
    assert not rec.feasible
    assert math.isnan(rec.return_time_days)


def test_vacuum_perigee_from_post_lunar_earth_periapsis_without_return() -> None:
    # A miss of the shell: h_vp comes from the first Earth periapsis after the perilune
    # (a tangential state, so the osculating perigee is that periapsis when v > v_circ).
    peris = (_moon_periapsis(0.8, 1000.0), _earth_periapsis(1.4, 300.0, 10.5))
    final = np.array([-MU + 0.1, 0.0, 0.0, 1.0])
    rec = classify(_result(Termination.HORIZON, 2.0, final, peris))
    assert rec.outcome is Outcome.NO_RETURN_WITHIN_T_MAX
    assert rec.perigee_source == "earth_periapsis"
    assert rec.vacuum_perigee_altitude_km == pytest.approx(300.0, abs=1e-6)
    f = targeting_residual(rec)
    assert f is not None
    assert f == pytest.approx((0.0, (300.0 - 38.0) / 10.0), abs=1e-6)


def test_classify_rejects_result_without_the_return_shell() -> None:
    result = _result(Termination.HORIZON, 1.0, _shell_state())
    other = FreeReturnDefinition(return_shell_altitude_km=150.0)
    with pytest.raises(ValueError):
        classify(result, other)


def test_definition_rejects_shell_above_parking_orbit() -> None:
    with pytest.raises(ValueError):
        FreeReturnDefinition(return_shell_altitude_km=200.0)


# --- two-body formulas, independent forms ---------------------------------------------


def test_vacuum_perigee_of_apollo_11_entry_state() -> None:
    # Apollo 11 entry interface (Mission Report Table 7-VII): 11.032 km/s at -6.48 deg at
    # 400 000 ft. Design section 4: vacuum perigee 38.5 km (Apollo targeted 37.8 km). The
    # reference here solves 2 E rp^2 + 2 GM rp - h^2 = 0 (vis-viva at perigee) in km.
    r, v, gamma = 6371.0 + 121.92, 11.032, math.radians(-6.48)
    gm = GM_EARTH_KM3_S2
    h = r * v * math.cos(gamma)
    energy = 0.5 * v * v - gm / r
    rp_ref = (-gm + math.sqrt(gm * gm + 2.0 * energy * h * h)) / (2.0 * energy)
    state = np.array([r, 0.0, v * math.sin(gamma), v * math.cos(gamma)]) / np.array(
        [L, L, SYSTEM.velocity_scale_km_s, SYSTEM.velocity_scale_km_s]
    )
    rp = osculating_perigee_radius(state, 1.0 - MU) * L
    assert rp == pytest.approx(rp_ref, rel=1e-12)
    assert rp - 6371.0 == pytest.approx(38.5, abs=0.05)
    assert flight_path_angle_deg(state) == pytest.approx(-6.48, abs=1e-12)


def test_vacuum_perigee_of_circular_and_apogee_states() -> None:
    gm = 1.0 - MU
    r = 0.02
    vc = math.sqrt(gm / r)
    assert osculating_perigee_radius(np.array([r, 0.0, 0.0, vc]), gm) == pytest.approx(r)
    # At apogee of an ellipse with perigee rp: v_a = sqrt(2 gm rp / (r (r + rp))).
    rp = 0.017
    va = math.sqrt(2.0 * gm * rp / (r * (r + rp)))
    assert osculating_perigee_radius(np.array([0.0, r, -va, 0.0]), gm) == pytest.approx(rp)


# --- propagated cases --------------------------------------------------------------------


def test_propagated_free_return_pilot_sample() -> None:
    # Pilot sample of the design (h_p ~ 293 km, t_ret ~ 5.8 d).
    result = propagate_departure(227.63, 3.1628)
    rec = classify(result, phase_deg=227.63, delta_v_km_s=3.1628, impulse_angle_deg=0.0)
    assert rec.outcome is Outcome.FEASIBLE
    assert rec.perilune_altitude_km == pytest.approx(293.0, abs=1.0)
    assert 5.7 < rec.return_time_days < 5.9
    assert rec.lunar_encounter_time_days < rec.return_time_days
    assert result.termination is Termination.EARTH_RETURN_SHELL
    final = result.final_state
    r1, _ = primary_distances(final, MU)
    assert float(r1) == pytest.approx(SHELL, abs=1e-13)
    # The departure itself (t0 apsis outside the shell) is not a return.
    assert result.t_final > 1.0
    # Perilune from the event equals the minimum of r2 over dense samples to < 10 m.
    times = np.linspace(0.0, result.t_final, 20_001)
    _, r2 = primary_distances(result.sample(times), MU)
    dense_alt = float(np.min(r2)) * L - SYSTEM.moon_radius_km
    assert dense_alt >= rec.perilune_altitude_km - 1e-6
    assert dense_alt == pytest.approx(rec.perilune_altitude_km, abs=0.01)
    assert rec.return_speed_km_s == pytest.approx(11.0, abs=0.1)
    assert rec.jacobi_max_abs_drift_steps < 1e-9


def test_propagated_moon_collision() -> None:
    rec = evaluate(225.0, 3.17)
    assert rec.outcome is Outcome.MOON_COLLISION
    assert rec.termination is Termination.MOON_COLLISION


def test_propagated_non_return() -> None:
    rec = evaluate(225.0, 3.15)
    assert rec.outcome is Outcome.NO_RETURN_WITHIN_T_MAX
    assert rec.t_final_days == pytest.approx(DEF.t_max_days, rel=1e-12)


def test_propagated_return_before_lunar_encounter() -> None:
    # A retrograde 0.1 km/s burn at theta = 90 deg (moving away from the Moon) drops the
    # perigee below the shell: the shell is crossed within one revolution, before any
    # Moon periapsis (r2 grows from theta = 90 to 180 deg and the crossing comes first).
    rec = evaluate(90.0, 0.1, 180.0)
    assert rec.termination is Termination.EARTH_RETURN_SHELL
    assert rec.outcome is Outcome.RETURN_BEFORE_LUNAR_ENCOUNTER
    assert rec.t_final_days < 0.05


def test_inward_impulse_crossing_before_the_moon_is_rejected() -> None:
    # Near-Earth transient: a strongly inward impulse crosses the shell long before any
    # lunar encounter; it must not be counted as a return.
    rec = evaluate(227.63, 3.1628, -80.0)
    assert rec.termination is Termination.EARTH_RETURN_SHELL
    assert rec.outcome in (Outcome.RETURN_BEFORE_LUNAR_ENCOUNTER, Outcome.PERILUNE_OUT_OF_BAND)
    assert rec.t_final_days < 0.1


def test_targeting_perigee_uses_the_first_post_lunar_earth_approach() -> None:
    # A return whose first post-lunar Earth pass stays above the shell: the targeting
    # value comes from that pass (continuity), the shell value is recorded separately.
    peris = (_moon_periapsis(0.8, 1000.0), _earth_periapsis(1.2, 500.0, 10.5))
    rec = classify(_result(Termination.EARTH_RETURN_SHELL, 1.5, _shell_state(), peris))
    assert rec.outcome is Outcome.FEASIBLE
    assert rec.perigee_source == "earth_periapsis"
    assert rec.vacuum_perigee_altitude_km == pytest.approx(500.0, abs=1e-6)
    assert math.isfinite(rec.return_vacuum_perigee_altitude_km)
    assert rec.return_vacuum_perigee_altitude_km != rec.vacuum_perigee_altitude_km
    assert rec.earth_periapses_after_encounter == 1


def test_classify_rejects_a_different_horizon() -> None:
    result = _result(Termination.HORIZON, 1.0, _shell_state())
    longer = replace(result, duration=3.0 * result.duration)
    with pytest.raises(ValueError):
        classify(longer)


def test_flight_path_angle_of_a_radial_state_is_clamped() -> None:
    assert flight_path_angle_deg(np.array([0.1, 0.0, -3.0, 0.0])) == -90.0
    assert flight_path_angle_deg(np.array([0.1 / 3.0, 0.1, -1.0, -3.0])) == pytest.approx(-90.0)


@pytest.mark.parametrize(
    ("vy_rot", "expected"),
    [
        (1.0, 1),  # posigrade in both frames
        (-5.0, -1),  # retrograde in both frames
        # Slightly negative in the rotating frame but posigrade inertially: the frame
        # rotation adds w x r = +r (here 0.017) to vy, so vy_in = -0.01 + 0.017 > 0.
        (-0.01, 1),
    ],
)
def test_entry_sense_uses_the_inertial_frame(vy_rot: float, expected: int) -> None:
    # Earth-relative position (r, 0) at t = 0; inertial vy = vy_rot + r (hand formula).
    r = 0.017
    state = np.array([-MU + r, 0.0, 0.0, vy_rot])
    assert earth_angular_momentum_sign(state, 0.0, MU) == expected
    # Rotating the frame (t != 0) does not change the sign.
    assert earth_angular_momentum_sign(state, 1.234, MU) == expected
