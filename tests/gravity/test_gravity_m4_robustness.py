"""M4 robustness evaluators (protocol 9.1, 9.2) on constructed cases.

No held-out error set or held-out constraint set is evaluated here: the held-out
accessors are only checked to refuse while the protocol is a draft.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from m4_toy import surface, toy_evaluate
from scipy.stats import truncnorm

from phasespace.gravity.m4.config import (
    GATES_PRIMARY,
    GATES_STRESS,
    H_E_NOMINAL,
    H_E_WIDE,
    PROTOCOL,
    T_E,
    CorrectionConfig,
    ProtocolNotFrozenError,
)
from phasespace.gravity.m4.evaluation import evaluate_unit, to_unit
from phasespace.gravity.m4.robustness import (
    ConstraintSet,
    DrawResult,
    ErrorDraw,
    availability,
    calibrate_budget,
    constraint_families,
    error_draws,
    executed_departure,
    gates_error,
    gates_normals,
    held_out_sets,
    p2,
    parking_mean_motion,
    solution_robustness,
    split_indices,
    truncated_normal,
    tuning_sets,
)
from phasespace.gravity.system import EARTH_MOON

C1 = (229.14224099316039, 3.1567704135102614, 0.0)


def test_truncated_normal_matches_scipy_truncnorm() -> None:
    u = np.linspace(0.0, 0.999, 37)
    expected = truncnorm.ppf(u, -3.0, 3.0)
    assert truncated_normal(u, 3.0) == pytest.approx(expected, abs=1e-9)


def test_tuning_error_draws_are_seeded_truncated_and_scaled() -> None:
    draws = error_draws(T_E)
    assert len(draws) == 32
    assert draws == error_draws(T_E)
    for d in draws:
        assert abs(d.magnitude) <= 3 * T_E.sigma_magnitude
        assert abs(d.direction_deg) <= 3 * T_E.sigma_direction_deg
        assert abs(d.timing_s) <= 3 * T_E.sigma_timing_s


def test_held_out_error_draws_refuse_before_the_freeze() -> None:
    for dist in (H_E_NOMINAL, H_E_WIDE):
        with pytest.raises(ProtocolNotFrozenError):
            error_draws(dist, confirm_frozen=True)


def test_parking_orbit_rate_matches_the_protocol_values() -> None:
    n = parking_mean_motion()
    assert n == pytest.approx(446.233, abs=5e-4)  # protocol 9.1
    deg_per_s = math.degrees(n - 1.0) / EARTH_MOON.time_scale_s
    assert deg_per_s == pytest.approx(0.06799, abs=5e-6)
    assert 2 * math.pi / n * EARTH_MOON.time_scale_s / 60.0 == pytest.approx(88.05, abs=0.005)


def test_executed_departure_hand_values() -> None:
    phase, dv, alpha = executed_departure(229.0, 3.15, 1.0, ErrorDraw(1e-3, 0.01, 1.0))
    assert dv == pytest.approx(3.15 * 1.001, rel=1e-15)
    assert alpha == pytest.approx(1.01, rel=1e-15)
    assert phase - 229.0 == pytest.approx(0.0679921, abs=1e-6)


def test_success_at_budget_rules() -> None:
    corrected = DrawResult(False, True, 2.5, True, True, (2.0, 1.5), 2, (5.0, 5.0), 4, 0)
    assert not corrected.survives(0.0)
    assert not corrected.survives(1.0)
    assert corrected.survives(3.0)
    infeasible_after = replace(corrected, corrected_feasible=False, executed_feasible=False)
    assert not infeasible_after.survives(30.0)
    # D24: survival is judged after the executed (perturbed) correction.
    execution_fails = replace(corrected, executed_feasible=False)
    assert not execution_fails.survives(30.0)
    nan2 = (math.nan, math.nan)
    free = DrawResult(True, False, math.inf, False, False, nan2, 3, (500.0, 5.0), 5, 0)
    assert free.survives(0.0)
    assert free.survives(10.0)  # uncorrected feasible counts at every B (default)
    literal = CorrectionConfig(uncorrected_feasible_counts=False)
    assert free.survives(0.0, literal)
    assert not free.survives(10.0, literal)


def test_budget_calibration_rule() -> None:
    survival = {1.0: [0.0, 0.1], 3.0: [0.1, 0.15], 10.0: [0.3, 0.5], 30.0: [0.9, 1.0]}
    assert calibrate_budget(survival) == 10.0  # smallest median in [0.2, 0.8]
    none_in_band = {1.0: [0.0], 3.0: [0.1], 10.0: [0.9], 30.0: [1.0]}
    # Distances to 0.5: 0.5, 0.4, 0.4, 0.5 -> tie between 3 and 10 -> the smaller B.
    assert calibrate_budget(none_in_band) == 3.0


def test_zero_error_draw_needs_no_correction_and_a_small_error_is_corrected() -> None:
    elite = evaluate_unit(to_unit(C1))
    small = ErrorDraw(2e-7, 0.0, 0.0)  # ~0.6 mm/s along the burn
    rob = solution_robustness(elite, [ErrorDraw(0.0, 0.0, 0.0), small])
    assert not rob.singular and rob.condition < 1e3
    zero, corrected = rob.draws
    # Re-propagating the nominal through t_c reproduces its targets (restart only).
    assert max(abs(v) for v in zero.initial_residual_km) < 1.0
    assert zero.converged and zero.delta_v_required_m_s == 0.0 and zero.corrected_feasible
    assert zero.survives(0.0) and zero.survives(1.0)
    # A sub-mm/s execution error moves h_vp by kilometres (M3: ~10 km per mm/s) and the
    # chord MCC brings it back with a tiny correction.
    assert max(abs(v) for v in corrected.initial_residual_km) > 1.0
    assert corrected.converged and corrected.corrected_feasible
    assert 0.0 < corrected.delta_v_required_m_s < 0.1
    assert rob.survival(1.0) == 1.0


def test_gates_error_covariance_matches_hand_built_q() -> None:
    # Commanded burn (3, 4) m/s: V = 5 m/s, e = (0.6, 0.8), in-plane transverse (-0.8, 0.6).
    model = GATES_PRIMARY
    along = 0.010**2 + 25.0 * (5e-3) ** 2  # sigma_r^2 + V^2 sigma_s^2 = 7.25e-4 (m/s)^2
    across = 0.0009**2 + 25.0 * (3e-4) ** 2  # sigma_a^2 + V^2 sigma_p^2 = 3.06e-6 (m/s)^2
    e = np.array([0.6, 0.8])
    t = np.array([-0.8, 0.6])
    q = along * np.outer(e, e) + across * np.outer(t, t)
    # By hand: Q00 = 0.36 a + 0.64 b, Q01 = 0.48 (a - b), Q11 = 0.64 a + 0.36 b.
    by_hand = [[2.629584e-4, 3.465312e-4], [3.465312e-4, 4.651016e-4]]
    assert q == pytest.approx(np.array(by_hand), rel=1e-12)
    z = np.random.default_rng(11).standard_normal((40_000, 2))
    samples = np.array([gates_error(np.array([3.0, 4.0]), row, model) for row in z])
    cov = np.cov(samples.T)
    assert np.abs(samples.mean(axis=0)).max() < 4 * math.sqrt(along / 40_000)
    # Sample covariance within ~4 standard errors (relative SE ~ sqrt(2/n) = 0.7 %).
    assert cov[0, 0] == pytest.approx(q[0, 0], rel=0.03)
    assert cov[1, 1] == pytest.approx(q[1, 1], rel=0.03)
    assert cov[0, 1] == pytest.approx(q[0, 1], rel=0.03)
    along_var = float(np.var(samples @ e))
    across_var = float(np.var(samples @ t))
    assert along_var == pytest.approx(along, rel=0.03)
    assert across_var == pytest.approx(across, rel=0.03)
    with pytest.raises(ValueError, match="commanded"):
        gates_error(np.zeros(2), z[0], model)


def test_gates_normals_are_seeded_and_held_out_sets_refuse() -> None:
    normals = gates_normals(GATES_PRIMARY, T_E)
    assert normals.shape == (32, 2)
    assert np.array_equal(normals, gates_normals(GATES_PRIMARY, T_E))
    expected = np.random.default_rng((GATES_PRIMARY.seed, T_E.seed)).standard_normal((32, 2))
    assert np.array_equal(normals, expected)
    assert GATES_STRESS.sigma_s == 3e-2 and GATES_STRESS.sigma_r_m_s == 0.010
    with pytest.raises(ProtocolNotFrozenError):
        gates_normals(GATES_STRESS, T_E, confirm_frozen=True)
    with pytest.raises(ProtocolNotFrozenError):
        gates_normals(GATES_PRIMARY, H_E_NOMINAL, confirm_frozen=True)


def test_executed_correction_adds_the_gates_error() -> None:
    elite = evaluate_unit(to_unit(C1))
    draws = [ErrorDraw(0.0, 0.0, 0.0), ErrorDraw(2e-7, 0.0, 0.0)]
    exact = solution_robustness(elite, draws, execution=GATES_PRIMARY, normals=np.zeros((2, 2)))
    zero, small = exact.draws
    assert zero.delta_v_required_m_s == 0.0 and zero.executed_feasible  # no burn, no error
    assert small.converged and small.executed_feasible == small.corrected_feasible
    commanded = np.array(small.executed_delta_v_m_s)  # zero normals: executed = commanded
    assert np.linalg.norm(commanded) == pytest.approx(small.delta_v_required_m_s, rel=1e-12)
    normals = np.array([[0.0, 0.0], [1.0, -2.0]])
    rob = solution_robustness(elite, draws, execution=GATES_PRIMARY, normals=normals)
    executed = np.array(rob.draws[1].executed_delta_v_m_s)
    expected = gates_error(commanded, normals[1], GATES_PRIMARY)
    assert executed - commanded == pytest.approx(expected, abs=1e-12)
    # |error| >= sigma_r = 10 mm/s along the burn here, far above the ~0.6 mm/s commanded.
    assert np.linalg.norm(executed - commanded) > 0.009
    with pytest.raises(ValueError, match="normal pair"):
        solution_robustness(elite, draws, execution=GATES_PRIMARY)


def test_infeasible_solutions_are_rejected() -> None:
    bad = toy_evaluate((0.05, 0.5, 0.5))
    with pytest.raises(ValueError, match="feasible"):
        solution_robustness(bad, [])


# --- changed constraints ----------------------------------------------------------------------


def test_constraint_families_ranges_and_split() -> None:
    cfg = PROTOCOL.constraints
    sets = constraint_families()
    assert [c.family for c in sets] == ["F1"] * 32 + ["F2"] * 32 + ["F3"] * 32
    assert sets == constraint_families()
    for c in sets:
        assert c.h_hi_km <= cfg.h_cap_km
        if c.family == "F2":
            assert (c.h_lo_km, c.h_hi_km, c.t_max_days) == (100.0, 30_000.0, 10.0)
        else:
            assert 100.0 <= c.h_lo_km <= 10_000.0 and c.h_lo_km < c.h_hi_km
            assert 6.5 <= c.t_max_days <= 10.0
            width = math.log10(c.h_hi_km / c.h_lo_km)
            assert width <= 2.0 + 1e-12 and (width >= 0.5 - 1e-12 or c.h_hi_km == 30_000.0)
        if c.family == "F1":
            assert c.gamma_c_deg is None and c.width_deg is None
        else:
            assert c.gamma_c_deg is not None and -8.0 <= c.gamma_c_deg <= -5.0
            assert c.width_deg is not None and 0.5 <= c.width_deg <= 2.0
    perms = split_indices()
    for perm in perms.values():
        assert sorted(perm) == list(range(32))
    assert len(tuning_sets()) == 30


def test_held_out_constraint_sets_refuse_before_the_freeze() -> None:
    with pytest.raises(ProtocolNotFrozenError):
        held_out_sets(confirm_frozen=True)


def test_availability_on_constructed_catalog() -> None:
    feasible = toy_evaluate((0.5, surface(0.5, 0.4), 0.4))  # h_p = 10^3 km, t_ret 7.75 d
    elite = replace(feasible, return_flight_path_angle_deg=-6.0)
    infeasible = toy_evaluate((0.05, 0.5, 0.5))
    geometry = ConstraintSet("F1", 0, 500.0, 2000.0, 8.0, None, None)
    too_fast = ConstraintSet("F1", 1, 500.0, 2000.0, 7.0, None, None)
    corridor = ConstraintSet("F2", 0, 100.0, 30_000.0, 10.0, -6.5, 0.6)
    narrow = ConstraintSet("F2", 1, 100.0, 30_000.0, 10.0, -7.0, 0.5)
    sets = [geometry, too_fast, corridor, narrow]
    assert availability([elite], sets) == [True, False, True, False]
    assert availability([infeasible], sets) == [False] * 4
    value, kept = p2([elite], sets, [True, True, True, False])
    assert (value, kept) == (pytest.approx(2 / 3), 3)
    assert math.isnan(p2([elite], sets, [False] * 4)[0])
