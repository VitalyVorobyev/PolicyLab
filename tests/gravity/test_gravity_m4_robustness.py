"""M4 robustness evaluators (protocol 9.1, 9.2) on constructed cases.

No held-out error set or held-out constraint set is evaluated here: the held-out
accessors are checked to refuse under a draft protocol (a temporary file) and without
confirmation under the frozen one, and to return correctly shaped data once allowed.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from m4_protocol import protocol_file
from m4_toy import surface, toy_evaluate
from scipy.stats import truncnorm

from phasespace.gravity.m4.config import (
    GATES_PRIMARY,
    GATES_STRESS,
    H_E_NOMINAL,
    H_E_WIDE,
    PROTOCOL,
    PROTOCOL_PATH,
    T_E,
    ProtocolNotFrozenError,
)
from phasespace.gravity.m4.evaluation import evaluate_unit, to_unit
from phasespace.gravity.m4.robustness import (
    ConstraintSet,
    DrawResult,
    ErrorDraw,
    Execution,
    availability,
    calibrate_budget,
    constraint_families,
    error_draws,
    executed_departure,
    execution_models,
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


def test_held_out_error_draws_refuse_before_the_freeze(tmp_path: Path) -> None:
    draft = protocol_file(tmp_path, "draft")
    frozen = protocol_file(tmp_path, "frozen")
    for dist in (H_E_NOMINAL, H_E_WIDE):
        with pytest.raises(ProtocolNotFrozenError, match="draft"):
            error_draws(dist, confirm_frozen=True, protocol_path=draft)
        for path in (frozen, PROTOCOL_PATH):
            with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
                error_draws(dist, protocol_path=path)
        draws = error_draws(dist, confirm_frozen=True, protocol_path=frozen)
        assert len(draws) == dist.draws == 50
        assert draws == error_draws(dist, confirm_frozen=True)
    # The tuning set needs no freeze, whatever the protocol says.
    assert error_draws(T_E, protocol_path=draft) == error_draws(T_E)


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


def _draw_result(
    uncorrected: bool, commanded: tuple[float, float], executed: bool | None
) -> DrawResult:
    nan2 = (math.nan, math.nan)
    return DrawResult(
        uncorrected_feasible=uncorrected,
        commanded_m_s=commanded,
        executions=(Execution("gates_primary", commanded, executed),),
        converged=False,
        diverged=executed is None,
        commanded_feasible=None,
        iterations=3,
        initial_residual_km=nan2,
        propagations=6,
        nfev=0,
    )


def test_success_at_budget_rules() -> None:
    # D25: commanded |dv| <= B and feasible after the executed correction; chord
    # convergence plays no part (converged=False in every case here).
    corrected = _draw_result(False, (2.0, 1.5), True)  # |dv| = 2.5 m/s
    assert corrected.commanded_m_s_norm == pytest.approx(2.5)
    assert not corrected.survives(0.0)
    assert not corrected.survives(1.0)
    assert corrected.survives(2.5) and corrected.survives(3.0)
    execution_fails = _draw_result(False, (2.0, 1.5), False)
    assert not any(execution_fails.survives(b) for b in (0.0, 1.0, 3.0, 10.0, 30.0))
    diverged = _draw_result(False, (150.0, 0.0), None)
    assert not any(diverged.survives(b) for b in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0))
    # A draw feasible without correction survives at every B, whatever its MCC does.
    free = _draw_result(True, (20.0, 0.0), False)
    assert all(free.survives(b) for b in (0.0, 1.0, 3.0, 10.0, 30.0))
    with pytest.raises(ValueError, match="defined"):
        corrected.survives(101.0)
    with pytest.raises(ValueError, match="defined"):
        corrected.survives(-1.0)


def test_survival_is_monotone_in_budget_for_every_outcome_combination() -> None:
    budgets = np.linspace(0.0, 100.0, 401)
    for uncorrected in (False, True):
        for norm in (0.0, 0.7, 2.5, 9.9, 30.0, 64.0, 150.0):
            for executed in (False, True, None):
                if executed is None and norm <= 100.0:
                    continue
                d = _draw_result(uncorrected, (norm, 0.0), executed)
                s = [d.survives(float(b)) for b in budgets]
                assert all(a <= b for a, b in pairwise(s))


def test_budget_calibration_rule() -> None:
    survival = {1.0: [0.0, 0.1], 3.0: [0.1, 0.15], 10.0: [0.3, 0.5], 30.0: [0.9, 1.0]}
    assert calibrate_budget(survival) == 10.0  # smallest median in [0.2, 0.8]
    none_in_band = {1.0: [0.0], 3.0: [0.1], 10.0: [0.9], 30.0: [1.0]}
    # Distances to 0.5: 0.5, 0.4, 0.4, 0.5 -> tie between 3 and 10 -> the smaller B.
    assert calibrate_budget(none_in_band) == 3.0


def test_zero_error_draw_needs_no_correction_and_a_small_error_is_corrected() -> None:
    elite = evaluate_unit(to_unit(C1))
    small = ErrorDraw(2e-7, 0.0, 0.0)  # ~0.6 mm/s along the burn
    zeros = np.zeros((2, 2))  # zero execution errors: executed = commanded
    rob = solution_robustness(
        elite, [ErrorDraw(0.0, 0.0, 0.0), small], executions=(GATES_PRIMARY,), normals=zeros
    )
    assert not rob.singular and rob.condition < 1e3
    zero, corrected = rob.draws
    # Re-propagating the nominal through t_c reproduces its targets (restart only).
    assert max(abs(v) for v in zero.initial_residual_km) < 1.0
    assert zero.converged and zero.commanded_m_s == (0.0, 0.0) and zero.uncorrected_feasible
    assert zero.executions[0].feasible and zero.survives(0.0) and zero.survives(1.0)
    # A sub-mm/s execution error moves h_vp by kilometres (M3: ~10 km per mm/s) and the
    # chord MCC brings it back with a tiny correction.
    assert max(abs(v) for v in corrected.initial_residual_km) > 1.0
    assert corrected.converged and corrected.commanded_feasible
    assert 0.0 < corrected.commanded_m_s_norm < 0.1
    assert corrected.executions[0].feasible
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


def test_gates_normals_and_pairing(tmp_path: Path) -> None:
    normals = gates_normals(T_E)
    assert normals.shape == (32, 2)
    assert np.array_equal(normals, gates_normals(T_E))
    expected = np.random.default_rng((7401, T_E.seed)).standard_normal((32, 2))
    assert np.array_equal(normals, expected)
    assert GATES_STRESS.sigma_s == 3e-2 and GATES_STRESS.sigma_r_m_s == 0.010
    assert GATES_STRESS.sigma_p_rad == GATES_PRIMARY.sigma_p_rad
    assert GATES_STRESS.sigma_a_m_s == GATES_PRIMARY.sigma_a_m_s
    # 9.1 pairing: the stress set goes only with the primary held-out injection set.
    assert execution_models(T_E) == (GATES_PRIMARY,)
    assert H_E_NOMINAL.executions == ("gates_primary", "gates_stress")
    assert H_E_WIDE.executions == ("gates_primary",)
    draft = protocol_file(tmp_path, "draft")
    with pytest.raises(ProtocolNotFrozenError, match="draft"):
        execution_models(H_E_NOMINAL, confirm_frozen=True, protocol_path=draft)
    with pytest.raises(ProtocolNotFrozenError, match="draft"):
        gates_normals(H_E_NOMINAL, confirm_frozen=True, protocol_path=draft)
    with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
        execution_models(H_E_NOMINAL)
    with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
        gates_normals(H_E_NOMINAL)
    frozen = protocol_file(tmp_path, "frozen")
    assert execution_models(H_E_NOMINAL, confirm_frozen=True, protocol_path=frozen) == (
        GATES_PRIMARY,
        GATES_STRESS,
    )
    held_out = gates_normals(H_E_NOMINAL, confirm_frozen=True, protocol_path=frozen)
    expected_held_out = np.random.default_rng((7401, 7101)).standard_normal((50, 2))
    assert np.array_equal(held_out, expected_held_out)
    assert execution_models(T_E, protocol_path=draft) == (GATES_PRIMARY,)


def test_executed_correction_adds_the_gates_error() -> None:
    elite = evaluate_unit(to_unit(C1))
    draws = [ErrorDraw(0.0, 0.0, 0.0), ErrorDraw(2e-7, 0.0, 0.0)]
    models = (GATES_PRIMARY, GATES_STRESS)
    exact = solution_robustness(elite, draws, executions=models, normals=np.zeros((2, 2)))
    zero, small = exact.draws
    assert zero.commanded_m_s == (0.0, 0.0)  # no burn, no execution error
    assert [x.feasible for x in zero.executions] == [True, True]
    commanded = np.array(small.commanded_m_s)
    for x in small.executions:  # zero normals: executed = commanded
        assert x.delta_v_m_s == small.commanded_m_s and x.feasible == small.commanded_feasible
    normals = np.array([[0.0, 0.0], [1.0, -2.0]])
    rob = solution_robustness(elite, draws, executions=models, normals=normals)
    d = rob.draws[1]
    # The commanded MCC does not depend on the execution sets (common random numbers).
    assert d.commanded_m_s == small.commanded_m_s
    for k, model in enumerate(models):
        executed = np.array(d.executions[k].delta_v_m_s)
        expected = gates_error(commanded, normals[1], model)
        assert executed - commanded == pytest.approx(expected, abs=1e-12)
        # |error| >= sigma_r = 10 mm/s along the burn here, far above the commanded.
        assert np.linalg.norm(executed - commanded) > 0.009
    assert d.propagations == small.propagations
    with pytest.raises(ValueError, match="normal pair"):
        solution_robustness(elite, draws, executions=models, normals=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="executed"):
        solution_robustness(elite, draws, executions=(), normals=np.zeros((2, 2)))


def test_survival_on_c1_is_monotone_in_budget_for_fixed_seeds() -> None:
    """D25 regression: with fixed injection and execution seeds, s(B) never decreases.

    The draws use the T_E scales with a non-protocol seed (9002, the smoke error seed).
    """
    dist = replace(T_E, name="test", draws=4, seed=9002)
    elite = evaluate_unit(to_unit(C1))
    models = (GATES_PRIMARY, GATES_STRESS)
    rob = solution_robustness(
        elite, error_draws(dist), executions=models, normals=gates_normals(dist)
    )
    budgets = [0.0, *PROTOCOL.correction.budget_grid_m_s, 100.0]
    fine = [*budgets, *np.linspace(0.0, 30.0, 61).tolist()]
    for k in range(len(models)):
        s = [rob.survival(b, k) for b in sorted(fine)]
        assert all(a <= b for a, b in pairwise(s))
        for d in rob.draws:
            per_draw = [d.survives(b, k) for b in sorted(fine)]
            assert all(a <= b for a, b in pairwise(per_draw))
    # The c1 smoke observation behind D25: these draws move h_vp by far more than 1 km.
    assert all(max(abs(v) for v in d.initial_residual_km) > 1.0 for d in rob.draws)


def test_infeasible_solutions_are_rejected() -> None:
    bad = toy_evaluate((0.05, 0.5, 0.5))
    with pytest.raises(ValueError, match="feasible"):
        solution_robustness(bad, [], executions=(GATES_PRIMARY,), normals=np.zeros((0, 2)))


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


def test_held_out_constraint_sets_refuse_before_the_freeze(tmp_path: Path) -> None:
    with pytest.raises(ProtocolNotFrozenError, match="draft"):
        held_out_sets(confirm_frozen=True, protocol_path=protocol_file(tmp_path, "draft"))
    for path in (protocol_file(tmp_path, "frozen"), PROTOCOL_PATH):
        with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
            held_out_sets(confirm_frozen=False, protocol_path=path)
    held_out = held_out_sets(confirm_frozen=True, protocol_path=protocol_file(tmp_path, "frozen"))
    tuning = tuning_sets()
    assert len(held_out) == 66 and [c.family for c in held_out].count("F2") == 22
    # The split is a partition: no set is both tuning and held out, and all 96 are used.
    keys = {(c.family, c.index) for c in held_out} | {(c.family, c.index) for c in tuning}
    assert len(keys) == 96 and len(keys) == len(held_out) + len(tuning)


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
