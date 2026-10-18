"""Validation of a free-return candidate (M3 stage S3; design section 5).

The candidate is re-propagated three ways: the run settings (DOP853 1e-12), a tighter
tolerance (DOP853 1e-13) and an independent integrator (LSODA 1e-13). Checks (each passes
when observed <= threshold, `export.Check`):

- every predicate holds under all three integrations, with margins of at least 1 km to
  the perilune band edges and 1 h to T_MAX;
- |dh_p|, |dt_ret| and the return-shell crossing position agree with the tight and
  independent runs to 10 m, 0.1 s and 10 m;
- Jacobi drift max |C(t) - C(0)| over >= 1e5 dense samples of each integration
  <= 1e-9 (terms of C reach ~116 near the departure, so 1e-9 is ~1e-11 relative);
- a backward DOP853 1e-13 integration (plain `solve_ivp`, no events) from the exported
  run's return-shell state to t = 0 recovers the departure state: the error is split into
  an along-track time shift (<= 1 ms) and the transverse remainder in position (<= 1 m)
  and velocity (<= 1 mm/s), see `decompose_state_error`. This replaced (user decision,
  2026-10-18) the design's single max-norm criterion max |ds0| <= 1e-7, which is kept as
  a recorded, non-gating diagnostic;
- the closest lunar approach found by brute dense sampling of r2(t) (>= 1e5 points
  plus a bounded scalar minimization) matches the perilune event to 1 m;
- no dense sample on [0, t_ret) lies inside the return shell or the Moon;
- the exported samples are finite with strictly increasing times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from phasespace.gravity.dynamics import (
    FloatArray,
    jacobi_constant,
    primary_distances,
    state_derivative,
)
from phasespace.gravity.export import Check
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    FreeReturnRecord,
    Outcome,
    classify,
    propagate_departure,
)
from phasespace.gravity.propagation import PropagationResult, SolverSettings
from phasespace.gravity.system import EarthMoonSystem

TIGHT_SOLVER: Final = SolverSettings(method="DOP853", rtol=1e-13, atol=1e-13)
INDEPENDENT_SOLVER: Final = SolverSettings(method="LSODA", rtol=1e-13, atol=1e-13)
BACKWARD_SOLVER: Final = SolverSettings(method="DOP853", rtol=1e-13, atol=1e-13)

PERILUNE_MARGIN_KM: Final = 1.0
T_MAX_MARGIN_HOURS: Final = 1.0
PERILUNE_AGREEMENT_KM: Final = 0.010
RETURN_TIME_AGREEMENT_S: Final = 0.1
RETURN_POSITION_AGREEMENT_KM: Final = 0.010
JACOBI_DRIFT_THRESHOLD: Final = 1e-9
SUPERSEDED_BACKWARD_STATE_THRESHOLD: Final = 1e-7
"""The design's original max-norm threshold on max |ds0| (normalized). Superseded: at the
parking orbit |ds/dt| ~ 3.4e3, so a pure time shift tau gives max |ds0| ~ 3.4e3 tau and
1e-7 amounts to a ~0.1 ms timing test. Recorded as a non-gating diagnostic only."""
# Decomposed backward check (user decision, 2026-10-18). The backward pass starts from the
# exported run's own return-shell state, so the check covers the exported trajectory.
BACKWARD_TIME_SHIFT_THRESHOLD_S: Final = 1e-3
"""|tau| <= 1 ms: 100x tighter than the 0.1 s return-time agreement check; a 1 ms shift
along the ~11 km/s departure arc is ~11 m of along-track position, the scale of the 10 m
perilune and return-position checks."""
BACKWARD_POSITION_THRESHOLD_M: Final = 1.0
"""Transverse position <= 1 m: 10x tighter than the 10 m perilune-altitude and
return-position agreement checks."""
BACKWARD_VELOCITY_THRESHOLD_MM_S: Final = 1.0
"""Transverse velocity <= 1 mm/s (~1e-7 of the ~10.9 km/s departure speed). This bounds
reversibility, not observable accuracy: with the local sensitivities of the M3 candidates
(|dh_p/d dv| ~ 6e5 km and |dh_vp/d dv| ~ 1e7 km per km/s) a 1 mm/s departure error would
move h_p by ~0.6 km and the vacuum perigee by ~10 km; the accuracy of the observables rests
on the forward tight / independent-integrator agreement checks."""
BRUTE_PERILUNE_AGREEMENT_KM: Final = 0.001
DENSE_SAMPLES: Final = 100_001
EXPORT_GRID_POINTS: Final = 2000


@dataclass(frozen=True, slots=True)
class BackwardDiagnostic:
    """Decomposition of a backward-integration error ds0 = s0_back - s0.

    With f = ds/dt at s0 (rotating-frame equations of motion, normalized), the along-track
    part is tau f with tau = (ds0 . f) / (f . f), the least-squares time shift in the
    normalized 4-D state metric; a pure time shift ds0 = tau f gives exactly tau. The
    transverse remainder r = ds0 - tau f is reported as |(r_x, r_y)| (position) and
    |(r_vx, r_vy)| (rotating-frame velocity), converted to physical units.
    """

    max_abs: float
    """max |ds0| over (x, y, vx, vy), normalized (the quantity of the section-5 check)."""
    along_track_time_s: float
    transverse_position_km: float
    transverse_velocity_m_s: float

    @staticmethod
    def undefined() -> BackwardDiagnostic:
        return BackwardDiagnostic(math.nan, math.nan, math.nan, math.nan)


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    """Propagations, their classifications, the checks and the exported samples."""

    run: PropagationResult
    tight: PropagationResult
    independent: PropagationResult
    records: tuple[FreeReturnRecord, FreeReturnRecord, FreeReturnRecord]
    checks: tuple[Check, ...]
    samples: FloatArray
    """Rows (t, x, y, vx, vy): union of accepted solver steps and a uniform grid."""
    perilune_t: float
    """Time of the run's closest Moon periapsis before t_final (NaN if none)."""
    perilune_state: FloatArray
    backward_run: BackwardDiagnostic
    """Diagnostics of the backward integration from the exported run's shell state."""
    backward_tight: BackwardDiagnostic
    """The same started from the tight (DOP853 1e-13) run's shell state: separates the
    amplification of the run's own integration error from that of the backward pass."""

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def _predicate_failures(record: FreeReturnRecord, definition: FreeReturnDefinition) -> int:
    if record.outcome is not Outcome.FEASIBLE:
        return 1
    lo, hi = definition.perilune_band_km
    h_p = record.perilune_altitude_km
    margin_t = (definition.t_max_days - record.return_time_days) * 24.0
    ok = (
        h_p - lo >= PERILUNE_MARGIN_KM
        and hi - h_p >= PERILUNE_MARGIN_KM
        and margin_t >= T_MAX_MARGIN_HOURS
    )
    return 0 if ok else 1


def _jacobi_drift(result: PropagationResult) -> float:
    times = np.linspace(0.0, result.t_final, DENSE_SAMPLES)
    c = jacobi_constant(result.sample(times), result.system.mu)
    c0 = float(jacobi_constant(result.states[0], result.system.mu))
    return float(np.max(np.abs(c - c0)))


def _closest_perilune(result: PropagationResult) -> tuple[float, FloatArray]:
    """Time and state of the closest Moon periapsis before t_final (the event)."""
    mu = result.system.mu
    lunar = [p for p in result.periapses if p.name == "moon_periapsis" and p.t < result.t_final]
    if not lunar:
        raise ValueError("no lunar periapsis before the return")
    best = min(lunar, key=lambda p: math.hypot(p.state[0] - 1.0 + mu, p.state[1]))
    return best.t, np.array(best.state)


def _brute_min_moon_distance(result: PropagationResult) -> float:
    """min r2(t) over [0, t_final] by dense sampling, refined by bounded minimization."""
    mu = result.system.mu
    times = np.linspace(0.0, result.t_final, DENSE_SAMPLES)
    _, r2 = primary_distances(result.sample(times), mu)
    k = int(np.argmin(r2))
    lo = float(times[max(k - 1, 0)])
    hi = float(times[min(k + 1, times.size - 1)])
    dense = result.dense
    assert dense is not None

    def r2_at(t: float) -> float:
        s = np.asarray(dense(t), dtype=np.float64)
        return math.hypot(float(s[0]) - 1.0 + mu, float(s[1]))

    opt = minimize_scalar(r2_at, bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
    return min(float(opt.fun), float(r2[k]))


def _inside_counts(result: PropagationResult, definition: FreeReturnDefinition) -> int:
    """Dense samples on [0, t_ret) inside the return shell or the Moon."""
    system = result.system
    times = np.linspace(0.0, result.t_final, DENSE_SAMPLES)[:-1]
    r1, r2 = primary_distances(result.sample(times), system.mu)
    shell = definition.return_shell_km / system.length_scale_km
    moon = system.moon_radius_km / system.length_scale_km
    return int(np.count_nonzero(r1 <= shell) + np.count_nonzero(r2 <= moon))


def _backward(result: PropagationResult) -> BackwardDiagnostic:
    """Backward integration from the final state to t = 0, compared with the start."""
    system = result.system
    mu = system.mu
    sol = solve_ivp(
        lambda _t, y: state_derivative(y, mu),
        (result.t_final, 0.0),
        result.final_state,
        method=BACKWARD_SOLVER.method,
        rtol=BACKWARD_SOLVER.rtol,
        atol=BACKWARD_SOLVER.atol,
    )
    if sol.status != 0:
        return BackwardDiagnostic(math.inf, math.inf, math.inf, math.inf)
    s0 = result.states[0]
    return decompose_state_error(np.asarray(sol.y, dtype=np.float64)[:, -1] - s0, s0, system)


def decompose_state_error(
    ds: FloatArray, s0: FloatArray, system: EarthMoonSystem
) -> BackwardDiagnostic:
    """Split ds = s_other - s0 into a time shift along f(s0) and the transverse rest."""
    f = state_derivative(s0, system.mu)
    tau = float(ds @ f / (f @ f))
    transverse = ds - tau * f
    return BackwardDiagnostic(
        max_abs=float(np.max(np.abs(ds))),
        along_track_time_s=tau * system.time_scale_s,
        transverse_position_km=float(np.hypot(transverse[0], transverse[1]))
        * system.length_scale_km,
        transverse_velocity_m_s=float(np.hypot(transverse[2], transverse[3]))
        * system.velocity_scale_km_s
        * 1000.0,
    )


def _export_samples(result: PropagationResult) -> FloatArray:
    grid = np.linspace(0.0, result.t_final, EXPORT_GRID_POINTS)
    times = np.union1d(result.t, grid)
    return np.column_stack([times, result.sample(times)])


def validate_candidate(
    phase_deg: float,
    delta_v_km_s: float,
    impulse_angle_deg: float = 0.0,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
) -> CandidateValidation:
    """Re-propagate one candidate three ways and evaluate every section-5 check."""
    system = definition.system
    scale = system.length_scale_km
    runs = tuple(
        propagate_departure(phase_deg, delta_v_km_s, impulse_angle_deg, definition, solver=s)
        for s in (definition.solver, TIGHT_SOLVER, INDEPENDENT_SOLVER)
    )
    run, tight, independent = runs
    records = tuple(
        classify(
            r,
            definition,
            phase_deg=phase_deg,
            delta_v_km_s=delta_v_km_s,
            impulse_angle_deg=impulse_angle_deg,
        )
        for r in runs
    )
    rec, rec_tight, rec_ind = records

    failures = sum(_predicate_failures(r, definition) for r in records)
    checks: list[Check] = [
        Check(
            "predicate_failures_run_tight_independent",
            float(failures),
            0.0,
            "number of the three integrations (run, DOP853 1e-13, LSODA 1e-13) in which the "
            "candidate is not feasible or has a margin below "
            f"{PERILUNE_MARGIN_KM} km to the perilune band edges or {T_MAX_MARGIN_HOURS} h to "
            "T_MAX",
        )
    ]
    feasible_all = failures == 0
    for label, other, other_rec in (
        ("tight_tolerance", tight, rec_tight),
        ("independent_integrator", independent, rec_ind),
    ):
        ref = f"{other.solver.method} rtol = atol = {other.solver.rtol:g}"
        if feasible_all:
            d_hp = abs(rec.perilune_altitude_km - other_rec.perilune_altitude_km)
            d_t = abs(rec.return_time_days - other_rec.return_time_days) * 86400.0
            d_r = float(np.hypot(*(run.final_state[:2] - other.final_state[:2]))) * scale
        else:
            d_hp = d_t = d_r = math.nan
        checks += [
            Check(
                f"{label}_abs_diff_perilune_altitude_km",
                d_hp,
                PERILUNE_AGREEMENT_KM,
                f"|h_p - h_p_ref| in km; ref: {ref}",
            ),
            Check(
                f"{label}_abs_diff_return_time_s",
                d_t,
                RETURN_TIME_AGREEMENT_S,
                f"|t_ret - t_ret_ref| in seconds at the return-shell crossing; ref: {ref}",
            ),
            Check(
                f"{label}_return_position_difference_km",
                d_r,
                RETURN_POSITION_AGREEMENT_KM,
                f"distance in km between the return-shell crossing positions; ref: {ref}",
            ),
        ]
    for label, result in (
        ("run", run),
        ("tight_tolerance", tight),
        ("independent_integrator", independent),
    ):
        checks.append(
            Check(
                f"jacobi_max_abs_drift_{label}",
                _jacobi_drift(result) if result.dense is not None else math.nan,
                JACOBI_DRIFT_THRESHOLD,
                f"max |C(t) - C(0)| at {DENSE_SAMPLES} uniform times over [0, t_final] of the "
                f"{result.solver.method} rtol = atol = {result.solver.rtol:g} integration; a "
                "conservation check only",
            )
        )
    try:
        t_p, s_p = _closest_perilune(run)
    except ValueError:
        t_p, s_p = math.nan, np.full(4, math.nan)
    samples = _export_samples(run) if run.dense is not None else np.empty((0, 5))
    if not feasible_all:
        return CandidateValidation(
            run,
            tight,
            independent,
            (rec, rec_tight, rec_ind),
            tuple(checks),
            samples,
            t_p,
            s_p,
            BackwardDiagnostic.undefined(),
            BackwardDiagnostic.undefined(),
        )

    _, r2_event = primary_distances(s_p, system.mu)
    backward_run = _backward(run)
    backward = (
        f"s0_back from a backward {BACKWARD_SOLVER.method} rtol = atol = "
        f"{BACKWARD_SOLVER.rtol:g} solve_ivp integration (no events) started from the "
        "exported run's return-shell state"
    )
    checks += [
        Check(
            "backward_integration_along_track_time_shift_s",
            abs(backward_run.along_track_time_s),
            BACKWARD_TIME_SHIFT_THRESHOLD_S,
            f"|tau| in seconds, tau = (ds0 . f) / (f . f), ds0 = s0_back - s0, f = ds/dt at "
            f"s0 (normalized rotating-frame state); {backward}. Threshold 1 ms: 100x tighter "
            "than the 0.1 s return-time agreement check; ~11 m along the ~11 km/s "
            "departure arc, the scale of the 10 m position checks",
        ),
        Check(
            "backward_integration_transverse_position_error_m",
            backward_run.transverse_position_km * 1000.0,
            BACKWARD_POSITION_THRESHOLD_M,
            f"|position part of ds0 - tau f| in metres; {backward}. Threshold 1 m: 10x "
            "tighter than the 10 m perilune and return-position agreement checks",
        ),
        Check(
            "backward_integration_transverse_velocity_error_mm_s",
            backward_run.transverse_velocity_m_s * 1000.0,
            BACKWARD_VELOCITY_THRESHOLD_MM_S,
            f"|velocity part of ds0 - tau f| in mm/s (rotating-frame components); {backward}. "
            "Threshold 1 mm/s (~1e-7 of the departure speed) bounds reversibility, not "
            "observable accuracy: at the local sensitivities of the M3 candidates 1 mm/s at "
            "departure moves h_p by ~0.6 km and the vacuum perigee by ~10 km; observable "
            "accuracy rests on the forward agreement checks",
        ),
        Check(
            "brute_force_abs_diff_perilune_distance_km",
            abs(_brute_min_moon_distance(run) - float(r2_event)) * scale,
            BRUTE_PERILUNE_AGREEMENT_KM,
            f"|min r2 (dense sampling, {DENSE_SAMPLES} points + bounded minimization) - r2 at "
            "the perilune event| in km",
        ),
        Check(
            "dense_samples_inside_return_shell_or_moon",
            float(_inside_counts(run, definition)),
            0.0,
            f"count of {DENSE_SAMPLES - 1} uniform dense samples on [0, t_ret) with "
            "r1 <= R_shell or r2 <= R_moon (a missed earlier crossing)",
        ),
        Check(
            "non_finite_sample_values",
            float(np.count_nonzero(~np.isfinite(samples))),
            0.0,
            "count of non-finite values in trajectory samples",
        ),
        Check(
            "non_increasing_sample_times",
            float(np.count_nonzero(np.diff(samples[:, 0]) <= 0.0)),
            0.0,
            "count of consecutive sample times with t[i+1] <= t[i]",
        ),
    ]
    return CandidateValidation(
        run,
        tight,
        independent,
        (rec, rec_tight, rec_ind),
        tuple(checks),
        samples,
        t_p,
        s_p,
        backward_run,
        _backward(tight),
    )
