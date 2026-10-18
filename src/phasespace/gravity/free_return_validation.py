"""Validation of a free-return candidate (M3 stage S3; design section 5).

The candidate is re-propagated three ways: the run settings (DOP853 1e-12), a tighter
tolerance (DOP853 1e-13) and an independent integrator (LSODA 1e-13). Checks (each passes
when observed <= threshold, `export.Check`):

- every predicate holds under all three integrations, with margins of at least 1 km to
  the perilune band edges and 1 h to T_MAX;
- |dh_p|, |dt_ret| and the return-shell crossing position agree with the tight and
  independent runs to 10 m, 0.1 s and 10 m;
- the targeted entry quantities agree with the tight and independent runs: entry speed
  to 1 mm/s, flight-path angle to 1e-3 deg and vacuum perigee to 50 m (thresholds derived
  from energy conservation and two-body sensitivities at the shell, see the constants);
- Jacobi drift max |C(t) - C(0)| over >= 1e5 dense samples of each integration
  <= 1e-9 (terms of C reach ~116 near the departure, so 1e-9 is ~1e-11 relative);
- a backward DOP853 1e-13 integration (plain `solve_ivp`, no events) from the exported
  run's return-shell state to t = 0 recovers the departure state: the error is split into
  a time-shift-like component (<= 1 ms) and the remainder in position (<= 1 m) and
  velocity (<= 1 mm/s), see `decompose_state_error`. These are reversibility bounds, not
  accuracy bounds on the observables. This replaced (user decision, 2026-10-18) the
  design's single max-norm criterion max |ds0| <= 1e-7, which is kept as a recorded,
  non-gating diagnostic;
- the closest lunar approach found by brute dense sampling of r2(t) (>= 1e5 points
  plus a bounded scalar minimization) matches the perilune event to 1 m;
- no dense sample on [0, t_ret) lies inside the return shell or the Moon;
- the exported samples are finite with strictly increasing times.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    earth_angular_momentum_sign,
    evaluate,
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
"""M3 threshold, unchanged. Margins are small for the targeted candidates (c1: observed
8.0e-10, a factor of only 1.26); a scaled Jacobi criterion must be frozen in the M4
protocol before any M4 data is produced."""
# Entry-quantity agreement (tight / independent vs run). Derived from integrator-agreement
# physics, not from observed values:
ENTRY_SPEED_AGREEMENT_KM_S: Final = 1e-6
"""1 mm/s. Both integrations conserve C to <= 1e-9, so at the same radius their speeds can
differ by |dC| / (2 v) + (potential difference between their shell points) ~ 1e-9 / v
~ 1e-10 normalized ~ 0.1 mm/s for v ~ 10.7; 1 mm/s leaves a factor ~10 for the potential
difference across the <= 10 m separation of the crossing points."""
ENTRY_FLIGHT_PATH_ANGLE_AGREEMENT_DEG: Final = 1e-3
"""1e-3 deg. A 10 m separation of the crossing points (the position-agreement threshold)
rotates the local horizontal by 10 m / R_S ~ 1.5e-6 rad and the velocity direction by
kappa * 10 m with path curvature kappa ~ g cos(gamma) / v^2 ~ 8e-8 /m, together ~ 2.3e-6
rad ~ 1.3e-4 deg; 1e-3 deg leaves a factor ~7."""
ENTRY_VACUUM_PERIGEE_AGREEMENT_KM: Final = 0.050
"""50 m. Two-body sensitivities at the shell for a near-parabolic entry (r_p ~ h^2 / (2 GM),
h = r v cos(gamma)): |dr_p/d gamma| ~ 2 r_p |tan(gamma)| ~ 25 km/deg at gamma ~ -6.5 deg
and |dr_p/dv| ~ 2 r_p / v ~ 1.2 km per m/s, so the angle and speed thresholds above map to
~25 m + ~1 m; 50 m leaves a factor ~2."""
SUPERSEDED_BACKWARD_STATE_THRESHOLD: Final = 1e-7
"""The design's original max-norm threshold on max |ds0| (normalized). Superseded: at the
parking orbit |ds/dt| ~ 3.4e3, so a pure time shift tau gives max |ds0| ~ 3.4e3 tau and
1e-7 amounts to a ~0.1 ms timing test. Recorded as a non-gating diagnostic only."""
# Decomposed backward check (user decision, 2026-10-18). The backward pass starts from the
# exported run's own return-shell state, so the check covers the exported trajectory. All
# three are reversibility bounds, NOT accuracy bounds on the observables: departure errors
# are amplified through the lunar flyby (at the candidates' sensitivities, 1 mm/s at
# departure moves h_p by ~0.6 km and the vacuum perigee by ~10 km). Observable accuracy
# rests on the forward tight / independent-integrator agreement checks.
#
# tau = (ds0 . f) / (f . f) is a projection in the unweighted normalized 4-D metric. At the
# parking orbit f = (v, a) with |a| ~ 3.4e3 >> |v| ~ 10.6, so tau ~ (dv . a) / |a|^2: in
# effect the velocity error along the local acceleration (~ radial) divided by |a|. A pure
# time shift and a velocity error along a are indistinguishable here; position-only errors
# land almost entirely in the remainder.
BACKWARD_TIME_SHIFT_THRESHOLD_S: Final = 1e-3
"""|tau| <= 1 ms (100x below the 0.1 s return-time agreement threshold). Because tau is
dominated by the acceleration, this admits a velocity error along a of up to
|a| * 1 ms ~ 9 mm/s at the parking orbit, or the equivalent pure time shift."""
BACKWARD_POSITION_THRESHOLD_M: Final = 1.0
"""Remainder position <= 1 m: a reversibility bound on the departure position."""
BACKWARD_VELOCITY_THRESHOLD_MM_S: Final = 1.0
"""Remainder velocity <= 1 mm/s (~1e-7 of the ~10.9 km/s departure speed): a
reversibility bound on the departure velocity transverse to the acceleration."""
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
class DeltaVEdge:
    """Distance in delta-v (fixed phase and impulse angle) to the nearest infeasible value."""

    margin_km_s: float
    """Offset to the first infeasible delta-v found (inf: none within the search range)."""
    outcome_beyond: Outcome | None
    """Outcome just beyond the edge (None: no edge found)."""
    propagations: int


def find_edge(
    is_feasible: Callable[[float], bool],
    *,
    step: float,
    tolerance: float,
    max_offset: float,
) -> tuple[float, int]:
    """Smallest offset o > 0 (to `tolerance`) with is_feasible(o) False, given is_feasible(0).

    Steps outward with doubling steps from `step` until an infeasible offset is found, then
    bisects between the last feasible and the first infeasible offset. Returns (offset,
    evaluations); offset is inf if every probed offset up to `max_offset` is feasible. A
    feasible island beyond the first infeasible offset is not seen.
    """
    evaluations = 0
    lo, hi = 0.0, step
    while True:
        evaluations += 1
        if not is_feasible(hi):
            break
        if hi >= max_offset:
            return math.inf, evaluations
        lo, hi = hi, min(2.0 * hi, max_offset)
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        evaluations += 1
        if is_feasible(mid):
            lo = mid
        else:
            hi = mid
    return hi, evaluations


EDGE_STEP_KM_S: Final = 1e-6
EDGE_TOLERANCE_KM_S: Final = 1e-8
EDGE_MAX_OFFSET_KM_S: Final = 0.05


def delta_v_edges(
    phase_deg: float,
    delta_v_km_s: float,
    impulse_angle_deg: float = 0.0,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
) -> tuple[DeltaVEdge, DeltaVEdge]:
    """(below, above): delta-v margins of a feasible point to the nearest infeasible values.

    Non-gating diagnostics: search from 1 mm/s with doubling steps up to 50 m/s, bisected to
    0.01 mm/s, using the run settings. Below the point the edge of a figure-eight free
    return is typically the skip-out (no return within T_MAX).
    """
    edges: list[DeltaVEdge] = []
    for sign in (-1.0, 1.0):
        outcomes: dict[float, Outcome] = {}

        def feasible(
            offset: float, sign: float = sign, outcomes: dict[float, Outcome] = outcomes
        ) -> bool:
            dv = delta_v_km_s + sign * offset
            outcome = evaluate(phase_deg, dv, impulse_angle_deg, definition).outcome
            outcomes[offset] = outcome
            return outcome is Outcome.FEASIBLE

        margin, n = find_edge(
            feasible,
            step=EDGE_STEP_KM_S,
            tolerance=EDGE_TOLERANCE_KM_S,
            max_offset=EDGE_MAX_OFFSET_KM_S,
        )
        edges.append(DeltaVEdge(margin, outcomes.get(margin), n))
    return edges[0], edges[1]


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
    entry_sense: int
    """Sign of the Earth-relative inertial angular momentum at the return shell (+1
    posigrade, -1 retrograde, 0 without a return)."""

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
            d_v = abs(rec.return_speed_km_s - other_rec.return_speed_km_s)
            d_g = abs(rec.return_flight_path_angle_deg - other_rec.return_flight_path_angle_deg)
            d_vp = abs(
                rec.return_vacuum_perigee_altitude_km - other_rec.return_vacuum_perigee_altitude_km
            )
        else:
            d_hp = d_t = d_r = d_v = d_g = d_vp = math.nan
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
            Check(
                f"{label}_abs_diff_entry_speed_km_s",
                d_v,
                ENTRY_SPEED_AGREEMENT_KM_S,
                "|v - v_ref| of the Earth-relative inertial speed at the return shell, km/s; "
                f"ref: {ref}. Threshold 1 mm/s from Jacobi conservation (<= 1e-9 each: "
                "|dv| ~ 1e-9 / v ~ 0.1 mm/s at the same radius) with a factor ~10 for the "
                "potential difference across the <= 10 m separation of the crossing points",
            ),
            Check(
                f"{label}_abs_diff_entry_flight_path_angle_deg",
                d_g,
                ENTRY_FLIGHT_PATH_ANGLE_AGREEMENT_DEG,
                f"|gamma - gamma_ref| at the return shell, deg; ref: {ref}. Threshold 1e-3 deg: "
                "a 10 m crossing-point separation changes gamma by ~10 m / R_S + kappa * 10 m "
                "~ 1.3e-4 deg (kappa ~ g cos(gamma) / v^2), a factor ~7 below",
            ),
            Check(
                f"{label}_abs_diff_entry_vacuum_perigee_km",
                d_vp,
                ENTRY_VACUUM_PERIGEE_AGREEMENT_KM,
                f"|h_vp - h_vp_ref| at the return shell, km; ref: {ref}. Threshold 50 m from "
                "the two-body sensitivities of a near-parabolic entry (~25 km/deg in gamma, "
                "~1.2 km per m/s in speed) applied to the angle and speed thresholds",
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
            0,
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
            f"s0 (normalized rotating-frame state, unweighted metric); {backward}. With "
            "|a| ~ 3.4e3 >> |v| ~ 10.6 at the parking orbit, tau is in effect the velocity "
            "error along the local acceleration divided by |a| (a time shift and such a "
            "velocity error are indistinguishable): 1 ms admits ~9 mm/s along a. A "
            "reversibility bound, not an accuracy bound: departure errors are amplified "
            "through the flyby; 1 ms is 100x below the 0.1 s return-time threshold",
        ),
        Check(
            "backward_integration_transverse_position_error_m",
            backward_run.transverse_position_km * 1000.0,
            BACKWARD_POSITION_THRESHOLD_M,
            f"|position part of ds0 - tau f| in metres; {backward}. A reversibility bound on "
            "the departure position (position-only errors land here), not an accuracy bound "
            "on the observables, which are amplified through the flyby",
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
        earth_angular_momentum_sign(run.final_state, run.t_final, system.mu),
    )
