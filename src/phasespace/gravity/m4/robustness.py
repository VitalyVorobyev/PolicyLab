"""Robustness evaluators of protocol section 9 (D17).

(a) Execution-error tolerance (9.1). For a catalog solution (theta, delta-v, alpha):

- error draws: independent truncated normals (|z| <= 3) per component, by inversion
  z = Phi^-1(Phi(-3) + U (Phi(3) - Phi(-3))) of `default_rng(seed).random((K, 3))`
  (columns: magnitude, direction, timing). The draws are common random numbers: draw k
  is the same triple for every solution of every method;
- executed TLI: delta-v' = delta-v (1 + eps_m), alpha' = alpha + eps_d and, for a burn
  dt late, theta' = theta + (n_park - 1) dt (the parking orbit's rotating-frame rate;
  the CR3BP is autonomous, so a late burn is a shifted departure phase);
- one in-plane impulsive MCC at t_c = 1 d after the burn retargets the solution's own
  nominal (h_p, h_vp) with chord iterations on a fixed Jacobian
  J = d(h_p, h_vp)/d(dv) (forward differences, 0.1 m/s, at the nominal t_c state):
  dv_{k+1} = dv_k - J^-1 R(dv_k), at most 3 times, stopping early when |R| <= 1 km per
  component (converged, a diagnostic), when R is undefined, or when |dv| > 100 m/s
  (diverged). The commanded MCC dv_m is the final iterate; with cond(J) > 1e8 none is
  commanded;
- the trajectory after a correction is the arc 0 -> t_c joined to a new propagation
  from t_c; predicates are evaluated on the joined trajectory (`classify`) and the
  Jacobi gate per segment (the impulse changes C);
- D24 / D25: dv_m is executed with Gates-model errors (`gates_error`, parameters
  `config.GatesModel`). A draw survives at budget B iff it is feasible without
  correction, or |dv_m| <= B and the trajectory after the *executed* correction is
  feasible. Survival is therefore non-decreasing in B.

The MCC dv is a rotating-frame velocity increment (dvx, dvy); an impulse is the same
vector in the inertial frame up to the frame rotation, so |dv| is frame independent.

(b) Changed-constraint availability (9.2): constraint sets C = (h_lo, h_hi, T', gamma_c,
w) in three Sobol-generated families with a seeded tuning / held-out split. Held-out
sets are only returned after the freeze (`held_out_sets` calls `require_frozen`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from phasespace.gravity.departure import parking_radius, tli_state
from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    PERILUNE_BAND_KM,
    FreeReturnDefinition,
    FreeReturnRecord,
    classify,
)
from phasespace.gravity.m4.config import (
    GATES_MODELS,
    GATES_NORMALS_SEED,
    PROTOCOL,
    ConstraintFamilies,
    CorrectionConfig,
    ErrorDistribution,
    GatesModel,
    require_frozen,
)
from phasespace.gravity.m4.evaluation import Evaluation, segment_drift_rel
from phasespace.gravity.propagation import (
    CollisionRadii,
    PropagationResult,
    Termination,
    propagate,
)
from phasespace.gravity.units import time_from_days, time_from_seconds, velocity_from_km_s

_CORR: Final = PROTOCOL.correction


# --- error draws ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorDraw:
    magnitude: float
    """eps_m: relative TLI magnitude error."""
    direction_deg: float
    timing_s: float


def truncated_normal(uniform: FloatArray, truncation: float) -> FloatArray:
    """Standard normal truncated to [-a, a] by inversion of uniforms in [0, 1)."""
    lo = ndtr(-truncation)
    return np.asarray(ndtri(lo + uniform * (ndtr(truncation) - lo)), dtype=np.float64)


def error_draws(dist: ErrorDistribution, *, confirm_frozen: bool = False) -> list[ErrorDraw]:
    """The K draws of a distribution; a held-out one requires the frozen protocol."""
    if dist.held_out:
        require_frozen(confirm_frozen)
    return _draws(dist)


def _draws(dist: ErrorDistribution) -> list[ErrorDraw]:
    rng = np.random.default_rng(dist.seed)
    z = truncated_normal(rng.random((dist.draws, 3)), dist.truncation_sigmas)
    return [
        ErrorDraw(
            float(row[0]) * dist.sigma_magnitude,
            float(row[1]) * dist.sigma_direction_deg,
            float(row[2]) * dist.sigma_timing_s,
        )
        for row in z
    ]


def parking_mean_motion(definition: FreeReturnDefinition = DEFAULT_DEFINITION) -> float:
    """Inertial mean motion of the circular parking orbit, normalized (protocol: 446.233)."""
    r0 = parking_radius(definition.system, definition.parking_altitude_km)
    return math.sqrt((1.0 - definition.system.mu) / r0**3)


def executed_departure(
    phase_deg: float,
    delta_v_km_s: float,
    impulse_angle_deg: float,
    draw: ErrorDraw,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
) -> tuple[float, float, float]:
    """(theta', delta-v', alpha') of the executed TLI (9.1 error model)."""
    rate = parking_mean_motion(definition) - 1.0
    dt = time_from_seconds(draw.timing_s, definition.system)
    return (
        phase_deg + math.degrees(rate * dt),
        delta_v_km_s * (1.0 + draw.magnitude),
        impulse_angle_deg + draw.direction_deg,
    )


# --- arcs and joined trajectories ----------------------------------------------------------


def _propagate(
    state: FloatArray, t0: float, duration: float, definition: FreeReturnDefinition
) -> PropagationResult:
    system = definition.system
    return propagate(
        state,
        duration,
        system,
        solver=definition.solver,
        collision=CollisionRadii.of(system),
        return_shell_km=definition.return_shell_km,
        t0=t0,
        max_evaluations=definition.max_evaluations,
    )


def _joined(arc: PropagationResult, tail: PropagationResult, duration: float) -> PropagationResult:
    """The arc 0 -> t_c followed by the tail from t_c, as one result for `classify`."""
    return PropagationResult(
        system=tail.system,
        solver=tail.solver,
        collision=tail.collision,
        return_shell_km=tail.return_shell_km,
        duration=duration,
        t=np.concatenate([arc.t, tail.t]),
        states=np.vstack([arc.states, tail.states]),
        termination=tail.termination,
        events=tail.events,
        periapses=(*arc.periapses, *tail.periapses),
        nfev=arc.nfev + tail.nfev,
        message=tail.message,
        dense=None,
    )


@dataclass(frozen=True, slots=True)
class Leg:
    """The joined trajectory for one MCC dv: its record, gate and targeting values."""

    record: FreeReturnRecord
    drift_rel: float
    feasible: bool
    nfev: int

    @property
    def targets(self) -> tuple[float, float]:
        return self.record.perilune_altitude_km, self.record.vacuum_perigee_altitude_km


def _leg(
    arc: PropagationResult,
    dv_m_s: FloatArray,
    definition: FreeReturnDefinition,
    t_c: float,
) -> Leg:
    state = arc.final_state
    dv = velocity_from_km_s(np.asarray(dv_m_s, dtype=np.float64) / 1000.0, definition.system)
    state[2:] = state[2:] + dv
    tail = _propagate(state, t_c, definition.duration - t_c, definition)
    record = classify(_joined(arc, tail, definition.duration), definition)
    drift = max(segment_drift_rel(arc)[0], segment_drift_rel(tail)[0])
    gate = math.isfinite(drift) and drift <= PROTOCOL.gate.jacobi_drift_rel_max
    return Leg(record, drift, gate and record.feasible, tail.nfev)


def _arc(
    phase_deg: float, dv_km_s: float, alpha_deg: float, definition: FreeReturnDefinition, t_c: float
) -> PropagationResult:
    state0 = tli_state(
        math.radians(phase_deg),
        dv_km_s,
        math.radians(alpha_deg),
        definition.system,
        parking_altitude_km=definition.parking_altitude_km,
    )
    return _propagate(state0, 0.0, t_c, definition)


# --- the correction ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Execution:
    """The commanded MCC of one draw, executed under one Gates set (D24)."""

    model: str
    delta_v_m_s: tuple[float, float]
    """Executed MCC (rotating-frame components, m/s); (0, 0) when no burn is commanded."""
    feasible: bool | None
    """5.1 and 5.2 hold after the executed MCC; None if not propagated (the chord diverged,
    so the commanded |dv| exceeds every budget for which survival is defined)."""


@dataclass(frozen=True, slots=True)
class DrawResult:
    uncorrected_feasible: bool
    commanded_m_s: tuple[float, float]
    """The commanded MCC: the final chord iterate; (0, 0) when none is commanded (the
    uncorrected residual is within tolerance or undefined, J is singular, or the
    trajectory ended before t_c)."""
    executions: tuple[Execution, ...]
    """One per Gates set, in the order given to `solution_robustness`."""
    converged: bool
    """Diagnostic only (D25): the final chord residual is within 1 km per component."""
    diverged: bool
    """Diagnostic: the chord stopped because |dv| exceeded the divergence threshold."""
    commanded_feasible: bool | None
    """Diagnostic: 5.1 and 5.2 hold with the commanded MCC applied exactly; None if the
    commanded MCC was not propagated (divergence)."""
    iterations: int
    initial_residual_km: tuple[float, float]
    propagations: int
    """Arc plus tail propagations of this draw (each a partial propagation)."""
    nfev: int

    @property
    def commanded_m_s_norm(self) -> float:
        """|dv| of the commanded MCC (m/s)."""
        return math.hypot(*self.commanded_m_s)

    def survives(
        self, budget_m_s: float, execution: int = 0, cfg: CorrectionConfig = _CORR
    ) -> bool:
        """Success at budget B (9.1, D25): feasible without correction, or commanded
        |dv| <= B and feasible after the executed correction. Non-decreasing in B."""
        if not 0.0 <= budget_m_s <= cfg.divergence_m_s:
            raise ValueError(f"survival is defined for 0 <= B <= {cfg.divergence_m_s} m/s")
        if self.uncorrected_feasible:
            return True
        executed = self.executions[execution]
        return self.commanded_m_s_norm <= budget_m_s and executed.feasible is True


@dataclass(frozen=True, slots=True)
class SolutionRobustness:
    nominal_targets: tuple[float, float]
    jacobian_km_per_m_s: tuple[tuple[float, float], tuple[float, float]]
    condition: float
    singular: bool
    executions: tuple[str, ...]
    draws: tuple[DrawResult, ...]
    setup_propagations: int
    setup_nfev: int

    def survival(self, budget_m_s: float, execution: int = 0) -> float:
        """s_i(B): the fraction of draws that succeed at budget B under one Gates set."""
        if not self.draws:
            return math.nan
        return sum(d.survives(budget_m_s, execution) for d in self.draws) / len(self.draws)


def _undefined(values: Iterable[float]) -> bool:
    return not all(math.isfinite(v) for v in values)


def solution_robustness(
    elite: Evaluation,
    draws: Sequence[ErrorDraw],
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    cfg: CorrectionConfig = _CORR,
    *,
    executions: Sequence[GatesModel],
    normals: FloatArray,
) -> SolutionRobustness:
    """Execution-error tolerance of one feasible catalog solution under the given draws.

    The commanded MCC of draw k is computed once and executed under every Gates set in
    `executions` with the same standard normal pair `normals[k]` (`gates_normals`), so
    the sets differ only through their sigmas (common random numbers, 9.1).
    """
    if not executions:
        raise ValueError("survival is judged after an executed correction (D25)")
    if normals.shape != (len(draws), 2):
        raise ValueError("the execution models need one standard normal pair per draw")
    if not elite.feasible:
        raise ValueError("only feasible catalog solutions are evaluated")
    t_c = time_from_days(cfg.t_c_days, definition.system)
    nominal = (elite.perilune_altitude_km, elite.vacuum_perigee_altitude_km)
    arc = _arc(elite.phase_deg, elite.delta_v_km_s, elite.impulse_angle_deg, definition, t_c)
    setup_nfev = arc.nfev
    h = cfg.fd_step_m_s
    columns: list[tuple[float, float]] = []
    base: Leg | None = None
    if arc.termination is Termination.HORIZON:
        base = _leg(arc, np.zeros(2), definition, t_c)
        setup_nfev += base.nfev
        for k in range(2):
            step = np.zeros(2)
            step[k] = h
            leg = _leg(arc, step, definition, t_c)
            setup_nfev += leg.nfev
            columns.append(
                ((leg.targets[0] - base.targets[0]) / h, (leg.targets[1] - base.targets[1]) / h)
            )
    jac = np.array(columns).T if len(columns) == 2 else np.full((2, 2), math.nan)
    finite = bool(np.all(np.isfinite(jac)))
    cond = float(np.linalg.cond(jac)) if finite else math.inf
    singular = not (finite and cond <= cfg.cond_max)
    results = tuple(
        _draw(
            elite,
            d,
            arc_ok=not singular,
            jac=jac,
            nominal=nominal,
            definition=definition,
            executions=executions,
            normal=normals[k],
            cfg=cfg,
        )
        for k, d in enumerate(draws)
    )
    return SolutionRobustness(
        nominal_targets=nominal,
        jacobian_km_per_m_s=(
            (float(jac[0, 0]), float(jac[0, 1])),
            (float(jac[1, 0]), float(jac[1, 1])),
        ),
        condition=cond,
        singular=singular,
        executions=tuple(g.name for g in executions),
        draws=results,
        setup_propagations=1 + (3 if base is not None else 0),
        setup_nfev=setup_nfev,
    )


def _draw(
    elite: Evaluation,
    draw: ErrorDraw,
    *,
    arc_ok: bool,
    jac: FloatArray,
    nominal: tuple[float, float],
    definition: FreeReturnDefinition,
    executions: Sequence[GatesModel],
    normal: FloatArray,
    cfg: CorrectionConfig,
) -> DrawResult:
    t_c = time_from_days(cfg.t_c_days, definition.system)
    phase, dv, alpha = executed_departure(
        elite.phase_deg, elite.delta_v_km_s, elite.impulse_angle_deg, draw, definition
    )
    arc = _arc(phase, dv, alpha, definition, t_c)
    nfev = arc.nfev
    if arc.termination is not Termination.HORIZON:
        # The trajectory ended (collision or shell) before t_c: nothing to correct, and
        # it is infeasible (no lunar periapsis before the return).
        no_burn = tuple(Execution(g.name, (0.0, 0.0), False) for g in executions)
        nan2 = (math.nan, math.nan)
        return DrawResult(False, (0.0, 0.0), no_burn, False, False, False, 0, nan2, 1, nfev)
    delta = np.zeros(2)
    leg = _leg(arc, delta, definition, t_c)
    nfev += leg.nfev
    propagations = 2
    uncorrected = leg.feasible

    def residual_of(lg: Leg) -> FloatArray:
        return np.array([lg.targets[0] - nominal[0], lg.targets[1] - nominal[1]])

    # Chord iterations; the commanded MCC is the final iterate (D25).
    r = residual_of(leg)
    r0 = (float(r[0]), float(r[1]))
    iterations = 0
    converged = diverged = False
    commanded_feasible: bool | None = leg.feasible
    while not _undefined(r):
        if float(np.max(np.abs(r))) <= cfg.tolerance_km:
            converged = True
            break
        if not arc_ok or iterations >= cfg.max_iterations:
            break
        delta = delta - np.linalg.solve(jac, r)
        iterations += 1
        if float(np.linalg.norm(delta)) > cfg.divergence_m_s:
            diverged = True
            commanded_feasible = None
            break
        leg = _leg(arc, delta, definition, t_c)
        nfev += leg.nfev
        propagations += 1
        commanded_feasible = leg.feasible
        r = residual_of(leg)
    commanded = (float(delta[0]), float(delta[1]))
    executed: list[Execution] = []
    for model in executions:
        if not np.any(delta != 0.0):
            # No burn is commanded, so none is executed: the uncorrected trajectory.
            executed.append(Execution(model.name, (0.0, 0.0), uncorrected))
            continue
        burn = delta + gates_error(delta, normal, model)
        if diverged:
            executed.append(Execution(model.name, (float(burn[0]), float(burn[1])), None))
            continue
        executed_leg = _leg(arc, burn, definition, t_c)
        nfev += executed_leg.nfev
        propagations += 1
        executed.append(
            Execution(model.name, (float(burn[0]), float(burn[1])), executed_leg.feasible)
        )
    return DrawResult(
        uncorrected_feasible=uncorrected,
        commanded_m_s=commanded,
        executions=tuple(executed),
        converged=converged,
        diverged=diverged,
        commanded_feasible=commanded_feasible,
        iterations=iterations,
        initial_residual_km=r0,
        propagations=propagations,
        nfev=nfev,
    )


def gates_error(commanded_m_s: FloatArray, normal: FloatArray, model: GatesModel) -> FloatArray:
    """Gates-model execution error (m/s) of a commanded in-plane burn (D24).

    eta = sqrt(sigma_r^2 + V^2 sigma_s^2) z1 e + sqrt(sigma_a^2 + V^2 sigma_p^2) z2 e_perp,
    whose covariance is Q = (sigma_r^2 + V^2 sigma_s^2) e e^T
    + (sigma_a^2 + V^2 sigma_p^2) (I - e e^T) for z ~ N(0, I_2); e_perp = (-e_y, e_x).
    """
    v = float(np.linalg.norm(commanded_m_s))
    if v == 0.0:
        raise ValueError("no execution error without a commanded burn")
    e = np.asarray(commanded_m_s, dtype=np.float64) / v
    e_perp = np.array([-e[1], e[0]])
    along = math.sqrt(model.sigma_r_m_s**2 + (v * model.sigma_s) ** 2)
    across = math.sqrt(model.sigma_a_m_s**2 + (v * model.sigma_p_rad) ** 2)
    return along * float(normal[0]) * e + across * float(normal[1]) * e_perp


def gates_normals(
    injection: ErrorDistribution,
    *,
    seed: int = GATES_NORMALS_SEED,
    confirm_frozen: bool = False,
) -> FloatArray:
    """Standard normal pairs for the K draws of an injection set, from
    default_rng((7401, injection seed)), shared by every Gates set; a held-out injection
    set requires the frozen protocol."""
    if injection.held_out:
        require_frozen(confirm_frozen)
    rng = np.random.default_rng((seed, injection.seed))
    return np.asarray(rng.standard_normal((injection.draws, 2)), dtype=np.float64)


def execution_models(
    injection: ErrorDistribution, *, confirm_frozen: bool = False
) -> tuple[GatesModel, ...]:
    """The Gates sets paired with an injection set (9.1); held-out ones need the freeze."""
    models = tuple(GATES_MODELS[name] for name in injection.executions)
    if any(m.held_out for m in models):
        require_frozen(confirm_frozen)
    return models


def calibrate_budget(
    survival_by_budget: Mapping[float, Sequence[float]], cfg: CorrectionConfig = _CORR
) -> float:
    """Smallest B in the grid whose median per-solution survival lies in the band, else the
    grid value whose median is closest to 0.5 (ties: the smaller B) (9.1)."""
    lo, hi = cfg.calibration_band
    medians = {b: float(np.median(survival_by_budget[b])) for b in cfg.budget_grid_m_s}
    for b in cfg.budget_grid_m_s:
        if lo <= medians[b] <= hi:
            return b
    return min(cfg.budget_grid_m_s, key=lambda b: (abs(medians[b] - 0.5), b))


# --- changed constraints (9.2) ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    family: str
    index: int
    h_lo_km: float
    h_hi_km: float
    t_max_days: float
    gamma_c_deg: float | None
    width_deg: float | None

    def satisfied_by(self, ev: Evaluation) -> bool:
        """Feasible, h_p in [h_lo, h_hi], t_ret <= T' and inside the corridor (if any)."""
        if not ev.feasible:
            return False
        if not (self.h_lo_km <= ev.perilune_altitude_km <= self.h_hi_km):
            return False
        if not ev.return_time_days <= self.t_max_days:
            return False
        if self.gamma_c_deg is not None and self.width_deg is not None:
            return abs(ev.return_flight_path_angle_deg - self.gamma_c_deg) <= self.width_deg
        return True


def _scaled(sample: FloatArray, column: int, bounds: tuple[float, float]) -> FloatArray:
    lo, hi = bounds
    return sample[:, column] * (hi - lo) + lo


def constraint_families(cfg: ConstraintFamilies = PROTOCOL.constraints) -> list[ConstraintSet]:
    """The 3 x 32 sets of F1 (geometry), F2 (corridor) and F3 (combined), in order."""
    m = int(math.log2(cfg.sets_per_family))
    if 2**m != cfg.sets_per_family:
        raise ValueError("sets_per_family must be a power of two (Sobol balance)")

    def sobol(d: int) -> FloatArray:
        sampler = qmc.Sobol(d=d, scramble=True, rng=np.random.default_rng(cfg.generation_seed))
        return np.asarray(sampler.random_base2(m), dtype=np.float64)

    nominal_lo, nominal_hi = PERILUNE_BAND_KM  # F2 keeps the nominal (5.1) band
    out: list[ConstraintSet] = []

    def geometry(s: FloatArray, c0: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        h_lo = 10.0 ** _scaled(s, c0, cfg.log_h_lo)
        h_hi = np.minimum(h_lo * 10.0 ** _scaled(s, c0 + 1, cfg.width_dex), cfg.h_cap_km)
        return h_lo, h_hi, _scaled(s, c0 + 2, cfg.t_max_days)

    s1 = sobol(3)
    lo1, hi1, t1 = geometry(s1, 0)
    for i in range(cfg.sets_per_family):
        out.append(ConstraintSet("F1", i, float(lo1[i]), float(hi1[i]), float(t1[i]), None, None))
    s2 = sobol(2)
    g2, w2 = _scaled(s2, 0, cfg.gamma_c_deg), _scaled(s2, 1, cfg.corridor_width_deg)
    for i in range(cfg.sets_per_family):
        out.append(ConstraintSet("F2", i, nominal_lo, nominal_hi, 10.0, float(g2[i]), float(w2[i])))
    s3 = sobol(5)
    lo3, hi3, t3 = geometry(s3, 0)
    g3, w3 = _scaled(s3, 3, cfg.gamma_c_deg), _scaled(s3, 4, cfg.corridor_width_deg)
    for i in range(cfg.sets_per_family):
        out.append(
            ConstraintSet(
                "F3", i, float(lo3[i]), float(hi3[i]), float(t3[i]), float(g3[i]), float(w3[i])
            )
        )
    return out


def split_indices(cfg: ConstraintFamilies = PROTOCOL.constraints) -> dict[str, tuple[int, ...]]:
    """Per family, the permuted set indices; the first `tuning_per_family` are tuning."""
    rng = np.random.default_rng(cfg.split_seed)
    return {
        f: tuple(int(i) for i in rng.permutation(cfg.sets_per_family)) for f in ("F1", "F2", "F3")
    }


def tuning_sets(cfg: ConstraintFamilies = PROTOCOL.constraints) -> list[ConstraintSet]:
    sets = {(c.family, c.index): c for c in constraint_families(cfg)}
    return [
        sets[(f, i)]
        for f, perm in split_indices(cfg).items()
        for i in perm[: cfg.tuning_per_family]
    ]


def held_out_sets(
    *, confirm_frozen: bool, cfg: ConstraintFamilies = PROTOCOL.constraints
) -> list[ConstraintSet]:
    """The held-out sets; only after the freeze (9.2: evaluated in the final analysis)."""
    require_frozen(confirm_frozen)
    sets = {(c.family, c.index): c for c in constraint_families(cfg)}
    return [
        sets[(f, i)]
        for f, perm in split_indices(cfg).items()
        for i in perm[cfg.tuning_per_family :]
    ]


def availability(elites: Sequence[Evaluation], sets: Sequence[ConstraintSet]) -> list[bool]:
    """Per set: does at least one catalog solution satisfy it (no correction)?"""
    return [any(c.satisfied_by(e) for e in elites) for c in sets]


def p2(
    elites: Sequence[Evaluation], sets: Sequence[ConstraintSet], satisfiable: Sequence[bool]
) -> tuple[float, int]:
    """(fraction of satisfiable sets the catalog satisfies, number of satisfiable sets)."""
    kept = [c for c, ok in zip(sets, satisfiable, strict=True) if ok]
    if not kept:
        return math.nan, 0
    hits = availability(elites, kept)
    return sum(hits) / len(kept), len(kept)
