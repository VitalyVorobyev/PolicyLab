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
  dv_{k+1} = dv_k - J^-1 R(dv_k), at most 3 times; converged when |R| <= 1 km per
  component, diverged when |dv| > 100 m/s; cond(J) > 1e8 fails every draw;
- the trajectory after a correction is the arc 0 -> t_c joined to a new propagation
  from t_c; predicates are evaluated on the joined trajectory (`classify`) and the
  Jacobi gate per segment (the impulse changes C).
- D24: the converged *commanded* correction dv_m is then executed with Gates-model
  errors (`gates_error`, parameters `config.GatesModel`), and survival at budget B
  needs |dv_m| <= B and a feasible trajectory after the *executed* correction.

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
class DrawResult:
    uncorrected_feasible: bool
    converged: bool
    delta_v_required_m_s: float
    """|dv| at convergence; inf when not converged or diverged."""
    corrected_feasible: bool
    """The trajectory with the commanded (exact) converged dv satisfies 5.1 and 5.2."""
    executed_feasible: bool
    """D24: the trajectory with the *executed* correction (commanded dv plus Gates-model
    error) satisfies 5.1 and 5.2. Equals `corrected_feasible` without an execution model
    or when no burn is commanded (dv = 0)."""
    executed_delta_v_m_s: tuple[float, float]
    """Executed MCC (rotating-frame components, m/s); (nan, nan) if not converged."""
    iterations: int
    initial_residual_km: tuple[float, float]
    propagations: int
    """Arc plus tail propagations of this draw (each a partial propagation)."""
    nfev: int

    def survives(self, budget_m_s: float, cfg: CorrectionConfig = _CORR) -> bool:
        """Success at budget B (9.1); see `CorrectionConfig.uncorrected_feasible_counts`."""
        if self.uncorrected_feasible and (budget_m_s == 0.0 or cfg.uncorrected_feasible_counts):
            return True
        return (
            budget_m_s > 0.0
            and self.converged
            and self.executed_feasible
            and self.delta_v_required_m_s <= budget_m_s
        )


@dataclass(frozen=True, slots=True)
class SolutionRobustness:
    nominal_targets: tuple[float, float]
    jacobian_km_per_m_s: tuple[tuple[float, float], tuple[float, float]]
    condition: float
    singular: bool
    draws: tuple[DrawResult, ...]
    setup_propagations: int
    setup_nfev: int

    def survival(self, budget_m_s: float, cfg: CorrectionConfig = _CORR) -> float:
        """s_i(B): the fraction of draws that succeed at budget B."""
        if not self.draws:
            return math.nan
        return sum(d.survives(budget_m_s, cfg) for d in self.draws) / len(self.draws)


def _undefined(values: Iterable[float]) -> bool:
    return not all(math.isfinite(v) for v in values)


def solution_robustness(
    elite: Evaluation,
    draws: Sequence[ErrorDraw],
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    cfg: CorrectionConfig = _CORR,
    *,
    execution: GatesModel | None = None,
    normals: FloatArray | None = None,
) -> SolutionRobustness:
    """Execution-error tolerance of one feasible catalog solution under the given draws.

    With `execution` (D24) the converged commanded MCC of draw k is executed with the
    Gates-model error built from `normals[k]` (standard normal pairs, `gates_normals`);
    without it the correction is applied exactly (diagnostic only).
    """
    if execution is not None and (normals is None or normals.shape != (len(draws), 2)):
        raise ValueError("an execution model needs one standard normal pair per draw")
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
            execution=execution,
            normal=None if normals is None or execution is None else normals[k],
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
    execution: GatesModel | None,
    normal: FloatArray | None,
    cfg: CorrectionConfig = _CORR,
) -> DrawResult:
    t_c = time_from_days(cfg.t_c_days, definition.system)
    phase, dv, alpha = executed_departure(
        elite.phase_deg, elite.delta_v_km_s, elite.impulse_angle_deg, draw, definition
    )
    arc = _arc(phase, dv, alpha, definition, t_c)
    nfev = arc.nfev
    if arc.termination is not Termination.HORIZON:
        # The trajectory ended (collision or shell) before t_c: nothing to correct.
        nan2 = (math.nan, math.nan)
        return DrawResult(False, False, math.inf, False, False, nan2, 0, nan2, 1, nfev)
    delta = np.zeros(2)
    leg = _leg(arc, delta, definition, t_c)
    nfev += leg.nfev
    propagations = 2
    uncorrected = leg.feasible

    def residual_of(lg: Leg) -> FloatArray:
        return np.array([lg.targets[0] - nominal[0], lg.targets[1] - nominal[1]])

    r = residual_of(leg)
    r0 = (float(r[0]), float(r[1]))
    iterations = 0
    converged = False
    while True:
        if _undefined(r):
            break
        if float(np.max(np.abs(r))) <= cfg.tolerance_km:
            converged = True
            break
        if not arc_ok or iterations >= cfg.max_iterations:
            break
        delta = delta - np.linalg.solve(jac, r)
        iterations += 1
        if float(np.linalg.norm(delta)) > cfg.divergence_m_s:
            break
        leg = _leg(arc, delta, definition, t_c)
        nfev += leg.nfev
        propagations += 1
        r = residual_of(leg)
    executed = delta
    executed_feasible = converged and leg.feasible
    if converged and execution is not None and normal is not None and np.any(delta != 0.0):
        executed = delta + gates_error(delta, normal, execution)
        executed_leg = _leg(arc, executed, definition, t_c)
        nfev += executed_leg.nfev
        propagations += 1
        executed_feasible = executed_leg.feasible
    return DrawResult(
        uncorrected_feasible=uncorrected,
        converged=converged,
        delta_v_required_m_s=float(np.linalg.norm(delta)) if converged else math.inf,
        corrected_feasible=converged and leg.feasible,
        executed_feasible=executed_feasible,
        executed_delta_v_m_s=(
            (float(executed[0]), float(executed[1])) if converged else (math.nan, math.nan)
        ),
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
    model: GatesModel, injection: ErrorDistribution, *, confirm_frozen: bool = False
) -> FloatArray:
    """Standard normal pairs for the K draws of an injection set, from
    default_rng((model.seed, injection.seed)); held-out sets require the frozen protocol."""
    if model.held_out or injection.held_out:
        require_frozen(confirm_frozen)
    rng = np.random.default_rng((model.seed, injection.seed))
    return np.asarray(rng.standard_normal((injection.draws, 2)), dtype=np.float64)


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
