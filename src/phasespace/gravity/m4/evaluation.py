"""Common evaluation for every M4 method (protocol sections 4, 5 and 7.1).

- `to_physical` maps u in [0, 1]^3 to x = (theta deg, delta-v km/s, alpha deg) with the
  arithmetic of `scipy.stats.qmc.scale` (u * (hi - lo) + lo), so Sobol rows reproduce the
  M3 S4 map bit for bit.
- `evaluate_unit` is one propagation: the M3 departure and predicates
  (`free_return.propagate_departure` + `classify`), plus the scaled Jacobi gate of 5.2
  (a propagation failing it is `numerical_failure`), the time at which the vacuum perigee
  was evaluated (the corrector descriptor of CC) and the entry sense.
- `Evaluator` is the counting, caching evaluator shared by all methods: an exactly
  repeated u is neither re-propagated nor re-counted; every other evaluation (starts,
  finite-difference columns, line-search and trust-region trials, offspring) costs one
  unit of the run budget N, and a run stops at the first evaluation that would exceed
  N (`BudgetExhaustedError`). Batches are evaluated in input order; with an executor the
  propagations run in parallel but the log, the counts and every result are identical
  to a serial run, because each evaluation is a pure function of u.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from typing import Final

import numpy as np

from phasespace.gravity.dynamics import FloatArray, primary_distances
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    Outcome,
    classify,
    earth_angular_momentum_sign,
    propagate_departure,
)
from phasespace.gravity.m4.config import PROTOCOL, Descriptors, Objective, SearchBox, Unit3
from phasespace.gravity.propagation import (
    PropagationResult,
    SolverSettings,
    Termination,
)
from phasespace.gravity.units import time_to_days

type Cell = tuple[int, int]

_BOX: Final = PROTOCOL.box
_OBJ: Final = PROTOCOL.objective
_DESC: Final = PROTOCOL.descriptors


def to_physical(u: Unit3, box: SearchBox = _BOX) -> tuple[float, float, float]:
    """(theta deg, delta-v km/s, alpha deg) = u * (hi - lo) + lo (qmc.scale arithmetic)."""
    lo = np.array(box.lo)
    hi = np.array(box.hi)
    x = np.asarray(u, dtype=np.float64) * (hi - lo) + lo
    return float(x[0]), float(x[1]), float(x[2])


def to_unit(x: tuple[float, float, float], box: SearchBox = _BOX) -> Unit3:
    """Inverse of `to_physical` (rounding aside)."""
    lo = np.array(box.lo)
    hi = np.array(box.hi)
    u = (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
    return float(u[0]), float(u[1]), float(u[2])


def jacobi_scale(states: FloatArray, mu: float) -> float:
    """S = max_i [x^2 + y^2 + 2(1 - mu)/r1 + 2 mu/r2 + vx^2 + vy^2] over the states (5.2)."""
    if states.shape[0] == 0 or not np.all(np.isfinite(states)):
        return math.nan
    r1, r2 = primary_distances(states, mu)
    x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    terms = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 + vx * vx + vy * vy
    return float(np.max(terms))


def segment_drift_rel(result: PropagationResult) -> tuple[float, float]:
    """(drift_rel, S) of one propagation: max |C(s_i) - C(s_0)| / S at accepted steps."""
    mu = result.system.mu
    states = result.states
    s = jacobi_scale(states, mu)
    if not math.isfinite(s):
        return math.nan, s
    x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    r1, r2 = primary_distances(states, mu)
    c = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - vx * vx - vy * vy
    return float(np.max(np.abs(c - c[0]))) / s, s


def perigee_time(result: PropagationResult) -> float:
    """Normalized time at which `classify` evaluates the targeting vacuum perigee (or NaN).

    Mirrors `free_return.classify`: the first Earth periapsis after the closest Moon
    periapsis (before t_final), else the return-shell crossing if the shell was reached.
    """
    t_final = result.t_final
    moon_x = 1.0 - result.system.mu
    lunar = [p for p in result.periapses if p.name == "moon_periapsis" and p.t < t_final]
    if not lunar:
        return math.nan
    closest = min(lunar, key=lambda p: math.hypot(p.state[0] - moon_x, p.state[1]))
    later = [
        p for p in result.periapses if p.name == "earth_periapsis" and closest.t < p.t < t_final
    ]
    if later:
        return later[0].t
    if result.termination is Termination.EARTH_RETURN_SHELL:
        return t_final
    return math.nan


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One propagation of one decision vector. NaN marks an undefined quantity."""

    u: Unit3
    phase_deg: float
    delta_v_km_s: float
    impulse_angle_deg: float
    outcome: Outcome
    """M3 outcome, overridden to `numerical_failure` when the 5.2 gate fails."""
    m3_outcome: Outcome
    gate_passed: bool
    perilune_altitude_km: float
    lunar_encounter_time_days: float
    lunar_passage_sense: int
    return_time_days: float
    return_speed_km_s: float
    return_flight_path_angle_deg: float
    vacuum_perigee_altitude_km: float
    perigee_time_days: float
    entry_sense: int
    """Sign of the Earth-relative inertial angular momentum at the return (0: no return)."""
    jacobi_scale: float
    jacobi_drift_rel: float
    nfev: int

    @property
    def feasible(self) -> bool:
        return self.outcome is Outcome.FEASIBLE

    @property
    def objective_deg(self) -> float:
        """e = |gamma_S - gamma*| for a feasible evaluation, else NaN (5.4)."""
        if not self.feasible:
            return math.nan
        return abs(self.return_flight_path_angle_deg - _OBJ.gamma_target_deg)

    @property
    def descriptors(self) -> tuple[float, float]:
        """(log10 h_p, t_ret days); NaN where undefined."""
        h_p = self.perilune_altitude_km
        d1 = math.log10(h_p) if math.isfinite(h_p) and h_p > 0.0 else math.nan
        return d1, self.return_time_days


def evaluate_unit(
    u: Unit3,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    solver: SolverSettings | None = None,
) -> Evaluation:
    """One propagation of the departure at u, classified with the M3 predicates + gate."""
    for v in u:
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"u must lie in [0, 1]^3, got {u}")
    phase, dv, alpha = to_physical(u)
    result = propagate_departure(phase, dv, alpha, definition, solver=solver)
    record = classify(result, definition, phase_deg=phase, delta_v_km_s=dv, impulse_angle_deg=alpha)
    drift_rel, scale = segment_drift_rel(result)
    gate = math.isfinite(drift_rel) and drift_rel <= PROTOCOL.gate.jacobi_drift_rel_max
    outcome = record.outcome if gate else Outcome.NUMERICAL_FAILURE
    returned = result.termination is Termination.EARTH_RETURN_SHELL
    t_vp = perigee_time(result)
    return Evaluation(
        u=(float(u[0]), float(u[1]), float(u[2])),
        phase_deg=phase,
        delta_v_km_s=dv,
        impulse_angle_deg=alpha,
        outcome=outcome,
        m3_outcome=record.outcome,
        gate_passed=gate,
        perilune_altitude_km=record.perilune_altitude_km,
        lunar_encounter_time_days=record.lunar_encounter_time_days,
        lunar_passage_sense=record.lunar_passage_sense,
        return_time_days=record.return_time_days,
        return_speed_km_s=record.return_speed_km_s,
        return_flight_path_angle_deg=record.return_flight_path_angle_deg,
        vacuum_perigee_altitude_km=record.vacuum_perigee_altitude_km,
        perigee_time_days=time_to_days(t_vp, definition.system) if math.isfinite(t_vp) else t_vp,
        entry_sense=(
            earth_angular_momentum_sign(result.final_state, result.t_final, definition.system.mu)
            if returned
            else 0
        ),
        jacobi_scale=scale,
        jacobi_drift_rel=drift_rel,
        nfev=record.nfev,
    )


# --- objective, residuals, merit, descriptors ------------------------------------------------


def residual(ev: Evaluation, obj: Objective = _OBJ) -> FloatArray:
    """r = (r1, r2) of section 6, or (UNDEFINED, UNDEFINED) where h_vp or h_p is undefined."""
    h_vp = ev.vacuum_perigee_altitude_km
    h_p = ev.perilune_altitude_km
    if not (math.isfinite(h_vp) and math.isfinite(h_p) and h_p > 0.0):
        return np.array([obj.undefined_residual, obj.undefined_residual])
    lh = math.log10(h_p)
    lo, hi = obj.perilune_log_band
    r1 = (h_vp - obj.vacuum_perigee_target_km) / obj.vacuum_perigee_scale_km
    r2 = (max(0.0, lo - lh) + max(0.0, lh - hi)) / obj.perilune_log_scale
    return np.array([r1, r2])


def residual_defined(ev: Evaluation) -> bool:
    h_vp, h_p = ev.vacuum_perigee_altitude_km, ev.perilune_altitude_km
    return math.isfinite(h_vp) and math.isfinite(h_p) and h_p > 0.0


def merit(ev: Evaluation, obj: Objective = _OBJ) -> float:
    """J = e if feasible, else 100 + min(||r||, 1000): every feasible point ranks first."""
    if ev.feasible:
        return ev.objective_deg
    norm = float(np.linalg.norm(residual(ev, obj)))
    return obj.infeasible_merit_offset + min(norm, obj.infeasible_merit_cap)


def bin_index(value: float, bounds: tuple[float, float], bins: int) -> int:
    """k = floor(bins (d - lo) / (hi - lo)), clipped to [0, bins - 1] (5.5)."""
    lo, hi = bounds
    k = math.floor(bins * (value - lo) / (hi - lo))
    return min(bins - 1, max(0, k))


def cell_of(ev: Evaluation, bins: int = _DESC.bins, desc: Descriptors = _DESC) -> Cell | None:
    """Descriptor cell of a feasible evaluation (None for an infeasible one)."""
    if not ev.feasible:
        return None
    d1, d2 = ev.descriptors
    return bin_index(d1, desc.d1_bounds, bins), bin_index(d2, desc.d2_bounds, bins)


def cell_centre(
    cell: Cell, bins: int = _DESC.bins, desc: Descriptors = _DESC
) -> tuple[tuple[float, float], tuple[float, float]]:
    """((c1, c2), (w1, w2)): the cell's centre and widths in descriptor units."""
    (lo1, hi1), (lo2, hi2) = desc.d1_bounds, desc.d2_bounds
    w1, w2 = (hi1 - lo1) / bins, (hi2 - lo2) / bins
    return (lo1 + (cell[0] + 0.5) * w1, lo2 + (cell[1] + 0.5) * w2), (w1, w2)


# --- the counting, caching evaluator -------------------------------------------------------


class BudgetExhaustedError(Exception):
    """The next evaluation would exceed the run budget."""


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One propagation of a run: its 0-based index, result, purpose tag and wall time."""

    index: int
    evaluation: Evaluation
    tag: str
    seconds: float


type EvaluateFn = Callable[[Unit3], Evaluation]


def _timed(fn: EvaluateFn, u: Unit3) -> tuple[Evaluation, float]:
    start = time.perf_counter()
    ev = fn(u)
    return ev, time.perf_counter() - start


def _key(u: Iterable[float]) -> Unit3:
    a, b, c = (float(v) for v in u)
    return a, b, c


class Evaluator:
    """Counting, caching evaluator with a hard propagation budget (protocol 6, 7.1)."""

    def __init__(
        self,
        budget: int,
        fn: EvaluateFn = evaluate_unit,
        *,
        executor: Executor | None = None,
        chunksize: int = 4,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self.budget = budget
        self.fn = fn
        self.executor = executor
        self.chunksize = chunksize
        self.log: list[LogEntry] = []
        self.cache: dict[Unit3, Evaluation] = {}
        self.cache_hits = 0

    @property
    def used(self) -> int:
        return len(self.log)

    @property
    def remaining(self) -> int:
        return self.budget - len(self.log)

    def is_cached(self, u: Iterable[float]) -> bool:
        return _key(u) in self.cache

    def _store(self, key: Unit3, ev: Evaluation, tag: str, seconds: float) -> None:
        self.cache[key] = ev
        self.log.append(LogEntry(len(self.log), ev, tag, seconds))

    def evaluate(self, u: Iterable[float], tag: str) -> Evaluation:
        """The evaluation at u; propagates (and counts) only if u was never evaluated."""
        key = _key(u)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.remaining <= 0:
            raise BudgetExhaustedError
        ev, seconds = _timed(self.fn, key)
        self._store(key, ev, tag, seconds)
        return ev

    def evaluate_batch(self, us: Sequence[Iterable[float]], tag: str) -> list[Evaluation | None]:
        """Evaluate in order; new points beyond the budget are not evaluated (None)."""
        keys = [_key(u) for u in us]
        new: list[Unit3] = []
        seen: set[Unit3] = set()
        for key in keys:
            if key not in self.cache and key not in seen:
                seen.add(key)
                new.append(key)
        new = new[: max(0, self.remaining)]
        task = partial(_timed, self.fn)
        if self.executor is None or len(new) < 2:
            results = [task(k) for k in new]
        else:
            results = list(self.executor.map(task, new, chunksize=self.chunksize))
        for key, (ev, seconds) in zip(new, results, strict=True):
            self._store(key, ev, tag, seconds)
        fresh = set(new)
        out: list[Evaluation | None] = []
        for key in keys:
            ev = self.cache.get(key)
            if ev is not None and key not in fresh:
                self.cache_hits += 1
            fresh.discard(key)
            out.append(ev)
        return out

    def evaluations(self) -> list[Evaluation]:
        return [entry.evaluation for entry in self.log]
