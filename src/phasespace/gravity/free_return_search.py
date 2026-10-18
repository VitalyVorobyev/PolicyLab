"""Local targeting of free-return candidates from feasible map points (M3 stage S2).

From the feasible map points with the smallest targeting residual
F = [(h_p - 1000 km)/1000 km, (h_vp - 38 km)/10 km] (`free_return.targeting_residual`),
`scipy.optimize.least_squares` (trf, box bounds, forward differences) solves F = 0 in
(phase_deg, delta_v_km_s) with alpha = 0 and unit variable scaling (x_scale = 1: phase
in degrees, delta-v in km/s; x_scale = "jac" stalled in trials). Every objective evaluation is one propagation
(Jacobian columns included) and is counted against a per-start cap. A point where F is
undefined (e.g. a Moon collision) returns a large constant residual, which the trust
region rejects.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import least_squares

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    FreeReturnRecord,
    evaluate,
    targeting_residual,
)

UNDEFINED_RESIDUAL: Final = 1.0e3
"""Residual component returned where F is undefined (no perilune or no vacuum perigee)."""


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Declared S2 settings (design section 5)."""

    starts: int = 10
    max_propagations_per_start: int = 1000
    """Raised from the design's proposed 200 (user decision, 2026-10-18): with 200, 0 of 10
    starts converged on the S1 map because trf crawls along the narrow, curved h_vp valley;
    with 1000 all 10 converged (257-648 propagations each)."""
    convergence_abs_residual: float = 1e-5
    """Converged when max |F_i| <= this (10 m in h_p, 10 cm in h_vp) and feasible."""
    phase_bounds_deg: tuple[float, float] = (0.0, 360.0)
    delta_v_bounds_km_s: tuple[float, float] = (3.00, 3.25)
    tolerance: float = 1e-12
    """ftol = xtol = gtol of least_squares (the propagation cap normally stops it first)."""
    distinct_phase_deg: float = 1e-4
    distinct_delta_v_km_s: float = 1e-7
    """Converged starts closer than both of these are the same candidate."""


DEFAULT_SEARCH: Final = SearchSettings()


@dataclass(frozen=True, slots=True)
class StartResult:
    """Outcome of one least-squares start."""

    start: FreeReturnRecord
    final: FreeReturnRecord
    residual: tuple[float, float] | None
    propagations: int
    converged: bool
    message: str

    @property
    def residual_max_abs(self) -> float:
        if self.residual is None:
            return math.inf
        return max(abs(self.residual[0]), abs(self.residual[1]))


def _norm(record: FreeReturnRecord) -> float:
    f = targeting_residual(record)
    return math.inf if f is None else math.hypot(f[0], f[1])


def select_starts(records: Sequence[FreeReturnRecord], count: int) -> list[FreeReturnRecord]:
    """The `count` feasible records with the smallest ||F|| (ties keep map order)."""
    feasible = [r for r in records if r.feasible and targeting_residual(r) is not None]
    return sorted(feasible, key=_norm)[:count]


class _BudgetExhaustedError(Exception):
    pass


class _Objective:
    """Counting, caching residual function of (phase_deg, delta_v_km_s)."""

    def __init__(self, definition: FreeReturnDefinition, cap: int) -> None:
        self.definition = definition
        self.cap = cap
        self.propagations = 0
        self.records: dict[tuple[float, float], FreeReturnRecord] = {}

    def record(self, x: FloatArray) -> FreeReturnRecord:
        key = (float(x[0]), float(x[1]))
        cached = self.records.get(key)
        if cached is not None:
            return cached
        if self.propagations >= self.cap:
            raise _BudgetExhaustedError
        self.propagations += 1
        rec = evaluate(key[0], key[1], 0.0, self.definition)
        self.records[key] = rec
        return rec

    def __call__(self, x: FloatArray) -> FloatArray:
        f = targeting_residual(self.record(x))
        if f is None:
            return np.array([UNDEFINED_RESIDUAL, UNDEFINED_RESIDUAL])
        return np.array(f)

    def best(self) -> FreeReturnRecord:
        return min(self.records.values(), key=_norm)


def refine(
    start: FreeReturnRecord,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    settings: SearchSettings = DEFAULT_SEARCH,
) -> StartResult:
    """Solve F = 0 from one start; the start's own (map) propagation is not re-counted."""
    objective = _Objective(definition, settings.max_propagations_per_start)
    objective.records[(start.phase_deg, start.delta_v_km_s)] = start
    lo = (settings.phase_bounds_deg[0], settings.delta_v_bounds_km_s[0])
    hi = (settings.phase_bounds_deg[1], settings.delta_v_bounds_km_s[1])
    try:
        solution = least_squares(
            objective,
            np.array([start.phase_deg, start.delta_v_km_s]),
            jac="2-point",
            bounds=(lo, hi),
            method="trf",
            ftol=settings.tolerance,
            xtol=settings.tolerance,
            gtol=settings.tolerance,
            x_scale=1.0,
            max_nfev=settings.max_propagations_per_start,
        )
        final = objective.record(np.asarray(solution.x, dtype=np.float64))
        message = str(solution.message)
    except _BudgetExhaustedError:
        final = objective.best()
        message = f"propagation cap of {settings.max_propagations_per_start} reached"
    residual = targeting_residual(final)
    converged = (
        residual is not None
        and final.feasible
        and max(abs(residual[0]), abs(residual[1])) <= settings.convergence_abs_residual
    )
    return StartResult(
        start=start,
        final=final,
        residual=residual,
        propagations=objective.propagations,
        converged=converged,
        message=message,
    )


def distinct_converged(
    results: Sequence[StartResult], settings: SearchSettings = DEFAULT_SEARCH
) -> list[StartResult]:
    """Converged results with duplicates removed, ordered by increasing delta-v."""
    kept: list[StartResult] = []
    for r in sorted((r for r in results if r.converged), key=lambda r: r.final.delta_v_km_s):
        duplicate = any(
            abs(r.final.phase_deg - k.final.phase_deg) <= settings.distinct_phase_deg
            and abs(r.final.delta_v_km_s - k.final.delta_v_km_s) <= settings.distinct_delta_v_km_s
            for k in kept
        )
        if not duplicate:
            kept.append(r)
    return kept
