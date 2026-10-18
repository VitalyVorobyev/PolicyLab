"""Catalog construction (protocol 5.5), elite re-check (5.6) and anytime checkpoints (7.3).

Every method's output is built the same way from its full propagation log: per
descriptor cell, the feasible evaluation with the lowest e among *all* propagations the
method made (ties: the earlier propagation index). The H2 elite-selection margin of an
elite is the e of the next-best feasible evaluation in its cell minus the elite's e
(NaN for a cell with a single feasible evaluation).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from phasespace.gravity.free_return import DEFAULT_DEFINITION, FreeReturnDefinition
from phasespace.gravity.m4.config import PROTOCOL, NumericalGate
from phasespace.gravity.m4.evaluation import Cell, Evaluation, cell_of, evaluate_unit
from phasespace.gravity.propagation import SolverSettings

RECHECK_SOLVER = SolverSettings(method="DOP853", rtol=1e-13, atol=1e-13)


@dataclass(frozen=True, slots=True)
class Elite:
    cell: Cell
    index: int
    """Propagation index of the elite in its run's log."""
    objective_deg: float
    margin_deg: float


def cell_members(
    log: Sequence[Evaluation], bins: int = PROTOCOL.descriptors.bins, limit: int | None = None
) -> dict[Cell, list[int]]:
    """Feasible propagation indices per cell, sorted by (e, index); first `limit` only."""
    members: dict[Cell, list[int]] = {}
    end = len(log) if limit is None else min(limit, len(log))
    for i in range(end):
        cell = cell_of(log[i], bins)
        if cell is not None:
            members.setdefault(cell, []).append(i)
    for indices in members.values():
        indices.sort(key=lambda i: (log[i].objective_deg, i))
    return dict(sorted(members.items()))


def build_catalog(
    log: Sequence[Evaluation], bins: int = PROTOCOL.descriptors.bins, limit: int | None = None
) -> dict[Cell, Elite]:
    """Cell -> elite over the first `limit` propagations (all if None), cells sorted."""
    out: dict[Cell, Elite] = {}
    for cell, indices in cell_members(log, bins, limit).items():
        best = indices[0]
        e = log[best].objective_deg
        margin = log[indices[1]].objective_deg - e if len(indices) > 1 else math.nan
        out[cell] = Elite(cell, best, e, margin)
    return out


def checkpoint_catalogs(
    log: Sequence[Evaluation], checkpoints: Sequence[int], bins: int = PROTOCOL.descriptors.bins
) -> dict[int, dict[Cell, Elite]]:
    """Catalog snapshots after the first n propagations for each checkpoint n (7.3)."""
    return {n: build_catalog(log, bins, n) for n in checkpoints}


@dataclass(frozen=True, slots=True)
class Recheck:
    """Re-propagation of an elite at DOP853 1e-13 (5.6)."""

    index: int
    passed: bool
    same_outcome: bool
    same_cell: bool
    perilune_diff_km: float
    return_time_diff_s: float
    gamma_diff_deg: float


def recheck(
    elite: Evaluation,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    gate: NumericalGate = PROTOCOL.gate,
    bins: int = PROTOCOL.descriptors.bins,
    *,
    index: int = -1,
    tight: Callable[[Evaluation], Evaluation] | None = None,
) -> Recheck:
    """Same outcome, |dh_p| <= 10 m, |dt_ret| <= 0.1 s, |dgamma| <= 1e-3 deg, same cell."""
    redo = (
        tight(elite)
        if tight is not None
        else evaluate_unit(elite.u, definition, solver=RECHECK_SOLVER)
    )
    dh = abs(redo.perilune_altitude_km - elite.perilune_altitude_km)
    dt = abs(redo.return_time_days - elite.return_time_days) * 86400.0
    dg = abs(redo.return_flight_path_angle_deg - elite.return_flight_path_angle_deg)
    same_outcome = redo.outcome is elite.outcome
    same_cell = cell_of(redo, bins) == cell_of(elite, bins)

    def ok(value: float, threshold: float) -> bool:
        return math.isfinite(value) and value <= threshold

    passed = (
        same_outcome
        and same_cell
        and ok(dh, gate.recheck_perilune_km)
        and ok(dt, gate.recheck_return_time_s)
        and ok(dg, gate.recheck_gamma_deg)
    )
    return Recheck(index, passed, same_outcome, same_cell, dh, dt, dg)


def rechecked_catalog(
    log: Sequence[Evaluation],
    check: Callable[[int], Recheck],
    bins: int = PROTOCOL.descriptors.bins,
    limit: int | None = None,
    max_replacements: int = PROTOCOL.gate.recheck_max_replacements,
) -> tuple[dict[Cell, Elite], list[Recheck]]:
    """Catalog after 5.6: a failing elite is replaced by the cell's next best, re-checked.

    `check(i)` re-checks propagation i. Returns the catalog and every re-check made.
    """
    out: dict[Cell, Elite] = {}
    made: list[Recheck] = []
    for cell, indices in cell_members(log, bins, limit).items():
        for rank, i in enumerate(indices[: 1 + max_replacements]):
            result = check(i)
            made.append(result)
            if result.passed:
                e = log[i].objective_deg
                rest = indices[rank + 1 :]
                margin = log[rest[0]].objective_deg - e if rest else math.nan
                out[cell] = Elite(cell, i, e, margin)
                break
    return out, made


def lsoda_sample(count: int, gate: NumericalGate = PROTOCOL.gate) -> list[int]:
    """Seeded 5 % sample (seed 7301) of `count` pooled elites for the full M3 S3 check."""
    if count <= 0:
        return []
    k = math.ceil(gate.lsoda_fraction * count)
    rng = np.random.default_rng(gate.lsoda_seed)
    return sorted(int(i) for i in rng.choice(count, size=k, replace=False))
