"""Per-run metrics of protocol section 10, computed from a catalog and its run log."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from phasespace.gravity.free_return import Outcome
from phasespace.gravity.m4.catalog import Elite
from phasespace.gravity.m4.evaluation import Cell, Evaluation


def p1(survivals: Sequence[float]) -> float:
    """P1 = sum_i s_i over the catalog's solutions."""
    return float(sum(survivals))


def coverage(catalog: Mapping[Cell, Elite], ground_truth: set[Cell]) -> float:
    """|catalog cells & G| / |G|."""
    if not ground_truth:
        return math.nan
    return len(set(catalog) & ground_truth) / len(ground_truth)


def quality(catalog: Mapping[Cell, Elite]) -> dict[str, float]:
    """Mean and median e, the 10th percentile of e, and QD-score sum (90 deg - e)."""
    e = np.array([el.objective_deg for el in catalog.values()], dtype=np.float64)
    if e.size == 0:
        return {"mean": math.nan, "median": math.nan, "p10": math.nan, "qd_score": 0.0}
    return {
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "p10": float(np.quantile(e, 0.1)),
        "qd_score": float(np.sum(90.0 - e)),
    }


def feasibility_yield(log: Sequence[Evaluation]) -> tuple[float, int | None]:
    """(feasible propagations / propagations, index of the first feasible one)."""
    if not log:
        return math.nan, None
    first = next((i for i, e in enumerate(log) if e.feasible), None)
    return sum(e.feasible for e in log) / len(log), first


def diversity_classes(catalog: Mapping[Cell, Elite], log: Sequence[Evaluation]) -> int:
    """Distinct (lunar passage sense, entry sense) classes among the elites."""
    return len(
        {(log[el.index].lunar_passage_sense, log[el.index].entry_sense) for el in catalog.values()}
    )


def numerical_failures(log: Sequence[Evaluation]) -> tuple[int, int]:
    """(propagations classified numerical_failure, of which by the 5.2 gate alone)."""
    total = sum(e.outcome is Outcome.NUMERICAL_FAILURE for e in log)
    gate = sum(not e.gate_passed and e.m3_outcome is not Outcome.NUMERICAL_FAILURE for e in log)
    return total, gate
