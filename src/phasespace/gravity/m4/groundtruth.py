"""Coverage ground truth with adaptive band-edge refinement (protocol 5.3).

G is the set of descriptor cells occupied by any feasible propagation from:
(a) the GT map, 2^20 scrambled Sobol points with seed 20261018 (its 2^17 prefix must
reproduce the M3 S4 map, check K1); (b) level 1, 32 Sobol points (seed 20261019) in a
box of half-width 2^(-20/3) in u around every feasible GT-map sample; (c) levels 2 and
3, the same around every feasible point of the previous level whose cell was not yet in
G before that level, with half the previous half-width (seeds 20261020, 20261021);
(d) every feasible propagation of every method run (added by the analysis).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

from phasespace.gravity.free_return import FreeReturnRecord
from phasespace.gravity.m4.config import PROTOCOL, GroundTruth, Unit3
from phasespace.gravity.m4.evaluation import Cell, Evaluation, cell_of


def refinement_box(
    centre: Unit3, level: int, gt: GroundTruth = PROTOCOL.ground_truth
) -> list[Unit3]:
    """The level's 32 Sobol points in the box centred on `centre`, clipped to [0, 1]^3."""
    if not 1 <= level <= len(gt.refinement_seeds):
        raise ValueError(f"level must be 1..{len(gt.refinement_seeds)}")
    m = int(math.log2(gt.refinement_points))
    sampler = qmc.Sobol(
        d=3, scramble=True, rng=np.random.default_rng(gt.refinement_seeds[level - 1])
    )
    unit = np.asarray(sampler.random_base2(m), dtype=np.float64)
    half = gt.refinement_half_width / 2.0 ** (level - 1)
    pts = np.clip(np.asarray(centre) + (2.0 * unit - 1.0) * half, 0.0, 1.0)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]


@dataclass(slots=True)
class GroundTruthCells:
    """Cells per source, for the saturation report (5.3)."""

    bins: int
    by_source: dict[str, set[Cell]] = field(default_factory=dict[str, set[Cell]])

    def add(self, source: str, evaluations: Sequence[Evaluation]) -> None:
        cells = self.by_source.setdefault(source, set())
        for e in evaluations:
            c = cell_of(e, self.bins)
            if c is not None:
                cells.add(c)

    def union(self, *sources: str) -> set[Cell]:
        out: set[Cell] = set()
        for s in sources or tuple(self.by_source):
            out |= self.by_source.get(s, set())
        return out


def next_level_centres(
    level_points: Sequence[Evaluation], known: set[Cell], bins: int
) -> list[Unit3]:
    """Feasible points of a level whose cell was not in G before that level."""
    out: list[Unit3] = []
    for e in level_points:
        c = cell_of(e, bins)
        if c is not None and c not in known:
            out.append(e.u)
    return out


def saturated(prefix_cells: int, all_cells: int, gt: GroundTruth = PROTOCOL.ground_truth) -> bool:
    """K7: False when all sources exceed the 2^19-prefix count by more than 10 %."""
    return all_cells <= (1.0 + gt.saturation_tolerance) * prefix_cells


@dataclass(frozen=True, slots=True)
class PrefixMismatch:
    row: int
    field: str
    gt_value: object
    m3_value: object


def compare_with_m3(
    evaluations: Sequence[Evaluation], m3_records: Sequence[FreeReturnRecord]
) -> list[PrefixMismatch]:
    """K1: the GT map's prefix must reproduce the M3 S4 map's outcomes and h_p, t_ret,
    gamma_S exactly (JSON floats round-trip exactly, so 'stored precision' is equality).

    The M3 outcome is compared with the M4 record's `m3_outcome` (before the 5.2 gate).
    """
    if len(evaluations) < len(m3_records):
        raise ValueError("fewer GT evaluations than M3 rows")
    out: list[PrefixMismatch] = []
    for i, (e, r) in enumerate(zip(evaluations, m3_records, strict=False)):
        pairs: tuple[tuple[str, object, object], ...] = (
            ("phase_deg", e.phase_deg, r.phase_deg),
            ("delta_v_km_s", e.delta_v_km_s, r.delta_v_km_s),
            ("impulse_angle_deg", e.impulse_angle_deg, r.impulse_angle_deg),
            ("outcome", e.m3_outcome.value, r.outcome.value),
            ("perilune_altitude_km", e.perilune_altitude_km, r.perilune_altitude_km),
            ("return_time_days", e.return_time_days, r.return_time_days),
            (
                "return_flight_path_angle_deg",
                e.return_flight_path_angle_deg,
                r.return_flight_path_angle_deg,
            ),
        )
        for name, a, b in pairs:
            both_nan = (
                isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
            )
            if not both_nan and a != b:
                out.append(PrefixMismatch(i, name, a, b))
    return out


def compute_ground_truth(
    evaluate_batch: Callable[[Sequence[Unit3], str], list[Evaluation | None]],
    map_points: Sequence[Unit3],
    gt: GroundTruth = PROTOCOL.ground_truth,
    bins: int = PROTOCOL.descriptors.bins,
) -> tuple[GroundTruthCells, list[Evaluation], list[Evaluation]]:
    """Sources (a)-(c): returns (cells per source, map evaluations, feasible refinements).

    Sources: 'map_prefix' (the 2^19 prefix), 'map', 'level1', 'level2', 'level3'.
    """
    cells = GroundTruthCells(bins)
    map_evals = [e for e in evaluate_batch(map_points, "gt_map") if e is not None]
    if len(map_evals) != len(map_points):
        raise RuntimeError("the ground-truth evaluator ran out of budget")
    cells.add("map_prefix", map_evals[: 2**gt.saturation_prefix_log2])
    cells.add("map", map_evals)
    known = set(cells.union("map"))
    centres = [e.u for e in map_evals if e.feasible]
    refined: list[Evaluation] = []
    for level in range(1, len(gt.refinement_seeds) + 1):
        points = [p for c in centres for p in refinement_box(c, level, gt)]
        evals = [e for e in evaluate_batch(points, f"gt_level{level}") if e is not None]
        cells.add(f"level{level}", evals)
        refined.extend(e for e in evals if e.feasible)
        centres = next_level_centres(evals, known, bins)
        known |= cells.by_source[f"level{level}"]
    return cells, map_evals, refined
