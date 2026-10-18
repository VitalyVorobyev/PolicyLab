"""Synthetic stored M4 runs (method, robustness and ground-truth runs) for analysis tests.

Every run is written with the production writers (`write_method_runs`, `build_manifest`,
`write_run_directory`) and the production column layouts, under the real (frozen)
protocol, so the analysis reads them exactly as it would read real runs. Method logs use
the analytic stand-in `m4_toy.toy_evaluate`; robustness rows are specified per elite as
(free, corrected): `free` draws feasible without correction, then `corrected` draws whose
commanded 2.5 m/s correction is feasible after execution under the primary Gates set and
infeasible under the stress set, then failures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from m4_toy import surface, toy_evaluate

from phasespace.gravity.export import write_run_directory
from phasespace.gravity.free_return_map import dumps_table, sample_table
from phasespace.gravity.m4.catalog import build_catalog
from phasespace.gravity.m4.config import ERROR_DISTRIBUTIONS, method_grid, to_jsonable
from phasespace.gravity.m4.evaluation import Evaluation, Evaluator, LogEntry
from phasespace.gravity.m4.methods import MethodRun
from phasespace.gravity.m4.runs import (
    build_manifest,
    evaluations_table,
    method_run_id,
    robustness_columns,
    robustness_run_id,
    write_method_runs,
)

FAKE_COMMIT: Final = "1" * 40
CORRECTED_M_S: Final = 2.5
NFEV_FREE: Final = 1000
NFEV_CORRECTED: Final = 3000
NFEV_FAILED: Final = 500

type Spec = Callable[[int], tuple[int, int]]
"""Elite rank (cell order) -> (free, corrected) draw counts."""


def feasible(u0: float, u2: float, gamma: float) -> Evaluation:
    """A feasible toy evaluation on the slab, with the return flight-path angle set."""
    e = toy_evaluate((u0, surface(u0, u2), u2))
    assert e.feasible
    return replace(e, return_flight_path_angle_deg=gamma)


WORLD_LOG: Final[tuple[Evaluation, ...]] = (
    toy_evaluate((0.05, 0.5, 0.5)),  # undefined r: infeasible
    feasible(0.25, 0.30, -7.0),  # cell (3, 2), e = 0.5
    feasible(0.35, 0.50, -6.0),  # (5, 3)
    feasible(0.45, 0.10, -6.5),  # (1, 4), e = 0: the elite
    toy_evaluate((0.5, 0.9, 0.5)),  # far from the slab: infeasible
    feasible(0.251, 0.301, -6.8),  # (3, 2), e = 0.3: the elite, margin 0.2
    feasible(0.451, 0.101, -5.0),  # (1, 4), e = 1.5: margin 1.5 for the (1, 4) elite
    feasible(0.55, 0.70, -8.0),  # (7, 6)
    feasible(0.65, 0.20, -6.4),  # (2, 7)
    feasible(0.75, 0.90, -7.5),  # (9, 8)
    feasible(0.85, 0.40, -6.0),  # (4, 9): only method runs find this cell
    toy_evaluate((0.3, 0.2, 0.5)),  # infeasible
)
"""The log every synthetic method run shares (7 catalog cells)."""
WORLD_BUDGET: Final = 16


def write_method(
    runs_dir: Path,
    *,
    stage: str,
    method: str,
    config_index: int,
    seed: int,
    log: Sequence[Evaluation] = WORLD_LOG,
    budget: int = WORLD_BUDGET,
    git_dirty: bool = False,
    run_id: str | None = None,
) -> str:
    lookup = {e.u: e for e in log}
    ev = Evaluator(budget, fn=lambda u: lookup[u])
    for e in log:
        ev.evaluate(e.u, "test")
    cfg = method_grid("CC" if method == "CCstar" else method)[config_index]
    run = MethodRun(
        method=method,
        seed=seed,
        budget=budget,
        config={} if cfg is None else to_jsonable(cfg),
        evaluator=ev,
        wall_seconds=1.0,
        status="stopped_early",
    )
    rid = run_id or method_run_id(stage, method, config_index, seed)
    write_method_runs(
        runs_dir / rid,
        [run],
        stage=stage,
        experiment_id="test",
        git_commit=FAKE_COMMIT,
        git_dirty=git_dirty,
    )
    return rid


def draw_rows(
    cell: tuple[int, int], index: int, free: int, corrected: int, k: int, gates: int
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for draw in range(k):
        if draw < free:
            row = [cell[0], cell[1], index, draw, 0, 1, 0.0, *([1] * gates), 1, 0, 1, 0, NFEV_FREE]
        elif draw < free + corrected:
            executed = [1, *([0] * (gates - 1))]
            dv = CORRECTED_M_S / 1000.0
            row = [cell[0], cell[1], index, draw, 0, 0, dv, *executed, 0, 0, 1, 3, NFEV_CORRECTED]
        else:
            row = [
                cell[0],
                cell[1],
                index,
                draw,
                0,
                0,
                0.0,
                *([0] * gates),
                0,
                0,
                0,
                0,
                NFEV_FAILED,
            ]
        rows.append(row)
    return rows


def write_robustness(
    runs_dir: Path,
    source_run: str,
    errors: str,
    spec: Spec,
    *,
    log: Sequence[Evaluation] = WORLD_LOG,
    rechecks: int | None = None,
    recheck_failures: int = 0,
    git_dirty: bool = False,
) -> str:
    """The robustness run of `source_run`: its catalog elites (all pass the re-check) with
    per-draw rows following `spec`."""
    dist = ERROR_DISTRIBUTIONS[errors]
    catalog = build_catalog(log)
    rows: list[list[Any]] = []
    for rank, el in enumerate(catalog.values()):
        free, corrected = spec(rank)
        rows += draw_rows(el.cell, el.index, free, corrected, dist.draws, len(dist.executions))
    run_id = robustness_run_id(source_run, errors)
    manifest = build_manifest(
        run_id,
        stage="robustness",
        experiment_id="test",
        seed=dist.seed,
        methods=[],
        metrics={
            "elites": {"value": len(catalog), "unit": "count", "definition": "elites"},
            "rechecks": {
                "value": len(catalog) if rechecks is None else rechecks,
                "unit": "count",
                "definition": "re-checks",
            },
            "recheck_failures": {"value": recheck_failures, "unit": "count", "definition": "x"},
        },
        limitations=[],
        git_commit=FAKE_COMMIT,
        git_dirty=git_dirty,
        created_at=datetime(2026, 10, 18, tzinfo=UTC),
    )
    table = dumps_table(sample_table(run_id, robustness_columns(dist.executions), rows))
    write_run_directory(runs_dir / run_id, manifest, [("sample_table", "robustness.json", table)])
    return run_id


def write_groundtruth(
    runs_dir: Path, entries: Sequence[LogEntry], run_id: str = "gravity-m4-groundtruth-v1"
) -> str:
    manifest = build_manifest(
        run_id,
        stage="ground_truth",
        experiment_id="test",
        seed=20261018,
        methods=[],
        metrics={},
        limitations=[],
        git_commit=FAKE_COMMIT,
        git_dirty=False,
        created_at=datetime(2026, 10, 18, tzinfo=UTC),
    )
    table = evaluations_table(run_id, entries)
    write_run_directory(runs_dir / run_id, manifest, [("sample_table", "evaluations.json", table)])
    return run_id


GT_ENTRIES: Final[tuple[LogEntry, ...]] = (
    LogEntry(3, WORLD_LOG[1], "gt_map", 0.0),  # (3, 2)
    LogEntry(10, WORLD_LOG[2], "gt_map", 0.0),  # (5, 3)
    LogEntry(17, WORLD_LOG[3], "gt_map", 0.0),  # (1, 4)
    LogEntry(2**19 - 1, WORLD_LOG[7], "gt_map", 0.0),  # (7, 6): last prefix row
    LogEntry(2**19, WORLD_LOG[8], "gt_map", 0.0),  # (2, 7): map, not the 2^19 prefix
    LogEntry(2**20 + 4, WORLD_LOG[9], "gt_level1", 0.0),  # (9, 8): refinement only
    LogEntry(2**20 + 40, feasible(0.15, 0.60, -6.5), "gt_level2", 0.0),  # (6, 1): GT only
)
"""Ground truth: 4 prefix cells, 5 map cells, 7 from (a)-(c); method runs add an 8th."""
