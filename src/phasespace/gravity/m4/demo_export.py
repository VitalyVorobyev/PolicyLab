"""Curated public demo of the closed M4 comparison: run `gravity-m4-demo-atlas-v1`.

    uv run python -m phasespace.gravity.m4 demo-export --confirm-frozen [--workers 10]

Reads stored evidence runs only (never modifies them) and writes one new run with

- `atlas.json`: the 10 x 10 descriptor grid and its reachable cells G, per arm the
  verbatim 16-seed summaries, anytime coverage and primary contrasts of the analysis run,
  and the seed-1001 catalog after the 5.6 re-check: every elite's decision vector,
  record quantities, stored survival s_i at the calibrated B and its nominal trajectory
  as a simplified polyline;
- `fan-<arm>.json`: per elite, the 50 H_E_nominal injection-error draws re-propagated
  without correction, as simplified polylines with their outcome, the recomputed
  uncorrected feasibility and the stored survival at B.

Provenance rules: the arm configurations, main seeds and B are read from the analysis
run; one seed is shown, the first main seed (1001), fixed before looking at any catalog.
G is the ground-truth run's cell set, accepted only if the analysis recorded the same
count for sources (a)-(c) and (a)-(d) (so (d) added no cell). Every source run must be
an input of the analysis run with the same manifest sha256. Re-propagated nominal
records must match the stored ones within the 5.6 re-check thresholds, and every
recomputed uncorrected feasibility must equal the stored robustness row; otherwise the
export fails and writes nothing.

Polylines are in the barycentric synodic frame in km (normalized coordinates times the
length scale), with times in days from TLI. They are simplified by a Douglas-Peucker
pass with a per-point tolerance min(abs_tol, rel_tol * distance to the nearer primary
centre) over the solver's accepted steps refined with dense output, so the departure,
the lunar flyby and the return keep detail at their own scale; the perilune is always
kept. If a path would keep more points than its cap, the tolerance is scaled by 1.5
until it fits (the largest scale used is recorded).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from phasespace.gravity.departure import parking_radius
from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.export import write_run_directory
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    Outcome,
    propagate_departure,
)
from phasespace.gravity.m4 import analysis
from phasespace.gravity.m4.catalog import Recheck, recheck
from phasespace.gravity.m4.config import (
    ERROR_DISTRIBUTIONS,
    METHODS,
    PROTOCOL,
    PROTOCOL_PATH,
    require_frozen,
)
from phasespace.gravity.m4.evaluation import Evaluation, evaluate_unit, to_physical
from phasespace.gravity.m4.robustness import ErrorDraw, error_draws, uncorrected_trajectory
from phasespace.gravity.m4.runs import build_manifest, load_table
from phasespace.gravity.propagation import PropagationResult, Termination
from phasespace.gravity.system import EarthMoonSystem
from phasespace.gravity.units import time_to_days

DEMO_RUN_ID: Final = "gravity-m4-demo-atlas-v1"
DEMO_EXPERIMENT_ID: Final = "cr3bp-free-return-m4-demo-atlas-v1"
DEMO_SEED: Final = PROTOCOL.seeds.main[0]
"""The first main seed: fixed in advance, so the shown catalogs are not selected."""
DEMO_ERRORS: Final = "H_E_nominal"
DEMO_GATES: Final = analysis.PRIMARY_GATES
ATLAS_FILE: Final = "atlas.json"
SUMMARY_METRICS: Final = ("p1", "p2", "coverage", "mean_s", "elites")
SUMMARY_STATISTICS: Final = ("mean", "median", "sd", "min", "max")


def fan_file(method: str) -> str:
    """Artifact path of one arm's error fan (a plain file name, as the contract requires)."""
    return f"fan-{method.lower()}.json"


# --- polyline simplification ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Simplification:
    """Douglas-Peucker settings: tolerance_i = min(abs_tol, rel_tol * d_i), d_i the distance
    of sample i to the nearer primary centre; output rounded to the given decimals."""

    rel_tolerance: float
    abs_tolerance_km: float
    max_points: int
    substeps: int
    """Dense-output samples per accepted solver step (the simplification's input)."""
    km_decimals: int
    day_decimals: int

    def json(self) -> dict[str, Any]:
        return {
            "method": "douglas_peucker_per_point_tolerance",
            "tolerance": "min(abs_tolerance_km, rel_tolerance * distance to the nearer "
            "primary centre), point-to-segment distance in the synodic frame",
            "input": "accepted solver steps, each refined into `substeps` dense-output samples",
            "rel_tolerance": self.rel_tolerance,
            "abs_tolerance_km": self.abs_tolerance_km,
            "max_points": self.max_points,
            "substeps": self.substeps,
            "km_decimals": self.km_decimals,
            "day_decimals": self.day_decimals,
            "cap_rule": "if more than max_points are kept, the tolerance is scaled by 1.5 "
            "until they fit",
        }


NOMINAL_SIMPLIFICATION: Final = Simplification(0.002, 100.0, 200, 16, 1, 5)
"""Sub-pixel in the viewer's Earth (~37 km/px) and Moon (~22 km/px) close-ups."""
FAN_SIMPLIFICATION: Final = Simplification(0.015, 1000.0, 120, 8, 0, 4)
"""Coarser: fans show where a draw goes at system scale (~500 km/px); size budget (8 MB)."""


def segment_distances(points: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Euclidean distance of each row of `points` to the closed segment a-b."""
    ab = b - a
    length2 = float(ab @ ab)
    rel = points - a
    if length2 == 0.0:
        return np.asarray(np.hypot(rel[:, 0], rel[:, 1]), dtype=np.float64)
    s = np.clip((rel @ ab) / length2, 0.0, 1.0)
    off = rel - s[:, np.newaxis] * ab
    return np.asarray(np.hypot(off[:, 0], off[:, 1]), dtype=np.float64)


def simplify(
    points: FloatArray, tolerance: FloatArray, keep: Iterable[int] = ()
) -> npt.NDArray[np.intp]:
    """Douglas-Peucker with a per-point tolerance: sorted indices of the kept points.

    The first, last and `keep` points are always kept. Every dropped point lies within its
    own tolerance of the kept segment that spans it (`deviations`).
    """
    n = points.shape[0]
    if n == 0:
        raise ValueError("no points")
    if tolerance.shape != (n,) or not np.all(tolerance > 0.0):
        raise ValueError("one positive tolerance per point is required")
    anchors = sorted({0, n - 1, *keep})
    if anchors[0] < 0 or anchors[-1] > n - 1:
        raise ValueError("keep indices out of range")
    kept = set(anchors)
    stack = list(pairwise(anchors))
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        ratio = segment_distances(points[i + 1 : j], points[i], points[j]) / tolerance[i + 1 : j]
        k = int(np.argmax(ratio))
        if ratio[k] > 1.0:
            m = i + 1 + k
            kept.add(m)
            stack += [(i, m), (m, j)]
    return np.array(sorted(kept), dtype=np.intp)


def deviations(points: FloatArray, kept: npt.NDArray[np.intp]) -> FloatArray:
    """Distance of every point to the kept segment that spans it (0 at kept points)."""
    out = np.zeros(points.shape[0])
    for i, j in pairwise(kept.tolist()):
        if j - i > 1:
            out[i + 1 : j] = segment_distances(points[i + 1 : j], points[i], points[j])
    return out


def dense_track(
    results: Sequence[PropagationResult], substeps: int, marks: Sequence[float] = ()
) -> tuple[FloatArray, FloatArray]:
    """(t, xy) in normalized units: each result's accepted steps refined into `substeps`
    dense-output samples, plus the `marks` times inside it; consecutive results are joined
    without repeating the shared time."""
    times: list[FloatArray] = []
    states: list[FloatArray] = []
    for res in results:
        t = res.t
        if res.dense is None or t.size < 2:
            ts, ys = t.copy(), res.states[:, :2].copy()
        else:
            frac = np.arange(substeps) / substeps
            ts = (t[:-1, np.newaxis] + np.diff(t)[:, np.newaxis] * frac).ravel()
            inside = [m for m in marks if float(t[0]) < m < float(t[-1])]
            ts = np.unique(np.concatenate([ts, [float(t[-1])], inside]))
            ys = res.sample(ts)[:, :2]
        if times and ts.size and ts[0] <= times[-1][-1]:
            ts, ys = ts[1:], ys[1:]
        times.append(ts)
        states.append(ys)
    return np.concatenate(times), np.vstack(states)


@dataclass(frozen=True, slots=True)
class Polyline:
    t_days: list[float]
    x_km: list[float] | list[int]
    y_km: list[float] | list[int]
    max_deviation_km: float
    """Largest distance of a dense sample to the simplified path, before rounding."""
    tolerance_scale: float

    def json(self) -> dict[str, Any]:
        return {"t_days": self.t_days, "x_km": self.x_km, "y_km": self.y_km}

    @property
    def points(self) -> int:
        return len(self.t_days)


def _rounded(values: FloatArray, decimals: int) -> list[float] | list[int]:
    if decimals == 0:
        return [round(float(v)) for v in values]
    return [round(float(v), decimals) for v in values]


def primary_distances_km(km: FloatArray, system: EarthMoonSystem) -> FloatArray:
    """Distance (km) of each synodic point (km) to the nearer primary centre."""
    scale, mu = system.length_scale_km, system.mu
    to_earth = np.hypot(km[:, 0] + mu * scale, km[:, 1])
    to_moon = np.hypot(km[:, 0] - (1.0 - mu) * scale, km[:, 1])
    return np.asarray(np.minimum(to_earth, to_moon), dtype=np.float64)


def polyline(
    results: Sequence[PropagationResult],
    system: EarthMoonSystem,
    spec: Simplification,
    marks: Sequence[float] = (),
) -> Polyline:
    """The simplified, rounded path of one (possibly joined) trajectory; `marks` (normalized
    times) are always kept."""
    t, xy = dense_track(results, spec.substeps, marks)
    km = xy * system.length_scale_km
    tolerance = np.minimum(
        spec.abs_tolerance_km, spec.rel_tolerance * primary_distances_km(km, system)
    )
    keep = [int(i) for i in np.flatnonzero(np.isin(t, np.asarray(marks, dtype=np.float64)))]
    if len(keep) + 2 > spec.max_points:
        raise ValueError("more marked points than the cap allows")
    scale = 1.0
    kept = simplify(km, tolerance, keep)
    while kept.size > spec.max_points:
        scale *= 1.5
        kept = simplify(km, tolerance * scale, keep)
    t_days = _rounded(np.asarray(time_to_days(t[kept], system)), spec.day_decimals)
    if any(b <= a for a, b in pairwise(t_days)):
        raise ValueError("rounded polyline times are not strictly increasing")
    return Polyline(
        t_days=[float(v) for v in t_days],
        x_km=_rounded(km[kept, 0], spec.km_decimals),
        y_km=_rounded(km[kept, 1], spec.km_decimals),
        max_deviation_km=float(np.max(deviations(km, kept))),
        tolerance_scale=scale,
    )


# --- per-elite computation (runs in worker processes) ------------------------------------------


@dataclass(frozen=True, slots=True)
class EliteTask:
    method: str
    cell: tuple[int, int]
    index: int
    stored: Evaluation
    draws: tuple[ErrorDraw, ...]


@dataclass(frozen=True, slots=True)
class DrawPath:
    outcome: str
    uncorrected_feasible: bool
    path: Polyline


@dataclass(frozen=True, slots=True)
class EliteResult:
    method: str
    cell: tuple[int, int]
    index: int
    recheck: Recheck
    path: Polyline
    perilune: dict[str, float]
    draws: tuple[DrawPath, ...]


def _closest_perilune(result: PropagationResult) -> float:
    """Normalized time of the closest Moon periapsis before the end (as `classify`)."""
    moon_x = 1.0 - result.system.mu
    lunar = [p for p in result.periapses if p.name == "moon_periapsis" and p.t < result.t_final]
    if not lunar:
        raise ValueError("a feasible nominal has a lunar periapsis")
    return min(lunar, key=lambda p: math.hypot(p.state[0] - moon_x, p.state[1])).t


def nominal_path(
    stored: Evaluation, definition: FreeReturnDefinition = DEFAULT_DEFINITION, index: int = -1
) -> tuple[Recheck, Polyline, dict[str, float]]:
    """Re-propagate a stored elite with the production evaluator settings, check it against
    the stored record (5.6 thresholds: same outcome and cell, |dh_p| <= 10 m, |dt_ret| <=
    0.1 s, |dgamma| <= 1e-3 deg) and return its simplified path and perilune point."""
    check = recheck(stored, definition, index=index, tight=lambda e: evaluate_unit(e.u, definition))
    if not check.passed:
        raise ValueError(f"elite {index}: re-propagation does not reproduce the record: {check}")
    result = propagate_departure(*to_physical(stored.u), definition)
    if result.termination is not Termination.EARTH_RETURN_SHELL:
        raise ValueError(f"elite {index}: the nominal path does not end on the return shell")
    system = definition.system
    if abs(time_to_days(result.t_final, system) - stored.return_time_days) * 86400.0 > (
        PROTOCOL.gate.recheck_return_time_s
    ):
        raise ValueError(f"elite {index}: path end differs from the stored return time")
    t_p = _closest_perilune(result)
    state = result.sample([t_p])[0]
    perilune = {
        "t_days": round(float(time_to_days(t_p, system)), NOMINAL_SIMPLIFICATION.day_decimals),
        "x_km": round(float(state[0]) * system.length_scale_km, NOMINAL_SIMPLIFICATION.km_decimals),
        "y_km": round(float(state[1]) * system.length_scale_km, NOMINAL_SIMPLIFICATION.km_decimals),
    }
    return check, polyline([result], system, NOMINAL_SIMPLIFICATION, [t_p]), perilune


def fan_path(
    stored: Evaluation, draw: ErrorDraw, definition: FreeReturnDefinition = DEFAULT_DEFINITION
) -> DrawPath:
    """One draw re-propagated without correction (as the robustness runs judge it)."""
    traj = uncorrected_trajectory(stored, draw, definition)
    outcome = traj.record.outcome if traj.gate_passed else Outcome.NUMERICAL_FAILURE
    segments = [traj.arc] if traj.tail is None else [traj.arc, traj.tail]
    return DrawPath(
        outcome=outcome.value,
        uncorrected_feasible=traj.feasible,
        path=polyline(segments, definition.system, FAN_SIMPLIFICATION),
    )


def run_elite(task: EliteTask) -> EliteResult:
    check, path, perilune = nominal_path(task.stored, index=task.index)
    return EliteResult(
        method=task.method,
        cell=task.cell,
        index=task.index,
        recheck=check,
        path=path,
        perilune=perilune,
        draws=tuple(fan_path(task.stored, d) for d in task.draws),
    )


# --- stored evidence ------------------------------------------------------------------------


def _rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    names = [str(c["name"]) for c in table["columns"]]
    return [dict(zip(names, row, strict=True)) for row in table["rows"]]


@dataclass(frozen=True, slots=True)
class Arm:
    method: str
    config_index: int
    source: analysis.MethodData
    robustness: analysis.RobustnessData


def _require_input(inputs: dict[str, str], run: analysis.StoredRun) -> None:
    if inputs.get(run.run_id) != run.manifest_sha256:
        raise analysis.EvidenceError(
            f"{run.run_id}: not an input of {analysis.ANALYSIS_RUN_ID} with this manifest sha256"
        )


def survived_flags(el: analysis.EliteDraws, budget_m_s: float) -> list[bool]:
    """Per draw, success at B under the primary Gates set (D25), from the stored rows with
    the comparison `EliteDraws.successes` makes (B in km/s)."""
    limit = budget_m_s / 1000.0
    flags = [
        u or (c <= limit and x is True)
        for u, c, x in zip(el.uncorrected, el.commanded_km_s, el.executed[DEMO_GATES], strict=True)
    ]
    if sum(flags) != el.successes(budget_m_s, DEMO_GATES):
        raise analysis.EvidenceError(f"elite {el.index}: per-draw survival != stored successes")
    return flags


def _summary(results: list[dict[str, Any]], method: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        rows = {
            r["statistic"]: r["value"]
            for r in results
            if r["section"] == "summary" and r["metric"] == metric and r["arm"] == method
        }
        out[metric] = {s: rows.get(s) for s in SUMMARY_STATISTICS}
    return out


def _anytime(results: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    rows = [
        r
        for r in results
        if r["section"] == "anytime"
        and r["metric"] == "coverage"
        and r["arm"] == method
        and r["statistic"] == "mean"
    ]
    return [
        {
            "checkpoint": r["checkpoint"],
            "mean": r["value"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
        }
        for r in sorted(rows, key=lambda r: int(r["checkpoint"]))
    ]


def _contrasts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "endpoint": r["metric"],
            "arm": r["arm"],
            "baseline": r["baseline"],
            "mean_difference": r["value"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p_value": r["p_value"],
            "p_holm": r["p_holm"],
            "significant": r["significant"],
        }
        for r in results
        if r["section"] == "primary"
    ]


def _grid(cells: set[tuple[int, int]]) -> dict[str, Any]:
    desc = PROTOCOL.descriptors
    (lo1, hi1), (lo2, hi2) = desc.d1_bounds, desc.d2_bounds
    n = desc.bins
    e1 = [lo1 + k * (hi1 - lo1) / n for k in range(n + 1)]
    e2 = [lo2 + k * (hi2 - lo2) / n for k in range(n + 1)]
    return {
        "d1": {
            "name": "log10_perilune_altitude_km",
            "definition": "d1 = log10(h_p / km), h_p the closest perilune altitude",
            "min": lo1,
            "max": hi1,
            "bins": n,
            "edges": e1,
            "edges_km": [10.0**v for v in e1],
        },
        "d2": {
            "name": "return_time_days",
            "definition": "d2 = t_ret, the return-shell crossing time after TLI (days)",
            "min": lo2,
            "max": hi2,
            "bins": n,
            "edges": e2,
            "edges_days": e2,
        },
        "binning": "k = floor(bins (d - min) / (max - min)) clipped to [0, bins - 1] (protocol 5.5)",
        "reachable_cells": [list(c) for c in sorted(cells)],
        "reachable_definition": "G (protocol 5.3): cells with a feasible propagation in the "
        "ground truth, sources (a)-(d); (d) added none at 10 x 10",
    }


# --- the export --------------------------------------------------------------------------------


def _dump(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def _check(name: str, observed: float, threshold: float, definition: str) -> dict[str, Any]:
    ok = math.isfinite(observed)
    return {
        "name": name,
        "observed": observed if ok else None,
        "threshold": threshold,
        "passed": ok and observed <= threshold,
        "definition": definition,
    }


def run_demo_export(
    runs_dir: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    confirm_frozen: bool,
    run_id: str = DEMO_RUN_ID,
    workers: int = 1,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Build and write the demo run (see the module docstring); returns its manifest."""
    require_frozen(confirm_frozen, protocol_path)
    if (runs_dir / run_id).exists():
        raise FileExistsError(f"{runs_dir / run_id} already exists; runs are never overwritten")
    definition = DEFAULT_DEFINITION
    system = definition.system
    dist = ERROR_DISTRIBUTIONS[DEMO_ERRORS]

    # The analysis run fixes the arms, the main seeds and B.
    an = analysis.open_run(runs_dir, analysis.ANALYSIS_RUN_ID)
    analysis.require_evidence([an])
    block: dict[str, Any] = an.m4["analysis"]
    configs = {str(k): int(v) for k, v in block["configs"].items()}
    if list(configs) != list(METHODS):
        raise analysis.EvidenceError("the analysis configs do not list the M4 arms in order")
    if int(block["main_seeds"][0]) != DEMO_SEED:
        raise analysis.EvidenceError(f"the first main seed is not {DEMO_SEED}")
    budget = float(block["budget_m_s"])
    inputs = {str(r["run_id"]): str(r["manifest_sha256"]) for r in an.m4["inputs"]}
    results = _rows(load_table(an.directory, "results.json"))
    per_run = _rows(load_table(an.directory, "per_run.json"))
    anytime = _rows(load_table(an.directory, "anytime.json"))

    # G: the ground-truth cells, accepted only if (d) added none in the analysis.
    gt = analysis.load_groundtruth(runs_dir, analysis.GROUNDTRUTH_RUN_ID)
    _require_input(inputs, gt.run)
    cells = gt.cells(PROTOCOL.descriptors.bins)
    counts = {
        r["statistic"]: r["value"]
        for r in results
        if r["section"] == "ground_truth" and r["metric"] == "cells_10"
    }
    if not (
        counts.get("a_c") == counts.get("a_d") == len(cells) == an.metric("ground_truth_cells")
    ):
        raise analysis.EvidenceError(f"G: {len(cells)} ground-truth cells vs analysis {counts}")

    arms: list[Arm] = []
    for method in METHODS:
        src = analysis.load_stage_run(runs_dir, "main", method, configs[method], DEMO_SEED)
        rob = analysis.load_robustness(
            runs_dir, src, src.run.run_id, DEMO_ERRORS, confirm_frozen=confirm_frozen
        )
        analysis.require_evidence([src.run, rob.run])
        _require_input(inputs, src.run)
        _require_input(inputs, rob.run)
        if not {el.cell for el in rob.elites} <= cells:
            raise analysis.EvidenceError(f"{src.run.run_id}: an elite lies outside G")
        arms.append(Arm(method, configs[method], src, rob))

    draws = tuple(error_draws(dist, confirm_frozen=confirm_frozen))
    tasks = [
        EliteTask(arm.method, el.cell, el.index, arm.source.log[el.index], draws)
        for arm in arms
        for el in arm.robustness.elites
    ]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            computed = list(pool.map(run_elite, tasks))
    else:
        computed = [run_elite(t) for t in tasks]
    by_key = {(r.method, r.index): r for r in computed}

    mismatches: list[str] = []
    atlas_arms: list[dict[str, Any]] = []
    fan_files: list[tuple[str, str, bytes]] = []
    elite_counts: dict[str, int] = {}
    for arm in arms:
        method = arm.method
        seed_row = next(r for r in per_run if r["method"] == method and r["seed"] == DEMO_SEED)
        elites_json: list[dict[str, Any]] = []
        fans_json: list[dict[str, Any]] = []
        survival_sum = 0.0
        for el in arm.robustness.elites:
            res = by_key[(method, el.index)]
            stored = arm.source.log[el.index]
            survived = survived_flags(el, budget)
            for k, (d, u) in enumerate(zip(res.draws, el.uncorrected, strict=True)):
                if d.uncorrected_feasible != u:
                    mismatches.append(
                        f"{method} elite {el.index} draw {k}: {d.uncorrected_feasible} != {u}"
                    )
            s_i = el.survival(budget, DEMO_GATES)
            survival_sum += s_i
            elites_json.append(
                {
                    "cell": list(el.cell),
                    "index": el.index,
                    "theta_deg": stored.phase_deg,
                    "delta_v_km_s": stored.delta_v_km_s,
                    "alpha_deg": stored.impulse_angle_deg,
                    "perilune_altitude_km": stored.perilune_altitude_km,
                    "lunar_encounter_time_days": stored.lunar_encounter_time_days,
                    "return_time_days": stored.return_time_days,
                    "entry_flight_path_angle_deg": stored.return_flight_path_angle_deg,
                    "objective_deg": stored.objective_deg,
                    "survival": s_i,
                    "uncorrected_feasible_draws": sum(el.uncorrected),
                    "perilune": res.perilune,
                    "path": res.path.json(),
                }
            )
            fans_json.append(
                {
                    "cell": list(el.cell),
                    "index": el.index,
                    "draws": [
                        {
                            "outcome": d.outcome,
                            "uncorrected_feasible": d.uncorrected_feasible,
                            "survived": s,
                            **d.path.json(),
                        }
                        for d, s in zip(res.draws, survived, strict=True)
                    ],
                }
            )
        if len(elites_json) != int(seed_row["elites"]) or (
            seed_row["p1"] is not None and abs(survival_sum - float(seed_row["p1"])) > 1e-9
        ):
            raise analysis.EvidenceError(f"{method}: catalog does not reproduce per_run P1/elites")
        elite_counts[method] = len(elites_json)
        seed_anytime = sorted(
            (r for r in anytime if r["method"] == method and r["seed"] == DEMO_SEED),
            key=lambda r: int(r["checkpoint"]),
        )
        fan_path_name = fan_file(method)
        atlas_arms.append(
            {
                "arm": method,
                "config_index": arm.config_index,
                "source_run_id": arm.source.run.run_id,
                "robustness_run_id": arm.robustness.run.run_id,
                "fan_path": fan_path_name,
                "summary": _summary(results, method),
                "anytime_coverage": _anytime(results, method),
                "seed_values": {m: seed_row[m] for m in SUMMARY_METRICS},
                "seed_anytime_coverage": [
                    {"checkpoint": r["checkpoint"], "coverage": r["coverage"]} for r in seed_anytime
                ],
                "elites": elites_json,
            }
        )
        fan_files.append(
            (
                "error_fan",
                fan_path_name,
                _dump(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "arm": method,
                        "frame": "barycentric_synodic",
                        "length_unit": "km",
                        "time_unit": "day",
                        "errors": dist.name,
                        "error_seed": dist.seed,
                        "budget_m_s": budget,
                        "gates": DEMO_GATES,
                        "error_draws": [
                            {
                                "magnitude": d.magnitude,
                                "direction_deg": d.direction_deg,
                                "timing_s": d.timing_s,
                            }
                            for d in draws
                        ],
                        "elites": fans_json,
                    }
                ),
            )
        )
    if mismatches:
        raise analysis.EvidenceError(
            f"{len(mismatches)} recomputed uncorrected feasibilities differ from the stored "
            "robustness rows:\n  " + "\n  ".join(mismatches[:20])
        )

    scale = system.length_scale_km
    earth_x_km = -system.mu * scale
    parking_km = parking_radius(system, definition.parking_altitude_km) * scale
    atlas: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "frame": "barycentric_synodic",
        "length_unit": "km",
        "time_unit": "day",
        "frame_definition": "origin at the Earth-Moon barycentre, +x toward the Moon, rotating "
        "counterclockwise; km = normalized coordinate x length_scale_km; t = days after TLI",
        "mu": system.mu,
        "length_scale_km": scale,
        "time_scale_seconds": system.time_scale_s,
        "earth": {"x_km": earth_x_km, "y_km": 0.0, "radius_km": system.earth_radius_km},
        "moon": {
            "x_km": (1.0 - system.mu) * scale,
            "y_km": 0.0,
            "radius_km": system.moon_radius_km,
        },
        "parking_orbit_radius_km": parking_km,
        "return_shell_radius_km": definition.return_shell_km,
        "perilune_band_km": list(definition.perilune_band_km),
        "gamma_target_deg": PROTOCOL.objective.gamma_target_deg,
        "seed": DEMO_SEED,
        "seed_choice": "the first main seed, fixed before any catalog was inspected (no "
        "seed selection); the 16-seed summaries are the evidence, one seed is an example",
        "main_seeds": [int(s) for s in block["main_seeds"]],
        "analysis_run_id": an.run_id,
        "budget_m_s": budget,
        "errors": dist.name,
        "draws": dist.draws,
        "gates": DEMO_GATES,
        "survival_definition": "s_i = fraction of the draws feasible without correction, or "
        "with a commanded MCC |dv| <= B that is feasible after its execution under the "
        "primary Gates set (D25); stored robustness rows, not recomputed",
        "grid": _grid(cells),
        "h1": block["h1"],
        "contrasts": _contrasts(results),
        "simplification": NOMINAL_SIMPLIFICATION.json(),
        "fan_simplification": FAN_SIMPLIFICATION.json(),
        "arms": atlas_arms,
    }

    nominal_paths = [r.path for r in computed]
    fan_paths = [d.path for r in computed for d in r.draws]
    rechecks = [r.recheck for r in computed]

    def worst(values: Iterable[float]) -> float:
        return max(values, default=0.0)

    shell_error = worst(
        abs(math.hypot(p.x_km[-1] - earth_x_km, p.y_km[-1]) - definition.return_shell_km)
        for p in nominal_paths
    )
    start_error = worst(
        abs(math.hypot(p.x_km[0] - earth_x_km, p.y_km[0]) - parking_km)
        for p in [*nominal_paths, *fan_paths]
    )
    gate = PROTOCOL.gate
    checks = [
        _check(
            "nominal_recheck_failures",
            float(sum(not c.passed for c in rechecks)),
            0.0,
            "elites whose re-propagation (production DOP853 1e-12 settings) differs from the "
            "stored record in outcome or cell or beyond the 5.6 thresholds",
        ),
        _check(
            "nominal_max_perilune_diff_km",
            worst(c.perilune_diff_km for c in rechecks),
            gate.recheck_perilune_km,
            "max |h_p re-propagated - h_p stored| over the elites (km)",
        ),
        _check(
            "nominal_max_return_time_diff_s",
            worst(c.return_time_diff_s for c in rechecks),
            gate.recheck_return_time_s,
            "max |t_ret re-propagated - t_ret stored| over the elites (s)",
        ),
        _check(
            "nominal_max_gamma_diff_deg",
            worst(c.gamma_diff_deg for c in rechecks),
            gate.recheck_gamma_deg,
            "max |gamma_S re-propagated - gamma_S stored| over the elites (deg)",
        ),
        _check(
            "fan_uncorrected_feasibility_mismatches",
            float(len(mismatches)),
            0.0,
            "draws whose recomputed feasibility without correction differs from the stored "
            "robustness row (uncorrected_feasible)",
        ),
        _check(
            "nominal_end_return_shell_error_km",
            shell_error,
            0.1,
            "max | |last path point - Earth centre| - R_S | over the nominal paths, after "
            "rounding (km)",
        ),
        _check(
            "path_start_parking_radius_error_km",
            start_error,
            1.0,
            "max | |first path point - Earth centre| - parking radius | over all paths, after "
            "rounding (km)",
        ),
    ]
    failed = [c["name"] for c in checks if not c["passed"]]
    if failed:
        raise analysis.EvidenceError(f"demo export checks failed: {failed}")

    metrics = {
        "seed": _metric(DEMO_SEED, "count", "main seed of the shown catalogs"),
        **{
            f"elites_{m.lower()}": _metric(n, "count", f"{m} elites at seed {DEMO_SEED} after 5.6")
            for m, n in elite_counts.items()
        },
        "fan_draws": _metric(len(fan_paths), "count", "uncorrected draws re-propagated"),
        "nominal_points_max": _metric(
            max(p.points for p in nominal_paths), "count", "largest nominal polyline"
        ),
        "fan_points_max": _metric(
            max(p.points for p in fan_paths), "count", "largest fan polyline"
        ),
        "fan_points_mean": _metric(
            sum(p.points for p in fan_paths) / len(fan_paths), "count", "mean fan polyline size"
        ),
        "nominal_max_deviation_km": _metric(
            worst(p.max_deviation_km for p in nominal_paths),
            "km",
            "largest distance of a dense sample to its simplified nominal path, before rounding",
        ),
        "fan_max_deviation_km": _metric(
            worst(p.max_deviation_km for p in fan_paths),
            "km",
            "largest distance of a dense sample to its simplified fan path, before rounding",
        ),
        "nominal_tolerance_scale_max": _metric(
            worst(p.tolerance_scale for p in nominal_paths),
            "dimensionless",
            "largest tolerance scale the point cap forced on a nominal path (1 = none)",
        ),
        "fan_tolerance_scale_max": _metric(
            worst(p.tolerance_scale for p in fan_paths),
            "dimensionless",
            "largest tolerance scale the point cap forced on a fan path (1 = none)",
        ),
    }
    sources = [an, gt.run, *(r for a in arms for r in (a.source.run, a.robustness.run))]
    created = datetime.now(UTC)
    manifest = build_manifest(
        run_id,
        stage="demo_export",
        experiment_id=DEMO_EXPERIMENT_ID,
        seed=dist.seed,
        methods=[],
        metrics=metrics,
        limitations=[
            f"A curated public sample: the catalogs of one main seed ({DEMO_SEED}, the first, "
            "fixed in advance) per arm. Comparative claims rest on the 16-seed summaries and "
            f"contrasts copied verbatim from {an.run_id}, not on this seed.",
            f"random_seed is the {dist.name} error-draw seed ({dist.seed}); the draws are "
            "regenerated by the frozen generator in the robustness runs' order.",
            "Paths are simplified polylines for display (see m4.demo.simplification); "
            "quantities in the records come from the stored runs, not from the polylines.",
            "Status computed_unvalidated: the recorded checks verify reproduction of the "
            "stored (computed_unvalidated) runs, not an independent validation.",
        ],
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=created,
        definition=definition,
        protocol_path=protocol_path,
        extra_m4={
            "inputs": [r.input_record() for r in sources],
            "demo": {
                "seed": DEMO_SEED,
                "seed_choice": atlas["seed_choice"],
                "analysis_run_id": an.run_id,
                "groundtruth_run_id": gt.run.run_id,
                "configs": configs,
                "errors": dist.name,
                "gates": DEMO_GATES,
                "budget_m_s": budget,
                "simplification": {
                    "nominal": NOMINAL_SIMPLIFICATION.json(),
                    "fan": FAN_SIMPLIFICATION.json(),
                },
            },
        },
    )
    manifest["validation"] = {
        "checks": checks,
        "validated_by": f"{__name__} (automatic consistency checks with the stored runs)",
        "validated_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    files = [("atlas", ATLAS_FILE, _dump(atlas)), *fan_files]
    return write_run_directory(runs_dir / run_id, manifest, files)
