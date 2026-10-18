"""Post-freeze analysis of the M4 comparison: protocol freeze-checklist steps 4, 5, 6 and 8.

Each step reads stored runs only and writes one new run directory (never overwriting):

- `run_calibration` (step 4, section 9.1): the correction budget B from the T_E
  robustness runs of the two D tuning catalogs (`robustness.calibrate_budget`).
- `run_selection` (step 5, section 6): per method, the configuration with the lowest sum
  of its ranks in P1 (at B, T_E, primary Gates set) and P2 (satisfiable tuning sets);
  ties by coverage against G_tune, then by the lower index. It also fixes the list of
  satisfiable tuning constraint sets (9.2: GT sources (a)-(c) and every tuning run).
- `run_pilot_analysis` (step 6, sections 8, 7.4, 12 K3/K4): P1 and P2 of the selected
  configurations on the 2 tuning + 3 pilot seeds, the seed count n, the K3 check and the
  projection (b) against the 7.8 M PE ceiling with the pre-declared reductions.
- `run_analysis` (step 8, sections 5.3 d, 5.6, 10-12): the ground-truth union and its
  saturation (K7), held-out satisfiability, every section-10 metric per main run and for
  CC*, the section-11 statistics (paired sign-flip tests with Holm, bootstrap CIs, H2),
  the K criteria and the seeded 5 % LSODA re-check sample. The LSODA sample is the only
  computation here that propagates.

Every input run must be evidence (protocol freeze checklist item 4): the frozen config
hash, `protocol_status = "frozen"` and `git_dirty = false` (`evidence_problems`). Each
output manifest records its inputs (run id, manifest sha256, commit) in `m4.inputs`.
Where the protocol text leaves a detail open, the docstrings below say which literal
reading is implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy.special import ndtr

from phasespace.gravity.export import validate_run_id, write_run_directory
from phasespace.gravity.free_return import Outcome
from phasespace.gravity.free_return_map import Column, dumps_table, sample_table
from phasespace.gravity.m4.catalog import Elite, build_catalog, cell_members, lsoda_sample
from phasespace.gravity.m4.config import (
    ERROR_DISTRIBUTIONS,
    METHODS,
    PROTOCOL,
    PROTOCOL_PATH,
    ErrorDistribution,
    checkpoints_for,
    config_sha256,
    method_grid,
    require_frozen,
    to_jsonable,
)
from phasespace.gravity.m4.evaluation import Cell, Evaluation, LogEntry, cell_of
from phasespace.gravity.m4.groundtruth import saturated
from phasespace.gravity.m4.metrics import diversity_classes, quality
from phasespace.gravity.m4.robustness import (
    ConstraintSet,
    availability,
    calibrate_budget,
    held_out_sets,
    tuning_sets,
)
from phasespace.gravity.m4.runs import (
    build_manifest,
    load_log,
    load_table,
    method_run_id,
    robustness_columns,
    robustness_run_id,
)
from phasespace.gravity.m4.stats import (
    bootstrap_ci,
    holm,
    paired_differences,
    seed_count,
    sign_flip_pvalue,
    stratified_permutation_pvalue,
    summary,
)

GROUNDTRUTH_RUN_ID: Final = "gravity-m4-groundtruth-v1"
CCSTAR_RUN_ID: Final = "gravity-m4-ccstar-v1"
CALIBRATION_RUN_ID: Final = "gravity-m4-calibration-v1"
SELECTION_RUN_ID: Final = "gravity-m4-selection-v1"
PILOT_ANALYSIS_RUN_ID: Final = "gravity-m4-pilot-analysis-v1"
ANALYSIS_RUN_ID: Final = "gravity-m4-analysis-v1"

BASELINES: Final = ("D", "MS", "MBH", "DE", "CC")
"""The five baselines of H1 (section 2); T - X for each forms the Holm family (11)."""
PRIMARY_GATES: Final = "gates_primary"
STRESS_GATES: Final = "gates_stress"
C1_PROPAGATION_NFEV: Final = 4106
"""Section 7.2: one propagation equivalent (PE) is the RHS evaluations of c1's propagation."""
PE_CEILING: Final = 7_800_000
"""Section 7.4: the compute ceiling in propagation equivalents."""

_STATS: Final = PROTOCOL.statistics
_GRID: Final = PROTOCOL.correction.budget_grid_m_s


class EvidenceError(ValueError):
    """An input run is not evidence (freeze checklist item 4) or is inconsistent."""


# --- stored runs -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredRun:
    run_id: str
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    @property
    def m4(self) -> dict[str, Any]:
        block: dict[str, Any] = self.manifest["m4"]
        return block

    def metric(self, name: str) -> float:
        return float(self.manifest["metrics"][name]["value"])

    def input_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "git_commit": self.manifest["source"]["git_commit"],
        }


def open_run(runs_dir: Path, run_id: str) -> StoredRun:
    """A stored run's manifest (FileNotFoundError if the run does not exist)."""
    directory = runs_dir / run_id
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{directory}: no such run (missing manifest.json)")
    data = path.read_bytes()
    manifest: dict[str, Any] = json.loads(data)
    if manifest.get("run_id") != run_id:
        raise EvidenceError(f"{path}: run_id {manifest.get('run_id')!r} != {run_id!r}")
    return StoredRun(run_id, directory, manifest, hashlib.sha256(data).hexdigest())


def evidence_problems(run: StoredRun) -> list[str]:
    """Why a run is not evidence: config hash, protocol version and status, git_dirty."""
    m4 = run.manifest.get("m4")
    if not isinstance(m4, dict):
        return [f"{run.run_id}: no m4 block"]
    block: dict[str, Any] = m4  # pyright: ignore[reportUnknownVariableType]
    out: list[str] = []
    if block.get("config_sha256") != config_sha256():
        out.append(f"{run.run_id}: config_sha256 {block.get('config_sha256')} != the frozen hash")
    if block.get("protocol_version") != PROTOCOL.version:
        out.append(f"{run.run_id}: protocol_version {block.get('protocol_version')!r}")
    if block.get("protocol_status") != "frozen":
        out.append(f"{run.run_id}: protocol_status {block.get('protocol_status')!r}")
    if run.manifest.get("source", {}).get("git_dirty") is not False:
        out.append(f"{run.run_id}: git_dirty is not false")
    return out


def require_evidence(runs: Iterable[StoredRun]) -> None:
    problems = [p for r in runs for p in evidence_problems(r)]
    if problems:
        raise EvidenceError(
            "inputs are not evidence (frozen config hash, protocol_status 'frozen' and "
            "git_dirty false are required):\n  " + "\n  ".join(problems)
        )


def _require_stage(run: StoredRun, stage: str) -> None:
    if run.m4.get("stage") != stage:
        raise EvidenceError(f"{run.run_id}: stage {run.m4.get('stage')!r}, expected {stage!r}")


def _refuse_existing(runs_dir: Path, run_id: str) -> None:
    validate_run_id(run_id)
    if (runs_dir / run_id).exists():
        raise FileExistsError(f"{runs_dir / run_id} already exists; runs are never overwritten")


@dataclass(frozen=True, slots=True)
class MethodData:
    """One stored method run with its full propagation log."""

    run: StoredRun
    method: str
    config_index: int
    seed: int
    budget: int
    log: tuple[Evaluation, ...]
    wall_time_hours: float


def load_method_run(
    runs_dir: Path, run_id: str, *, stage: str, method: str, config_index: int, seed: int
) -> MethodData:
    """A method run, checked against its expected stage, method, configuration and seed."""
    run = open_run(runs_dir, run_id)
    _require_stage(run, stage)
    blocks: list[dict[str, Any]] = run.m4["methods"]
    if len(blocks) != 1:
        raise EvidenceError(f"{run_id}: expected one method block")
    block = blocks[0]
    cfg = method_grid("CC" if method == "CCstar" else method)[config_index]
    expected_config: Any = {} if cfg is None else to_jsonable(cfg)
    if block["name"] != method or int(block["seed"]) != seed or block["config"] != expected_config:
        raise EvidenceError(f"{run_id}: method block does not match {method} c{config_index}")
    log = tuple(e.evaluation for e in load_log(run.directory, str(block["evaluations_path"])))
    if len(log) != int(block["propagations"]):
        raise EvidenceError(f"{run_id}: log length != recorded propagations")
    return MethodData(
        run=run,
        method=method,
        config_index=config_index,
        seed=seed,
        budget=int(block["budget"]),
        log=log,
        wall_time_hours=run.metric("wall_time_hours"),
    )


def load_stage_run(runs_dir: Path, stage: str, method: str, k: int, seed: int) -> MethodData:
    return load_method_run(
        runs_dir,
        method_run_id(stage, method, k, seed),
        stage=stage,
        method=method,
        config_index=k,
        seed=seed,
    )


# --- robustness runs -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EliteDraws:
    """The stored per-draw results of one re-checked catalog elite (9.1)."""

    cell: Cell
    index: int
    singular: bool
    uncorrected: tuple[bool, ...]
    commanded_km_s: tuple[float, ...]
    executed: dict[str, tuple[bool | None, ...]]
    converged: tuple[bool, ...]
    diverged: tuple[bool, ...]
    nfev: tuple[int, ...]

    @property
    def draws(self) -> int:
        return len(self.uncorrected)

    def successes(self, budget_m_s: float, gates: str = PRIMARY_GATES) -> int:
        """Draws that succeed at B (D25): uncorrected feasible, or commanded |dv| <= B and
        feasible after the executed correction. B is compared in the stored unit (km/s,
        B / 1000), the same rounding the stored |dv| went through."""
        if not 0.0 <= budget_m_s <= PROTOCOL.correction.divergence_m_s:
            raise ValueError("survival is defined for 0 <= B <= 100 m/s")
        limit = budget_m_s / 1000.0
        return sum(
            u or (c <= limit and x is True)
            for u, c, x in zip(
                self.uncorrected, self.commanded_km_s, self.executed[gates], strict=True
            )
        )

    def survival(self, budget_m_s: float, gates: str = PRIMARY_GATES) -> float:
        """s_i(B): the fraction of the draws that succeed at B."""
        return self.successes(budget_m_s, gates) / self.draws


@dataclass(frozen=True, slots=True)
class RobustnessData:
    run: StoredRun
    errors: ErrorDistribution
    elites: tuple[EliteDraws, ...]
    """Sorted by cell (the catalog after the 5.6 re-check)."""
    rechecks: int
    recheck_failures: int

    def successes(self, budget_m_s: float, gates: str = PRIMARY_GATES) -> int:
        return sum(el.successes(budget_m_s, gates) for el in self.elites)

    def p1(self, budget_m_s: float, gates: str = PRIMARY_GATES) -> Fraction:
        """P1 = sum_i s_i, exact (every elite has the same K draws)."""
        return Fraction(self.successes(budget_m_s, gates), self.errors.draws)


def _flag(value: Any) -> bool | None:
    return None if value is None else bool(int(value))


def load_robustness(
    runs_dir: Path,
    source: MethodData | None,
    source_run_id: str,
    errors: str,
    *,
    confirm_frozen: bool = False,
    protocol_path: Path = PROTOCOL_PATH,
) -> RobustnessData:
    """The robustness run of `source_run_id` under `errors`. With `source`, every elite is
    checked to be a feasible propagation of the source run in the recorded cell, ranked
    within the first 1 + 3 members of its cell (5.6). Held-out sets need the freeze."""
    dist = ERROR_DISTRIBUTIONS[errors]
    if dist.held_out:
        require_frozen(confirm_frozen, protocol_path)
    run = open_run(runs_dir, robustness_run_id(source_run_id, errors))
    _require_stage(run, "robustness")
    if run.manifest.get("random_seed") != dist.seed:
        raise EvidenceError(f"{run.run_id}: random_seed is not the {errors} seed {dist.seed}")
    table = load_table(run.directory, "robustness.json")
    names = [str(c["name"]) for c in table["columns"]]
    if names != [c.name for c in robustness_columns(dist.executions)]:
        raise EvidenceError(f"{run.run_id}: unexpected robustness columns")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in table["rows"]:
        record = dict(zip(names, row, strict=True))
        grouped.setdefault(int(record["index"]), []).append(record)
    members = cell_members(source.log) if source is not None else {}
    elites: list[EliteDraws] = []
    for index, rows in grouped.items():
        rows.sort(key=lambda r: int(r["draw"]))
        if [int(r["draw"]) for r in rows] != list(range(dist.draws)):
            raise EvidenceError(f"{run.run_id}: elite {index} lacks draws 0..{dist.draws - 1}")
        cells = {(int(r["cell_d1"]), int(r["cell_d2"])) for r in rows}
        singular = {int(r["singular"]) for r in rows}
        if len(cells) != 1 or len(singular) != 1:
            raise EvidenceError(f"{run.run_id}: elite {index} has inconsistent rows")
        cell = cells.pop()
        if source is not None:
            if not (0 <= index < len(source.log)) or cell_of(source.log[index]) != cell:
                raise EvidenceError(
                    f"{run.run_id}: elite {index} is not a feasible propagation of "
                    f"{source_run_id} in cell {cell}"
                )
            ranked = members[cell][: 1 + PROTOCOL.gate.recheck_max_replacements]
            if index not in ranked:
                raise EvidenceError(f"{run.run_id}: elite {index} is not a 5.6 candidate")
        elites.append(
            EliteDraws(
                cell=cell,
                index=index,
                singular=bool(singular.pop()),
                uncorrected=tuple(bool(int(r["uncorrected_feasible"])) for r in rows),
                commanded_km_s=tuple(float(r["commanded_delta_v_km_s"]) for r in rows),
                executed={
                    g: tuple(_flag(r[f"executed_feasible_{g}"]) for r in rows)
                    for g in dist.executions
                },
                converged=tuple(bool(int(r["converged"])) for r in rows),
                diverged=tuple(bool(int(r["diverged"])) for r in rows),
                nfev=tuple(int(r["nfev"]) for r in rows),
            )
        )
    elites.sort(key=lambda e: e.cell)
    if len({e.cell for e in elites}) != len(elites):
        raise EvidenceError(f"{run.run_id}: more than one elite in a cell")
    if len(elites) != int(run.metric("elites")):
        raise EvidenceError(f"{run.run_id}: elite count != the recorded 'elites' metric")
    return RobustnessData(
        run=run,
        errors=dist,
        elites=tuple(elites),
        rechecks=int(run.metric("rechecks")),
        recheck_failures=int(run.metric("recheck_failures")),
    )


def rechecked_catalog(source: MethodData, rob: RobustnessData) -> dict[Cell, Elite]:
    """The catalog after 5.6 as recorded by a robustness run, with the H2 margin: the e of
    the next evaluation after the elite in its cell's (e, index) order minus the elite's
    (as `catalog.rechecked_catalog` computes it)."""
    members = cell_members(source.log)
    out: dict[Cell, Elite] = {}
    for el in rob.elites:
        indices = members[el.cell]
        rest = indices[indices.index(el.index) + 1 :]
        e = source.log[el.index].objective_deg
        margin = source.log[rest[0]].objective_deg - e if rest else math.nan
        out[el.cell] = Elite(el.cell, el.index, e, margin)
    return out


# --- ground truth, satisfiability ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroundTruthData:
    run: StoredRun
    entries: tuple[LogEntry, ...]
    """The feasible propagations of sources (a)-(c), with their GT log index and tag."""

    def cells(self, bins: int, source: str = "all") -> set[Cell]:
        """Cells of 'prefix' (the 2^19 map prefix), 'map' (a) or 'all' ((a)-(c))."""
        prefix = 2**PROTOCOL.ground_truth.saturation_prefix_log2
        out: set[Cell] = set()
        for entry in self.entries:
            is_map = entry.tag == "gt_map"
            if source == "map" and not is_map:
                continue
            if source == "prefix" and not (is_map and entry.index < prefix):
                continue
            cell = cell_of(entry.evaluation, bins)
            if cell is not None:
                out.add(cell)
        return out


def load_groundtruth(runs_dir: Path, run_id: str) -> GroundTruthData:
    run = open_run(runs_dir, run_id)
    _require_stage(run, "ground_truth")
    entries = tuple(load_log(run.directory, "evaluations.json"))
    if not all(e.evaluation.feasible for e in entries):
        raise EvidenceError(f"{run_id}: the ground-truth table holds an infeasible row")
    return GroundTruthData(run, entries)


@dataclass(slots=True)
class FeasiblePool:
    """(h_p, t_ret, gamma_S) of feasible propagations, for satisfiability (9.2)."""

    perilune_km: list[float] = field(default_factory=list[float])
    return_days: list[float] = field(default_factory=list[float])
    gamma_deg: list[float] = field(default_factory=list[float])

    def add(self, evaluations: Iterable[Evaluation]) -> None:
        for e in evaluations:
            if e.feasible:
                self.perilune_km.append(e.perilune_altitude_km)
                self.return_days.append(e.return_time_days)
                self.gamma_deg.append(e.return_flight_path_angle_deg)

    def satisfiable(self, sets: Sequence[ConstraintSet]) -> list[bool]:
        """Per set: does some pooled propagation satisfy it (`ConstraintSet.satisfied_by`)?"""
        h = np.asarray(self.perilune_km, dtype=np.float64)
        t = np.asarray(self.return_days, dtype=np.float64)
        g = np.asarray(self.gamma_deg, dtype=np.float64)
        out: list[bool] = []
        for c in sets:
            mask = (h >= c.h_lo_km) & (h <= c.h_hi_km) & (t <= c.t_max_days)
            if c.gamma_c_deg is not None and c.width_deg is not None:
                mask &= np.abs(g - c.gamma_c_deg) <= c.width_deg
            out.append(bool(mask.any()))
        return out


def _cells(evaluations: Iterable[Evaluation], bins: int) -> set[Cell]:
    return {c for c in (cell_of(e, bins) for e in evaluations) if c is not None}


def average_ranks(values: Sequence[Fraction]) -> list[Fraction]:
    """Rank 1 for the largest value; tied values share the average of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [Fraction(0)] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for j in range(start, end + 1):
            ranks[order[j]] = Fraction(start + end + 2, 2)
        start = end + 1
    return ranks


# --- small helpers ---------------------------------------------------------------------------


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def _num(value: float | Fraction | None) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _table(run_id: str, columns: Sequence[Column], rows: Sequence[Sequence[Any]]) -> bytes:
    return dumps_table(sample_table(run_id, columns, rows))


def _write(
    runs_dir: Path,
    run_id: str,
    *,
    stage: str,
    seed: int | None,
    metrics: dict[str, dict[str, Any]],
    limitations: Sequence[str],
    inputs: Sequence[StoredRun],
    extra: dict[str, Any],
    files: Sequence[tuple[str, bytes]],
    git_commit: str,
    git_dirty: bool,
    protocol_path: Path,
) -> dict[str, Any]:
    manifest = build_manifest(
        run_id,
        stage=stage,
        experiment_id=f"cr3bp-free-return-m4-{stage.replace('_', '-')}-v1",
        seed=seed,
        methods=[],
        metrics=metrics,
        limitations=limitations,
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now(UTC),
        protocol_path=protocol_path,
        extra_m4={"inputs": [r.input_record() for r in inputs], **extra},
    )
    return write_run_directory(
        runs_dir / run_id, manifest, [("sample_table", path, data) for path, data in files]
    )


def _budget_label(b: float) -> str:
    return f"b{b:g}"


def read_budget(runs_dir: Path, run_id: str) -> tuple[StoredRun, float]:
    """B as recorded by a calibration run (an evidence-checked input)."""
    run = open_run(runs_dir, run_id)
    _require_stage(run, "calibration")
    budget = float(run.m4["calibration"]["budget_m_s"])
    if budget not in _GRID:
        raise EvidenceError(f"{run_id}: B = {budget} is not a grid value")
    return run, budget


@dataclass(frozen=True, slots=True)
class Selection:
    run: StoredRun
    budget_m_s: float
    configs: dict[str, int]
    satisfiable_tuning: tuple[tuple[str, int], ...]


def read_selection(runs_dir: Path, run_id: str) -> Selection:
    run = open_run(runs_dir, run_id)
    _require_stage(run, "selection")
    block: dict[str, Any] = run.m4["selection"]
    configs = {str(m): int(k) for m, k in block["configs"].items()}
    if set(configs) != set(METHODS):
        raise EvidenceError(f"{run_id}: the selection does not cover every method")
    return Selection(
        run=run,
        budget_m_s=float(block["budget_m_s"]),
        configs=configs,
        satisfiable_tuning=tuple((str(f), int(i)) for f, i in block["satisfiable_tuning_sets"]),
    )


def satisfiable_tuning_sets(selection: Selection) -> list[ConstraintSet]:
    keep = set(selection.satisfiable_tuning)
    return [c for c in tuning_sets() if (c.family, c.index) in keep]


def _successes_p2(elites: Sequence[Evaluation], kept: Sequence[ConstraintSet]) -> Fraction:
    if not kept:
        raise EvidenceError("no satisfiable constraint set: P2 is undefined")
    return Fraction(sum(availability(elites, kept)), len(kept))


def main_seeds(n: int) -> tuple[int, ...]:
    """The first n main seeds: 1001-1010, then the extension 1011-1016 (section 8)."""
    seeds = PROTOCOL.seeds.main + PROTOCOL.seeds.main_extension
    lo, hi = _STATS.seed_count_bounds
    if not lo <= n <= hi:
        raise ValueError(f"n must be in [{lo}, {hi}]")
    return seeds[:n]


# --- step 4: calibration of B (9.1) -------------------------------------------------------------


def run_calibration(
    runs_dir: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    run_id: str = CALIBRATION_RUN_ID,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """B: the smallest grid value whose median s_i(B), pooled over the elites of the two D
    tuning catalogs under T_E (primary Gates set), lies in [0.2, 0.8]; else the value whose
    median is closest to 0.5 (ties: the smaller B) (9.1)."""
    _refuse_existing(runs_dir, run_id)
    sources = [load_stage_run(runs_dir, "tune", "D", 0, s) for s in PROTOCOL.seeds.tuning]
    robs = [load_robustness(runs_dir, s, s.run.run_id, "T_E") for s in sources]
    inputs = [*(s.run for s in sources), *(r.run for r in robs)]
    require_evidence(inputs)
    survival = {b: [el.survival(b) for r in robs for el in r.elites] for b in _GRID}
    if not survival[_GRID[0]]:
        raise EvidenceError("the D tuning catalogs hold no elites: B cannot be calibrated")
    budget = calibrate_budget(survival)
    medians = {b: float(np.median(v)) for b, v in survival.items()}
    lo, hi = PROTOCOL.correction.calibration_band
    columns = (
        Column("budget_km_s", "km/s", "grid value of the correction budget B"),
        Column("median_survival", "dimensionless", "median s_i(B), pooled D tuning elites"),
        Column("elites", "count", "pooled elites of the D tuning catalogs (after 5.6)"),
        Column("in_band", "count", "1 if the median lies in [0.2, 0.8]"),
        Column("selected", "count", "1 for the calibrated B"),
    )
    rows = [
        [b / 1000.0, medians[b], len(survival[b]), int(lo <= medians[b] <= hi), int(b == budget)]
        for b in _GRID
    ]
    metrics = {
        "budget_km_s": _metric(budget / 1000.0, "km/s", "calibrated correction budget B (9.1)"),
        "elites_pooled": _metric(len(survival[budget]), "count", "D tuning elites pooled"),
        **{
            f"median_survival_{_budget_label(b)}": _metric(
                medians[b], "dimensionless", f"median s_i at B = {b:g} m/s (T_E, primary)"
            )
            for b in _GRID
        },
    }
    return _write(
        runs_dir,
        run_id,
        stage="calibration",
        seed=None,
        metrics=metrics,
        limitations=[
            "Reads stored runs only: the T_E robustness runs of the D tuning catalogs "
            "(seeds 9101, 9102) under the primary Gates set (protocol 9.1)."
        ],
        inputs=inputs,
        extra={
            "calibration": {"budget_m_s": budget, "medians": {f"{b:g}": medians[b] for b in _GRID}}
        },
        files=[("calibration.json", _table(run_id, columns, rows))],
        git_commit=git_commit,
        git_dirty=git_dirty,
        protocol_path=protocol_path,
    )


# --- step 5: configuration selection (section 6) ----------------------------------------------


@dataclass(frozen=True, slots=True)
class CatalogData:
    """A method run reduced to its re-checked catalog and its T_E robustness."""

    run: StoredRun
    method: str
    config_index: int
    seed: int
    catalog: dict[Cell, Elite]
    elites: tuple[Evaluation, ...]
    """The catalog's elite evaluations, in cell order."""
    robustness: RobustnessData


def catalog_data(source: MethodData, rob: RobustnessData) -> CatalogData:
    catalog = rechecked_catalog(source, rob)
    return CatalogData(
        run=source.run,
        method=source.method,
        config_index=source.config_index,
        seed=source.seed,
        catalog=catalog,
        elites=tuple(source.log[el.index] for el in catalog.values()),
        robustness=rob,
    )


def tuning_tasks() -> list[tuple[str, int, int]]:
    """(method, config index, seed) of the 32 tuning runs (section 6)."""
    return [
        (m, k, s)
        for m in METHODS
        for k in range(len(method_grid(m)))
        for s in PROTOCOL.seeds.tuning
    ]


@dataclass(frozen=True, slots=True)
class ConfigScore:
    method: str
    config_index: int
    p1: Fraction
    """Mean over the tuning seeds of P1 at B under T_E (primary Gates set)."""
    p2: Fraction
    """Mean over the tuning seeds of P2 on the satisfiable tuning sets."""
    coverage: Fraction
    """Mean over the tuning seeds of |catalog cells & G_tune| / |G_tune|."""
    rank_p1: Fraction
    rank_p2: Fraction
    selected: bool

    @property
    def rank_sum(self) -> Fraction:
        return self.rank_p1 + self.rank_p2


def select_configurations(
    scores: Mapping[tuple[str, int, int], tuple[Fraction, Fraction, Fraction]],
) -> tuple[dict[str, int], list[ConfigScore]]:
    """Section 6 selection from per-run (P1, P2, coverage) keyed by (method, k, seed).

    Per method: each configuration's quantities are averaged over its tuning seeds (the
    protocol scores configurations, not runs; the mean is the implemented reading); P1
    and P2 are ranked separately (rank 1 = highest, average ranks for ties); the lowest
    rank sum wins; ties go to the higher mean coverage against G_tune, then to the lower
    listed index. All arithmetic is exact (fractions), so ties are exact ties.
    """
    selected: dict[str, int] = {}
    rows: list[ConfigScore] = []
    for method in METHODS:
        ks = sorted({k for (m, k, _) in scores if m == method})
        if not ks:
            raise EvidenceError(f"no tuning scores for {method}")
        means: list[tuple[Fraction, Fraction, Fraction]] = []
        for k in ks:
            per_seed = [v for (m, kk, _), v in scores.items() if m == method and kk == k]
            count = len(per_seed)
            means.append(
                (
                    sum((v[0] for v in per_seed), Fraction(0)) / count,
                    sum((v[1] for v in per_seed), Fraction(0)) / count,
                    sum((v[2] for v in per_seed), Fraction(0)) / count,
                )
            )
        r1 = average_ranks([m[0] for m in means])
        r2 = average_ranks([m[1] for m in means])
        best = min(range(len(ks)), key=lambda i: (r1[i] + r2[i], -means[i][2], ks[i]))
        selected[method] = ks[best]
        rows.extend(
            ConfigScore(method, k, *means[i], r1[i], r2[i], i == best) for i, k in enumerate(ks)
        )
    return selected, rows


def configs_argument(configs: Mapping[str, int]) -> str:
    """The `--configs` value of the pilot and main stages (tuned methods only)."""
    return ",".join(f"{m}={configs[m]}" for m in METHODS if len(method_grid(m)) > 1)


def run_selection(
    runs_dir: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    calibration_run: str = CALIBRATION_RUN_ID,
    groundtruth_run: str = GROUNDTRUTH_RUN_ID,
    run_id: str = SELECTION_RUN_ID,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Configuration selection (section 6) at the calibrated B, and the satisfiable tuning
    constraint sets (9.2: some feasible propagation of GT sources (a)-(c) or of any tuning
    run satisfies the set; fixed here, before any pilot data)."""
    _refuse_existing(runs_dir, run_id)
    cal_run, budget = read_budget(runs_dir, calibration_run)
    gt = load_groundtruth(runs_dir, groundtruth_run)
    require_evidence([cal_run, gt.run])
    bins = PROTOCOL.descriptors.bins
    pool = FeasiblePool()
    pool.add(e.evaluation for e in gt.entries)
    g_tune = gt.cells(bins)
    data: list[CatalogData] = []
    for method, k, seed in tuning_tasks():
        src = load_stage_run(runs_dir, "tune", method, k, seed)
        rob = load_robustness(runs_dir, src, src.run.run_id, "T_E")
        require_evidence([src.run, rob.run])
        pool.add(src.log)
        g_tune |= _cells(src.log, bins)
        data.append(catalog_data(src, rob))
    sets = tuning_sets()
    mask = pool.satisfiable(sets)
    kept = [c for c, ok in zip(sets, mask, strict=True) if ok]
    scores: dict[tuple[str, int, int], tuple[Fraction, Fraction, Fraction]] = {}
    for d in data:
        scores[(d.method, d.config_index, d.seed)] = (
            d.robustness.p1(budget),
            _successes_p2(d.elites, kept),
            Fraction(len(set(d.catalog) & g_tune), len(g_tune)),
        )
    selected, configs = select_configurations(scores)
    run_cols = (
        Column("method", None, "method"),
        Column("config_index", "count", "grid index of the configuration"),
        Column("seed", "count", "tuning seed"),
        Column("p1", "dimensionless", "P1 = sum_i s_i at B, T_E, primary Gates set"),
        Column("p2", "dimensionless", "fraction of the satisfiable tuning sets available"),
        Column("coverage_g_tune", "dimensionless", "|catalog cells & G_tune| / |G_tune|"),
        Column("elites", "count", "catalog elites after the 5.6 re-check"),
    )
    run_rows: list[list[Any]] = []
    for d in data:
        v = scores[(d.method, d.config_index, d.seed)]
        run_rows.append([d.method, d.config_index, d.seed, *(float(x) for x in v), len(d.catalog)])
    cfg_cols = (
        Column("method", None, "method"),
        Column("config_index", "count", "grid index of the configuration"),
        Column("mean_p1", "dimensionless", "mean P1 over the tuning seeds"),
        Column("mean_p2", "dimensionless", "mean P2 over the tuning seeds"),
        Column("mean_coverage_g_tune", "dimensionless", "mean coverage against G_tune"),
        Column("rank_p1", "dimensionless", "rank of mean P1 within the method (1 = best)"),
        Column("rank_p2", "dimensionless", "rank of mean P2 within the method (1 = best)"),
        Column("rank_sum", "dimensionless", "rank_p1 + rank_p2 (lowest wins)"),
        Column("selected", "count", "1 for the selected configuration"),
    )
    cfg_rows = [
        [
            c.method,
            c.config_index,
            float(c.p1),
            float(c.p2),
            float(c.coverage),
            float(c.rank_p1),
            float(c.rank_p2),
            float(c.rank_sum),
            int(c.selected),
        ]
        for c in configs
    ]
    set_cols = (
        Column("family", None, "constraint family (F1, F2, F3)"),
        Column("index", "count", "set index within the family (generation order)"),
        Column("h_lo_km", "km", "lower perilune altitude"),
        Column("h_hi_km", "km", "upper perilune altitude"),
        Column("t_max_days", "day", "maximum return time T'"),
        Column("gamma_c_deg", "deg", "corridor centre (null: no corridor)"),
        Column("width_deg", "deg", "corridor half-width (null: no corridor)"),
        Column("satisfiable", "count", "1 if satisfiable by GT (a)-(c) or a tuning run"),
    )
    set_rows = [
        [c.family, c.index, c.h_lo_km, c.h_hi_km, c.t_max_days, c.gamma_c_deg, c.width_deg, int(ok)]
        for c, ok in zip(sets, mask, strict=True)
    ]
    metrics = {
        "budget_km_s": _metric(budget / 1000.0, "km/s", "calibrated B (input)"),
        "g_tune_cells": _metric(len(g_tune), "count", "|G_tune|: GT (a)-(c) and tuning runs"),
        "satisfiable_tuning_sets": _metric(len(kept), "count", "satisfiable tuning sets"),
        "unsatisfiable_tuning_sets": _metric(
            len(sets) - len(kept), "count", "tuning sets dropped as unsatisfiable"
        ),
        **{
            f"selected_config_{m.lower()}": _metric(k, "count", f"selected grid index of {m}")
            for m, k in selected.items()
        },
    }
    return _write(
        runs_dir,
        run_id,
        stage="selection",
        seed=None,
        metrics=metrics,
        limitations=[
            "Reads stored runs only: the 32 tuning runs, their T_E robustness runs, the "
            "ground-truth run (sources a-c) and the calibration run. Scores are means over "
            "the two tuning seeds; held-out sets were not evaluated.",
        ],
        inputs=[cal_run, gt.run, *(d.run for d in data), *(d.robustness.run for d in data)],
        extra={
            "selection": {
                "budget_m_s": budget,
                "configs": selected,
                "configs_argument": configs_argument(selected),
                "satisfiable_tuning_sets": [[c.family, c.index] for c in kept],
            }
        },
        files=[
            ("tuning_scores.json", _table(run_id, run_cols, run_rows)),
            ("selection.json", _table(run_id, cfg_cols, cfg_rows)),
            ("tuning_sets.json", _table(run_id, set_cols, set_rows)),
        ],
        git_commit=git_commit,
        git_dirty=git_dirty,
        protocol_path=protocol_path,
    )


# --- step 6: pilot (sections 8, 7.4, 12) ------------------------------------------------------


def dispersion(
    p1: Mapping[str, Mapping[int, float]], p2: Mapping[str, Mapping[int, float]]
) -> list[tuple[str, str, float, float]]:
    """(baseline X, endpoint, sigma_d, delta) for every X and endpoint: sigma_d is the SD
    (n - 1 denominator) of the paired differences T - X over the pilot-dispersion seeds."""
    rows: list[tuple[str, str, float, float]] = []
    for x in BASELINES:
        for endpoint, values, delta in (
            ("P1", p1, _STATS.effect_p1),
            ("P2", p2, _STATS.effect_p2),
        ):
            _, d = paired_differences(values["T"], values[x])
            rows.append((x, endpoint, float(np.std(d, ddof=1)), delta))
    return rows


def achieved_power(n: int, worst_ratio: float) -> float:
    """Normal-approximation power at n seeds, the inverse of the section-8 rule:
    Phi(sqrt(n) / max(sigma_d / delta) - z_alpha) (the protocol asks for the achieved power
    when n hits its cap but does not define it; this is the implemented reading)."""
    if worst_ratio <= 0.0:
        return 1.0
    return float(ndtr(math.sqrt(n) / worst_ratio - _STATS.z_alpha))


REDUCED_HELD_OUT_DRAWS: Final = 32
"""Section 7.4, pre-declared reduction 2: K per held-out error set drops from 50 to 32."""
TABLE_DRAW_PE_PRIMARY: Final = 3.65
"""Section 7.4 table: PE of one draw with the primary Gates set only (T_E, H_E-wide)."""
TABLE_DRAW_PE_BOTH: Final = 4.39
"""Section 7.4 table: PE of one draw with the primary and stress sets (H_E-nominal)."""


@dataclass(frozen=True, slots=True)
class ProjectionInputs:
    """The per-stage PE costs of the section 7.4 table (defaults: the table's values)."""

    gt_map: int = 2**20
    gt_refinement: float = 82_000.0
    tuning_runs: int = 32
    pilot_seeds: int = 3
    methods: int = 7
    elites_per_catalog: float = 30.0
    recheck_pe: float = 1.2
    setup_pe: float = 2.52
    draw_pe_primary: float = TABLE_DRAW_PE_PRIMARY
    draw_pe_both: float = TABLE_DRAW_PE_BOTH
    tuning_draws: int = 32
    held_out_draws: int = 50
    ccstar_pe: float = 18_000.0
    lsoda_pe: float = 5_000.0
    main_seeds: int = 10
    include_t2: bool = True


def projection(p: ProjectionInputs) -> dict[str, float]:
    """PE per stage and the total, counted as in the section 7.4 table."""
    n = PROTOCOL.run_budget
    t_e_elite = p.recheck_pe + p.setup_pe + p.tuning_draws * p.draw_pe_primary
    held_out_elite = (p.recheck_pe + p.setup_pe + p.held_out_draws * p.draw_pe_both) + (
        p.recheck_pe + p.setup_pe + p.held_out_draws * p.draw_pe_primary
    )
    main_methods = p.methods if p.include_t2 else p.methods - 1
    items = {
        "gt_map": float(p.gt_map),
        "gt_refinement": p.gt_refinement,
        "tuning_search": float(p.tuning_runs * n),
        "tuning_robustness": p.tuning_runs * p.elites_per_catalog * t_e_elite,
        "pilot_search": float(p.pilot_seeds * p.methods * n),
        "pilot_robustness": p.pilot_seeds * p.methods * p.elites_per_catalog * t_e_elite,
        "main_search": float(main_methods * p.main_seeds * n),
        "main_robustness": main_methods * p.main_seeds * p.elites_per_catalog * held_out_elite,
        "ccstar": p.ccstar_pe,
        "lsoda": p.lsoda_pe,
    }
    items["total"] = sum(items.values())
    return items


def projection_after_pilot(
    pe_per_draw: float, n: int
) -> tuple[list[tuple[str, dict[str, float]]], list[str], bool]:
    """Projection (b) of 7.4 with the measured PE per draw and the seed count n, then the
    pre-declared reductions 2 (K 50 -> 32) and 3 (drop T2) while the total exceeds the
    ceiling (reduction 1 is available at projection (a) only). Returns the scenarios, the
    reductions applied and whether K4 is triggered.

    Only primary-set draws are measured before the main runs; the H_E-nominal draw (primary
    and stress sets) is scaled by the table's ratio 4.39 / 3.65 (implemented reading)."""
    current = ProjectionInputs(
        draw_pe_primary=pe_per_draw,
        draw_pe_both=pe_per_draw * TABLE_DRAW_PE_BOTH / TABLE_DRAW_PE_PRIMARY,
        main_seeds=n,
    )
    scenarios = [("as_planned", projection(current))]
    applied: list[str] = []
    steps: tuple[tuple[str, Callable[[ProjectionInputs], ProjectionInputs]], ...] = (
        ("k_50_to_32", lambda p: replace(p, held_out_draws=REDUCED_HELD_OUT_DRAWS)),
        ("drop_t2", lambda p: replace(p, include_t2=False)),
    )
    for name, step in steps:
        if scenarios[-1][1]["total"] <= PE_CEILING:
            break
        current = step(current)
        applied.append(name)
        scenarios.append((name, projection(current)))
    return scenarios, applied, scenarios[-1][1]["total"] > PE_CEILING


def measured_pe_per_draw(robs: Iterable[RobustnessData]) -> float:
    draws = 0
    nfev = 0
    for rob in robs:
        for el in rob.elites:
            draws += el.draws
            nfev += sum(el.nfev)
    if draws == 0:
        raise EvidenceError("no robustness draws: the PE per draw cannot be measured")
    return nfev / draws / C1_PROPAGATION_NFEV


def run_pilot_analysis(
    runs_dir: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    calibration_run: str = CALIBRATION_RUN_ID,
    selection_run: str = SELECTION_RUN_ID,
    run_id: str = PILOT_ANALYSIS_RUN_ID,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """The pilot record of section 8 (seed count n), K3 and the projection (b) of 7.4.

    P1 (at B, T_E, primary Gates set) and P2 (satisfiable tuning sets) are taken on the 5
    seeds (2 tuning + 3 pilot) of each selected configuration. K3 uses the median s_i of
    the elites of each method's 3 pilot runs (the literal "pilot median s_i"). The PE per
    draw is measured on every T_E robustness run of the tuning and pilot stages.
    """
    _refuse_existing(runs_dir, run_id)
    cal_run, budget = read_budget(runs_dir, calibration_run)
    selection = read_selection(runs_dir, selection_run)
    require_evidence([cal_run, selection.run])
    if selection.budget_m_s != budget:
        raise EvidenceError("the selection run used another B than the calibration run")
    kept = satisfiable_tuning_sets(selection)
    inputs: list[StoredRun] = [cal_run, selection.run]
    p1: dict[str, dict[int, float]] = {}
    p2: dict[str, dict[int, float]] = {}
    medians: dict[str, float] = {}
    score_rows: list[list[Any]] = []
    robs: list[RobustnessData] = []
    for method in METHODS:
        k = selection.configs[method]
        pilot_survival: list[float] = []
        for stage, seeds in (("tune", PROTOCOL.seeds.tuning), ("pilot", PROTOCOL.seeds.pilot)):
            for seed in seeds:
                src = load_stage_run(runs_dir, stage, method, k, seed)
                rob = load_robustness(runs_dir, src, src.run.run_id, "T_E")
                require_evidence([src.run, rob.run])
                inputs += [src.run, rob.run]
                data = catalog_data(src, rob)
                v1, v2 = rob.p1(budget), _successes_p2(data.elites, kept)
                p1.setdefault(method, {})[seed] = float(v1)
                p2.setdefault(method, {})[seed] = float(v2)
                survival = [el.survival(budget) for el in rob.elites]
                if stage == "pilot":
                    pilot_survival += survival
                    robs.append(rob)
                score_rows.append([method, k, stage, seed, float(v1), float(v2), len(data.catalog)])
        medians[method] = float(np.median(pilot_survival)) if pilot_survival else math.nan
    for method, k, seed in tuning_tasks():
        rid = method_run_id("tune", method, k, seed)
        rob = load_robustness(runs_dir, None, rid, "T_E")
        require_evidence([rob.run])
        robs.append(rob)
    k3 = all(m < 0.05 or m > 0.95 for m in medians.values())
    disp = dispersion(p1, p2)
    ratios = [sd / delta for _, _, sd, delta in disp]
    n, raw = seed_count(ratios)
    power = achieved_power(n, max(ratios))
    pe_draw = measured_pe_per_draw(robs)
    scenarios, reductions, k4 = projection_after_pilot(pe_draw, n)
    final_total = scenarios[-1][1]["total"]
    seeds = main_seeds(n)
    score_cols = (
        Column("method", None, "method"),
        Column("config_index", "count", "selected grid index"),
        Column("stage", None, "tune or pilot"),
        Column("seed", "count", "seed"),
        Column("p1", "dimensionless", "P1 at B under T_E (primary Gates set)"),
        Column("p2", "dimensionless", "P2 on the satisfiable tuning sets"),
        Column("elites", "count", "catalog elites after the 5.6 re-check"),
    )
    disp_cols = (
        Column("baseline", None, "baseline X of the paired difference T - X"),
        Column("endpoint", None, "P1 or P2"),
        Column("sd", "dimensionless", "SD (n - 1) of T - X over the 5 dispersion seeds"),
        Column("effect", "dimensionless", "smallest effect of interest delta"),
        Column("ratio", "dimensionless", "sd / effect"),
    )
    proj_cols = (
        Column("scenario", None, "as_planned, then each pre-declared reduction applied"),
        Column("item", None, "stage of the section 7.4 table"),
        Column("pe", "count", "propagation equivalents"),
    )
    proj_rows = [[name, item, pe] for name, items in scenarios for item, pe in items.items()]
    metrics = {
        "budget_km_s": _metric(budget / 1000.0, "km/s", "calibrated B (input)"),
        "seed_count": _metric(n, "count", "main seeds n (section 8 count rule)"),
        "seed_count_unclamped": _metric(raw, "count", "ceil(13.315 max (sigma_d/delta)^2)"),
        "achieved_power": _metric(power, "dimensionless", "normal-approximation power at n"),
        "k3_triggered": _metric(int(k3), "count", "1 if every pilot median s_i is < .05 or > .95"),
        "pe_per_draw_t_e": _metric(pe_draw, "dimensionless", "measured PE per T_E draw"),
        "projection_total_pe": _metric(final_total, "count", "projection (b) after reductions"),
        "reductions_applied": _metric(len(reductions), "count", "pre-declared reductions needed"),
        "k4_triggered": _metric(int(k4), "count", "1 if the projection exceeds 7.8 M PE"),
        **{
            f"pilot_median_survival_{m.lower()}": _metric(
                v if math.isfinite(v) else -1.0,
                "dimensionless",
                f"median s_i of {m}'s pilot elites at B (T_E, primary; -1: no elites)",
            )
            for m, v in medians.items()
        },
    }
    notes = [
        "Reads stored runs only. P1 and P2 of the selected configurations on the tuning "
        "seeds 9101-9102 and pilot seeds 9201-9203 (section 8); the seed count depends on "
        "dispersion only.",
    ]
    if raw > n:
        notes.append(f"The count rule asked for {raw} seeds; n is capped at {n}.")
    if k3:
        notes.append("K3 TRIGGERED: P1 cannot discriminate; revise to v2 before the main runs.")
    if k4:
        notes.append("K4 TRIGGERED: the projection exceeds 7.8 M PE after the reductions.")
    return _write(
        runs_dir,
        run_id,
        stage="pilot_analysis",
        seed=None,
        metrics=metrics,
        limitations=notes,
        inputs=inputs + [r.run for r in robs if r.run not in inputs],
        extra={
            "pilot": {
                "budget_m_s": budget,
                "configs": selection.configs,
                "seed_count": n,
                "seed_count_unclamped": raw,
                "main_seeds": list(seeds),
                "achieved_power": power,
                "k3_triggered": k3,
                "pilot_median_survival": {m: _num(v) for m, v in medians.items()},
                "pe_per_draw_t_e": pe_draw,
                "projection_total_pe": final_total,
                "reductions": reductions,
                "k4_triggered": k4,
            }
        },
        files=[
            ("pilot_scores.json", _table(run_id, score_cols, score_rows)),
            ("seed_count.json", _table(run_id, disp_cols, [[*r, r[2] / r[3]] for r in disp])),
            ("projection.json", _table(run_id, proj_cols, proj_rows)),
        ],
        git_commit=git_commit,
        git_dirty=git_dirty,
        protocol_path=protocol_path,
    )


@dataclass(frozen=True, slots=True)
class PilotRecord:
    run: StoredRun
    seed_count: int
    main_seeds: tuple[int, ...]


def read_pilot(runs_dir: Path, run_id: str) -> PilotRecord:
    run = open_run(runs_dir, run_id)
    _require_stage(run, "pilot_analysis")
    block: dict[str, Any] = run.m4["pilot"]
    n = int(block["seed_count"])
    seeds = tuple(int(s) for s in block["main_seeds"])
    if seeds != main_seeds(n):
        raise EvidenceError(f"{run_id}: main seeds do not match n = {n}")
    return PilotRecord(run, n, seeds)


# --- step 8: the final analysis (sections 5.3 d, 5.6, 10, 11, 12) -------------------------------

type Values = dict[str, float | None]
"""Metric name -> value (None: undefined, e.g. the mean e of an empty catalog)."""

FAMILIES: Final = ("F1", "F2", "F3")


@dataclass(slots=True)
class ArmRun:
    """One main run (or CC*) reduced to what the section-10 metrics need."""

    method: str
    seed: int | None
    run: StoredRun
    budget: int
    catalog: dict[Cell, Elite]
    evaluations: dict[int, Evaluation]
    """The evaluations of every elite of `catalog` and of the checkpoint snapshots."""
    snapshots: dict[int, dict[Cell, Elite]]
    cells20: set[Cell]
    nominal: RobustnessData
    wide: RobustnessData
    values: Values
    anytime: dict[int, Values] = field(default_factory=dict[int, Values])


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def robustness_values(nominal: RobustnessData, wide: RobustnessData, budget: float) -> Values:
    """P1 and the per-solution robustness and correction diagnostics of section 10."""
    survival = [el.survival(budget) for el in nominal.elites]
    out: Values = {
        "p1": float(nominal.p1(budget)),
        "mean_s": _mean(survival),
        "min_s": min(survival) if survival else None,
        "p1_stress": float(nominal.p1(budget, STRESS_GATES)),
        "p1_wide": float(wide.p1(budget)),
    }
    for b in (0.0, *_GRID):
        label = _budget_label(b)
        out[f"p1_{label}"] = float(nominal.p1(b))
        out[f"mean_s_{label}"] = _mean([el.survival(b) for el in nominal.elites])
    for prefix, rob in (("", nominal), ("wide_", wide)):
        commanded = [c for el in rob.elites for c in el.commanded_km_s]
        draws = len(commanded)
        out[f"{prefix}commanded_fraction"] = (
            sum(c > 0.0 for c in commanded) / draws if draws else None
        )
        out[f"{prefix}median_commanded_km_s"] = float(np.median(commanded)) if draws else None
        out[f"{prefix}convergence_rate"] = (
            sum(sum(el.converged) for el in rob.elites) / draws if draws else None
        )
        out[f"{prefix}divergence_rate"] = (
            sum(sum(el.diverged) for el in rob.elites) / draws if draws else None
        )
        out[f"{prefix}singular_solutions"] = float(sum(el.singular for el in rob.elites))
    return out


def arm_run(
    source: MethodData, nominal: RobustnessData, wide: RobustnessData, budget: float
) -> ArmRun:
    """Everything of a main run that does not need G or the held-out sets."""
    if [(e.cell, e.index) for e in nominal.elites] != [(e.cell, e.index) for e in wide.elites]:
        raise EvidenceError(f"{source.run.run_id}: the 5.6 re-check differs between H_E sets")
    log = source.log
    catalog = rechecked_catalog(source, nominal)
    snapshots = {n: build_catalog(log, limit=n) for n in checkpoints_for(source.budget)}
    evaluations = {el.index: log[el.index] for el in catalog.values()}
    for snap in snapshots.values():
        evaluations.update({el.index: log[el.index] for el in snap.values()})
    first = next((i for i, e in enumerate(log) if e.feasible), None)
    q = quality(catalog)
    values: Values = {
        "elites": float(len(catalog)),
        "quality_mean_e": _num(q["mean"]),
        "quality_median_e": _num(q["median"]),
        "quality_p10_e": _num(q["p10"]),
        "qd_score": q["qd_score"],
        # CC* has no run budget N (its cap only guards a runaway): its yield is per
        # propagation used.
        "feasibility_yield": sum(e.feasible for e in log)
        / (max(1, len(log)) if source.method == "CCstar" else source.budget),
        "first_feasible_index": None if first is None else float(first),
        "diversity_classes": float(diversity_classes(catalog, log)),
        "propagations": float(len(log)),
        "rhs_evaluations": float(sum(e.nfev for e in log)),
        "wall_time_hours": source.wall_time_hours,
        "numerical_failures": float(sum(e.outcome is Outcome.NUMERICAL_FAILURE for e in log)),
        "gate_failures": float(sum(not e.gate_passed for e in log)),
        "rechecks": float(nominal.rechecks),
        "recheck_failures": float(nominal.recheck_failures),
        **robustness_values(nominal, wide, budget),
    }
    return ArmRun(
        method=source.method,
        seed=None if source.method == "CCstar" else source.seed,
        run=source.run,
        budget=source.budget,
        catalog=catalog,
        evaluations=evaluations,
        snapshots=snapshots,
        cells20=set(build_catalog(log, PROTOCOL.descriptors.secondary_bins)),
        nominal=nominal,
        wide=wide,
        values=values,
    )


def finish_arm(arm: ArmRun, g10: set[Cell], g20: set[Cell], kept: Sequence[ConstraintSet]) -> None:
    """Coverage (10 x 10 and 20 x 20), P2, P2 by family and the anytime values."""
    elites = [arm.evaluations[el.index] for el in arm.catalog.values()]
    arm.values["coverage"] = len(set(arm.catalog) & g10) / len(g10)
    arm.values["coverage_20"] = len(arm.cells20 & g20) / len(g20)
    arm.values["p2"] = float(_successes_p2(elites, kept))
    for family in FAMILIES:
        sets = [c for c in kept if c.family == family]
        arm.values[f"p2_{family}"] = float(_successes_p2(elites, sets)) if sets else None
    for n, snap in arm.snapshots.items():
        snap_elites = [arm.evaluations[el.index] for el in snap.values()]
        q = quality(snap)
        arm.anytime[n] = {
            "coverage": len(set(snap) & g10) / len(g10),
            "quality_mean_e": _num(q["mean"]),
            "qd_score": q["qd_score"],
            "p2": float(_successes_p2(snap_elites, kept)),
        }


PER_RUN_COLUMNS: Final[tuple[Column, ...]] = (
    Column("p1", "dimensionless", "P1 = sum_i s_i at B (H_E-nominal, primary Gates set)"),
    Column("p2", "dimensionless", "fraction of the satisfiable held-out sets available"),
    Column("p2_F1", "dimensionless", "P2 on the satisfiable held-out F1 sets"),
    Column("p2_F2", "dimensionless", "P2 on the satisfiable held-out F2 sets"),
    Column("p2_F3", "dimensionless", "P2 on the satisfiable held-out F3 sets"),
    Column("coverage", "dimensionless", "|catalog cells & G| / |G| (10 x 10, after 5.6)"),
    Column("coverage_20", "dimensionless", "the same on the 20 x 20 grid (no re-check)"),
    Column("elites", "count", "catalog elites after the 5.6 re-check"),
    Column("quality_mean_e", "deg", "mean e over the elites"),
    Column("quality_median_e", "deg", "median e over the elites"),
    Column("quality_p10_e", "deg", "10th percentile of e"),
    Column("qd_score", "deg", "QD-score sum over cells of (90 deg - e)"),
    Column("mean_s", "dimensionless", "mean s_i at B (H_E-nominal, primary)"),
    Column("min_s", "dimensionless", "min s_i at B (H_E-nominal, primary)"),
    *(
        Column(f"p1_{_budget_label(b)}", "dimensionless", f"P1 at B = {b:g} m/s")
        for b in (0.0, *_GRID)
    ),
    *(
        Column(f"mean_s_{_budget_label(b)}", "dimensionless", f"mean s_i at B = {b:g} m/s")
        for b in (0.0, *_GRID)
    ),
    Column("p1_stress", "dimensionless", "P1 at B under the Gates stress set (H_E-nominal)"),
    Column("p1_wide", "dimensionless", "P1 at B under H_E-wide (primary)"),
    *(
        col
        for prefix, label in (("", "H_E-nominal"), ("wide_", "H_E-wide"))
        for col in (
            Column(f"{prefix}commanded_fraction", "dimensionless", f"{label}: draws with an MCC"),
            Column(
                f"{prefix}median_commanded_km_s", "km/s", f"{label}: median |dv_cmd|, all draws"
            ),
            Column(f"{prefix}convergence_rate", "dimensionless", f"{label}: chord converged"),
            Column(f"{prefix}divergence_rate", "dimensionless", f"{label}: chord diverged"),
            Column(f"{prefix}singular_solutions", "count", f"{label}: elites with singular J"),
        )
    ),
    Column("feasibility_yield", "dimensionless", "feasible propagations / N (CC*: / used)"),
    Column("first_feasible_index", "count", "index of the first feasible propagation"),
    Column("diversity_classes", "count", "distinct (lunar passage, entry sense) classes"),
    Column("propagations", "count", "propagations used"),
    Column("rhs_evaluations", "count", "right-hand-side evaluations"),
    Column("wall_time_hours", "hour", "method wall time"),
    Column("numerical_failures", "count", "propagations classified numerical_failure"),
    Column("gate_failures", "count", "propagations failing the 5.2 gate"),
    Column("rechecks", "count", "5.6 re-checks (H_E-nominal robustness run)"),
    Column("recheck_failures", "count", "elites rejected by the 5.6 re-check"),
)
ANYTIME_METRICS: Final = ("coverage", "quality_mean_e", "qd_score", "p2")
SUMMARY_METRICS: Final = tuple(c.name for c in PER_RUN_COLUMNS)
RESULT_COLUMNS: Final[tuple[Column, ...]] = (
    Column(
        "section",
        None,
        "summary, anytime, primary, h1, secondary, cc_star, h2, criterion, "
        "ground_truth, constraints, lsoda or design",
    ),
    Column("metric", None, "metric (per_run.json column) or quantity"),
    Column("arm", None, "method, contrast arm or null"),
    Column("baseline", None, "baseline X of a contrast arm - X, or null"),
    Column("checkpoint", "count", "anytime checkpoint (propagations), or null"),
    Column("statistic", None, "mean, median, sd, min, max, worst_seed, mean_difference, ..."),
    Column("value", "dimensionless", "value in the metric's unit (see per_run.json)"),
    Column("ci_low", "dimensionless", "95 % percentile bootstrap CI over seeds (seed 4242)"),
    Column("ci_high", "dimensionless", "upper end of that CI"),
    Column("p_value", "dimensionless", "exact two-sided sign-flip p (H2: permutation p)"),
    Column("p_holm", "dimensionless", "Holm-adjusted p (the ten primary contrasts only)"),
    Column("significant", "count", "1 if p_holm <= 0.05 and the difference favours T"),
    Column("note", None, "remarks"),
)


@dataclass(slots=True)
class Results:
    rows: list[list[Any]] = field(default_factory=list[list[Any]])

    def add(
        self,
        section: str,
        metric: str | None,
        statistic: str,
        value: float | None,
        *,
        arm: str | None = None,
        baseline: str | None = None,
        checkpoint: int | None = None,
        ci: tuple[float, float] | None = None,
        p: float | None = None,
        p_holm: float | None = None,
        significant: bool | None = None,
        note: str | None = None,
    ) -> None:
        self.rows.append(
            [
                section,
                metric,
                arm,
                baseline,
                checkpoint,
                statistic,
                _num(value),
                None if ci is None else _num(ci[0]),
                None if ci is None else _num(ci[1]),
                _num(p),
                _num(p_holm),
                None if significant is None else int(significant),
                note,
            ]
        )


def _by_seed(arms: Sequence[ArmRun], method: str, metric: str) -> dict[int, float | None]:
    return {a.seed: a.values.get(metric) for a in arms if a.method == method and a.seed is not None}


def _defined(values: Mapping[int, float | None]) -> dict[int, float] | None:
    out = {s: v for s, v in values.items() if v is not None}
    return out if len(out) == len(values) and out else None


@dataclass(frozen=True, slots=True)
class PairedEstimate:
    mean_difference: float
    ci: tuple[float, float]
    p_value: float


def paired_estimate(
    treatment: Mapping[int, float | None], baseline: Mapping[int, float | None]
) -> PairedEstimate | None:
    """Mean paired difference, bootstrap CI and sign-flip p; None if a value is undefined."""
    t, x = _defined(treatment), _defined(baseline)
    if t is None or x is None:
        return None
    _, d = paired_differences(t, x)
    return PairedEstimate(float(d.mean()), bootstrap_ci(d), sign_flip_pvalue(d))


@dataclass(frozen=True, slots=True)
class PrimaryContrast:
    endpoint: str
    baseline: str
    estimate: PairedEstimate
    p_holm: float
    favours_t: bool
    """Holm-significant at alpha = 0.05 with a positive mean difference T - X."""


def primary_contrasts(arms: Sequence[ArmRun]) -> list[PrimaryContrast]:
    """The ten contrasts T - X (X in the baselines) on P1 and P2, Holm across the ten."""
    estimates: list[tuple[str, str, PairedEstimate]] = []
    for endpoint, metric in (("P1", "p1"), ("P2", "p2")):
        for x in BASELINES:
            est = paired_estimate(_by_seed(arms, "T", metric), _by_seed(arms, x, metric))
            if est is None:
                raise EvidenceError(f"{endpoint} is undefined for T or {x}")
            estimates.append((endpoint, x, est))
    adjusted = holm([e.p_value for _, _, e in estimates])
    return [
        PrimaryContrast(ep, x, est, q, q <= _STATS.alpha and est.mean_difference > 0.0)
        for (ep, x, est), q in zip(estimates, adjusted, strict=True)
    ]


def h1_status(arms: Sequence[ArmRun], contrasts: Sequence[PrimaryContrast]) -> dict[str, str]:
    """Per baseline (section 2): 'refuted' if X's mean >= T's on both P1 and P2, else
    'supported' if T is Holm-significantly better on both endpoints, else 'not_supported'."""
    out: dict[str, str] = {}
    for x in BASELINES:
        means = {
            ep: (
                float(np.mean([v for v in _by_seed(arms, "T", m).values() if v is not None])),
                float(np.mean([v for v in _by_seed(arms, x, m).values() if v is not None])),
            )
            for ep, m in (("P1", "p1"), ("P2", "p2"))
        }
        better = [c.favours_t for c in contrasts if c.baseline == x]
        if all(xm >= tm for tm, xm in means.values()):
            out[x] = "refuted"
        elif all(better):
            out[x] = "supported"
        else:
            out[x] = "not_supported"
    return out


def h2_test(arms: Sequence[ArmRun], budget: float) -> tuple[float | None, float | None, int]:
    """Spearman rho of s_i (B, H_E-nominal, primary) with the elite-selection margin within
    T over the main seeds (elites without a next-best evaluation excluded), and the p of
    10 000 permutations of the margins within each seed (generator default_rng(4242))."""
    s: list[float] = []
    margins: list[float] = []
    strata: list[int] = []
    for arm in arms:
        if arm.method != "T" or arm.seed is None:
            continue
        for el in arm.nominal.elites:
            margin = arm.catalog[el.cell].margin_deg
            if math.isfinite(margin):
                s.append(el.survival(budget))
                margins.append(margin)
                strata.append(arm.seed)
    if len(s) < 3 or len(set(s)) < 2 or len(set(margins)) < 2:
        return None, None, len(s)
    rho, p = stratified_permutation_pvalue(s, margins, strata)
    return rho, p, len(s)


def lsoda_validation(elite: Evaluation) -> tuple[bool, tuple[str, ...]]:
    """The full M3 S3 validation (run, DOP853 1e-13 and LSODA 1e-13, plus the S3 checks) of
    one elite: (passed, names of the failed checks)."""
    from phasespace.gravity.free_return_validation import validate_candidate

    v = validate_candidate(elite.phase_deg, elite.delta_v_km_s, elite.impulse_angle_deg)
    return v.passed, tuple(c.name for c in v.checks if not c.passed)


type Validator = Callable[[Evaluation], tuple[bool, tuple[str, ...]]]


def _validate_all(
    validate: Validator, elites: Sequence[Evaluation], workers: int
) -> list[tuple[bool, tuple[str, ...]]]:
    if workers <= 1 or len(elites) < 2:
        return [validate(e) for e in elites]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(validate, elites))


def run_analysis(
    runs_dir: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    confirm_frozen: bool,
    calibration_run: str = CALIBRATION_RUN_ID,
    selection_run: str = SELECTION_RUN_ID,
    pilot_run: str = PILOT_ANALYSIS_RUN_ID,
    groundtruth_run: str = GROUNDTRUTH_RUN_ID,
    ccstar_run: str = CCSTAR_RUN_ID,
    run_id: str = ANALYSIS_RUN_ID,
    workers: int = 1,
    validate: Validator = lsoda_validation,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    """The final analysis (freeze checklist step 8). Reads stored runs only, except the
    seeded 5 % LSODA re-check sample of the pooled main-run elites (5.6), which propagates.

    Implemented readings where the text leaves room: (d) of 5.3 uses the tuning, pilot and
    main runs of the protocol (not CC*); the anytime values use the stored checkpoint
    snapshots (no re-check at checkpoints: 7.3 says they need no new propagations); the
    median |dv_cmd| is over all draws; K2 uses the re-check failures of the main
    H_E-nominal runs over their re-checks and each method's gate failures over its main
    propagations, and the LSODA sample's failure fraction is reported beside it.
    """
    require_frozen(confirm_frozen, protocol_path)
    _refuse_existing(runs_dir, run_id)
    bins, bins20 = PROTOCOL.descriptors.bins, PROTOCOL.descriptors.secondary_bins
    cal_run, budget = read_budget(runs_dir, calibration_run)
    selection = read_selection(runs_dir, selection_run)
    pilot = read_pilot(runs_dir, pilot_run)
    gt = load_groundtruth(runs_dir, groundtruth_run)
    require_evidence([cal_run, selection.run, pilot.run, gt.run])
    if selection.budget_m_s != budget:
        raise EvidenceError("the selection run used another B than the calibration run")
    inputs: list[StoredRun] = [cal_run, selection.run, pilot.run, gt.run]
    pool = FeasiblePool()
    pool.add(e.evaluation for e in gt.entries)
    gt_evals = [e.evaluation for e in gt.entries]
    g10 = gt.cells(bins)
    g20 = _cells(gt_evals, bins20)
    method_cells10: set[Cell] = set()
    method_cells20: set[Cell] = set()

    def absorb(src: MethodData) -> None:
        require_evidence([src.run])
        inputs.append(src.run)
        pool.add(src.log)
        method_cells10.update(_cells(src.log, bins))
        method_cells20.update(_cells(src.log, bins20))

    for method, k, seed in tuning_tasks():
        absorb(load_stage_run(runs_dir, "tune", method, k, seed))
    for method in METHODS:
        for seed in PROTOCOL.seeds.pilot:
            absorb(load_stage_run(runs_dir, "pilot", method, selection.configs[method], seed))
    arms: list[ArmRun] = []
    for method in METHODS:
        for seed in pilot.main_seeds:
            src = load_stage_run(runs_dir, "main", method, selection.configs[method], seed)
            absorb(src)
            robs = [
                load_robustness(
                    runs_dir,
                    src,
                    src.run.run_id,
                    errors,
                    confirm_frozen=confirm_frozen,
                    protocol_path=protocol_path,
                )
                for errors in ("H_E_nominal", "H_E_wide")
            ]
            require_evidence([r.run for r in robs])
            inputs += [r.run for r in robs]
            arms.append(arm_run(src, robs[0], robs[1], budget))
    cc_src = load_method_run(
        runs_dir, ccstar_run, stage="reference", method="CCstar", config_index=0, seed=0
    )
    cc_robs = [
        load_robustness(
            runs_dir,
            cc_src,
            ccstar_run,
            errors,
            confirm_frozen=confirm_frozen,
            protocol_path=protocol_path,
        )
        for errors in ("H_E_nominal", "H_E_wide")
    ]
    require_evidence([cc_src.run, *(r.run for r in cc_robs)])
    inputs += [cc_src.run, *(r.run for r in cc_robs)]
    cc_star = arm_run(cc_src, cc_robs[0], cc_robs[1], budget)

    # Ground truth G = (a)-(d) and the saturation check (5.3, K7).
    gt_counts = {
        "prefix": (
            len(gt.cells(bins, "prefix")),
            len(
                _cells(
                    [
                        e.evaluation
                        for e in gt.entries
                        if e.tag == "gt_map"
                        and e.index < 2**PROTOCOL.ground_truth.saturation_prefix_log2
                    ],
                    bins20,
                )
            ),
        ),
        "map": (
            len(gt.cells(bins, "map")),
            len(_cells([e.evaluation for e in gt.entries if e.tag == "gt_map"], bins20)),
        ),
        "a_c": (len(g10), len(g20)),
    }
    g10 |= method_cells10
    g20 |= method_cells20
    gt_counts["a_d"] = (len(g10), len(g20))
    k7 = not saturated(gt_counts["prefix"][0], gt_counts["a_d"][0])

    # Held-out constraint sets and their satisfiability (9.2).
    sets = held_out_sets(confirm_frozen=confirm_frozen, protocol_path=protocol_path)
    mask = pool.satisfiable(sets)
    kept = [c for c, ok in zip(sets, mask, strict=True) if ok]
    for arm in (*arms, cc_star):
        finish_arm(arm, g10, g20, kept)

    results = Results()
    results.add("design", "budget_m_s", "value", budget, note="calibrated B (9.1)")
    results.add("design", "seed_count", "value", float(pilot.seed_count))
    for name, (c10, c20) in gt_counts.items():
        results.add("ground_truth", "cells_10", name, float(c10))
        results.add("ground_truth", "cells_20", name, float(c20))
    for family in FAMILIES:
        fam = [ok for c, ok in zip(sets, mask, strict=True) if c.family == family]
        results.add("constraints", f"satisfiable_{family}", "count", float(sum(fam)))
        results.add("constraints", f"unsatisfiable_{family}", "count", float(len(fam) - sum(fam)))

    # Summaries per method and metric over the main seeds (11), CC* as a single value.
    for method in METHODS:
        for metric in SUMMARY_METRICS:
            by_seed = _by_seed(arms, method, metric)
            values = [v for v in by_seed.values() if v is not None]
            if not values:
                continue
            note = None if len(values) == len(by_seed) else f"defined for {len(values)} seeds"
            for stat, value in summary(values).items():
                results.add("summary", metric, stat, value, arm=method, note=note)
            if metric in ("p1", "p2"):
                worst = min((v, s) for s, v in by_seed.items() if v is not None)
                results.add(
                    "summary", metric, "worst_seed", worst[0], arm=method, note=f"seed {worst[1]}"
                )
        for checkpoint in sorted(next(a for a in arms if a.method == method).anytime):
            for metric in ANYTIME_METRICS:
                values = [
                    v
                    for a in arms
                    if a.method == method and (v := a.anytime[checkpoint][metric]) is not None
                ]
                if values:
                    results.add(
                        "anytime",
                        metric,
                        "mean",
                        float(np.mean(values)),
                        arm=method,
                        checkpoint=checkpoint,
                        ci=bootstrap_ci(values),
                    )
    for metric in SUMMARY_METRICS:
        results.add("summary", metric, "value", cc_star.values.get(metric), arm="CCstar")

    # Primary contrasts with Holm (11), H1 per baseline (2).
    contrasts = primary_contrasts(arms)
    for c in contrasts:
        results.add(
            "primary",
            c.endpoint,
            "mean_difference",
            c.estimate.mean_difference,
            arm="T",
            baseline=c.baseline,
            ci=c.estimate.ci,
            p=c.estimate.p_value,
            p_holm=c.p_holm,
            significant=c.favours_t,
        )
    status = h1_status(arms, contrasts)
    for x, verdict in status.items():
        results.add("h1", "H1", "status", None, arm="T", baseline=x, note=verdict)

    # Secondary analyses: estimates with CIs and unadjusted p, outside the Holm family.
    def secondary(metric: str, arm: str, baseline: str, note: str | None = None) -> None:
        est = paired_estimate(_by_seed(arms, arm, metric), _by_seed(arms, baseline, metric))
        if est is None:
            results.add(
                "secondary",
                metric,
                "mean_difference",
                None,
                arm=arm,
                baseline=baseline,
                note="undefined for some seed",
            )
            return
        results.add(
            "secondary",
            metric,
            "mean_difference",
            est.mean_difference,
            arm=arm,
            baseline=baseline,
            ci=est.ci,
            p=est.p_value,
            note=note,
        )

    for x in (*BASELINES, "T"):
        for metric in ("p1", "p2"):
            secondary(metric, "T2", x)
    for x in BASELINES:
        for metric in (
            "coverage",
            "coverage_20",
            "quality_mean_e",
            "quality_median_e",
            "qd_score",
            "p1_wide",
            "p1_stress",
            *(f"p1_{_budget_label(b)}" for b in (0.0, *_GRID)),
            *(f"p2_{f}" for f in FAMILIES),
        ):
            secondary(metric, "T", x)
    for method in METHODS:
        est = paired_estimate(_by_seed(arms, method, "p1_stress"), _by_seed(arms, method, "p1"))
        if est is not None:
            results.add(
                "secondary",
                "p1_stress_minus_p1",
                "mean_difference",
                est.mean_difference,
                arm=method,
                ci=est.ci,
                p=est.p_value,
                note="stress - primary Gates set, paired",
            )
    for x in METHODS:
        for metric in ("p1", "p2", "coverage", "mean_s", "quality_mean_e", "qd_score"):
            cc_value = cc_star.values.get(metric)
            xs = _defined(_by_seed(arms, x, metric))
            if cc_value is None or xs is None:
                continue
            d = [cc_value - v for v in xs.values()]
            results.add(
                "cc_star",
                metric,
                "mean_difference",
                float(np.mean(d)),
                arm="CCstar",
                baseline=x,
                ci=bootstrap_ci(d),
                note="unmatched reference (not budget-matched)",
            )

    # H2 (11): Spearman within T, and the mean s_i of T vs D.
    rho, p_h2, pairs = h2_test(arms, budget)
    results.add("h2", "spearman_rho_s_margin", "rho", rho, arm="T", p=p_h2, note=f"{pairs} elites")
    est = paired_estimate(_by_seed(arms, "T", "mean_s"), _by_seed(arms, "D", "mean_s"))
    if est is not None:
        results.add(
            "h2",
            "mean_s",
            "mean_difference",
            est.mean_difference,
            arm="T",
            baseline="D",
            ci=est.ci,
            p=est.p_value,
        )

    # The 5 % LSODA sample of the pooled main-run elites, order (method, seed, cell) (5.6).
    pooled = [
        (arm, el)
        for method in METHODS
        for arm in sorted((a for a in arms if a.method == method), key=lambda a: a.seed or 0)
        for el in arm.catalog.values()
    ]
    sample = lsoda_sample(len(pooled))
    checked = _validate_all(
        validate, [pooled[i][0].evaluations[pooled[i][1].index] for i in sample], workers
    )
    lsoda_rows = [
        [
            i,
            pooled[i][0].method,
            pooled[i][0].seed,
            *pooled[i][1].cell,
            pooled[i][1].index,
            int(ok),
            ";".join(failed),
        ]
        for i, (ok, failed) in zip(sample, checked, strict=True)
    ]
    lsoda_failures = sum(not ok for ok, _ in checked)
    results.add("lsoda", "pooled_elites", "count", float(len(pooled)))
    results.add("lsoda", "sample", "count", float(len(sample)))
    results.add("lsoda", "failures", "count", float(lsoda_failures))

    # Stop criteria (12).
    stop = PROTOCOL.gate.stop_fraction
    rechecks = sum(a.nominal.rechecks for a in arms)
    recheck_failures = sum(a.nominal.recheck_failures for a in arms)
    recheck_fraction = recheck_failures / rechecks if rechecks else 0.0
    gate_fraction = {
        m: sum(a.values["gate_failures"] or 0.0 for a in arms if a.method == m)
        / sum(a.values["propagations"] or 0.0 for a in arms if a.method == m)
        for m in METHODS
    }
    lsoda_fraction = lsoda_failures / len(sample) if sample else 0.0
    k2 = recheck_fraction > stop or any(f > stop for f in gate_fraction.values())
    results.add("criterion", "K2_recheck_fraction", "value", recheck_fraction)
    for m, f in gate_fraction.items():
        results.add("criterion", "K2_gate_fraction", "value", f, arm=m)
    results.add(
        "criterion",
        "K2_lsoda_fraction",
        "value",
        lsoda_fraction,
        note="failure fraction of the 5 % LSODA sample (5.6); reported beside K2",
    )
    results.add("criterion", "K2", "triggered", float(k2))
    means = {
        (m, metric): float(
            np.mean([v for v in _by_seed(arms, m, metric).values() if v is not None])
        )
        for m in ("T", "D", "CC")
        for metric in ("p1", "p2", "coverage")
    }
    k5 = {
        x: all(means[(x, metric)] >= means[("T", metric)] for metric in ("p1", "p2", "coverage"))
        for x in ("D", "CC")
    }
    for x, flag in k5.items():
        results.add("criterion", "K5", "triggered", float(flag), baseline=x)
    t_beats_d = any(c.favours_t for c in contrasts if c.baseline == "D")
    k6 = all(v != "supported" for v in status.values()) and not t_beats_d
    results.add("criterion", "K6", "triggered", float(k6))
    results.add("criterion", "K7", "triggered", float(k7))

    per_run_cols = (
        Column("method", None, "method (CCstar: the unmatched reference)"),
        Column("config_index", "count", "selected grid index"),
        Column("seed", "count", "main seed (null for CC*)"),
        Column("run_id", None, "method run"),
        *PER_RUN_COLUMNS,
    )
    per_run_rows = [
        [
            a.method,
            selection.configs.get(a.method, 0),
            a.seed,
            a.run.run_id,
            *(_num(a.values.get(c.name)) for c in PER_RUN_COLUMNS),
        ]
        for a in (*arms, cc_star)
    ]
    anytime_cols = (
        Column("method", None, "method"),
        Column("seed", "count", "main seed (null for CC*)"),
        Column("checkpoint", "count", "catalog over the first n propagations (pre-re-check)"),
        Column("coverage", "dimensionless", "|snapshot cells & G| / |G|"),
        Column("quality_mean_e", "deg", "mean e of the snapshot"),
        Column("qd_score", "deg", "QD-score of the snapshot"),
        Column("p2", "dimensionless", "P2 (held-out) of the snapshot"),
    )
    anytime_rows = [
        [a.method, a.seed, n, *(_num(v[m]) for m in ANYTIME_METRICS)]
        for a in (*arms, cc_star)
        for n, v in sorted(a.anytime.items())
    ]
    lsoda_cols = (
        Column("pool_index", "count", "index in the pooled (method, seed, cell) order"),
        Column("method", None, "method"),
        Column("seed", "count", "main seed"),
        Column("cell_d1", "count", "bin of log10 h_p"),
        Column("cell_d2", "count", "bin of t_ret"),
        Column("index", "count", "propagation index of the elite"),
        Column("passed", "count", "1 if every M3 S3 check passed"),
        Column("failed_checks", None, "names of the failed checks (;-separated)"),
    )
    set_cols = (
        Column("family", None, "constraint family"),
        Column("index", "count", "set index within the family"),
        Column("satisfiable", "count", "1 if some feasible propagation of (a)-(d) satisfies it"),
    )
    set_rows = [[c.family, c.index, int(ok)] for c, ok in zip(sets, mask, strict=True)]
    metrics = {
        "budget_km_s": _metric(budget / 1000.0, "km/s", "calibrated B"),
        "seed_count": _metric(pilot.seed_count, "count", "main seeds n"),
        "ground_truth_cells": _metric(len(g10), "count", "|G| (10 x 10) from sources (a)-(d)"),
        "heldout_satisfiable_sets": _metric(len(kept), "count", "satisfiable held-out sets"),
        "lsoda_sample": _metric(len(sample), "count", "elites in the 5 % LSODA sample"),
        "lsoda_failures": _metric(lsoda_failures, "count", "failed full M3 S3 validations"),
        "k2_triggered": _metric(int(k2), "count", "K2 (re-check or gate failures > 1 %)"),
        "k5_triggered": _metric(int(any(k5.values())), "count", "K5 (D or CC >= T)"),
        "k6_triggered": _metric(int(k6), "count", "K6 (negative result)"),
        "k7_triggered": _metric(int(k7), "count", "K7 (coverage only an upper bound)"),
        **{
            f"h1_{x.lower()}_supported": _metric(
                int(v == "supported"), "count", f"1 if H1 is supported against {x}"
            )
            for x, v in status.items()
        },
    }
    limitations = [
        "Reads stored runs only, except the 5 % LSODA sample (full M3 S3 validation), which "
        "propagates. No elite was designated as shown on the site, so the LSODA check covers "
        "the seeded sample only.",
        "Anytime values use the stored checkpoint snapshots, which are not re-checked (5.6 "
        "re-checks are made for the final catalogs only).",
        "Ground truth (d) unions the tuning, pilot and main runs; CC* is not included.",
        "CC* is an oracle-seeded reference outside the hypothesis tests, not budget-matched.",
    ]
    return _write(
        runs_dir,
        run_id,
        stage="analysis",
        seed=_STATS.bootstrap_seed,
        metrics=metrics,
        limitations=limitations,
        inputs=inputs,
        extra={
            "analysis": {
                "budget_m_s": budget,
                "configs": selection.configs,
                "main_seeds": list(pilot.main_seeds),
                "h1": status,
                "criteria": {"K2": k2, "K5": k5, "K6": k6, "K7": k7},
            }
        },
        files=[
            ("per_run.json", _table(run_id, per_run_cols, per_run_rows)),
            ("anytime.json", _table(run_id, anytime_cols, anytime_rows)),
            ("results.json", _table(run_id, RESULT_COLUMNS, results.rows)),
            ("lsoda.json", _table(run_id, lsoda_cols, lsoda_rows)),
            ("heldout_sets.json", _table(run_id, set_cols, set_rows)),
        ],
        git_commit=git_commit,
        git_dirty=git_dirty,
        protocol_path=protocol_path,
    )
