"""M4 run artifacts (docs/data-contract.md, "M4 runs"): manifests and sample tables.

A method run writes `evaluations[-<method>].json` (every propagation, in order, with its
purpose tag) and `catalog[-<method>].json` (the 10 x 10 catalog at each anytime
checkpoint plus the 20 x 20 sensitivity catalog at the end). The manifest's `m4` block
records the protocol version, its status when the run was made, the configuration
hash, the stage, and per method the hyperparameters, seed, budget and propagations used.
Runs are written atomically and never overwrite an existing directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from phasespace.gravity.export import (
    REPOSITORY,
    SCHEMA_VERSION,
    STUDY_ID,
    collision_events_json,
    environment_info,
    model_json,
    solver_json,
    write_run_directory,
)
from phasespace.gravity.free_return import DEFAULT_DEFINITION, FreeReturnDefinition, Outcome
from phasespace.gravity.free_return_map import Column, dumps_table, sample_table
from phasespace.gravity.m4.catalog import build_catalog, checkpoint_catalogs
from phasespace.gravity.m4.config import (
    PROTOCOL,
    PROTOCOL_PATH,
    PROTOCOL_RELATIVE_PATH,
    checkpoints_for,
    config_sha256,
    protocol_status,
)
from phasespace.gravity.m4.evaluation import Evaluation, LogEntry, cell_of
from phasespace.gravity.m4.methods import MethodRun
from phasespace.gravity.m4.metrics import feasibility_yield, numerical_failures

GENERATOR: Final = "phasespace.gravity.m4.cli"
ASSUMPTIONS: Final = (
    "Planar circular restricted three-body model with a single impulsive departure from an "
    "idealized circular parking orbit (M3 definitions, D20): no lunar eccentricity, "
    "out-of-plane motion, Earth oblateness, solar or other perturbations, finite burns, "
    "navigation errors or ephemeris effects."
)

EVALUATION_COLUMNS: Final[tuple[Column, ...]] = (
    Column("index", "count", "propagation index in the run (0-based; the budget counter)"),
    Column("tag", None, "purpose of the propagation within the method (e.g. sobol, local, de)"),
    Column("u_phase", "dimensionless", "unit-cube coordinate of the departure phase"),
    Column("u_delta_v", "dimensionless", "unit-cube coordinate of the TLI delta-v"),
    Column("u_impulse_angle", "dimensionless", "unit-cube coordinate of the impulse angle"),
    Column("phase_deg", "deg", "departure phase theta = 360 u_phase"),
    Column("delta_v_km_s", "km/s", "TLI magnitude = 3.00 + 0.25 u_delta_v"),
    Column("impulse_angle_deg", "deg", "impulse angle = -10 + 20 u_impulse_angle"),
    Column("outcome", None, "M4 outcome: the M3 outcome, numerical_failure if the gate fails"),
    Column("m3_outcome", None, "M3 feasibility outcome before the scaled Jacobi gate"),
    Column("gate_passed", "count", "1 if max |dC| / S <= 1e-11 (protocol 5.2), else 0"),
    Column("perilune_altitude_km", "km", "closest Moon periapsis altitude before the return"),
    Column("lunar_encounter_time_days", "day", "time of that periapsis after TLI"),
    Column("lunar_passage_sense", "dimensionless", "sign of the Moon-relative angular momentum"),
    Column("return_time_days", "day", "inbound return-shell crossing time; null if none"),
    Column("return_speed_km_s", "km/s", "Earth-relative inertial speed at the return shell"),
    Column("return_flight_path_angle_deg", "deg", "Earth-relative flight-path angle at return"),
    Column("vacuum_perigee_altitude_km", "km", "targeting vacuum perigee altitude (M3 h_vp)"),
    Column("perigee_time_days", "day", "time at which h_vp is evaluated; null if undefined"),
    Column("entry_sense", "dimensionless", "sign of Earth-relative angular momentum at return"),
    Column("jacobi_scale", "normalized_cr3bp", "S = max over accepted steps of the C terms"),
    Column("jacobi_drift_rel", "dimensionless", "max |C(s_i) - C(s_0)| / S at accepted steps"),
    Column("nfev", "count", "right-hand-side evaluations of the propagation"),
    Column("objective_deg", "deg", "e = |gamma_S + 6.5 deg| if feasible, else null"),
    Column("cell_d1", "count", "10 x 10 bin of log10 h_p if feasible, else null"),
    Column("cell_d2", "count", "10 x 10 bin of t_ret if feasible, else null"),
    Column("wall_time_hours", "hour", "wall time of the propagation (not reproducible)"),
)

CATALOG_COLUMNS: Final[tuple[Column, ...]] = (
    Column("bins", "count", "grid resolution (10: primary, 20: sensitivity)"),
    Column("checkpoint", "count", "catalog over the first n propagations"),
    Column("cell_d1", "count", "bin of log10 h_p"),
    Column("cell_d2", "count", "bin of t_ret"),
    Column("index", "count", "propagation index of the elite"),
    Column("objective_deg", "deg", "e of the elite"),
    Column("margin_deg", "deg", "next-best feasible e in the cell minus the elite's; null if none"),
)


def _num(value: float) -> float | None:
    return value if math.isfinite(value) else None


def evaluation_row(entry: LogEntry) -> list[Any]:
    e = entry.evaluation
    cell = cell_of(e)
    return [
        entry.index,
        entry.tag,
        *e.u,
        e.phase_deg,
        e.delta_v_km_s,
        e.impulse_angle_deg,
        e.outcome.value,
        e.m3_outcome.value,
        int(e.gate_passed),
        _num(e.perilune_altitude_km),
        _num(e.lunar_encounter_time_days),
        e.lunar_passage_sense,
        _num(e.return_time_days),
        _num(e.return_speed_km_s),
        _num(e.return_flight_path_angle_deg),
        _num(e.vacuum_perigee_altitude_km),
        _num(e.perigee_time_days),
        e.entry_sense,
        _num(e.jacobi_scale),
        _num(e.jacobi_drift_rel),
        e.nfev,
        _num(e.objective_deg),
        None if cell is None else cell[0],
        None if cell is None else cell[1],
        entry.seconds / 3600.0,
    ]


def evaluation_from_row(row: Sequence[Any]) -> LogEntry:
    """Inverse of `evaluation_row` (None becomes NaN)."""
    v = dict(zip((c.name for c in EVALUATION_COLUMNS), row, strict=True))

    def num(name: str) -> float:
        x = v[name]
        return math.nan if x is None else float(x)

    ev = Evaluation(
        u=(float(v["u_phase"]), float(v["u_delta_v"]), float(v["u_impulse_angle"])),
        phase_deg=num("phase_deg"),
        delta_v_km_s=num("delta_v_km_s"),
        impulse_angle_deg=num("impulse_angle_deg"),
        outcome=Outcome(v["outcome"]),
        m3_outcome=Outcome(v["m3_outcome"]),
        gate_passed=bool(v["gate_passed"]),
        perilune_altitude_km=num("perilune_altitude_km"),
        lunar_encounter_time_days=num("lunar_encounter_time_days"),
        lunar_passage_sense=int(v["lunar_passage_sense"]),
        return_time_days=num("return_time_days"),
        return_speed_km_s=num("return_speed_km_s"),
        return_flight_path_angle_deg=num("return_flight_path_angle_deg"),
        vacuum_perigee_altitude_km=num("vacuum_perigee_altitude_km"),
        perigee_time_days=num("perigee_time_days"),
        entry_sense=int(v["entry_sense"]),
        jacobi_scale=num("jacobi_scale"),
        jacobi_drift_rel=num("jacobi_drift_rel"),
        nfev=int(v["nfev"]),
    )
    return LogEntry(int(v["index"]), ev, str(v["tag"]), 3600.0 * float(v["wall_time_hours"]))


def evaluations_table(run_id: str, log: Sequence[LogEntry]) -> bytes:
    return dumps_table(sample_table(run_id, EVALUATION_COLUMNS, [evaluation_row(e) for e in log]))


def catalog_table(run_id: str, log: Sequence[Evaluation], budget: int) -> bytes:
    rows: list[list[Any]] = []
    for n, catalog in checkpoint_catalogs(log, checkpoints_for(budget)).items():
        for el in catalog.values():
            rows.append(
                [10, n, el.cell[0], el.cell[1], el.index, el.objective_deg, _num(el.margin_deg)]
            )
    bins2 = PROTOCOL.descriptors.secondary_bins
    for el in build_catalog(log, bins2).values():
        rows.append(
            [bins2, budget, el.cell[0], el.cell[1], el.index, el.objective_deg, _num(el.margin_deg)]
        )
    return dumps_table(sample_table(run_id, CATALOG_COLUMNS, rows))


def load_log(run_dir: Path, path: str) -> list[LogEntry]:
    """Log entries of an evaluations table (sha256 verified against the manifest)."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (artifact,) = [a for a in manifest["artifacts"] if a["path"] == path]
    data = (run_dir / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
        raise ValueError(f"{run_dir / path}: sha256 mismatch")
    return [evaluation_from_row(r) for r in json.loads(data)["rows"]]


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def solver_block(definition: FreeReturnDefinition = DEFAULT_DEFINITION) -> dict[str, Any]:
    return {
        **solver_json(definition.solver),
        "dense_output": True,
        "max_evaluations": definition.max_evaluations,
        "collision_events": collision_events_json(definition.system),
        "return_shell": {
            "radius_km": definition.return_shell_km,
            "altitude_km": definition.return_shell_altitude_km,
            "function": "distance_minus_radius",
            "direction": "decreasing",
            "terminal": True,
            "grazing_detection": "periapsis_below_radius",
        },
    }


@dataclass(frozen=True, slots=True)
class MethodFiles:
    evaluations: str
    catalog: str


def method_metrics(run: MethodRun, prefix: str) -> dict[str, dict[str, Any]]:
    """Budget counters, cost and numerical-health metrics of one method run."""
    log = run.evaluator.log
    evals = [e.evaluation for e in log]
    prop_s = sum(e.seconds for e in log)
    n = len(log)
    failures, gate = numerical_failures(evals)
    yield_, first = feasibility_yield(evals)
    p = f"{prefix}_" if prefix else ""
    out = {
        f"{p}budget": _metric(run.budget, "count", "propagation budget N of the run"),
        f"{p}propagations": _metric(n, "count", "propagations used (every evaluation counted)"),
        f"{p}cache_hits": _metric(
            run.evaluator.cache_hits, "count", "exactly repeated points served from the cache"
        ),
        f"{p}rhs_evaluations": _metric(
            sum(e.nfev for e in evals), "count", "right-hand-side evaluations (secondary cost)"
        ),
        f"{p}numerical_failures": _metric(failures, "count", "propagations numerical_failure"),
        f"{p}gate_failures": _metric(gate, "count", "numerical failures due to the 5.2 gate alone"),
        f"{p}feasible_count": _metric(
            sum(e.feasible for e in evals), "count", "feasible propagations"
        ),
        f"{p}feasibility_yield": _metric(
            yield_ if math.isfinite(yield_) else 0.0, "dimensionless", "feasible / propagations"
        ),
        f"{p}first_feasible_index": _metric(
            -1 if first is None else first, "count", "index of the first feasible one (-1: none)"
        ),
        f"{p}catalog_cells": _metric(
            len(build_catalog(evals)), "count", "occupied cells of the final 10 x 10 catalog"
        ),
        f"{p}wall_time_hours": _metric(run.wall_seconds / 3600.0, "hour", "method wall time"),
        f"{p}propagation_time_hours": _metric(
            prop_s / 3600.0, "hour", "sum of per-propagation wall times"
        ),
        f"{p}overhead_fraction": _metric(
            (run.wall_seconds - prop_s) / prop_s if prop_s > 0.0 else 0.0,
            "dimensionless",
            "(method wall time - propagation time) / propagation time: method-internal work "
            "per unit of propagation work (meaningful for a serial run only)",
        ),
    }
    for key, value in sorted(run.counters.items()):
        out[f"{p}counter_{key}"] = _metric(value, "count", f"method counter '{key}'")
    return out


def method_block(run: MethodRun, files: MethodFiles) -> dict[str, Any]:
    return {
        "name": run.method,
        "config": run.config,
        "seed": run.seed,
        "budget": run.budget,
        "propagations": run.evaluator.used,
        "status": run.status,
        "evaluations_path": files.evaluations,
        "catalog_path": files.catalog,
    }


def build_manifest(
    run_id: str,
    *,
    stage: str,
    experiment_id: str,
    seed: int | None,
    methods: Sequence[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    limitations: Sequence[str],
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    status = protocol_status(protocol_path)
    extra = list(limitations)
    if status != "frozen":
        extra.append(
            f"Made while the M4 protocol was '{status}' (not frozen): not comparative evidence."
        )
    if git_dirty:
        extra.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": experiment_id,
        "status": "computed_unvalidated",
        "created_at": created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "repository": REPOSITORY,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "environment": environment_info(),
        "model": model_json(definition.system),
        "solver": solver_block(definition),
        "random_seed": seed,
        "artifacts": [],
        "metrics": metrics,
        "limitations": [ASSUMPTIONS, *extra],
        "m4": {
            "protocol_path": PROTOCOL_RELATIVE_PATH,
            "protocol_version": PROTOCOL.version,
            "protocol_status": status,
            "config_sha256": config_sha256(),
            "stage": stage,
            "methods": list(methods),
        },
    }


def write_method_runs(
    out_dir: Path,
    runs: Sequence[MethodRun],
    *,
    stage: str,
    experiment_id: str,
    extra_metrics: dict[str, dict[str, Any]] | None = None,
    limitations: Sequence[str] = (),
    git_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    """Write one run directory holding the given method runs (one file pair per method)."""
    run_id = out_dir.name
    single = len(runs) == 1
    files: list[tuple[str, str, bytes]] = []
    blocks: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    for run in runs:
        suffix = "" if single else f"-{run.method.lower()}"
        names = MethodFiles(f"evaluations{suffix}.json", f"catalog{suffix}.json")
        log = run.evaluator.log
        files.append(("sample_table", names.evaluations, evaluations_table(run_id, log)))
        files.append(
            (
                "sample_table",
                names.catalog,
                catalog_table(run_id, [e.evaluation for e in log], run.budget),
            )
        )
        blocks.append(method_block(run, names))
        metrics.update(method_metrics(run, "" if single else run.method.lower()))
    metrics.update(extra_metrics or {})
    seeds = {r.seed for r in runs}
    if len(seeds) != 1:
        raise ValueError("a run directory holds method runs of one seed")
    manifest = build_manifest(
        run_id,
        stage=stage,
        experiment_id=experiment_id,
        seed=seeds.pop(),
        methods=blocks,
        metrics=metrics,
        limitations=limitations,
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now(UTC),
    )
    return write_run_directory(out_dir, manifest, files)
