"""M3 free-return runs: the dense feasibility map and the targeted, validated candidates.

Usage::

    uv run python -m phasespace.gravity.free_return_export map \\
        --run-id gravity-free-return-map-2d-grid-v1 [--design grid2d] [--workers 10]
    uv run python -m phasespace.gravity.free_return_export map --design sobol3d \\
        --seed 20261018 --log2-samples 17 --run-id gravity-free-return-map-3d-sobol17-v1
    uv run python -m phasespace.gravity.free_return_export candidates \\
        --map-run gravity-free-return-map-2d-grid-v1 --run-prefix gravity-free-return \\
        --version v1
    uv run python -m phasespace.gravity.free_return_export web-map \\
        --map-run gravity-free-return-map-2d-grid-v1 \\
        --run-id gravity-free-return-map-2d-grid-v1-web

`map` writes one run with a `sample_table` artifact (`samples.json`, one row per sample).
`candidates` reads a 2-D map run (verifying its sha256), refines the feasible points with
the smallest targeting residual (`free_return_search`), validates every distinct
converged candidate (`free_return_validation`) and writes one trajectory run per
candidate (`<prefix>-c<k>-<version>`, status `validated` only if every check passes) plus
a search run (`<prefix>-search-<version>`) recording every start and its propagation
count. `web-map` reads a 2-D map run (verifying its sha256) and writes a derived run with a
compact, lossless `outcome_grid` artifact for the website (`free_return_grid`). Runs are written atomically and never overwrite an existing directory
(`export.write_run_directory`). The definitions are those of D20
(docs/studies/gravity-atlas/m3-feasibility-design.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.export import (
    REPOSITORY,
    SCHEMA_VERSION,
    STATE_ORDER,
    STUDY_ID,
    UNITS,
    collision_events_json,
    dumps_json,
    environment_info,
    git_source,
    model_json,
    run_status,
    solver_json,
    validate_run_id,
    write_run_directory,
)
from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    RESIDUAL_SCALES_KM,
    TARGET_PERILUNE_ALTITUDE_KM,
    TARGET_VACUUM_PERIGEE_ALTITUDE_KM,
    FreeReturnDefinition,
    FreeReturnRecord,
    Outcome,
    targeting_residual,
)
from phasespace.gravity.free_return_grid import (
    ARTIFACT_KIND as OUTCOME_GRID_KIND,
)
from phasespace.gravity.free_return_grid import (
    ENCODING,
    OUTCOMES,
    decode_outcomes,
    design_from_metrics,
    dumps_grid,
    outcome_grid,
)
from phasespace.gravity.free_return_map import (
    DV_PROBE,
    Column,
    GridDesign,
    SobolDesign,
    dumps_table,
    evaluate_points,
    feasible_intervals,
    outcome_counts,
    record_from_row,
    records_table,
    sample_table,
)
from phasespace.gravity.free_return_search import (
    DEFAULT_SEARCH,
    SearchSettings,
    StartResult,
    distinct_converged,
    refine,
    select_starts,
)
from phasespace.gravity.free_return_validation import (
    BACKWARD_SOLVER,
    EXPORT_GRID_POINTS,
    INDEPENDENT_SOLVER,
    SUPERSEDED_BACKWARD_STATE_THRESHOLD,
    TIGHT_SOLVER,
    BackwardDiagnostic,
    CandidateValidation,
    DeltaVEdge,
    delta_v_edges,
    validate_candidate,
)
from phasespace.gravity.propagation import Termination

GENERATOR: Final = "phasespace.gravity.free_return_export"
DEFAULT_RUNS_DIR: Final = Path("runs")
SAMPLES_FILE: Final = "samples.json"
TRAJECTORY_FILE: Final = "trajectory.json"
MAP_2D_EXPERIMENT: Final = "cr3bp-free-return-map-2d-grid-v1"
PROBE_EXPERIMENT: Final = "cr3bp-free-return-dv-probe-v1"
OUTCOME_GRID_EXPERIMENT: Final = "cr3bp-free-return-map-2d-outcome-grid-v1"
OUTCOME_GRID_FILE: Final = "outcome_grid.json"
DESIGN_PROPOSED_CAP: Final = 200
"""S2 propagation cap per start proposed in the M3 design (superseded default)."""
FRAME: Final = "barycentric_synodic"
ASSUMPTIONS_LIMITATION: Final = (
    "Planar circular restricted three-body model with a single impulsive departure from an "
    "idealized circular parking orbit: no lunar eccentricity, out-of-plane motion, Earth "
    "oblateness, solar or other perturbations, finite burns or ephemeris effects. A "
    "'return' is an inbound crossing of a spherical entry-interface-like shell, not an "
    "atmospheric re-entry model."
)


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def _timestamp(created_at: datetime) -> str:
    return created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source(git_commit: str, git_dirty: bool) -> dict[str, Any]:
    return {
        "repository": REPOSITORY,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "generator": GENERATOR,
        "generator_version": version("phasespace"),
    }


def _solver_block(definition: FreeReturnDefinition) -> dict[str, Any]:
    system = definition.system
    return {
        **solver_json(definition.solver),
        "dense_output": True,
        "max_evaluations": definition.max_evaluations,
        "collision_events": collision_events_json(system),
        "return_shell": {
            "radius_km": definition.return_shell_km,
            "altitude_km": definition.return_shell_altitude_km,
            "function": "distance_minus_radius",
            "direction": "decreasing",
            "terminal": True,
            "grazing_detection": "periapsis_below_radius",
        },
    }


def _definition_metrics(definition: FreeReturnDefinition) -> dict[str, dict[str, Any]]:
    lo, hi = definition.perilune_band_km
    return {
        "parking_altitude_km": _metric(
            definition.parking_altitude_km,
            "km",
            "circular parking-orbit altitude above the mean Earth radius",
        ),
        "return_shell_altitude_km": _metric(
            definition.return_shell_altitude_km,
            "km",
            "return-shell altitude above the mean Earth radius (400 000 ft entry interface)",
        ),
        "perilune_band_min_km": _metric(lo, "km", "lower edge of the feasible perilune band"),
        "perilune_band_max_km": _metric(hi, "km", "upper edge of the feasible perilune band"),
        "t_max_days": _metric(
            definition.t_max_days, "day", "flight-time window and propagation horizon"
        ),
    }


def _limitations(git_dirty: bool, extra: Sequence[str]) -> list[str]:
    out = [ASSUMPTIONS_LIMITATION, *extra]
    if git_dirty:
        out.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    return out


# --- map ---------------------------------------------------------------------------


def build_map_documents(
    run_id: str,
    design: GridDesign | SobolDesign,
    records: Sequence[FreeReturnRecord],
    *,
    definition: FreeReturnDefinition,
    workers: int,
    wall_time_s: float,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
    experiment_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """(manifest without artifacts, samples.json bytes) of a map run.

    `experiment_id` overrides the design's default (e.g. for the fine delta-v probe).
    """
    counts = outcome_counts(records)
    n = len(records)
    feasible = counts[Outcome.FEASIBLE]
    metrics: dict[str, dict[str, Any]] = {
        "sample_count": _metric(n, "count", "departures evaluated (one propagation each)"),
        "feasible_count": _metric(feasible, "count", "samples with outcome 'feasible'"),
        "feasible_fraction": _metric(
            feasible / n if n else 0.0, "dimensionless", "feasible_count / sample_count"
        ),
    }
    for outcome, count in counts.items():
        metrics[f"outcome_{outcome.value}_count"] = _metric(
            count, "count", f"samples with outcome '{outcome.value}'"
        )
    metrics["total_rhs_evaluations"] = _metric(
        sum(r.nfev for r in records), "count", "sum of nfev over all propagations"
    )
    metrics["max_jacobi_drift_steps"] = _metric(
        max((r.jacobi_max_abs_drift_steps for r in records if r.feasible), default=0.0),
        UNITS,
        "max over feasible samples of max |C(s_i) - C(s_0)| at accepted solver steps",
    )
    metrics["wall_time_hours"] = _metric(
        wall_time_s / 3600.0, "hour", f"wall-clock time of the evaluation with {workers} workers"
    )
    metrics["workers"] = _metric(workers, "count", "worker processes (results do not depend on it)")
    metrics.update(_definition_metrics(definition))
    if isinstance(design, GridDesign):
        experiment = MAP_2D_EXPERIMENT if experiment_id is None else experiment_id
        seed: int | None = None
        if experiment_id == PROBE_EXPERIMENT:
            for phase, runs in feasible_intervals(records).items():
                key = f"{phase:g}".replace(".", "p")
                metrics[f"phase_{key}_feasible_intervals"] = _metric(
                    len(runs), "count", f"contiguous feasible delta-v runs at {phase:g} deg"
                )
                metrics[f"phase_{key}_feasible_points"] = _metric(
                    sum(n for _, _, n in runs),
                    "count",
                    f"feasible delta-v samples at {phase:g} deg (times the step: band width)",
                )
        metrics.update(
            {
                "grid_phase_count": _metric(design.n_phase, "count", "phase grid points"),
                "grid_phase_min_deg": _metric(design.phase_min_deg, "deg", "first phase"),
                "grid_phase_step_deg": _metric(design.phase_step_deg, "deg", "phase step"),
                "grid_delta_v_count": _metric(design.n_delta_v, "count", "delta-v grid points"),
                "grid_delta_v_min_km_s": _metric(design.delta_v_min_km_s, "km/s", "first delta-v"),
                "grid_delta_v_step_km_s": _metric(design.delta_v_step_km_s, "km/s", "delta-v step"),
                "impulse_angle_deg": _metric(
                    design.impulse_angle_deg, "deg", "impulse angle of every sample"
                ),
            }
        )
        sampling = (
            "Deterministic grid, phase-major row order (index = i_phase * n_delta_v + "
            "i_delta_v); grid resolution limits which narrow feasible regions are sampled."
        )
    else:
        experiment = "cr3bp-free-return-map-3d-sobol-v1"
        seed = design.seed
        metrics.update(
            {
                "sobol_log2_samples": _metric(
                    design.log2_samples, "count", "log2 of the sample count"
                ),
                "phase_min_deg": _metric(design.phase_bounds_deg[0], "deg", "phase lower bound"),
                "phase_max_deg": _metric(design.phase_bounds_deg[1], "deg", "phase upper bound"),
                "delta_v_min_km_s": _metric(
                    design.delta_v_bounds_km_s[0], "km/s", "delta-v lower bound"
                ),
                "delta_v_max_km_s": _metric(
                    design.delta_v_bounds_km_s[1], "km/s", "delta-v upper bound"
                ),
                "impulse_angle_min_deg": _metric(
                    design.impulse_angle_bounds_deg[0], "deg", "impulse-angle lower bound"
                ),
                "impulse_angle_max_deg": _metric(
                    design.impulse_angle_bounds_deg[1], "deg", "impulse-angle upper bound"
                ),
            }
        )
        sampling = (
            "Scrambled Sobol sample (scipy.stats.qmc.Sobol, d = 3, rng = "
            f"numpy.random.default_rng({design.seed}), random_base2) scaled to the box; "
            "row order is the generator's order."
        )
    timestamp = _timestamp(created_at)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": experiment,
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": _source(git_commit, git_dirty),
        "environment": environment_info(),
        "model": model_json(definition.system),
        "solver": _solver_block(definition),
        "random_seed": seed,
        "artifacts": [],
        "metrics": metrics,
        "limitations": _limitations(
            git_dirty,
            [
                sampling,
                "Each sample is classified from a single DOP853 rtol = atol = 1e-12 "
                "propagation; individual samples are not independently validated.",
            ],
        ),
    }
    return manifest, dumps_table(records_table(run_id, records))


def run_map(
    out_dir: Path,
    design: GridDesign | SobolDesign,
    *,
    workers: int,
    experiment_id: str | None = None,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    git_commit: str,
    git_dirty: bool,
) -> tuple[dict[str, Any], list[FreeReturnRecord]]:
    """Evaluate the design and write the map run to `out_dir` (must not exist)."""
    validate_run_id(out_dir.name)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")
    start = time.perf_counter()
    records = evaluate_points(design.points(), definition, workers=workers)
    wall = time.perf_counter() - start
    manifest, table = build_map_documents(
        out_dir.name,
        design,
        records,
        definition=definition,
        workers=workers,
        wall_time_s=wall,
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now(UTC),
        experiment_id=experiment_id,
    )
    written = write_run_directory(out_dir, manifest, [("sample_table", SAMPLES_FILE, table)])
    return written, records


def load_map_records(run_dir: Path) -> tuple[list[FreeReturnRecord], str]:
    """Records of a map run after verifying the sample table's sha256; and that sha256."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (artifact,) = [a for a in manifest["artifacts"] if a["kind"] == "sample_table"]
    data = (run_dir / artifact["path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError(f"{run_dir / artifact['path']}: sha256 mismatch")
    table = json.loads(data)
    return [record_from_row(row) for row in table["rows"]], digest


def check_map_compatible(run_dir: Path, definition: FreeReturnDefinition) -> None:
    """Raise ValueError unless the run is a 2-D (alpha = 0) grid map made with `definition`.

    Targeting varies (phase, delta-v) at alpha = 0 and reuses each start's map record, so
    the map must use the same model, solver, return shell and predicates.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("experiment_id") != MAP_2D_EXPERIMENT:
        raise ValueError(f"{run_dir}: not a {MAP_2D_EXPERIMENT} run")
    expected = {
        "model": model_json(definition.system),
        "solver": _solver_block(definition),
        "metrics": _definition_metrics(definition),
    }
    if manifest["model"] != expected["model"] or manifest["solver"] != expected["solver"]:
        raise ValueError(f"{run_dir}: model or solver differs from the definition")
    for key, metric in expected["metrics"].items():
        if manifest["metrics"].get(key, {}).get("value") != metric["value"]:
            raise ValueError(f"{run_dir}: {key} differs from the definition")
    if manifest["metrics"].get("impulse_angle_deg", {}).get("value") != 0.0:
        raise ValueError(f"{run_dir}: the map is not at impulse angle 0")


# --- candidates ----------------------------------------------------------------------

SEARCH_COLUMNS: Final[tuple[Column, ...]] = (
    Column("start_rank", "count", "rank of the start by ||F|| among feasible map samples"),
    Column("start_phase_deg", "deg", "phase of the start (a map sample)"),
    Column("start_delta_v_km_s", "km/s", "delta-v of the start"),
    Column("start_residual_norm", "dimensionless", "||F|| at the start"),
    Column("final_phase_deg", "deg", "phase at termination"),
    Column("final_delta_v_km_s", "km/s", "delta-v at termination"),
    Column("final_residual_perilune", "dimensionless", "(h_p - 1000 km) / 1000 km at termination"),
    Column("final_residual_perigee", "dimensionless", "(h_vp - 38 km) / 10 km at termination"),
    Column("final_outcome", None, "feasibility outcome at termination"),
    Column("propagations", "count", "propagations used by this start (Jacobian columns included)"),
    Column("converged", "count", "1 if max |F_i| <= tolerance and feasible, else 0"),
    Column("candidate_run_id", None, "run id of the exported candidate (null: duplicate or none)"),
    Column("message", None, "least_squares termination message"),
)


def _refine_task(
    start: FreeReturnRecord, definition: FreeReturnDefinition, settings: SearchSettings
) -> StartResult:
    return refine(start, definition, settings)


def run_search(
    starts: Sequence[FreeReturnRecord],
    *,
    definition: FreeReturnDefinition,
    settings: SearchSettings,
    workers: int,
) -> list[StartResult]:
    """Refine every start (in parallel; results in input order)."""
    task = partial(_refine_task, definition=definition, settings=settings)
    if workers == 1 or len(starts) <= 1:
        return [task(s) for s in starts]
    with ProcessPoolExecutor(max_workers=min(workers, len(starts))) as pool:
        return list(pool.map(task, starts))


def _norm_or_nan(f: tuple[float, float] | None) -> float:
    return math.nan if f is None else math.hypot(f[0], f[1])


def _backward_metrics(
    prefix: str, diag: BackwardDiagnostic, origin: str
) -> dict[str, dict[str, Any]]:
    note = (
        "diagnostic, not a validation check: backward DOP853 1e-13 integration (no events) "
        f"from {origin} return-shell state to t = 0, error ds0 = s0_back - s0 split into an "
        "along-track time shift tau (ds0 . f / |f|^2, f = ds/dt at s0) and the transverse rest"
    )
    return {
        f"{prefix}_max_abs_initial_state_error": _metric(
            diag.max_abs, UNITS, f"max |ds0| over (x, y, vx, vy); {note}"
        ),
        f"{prefix}_along_track_time_shift_hours": _metric(
            diag.along_track_time_s / 3600.0, "hour", f"tau; {note}"
        ),
        f"{prefix}_transverse_position_error_km": _metric(
            diag.transverse_position_km, "km", f"|position part of ds0 - tau f|; {note}"
        ),
        f"{prefix}_transverse_velocity_error_km_s": _metric(
            diag.transverse_velocity_m_s / 1000.0, "km/s", f"|velocity part of ds0 - tau f|; {note}"
        ),
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _event(name: str, t: float, state: FloatArray) -> dict[str, Any]:
    return {"name": name, "t": float(t), "state": [float(v) for v in state]}


def build_candidate_documents(
    run_id: str,
    validation: CandidateValidation,
    search: StartResult,
    *,
    definition: FreeReturnDefinition,
    map_run_id: str,
    map_sha256: str,
    search_run_id: str,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
    edges: tuple[DeltaVEdge, DeltaVEdge] | None = None,
) -> tuple[dict[str, Any], bytes]:
    """(manifest without artifacts, trajectory.json bytes) of one candidate run.

    `edges` (from `delta_v_edges`) adds the non-gating delta-v margin metrics.
    """
    system = definition.system
    rec = validation.records[0]
    run = validation.run
    convention = system.length_scale_convention
    km = f"km via length_scale_km = {system.length_scale_km} km ({convention})"
    days = f"time_scale_seconds = {system.time_scale_s} s, 1 day = 86400 s"
    # Events only where defined: the perilune when the run has one, the shell crossing
    # only when the run actually ended on the return shell.
    events = [_event("departure", float(run.t[0]), run.states[0])]
    if math.isfinite(validation.perilune_t):
        events.append(_event("perilune", validation.perilune_t, validation.perilune_state))
    if run.termination is Termination.EARTH_RETURN_SHELL:
        events.append(_event("return_shell_crossing", run.t_final, run.final_state))
    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "frame": FRAME,
        "time_unit": UNITS,
        "state_order": list(STATE_ORDER),
        "samples": validation.samples.tolist(),
        "events": events,
    }
    residual = search.residual if search.residual is not None else (math.nan, math.nan)
    metrics: dict[str, dict[str, Any]] = {
        "departure_phase_deg": _metric(
            rec.phase_deg, "deg", "departure phase theta, counterclockwise from +x (Earth->Moon)"
        ),
        "departure_delta_v_km_s": _metric(rec.delta_v_km_s, "km/s", "TLI impulse magnitude"),
        "impulse_angle_deg": _metric(
            rec.impulse_angle_deg, "deg", "impulse angle from local horizontal (tangential: 0)"
        ),
        **_definition_metrics(definition),
        "perilune_altitude_km": _metric(
            rec.perilune_altitude_km,
            "km",
            f"altitude above the {system.moon_radius_km} km mean Moon radius of the closest "
            f"Moon periapsis before the return (event); {km}",
        ),
        "lunar_encounter_time_days": _metric(
            rec.lunar_encounter_time_days, "day", f"time of that periapsis after TLI; {days}"
        ),
        "lunar_passage_sense": _metric(
            rec.lunar_passage_sense,
            "dimensionless",
            "sign of the rotating-frame Moon-relative angular momentum at perilune",
        ),
        "entry_sense": _metric(
            validation.entry_sense,
            "dimensionless",
            "sign of the Earth-relative inertial angular momentum at the return shell "
            "(+1 posigrade, -1 retrograde)",
        ),
        "perilune_margin_to_band_min_km": _metric(
            rec.perilune_altitude_km - definition.perilune_band_km[0],
            "km",
            "non-gating: h_p minus the lower perilune band edge",
        ),
        "perilune_margin_to_band_max_km": _metric(
            definition.perilune_band_km[1] - rec.perilune_altitude_km,
            "km",
            "non-gating: upper perilune band edge minus h_p",
        ),
        "return_shell_crossing_time_days": _metric(
            rec.return_time_days, "day", f"inbound return-shell crossing time after TLI; {days}"
        ),
        "return_shell_speed_km_s": _metric(
            rec.return_speed_km_s,
            "km/s",
            "Earth-relative inertial speed at the return shell; km/s via velocity_scale_km_s = "
            f"{system.velocity_scale_km_s} km/s ({convention})",
        ),
        "return_flight_path_angle_deg": _metric(
            rec.return_flight_path_angle_deg,
            "deg",
            "Earth-relative inertial flight-path angle at the return shell (< 0: descending)",
        ),
        "return_vacuum_perigee_altitude_km": _metric(
            rec.return_vacuum_perigee_altitude_km,
            "km",
            "perigee altitude of the osculating two-body Earth conic (GM_earth only, no "
            f"atmosphere) through the return-shell state; {km}",
        ),
        "jacobi_constant": _metric(rec.jacobi_constant, UNITS, "Jacobi constant after TLI"),
        "earth_periapses_after_encounter": _metric(
            rec.earth_periapses_after_encounter,
            "count",
            "Earth periapses between the perilune and the return",
        ),
        "targeting_residual_perilune": _metric(
            residual[0],
            "dimensionless",
            f"(h_p - {TARGET_PERILUNE_ALTITUDE_KM} km) / "
            f"{RESIDUAL_SCALES_KM[0]} km at the refined point",
        ),
        "targeting_residual_perigee": _metric(
            residual[1],
            "dimensionless",
            f"(h_vp - {TARGET_VACUUM_PERIGEE_ALTITUDE_KM} km) / "
            f"{RESIDUAL_SCALES_KM[1]} km at the refined point",
        ),
        "targeting_propagations": _metric(
            search.propagations,
            "count",
            "propagations of the least-squares start that found this candidate",
        ),
        "sample_count": _metric(
            int(validation.samples.shape[0]),
            "count",
            "rows in trajectory.json: accepted solver steps united with a uniform grid",
        ),
        "rhs_evaluations": _metric(run.nfev, "count", "nfev of the exported propagation"),
        "superseded_check_backward_max_abs_initial_state_error": _metric(
            validation.backward_run.max_abs,
            UNITS,
            "SUPERSEDED criterion, recorded and non-gating: max |s0_back - s0| over (x, y, vx, "
            "vy), normalized, of the backward integration from the exported run's shell "
            f"state; the M3 design's threshold was {SUPERSEDED_BACKWARD_STATE_THRESHOLD:g} "
            "(would "
            + (
                "pass"
                if validation.backward_run.max_abs <= SUPERSEDED_BACKWARD_STATE_THRESHOLD
                else "fail"
            )
            + "). Replaced (user decision 2026-10-18) by the decomposed checks "
            "backward_integration_{along_track_time_shift_s, transverse_position_error_m, "
            "transverse_velocity_error_mm_s}, because the max-norm is dominated by a pure "
            "time shift (|ds/dt| ~ 3.4e3 at the parking orbit, so 1e-7 is a ~0.1 ms test)",
        ),
        **_backward_metrics("backward_run", validation.backward_run, "the exported run's"),
        **_backward_metrics(
            "backward_tight",
            validation.backward_tight,
            "the tight-tolerance (DOP853 1e-13) run's",
        ),
    }
    if edges is not None:
        for side, edge in zip(("below", "above"), edges, strict=True):
            beyond = "none found" if edge.outcome_beyond is None else edge.outcome_beyond.value
            metrics[f"delta_v_margin_{side}_km_s"] = _metric(
                edge.margin_km_s,
                "km/s",
                f"non-gating: delta-v decrease/increase ({side}) at fixed phase and impulse "
                "angle to the first infeasible value (doubling steps from 1 mm/s up to 50 m/s, "
                f"bisected to 0.01 mm/s, {edge.propagations} propagations); outcome beyond "
                f"the edge: {beyond}",
            )
    # Metrics are only included when actually computed (finite).
    metrics = {k: v for k, v in metrics.items() if math.isfinite(v["value"])}
    timestamp = _timestamp(created_at)
    status = run_status(validation.checks)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": "cr3bp-free-return-candidate-v1",
        "status": status,
        "created_at": timestamp,
        "source": _source(git_commit, git_dirty),
        "environment": environment_info(),
        "model": model_json(system),
        "solver": {
            **_solver_block(definition),
            "comparisons": [
                {"role": "tight_tolerance", **solver_json(TIGHT_SOLVER)},
                {"role": "independent_integrator", **solver_json(INDEPENDENT_SOLVER)},
            ],
        },
        "termination": run.termination.value,
        "random_seed": None,
        "artifacts": [],
        "metrics": metrics,
        "limitations": _limitations(
            git_dirty,
            [
                f"Found by least-squares targeting (h_p = {TARGET_PERILUNE_ALTITUDE_KM} km, "
                f"vacuum perigee {TARGET_VACUUM_PERIGEE_ALTITUDE_KM} km) from a sample of map "
                f"run {map_run_id} (samples.json sha256 {map_sha256}); every start is recorded "
                f"in run {search_run_id}.",
                "Samples are the accepted DOP853 steps united with a uniform grid of "
                f"{EXPORT_GRID_POINTS} times over [0, t_ret], evaluated on the dense "
                "output; the last sample is the return-shell crossing.",
                f"The backward-integration check uses {BACKWARD_SOLVER.method} rtol = atol = "
                f"{BACKWARD_SOLVER.rtol:g} without events from the exported return-shell state.",
                "'validated' means the recorded numerical checks passed; it is not a new "
                "scientific result or a mission-grade trajectory.",
            ],
        ),
        "validation": {
            "checks": [c.to_json() for c in validation.checks],
            "validated_by": f"{GENERATOR} automated checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
    }
    return manifest, dumps_json(trajectory)


def run_candidates(
    runs_dir: Path,
    map_run_id: str,
    *,
    run_prefix: str,
    run_version: str,
    workers: int,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    settings: SearchSettings = DEFAULT_SEARCH,
    git_commit: str,
    git_dirty: bool,
) -> tuple[list[StartResult], list[tuple[str, CandidateValidation]]]:
    """S2 + S3: refine, validate and export the candidates of a map run."""
    search_id = f"{run_prefix}-search-{run_version}"
    validate_run_id(search_id)
    if (runs_dir / search_id).exists():
        raise FileExistsError(f"{runs_dir / search_id} already exists; runs are never overwritten")
    check_map_compatible(runs_dir / map_run_id, definition)
    records, map_sha = load_map_records(runs_dir / map_run_id)
    starts = select_starts(records, settings.starts)
    results = run_search(starts, definition=definition, settings=settings, workers=workers)
    distinct = distinct_converged(results, settings)
    candidate_ids = {id(r): f"{run_prefix}-c{k}-{run_version}" for k, r in enumerate(distinct, 1)}
    for run_id in candidate_ids.values():
        validate_run_id(run_id)
        if (runs_dir / run_id).exists():
            raise FileExistsError(f"{runs_dir / run_id} already exists; runs are never overwritten")

    exported: list[tuple[str, CandidateValidation]] = []
    for r in distinct:
        run_id = candidate_ids[id(r)]
        validation = validate_candidate(
            r.final.phase_deg, r.final.delta_v_km_s, r.final.impulse_angle_deg, definition
        )
        edges = delta_v_edges(
            r.final.phase_deg, r.final.delta_v_km_s, r.final.impulse_angle_deg, definition
        )
        manifest, trajectory = build_candidate_documents(
            run_id,
            validation,
            r,
            definition=definition,
            map_run_id=map_run_id,
            map_sha256=map_sha,
            search_run_id=search_id,
            git_commit=git_commit,
            git_dirty=git_dirty,
            created_at=datetime.now(UTC),
            edges=edges,
        )
        write_run_directory(
            runs_dir / run_id, manifest, [("trajectory", TRAJECTORY_FILE, trajectory)]
        )
        exported.append((run_id, validation))

    rows = [
        [
            rank,
            r.start.phase_deg,
            r.start.delta_v_km_s,
            _finite_or_none(_norm_or_nan(targeting_residual(r.start))),
            r.final.phase_deg,
            r.final.delta_v_km_s,
            None if r.residual is None else r.residual[0],
            None if r.residual is None else r.residual[1],
            r.final.outcome.value,
            r.propagations,
            int(r.converged),
            candidate_ids.get(id(r)),
            r.message,
        ]
        for rank, r in enumerate(results)
    ]
    total = sum(r.propagations for r in results)
    timestamp = _timestamp(datetime.now(UTC))
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": search_id,
        "study_id": STUDY_ID,
        "experiment_id": "cr3bp-free-return-targeting-v1",
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": _source(git_commit, git_dirty),
        "environment": environment_info(),
        "model": model_json(definition.system),
        "solver": _solver_block(definition),
        "random_seed": None,
        "artifacts": [],
        "metrics": {
            "starts": _metric(len(results), "count", "least-squares starts"),
            "converged_starts": _metric(
                sum(r.converged for r in results), "count", "starts meeting the tolerance"
            ),
            "distinct_candidates": _metric(len(distinct), "count", "distinct converged points"),
            "total_propagations": _metric(
                total, "count", "propagations over all starts (map propagations excluded)"
            ),
            "max_propagations_per_start": _metric(
                settings.max_propagations_per_start, "count", "declared cap per start"
            ),
            "convergence_abs_residual": _metric(
                settings.convergence_abs_residual, "dimensionless", "max |F_i| tolerance"
            ),
            **_definition_metrics(definition),
        },
        "limitations": _limitations(
            git_dirty,
            [
                f"Starts are the {settings.starts} feasible samples of map run {map_run_id} "
                f"(samples.json sha256 {map_sha}) with the smallest ||F||; scipy.optimize."
                "least_squares (trf, 2-point forward differences, x_scale = 1, box bounds "
                f"phase {list(settings.phase_bounds_deg)} deg, delta-v "
                f"{list(settings.delta_v_bounds_km_s)} km/s, ftol = xtol = gtol = "
                f"{settings.tolerance:g}).",
                f"Propagation cap per start: {settings.max_propagations_per_start}. The M3 "
                f"design (section 5, S2) proposed {DESIGN_PROPOSED_CAP}; the M3 default is "
                f"{DEFAULT_SEARCH.max_propagations_per_start} (user decision 2026-10-18) because "
                f"with {DESIGN_PROPOSED_CAP} no start converged along the narrow h_vp valley."
                + (
                    ""
                    if settings.max_propagations_per_start
                    == DEFAULT_SEARCH.max_propagations_per_start
                    else " This run deviates from the M3 default."
                ),
            ],
        ),
    }
    table = dumps_table(sample_table(search_id, SEARCH_COLUMNS, rows))
    write_run_directory(runs_dir / search_id, manifest, [("sample_table", SAMPLES_FILE, table)])
    return results, exported


# --- web map (derived outcome grid) --------------------------------------------------

_SOURCE_PROCESS_METRICS: Final = frozenset({"wall_time_hours", "workers"})
"""Source metrics that describe the source run's process, not its data; not copied."""


def _load_verified_table(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """(manifest, sample table, sha256) of a map run; raises unless the sha256 matches."""
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text())
    tables = [a for a in manifest["artifacts"] if a["kind"] == "sample_table"]
    if len(tables) != 1:
        raise ValueError(f"{run_dir}: expected exactly one sample_table artifact")
    (artifact,) = tables
    data = (run_dir / artifact["path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError(f"{run_dir / artifact['path']}: sha256 mismatch")
    return manifest, json.loads(data), digest


def build_outcome_grid_documents(
    run_id: str,
    source_manifest: dict[str, Any],
    source_table: dict[str, Any],
    source_sha256: str,
    *,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
) -> tuple[dict[str, Any], bytes]:
    """(manifest without artifacts, outcome_grid.json bytes) of a derived web-map run.

    The grid's outcome counts are recounted from the encoded codes and must equal the
    source run's `outcome_<name>_count` metrics.
    """
    if source_manifest.get("experiment_id") != MAP_2D_EXPERIMENT:
        raise ValueError(f"source is not a {MAP_2D_EXPERIMENT} run")
    (artifact,) = [a for a in source_manifest["artifacts"] if a["kind"] == "sample_table"]
    design = design_from_metrics(source_manifest["metrics"])
    grid = outcome_grid(run_id, design, source_table)
    decoded = decode_outcomes(grid)
    source_metrics: dict[str, Any] = source_manifest["metrics"]
    for outcome in OUTCOMES:
        key = f"outcome_{outcome.value}_count"
        recount = decoded.count(outcome.value)
        if key in source_metrics and source_metrics[key]["value"] != recount:
            raise ValueError(
                f"{key}: source records {source_metrics[key]['value']}, grid {recount}"
            )
    metrics = {k: v for k, v in source_metrics.items() if k not in _SOURCE_PROCESS_METRICS}
    source = source_manifest["source"]
    derivation = (
        f"Lossless re-encoding of {artifact['path']}: the outcome of every one of the "
        f"{design.size} grid cells as uint8 codes ({ENCODING}, phase-major), plus grid "
        "indices, perilune altitude, lunar encounter time, return time and lunar passage "
        "sense of the feasible cells, copied verbatim; no downsampling, no recomputation."
    )
    inherited = [
        s
        for s in source_manifest["limitations"]
        if s != ASSUMPTIONS_LIMITATION and "source.git_dirty" not in s
    ]
    if source["git_dirty"]:
        inherited.append(
            "The source run was generated from a working tree with uncommitted changes "
            "(derived_from.git_dirty = true)."
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": OUTCOME_GRID_EXPERIMENT,
        "status": "computed_unvalidated",
        "created_at": _timestamp(created_at),
        "source": _source(git_commit, git_dirty),
        "environment": environment_info(),
        "model": source_manifest["model"],
        "solver": source_manifest["solver"],
        "random_seed": source_manifest["random_seed"],
        "derived_from": {
            "run_id": source_manifest["run_id"],
            "experiment_id": source_manifest["experiment_id"],
            "artifact_path": artifact["path"],
            "artifact_sha256": source_sha256,
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "derivation": derivation,
        },
        "artifacts": [],
        "metrics": metrics,
        "limitations": _limitations(
            git_dirty,
            [
                *inherited,
                f"Derived from run {source_manifest['run_id']} ({artifact['path']} sha256 "
                f"{source_sha256}, verified before derivation). Model, solver and metrics are "
                "the source run's (process metrics such as wall time excluded); the other "
                "per-sample columns stay in the source table.",
            ],
        ),
    }
    return manifest, dumps_grid(grid)


def run_outcome_grid(
    runs_dir: Path,
    map_run_id: str,
    run_id: str,
    *,
    git_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    """Derive the web outcome-grid run `run_id` from map run `map_run_id` (both in runs_dir)."""
    validate_run_id(run_id)
    out_dir = runs_dir / run_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")
    source_manifest, table, digest = _load_verified_table(runs_dir / map_run_id)
    manifest, data = build_outcome_grid_documents(
        run_id,
        source_manifest,
        table,
        digest,
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now(UTC),
    )
    return write_run_directory(out_dir, manifest, [(OUTCOME_GRID_KIND, OUTCOME_GRID_FILE, data)])


# --- CLI -----------------------------------------------------------------------------


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phasespace.gravity.free_return_export",
        description="M3 free-return feasibility map and candidate runs.",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--workers", type=int, default=_default_workers())
    sub = parser.add_subparsers(dest="command", required=True)
    p_map = sub.add_parser("map", help="evaluate a dense map and write it as a run")
    p_map.add_argument("--run-id", required=True)
    p_map.add_argument(
        "--design",
        choices=("grid2d", "sobol3d", "dvprobe"),
        default="grid2d",
        help="grid2d: S1 grid; sobol3d: S4 Sobol map; dvprobe: fine delta-v probe at "
        "theta = 229 and 251 deg (0.02 m/s)",
    )
    p_map.add_argument("--seed", type=int, help="Sobol scrambling seed (required for sobol3d)")
    p_map.add_argument("--log2-samples", type=int, default=17)
    p_cand = sub.add_parser("candidates", help="refine, validate and export candidates")
    p_cand.add_argument("--map-run", required=True, help="run id of a grid2d map run")
    p_cand.add_argument("--run-prefix", default="gravity-free-return")
    p_cand.add_argument("--version", default="v1")
    p_cand.add_argument(
        "--max-propagations-per-start",
        type=int,
        default=DEFAULT_SEARCH.max_propagations_per_start,
        help="propagation cap per least-squares start (M3 default "
        f"{DEFAULT_SEARCH.max_propagations_per_start}; the design proposed "
        f"{DESIGN_PROPOSED_CAP}); recorded in the search run",
    )
    p_web = sub.add_parser(
        "web-map", help="derive the compact outcome grid of a 2-D map run for the website"
    )
    p_web.add_argument("--map-run", required=True, help="run id of a grid2d map run")
    p_web.add_argument("--run-id", required=True, help="run id of the new derived run")
    args = parser.parse_args(argv)
    runs_dir: Path = args.runs_dir
    workers: int = args.workers
    if workers < 1:
        parser.error("--workers must be >= 1")
    commit, dirty = git_source()

    if args.command == "web-map":
        try:
            written = run_outcome_grid(
                runs_dir,
                str(args.map_run),
                str(args.run_id),
                git_commit=commit,
                git_dirty=dirty,
            )
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        out = runs_dir / str(args.run_id)
        size = sum(f.stat().st_size for f in out.iterdir())
        print(
            f"wrote {out}: {written['artifacts'][0]['path']} from {args.map_run} "
            f"({size / 1024:.1f} KiB in total), git_dirty={dirty}"
        )
        return 0

    if args.command == "map":
        run_id: str = args.run_id
        try:
            validate_run_id(run_id)
        except ValueError as exc:
            parser.error(str(exc))
        if (runs_dir / run_id).exists():
            parser.error(f"{runs_dir / run_id} already exists; runs are never overwritten")
        design: GridDesign | SobolDesign
        experiment_id: str | None = None
        if args.design == "grid2d":
            design = GridDesign()
        elif args.design == "dvprobe":
            design = DV_PROBE
            experiment_id = PROBE_EXPERIMENT
        else:
            if args.seed is None:
                parser.error("--seed is required for --design sobol3d")
            design = SobolDesign(seed=int(args.seed), log2_samples=int(args.log2_samples))
        manifest, records = run_map(
            runs_dir / run_id,
            design,
            workers=workers,
            experiment_id=experiment_id,
            git_commit=commit,
            git_dirty=dirty,
        )
        m = manifest["metrics"]
        print(
            f"wrote {runs_dir / run_id}: {m['sample_count']['value']} samples, "
            f"{m['feasible_count']['value']} feasible, "
            f"{m['wall_time_hours']['value'] * 3600:.1f} s with {workers} workers, "
            f"git_dirty={dirty}"
        )
        for key, metric in m.items():
            if key.startswith("outcome_"):
                print(f"  {key}: {metric['value']}")
        if experiment_id == PROBE_EXPERIMENT:
            step = DV_PROBE.delta_v_step_km_s * 1000.0
            for phase, runs in feasible_intervals(records).items():
                for lo, hi, count in runs:
                    print(
                        f"  theta {phase:g} deg: feasible delta-v {lo:.5f}-{hi:.5f} km/s "
                        f"({count} points, width {(count - 1) * step:.2f}-{(count + 1) * step:.2f}"
                        " m/s)"
                    )
        return 0

    results, exported = run_candidates(
        runs_dir,
        str(args.map_run),
        run_prefix=str(args.run_prefix),
        run_version=str(args.version),
        workers=workers,
        settings=replace(
            DEFAULT_SEARCH, max_propagations_per_start=int(args.max_propagations_per_start)
        ),
        git_commit=commit,
        git_dirty=dirty,
    )
    print(
        f"{len(results)} starts, {sum(r.converged for r in results)} converged, "
        f"{sum(r.propagations for r in results)} propagations; git_dirty={dirty}"
    )
    for r in results:
        print(
            f"  start ({r.start.phase_deg:.4f} deg, {r.start.delta_v_km_s:.4f} km/s) -> "
            f"({r.final.phase_deg:.8f} deg, {r.final.delta_v_km_s:.10f} km/s) "
            f"max|F|={r.residual_max_abs:.2e} props={r.propagations} converged={r.converged}"
        )
    all_valid = True
    for run_id, validation in exported:
        rec = validation.records[0]
        status = run_status(validation.checks)
        all_valid = all_valid and status == "validated"
        print(
            f"wrote {runs_dir / run_id} status={status}: h_p={rec.perilune_altitude_km:.3f} km "
            f"t_ret={rec.return_time_days:.4f} d gamma={rec.return_flight_path_angle_deg:.3f} deg "
            f"dv={rec.delta_v_km_s:.7f} km/s"
        )
        for check in validation.checks:
            mark = "ok  " if check.passed else "FAIL"
            print(f"    {mark} {check.name}: {check.observed:.3e} <= {check.threshold:.1e}")
    return 0 if exported and all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
