"""Command line for the M4 comparison (protocol section 13).

    uv run python -m phasespace.gravity.m4 smoke [--budget 256] [--workers 1]
    uv run python -m phasespace.gravity.m4 search --stage tune|pilot|main --method MS \\
        --config-index 0 --seed 9101 --confirm-frozen
    uv run python -m phasespace.gravity.m4 stage --stage tune|pilot|main \\
        [--configs MS=0,MBH=1,DE=0,T=2,T2=0] --workers 10 --confirm-frozen
    uv run python -m phasespace.gravity.m4 ccstar --confirm-frozen
    uv run python -m phasespace.gravity.m4 groundtruth --confirm-frozen
    uv run python -m phasespace.gravity.m4 robustness --run-id <run> --errors T_E \\
        --confirm-frozen
    uv run python -m phasespace.gravity.m4 calibrate --confirm-frozen
    uv run python -m phasespace.gravity.m4 select --confirm-frozen
    uv run python -m phasespace.gravity.m4 pilot-analysis --confirm-frozen
    uv run python -m phasespace.gravity.m4 analysis --confirm-frozen [--workers 10]
    uv run python -m phasespace.gravity.m4 config

`config` prints the configuration hash and the key constants (no computation). `smoke` is
the only computing command that runs before the freeze: every method once with a tiny
budget (at most 2^8 propagations, seed 9001, outside every protocol seed list), a
robustness cost probe on the M3 candidate c1 (smoke error seed 9002, never T_E or H_E
draws) and per-method timing. Its outputs verify that the code runs and measure cost;
they are not comparative evidence. Every other command refuses to run unless the
protocol's status line says `frozen` and `--confirm-frozen` is given. Every command
writes a new run directory and never overwrites one. `calibrate`, `select`,
`pilot-analysis` and `analysis` (freeze checklist steps 4, 5, 6 and 8; `analysis.py`) read
stored runs only; `analysis` also propagates its 5 % LSODA re-check sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from phasespace.gravity.export import git_source, validate_run_id, write_run_directory
from phasespace.gravity.free_return_export import load_map_records
from phasespace.gravity.free_return_map import dumps_table, sample_table
from phasespace.gravity.m4 import analysis
from phasespace.gravity.m4.catalog import recheck, rechecked_catalog
from phasespace.gravity.m4.config import (
    CC_STAR_SAFETY_CAP,
    CONTINUATION,
    ERROR_DISTRIBUTIONS,
    GATES_PRIMARY,
    GATES_STRESS,
    METHODS,
    ORACLE_SEEDS,
    PROTOCOL,
    PROTOCOL_PATH,
    RUN_BUDGET,
    SMOKE_ERROR_SEED,
    SMOKE_MAX_BUDGET,
    SMOKE_SEED,
    T_E,
    ContinuationConfig,
    CorridorEmitterConfig,
    ErrorDistribution,
    GatesModel,
    MapElitesConfig,
    MethodConfig,
    ProtocolNotFrozenError,
    config_json,
    config_sha256,
    method_grid,
    protocol_status,
    require_frozen,
    to_jsonable,
)
from phasespace.gravity.m4.evaluation import Evaluator, LogEntry, evaluate_unit, to_unit
from phasespace.gravity.m4.groundtruth import compare_with_m3, compute_ground_truth
from phasespace.gravity.m4.methods import MethodRun, run_method, sobol_prefix
from phasespace.gravity.m4.robustness import (
    error_draws,
    execution_models,
    gates_normals,
    solution_robustness,
)
from phasespace.gravity.m4.runs import (
    build_manifest,
    evaluations_table,
    load_log,
    method_run_id,
    robustness_columns,
    robustness_run_id,
    write_method_runs,
)

DEFAULT_RUNS_DIR: Final = Path("runs")
STAGE_SEEDS: Final = {
    "tune": PROTOCOL.seeds.tuning,
    "pilot": PROTOCOL.seeds.pilot,
    "main": PROTOCOL.seeds.main + PROTOCOL.seeds.main_extension,
}
ALL_PROTOCOL_SEEDS: Final = frozenset(
    (
        *PROTOCOL.seeds.tuning,
        *PROTOCOL.seeds.pilot,
        *PROTOCOL.seeds.main,
        *PROTOCOL.seeds.main_extension,
        PROTOCOL.ground_truth.map_seed,
        *PROTOCOL.ground_truth.refinement_seeds,
        *(d.seed for d in PROTOCOL.errors),
        PROTOCOL.gates_normals_seed,
        PROTOCOL.constraints.generation_seed,
        PROTOCOL.constraints.split_seed,
        PROTOCOL.gate.lsoda_seed,
        PROTOCOL.seeds.bootstrap,
    )
)


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def smoke_config(method: str, budget: int) -> MethodConfig:
    """The first grid entry, with the T/T2 initialization and CC seeding batch scaled to a
    smoke budget (n_init = N/4, batches N/8 and N/4) so that every loop is exercised."""
    cfg = method_grid(method)[0]
    if isinstance(cfg, MapElitesConfig):
        return replace(cfg, n_init=budget // 4, init_batch=max(1, budget // 8))
    if isinstance(cfg, CorridorEmitterConfig):
        inner = replace(cfg.map_elites, n_init=budget // 4, init_batch=max(1, budget // 8))
        return replace(cfg, map_elites=inner)
    if isinstance(cfg, ContinuationConfig):
        return replace(cfg, seed_batch=max(1, budget // 4))
    return cfg


def _executor(workers: int) -> ProcessPoolExecutor | None:
    return ProcessPoolExecutor(max_workers=workers) if workers > 1 else None


def run_smoke(
    runs_dir: Path,
    run_id: str,
    *,
    budget: int,
    seed: int,
    workers: int,
    probe_draws: int,
    git_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    if not 1 <= budget <= SMOKE_MAX_BUDGET:
        raise ValueError(f"smoke budget must be in [1, {SMOKE_MAX_BUDGET}]")
    if seed in ALL_PROTOCOL_SEEDS:
        raise ValueError(f"seed {seed} is a protocol seed; smoke runs must not use it")
    out_dir = runs_dir / run_id
    validate_run_id(run_id)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")
    runs: list[MethodRun] = []
    executor = _executor(workers)
    try:
        for method in (*METHODS, "CCstar"):
            cfg = smoke_config("CC" if method == "CCstar" else method, budget)
            ev = Evaluator(budget, executor=executor)
            run = run_method(method, seed, cfg, ev)
            runs.append(run)
            prop = sum(e.seconds for e in ev.log)
            print(
                f"  {method:6s} {ev.used:4d} propagations, {run.wall_seconds:7.2f} s wall, "
                f"{1000 * prop / max(1, ev.used):6.1f} ms/propagation, overhead "
                f"{(run.wall_seconds - prop) / max(prop, 1e-9):6.1%}, status {run.status}"
            )
    finally:
        if executor is not None:
            executor.shutdown()
    extra = smoke_probe(probe_draws)
    limitations = [
        f"Smoke run: budget {budget} propagations per method (protocol N = {RUN_BUDGET}), "
        f"seed {seed} outside every protocol seed list, smoke-scaled T/T2 initialization and "
        "CC seeding batch (recorded in m4.methods[].config). It verifies that the code runs "
        "and measures cost; its catalogs are NOT comparative evidence and must not be "
        "summarized as such.",
        f"CCstar is capped at the smoke budget here (its protocol run is unmatched, cap "
        f"{CC_STAR_SAFETY_CAP}).",
        "Robustness probe: the M3 candidate c1 under smoke error draws (T_E scales, seed "
        f"{SMOKE_ERROR_SEED}, not a protocol error set); cost measurement only.",
    ]
    return write_method_runs(
        out_dir,
        runs,
        stage="smoke",
        experiment_id="cr3bp-free-return-m4-smoke-v1",
        extra_metrics=extra,
        limitations=limitations,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def smoke_probe(draws: int) -> dict[str, dict[str, Any]]:
    """Cost of the 9.1 evaluator and the 5.6 re-check on c1 (propagation equivalents)."""
    _, phase, dv, alpha = ORACLE_SEEDS[0]
    start = time.perf_counter()
    elite = evaluate_unit(to_unit((phase, dv, alpha)))
    t_full = time.perf_counter() - start
    dist = ErrorDistribution(
        "smoke_probe",
        T_E.sigma_magnitude,
        T_E.sigma_direction_deg,
        T_E.sigma_timing_s,
        draws,
        SMOKE_ERROR_SEED,
        held_out=False,
    )
    start = time.perf_counter()
    normals = gates_normals(dist)
    rob = solution_robustness(
        elite, error_draws(dist), executions=(GATES_PRIMARY,), normals=normals
    )
    t_rob = time.perf_counter() - start
    # The H_E-nominal shape: the same draws executed under both Gates sets (9.1 pairing).
    both = solution_robustness(
        elite, error_draws(dist), executions=(GATES_PRIMARY, GATES_STRESS), normals=normals
    )
    start = time.perf_counter()
    rc = recheck(elite)
    t_rc = time.perf_counter() - start
    pe = float(elite.nfev)
    per_draw = sum(d.nfev for d in rob.draws) / pe / max(1, draws)
    per_draw_both = sum(d.nfev for d in both.draws) / pe / max(1, draws)

    def metric(value: float, unit: str, definition: str) -> dict[str, Any]:
        return {"value": value, "unit": unit, "definition": definition}

    print(
        f"  probe: full propagation {1000 * t_full:.1f} ms; robustness setup "
        f"{rob.setup_nfev / pe:.2f} PE, {per_draw:.2f} PE per draw ({per_draw_both:.2f} with "
        f"the stress set), {t_rob:.2f} s for {draws} draws; re-check {t_rc * 1000:.0f} ms "
        f"(passed={rc.passed})"
    )
    survival = (
        {
            "probe_robustness_survival_b30": metric(
                rob.survival(30.0), "dimensionless", "c1 survival at B = 30 m/s (probe only)"
            )
        }
        if rob.draws
        else {}
    )
    return {
        **survival,
        "probe_full_propagation_nfev": metric(pe, "count", "nfev of c1's full propagation (1 PE)"),
        "probe_full_propagation_hours": metric(t_full / 3600.0, "hour", "its wall time"),
        "probe_robustness_setup_pe": metric(
            rob.setup_nfev / pe, "dimensionless", "Jacobian setup nfev / c1 nfev"
        ),
        "probe_robustness_pe_per_draw": metric(
            per_draw, "dimensionless", "mean draw nfev / c1 nfev (primary Gates set only)"
        ),
        "probe_robustness_pe_per_draw_with_stress": metric(
            per_draw_both,
            "dimensionless",
            "mean draw nfev / c1 nfev, primary and stress Gates sets (H_E-nominal shape)",
        ),
        "probe_robustness_draws": metric(draws, "count", "smoke error draws (seed 9002)"),
        "probe_robustness_hours": metric(t_rob / 3600.0, "hour", "setup + draws wall time"),
        "probe_recheck_hours": metric(t_rc / 3600.0, "hour", "DOP853 1e-13 re-check wall time"),
        "probe_recheck_passed": metric(int(rc.passed), "count", "1 if c1 passes the 5.6 re-check"),
        **gates_metrics((GATES_PRIMARY, GATES_STRESS)),
    }


# --- frozen-protocol stages ----------------------------------------------------------------


def _search_task(
    runs_dir: Path,
    stage: str,
    method: str,
    config_index: int,
    seed: int,
    git_commit: str,
    git_dirty: bool,
) -> str:
    """One budget-matched method run (serial inside; runs are parallelized across tasks)."""
    run_id = method_run_id(stage, method, config_index, seed)
    out_dir = runs_dir / run_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")
    cfg = method_grid(method)[config_index]
    run = run_method(method, seed, cfg, Evaluator(RUN_BUDGET))
    write_method_runs(
        out_dir,
        [run],
        stage=stage,
        experiment_id=f"cr3bp-free-return-m4-{stage}-v1",
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    return run_id


def stage_tasks(
    stage: str, configs: dict[str, int], seeds: Sequence[int]
) -> list[tuple[str, int, int]]:
    """(method, config index, seed) of a stage: every grid entry for 'tune' (including the
    single, untuned D and CC entries: B is calibrated on the D tuning catalogs and the
    pilot's dispersion uses the tuning seeds of every method), else the selected entry of
    every method (all methods must be given except D and CC)."""
    tasks: list[tuple[str, int, int]] = []
    for method in METHODS:
        grid = method_grid(method)
        if stage == "tune":
            indices = list(range(len(grid)))
        elif len(grid) == 1:
            indices = [0]
        else:
            if method not in configs:
                raise ValueError(f"--configs must select a configuration for {method}")
            indices = [configs[method]]
        tasks.extend((method, k, s) for k in indices for s in seeds)
    return tasks


def run_stage(
    runs_dir: Path,
    stage: str,
    tasks: Sequence[tuple[str, int, int]],
    workers: int,
    git_commit: str,
    git_dirty: bool,
) -> list[str]:
    if workers <= 1:
        return [_search_task(runs_dir, stage, m, k, s, git_commit, git_dirty) for m, k, s in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_search_task, runs_dir, stage, m, k, s, git_commit, git_dirty)
            for m, k, s in tasks
        ]
        return [f.result() for f in futures]


def run_ccstar(runs_dir: Path, git_commit: str, git_dirty: bool, workers: int) -> str:
    run_id = "gravity-m4-ccstar-v1"
    executor: Executor | None = _executor(workers)
    try:
        run = run_method(
            "CCstar", 0, CONTINUATION, Evaluator(CC_STAR_SAFETY_CAP, executor=executor)
        )
    finally:
        if executor is not None:
            executor.shutdown()
    write_method_runs(
        runs_dir / run_id,
        [run],
        stage="reference",
        experiment_id="cr3bp-free-return-m4-ccstar-v1",
        limitations=["CC* is oracle-seeded with M3 c1 and c2 and not budget-matched."],
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    return run_id


def run_groundtruth(runs_dir: Path, workers: int, git_commit: str, git_dirty: bool) -> str:
    """Sources (a)-(c) of 5.3 plus the K1 check against the M3 S4 map."""
    gt = PROTOCOL.ground_truth
    run_id = "gravity-m4-groundtruth-v1"
    if (runs_dir / run_id).exists():
        raise FileExistsError(f"{runs_dir / run_id} already exists; runs are never overwritten")
    m3_records, _ = load_map_records(runs_dir / gt.m3_prefix_run_id)
    executor = _executor(workers)
    ev = Evaluator(10**8, executor=executor, chunksize=16)
    start = time.perf_counter()
    try:
        cells, map_evals, _ = compute_ground_truth(
            ev.evaluate_batch, sobol_prefix(gt.map_seed, 2**gt.log2_map)
        )
    finally:
        if executor is not None:
            executor.shutdown()
    mismatches = compare_with_m3(map_evals, m3_records)
    wall = time.perf_counter() - start
    feasible: list[LogEntry] = [e for e in ev.log if e.evaluation.feasible]

    def metric(value: float, unit: str, definition: str) -> dict[str, Any]:
        return {"value": value, "unit": unit, "definition": definition}

    metrics = {
        "propagations": metric(ev.used, "count", "GT propagations (map and levels 1-3)"),
        "k1_mismatches": metric(len(mismatches), "count", "K1: fields differing from M3 S4"),
        "wall_time_hours": metric(wall / 3600.0, "hour", f"wall time with {workers} workers"),
    }
    for source, found in cells.by_source.items():
        metrics[f"cells_{source}"] = metric(len(found), "count", f"10 x 10 cells from {source}")
    metrics["cells_union"] = metric(len(cells.union()), "count", "|G| from sources (a)-(c)")
    manifest = build_manifest(
        run_id,
        stage="ground_truth",
        experiment_id="cr3bp-free-return-m4-groundtruth-v1",
        seed=gt.map_seed,
        methods=[],
        metrics=metrics,
        limitations=[
            "evaluations.json holds the feasible propagations only (every source); the "
            "infeasible ones are counted, not stored.",
        ],
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now(UTC),
    )
    write_run_directory(
        runs_dir / run_id,
        manifest,
        [("sample_table", "evaluations.json", evaluations_table(run_id, feasible))],
    )
    if mismatches:
        raise SystemExit(f"K1 failed: {len(mismatches)} mismatches with {gt.m3_prefix_run_id}")
    return run_id


def gates_metrics(models: Sequence[GatesModel]) -> dict[str, dict[str, Any]]:
    """The MCC execution-error parameters (D24) of each Gates set as manifest metrics."""
    out: dict[str, dict[str, Any]] = {}
    for gates in models:
        prefix = f"mcc_{gates.name}"
        out |= {
            f"{prefix}_sigma_s": {
                "value": gates.sigma_s,
                "unit": "dimensionless",
                "definition": f"{gates.name}: proportional magnitude error of the MCC (Gates)",
            },
            f"{prefix}_sigma_r_km_s": {
                "value": gates.sigma_r_m_s / 1000.0,
                "unit": "km/s",
                "definition": f"{gates.name}: fixed magnitude error along the MCC",
            },
            f"{prefix}_sigma_p_deg": {
                "value": math.degrees(gates.sigma_p_rad),
                "unit": "deg",
                "definition": f"{gates.name}: proportional pointing error (rad in config)",
            },
            f"{prefix}_sigma_a_km_s": {
                "value": gates.sigma_a_m_s / 1000.0,
                "unit": "km/s",
                "definition": f"{gates.name}: fixed in-plane transverse error",
            },
        }
    out["mcc_gates_normals_seed"] = {
        "value": PROTOCOL.gates_normals_seed,
        "unit": "count",
        "definition": "first element of the execution-error normal-stream seed (all sets)",
    }
    return out


def run_robustness(
    runs_dir: Path,
    source_run: str,
    errors: str,
    git_commit: str,
    git_dirty: bool,
    *,
    confirm_frozen: bool,
) -> str:
    """5.6 re-check of a method run's final catalog, then 9.1 for every elite under the
    injection set `errors` and every Gates set paired with it (D24, D25)."""
    dist = ERROR_DISTRIBUTIONS[errors]
    models = execution_models(dist, confirm_frozen=confirm_frozen)
    draws = error_draws(dist, confirm_frozen=confirm_frozen)
    normals = gates_normals(dist, confirm_frozen=confirm_frozen)
    run_id = robustness_run_id(source_run, errors)
    if (runs_dir / run_id).exists():
        raise FileExistsError(f"{runs_dir / run_id} already exists; runs are never overwritten")
    log = [e.evaluation for e in load_log(runs_dir / source_run, "evaluations.json")]
    catalog, checks = rechecked_catalog(log, lambda i: recheck(log[i], index=i))

    def flag(value: bool | None) -> int | None:
        return None if value is None else int(value)

    rows: list[list[Any]] = []
    for el in catalog.values():
        rob = solution_robustness(log[el.index], draws, executions=models, normals=normals)
        for k, d in enumerate(rob.draws):
            rows.append(
                [
                    el.cell[0],
                    el.cell[1],
                    el.index,
                    k,
                    int(rob.singular),
                    int(d.uncorrected_feasible),
                    d.commanded_m_s_norm / 1000.0,
                    *(flag(x.feasible) for x in d.executions),
                    int(d.converged),
                    int(d.diverged),
                    flag(d.commanded_feasible),
                    d.iterations,
                    d.nfev,
                ]
            )
    columns = robustness_columns([m.name for m in models])
    names = ", ".join(m.name for m in models)
    manifest = build_manifest(
        run_id,
        stage="robustness",
        experiment_id="cr3bp-free-return-m4-robustness-v1",
        seed=dist.seed,
        methods=[],
        metrics={
            "elites": {"value": len(catalog), "unit": "count", "definition": "after re-check"},
            "rechecks": {"value": len(checks), "unit": "count", "definition": "re-checks made"},
            "recheck_failures": {
                "value": sum(not c.passed for c in checks),
                "unit": "count",
                "definition": "elites rejected by the 5.6 re-check",
            },
            **gates_metrics(models),
        },
        limitations=[
            f"Injection errors {dist.name} (K = {dist.draws}, seed {dist.seed}); MCC execution "
            f"errors {names} (Gates model, D24), all with the standard normals of "
            f"default_rng(({PROTOCOL.gates_normals_seed}, {dist.seed})). Survival at B (D25): "
            "uncorrected_feasible, or commanded_delta_v <= B and executed_feasible."
        ],
        git_commit=git_commit,
        git_dirty=git_dirty,
        created_at=datetime.now(UTC),
    )
    table = dumps_table(sample_table(run_id, columns, rows))
    write_run_directory(runs_dir / run_id, manifest, [("sample_table", "robustness.json", table)])
    return run_id


def print_config(*, full: bool) -> None:
    """The configuration hash and the key constants (or the full canonical JSON)."""
    if full:
        print(json.dumps(config_json(), indent=2, sort_keys=True))
        return
    p = PROTOCOL
    corr = p.correction
    lines = [
        f"protocol        {PROTOCOL_PATH.relative_to(PROTOCOL_PATH.parents[3])} "
        f"{p.version} (status: {protocol_status()})",
        f"config_sha256   {config_sha256()}",
        f"run budget N    {p.run_budget}; checkpoints {list(p.checkpoints)}",
        f"seeds           tuning {list(p.seeds.tuning)}, pilot {list(p.seeds.pilot)}, "
        f"main {p.seeds.main[0]}-{p.seeds.main[-1]} "
        f"(+{p.seeds.main_extension[0]}-{p.seeds.main_extension[-1]})",
        f"ground truth    2^{p.ground_truth.log2_map} Sobol, seed {p.ground_truth.map_seed}; "
        f"refinement seeds {list(p.ground_truth.refinement_seeds)}",
    ]
    for d in p.errors:
        lines.append(
            f"errors {d.name:12s} sigma = ({d.sigma_magnitude:g}, {d.sigma_direction_deg:g} deg, "
            f"{d.sigma_timing_s:g} s), K = {d.draws}, seed {d.seed}, "
            f"executions {list(d.executions)}"
        )
    for g in p.correction_execution:
        lines.append(
            f"gates {g.name:13s} sigma_s {g.sigma_s:g}, sigma_r {g.sigma_r_m_s:g} m/s, "
            f"sigma_p {g.sigma_p_rad:g} rad, sigma_a {g.sigma_a_m_s:g} m/s"
        )
    lines += [
        f"gates normals   default_rng(({p.gates_normals_seed}, injection seed))",
        f"correction      t_c {corr.t_c_days:g} d, FD {corr.fd_step_m_s:g} m/s, cond <= "
        f"{corr.cond_max:g}, <= {corr.max_iterations} chord iterations, tol "
        f"{corr.tolerance_km:g} km, divergence {corr.divergence_m_s:g} m/s",
        f"budget B grid   {list(corr.budget_grid_m_s)} m/s, calibration band "
        f"{list(corr.calibration_band)}",
    ]
    for method in METHODS:
        for k, cfg in enumerate(method_grid(method)):
            text = "-" if cfg is None else json.dumps(to_jsonable(cfg), sort_keys=True)
            lines.append(f"grid {method}[{k}]{'':{8 - len(method)}s} {text}")
    print("\n".join(lines))


def _analysis_command(
    args: argparse.Namespace, runs_dir: Path, commit: str, dirty: bool, protocol_path: Path
) -> dict[str, Any]:
    """Steps 4, 5, 6 and 8 of the freeze checklist (they read stored runs only)."""
    common: dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": dirty,
        "run_id": str(args.run_id),
        "protocol_path": protocol_path,
    }
    if args.command == "calibrate":
        manifest = analysis.run_calibration(runs_dir, **common)
        print(f"B = {manifest['m4']['calibration']['budget_m_s']:g} m/s")
    elif args.command == "select":
        manifest = analysis.run_selection(
            runs_dir,
            calibration_run=str(args.calibration_run),
            groundtruth_run=str(args.groundtruth_run),
            **common,
        )
        print(f"selected: --configs {manifest['m4']['selection']['configs_argument']}")
    elif args.command == "pilot-analysis":
        manifest = analysis.run_pilot_analysis(
            runs_dir,
            calibration_run=str(args.calibration_run),
            selection_run=str(args.selection_run),
            **common,
        )
        pilot = manifest["m4"]["pilot"]
        print(
            f"n = {pilot['seed_count']} (unclamped {pilot['seed_count_unclamped']}), main seeds "
            f"{pilot['main_seeds']}; K3 {'TRIGGERED' if pilot['k3_triggered'] else 'not triggered'}"
            f"; projection {pilot['projection_total_pe']:.0f} PE, reductions "
            f"{pilot['reductions'] or 'none'}; K4 "
            f"{'TRIGGERED' if pilot['k4_triggered'] else 'not triggered'}"
        )
    else:
        manifest = analysis.run_analysis(
            runs_dir,
            confirm_frozen=bool(args.confirm_frozen),
            calibration_run=str(args.calibration_run),
            selection_run=str(args.selection_run),
            pilot_run=str(args.pilot_run),
            groundtruth_run=str(args.groundtruth_run),
            ccstar_run=str(args.ccstar_run),
            workers=int(args.workers),
            **common,
        )
        block = manifest["m4"]["analysis"]
        print(f"H1 per baseline: {block['h1']}; criteria: {block['criteria']}")
    return manifest


def _parse_configs(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, index = part.partition("=")
        out[name.strip()] = int(index)
    return out


def main(argv: Sequence[str] | None = None, *, protocol_path: Path = PROTOCOL_PATH) -> int:
    """Run the command line. `protocol_path` is the protocol whose status the freeze guard
    reads (tests substitute a temporary file; there is deliberately no flag for it)."""
    parser = argparse.ArgumentParser(
        prog="python -m phasespace.gravity.m4", description="Gravity Atlas M4 comparison runs."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    p_smoke = sub.add_parser("smoke", help="tiny-budget run of every method (pre-freeze)")
    p_smoke.add_argument("--run-id", default="gravity-m4-smoke-v1")
    p_smoke.add_argument("--budget", type=int, default=SMOKE_MAX_BUDGET)
    p_smoke.add_argument("--seed", type=int, default=SMOKE_SEED)
    p_smoke.add_argument("--workers", type=int, default=1)
    p_smoke.add_argument("--probe-draws", type=int, default=4)
    p_config = sub.add_parser("config", help="print the config hash and key constants")
    p_config.add_argument("--json", action="store_true", help="print the full config JSON")
    guarded = {
        "search": "one budget-matched method run",
        "stage": "every run of a stage (tune, pilot or main)",
        "ccstar": "the oracle-seeded continuation reference CC*",
        "groundtruth": "the coverage ground truth (5.3 a-c) and the K1 check",
        "robustness": "re-check and execution-error tolerance of a run's catalog",
        "calibrate": "step 4: the correction budget B from the D tuning catalogs (9.1)",
        "select": "step 5: configuration selection on the tuning runs (6)",
        "pilot-analysis": "step 6: seed count n, K3 and the 7.4 re-projection (8)",
        "analysis": "step 8: ground-truth union, metrics, statistics, LSODA sample (10-12)",
    }
    subs: dict[str, argparse.ArgumentParser] = {}
    for name, text in guarded.items():
        p = sub.add_parser(name, help=f"{text} (frozen protocol only)")
        p.add_argument("--confirm-frozen", action="store_true")
        p.add_argument("--workers", type=int, default=_default_workers())
        subs[name] = p
    subs["search"].add_argument("--stage", choices=sorted(STAGE_SEEDS), required=True)
    subs["search"].add_argument("--method", choices=METHODS, required=True)
    subs["search"].add_argument("--config-index", type=int, default=0)
    subs["search"].add_argument("--seed", type=int, required=True)
    subs["stage"].add_argument("--stage", choices=sorted(STAGE_SEEDS), required=True)
    subs["stage"].add_argument("--configs", default="", help="e.g. MS=0,MBH=1,DE=0,T=2,T2=0")
    subs["robustness"].add_argument("--run-id", required=True)
    subs["robustness"].add_argument("--errors", choices=sorted(ERROR_DISTRIBUTIONS), required=True)
    for name, default in (
        ("calibrate", analysis.CALIBRATION_RUN_ID),
        ("select", analysis.SELECTION_RUN_ID),
        ("pilot-analysis", analysis.PILOT_ANALYSIS_RUN_ID),
        ("analysis", analysis.ANALYSIS_RUN_ID),
    ):
        subs[name].add_argument("--run-id", default=default, help="the new run's id")
    for name in ("select", "pilot-analysis", "analysis"):
        subs[name].add_argument("--calibration-run", default=analysis.CALIBRATION_RUN_ID)
    for name in ("pilot-analysis", "analysis"):
        subs[name].add_argument("--selection-run", default=analysis.SELECTION_RUN_ID)
    for name in ("select", "analysis"):
        subs[name].add_argument("--groundtruth-run", default=analysis.GROUNDTRUTH_RUN_ID)
    subs["analysis"].add_argument("--pilot-run", default=analysis.PILOT_ANALYSIS_RUN_ID)
    subs["analysis"].add_argument("--ccstar-run", default=analysis.CCSTAR_RUN_ID)
    args = parser.parse_args(argv)
    runs_dir: Path = args.runs_dir
    if args.command == "config":
        print_config(full=bool(args.json))
        return 0
    commit, dirty = git_source()

    if args.command == "smoke":
        try:
            manifest = run_smoke(
                runs_dir,
                str(args.run_id),
                budget=int(args.budget),
                seed=int(args.seed),
                workers=int(args.workers),
                probe_draws=int(args.probe_draws),
                git_commit=commit,
                git_dirty=dirty,
            )
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        print(f"wrote {runs_dir / manifest['run_id']} (git_dirty={dirty}); not comparative")
        return 0

    try:
        require_frozen(bool(args.confirm_frozen), protocol_path)
    except ProtocolNotFrozenError as exc:
        parser.error(str(exc))
    workers = int(args.workers)
    if args.command == "search":
        stage, seed = str(args.stage), int(args.seed)
        if seed not in STAGE_SEEDS[stage]:
            parser.error(f"seed {seed} is not a {stage} seed {STAGE_SEEDS[stage]}")
        print(
            _search_task(
                runs_dir, stage, str(args.method), int(args.config_index), seed, commit, dirty
            )
        )
    elif args.command == "stage":
        stage = str(args.stage)
        seeds = PROTOCOL.seeds.main if stage == "main" else STAGE_SEEDS[stage]
        tasks = stage_tasks(stage, _parse_configs(str(args.configs)), seeds)
        for run_id in run_stage(runs_dir, stage, tasks, workers, commit, dirty):
            print(run_id)
    elif args.command == "ccstar":
        print(run_ccstar(runs_dir, commit, dirty, workers))
    elif args.command == "groundtruth":
        print(run_groundtruth(runs_dir, workers, commit, dirty))
    elif args.command in ("calibrate", "select", "pilot-analysis", "analysis"):
        try:
            manifest = _analysis_command(args, runs_dir, commit, dirty, protocol_path)
        except (analysis.EvidenceError, FileNotFoundError, FileExistsError) as exc:
            parser.error(str(exc))
        print(f"wrote {runs_dir / manifest['run_id']} (git_dirty={dirty})")
    else:
        print(
            run_robustness(
                runs_dir,
                str(args.run_id),
                str(args.errors),
                commit,
                dirty,
                confirm_frozen=bool(args.confirm_frozen),
            )
        )
    return 0
