"""Model Citizens C3 stages: baselines (with k* re-tuning), planner training, G3 (CLI).

Commands::

    python -m phasespace.citizens.rl.stage baselines --fit-run <dir> --run-id <id>
    python -m phasespace.citizens.rl.stage train --fit-run <dir> --model B1 --seed 1 \\
        --iterations 600 --run-id <id>
    python -m phasespace.citizens.rl.stage g3 --baselines-run <dir> --train-run <dir> ... \\
        --run-id <id>

- **baselines**: for every trainable citizen model (B1, B2, B3, B23; D35), Equal, Mixed,
  Proportional and the published Interpolating k = 22 on the evaluation games, the
  re-tuned Interpolating k* (grid ln k in [-5, 5], step 0.1, scored on separate tuning
  games with common random numbers, then evaluated on the evaluation games), and the
  port checks against the NumPy reference (pmf agreement, closed-loop z).
- **train**: one PPO planner against one citizen model and seed; its learning curve
  (deterministic monitoring games of the training model) and its evaluation on the
  evaluation games (deterministic policy, same citizen draws as the baselines).
- **g3**: planner versus Interpolating k = 22 and k* against B1, paired by evaluation
  game, with a bootstrap over training seeds and evaluation games; the 24 h budget.

Every run is a new immutable directory (`write_run_directory`, never overwritten) with a
manifest (commit, C3 config hash, seeds, artifact SHA-256). Inputs are verified against
their manifests' SHA-256. Status is always `computed_unvalidated` (pre-protocol pilot
runs; the checks are recorded for inspection). Only the frozen C2 fit run is read: no
human data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import equinox as eqx
import jax
import numpy as np
from jax import Array

from phasespace.citizens import config as c2config
from phasespace.citizens.fit import load_fitted, model_block
from phasespace.citizens.rl import checks, ppo, twin
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl.planner import Planner, deserialise, serialise
from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    validate_run_id,
    write_run_directory,
)

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

SCHEMA_VERSION: Final = 1
STUDY_ID: Final = "model-citizens"
GENERATOR: Final = "phasespace.citizens.rl.stage"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"
DEFAULT_RUNS_DIR: Final = Path("runs")
DEFAULT_FIT_RUN: Final = Path("runs/citizens-c2-fit-v1")
GAME_METRICS: Final = ("total_surplus", *c2config.METRICS)
BASELINES_FILE: Final = "baselines.json"
BASELINE_GAMES_FILE: Final = "baseline_games.json"
K_GRID_FILE: Final = "k_grid.json"
PLANNER_FILE: Final = "planner.eqx"
CURVE_FILE: Final = "learning_curve.json"
EVAL_GAMES_FILE: Final = "eval_games.json"
EVAL_SHARES_FILE: Final = "eval_shares.json"
G3_FILE: Final = "g3.json"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _table(
    run_id: str, columns: Sequence[tuple[str, str | None, str]], rows: list[list[Any]]
) -> bytes:
    return dumps_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "columns": [{"name": n, "unit": u, "definition": d} for n, u, d in columns],
            "rows": rows,
        }
    )


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def _manifest_sha256(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()


def _read_verified(run_dir: Path, path: str) -> bytes:
    """An artifact's bytes after checking its SHA-256 against the run's manifest."""
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    expected = next(a["sha256"] for a in manifest["artifacts"] if a["path"] == path)
    payload = (run_dir / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{run_dir / path}: SHA-256 differs from the manifest")
    return payload


def _rows(run_dir: Path, path: str) -> list[list[Any]]:
    rows: list[list[Any]] = json.loads(_read_verified(run_dir, path))["rows"]
    return rows


def _fit_run_record(fit_run: Path) -> dict[str, Any]:
    manifest = json.loads((fit_run / "manifest.json").read_text("utf-8"))
    return {
        "path": str(fit_run.resolve()),
        "run_id": manifest["run_id"],
        "manifest_sha256": _manifest_sha256(fit_run),
        "git_commit": manifest["source"]["git_commit"],
        "config_sha256": manifest["citizens"]["config_sha256"],
    }


def _environment() -> dict[str, str]:
    env = environment_info()
    for package in ("jax", "jaxlib", "equinox", "optax"):
        env[package] = version(package)
    env["jax_backend"] = jax.default_backend()
    devices: list[Any] = jax.devices()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    env["jax_devices"] = ", ".join(str(d) for d in devices)
    return env


def simulator_block() -> dict[str, Any]:
    return {
        "name": "phasespace.citizens.rl.citizens.rollout",
        "pool_rule": "full: R' = max(0.01, min(200, R - sum e + 1.4 sum c))",
        "offer_rule": "deployed for the baselines (D33); R * softmax shares for the planner",
        "rounds": c3config.ROUNDS,
        "precision": "float64 (jax_enable_x64); deployed offers float32 as deployed",
        "citizen_draws": "B1: one member per player at game start; uniforms per round and player",
        "common_random_numbers": "every mechanism faces the same citizen draws for a game key",
    }


def base_manifest(
    run_id: str, stage: str, fit_run: Path | None, seed: int | None
) -> dict[str, Any]:
    commit, dirty = git_source()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": f"citizens-c3-{stage}-v1",
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": {
            "repository": REPOSITORY,
            "git_commit": commit,
            "git_dirty": dirty,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "environment": _environment(),
        "model": model_block(),
        "simulator": simulator_block(),
        "random_seed": seed,
        "artifacts": [],
        "metrics": {},
        "limitations": [
            "Pre-protocol C3 pilot: hyperparameters and budgets are frozen only by C4's protocol.",
            "Citizen models keep their C2 validation labels (D35): only B1 is closed-loop "
            "validated; planners trained on B2, B3 or B23 optimise against unvalidated models.",
            "In-silico only: no planner is evaluated on new human players.",
        ],
        "validation": {
            "checks": [],
            "validated_by": f"{GENERATOR} automated checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
        "citizens": {
            "stage": f"c3_{stage}",
            "pilot_version": c3config.PILOT_VERSION,
            "c3_config_sha256": c3config.config_sha256(),
            "c3_config": c3config.config_dict(),
            "c2_config_sha256": c2config.config_sha256(),
            "fit_run": None if fit_run is None else _fit_run_record(fit_run),
            "held_out_access": "none (only the fitted-model run is read; no human data)",
        },
    }


# --- evaluation ---------------------------------------------------------------------------


def eval_key() -> Array:
    return jax.random.key(c3config.EVAL_KEY_SEED)


def _weight_runner(
    port: rc.JaxCitizen, n_games: int
) -> Callable[[Array, Array, Array], rc.Trajectory]:
    """jit(kind, parameter, key): kind 0 = constant weight w, 1 = Interpolating k."""

    def weight(kind: Array, param: Array) -> Callable[[Array], Array]:
        def w(pool: Array) -> Array:
            return jax.numpy.where(kind == 0, param, twin.interpolating_weight(pool, param))

        return w

    @eqx.filter_jit
    def run(kind: Array, param: Array, key: Array) -> rc.Trajectory:
        mech = rc.weight_rule_step(weight(kind, param))
        return rc.rollout(port, mech, None, key, jax.random.key(0), n_games)[0]

    return run


def baselines_for_model(
    name: str, port: rc.JaxCitizen
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    """(summary rows, per-game rows, k-grid rows) of one citizen model."""
    f64 = twin.f64
    run_eval = _weight_runner(port, c3config.EVAL_GAMES)
    run_tune = _weight_runner(port, c3config.TUNE_GAMES)
    tune_key = jax.random.key(c3config.TUNE_KEY_SEED)
    grid_rows: list[list[Any]] = []
    scores: list[tuple[float, float]] = []
    for ln_k in c3config.ln_k_grid():
        k = math.exp(ln_k)
        traj = run_tune(f64(1), f64(k), tune_key)
        mean = float(checks.trajectory_metrics(traj)["total_surplus"].mean())
        scores.append((ln_k, mean))
        grid_rows.append([name, ln_k, k, mean])
    best = max(s for _, s in scores)
    ln_k_star = min(lk for lk, s in scores if s == best)
    mechanisms: list[tuple[str, int, float, float | None]] = [
        *((m, 0, w, None) for m, w in c3config.FIXED_WEIGHTS.items()),
        ("Interpolating k=22", 1, c3config.K_PUBLISHED, math.log(c3config.K_PUBLISHED)),
        ("Interpolating k*", 1, math.exp(ln_k_star), ln_k_star),
    ]
    summary: list[list[Any]] = []
    games: list[list[Any]] = []
    for mech, kind, param, ln_k in mechanisms:
        m = checks.trajectory_metrics(run_eval(f64(kind), f64(param), eval_key()))
        k_value = param if kind == 1 else None
        summary.append(
            [name, mech, k_value, ln_k, c3config.EVAL_GAMES]
            + [
                x
                for metric in GAME_METRICS
                for x in (float(m[metric].mean()), float(m[metric].std(ddof=1)))
            ]
        )
        for j in range(c3config.EVAL_GAMES):
            games.append([name, mech, j, *(float(m[metric][j]) for metric in GAME_METRICS)])
        _log(f"{name:4s} {mech:20s} total surplus {m['total_surplus'].mean():8.2f}")
    return summary, games, grid_rows


SUMMARY_COLUMNS: Final = [
    ("model", None, "citizen model the mechanism was evaluated against"),
    ("mechanism", None, "baseline (deployed offer rule)"),
    ("k", None, "Interpolating exponent (null for fixed weights)"),
    ("ln_k", None, "natural log of k (k* from the tuning grid)"),
    ("games", "count", "evaluation games (common random numbers across mechanisms)"),
] + [
    col
    for metric in GAME_METRICS
    for col in (
        (f"{metric}_mean", None, f"mean over games of {metric}"),
        (f"{metric}_sd", None, f"SD over games (ddof 1) of {metric}"),
    )
]
GAME_COLUMNS: Final = [
    ("model", None, "citizen model"),
    ("mechanism", None, "mechanism"),
    ("game", "count", "evaluation game index (same citizen draws for every mechanism)"),
    ("total_surplus", "coin", "sum over 40 rounds and 4 players of e - c (the planner objective)"),
    ("surplus", "coin", "notebook surplus: mean over players of mean reward per round"),
    ("gini", "dimensionless", "notebook Gini of the players' mean rewards"),
    ("depletion_round", "count", "first round with R < 1, else 40"),
    ("players_active", "count", "#{i : e_i >= 1} in the last round"),
]
GRID_COLUMNS: Final = [
    ("model", None, "citizen model"),
    ("ln_k", None, "grid point ln k"),
    ("k", None, "exp(ln k)"),
    ("tune_total_surplus_mean", "coin", "mean total surplus over the tuning games"),
]


def port_checks(fit_run: Path, name: str) -> list[Check]:
    reference = load_fitted(fit_run).sim_models()[name]
    port = rc.load_citizens(fit_run)[name]
    diff, mass = checks.pmf_difference(reference, port, checks.random_histories(5), seed=6)
    out = [
        Check(
            f"{name.lower()}_pmf_max_abs_difference",
            diff,
            c3config.CHECK_PMF_ATOL,
            "max |P_jax(c) - P_numpy(c)| over 96 streams x 40 rounds x 201 c of random histories",
        ),
        Check(f"{name.lower()}_pmf_mass_error", mass, 1e-9, "max |sum_c P_jax(c) - 1|"),
    ]
    for i, mech in enumerate(c3config.CHECK_MECHANISMS):
        z = checks.closed_loop_z(
            reference, port, mech, c3config.CHECK_SIM_GAMES, c3config.CHECK_SIM_SEED + i
        )
        out.append(
            Check(
                f"{name.lower()}_closed_loop_{mech.lower()}_max_z",
                max(z.values()),
                c3config.CHECK_Z_MAX,
                f"max over the 4 metrics of |mean_jax - mean_numpy| / SE, {c3config.CHECK_SIM_GAMES} "
                f"games each under {mech}",
            )
        )
    return out


def baselines_command(args: argparse.Namespace) -> int:
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    fit_run: Path = args.fit_run
    models: list[str] = args.models
    manifest = base_manifest(run_id, "baselines", fit_run, c3config.EVAL_KEY_SEED)
    start = time.perf_counter()
    with twin.x64():
        ports = rc.load_citizens(fit_run)
        summary: list[list[Any]] = []
        games: list[list[Any]] = []
        grid: list[list[Any]] = []
        all_checks: list[Check] = []
        for name in models:
            s, g, k = baselines_for_model(name, ports[name])
            summary += s
            games += g
            grid += k
            all_checks += port_checks(fit_run, name)
    manifest["validation"]["checks"] = [c.to_json() for c in all_checks]
    manifest["citizens"]["models"] = models
    manifest["citizens"]["seeds"] = {
        "eval_key": c3config.EVAL_KEY_SEED,
        "tune_key": c3config.TUNE_KEY_SEED,
        "check_simulation": c3config.CHECK_SIM_SEED,
    }
    manifest["citizens"]["k_star"] = {
        row[0]: {"k": row[2], "ln_k": row[3]} for row in summary if row[1] == "Interpolating k*"
    }
    for row in summary:
        key = f"{row[0].lower()}_{row[1].lower().replace(' ', '_').replace('=', '').replace('*', '_star')}"
        manifest["metrics"][f"{key}_total_surplus"] = _metric(
            row[5], "coin", "mean total surplus over the evaluation games"
        )
    manifest["citizens"]["seconds"] = time.perf_counter() - start
    files = [
        ("sample_table", BASELINES_FILE, _table(run_id, SUMMARY_COLUMNS, summary)),
        ("sample_table", BASELINE_GAMES_FILE, _table(run_id, GAME_COLUMNS, games)),
        ("sample_table", K_GRID_FILE, _table(run_id, GRID_COLUMNS, grid)),
    ]
    written = write_run_directory(out_dir, manifest, files)
    failed = [c["name"] for c in written["validation"]["checks"] if not c["passed"]]
    _log(f"{out_dir}: {len(all_checks)} checks, failed {failed or 'none'}")
    return 0 if not failed else 1


# --- training -----------------------------------------------------------------------------


def ppo_config(**overrides: Any) -> ppo.PPOConfig:
    """The C3 PPO configuration (defaults `rl.config.PPO_*`) with explicit overrides."""
    return ppo.PPOConfig(**overrides)


def planner_trajectory(
    planner: Planner, port: rc.JaxCitizen, key: Array, n_games: int
) -> tuple[rc.Trajectory, Array]:
    return ppo.evaluate_planner(planner, port, key, n_games, c3config.ROUNDS)


CURVE_COLUMNS: Final = [
    ("iteration", "count", "PPO iterations completed"),
    ("env_steps", "count", "planner decisions (game rounds) simulated for training"),
    ("seconds", "s", "wall time since training started (includes compilation)"),
    ("sampled_total_surplus", "coin", "mean total surplus of the sampling policy, last chunk"),
    (
        "monitor_total_surplus",
        "coin",
        "deterministic policy on the monitoring games (training model)",
    ),
    ("approx_kl", "dimensionless", "PPO approximate KL, last minibatch of the chunk (mean)"),
    ("clip_fraction", "dimensionless", "fraction of clipped ratios"),
    ("value_loss", "dimensionless", "critic loss (scaled returns)"),
    ("log_std_mean", "dimensionless", "mean log sigma of the logistic-normal head"),
]
SHARES_COLUMNS: Final = [
    ("round", "count", "0-based round"),
    ("pool_mean", "coin", "mean pool before allocation"),
    ("kept_share_mean", "dimensionless", "mean share kept in the pool"),
    ("offer_gini_mean", "dimensionless", "mean notebook Gini of the round's offers"),
    ("excluded_mean", "count", "mean number of players offered < 1"),
]


def train_command(args: argparse.Namespace) -> int:
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    fit_run: Path = args.fit_run
    model: str = args.model
    seed: int = args.seed
    iterations: int = args.iterations
    overrides: dict[str, Any] = {}
    items: list[str] = args.set or []
    for item in items:
        key, _, value = item.partition("=")
        default = getattr(ppo_config(), key)
        overrides[key] = type(default)(value)
    cfg = ppo_config(**overrides)
    manifest = base_manifest(run_id, "train", fit_run, seed)
    with twin.x64():
        port = rc.load_citizens(fit_run)[model]
        monitor_key = jax.random.key(c3config.MONITOR_KEY_SEED)

        @eqx.filter_jit
        def monitor_fn(p: Planner) -> Array:
            traj, _ = planner_trajectory(p, port, monitor_key, c3config.MONITOR_GAMES)
            return checks_total(traj)

        @eqx.filter_jit
        def evaluate(p: Planner) -> tuple[rc.Trajectory, Array]:
            return planner_trajectory(p, port, eval_key(), c3config.EVAL_GAMES)

        start = time.perf_counter()
        planner, curve = ppo.train(
            port,
            cfg,
            seed,
            iterations,
            chunk=c3config.PPO_CHUNK,
            monitor=lambda p: float(monitor_fn(p)),
            log=_log,
        )
        seconds = time.perf_counter() - start
        traj, shares = evaluate(planner)
        m = checks.trajectory_metrics(traj)
        share_rows = _share_rows(traj, np.asarray(shares))
    steps = iterations * cfg.steps_per_iteration
    curve_rows = [
        [
            p.iteration,
            p.env_steps,
            p.seconds,
            p.episode_surplus,
            p.eval_surplus,
            p.approx_kl,
            p.clip_fraction,
            p.value_loss,
            p.log_std_mean,
        ]
        for p in curve
    ]
    game_rows = [
        [model, "Planner", j, *(float(m[metric][j]) for metric in GAME_METRICS)]
        for j in range(c3config.EVAL_GAMES)
    ]
    manifest["citizens"]["training"] = {
        "model": model,
        "seed": seed,
        "iterations": iterations,
        "env_steps": steps,
        "ppo": cfg.to_json(),
        "overrides": overrides,
        "seconds": seconds,
        "seconds_per_million_steps": seconds / steps * 1e6,
    }
    manifest["citizens"]["seeds"] = {
        "train": seed,
        "eval_key": c3config.EVAL_KEY_SEED,
        "monitor_key": c3config.MONITOR_KEY_SEED,
    }
    manifest["metrics"] = {
        "eval_total_surplus_mean": _metric(
            float(m["total_surplus"].mean()), "coin", "deterministic planner, evaluation games"
        ),
        "eval_gini_mean": _metric(
            float(m["gini"].mean()), "dimensionless", "notebook Gini, evaluation games"
        ),
        "training_seconds": _metric(seconds, "s", "wall time of training (including compilation)"),
        "seconds_per_million_steps": _metric(
            seconds / steps * 1e6, "s", "training wall time per 1e6 planner decisions"
        ),
    }
    manifest["validation"]["checks"] = [
        Check(
            "training_hours",
            seconds / 3600.0,
            c3config.G3_HOURS_PER_RUN,
            "wall time of the training run (G3 budget)",
        ).to_json()
    ]
    files = [
        ("model_parameters", PLANNER_FILE, serialise(planner)),
        ("sample_table", CURVE_FILE, _table(run_id, CURVE_COLUMNS, curve_rows)),
        ("sample_table", EVAL_GAMES_FILE, _table(run_id, GAME_COLUMNS, game_rows)),
        ("sample_table", EVAL_SHARES_FILE, _table(run_id, SHARES_COLUMNS, share_rows)),
    ]
    write_run_directory(out_dir, manifest, files)
    _log(f"{out_dir}: eval total surplus {m['total_surplus'].mean():.2f}, {seconds:.0f} s")
    return 0


def checks_total(traj: rc.Trajectory) -> Array:
    return ppo.round_surplus(traj).sum(axis=1).mean()


def _share_rows(traj: rc.Trajectory, shares: FloatArray) -> list[list[Any]]:
    from phasespace.citizens.metrics import gini

    offers = np.asarray(traj.offers)
    pool = np.asarray(traj.pool)
    s = np.swapaxes(shares, 0, 1)  # (G, T, 5)
    rows: list[list[Any]] = []
    for t in range(offers.shape[1]):
        rows.append(
            [
                t,
                float(pool[:, t].mean()),
                float(s[:, t, 4].mean()),
                float(gini(offers[:, t]).mean()),
                float((offers[:, t] < 1.0).sum(axis=1).mean()),
            ]
        )
    return rows


def load_planner(run_dir: Path) -> Planner:
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    hidden = int(manifest["citizens"]["training"]["ppo"]["hidden"])
    return deserialise(_read_verified(run_dir, PLANNER_FILE), hidden)


# --- G3 -------------------------------------------------------------------------------------


def paired_bootstrap(
    planner: FloatArray, baseline: FloatArray, n_resamples: int, seed: int
) -> tuple[float, float, float]:
    """Mean of planner (S seeds x J games) minus baseline (J games), paired by game, and
    its percentile CI: each resample draws S seeds and J games with replacement (the
    same games for every drawn seed and for the baseline)."""
    rng = np.random.default_rng(seed)
    s, j = planner.shape
    diff = planner - baseline[None, :]
    seeds = rng.integers(0, s, size=(n_resamples, s))
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        games = rng.integers(0, j, size=j)
        stats[b] = diff[np.ix_(seeds[b], games)].mean()
    alpha = (1.0 - c3config.CI_LEVEL) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(diff.mean()), float(lo), float(hi)


def g3_command(args: argparse.Namespace) -> int:
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    base_dir: Path = args.baselines_run
    train_dirs: list[Path] = args.train_run
    model = c3config.G3_MODEL
    base_manifest_doc = json.loads((base_dir / "manifest.json").read_text("utf-8"))
    fit_sha = base_manifest_doc["citizens"]["fit_run"]["manifest_sha256"]
    baseline: dict[str, FloatArray] = {}
    for row in _rows(base_dir, BASELINE_GAMES_FILE):
        if row[0] == model:
            baseline.setdefault(row[1], np.zeros(c3config.EVAL_GAMES))[row[2]] = row[3]
    planners: list[FloatArray] = []
    inputs: list[dict[str, Any]] = []
    hours: list[float] = []
    for d in train_dirs:
        tm = json.loads((d / "manifest.json").read_text("utf-8"))
        if tm["citizens"]["training"]["model"] != model:
            raise SystemExit(
                f"{d}: trained against {tm['citizens']['training']['model']}, not {model}"
            )
        if tm["citizens"]["fit_run"]["manifest_sha256"] != fit_sha:
            raise SystemExit(f"{d}: a different fit run than the baselines run")
        if (
            tm["citizens"]["seeds"]["eval_key"]
            != base_manifest_doc["citizens"]["seeds"]["eval_key"]
        ):
            raise SystemExit(f"{d}: different evaluation games")
        values = np.zeros(c3config.EVAL_GAMES)
        for row in _rows(d, EVAL_GAMES_FILE):
            values[row[2]] = row[3]
        planners.append(values)
        hours.append(tm["citizens"]["training"]["seconds"] / 3600.0)
        inputs.append(
            {
                "run_id": tm["run_id"],
                "manifest_sha256": _manifest_sha256(d),
                "seed": tm["citizens"]["training"]["seed"],
                "env_steps": tm["citizens"]["training"]["env_steps"],
                "hours": hours[-1],
            }
        )
    p = np.stack(planners)
    result: dict[str, Any] = {
        "model": model,
        "planner_total_surplus_mean": float(p.mean()),
        "planner_per_seed": [float(x) for x in p.mean(axis=1)],
        "comparisons": {},
    }
    checks_out: list[Check] = []
    for mech in ("Interpolating k=22", "Interpolating k*"):
        mean, lo, hi = paired_bootstrap(
            p, baseline[mech], c3config.BOOTSTRAP_RESAMPLES, c3config.BOOTSTRAP_SEED
        )
        result["comparisons"][mech] = {
            "baseline_total_surplus_mean": float(baseline[mech].mean()),
            "difference_mean": mean,
            "difference_ci": [lo, hi],
            "planner_above": lo > 0.0,
        }
        _log(f"planner - {mech}: {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    g3_pass = (
        bool(result["comparisons"]["Interpolating k=22"]["planner_above"])
        and max(hours) <= c3config.G3_HOURS_PER_RUN
    )
    result["g3_pass"] = g3_pass
    result["g3_rule"] = (
        "planner mean total surplus against B1 above Interpolating k = 22: the 95 % "
        "paired bootstrap CI of the difference (seeds and games resampled) lies above 0, "
        "and every training run took <= 24 h"
    )
    checks_out.append(
        Check(
            "training_hours_max", max(hours), c3config.G3_HOURS_PER_RUN, "longest training run (h)"
        )
    )
    k22 = result["comparisons"]["Interpolating k=22"]
    checks_out.append(
        Check(
            "planner_minus_interpolating_k22_ci_lower_negated",
            -k22["difference_ci"][0],
            0.0,
            "minus the lower 95 % CI bound of planner - Interpolating k=22 (passes when the bound is >= 0)",
        )
    )
    manifest = base_manifest(run_id, "g3", None, c3config.BOOTSTRAP_SEED)
    manifest["citizens"]["inputs"] = {
        "baselines_run": {
            "run_id": base_manifest_doc["run_id"],
            "manifest_sha256": _manifest_sha256(base_dir),
        },
        "train_runs": inputs,
        "fit_run_manifest_sha256": fit_sha,
    }
    manifest["citizens"]["g3"] = result
    manifest["validation"]["checks"] = [c.to_json() for c in checks_out]
    manifest["metrics"] = {
        "planner_total_surplus_mean": _metric(
            float(p.mean()), "coin", "mean over seeds and evaluation games"
        ),
        "planner_minus_k22": _metric(k22["difference_mean"], "coin", "paired mean difference"),
    }
    write_run_directory(out_dir, manifest, [("summary", G3_FILE, dumps_json(result))])
    _log(f"{out_dir}: G3 {'PASS' if g3_pass else 'FAIL'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.rl.stage")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--fit-run", type=Path, default=DEFAULT_FIT_RUN)
    b.add_argument("--run-id", required=True)
    b.add_argument(
        "--models", nargs="+", default=list(rc.TRAINABLE_MODELS), choices=rc.TRAINABLE_MODELS
    )
    t = sub.add_parser("train")
    t.add_argument("--fit-run", type=Path, default=DEFAULT_FIT_RUN)
    t.add_argument("--run-id", required=True)
    t.add_argument("--model", required=True, choices=rc.TRAINABLE_MODELS)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--iterations", type=int, required=True)
    t.add_argument("--set", nargs="*", help="PPO overrides key=value (recorded in the manifest)")
    g = sub.add_parser("g3")
    g.add_argument("--baselines-run", type=Path, required=True)
    g.add_argument("--train-run", type=Path, nargs="+", required=True)
    g.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    commands: Mapping[str, Callable[[argparse.Namespace], int]] = {
        "baselines": baselines_command,
        "train": train_command,
        "g3": g3_command,
    }
    return commands[str(args.command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
