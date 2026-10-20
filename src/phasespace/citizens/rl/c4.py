"""Model Citizens C4 stages (CLI): PPO feasibility pilots.

Commands::

    python -m phasespace.citizens.rl.c4 pilot --model B2 --seed 901 --run-id citizens-c4-pilot-b2-s901

- **pilot** (review item 2): one PPO run with the frozen C3 hyperparameters against a
  non-B1 model. The planner is scored only on its own training model and only on the C3
  monitoring games (never on evaluation games, never on another model): its learning
  curve, plateau, final log sigma, wall time, its final deterministic and sampled
  monitoring return, and the baselines (Equal, Mixed, Proportional, Interpolating k = 22
  and that model's k* from the C3 baselines run) re-scored on the same monitoring games.

Runs are new immutable directories (never overwritten), status `computed_unvalidated`.
Only the frozen C2 fit run and the C3 baselines run are read: no human data.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import equinox as eqx
import jax
import numpy as np
from jax import Array

from phasespace.citizens.rl import c4_config, checks, ppo, stage, twin
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl.planner import Planner, sampling_step, serialise
from phasespace.gravity.export import Check, dumps_json, validate_run_id, write_run_directory

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

GENERATOR: Final = "phasespace.citizens.rl.c4"
DEFAULT_BASELINES_RUN: Final = Path("runs/citizens-c3-baselines-v1")
PILOT_FILE: Final = "pilot.json"


def c4_manifest(
    run_id: str, stage_name: str, fit_run: Path | None, seed: int | None
) -> dict[str, Any]:
    """The C3 manifest layout with the C4 stage, generator and configuration hash."""
    manifest = stage.base_manifest(run_id, stage_name, fit_run, seed)
    manifest["experiment_id"] = f"citizens-c4-{stage_name}-v1"
    manifest["source"]["generator"] = GENERATOR
    manifest["validation"]["validated_by"] = f"{GENERATOR} automated checks"
    manifest["citizens"]["stage"] = f"c4_{stage_name}"
    manifest["citizens"]["c4_version"] = c4_config.C4_VERSION
    manifest["citizens"]["c4_config_sha256"] = c4_config.config_sha256()
    manifest["citizens"]["c4_config"] = c4_config.config_dict()
    manifest["limitations"] = [
        "Pre-protocol C4 preparation: the C4 protocol (seeds, iterations, statistics) is "
        "not frozen yet.",
        *manifest["limitations"][1:],
    ]
    return manifest


# --- learning-curve summaries ----------------------------------------------------------------


def plateau_iteration(iterations: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    """Where a learning curve flattens (definition in `c4_config`).

    F = mean of the last `PLATEAU_FINAL_POINTS` values, I = the first value; the smoothed
    curve is the trailing mean of `PLATEAU_SMOOTH_POINTS` values; the plateau iteration is
    the first iteration from which the smoothed curve stays >= F - tol (F - I) to the end.
    Also the least-squares slope over the last 20 % of the curve (coins per 100 iterations).
    """
    x = np.asarray(iterations, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    n = y.size
    final = float(y[-c4_config.PLATEAU_FINAL_POINTS :].mean())
    first = float(y[0])
    w = c4_config.PLATEAU_SMOOTH_POINTS
    smooth = np.array([y[max(0, j - w + 1) : j + 1].mean() for j in range(n)])
    threshold = final - c4_config.PLATEAU_TOL_FRACTION * (final - first)
    below = np.flatnonzero(smooth < threshold)
    start = 0 if below.size == 0 else int(below[-1]) + 1
    tail = max(2, n // 5)
    slope = float(np.polyfit(x[-tail:], y[-tail:], 1)[0]) * 100.0
    return {
        "plateau_iteration": int(x[min(start, n - 1)]),
        "final_level": final,
        "first_value": first,
        "threshold": threshold,
        "tail_slope_per_100_iterations": slope,
        "tail_points": tail,
    }


def decisions_per_game(traj: rc.Trajectory) -> FloatArray:
    """(G,) number of player-rounds offered at least 1 coin (each is one B4 call)."""
    return (np.asarray(traj.offers) >= 1.0).sum(axis=(1, 2)).astype(np.float64)


def _rule(name: str, k_star: float) -> tuple[int, float]:
    """(kind, parameter) of a baseline for `stage.weight_runner`."""
    if name in c3config.FIXED_WEIGHTS:
        return 0, c3config.FIXED_WEIGHTS[name]
    if name == c4_config.K_PUBLISHED_RULE:
        return 1, c3config.K_PUBLISHED
    if name == c4_config.K_STAR_RULE:
        return 1, k_star
    raise ValueError(f"unknown baseline {name!r}")


def k_star_of(baselines_manifest: Mapping[str, Any], model: str) -> float:
    return float(baselines_manifest["citizens"]["k_star"][model]["k"])


# --- pilot -------------------------------------------------------------------------------------


def pilot_run_id(model: str, seed: int) -> str:
    return f"citizens-c4-pilot-{model.lower()}-s{seed}"


def pilot_command(args: argparse.Namespace) -> int:
    model: str = args.model
    seed: int = args.seed
    run_id: str = args.run_id
    validate_run_id(run_id)
    if run_id != pilot_run_id(model, seed):
        raise SystemExit(f"pilot run ids are {pilot_run_id(model, seed)!r}")
    if c4_config.PILOT_SEEDS.get(model) != seed:
        raise SystemExit(f"the pilot seed of {model} is {c4_config.PILOT_SEEDS.get(model)}")
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    fit_run: Path = args.fit_run
    base_dir: Path = args.baselines_run
    base_doc = json.loads((base_dir / "manifest.json").read_text("utf-8"))
    if base_doc["citizens"]["fit_run"]["manifest_sha256"] != stage.manifest_sha256(fit_run):
        raise SystemExit("the baselines run used a different fit run")
    k_star = k_star_of(base_doc, model)
    iterations: int = args.iterations
    cfg = stage.ppo_config()
    manifest = c4_manifest(run_id, "pilot", fit_run, seed)
    n_mon = c3config.MONITOR_GAMES
    with twin.x64():
        port = rc.load_citizens(fit_run)[model]
        monitor_key = jax.random.key(c3config.MONITOR_KEY_SEED)
        noise_key = jax.random.key(c4_config.PILOT_SAMPLED_NOISE_KEY_SEED)

        @eqx.filter_jit
        def deterministic(p: Planner) -> tuple[rc.Trajectory, Array]:
            return ppo.evaluate_planner(p, port, monitor_key, n_mon, c3config.ROUNDS)

        @eqx.filter_jit
        def sampled(p: Planner) -> rc.Trajectory:
            return rc.rollout(
                port, sampling_step(p), p.initial_state(n_mon), monitor_key, noise_key, n_mon
            )[0]

        def monitor(p: Planner) -> float:
            return float(stage.checks_total(deterministic(p)[0]))

        start = time.perf_counter()
        planner, curve = ppo.train(
            port, cfg, seed, iterations, chunk=c3config.PPO_CHUNK, monitor=monitor, log=stage.log
        )
        seconds = time.perf_counter() - start
        det_traj, _ = deterministic(planner)
        smp_traj = sampled(planner)
        det = checks.trajectory_metrics(det_traj)
        smp = checks.trajectory_metrics(smp_traj)
        runner = stage.weight_runner(port, n_mon)
        rules = [*c4_config.FIXED_RULES, c4_config.K_PUBLISHED_RULE, c4_config.K_STAR_RULE]
        baselines: dict[str, Any] = {}
        for rule in rules:
            kind, param = _rule(rule, k_star)
            traj = runner(twin.f64(kind), twin.f64(param), monitor_key)
            m = checks.trajectory_metrics(traj)
            baselines[rule] = {
                "k": param if kind == 1 else None,
                "total_surplus_mean": float(m["total_surplus"].mean()),
                "total_surplus_sd": float(m["total_surplus"].std(ddof=1)),
                "gini_mean": float(m["gini"].mean()),
                "decisions_per_game_mean": float(decisions_per_game(traj).mean()),
            }
        kept_det = stage.kept_share(det_traj)
        kept_smp = stage.kept_share(smp_traj)
    det_totals = det["total_surplus"]
    paired = {
        rule: float(baselines[rule]["total_surplus_mean"])
        for rule in (c4_config.K_PUBLISHED_RULE, c4_config.K_STAR_RULE)
    }
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
    mon_values = [float(p.eval_surplus) for p in curve if p.eval_surplus is not None]
    plateau = plateau_iteration([p.iteration for p in curve], mon_values)
    sampled_plateau = plateau_iteration(
        [p.iteration for p in curve], [p.episode_surplus for p in curve]
    )
    log_std = [float(x) for x in np.asarray(planner.log_std)]
    steps = iterations * cfg.steps_per_iteration
    result: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "iterations": iterations,
        "env_steps": steps,
        "seconds": seconds,
        "seconds_per_million_steps": seconds / steps * 1e6,
        "games": "C3 monitoring games of the training model (MONITOR_KEY_SEED), never "
        "evaluation games or another model",
        "monitor_games": n_mon,
        "planner": {
            "deterministic_total_surplus_mean": float(det_totals.mean()),
            "deterministic_total_surplus_sd": float(det_totals.std(ddof=1)),
            "sampled_total_surplus_mean": float(smp["total_surplus"].mean()),
            "deterministic_gini_mean": float(det["gini"].mean()),
            "deterministic_depletion_round_mean": float(det["depletion_round"].mean()),
            "decisions_per_game_mean": float(decisions_per_game(det_traj).mean()),
            "final_log_std": log_std,
            "final_log_std_mean": float(np.mean(log_std)),
            "kept_share_per_round_deterministic": kept_det.tolist(),
            "kept_share_per_round_sampled": kept_smp.tolist(),
        },
        "baselines": baselines,
        "planner_minus_baseline": {
            rule: float(det_totals.mean()) - value for rule, value in paired.items()
        },
        "plateau_monitor": plateau,
        "plateau_sampled": sampled_plateau,
    }
    manifest["citizens"]["training"] = {
        "model": model,
        "seed": seed,
        "iterations": iterations,
        "env_steps": steps,
        "ppo": cfg.to_json(),
        "overrides": {},
        "seconds": seconds,
        "seconds_per_million_steps": seconds / steps * 1e6,
    }
    manifest["citizens"]["seeds"] = {
        "train": seed,
        "monitor_key": c3config.MONITOR_KEY_SEED,
        "sampled_noise_key": c4_config.PILOT_SAMPLED_NOISE_KEY_SEED,
    }
    manifest["citizens"]["inputs"] = {
        "baselines_run": {
            "run_id": base_doc["run_id"],
            "manifest_sha256": stage.manifest_sha256(base_dir),
        },
        "k_star": k_star,
    }
    manifest["metrics"] = {
        "monitor_deterministic_total_surplus_mean": stage.metric_entry(
            float(det_totals.mean()), "coin", "final deterministic planner, monitoring games"
        ),
        "monitor_sampled_total_surplus_mean": stage.metric_entry(
            float(smp["total_surplus"].mean()), "coin", "final sampled planner, monitoring games"
        ),
        "monitor_k22_total_surplus_mean": stage.metric_entry(
            paired[c4_config.K_PUBLISHED_RULE], "coin", "Interpolating k = 22, monitoring games"
        ),
        "monitor_k_star_total_surplus_mean": stage.metric_entry(
            paired[c4_config.K_STAR_RULE], "coin", "Interpolating k*, monitoring games"
        ),
        "plateau_iteration": stage.metric_entry(
            plateau["plateau_iteration"], "count", "monitoring-curve plateau (c4_config)"
        ),
        "final_log_std_mean": stage.metric_entry(
            float(np.mean(log_std)), "dimensionless", "mean of the final log sigma"
        ),
        "training_seconds": stage.metric_entry(seconds, "s", "training wall time"),
    }
    manifest["validation"]["checks"] = [
        Check(
            "training_hours",
            seconds / 3600.0,
            c3config.G3_HOURS_PER_RUN,
            "wall time of the training run (C3 budget)",
        ).to_json()
    ]
    files = [
        ("model_parameters", stage.PLANNER_FILE, serialise(planner)),
        (
            "sample_table",
            stage.CURVE_FILE,
            stage.table_bytes(run_id, stage.CURVE_COLUMNS, curve_rows),
        ),
        ("summary", PILOT_FILE, dumps_json(result)),
    ]
    write_run_directory(out_dir, manifest, files)
    stage.log(
        f"{out_dir}: monitoring det {det_totals.mean():.1f}, k22 "
        f"{paired[c4_config.K_PUBLISHED_RULE]:.1f}, k* {paired[c4_config.K_STAR_RULE]:.1f}, "
        f"plateau {plateau['plateau_iteration']}, {seconds:.0f} s"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.rl.c4")
    parser.add_argument("--runs-dir", type=Path, default=stage.DEFAULT_RUNS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--fit-run", type=Path, default=stage.DEFAULT_FIT_RUN)
    p.add_argument("--baselines-run", type=Path, default=DEFAULT_BASELINES_RUN)
    p.add_argument("--model", required=True, choices=sorted(c4_config.PILOT_SEEDS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, default=c4_config.PILOT_ITERATIONS)
    p.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    commands: Mapping[str, Callable[[argparse.Namespace], int]] = {"pilot": pilot_command}
    return commands[str(args.command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
