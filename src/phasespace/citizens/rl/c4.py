"""Model Citizens C4 stages (CLI): pilots, cross-play, distillation, planner versus B4.

Commands::

    python -m phasespace.citizens.rl.c4 pilot --model B2 --seed 901 --run-id citizens-c4-pilot-b2-s901
    python -m phasespace.citizens.rl.c4 crossplay --confirm-frozen --train-run <dir> ... --run-id <id>
    python -m phasespace.citizens.rl.c4 distill --confirm-frozen --train-run <dir> ... --run-id <id>
    python -m phasespace.citizens.rl.c4 b4 --confirm-frozen --train-run <dir> --variant plain \
        --run-id <id>

- **pilot** (review item 2): one PPO run with the frozen C3 hyperparameters against a
  non-B1 model. The planner is scored only on its own training model and only on the C3
  monitoring games (never on evaluation games, never on another model): its learning
  curve, plateau, final log sigma, wall time, its final deterministic and sampled
  monitoring return, and the baselines (Equal, Mixed, Proportional, Interpolating k = 22
  and that model's k* from the C3 baselines run) re-scored on the same monitoring games.

- **crossplay** (C4b): every planner (pooled per row, model and action head, after
  `group_training_runs` checks one shared training signature and the declared seeds)
  and every deployed rule (Equal, Mixed, Proportional, Interpolating
  k = 22, each model's k*) against every citizen model on the C4 evaluation games
  (`c4_config.EVAL_KEY_SEED`, common random numbers), deterministic and sampled, with
  per-seed values, paired differences and the extrapolation diagnostics
  (`rl.crossplay`). The input-range diagnostic reads the fit-split human training rows
  of the C2 fit (`data.load_training` and the fit run's split); never held-out rows.
- **distill** (C4c): closed-form rules fitted to each planner's trajectories on its own
  training model (`rl.distill`), with the surplus each distilled rule earns there.
- **b4**: LLM citizens (frozen `b4-prompt-v2`, calibrated or plain) against a planner's
  deterministic policy (`rl.b4play`).

crossplay, distill and b4 produce C4 evidence: they refuse to run unless the C4 protocol
(`c4_config.C4_PROTOCOL_PATH`) is frozen, committed and the tree is clean
(`--confirm-frozen`). Pilots never reach them: pilot runs are refused as inputs.

Runs are new immutable directories (never overwritten), status `computed_unvalidated`.
Pilots read only the frozen C2 fit run and the C3 baselines run: no human data.
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

from phasespace.citizens import data, fit
from phasespace.citizens.fit import load_fitted
from phasespace.citizens.models.base import streams_from_games
from phasespace.citizens.rl import b4play, c4_config, checks, crossplay, distill, ppo, stage, twin
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl.planner import Planner, sampling_step, serialise
from phasespace.gravity.export import (
    Check,
    dumps_json,
    validate_run_id,
    write_run_directory,
)

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

GENERATOR: Final = "phasespace.citizens.rl.c4"
DEFAULT_BASELINES_RUN: Final = Path("runs/citizens-c3-baselines-v1")
PILOT_FILE: Final = "pilot.json"


def c4_manifest(
    run_id: str, stage_name: str, fit_run: Path | None, seed: int | None
) -> dict[str, Any]:
    """The C3 manifest layout with the C4 stage, generator and configuration hash."""
    manifest = stage.base_manifest(run_id, stage_name, fit_run, seed, milestone="c4")
    manifest["source"]["generator"] = GENERATOR
    manifest["validation"]["validated_by"] = f"{GENERATOR} automated checks"
    if stage_name == "pilot":
        manifest["limitations"][0] = (
            "Pre-protocol C4 pilot: scored only on its own model's monitoring games; never "
            "C4 evidence (the C4 stages refuse pilot runs)."
        )
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
        kept_det = stage.coin_weighted_kept_share(*stage.kept_coins(det_traj))
        kept_smp = stage.coin_weighted_kept_share(*stage.kept_coins(smp_traj))
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
            "kept_share_coin_weighted_deterministic": kept_det,
            "kept_share_coin_weighted_sampled": kept_smp,
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


# --- the C4 evidence stages (refused before the C4 protocol freeze) ----------------------------

C4_TRAIN_STAGE: Final = "c4_train"
CROSSPLAY_FILE: Final = "crossplay.json"
SERIES_FILE: Final = "crossplay_series.json"
DISTILL_FILE: Final = "distill.json"


def require_c4_freeze(confirm_frozen: bool) -> tuple[data.FrozenProtocol, str]:
    """The C4 protocol must be frozen and committed, and the tree clean."""
    return stage.require_c4_freeze(confirm_frozen)


def reserved_seeds() -> set[int]:
    """Seeds no C4 training run may use: the C3 training seeds, the pilots' and every
    C3/C4 game key."""
    return {
        *c3config.PILOT_TRAIN_SEEDS,
        *c4_config.PILOT_SEEDS.values(),
        c3config.EVAL_KEY_SEED,
        c3config.TUNE_KEY_SEED,
        c3config.MONITOR_KEY_SEED,
        c3config.SAMPLED_EVAL_KEY_SEED,
        *(c3config.CHECK_SIM_SEED + i for i in range(len(c3config.CHECK_MECHANISMS))),
        c4_config.EVAL_KEY_SEED,
        c4_config.SAMPLED_NOISE_KEY_SEED,
        c4_config.DISTILL_KEY_SEED,
        c4_config.PILOT_SAMPLED_NOISE_KEY_SEED,
    }


def retention_of(manifest: Mapping[str, Any]) -> bool:
    return bool(manifest["citizens"]["training"]["ppo"].get("retention", True))


def shared_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What every C4 planner run must share (protocol §4, K6): iterations, PPO
    configuration and overrides (the retention flag excepted), code commit, fit run,
    C3/C4 configuration hashes and evaluation games. Rows differ only in training model
    (or mixture), seed and retention."""
    t = manifest["citizens"]["training"]
    return {
        "iterations": t["iterations"],
        "ppo": {k: v for k, v in t["ppo"].items() if k != "retention"},
        "overrides": {k: v for k, v in t["overrides"].items() if k != "retention"},
        "git_commit": manifest["source"]["git_commit"],
        "fit_run_manifest_sha256": manifest["citizens"]["fit_run"]["manifest_sha256"],
        "c3_config_sha256": manifest["citizens"].get("c3_config_sha256"),
        "c4_config_sha256": manifest["citizens"].get("c4_config_sha256"),
        "eval_key": manifest["citizens"]["seeds"]["eval_key"],
    }


ROW_ORDER: Final = (*rc.TRAINABLE_MODELS, c4_config.MIXTURE_MODEL)


def expected_rows() -> dict[tuple[str, bool], tuple[int, ...]]:
    """Every declared C4 planner row (model, retention) and its seeds (protocol §4)."""
    rows: dict[tuple[str, bool], tuple[int, ...]] = {}
    for model in ROW_ORDER:
        for retention in (True, False):
            seeds = stage.c4_declared_seeds(model, retention)
            if seeds is not None:
                rows[(model, retention)] = seeds
    return rows


def is_main_row(row: tuple[str, bool]) -> bool:
    return row[0] in rc.TRAINABLE_MODELS and row[1]


def _allowed_seed_sets(row: tuple[str, bool], seeds: tuple[int, ...]) -> list[tuple[int, ...]]:
    """The full declared set; a secondary row (no-retention or mixture) may also be its
    first `SECONDARY_REDUCED_SEEDS` seeds (the pre-declared reduction of §7)."""
    if is_main_row(row):
        return [seeds]
    return [seeds, seeds[: c4_config.SECONDARY_REDUCED_SEEDS]]


def check_c4_train_run(
    tm: Mapping[str, Any], reserved: set[int], declared: Mapping[tuple[str, bool], Any]
) -> tuple[str, bool]:
    """One run's own checks (C4 training stage, clean tree, unreserved seed declared for
    a declared row); returns its row."""
    name = tm["run_id"]
    if tm["citizens"]["stage"] != C4_TRAIN_STAGE:
        raise stage.RunMismatchError(
            f"{name}: stage {tm['citizens']['stage']} is not a C4 training run (C3 runs "
            "and pilot runs are never C4 inputs)"
        )
    if tm["source"]["git_dirty"]:
        raise stage.RunMismatchError(f"{name}: made from a dirty tree")
    seed = int(tm["citizens"]["training"]["seed"])
    if seed in reserved:
        raise stage.RunMismatchError(f"{name}: seed {seed} is reserved")
    row = (str(tm["citizens"]["training"]["model"]), retention_of(tm))
    if row not in declared:
        raise stage.RunMismatchError(f"{name}: row {row} is not declared by the protocol")
    if seed not in declared[row]:
        raise stage.RunMismatchError(
            f"{name}: seed {seed} is not declared for row {row} ({declared[row]})"
        )
    return row


def check_base_matches(base: Mapping[str, Any], signature: Mapping[str, Any]) -> None:
    """The baselines run is clean and shares the training runs' fit run and games."""
    if base["source"]["git_dirty"]:
        raise stage.RunMismatchError(f"{base['run_id']}: made from a dirty tree")
    if signature["fit_run_manifest_sha256"] != base["citizens"]["fit_run"]["manifest_sha256"]:
        raise stage.RunMismatchError("training runs and baselines run use different fit runs")
    if signature["eval_key"] != base["citizens"]["seeds"]["eval_key"]:
        raise stage.RunMismatchError(
            "training runs and baselines run use different evaluation games"
        )
    if signature["iterations"] != c4_config.TRAIN_ITERATIONS:
        raise stage.RunMismatchError(
            f"training runs have {signature['iterations']} iterations, the protocol "
            f"{c4_config.TRAIN_ITERATIONS}"
        )


def group_training_runs(
    base: Mapping[str, Any], trains: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, bool], list[int]]:
    """Indices of the training runs per row (model, retention), after refusing anything
    the protocol does not allow (`stage.RunMismatchError`):

    - a run that is not a C4 training run (C3 runs, pilots), made from a dirty tree, or
      with a reserved seed;
    - a row the protocol does not declare, or a seed not declared for its row;
    - a run whose shared signature (`shared_signature`) differs from the first run's:
      one signature across all rows (K6), with the protocol's iterations;
    - a duplicated seed in a row, a row whose seeds are not its declared set (a
      secondary row may be reduced to its first 2 seeds, the same count for every
      no-retention row), or a missing declared row;
    - a baselines run that is dirty or uses another fit run or other evaluation games.
    """
    if not trains:
        raise stage.RunMismatchError("no training runs")
    reserved = reserved_seeds()
    declared = expected_rows()
    first = shared_signature(trains[0])
    groups: dict[tuple[str, bool], list[int]] = {}
    for i, tm in enumerate(trains):
        row = check_c4_train_run(tm, reserved, declared)
        sig = shared_signature(tm)
        differing = sorted(k for k in sig if sig[k] != first[k])
        if differing:
            raise stage.RunMismatchError(
                f"{tm['run_id']}: training signature differs from {trains[0]['run_id']} in "
                f"{differing}"
            )
        groups.setdefault(row, []).append(i)
    check_base_matches(base, first)
    noret_counts: set[int] = set()
    for row, seeds in declared.items():
        idx = groups.get(row)
        if idx is None:
            raise stage.RunMismatchError(f"row {row} is missing (seeds {seeds})")
        got = [int(trains[i]["citizens"]["training"]["seed"]) for i in idx]
        if len(set(got)) != len(got):
            raise stage.RunMismatchError(f"row {row}: a seed appears twice ({got})")
        if tuple(sorted(got)) not in _allowed_seed_sets(row, seeds):
            raise stage.RunMismatchError(f"row {row}: seeds {sorted(got)}, declared {seeds}")
        if not row[1]:
            noret_counts.add(len(got))
    if len(noret_counts) > 1:
        raise stage.RunMismatchError("the no-retention rows have different seed counts")
    order = {m: j for j, m in enumerate(ROW_ORDER)}
    return dict(sorted(groups.items(), key=lambda kv: (order[kv[0][0]], not kv[0][1])))


def load_policies(
    base: Mapping[str, Any], train_dirs: Sequence[Path], trains: Sequence[Mapping[str, Any]]
) -> list[crossplay.Policy]:
    """Planner rows (grouped, seeds in input order) then the deployed rules."""
    policies: list[crossplay.Policy] = []
    for (model, retention), idx in group_training_runs(base, trains).items():
        seeds = [int(trains[i]["citizens"]["training"]["seed"]) for i in idx]
        planners = [stage.load_planner(train_dirs[i]) for i in idx]
        policies.append(crossplay.planner_policy(model, retention, seeds, planners))
    k_star = {m: k_star_of(base, m) for m in rc.TRAINABLE_MODELS}
    return policies + crossplay.rule_policies(k_star)


def _inputs_record(
    base_dir: Path,
    base: Mapping[str, Any],
    train_dirs: Sequence[Path],
    trains: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "baselines_run": {
            "run_id": base["run_id"],
            "manifest_sha256": stage.manifest_sha256(base_dir),
        },
        "shared_training_signature": shared_signature(trains[0]),
        "train_runs": [
            {
                "run_id": tm["run_id"],
                "manifest_sha256": stage.manifest_sha256(d),
                "model": tm["citizens"]["training"]["model"],
                "retention": retention_of(tm),
                "seed": tm["citizens"]["training"]["seed"],
            }
            for d, tm in zip(train_dirs, trains, strict=True)
        ],
    }


def _read_inputs(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], list[Path], list[dict[str, Any]], Path]:
    base_dir: Path = args.baselines_run
    train_dirs: list[Path] = args.train_run
    base = json.loads((base_dir / "manifest.json").read_text("utf-8"))
    trains = [json.loads((d / "manifest.json").read_text("utf-8")) for d in train_dirs]
    fit_run = Path(base["citizens"]["fit_run"]["path"])
    if stage.manifest_sha256(fit_run) != base["citizens"]["fit_run"]["manifest_sha256"]:
        raise SystemExit(f"{fit_run}: manifest differs from the one the runs used")
    return base_dir, base, train_dirs, trains, fit_run


def fit_split_supports(fit_run: Path, processed_dir: Path) -> dict[str, crossplay.TrainingSupport]:
    """Input supports of B2, B3 and B23 from the fit-split human training rows of the C2
    fit (`data.load_training` refuses held-out games; the split is the fit run's)."""
    training = data.load_training(processed_dir)
    fit_games, _ = fit.training_streams(training, load_fitted(fit_run).split)
    streams = streams_from_games(fit_games)
    return {m: crossplay.training_support(streams, m) for m in c4_config.OOD_MODELS}


def crossplay_command(args: argparse.Namespace) -> int:
    protocol, _ = require_c4_freeze(bool(args.confirm_frozen))
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    base_dir, base, train_dirs, trains, fit_run = _read_inputs(args)
    start = time.perf_counter()
    with twin.x64():
        policies = load_policies(base, train_dirs, trains)
        ports = rc.load_citizens(fit_run)
        b1 = ports["B1"]
        if not isinstance(b1, rc.B1Port):
            raise SystemExit("the B1 port is needed for the disagreement diagnostic")
        supports = fit_split_supports(fit_run, args.processed_dir)
        citizen_key, noise_key = crossplay.eval_keys()
        series: list[crossplay.Series] = []
        for column in rc.TRAINABLE_MODELS:
            diag = crossplay.Diagnostics(
                b1=b1, supports={column: supports[column]} if column in supports else {}
            )
            series += crossplay.evaluate_column(
                policies, column, ports[column], citizen_key, noise_key, c4_config.EVAL_GAMES, diag
            )
            stage.log(f"column {column}: {len(series)} series")
    columns = list(rc.TRAINABLE_MODELS)
    summary = crossplay.summarise(policies, columns, series)
    summary["paired_differences"] = crossplay.paired_differences(policies, columns, series)
    summary["training_support"] = {m: sp.to_json() for m, sp in supports.items()}
    manifest = c4_manifest(run_id, "crossplay", fit_run, c4_config.EVAL_KEY_SEED)
    manifest["citizens"]["c4_protocol"] = {"path": protocol.path, "sha256": protocol.sha256}
    manifest["citizens"]["inputs"] = _inputs_record(base_dir, base, train_dirs, trains)
    manifest["citizens"]["seeds"] = {
        "eval_key": c4_config.EVAL_KEY_SEED,
        "sampled_noise_key": c4_config.SAMPLED_NOISE_KEY_SEED,
        "bootstrap": c4_config.BOOTSTRAP_SEED,
    }
    manifest["citizens"]["held_out_access"] = (
        "none (fit-split human training rows only, for the input-range diagnostic)"
    )
    manifest["citizens"]["seconds"] = time.perf_counter() - start
    files = [
        ("summary", CROSSPLAY_FILE, dumps_json(summary)),
        (
            "summary",
            SERIES_FILE,
            dumps_json(
                {
                    "schema_version": stage.SCHEMA_VERSION,
                    "series": crossplay.series_document(series),
                }
            ),
        ),
    ]
    write_run_directory(out_dir, manifest, files)
    stage.log(f"{out_dir}: {len(series)} series")
    return 0


def distill_planner(
    planner: Planner, port: rc.JaxCitizen, key: Array, n_games: int
) -> dict[str, Any]:
    """Distil one planner on `n_games` games of `key` and play the distilled rules there."""
    traj, _ = eqx.filter_jit(ppo.evaluate_planner)(planner, port, key, n_games, c3config.ROUNDS)
    result = distill.distill(distill.trajectory_rows(traj))
    planner_total = checks.trajectory_metrics(traj)["total_surplus"]
    rules: dict[str, tuple[rc.MechStep, Any]] = {
        "w_rule": (distill.w_rule_step(result.w_rule), None),
        "interpolating_k": (distill.interpolating_step(result.interpolating["k"]), None),
        "tree": (
            distill.tree_rule_step(result.tree, result.first_share),
            distill.tree_initial_state(n_games),
        ),
    }
    surplus: dict[str, Any] = {"planner": float(planner_total.mean())}
    for name, (step, state) in rules.items():
        totals = distill.total_surplus(step, state, port, key, n_games)
        surplus[name] = float(totals.mean())
        surplus[f"{name}_minus_planner"] = float((totals - planner_total).mean())
    return {**result.to_json(), "total_surplus_mean": surplus}


def distill_command(args: argparse.Namespace) -> int:
    protocol, _ = require_c4_freeze(bool(args.confirm_frozen))
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    base_dir, base, train_dirs, trains, fit_run = _read_inputs(args)
    out: list[dict[str, Any]] = []
    with twin.x64():
        policies = load_policies(base, train_dirs, trains)
        ports = rc.load_citizens(fit_run)
        key = jax.random.key(c4_config.DISTILL_KEY_SEED)
        for policy in policies:
            if policy.kind != "planner" or policy.training_model is None:
                continue
            for seed, planner in zip(policy.seeds, policy.planners, strict=True):
                port = ports[policy.training_model]
                record = distill_planner(planner, port, key, c4_config.DISTILL_GAMES)
                out.append({"policy": policy.name, "seed": seed, **record})
                stage.log(f"{policy.name} seed {seed}: {record['total_surplus_mean']}")
    manifest = c4_manifest(run_id, "distill", fit_run, c4_config.DISTILL_KEY_SEED)
    manifest["citizens"]["c4_protocol"] = {"path": protocol.path, "sha256": protocol.sha256}
    manifest["citizens"]["inputs"] = _inputs_record(base_dir, base, train_dirs, trains)
    manifest["citizens"]["seeds"] = {"distill_key": c4_config.DISTILL_KEY_SEED}
    write_run_directory(
        out_dir,
        manifest,
        [
            (
                "summary",
                DISTILL_FILE,
                dumps_json({"schema_version": stage.SCHEMA_VERSION, "planners": out}),
            )
        ],
    )
    return 0


def b4_command(args: argparse.Namespace) -> int:
    """Planner versus B4 (never run before the C4 freeze; calls the Claude Code CLI)."""
    from phasespace.citizens import heldout
    from phasespace.citizens.llm import runner
    from phasespace.citizens.llm.backend import ClaudeCLI

    protocol, commit = require_c4_freeze(bool(args.confirm_frozen))
    validate_run_id(args.run_id)
    if (args.runs_dir / args.run_id).exists():
        raise SystemExit(f"{args.runs_dir / args.run_id} already exists")
    train_dir: Path = args.train_run
    tm = json.loads((train_dir / "manifest.json").read_text("utf-8"))
    if tm["citizens"]["stage"] != C4_TRAIN_STAGE:
        raise SystemExit(f"{tm['run_id']} is not a training run")
    backend = ClaudeCLI()
    heldout.require_cli_version(backend.describe())
    retention = retention_of(tm)
    model = tm["citizens"]["training"]["model"]
    name = f"{crossplay.planner_name(model, retention)} seed {tm['citizens']['training']['seed']}"
    with twin.x64():
        mechanism = b4play.PlannerGameMechanism(name, stage.load_planner(train_dir))
    weights_record: dict[str, Any] | None = None
    if args.variant == "calibrated":
        if args.calibration_run is None:
            raise SystemExit("--calibration-run is required for B4-calibrated")
        heldout.check_calibration_identity(args.calibration_run, backend.describe())
        weights_record = heldout.type_weights(args.calibration_run, args.processed_dir)
        w = weights_record["weights"]

        def spec(k: int) -> runner.GameSpec:
            return runner.GameSpec(k, heldout.N_ROUNDS, runner.evaluation_types(w, k))
    else:

        def spec(k: int) -> runner.GameSpec:
            return runner.GameSpec(k, heldout.N_ROUNDS, (None, None, None, None))

    start = time.perf_counter()
    try:
        config, outcomes = b4play.run_planner_games(
            mechanism,
            backend,
            args.runs_dir,
            args.run_id,
            int(args.games),
            spec,
            variant=args.variant,
            workers=int(args.workers),
            log=lambda m: print(m, flush=True),
        )
    except runner.InfrastructureError as exc:
        print(f"stopped: {exc}", flush=True)
        return 3
    extra = {
        "c4_protocol": {"path": protocol.path, "sha256": protocol.sha256},
        "c4_config_sha256": c4_config.config_sha256(),
        "git_commit_at_start": commit,
        "planner": b4play.describe_planner(
            tm["run_id"], stage.manifest_sha256(train_dir), retention
        ),
        "type_weights": weights_record,
    }
    manifest = runner.finalize_games_run(
        args.runs_dir, config, outcomes, wall_s=time.perf_counter() - start, extra=extra
    )
    print(f"{args.runs_dir / args.run_id}: {manifest['status']}")
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
    for name in ("crossplay", "distill"):
        q = sub.add_parser(name)
        q.add_argument("--confirm-frozen", action="store_true")
        q.add_argument("--baselines-run", type=Path, default=DEFAULT_BASELINES_RUN)
        q.add_argument("--train-run", type=Path, nargs="+", required=True)
        q.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
        q.add_argument("--run-id", required=True)
    b = sub.add_parser("b4")
    b.add_argument("--confirm-frozen", action="store_true")
    b.add_argument("--train-run", type=Path, required=True)
    b.add_argument("--variant", choices=("plain", "calibrated"), required=True)
    b.add_argument("--calibration-run", type=Path)
    b.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    b.add_argument("--games", type=int, default=c4_config.B4_PLANNER_GAMES)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    commands: Mapping[str, Callable[[argparse.Namespace], int]] = {
        "pilot": pilot_command,
        "crossplay": crossplay_command,
        "distill": distill_command,
        "b4": b4_command,
    }
    return commands[str(args.command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
