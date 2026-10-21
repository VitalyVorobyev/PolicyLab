"""Model Citizens public demonstration export (CLI; deterministic; reads recorded runs).

Command::

    python -m phasespace.citizens.web_export --runs-dir runs --run-id citizens-web-demo-v1 \\
        --confirm-frozen --crossplay-run runs/citizens-c4-crossplay-v1 \\
        --distill-run runs/citizens-c4-distill-v1 --baselines-run runs/citizens-c3-baselines-v1 \\
        --train-run runs/citizens-c4-train-b1-s1001-v1 ... runs/citizens-c4-train-mix-s3001-v1 \\
        [--heldout-eval-run <dir>] [--b4-eval-run <dir>] [--b4-run <dir> ...] \\
        [--hypotheses-run <dir>] [--e1-run <dir>]

One artifact, `citizens_demo.json` (kind `citizens_demo`, compact JSON), for the public
page (docs/data-contract.md, "Model Citizens addendum (web demo)"):

- **traces**: per-round pool, offers, returns, kept coins and running surplus of games 0,
  1 and 2 (the first three by index, fixed in advance) of every (governor, citizen model)
  cell. The governors are Equal, Proportional, Interpolating k = 22 and the main planners
  of seed 1001 (Planner[B1], [B2], [B3], [B23]) plus Planner[Mix] of seed 3001 (the first
  declared seed of each row, fixed in advance: no selection on outcome), deterministic
  policy. The cross-play run stores only per-game totals, so every column is re-simulated
  on the full C4 evaluation batch (the citizen draws are made for the whole batch) with
  `crossplay.deterministic_trajectories`, and the export fails, writing nothing, unless
  every re-simulated per-game total surplus equals the cross-play series within 1e-9
  coins (all 1,000 games of every simulated cell);
- **human games**: two real games per mechanism (Equal, Proportional, Interpolating,
  M2) from the Koster et al. (2025) data (CC BY 4.0): the first two by (launch id, CSV
  row) among the human games of Exps 1-3 with exactly 40 rounds and all four players
  present in every round. Held-out games are read through `data.load_heldout` under the
  frozen validation protocol (`--confirm-frozen`);
- **LLM games** (optional `--b4-run`): the first three complete games of each B4 run (C2
  held-out Interpolating runs, C4 planner-versus-B4 runs), parsed actions only (never the
  call records);
- **summaries** copied from the recorded runs: the cross-play matrix (mean total surplus
  and Gini, the paired difference against the column's k* with its CI and seed signs),
  the k-tuning curves and k*, the distilled w(R) of each planner (seed means), the two
  held-out exams per citizen model, the C2 B4 labels, and, when given, the hypothesis
  verdicts and an E1 run's sample tables.

Coin values are rounded to 2 decimals only after the reproduction check. Status
`computed_unvalidated`: the checks verify reproduction of computed_unvalidated runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.citizens import data
from phasespace.citizens.records import Games
from phasespace.gravity.export import Check, validate_run_id, write_run_directory

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

GENERATOR: Final = "phasespace.citizens.web_export"
EXPERIMENT_ID: Final = "citizens-web-demo-v1"
STAGE: Final = "web_export"
DEMO_FILE: Final = "citizens_demo.json"
DEMO_KIND: Final = "citizens_demo"

MAX_POOL: Final = 200.0
GROWTH_FACTOR: Final = 1.4
POOL_FLOOR: Final = 0.01
ROUNDS: Final = 40
PLAYERS: Final = 4

TRACE_GAMES: Final = (0, 1, 2)
"""Games shown per cell: the first three by index (fixed before any trace was seen)."""
TRACE_RULES: Final = ("Equal", "Proportional", "Interpolating k=22")
TRACE_PLANNER_SEED: Final = 1001
TRACE_MIX_SEED: Final = 3001
TRACE_PLANNER_MODELS: Final = ("B1", "B2", "B3", "B23")
COLUMNS: Final = ("B1", "B2", "B3", "B23")
SUMMARY_RULES: Final = ("Equal", "Mixed", "Proportional", "Interpolating k=22")
REPRODUCTION_ATOL: Final = 1e-9
DECIMALS: Final = 2

HUMAN_MECHANISMS: Final = (
    ("Equal", "Equal Baseline"),
    ("Proportional", "Proportional Baseline"),
    ("Interpolating k=22", "Interpolating Baseline"),
    ("M2", "RL Agent (M2)"),
)
"""(governor shown on the page, mechanism in the parsed data)."""
HUMAN_GAMES_PER_MECHANISM: Final = 2
HUMAN_COHORTS: Final = ("Exp 1", "Exp 2", "Exp 3")
HUMAN_SELECTION_RULE: Final = (
    "the first two by (launch id, CSV row) among the human games of Exps 1-3 of the "
    "mechanism with exactly 40 recorded rounds and all four players present in every round"
)
B4_TRACE_GAMES: Final = 3
B4_C2_STAGE: Final = "c2_b4_heldout"
B4_C4_STAGE: Final = "c2_b4_c4-planner"
DATA_ATTRIBUTION: Final = (
    "Koster, R. et al. (2025), Nature Communications 16:2824, "
    "doi:10.1038/s41467-025-58043-7; data CC BY 4.0"
)


class ReproductionError(RuntimeError):
    """A re-simulated value differs from the recorded run."""


# --- the game rule and per-game records -----------------------------------------------------------


def next_pool(pool: FloatArray, offers: FloatArray, returns: FloatArray) -> FloatArray:
    """R' = max(0.01, min(200, R - sum e + 1.4 sum c)) (players on the last axis)."""
    raw = pool - offers.sum(axis=-1) + GROWTH_FACTOR * returns.sum(axis=-1)
    return np.maximum(POOL_FLOOR, np.minimum(MAX_POOL, raw))


def _round(values: FloatArray | float) -> Any:
    return np.round(np.asarray(values, dtype=np.float64), DECIMALS).tolist()


def game_record(
    pool: FloatArray, offers: FloatArray, returns: FloatArray, final_pool: float
) -> dict[str, Any]:
    """One game, rounded for display: `pool` has T + 1 values (before each round, then
    after the last), `kept` = offers - returns, `cumulative_surplus` the running sum of
    kept over players and rounds; `total_surplus` is the exact total, rounded."""
    kept = offers - returns
    cumulative = np.cumsum(kept.sum(axis=1))
    return {
        "rounds": int(offers.shape[0]),
        "pool": _round(np.append(pool, final_pool)),
        "offers": _round(offers),
        "returns": _round(returns),
        "kept": _round(kept),
        "cumulative_surplus": _round(cumulative),
        "total_surplus": float(np.round(float(kept.sum()), DECIMALS)),
    }


def max_total_difference(simulated: FloatArray, recorded: Sequence[float]) -> float:
    """max |simulated - recorded| over games (inf when the lengths differ)."""
    rec = np.asarray(recorded, dtype=np.float64)
    if rec.shape != simulated.shape:
        return math.inf
    return float(np.abs(simulated - rec).max()) if rec.size else 0.0


def check_reproduction(
    simulated: FloatArray, recorded: Sequence[float], label: str, atol: float = REPRODUCTION_ATOL
) -> float:
    """The largest per-game difference; raises ReproductionError above `atol`."""
    worst = max_total_difference(simulated, recorded)
    if not worst <= atol:
        raise ReproductionError(
            f"{label}: re-simulated total surplus differs from the recorded series by {worst} "
            f"coins (> {atol})"
        )
    return worst


def trace_residuals(
    pool: FloatArray, offers: FloatArray, returns: FloatArray, final_pool: float
) -> dict[str, float]:
    """Largest violations of the game rule in one game: the pool update, offers above
    the pool, returns above the offer or below zero."""
    pools = np.append(pool, final_pool)
    update = np.abs(pools[1:] - next_pool(pool, offers, returns))
    return {
        "pool_update": float(update.max()),
        "offers_above_pool": float(max(0.0, (offers.sum(axis=1) - pool).max())),
        "returns_outside_offer": float(max(0.0, (returns - offers).max(), (-returns).max())),
    }


# --- selection of the human games ---------------------------------------------------------------


def select_human_games(
    games: Games, mechanism: str, n: int = HUMAN_GAMES_PER_MECHANISM
) -> list[int]:
    """Indices of the pre-declared human games of `mechanism` (`HUMAN_SELECTION_RULE`)."""
    candidates: list[tuple[str, int, int]] = []
    for i in range(games.n_games):
        if games.mechanism[i] != mechanism or games.cohort[i] not in HUMAN_COHORTS:
            continue
        rounds = int(games.n_rounds[i])
        if rounds != ROUNDS or not bool((games.four_players_present[i, :rounds] == 1).all()):
            continue
        candidates.append((str(games.launch_id[i]), int(games.first_row[i]), i))
    candidates.sort()
    return [i for _, _, i in candidates[:n]]


def human_record(games: Games, i: int, governor: str, split: str) -> dict[str, Any]:
    t = int(games.n_rounds[i])
    pool = games.pool[i, :t].astype(np.float64)
    offers = games.offers[i, :t].astype(np.float64)
    returns = games.actions[i, :t].astype(np.float64)
    record = game_record(pool, offers, returns, float(games.next_pool[i, t - 1]))
    return {
        "governor": governor,
        "mechanism": str(games.mechanism[i]),
        "cohort": str(games.cohort[i]),
        "game_id": str(games.launch_id[i]),
        "split": split,
        **record,
    }


# --- reading recorded runs ------------------------------------------------------------------------


def manifest_of(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    return out


def manifest_sha256(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()


def read_verified(run_dir: Path, path: str) -> bytes:
    """An artifact's bytes after checking its SHA-256 against the run's manifest."""
    manifest = manifest_of(run_dir)
    expected = next((a["sha256"] for a in manifest["artifacts"] if a["path"] == path), None)
    if expected is None:
        raise ValueError(f"{run_dir}: the manifest lists no artifact {path}")
    payload = (run_dir / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{run_dir / path}: SHA-256 differs from the manifest")
    return payload


def read_json(run_dir: Path, path: str) -> Any:
    return json.loads(read_verified(run_dir, path).decode("utf-8"))


def table_dicts(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = [c["name"] for c in table["columns"]]
    return [dict(zip(names, row, strict=True)) for row in table["rows"]]


def _input(run_dir: Path) -> dict[str, Any]:
    m = manifest_of(run_dir)
    return {
        "run_id": m["run_id"],
        "manifest_sha256": manifest_sha256(run_dir),
        "git_commit": m["source"]["git_commit"],
        "git_dirty": m["source"]["git_dirty"],
        "status": m["status"],
    }


def _require_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    m = manifest_of(run_dir)
    got = m.get("citizens", {}).get("stage")
    if got != stage:
        raise SystemExit(f"{run_dir}: stage {got!r}, expected {stage!r}")
    return m


# --- summaries (copied, never re-estimated) ---------------------------------------------------------


def k_star_name(column: str) -> str:
    return f"Interpolating k*[{column}]"


def crossplay_summary(cp: Mapping[str, Any], policies: Sequence[str]) -> dict[str, Any]:
    """Per (policy, column), deterministic mode: mean total surplus and Gini, per-seed
    means, and the differences against the column's k* and against k = 22 (the paired
    planner records with CI and seed signs where the cross-play run has them; for a rule
    the difference of the two means on the same games)."""
    paired = {
        (d["policy"], d["column"], d["versus"]): d
        for d in cp["paired_differences"]
        if d["mode"] == "deterministic"
    }
    cells: list[dict[str, Any]] = []
    for policy in policies:
        for column in cp["columns"]:
            cell = cp["cells"][policy][column]["deterministic"]
            mean = float(cell["total_surplus_mean"])
            diffs: dict[str, Any] = {}
            for key, versus in (
                ("vs_k_star", k_star_name(column)),
                ("vs_k22", "Interpolating k=22"),
            ):
                ref = float(cp["cells"][versus][column]["deterministic"]["total_surplus_mean"])
                d = paired.get((policy, column, versus))
                diffs[key] = {
                    "versus": versus,
                    "difference": mean - ref if d is None else float(d["difference_mean"]),
                    "relative": (mean - ref) / ref,
                    "ci": None if d is None else [float(x) for x in d["difference_ci"]],
                    "seeds_positive": None if d is None else int(d["seeds_positive"]),
                    "seeds_negative": None if d is None else int(d["seeds_negative"]),
                }
            cells.append(
                {
                    "policy": policy,
                    "column": column,
                    "total_surplus_mean": mean,
                    "total_surplus_per_seed": [float(x) for x in cell["total_surplus_per_seed"]],
                    "seeds": cell["seeds"],
                    "gini_mean": float(cell["gini_mean"]),
                    **diffs,
                }
            )
    k_star = {
        c: next(float(p["rule"]["param"]) for p in cp["policies"] if p["name"] == k_star_name(c))
        for c in cp["columns"]
    }
    return {
        "columns": list(cp["columns"]),
        "policies": list(policies),
        "mode": "deterministic",
        "k_star": k_star,
        "cells": cells,
    }


def k_tuning(k_grid: Mapping[str, Any], base_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Mean total surplus on the tuning games per grid ln k and model, and each k*."""
    rows = table_dicts(k_grid)
    ln_k = sorted({float(r["ln_k"]) for r in rows})
    curves: dict[str, list[float]] = {}
    for model in COLUMNS:
        by = {
            float(r["ln_k"]): float(r["tune_total_surplus_mean"])
            for r in rows
            if r["model"] == model
        }
        curves[model] = [by[x] for x in ln_k]
    k_star = {
        m: {"k": float(v["k"]), "ln_k": float(v["ln_k"])}
        for m, v in base_manifest["citizens"]["k_star"].items()
        if m in COLUMNS
    }
    cfg = base_manifest["citizens"]["c3_config"]
    return {
        "ln_k": ln_k,
        "curves": curves,
        "k_star": k_star,
        "k_published": float(cfg["k_published"]),
        "games": int(cfg["tune_games"]),
        "key_seed": int(cfg["tune_key_seed"]),
        "definition": "mean total surplus of the deployed Interpolating rule over the C3 "
        "tuning games (not the cross-play games) per grid point ln k; k* is the smallest "
        "ln k with the maximal mean",
    }


def interpolating_weight(fraction: FloatArray, k: float) -> FloatArray:
    """The deployed weight round((R/200)^k, 1) (`mechanisms.interpolating_weight`)."""
    from phasespace.citizens.mechanisms import interpolating_weight as deployed

    return np.asarray(deployed(fraction * MAX_POOL, k), dtype=np.float64)


def distilled_curves(doc: Mapping[str, Any], policies: Sequence[str], k22: float) -> dict[str, Any]:
    """Seed means of each planner's binned w(R) and kept share, with the per-seed
    curves, and the deployed k = 22 weight on a fine grid of R/200 for reference."""
    out: list[dict[str, Any]] = []
    bins = None
    for policy in policies:
        seeds = [p for p in doc["planners"] if p["policy"] == policy]
        if not seeds:
            continue
        w = np.array(
            [[np.nan if v is None else v for v in p["w_rule"]["w"]] for p in seeds],
            dtype=np.float64,
        )
        kept = np.array(
            [[np.nan if v is None else v for v in p["w_rule"]["kept_share"]] for p in seeds],
            dtype=np.float64,
        )
        bins = int(seeds[0]["w_rule"]["bins"])

        def mean_or_none(a: FloatArray) -> list[float | None]:
            return [
                None if np.isnan(a[:, j]).all() else float(np.nanmean(a[:, j]))
                for j in range(a.shape[1])
            ]

        out.append(
            {
                "policy": policy,
                "seeds": [int(p["seed"]) for p in seeds],
                "w_mean": mean_or_none(w),
                "kept_share_mean": mean_or_none(kept),
                "w_per_seed": [[None if np.isnan(v) else float(v) for v in row] for row in w],
                "r2_per_seed": [p["w_rule"]["r2"] for p in seeds],
                "fitted_k_per_seed": [float(p["interpolating_k"]["k"]) for p in seeds],
            }
        )
    grid = np.linspace(0.0, 1.0, 201)
    return {
        "bins": bins,
        "bin_edges": None if bins is None else np.linspace(0.0, 1.0, bins + 1).tolist(),
        "definition": "w = weight on the equal split in s_i = (1 - kappa)(w/4 + (1 - w) q_i), "
        "least squares per R/200 bin on the planner's own training model (C4 distillation "
        "games); seed means",
        "planners": out,
        "reference": {
            "rule": "Interpolating k=22",
            "pool_fraction": grid.tolist(),
            "w": interpolating_weight(grid, k22).tolist(),
        },
    }


def exams(eval_dir: Path) -> dict[str, Any]:
    """The two held-out exams per citizen model and regime: the next-move score (delta
    ell vs M0, with its CI) and the whole-game metrics in range, with the C2 labels."""
    m = manifest_of(eval_dir)
    criterion = table_dicts(read_json(eval_dir, "criterion_heldout.json"))
    outcomes = table_dicts(read_json(eval_dir, "outcomes_heldout.json"))
    labels = m["citizens"]["labels"]
    rows: list[dict[str, Any]] = []
    for r in criterion:
        if r["model"] not in COLUMNS:
            continue
        rows.append(
            {
                "regime": r["regime"],
                "model": r["model"],
                "delta_ell": r["delta_ell"],
                "delta_ci": [r["delta_ci_lo"], r["delta_ci_hi"]],
                "pass_likelihood": r["pass_likelihood"] == "true",
                "metrics_in_range": r["metrics_in_range"],
                "pass_outcomes": r["pass_outcomes"] == "true",
                "mean_normalised_w1": r["mean_normalised_w1"],
                "label": labels[r["regime"]][r["model"]],
                "units": r["units"],
            }
        )
    metrics = [
        {
            "regime": r["regime"],
            "model": r["model"],
            "metric": r["metric"],
            "human_ci": [r["human_ci_lo"], r["human_ci_hi"]],
            "tolerance": r["tolerance"],
            "sim_mean": r["sim_mean"],
            "in_range": r["in_range"] == "true",
        }
        for r in outcomes
        if r["model"] in COLUMNS
    ]
    tier = m["citizens"]["surrogates"]["closed_loop_tier"]
    return {
        "regimes": sorted({r["regime"] for r in rows}),
        "closed_loop_tier": tier,
        "rows": rows,
        "metrics": metrics,
        "rule_likelihood": "next move: delta ell > 0 and its 95 % CI above 0",
        "rule_outcomes": "whole game: at least 3 of 4 game metrics inside the widened human CI",
    }


def b4_c2(eval_dir: Path) -> dict[str, Any]:
    m = manifest_of(eval_dir)
    keys = (
        "variant",
        "regime",
        "games_complete",
        "metrics_in_range",
        "pass_outcomes",
        "mean_normalised_w1",
        "m0_sim_mean_normalised_w1",
        "closed_loop",
        "label",
        "counts_as_validated",
    )
    records = [{k: r.get(k) for k in keys} for r in m["citizens"]["b4"]]
    outcomes = table_dicts(read_json(eval_dir, "b4_outcomes.json"))
    metrics = [
        {
            "variant": r["variant"],
            "regime": r["regime"],
            "metric": r["metric"],
            "b4_mean": r["b4_mean"],
            "b4_ci": [r["b4_ci_lo"], r["b4_ci_hi"]],
            "human_ci": [r["human_ci_lo"], r["human_ci_hi"]],
            "in_range": r["in_range"] == "true",
        }
        for r in outcomes
    ]
    return {"records": records, "metrics": metrics}


def b4_runs(run_dirs: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """(rows, traces, worst total check) of B4 game runs: a C2 held-out Interpolating run
    is the k = 22 row, a C4 run the row in its `c4_row`. Totals come from the parsed
    rounds and are checked against the run's games.json surplus (x 4 players x 40 rounds)."""
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    worst = 0.0
    for run_dir in run_dirs:
        m = manifest_of(run_dir)
        c = m["citizens"]
        variant = str(c["config"]["variant"])
        if c.get("stage") == B4_C2_STAGE and c.get("mechanism_key") == "Interpolating":
            governor, origin = "Interpolating k=22", "c2"
        elif c.get("stage") == B4_C4_STAGE:
            governor, origin = str(c["c4_row"]["name"]), "c4"
        else:
            raise SystemExit(f"{run_dir}: not a C2 held-out Interpolating or C4 B4 run")
        games = table_dicts(read_json(run_dir, "games.json"))
        rounds = table_dicts(read_json(run_dir, "rounds.json"))
        complete = [int(g["game"]) for g in games if g["status"] == "complete"]
        totals: list[float] = []
        for k, g in enumerate(complete):
            rs = sorted((r for r in rounds if int(r["game"]) == g), key=lambda r: int(r["round"]))
            pool = np.array([r["pool"] for r in rs], dtype=np.float64)
            offers = np.array(
                [[r[f"offer_{i}"] for i in range(PLAYERS)] for r in rs], dtype=np.float64
            )
            returns = np.array(
                [[r[f"return_{i}"] for i in range(PLAYERS)] for r in rs], dtype=np.float64
            )
            total = float((offers - returns).sum())
            recorded = (
                next(float(x["surplus"]) for x in games if int(x["game"]) == g) * PLAYERS * ROUNDS
            )
            worst = max(worst, abs(total - recorded))
            totals.append(total)
            if k < B4_TRACE_GAMES:
                traces.append(
                    {
                        "governor": governor,
                        "citizens": f"B4-{variant}",
                        "game": g,
                        "source": origin,
                        **game_record(pool, offers, returns, float(rs[-1]["next_pool"])),
                    }
                )
        arr = np.array(totals, dtype=np.float64)
        rows.append(
            {
                "governor": governor,
                "variant": variant,
                "source": origin,
                "games_complete": len(complete),
                "games_discarded": sum(1 for g in games if g["status"] == "discarded"),
                "total_surplus_mean": float(arr.mean()) if arr.size else None,
                "total_surplus_sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
                "cli_version": m.get("simulator", {}).get("backend", {}).get("cli_version"),
            }
        )
    return rows, traces, worst


def hypotheses_block(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Verdicts, confirmed cells and row labels of the hypothesis stage (deterministic)."""
    main = doc["modes"]["deterministic"]["main"]
    confirmed = {name: list(f["confirmed"]) for name, f in main["families"].items()}
    out: dict[str, Any] = {
        "verdicts": doc["verdicts"],
        "confirmed": confirmed,
        "rows": main["rows"],
        "delta": doc["delta"],
    }
    if "H2_B4" in doc:
        out["h2_b4"] = {
            variant: {
                "validated_evaluation_population": v.get("validated_evaluation_population"),
                "verdict": v.get("H2-B4", {}).get("verdict"),
                "label": v.get("label"),
                "descriptive": v.get("descriptive"),
            }
            for variant, v in doc["H2_B4"].items()
        }
    return out


def e1_block(run_dir: Path) -> dict[str, Any]:
    """An E1 run's sample tables, verbatim (columns and rows), keyed by file name."""
    m = manifest_of(run_dir)
    tables = {
        a["path"]: read_json(run_dir, a["path"])
        for a in m["artifacts"]
        if a["kind"] == "sample_table"
    }
    return {
        "run_id": m["run_id"],
        "experiment_id": m["experiment_id"],
        "tables": {k: {"columns": v["columns"], "rows": v["rows"]} for k, v in tables.items()},
    }


# --- simulation ------------------------------------------------------------------------------------


def simulate_traces(
    fit_run: Path,
    train_dirs: Mapping[tuple[str, int], Path],
    k_star: Mapping[str, float],
    series: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Re-simulate every traced governor in every column on the C4 evaluation games,
    check the per-game totals against the recorded series, and keep games 0..2."""
    from phasespace.citizens.rl import c4_config, checks, crossplay, stage, twin
    from phasespace.citizens.rl import citizens as rc

    recorded = {
        (s["policy"], s["column"], s["seed"]): s["total_surplus"]
        for s in series
        if s["mode"] == "deterministic"
    }
    traces: list[dict[str, Any]] = []
    worst = {
        "total": 0.0,
        "pool_update": 0.0,
        "offers_above_pool": 0.0,
        "returns_outside_offer": 0.0,
    }
    with twin.x64():
        policies = [
            crossplay.planner_policy(model, True, [seed], [stage.load_planner(path)])
            for (model, seed), path in train_dirs.items()
        ]
        policies += [p for p in crossplay.rule_policies(k_star) if p.name in TRACE_RULES]
        ports = rc.load_citizens(fit_run)
        citizen_key, _ = crossplay.eval_keys()
        for column in COLUMNS:
            for name, seed, traj in crossplay.deterministic_trajectories(
                policies, ports[column], citizen_key, c4_config.EVAL_GAMES
            ):
                totals: FloatArray = checks.trajectory_metrics(traj)["total_surplus"]
                key = (name, column, seed)
                if key not in recorded:
                    raise ReproductionError(f"{key}: no recorded cross-play series")
                worst["total"] = max(
                    worst["total"], check_reproduction(totals, recorded[key], str(key))
                )
                offers = np.asarray(traj.offers, dtype=np.float64)
                returns = np.asarray(traj.actions, dtype=np.float64)
                pools = np.asarray(traj.pool, dtype=np.float64)
                finals = np.asarray(
                    twin.next_pool(traj.pool[:, -1], traj.offers[:, -1], traj.actions[:, -1]),
                    dtype=np.float64,
                )
                for g in TRACE_GAMES:
                    res = trace_residuals(pools[g], offers[g], returns[g], float(finals[g]))
                    for k, v in res.items():
                        worst[k] = max(worst[k], v)
                    record = game_record(pools[g], offers[g], returns[g], float(finals[g]))
                    record["total_surplus"] = round(float(totals[g]), DECIMALS)
                    traces.append(
                        {"governor": name, "seed": seed, "citizens": column, "game": g, **record}
                    )
                stage.log(f"{column} {name}: max |total - recorded| {worst['total']:.3g}")
    return traces, worst


# --- the run -----------------------------------------------------------------------------------------


def _limitations(dirty: bool) -> list[str]:
    out = [
        "Curated public sample: three games per cell (games 0-2, fixed in advance), "
        "deterministic planner policy, one training seed per planner row (1001; Mix 3001); "
        "every other number is copied from the recorded runs listed in citizens.inputs.",
        "Citizen models keep their C2 validation labels (D35): only B1 is closed-loop "
        "validated; every planner is judged by simulated citizens, not by new human players.",
        "Coin values in traces are rounded to 2 decimals after the reproduction check.",
    ]
    if dirty:
        out.append(
            "source.git_dirty is true: exported from an uncommitted working tree; for "
            "development only, replace with a clean re-export before release."
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m phasespace.citizens.web_export", description=__doc__
    )
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--confirm-frozen", action="store_true", help="allow the held-out human read")
    p.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    p.add_argument("--crossplay-run", type=Path, required=True)
    p.add_argument("--distill-run", type=Path, required=True)
    p.add_argument("--baselines-run", type=Path, required=True)
    p.add_argument("--train-run", type=Path, nargs="+", required=True)
    p.add_argument("--heldout-eval-run", type=Path)
    p.add_argument("--b4-eval-run", type=Path)
    p.add_argument("--b4-run", type=Path, nargs="*", default=[])
    p.add_argument("--hypotheses-run", type=Path)
    p.add_argument("--e1-run", type=Path)
    return p


def _trace_train_runs(
    train_runs: Sequence[Path], cp_manifest: Mapping[str, Any]
) -> dict[tuple[str, int], Path]:
    """The pre-declared trace planners among `train_runs`, each an input of the cross-play
    run (same manifest sha256); any other training run passed is ignored."""
    declared = {(m, TRACE_PLANNER_SEED) for m in TRACE_PLANNER_MODELS} | {("Mix", TRACE_MIX_SEED)}
    inputs = {r["manifest_sha256"]: r for r in cp_manifest["citizens"]["inputs"]["train_runs"]}
    out: dict[tuple[str, int], Path] = {}
    for d in train_runs:
        tm = manifest_of(d)
        training = tm["citizens"]["training"]
        key = (str(training["model"]), int(training["seed"]))
        retention = bool(training["ppo"].get("retention", True))
        if key not in declared or not retention:
            continue
        if manifest_sha256(d) not in inputs:
            raise SystemExit(f"{d}: not an input of the cross-play run")
        out[key] = d
    missing = sorted(declared - set(out))
    if missing:
        raise SystemExit(f"missing training runs for the traces: {missing}")
    order = [*((m, TRACE_PLANNER_SEED) for m in TRACE_PLANNER_MODELS), ("Mix", TRACE_MIX_SEED)]
    return {k: out[k] for k in order}


def export_command(args: argparse.Namespace) -> int:
    from phasespace.citizens.rl import stage

    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    cp_dir: Path = args.crossplay_run
    cp_m = _require_stage(cp_dir, "c4_crossplay")
    dist_m = _require_stage(args.distill_run, "c4_distill")
    base_m = _require_stage(args.baselines_run, "c3_baselines")
    if cp_m["citizens"]["inputs"]["baselines_run"]["manifest_sha256"] != manifest_sha256(
        args.baselines_run
    ):
        raise SystemExit("the baselines run is not the cross-play run's input")
    cp_train = {r["manifest_sha256"] for r in cp_m["citizens"]["inputs"]["train_runs"]}
    if cp_train != {r["manifest_sha256"] for r in dist_m["citizens"]["inputs"]["train_runs"]}:
        raise SystemExit("cross-play and distillation used different training runs")
    fit_run = Path(base_m["citizens"]["fit_run"]["path"])
    if manifest_sha256(fit_run) != base_m["citizens"]["fit_run"]["manifest_sha256"]:
        raise SystemExit(f"{fit_run}: manifest differs from the one the runs used")
    train_dirs = _trace_train_runs(args.train_run, cp_m)

    cp = read_json(cp_dir, "crossplay.json")
    series = read_json(cp_dir, "crossplay_series.json")["series"]
    k_star = {c: float(base_m["citizens"]["k_star"][c]["k"]) for c in COLUMNS}

    traces, worst = simulate_traces(fit_run, train_dirs, k_star, series)

    # Human games (training and, under the frozen protocol, held-out).
    training = data.load_training(args.processed_dir)
    heldout = data.load_heldout(
        confirm_frozen=bool(args.confirm_frozen), processed_dir=args.processed_dir
    )
    human: list[dict[str, Any]] = []
    for governor, mechanism in HUMAN_MECHANISMS:
        heldout_mech = mechanism not in ("Equal Baseline", "Proportional Baseline")
        games, split = (heldout.games, "heldout") if heldout_mech else (training, "training")
        chosen = select_human_games(games, mechanism)
        if len(chosen) < HUMAN_GAMES_PER_MECHANISM:
            raise SystemExit(f"{mechanism}: fewer than {HUMAN_GAMES_PER_MECHANISM} eligible games")
        human += [human_record(games, i, governor, split) for i in chosen]

    summary_policies = [
        *SUMMARY_RULES,
        *(f"Planner[{m}]" for m in TRACE_PLANNER_MODELS),
        "Planner[Mix]",
    ]
    demo: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "game": {
            "players": PLAYERS,
            "rounds": ROUNDS,
            "max_pool": MAX_POOL,
            "growth_factor": GROWTH_FACTOR,
            "pool_floor": POOL_FLOOR,
            "pool_rule": "R' = max(0.01, min(200, R - sum e + 1.4 sum c))",
        },
        "trace_games": list(TRACE_GAMES),
        "traces": traces,
        "human_games": human,
        "human_selection_rule": HUMAN_SELECTION_RULE,
        "data_attribution": DATA_ATTRIBUTION,
        "crossplay": {
            **crossplay_summary(cp, summary_policies),
            "games": len(series[0]["total_surplus"]),
        },
        "k_tuning": k_tuning(read_json(args.baselines_run, "k_grid.json"), base_m),
        "distilled": distilled_curves(
            read_json(args.distill_run, "distill.json"),
            [f"Planner[{m}]" for m in TRACE_PLANNER_MODELS],
            float(base_m["citizens"]["c3_config"]["k_published"]),
        ),
        "exams": None if args.heldout_eval_run is None else exams(args.heldout_eval_run),
        "b4_c2": None if args.b4_eval_run is None else b4_c2(args.b4_eval_run),
        "b4_rows": None,
        "hypotheses": None,
        "e1": None,
    }
    inputs: dict[str, Any] = {
        "crossplay_run": _input(cp_dir),
        "distill_run": _input(args.distill_run),
        "baselines_run": _input(args.baselines_run),
        "fit_run": _input(fit_run),
        "train_runs": [{**_input(d), "model": m, "seed": s} for (m, s), d in train_dirs.items()],
    }
    for key, path in (
        ("heldout_eval_run", args.heldout_eval_run),
        ("b4_eval_run", args.b4_eval_run),
        ("hypotheses_run", args.hypotheses_run),
        ("e1_run", args.e1_run),
    ):
        if path is not None:
            inputs[key] = _input(path)
    checks_list = [
        Check(
            "trace_totals_reproduce_crossplay",
            worst["total"],
            REPRODUCTION_ATOL,
            "max |re-simulated - recorded| per-game total surplus (coins) over all 1,000 "
            "evaluation games of every traced (governor, citizen model) cell",
        ),
        Check(
            "trace_pool_update_residual",
            worst["pool_update"],
            1e-9,
            "max |R_{t+1} - max(0.01, min(200, R_t - sum e + 1.4 sum c))| over the traced games",
        ),
        Check(
            "trace_offers_above_pool",
            worst["offers_above_pool"],
            1e-9,
            "max (sum e - R) over the traced rounds (0 when every round offers at most the pool)",
        ),
        Check(
            "trace_returns_outside_offer",
            worst["returns_outside_offer"],
            0.0,
            "max over traced decisions of (c - e) and of -c (0 when 0 <= c <= e)",
        ),
    ]
    if args.b4_run:
        rows, b4_traces, b4_worst = b4_runs(args.b4_run)
        demo["b4_rows"] = rows
        demo["traces"] = traces + b4_traces
        inputs["b4_runs"] = [_input(d) for d in args.b4_run]
        checks_list.append(
            Check(
                "b4_totals_match_games",
                b4_worst,
                1e-6,
                "max |sum of parsed (offer - return) - 160 x games.json surplus| per complete B4 game",
            )
        )
    if args.hypotheses_run is not None:
        _require_stage(args.hypotheses_run, "c4_hypotheses")
        demo["hypotheses"] = hypotheses_block(read_json(args.hypotheses_run, "hypotheses.json"))
    if args.e1_run is not None:
        demo["e1"] = e1_block(args.e1_run)
    failed = [c.name for c in checks_list if not c.passed]
    if failed:
        raise ReproductionError(f"checks failed: {failed}")

    eval_key = int(cp_m["citizens"]["seeds"]["eval_key"])
    manifest = stage.base_manifest(run_id, STAGE, fit_run, eval_key, milestone="c4")
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["source"]["generator"] = GENERATOR
    manifest["limitations"] = _limitations(bool(manifest["source"]["git_dirty"]))
    manifest["validation"]["checks"] = [c.to_json() for c in checks_list]
    manifest["validation"]["validated_by"] = (
        f"{GENERATOR} automated checks (pass: observed <= threshold)"
    )
    c = manifest["citizens"]
    c["stage"] = STAGE
    c["inputs"] = inputs
    c["c4_protocol"] = cp_m["citizens"]["c4_protocol"]
    c["seeds"] = {"eval_key": eval_key, "traced_games": list(TRACE_GAMES)}
    c["held_out_access"] = (
        f"data.load_heldout(confirm_frozen=True) under {heldout.protocol.path} "
        f"(sha256 {heldout.protocol.sha256}): {sum(1 for h in human if h['split'] == 'heldout')} "
        "held-out human games copied for display"
    )
    c["human_selection_rule"] = HUMAN_SELECTION_RULE
    c["data_attribution"] = DATA_ATTRIBUTION
    payload = (json.dumps(demo, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    write_run_directory(out_dir, manifest, [(DEMO_KIND, DEMO_FILE, payload)])
    stage.log(
        f"{out_dir}: {len(demo['traces'])} traces, {len(human)} human games, {len(payload)} bytes"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return export_command(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
