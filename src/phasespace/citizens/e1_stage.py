"""Model Citizens E1 stage (exploratory): mechanism-ranking check of the simulators.

Command::

    python -m phasespace.citizens.e1_stage --runs-dir runs --run-id citizens-e1-ranking-v1 \\
        [--audit-run runs/citizens-c1-data-audit-v1] [--g1b-run runs/citizens-c1-g1b-v1] \\
        [--evaluation-run runs/citizens-c2-heldout-eval-v1] \\
        [--b4-eval-run runs/citizens-c2-b4-eval-v1] [--processed-dir data/processed]

Reads stored runs only (each artifact checked against its manifest's SHA-256) and the
processed training games (no held-out game is loaded): no simulation, no LLM call, no
training. The pure parts live in `phasespace.citizens.e1`. Refuses a dirty working tree
(the run records ``git_dirty = false``) and never overwrites a run. Status is always
`computed_unvalidated`: the analysis is exploratory (held-out human outcomes were seen
before it was specified; it is in no frozen protocol; it cannot change a C4 verdict).
Its checks are consistency checks of the inputs; a failing check aborts the stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.citizens import config, data, e1, metrics
from phasespace.citizens.fit import model_block
from phasespace.citizens.records import Games
from phasespace.citizens.split import human_training_games
from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    validate_run_id,
    write_run_directory,
)

GENERATOR: Final = "phasespace.citizens.e1_stage"
STUDY_ID: Final = "model-citizens"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"
DEFAULT_RUN_ID: Final = "citizens-e1-ranking-v1"
INPUT_RUNS: Final = {
    "audit_run": ("citizens-c1-data-audit-v1", "c1_data_audit"),
    "g1b_run": ("citizens-c1-g1b-v1", "c1_g1b"),
    "evaluation_run": ("citizens-c2-heldout-eval-v1", "c2_heldout_eval"),
    "b4_eval_run": ("citizens-c2-b4-eval-v1", "c2_b4_eval"),
}
"""argparse destination -> (default run id under --runs-dir, expected citizens.stage)."""
SIM_MODELS: Final = ("B0", "M0-sim", "B1", "B2", "B3", "B23")
B4_MODELS: Final = {"plain": "B4-plain", "calibrated": "B4-calibrated"}
MODELS: Final = (*SIM_MODELS, *B4_MODELS.values())
CONTROLS: Final = ("B0", "M0-sim")
ENDPOINTS: Final = ("surplus", "gini")
IDENTITY_ATOL: Final = 1e-9
EXPLORATORY: Final = (
    "exploratory: held-out human outcomes were already seen; not part of any frozen "
    "protocol; cannot change a C4 verdict"
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _manifest(run_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    return payload


def _manifest_sha256(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()


def verified_table(run_dir: Path, name: str) -> dict[str, Any]:
    """A sample table of a run, after checking its SHA-256 against the run's manifest."""
    entry = next((a for a in _manifest(run_dir)["artifacts"] if a["path"] == name), None)
    if entry is None:
        raise ValueError(f"{run_dir}: {name} is not listed in the manifest's artifacts")
    payload = (run_dir / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"{run_dir / name}: SHA-256 differs from the manifest")
    table: dict[str, Any] = json.loads(payload)
    return table


def table_dicts(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    cols = [str(c["name"]) for c in table["columns"]]
    return [dict(zip(cols, row, strict=True)) for row in table["rows"]]


def input_record(run_dir: Path, stage: str) -> dict[str, Any]:
    m = _manifest(run_dir)
    got = m.get("citizens", {}).get("stage")
    if got != stage:
        raise ValueError(f"{run_dir}: stage {got!r}, expected {stage!r}")
    if m["source"].get("git_dirty"):
        raise ValueError(f"{run_dir}: made from a dirty working tree")
    return {
        "path": str(run_dir.resolve()),
        "run_id": m["run_id"],
        "manifest_sha256": _manifest_sha256(run_dir),
        "git_commit": m["source"]["git_commit"],
        "status": m["status"],
        "stage": stage,
    }


def simulated_means(
    in_sample: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    b4: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], float]:
    """(model, regime, metric) -> simulated mean: in-sample and gating rows of the
    evaluation run (`sim_mean`, 1,000 games) and the B4 evaluation (`b4_mean`)."""
    out: dict[tuple[str, str, str], float] = {}
    for r in [*in_sample, *heldout]:
        if r["sim_mean"] is not None:
            out[(str(r["model"]), str(r["regime"]), str(r["metric"]))] = float(r["sim_mean"])
    for r in b4:
        key = (B4_MODELS[str(r["variant"])], str(r["regime"]), str(r["metric"]))
        out[key] = float(r["b4_mean"])
    return out


def _endpoint(games: Games, endpoint: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    if endpoint == "surplus":
        return metrics.surplus(games.rewards, games.n_rounds)
    if endpoint == "gini":
        return metrics.reward_gini(games.rewards, games.n_rounds)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def human_means(
    training: Games, diagnostic: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str, str], float]:
    """(mechanism, cohort, metric) -> human mean over games: training mechanisms from the
    training games (Exps 1-2, 40 rounds); held-out mechanisms from the evaluation run's
    per-experiment diagnostic rows ("R-I Exp 2", ...; units are games in Exps 1-3)."""
    out: dict[tuple[str, str, str], float] = {}
    for mechanism in e1.TRAINING_REGIME:
        for cohort in ("Exp 1", "Exp 2"):
            g = training.where(mechanism=mechanism, cohort=cohort)
            if g.n_games:
                for k in ENDPOINTS:
                    out[(mechanism, cohort, k)] = float(_endpoint(g, k).mean())
    regime_mechanism = {v: k for k, v in e1.HELDOUT_REGIME.items()}
    for r in diagnostic:
        regime, _, cohort = str(r["regime"]).partition(" ")
        if regime in regime_mechanism and cohort.startswith("Exp") and r["metric"] in ENDPOINTS:
            out[(regime_mechanism[regime], cohort, str(r["metric"]))] = float(r["human_mean"])
    return out


def _b(x: bool | None) -> str | None:
    return None if x is None else ("true" if x else "false")


PAIR_COLUMNS: Final = [
    ("pair", None, "published comparison (C1 audit or G1b z-value table)"),
    ("source_run", None, "run holding the reproduced z-value"),
    ("cohort", None, "human experiment"),
    ("exp4_game", "count", "Exp 4 game 1-3; null otherwise"),
    ("endpoint", None, "surplus (per player-round) or offer Gini (notebook definitions)"),
    ("higher", None, "mechanism reported higher on the endpoint"),
    ("lower", None, "mechanism reported lower"),
    ("published_z", "dimensionless", "|z| printed in Koster et al. 2025"),
    ("reproduced_z", "dimensionless", "our rank-sum z (games as units)"),
    ("significant", None, "published |z| >= 1.96"),
    ("heldout_pair", None, "either mechanism is held out (Interpolating, M1, M2)"),
    ("human_mean_higher", None, "human mean of the endpoint, higher mechanism, this cohort"),
    ("human_mean_lower", None, "human mean, lower mechanism, this cohort"),
    ("human_difference", None, "human_mean_higher - human_mean_lower"),
    ("model", None, "citizen model (B0 and M0-sim are controls)"),
    ("regime_higher", None, "simulation regime matched to the higher mechanism"),
    ("regime_lower", None, "simulation regime matched to the lower mechanism"),
    ("sim_higher", None, "simulated mean under the higher mechanism's regime"),
    ("sim_lower", None, "simulated mean under the lower mechanism's regime"),
    ("sim_difference", None, "sim_higher - sim_lower"),
    ("status", None, "scored, excluded or not_scorable"),
    ("agrees", None, "scored: sim_difference > 0 (strict; a tie is no agreement)"),
    ("reason", None, "why a pair is excluded or not scorable"),
]
SCORE_COLUMNS: Final = [
    ("model", None, "citizen model (B0 and M0-sim are controls)"),
    ("pairs", "count", "published pairs (C1 audit + G1b)"),
    ("excluded", "count", "pairs involving R-M1 (surrogate not accepted)"),
    ("not_scorable", "count", "pairs without both simulated means (or Exp 4)"),
    ("scorable", "count", "scored pairs"),
    ("agreements", "count", "scored pairs with the human sign"),
    ("ties", "count", "scored pairs with equal simulated means (no agreement)"),
    ("agreement_rate", "dimensionless", "agreements / scorable; null if none"),
    ("significant_scorable", "count", "scored pairs with published |z| >= 1.96"),
    ("significant_agreements", "count", "agreements among them"),
    ("significant_rate", "dimensionless", "significant_agreements / significant_scorable"),
    ("heldout_scorable", "count", "scored pairs involving a held-out mechanism"),
    ("heldout_agreements", "count", "agreements among them"),
    ("heldout_rate", "dimensionless", "heldout_agreements / heldout_scorable"),
]
TAU_COLUMNS: Final = [
    ("cohort", None, "human experiment"),
    ("endpoint", None, "surplus or gini"),
    ("model", None, "citizen model"),
    ("mechanisms", None, "mechanisms compared (sorted, ' | ' separated)"),
    ("human_means", None, "human means in that order"),
    ("sim_means", None, "simulated means in that order; null where a model lacks one"),
    ("tau_b", "dimensionless", "Kendall tau-b, human vs simulated means; null if undefined"),
]


def _round_list(xs: Sequence[float]) -> str:
    return ", ".join(f"{x:.6g}" for x in xs)


def build(
    args: argparse.Namespace, commit: str
) -> tuple[dict[str, Any], list[tuple[str, str, bytes]], list[str]]:
    runs = {k: getattr(args, k) for k in INPUT_RUNS}
    records = {k: input_record(runs[k], INPUT_RUNS[k][1]) for k in INPUT_RUNS}
    ev_manifest = _manifest(runs["evaluation_run"])
    b4_inputs = _manifest(runs["b4_eval_run"])["citizens"]["inputs"]["evaluation_run"]
    if b4_inputs["manifest_sha256"] != records["evaluation_run"]["manifest_sha256"]:
        raise ValueError("the B4 evaluation was not made against this evaluation run")

    pairs = e1.pairs_from_table(verified_table(runs["audit_run"], "z_values.json"))
    pairs += e1.pairs_from_table(verified_table(runs["g1b_run"], "z_values.json"))
    in_sample = table_dicts(verified_table(runs["evaluation_run"], "outcomes_in_sample.json"))
    heldout = table_dicts(verified_table(runs["evaluation_run"], "outcomes_heldout.json"))
    diagnostic = table_dicts(verified_table(runs["evaluation_run"], "outcomes_diagnostic.json"))
    b4 = table_dicts(verified_table(runs["b4_eval_run"], "b4_outcomes.json"))
    sims = simulated_means(in_sample, heldout, b4)
    tier = ev_manifest["citizens"]["closed_loop_tier"]
    accepted = {r: bool(tier[r]) for r in e1.SURROGATE_REGIMES}
    training = human_training_games(data.load_training(args.processed_dir))
    humans = human_means(training, diagnostic)

    # Consistency checks (abort on failure).
    audit_rows = [p for p in pairs if p.source_run == records["audit_run"]["run_id"]]
    z_dev = max(
        abs(
            metrics.rank_sum_z(
                _endpoint(training.where(mechanism=p.higher, cohort=p.cohort), p.endpoint),
                _endpoint(training.where(mechanism=p.lower, cohort=p.cohort), p.endpoint),
            )
            - p.reproduced
        )
        for p in audit_rows
    )
    mean_dev = max(
        abs(humans[(mech, "Exp 1", str(r["metric"]))] - float(r["human_mean"]))
        for r in in_sample
        for mech, regime in e1.TRAINING_REGIME.items()
        if r["regime"] == regime and regime in ("Equal", "Mixed") and r["metric"] in ENDPOINTS
    )
    pooled = {(str(r["model"]), str(r["regime"]), str(r["metric"])): r for r in heldout}
    cohort_dev = 0.0
    for r in diagnostic:
        regime = str(r["regime"]).split(" ")[0]
        ref = pooled.get((str(r["model"]), regime, str(r["metric"])))
        if ref is not None and r["sim_mean"] is not None and str(r["regime"]) != "M2 Exp 4":
            cohort_dev = max(cohort_dev, abs(float(r["sim_mean"]) - float(ref["sim_mean"])))
    checks = [
        Check("input_checksums_verified", 0.0, 0.0,
              "every input table loaded by its manifest SHA-256 (a mismatch aborts); the B4 "
              "evaluation's evaluation-run manifest SHA-256 equals the evaluation run's"),
        Check("audit_z_recomputed_max_abs", z_dev, IDENTITY_ATOL,
              "max |rank-sum z recomputed from the processed training games - the C1 "
              "audit's reproduced z| over its four Exp 1 pairs (same games and definitions)"),
        Check("human_means_vs_evaluation_run_max_abs", mean_dev, IDENTITY_ATOL,
              "max |Exp 1 Equal and Mixed human means from the training games - the "
              "evaluation run's in-sample human_mean| (surplus, gini)"),
        Check("simulation_cohort_invariance_max_abs", cohort_dev, IDENTITY_ATOL,
              "max |per-experiment diagnostic sim_mean - pooled gating sim_mean| (R-I Exp 2/3, "
              "R-M1 Exp 1/2, R-M2 Exp 3; all models and metrics): the simulator does not "
              "depend on the cohort, so a regime's simulation stands for each of its cohorts"),
    ]  # fmt: skip
    failed = [c.name for c in checks if not c.passed]
    if failed:
        raise ValueError(f"E1 consistency checks failed: {failed}")

    matches = [e1.match_pair(p, m, sims, accepted) for p in pairs for m in MODELS]
    scores = [e1.score(m, matches) for m in MODELS]

    pair_rows: list[list[Any]] = []
    for mt in matches:
        p = mt.pair
        hh = humans.get((p.higher, p.cohort, p.endpoint)) if p.exp4_game is None else None
        hl = humans.get((p.lower, p.cohort, p.endpoint)) if p.exp4_game is None else None
        pair_rows.append([
            p.name, p.source_run, p.cohort, p.exp4_game, p.endpoint, p.higher, p.lower,
            p.published, p.reproduced, _b(p.significant), _b(p.heldout), hh, hl,
            None if hh is None or hl is None else hh - hl,
            mt.model, mt.regime_higher, mt.regime_lower, mt.sim_higher, mt.sim_lower,
            mt.sim_difference, mt.status, _b(mt.agrees), mt.reason or None,
        ])  # fmt: skip
    score_rows = [
        [s.model, s.pairs, s.excluded, s.not_scorable, s.scorable, s.agreements, s.ties,
         s.rate, s.significant_scorable, s.significant_agreements, s.significant_rate,
         s.heldout_scorable, s.heldout_agreements, s.heldout_rate]
        for s in scores
    ]  # fmt: skip

    usable = [mech for mech in (*e1.TRAINING_REGIME, *e1.HELDOUT_REGIME)
              if e1.regime_of(mech) not in e1.SURROGATE_REGIMES
              or accepted[e1.regime_of(mech)]]  # fmt: skip
    groups = e1.ranking_groups(pairs, usable)
    tau_rows: list[list[Any]] = []
    group_records: list[dict[str, Any]] = []
    for (cohort, endpoint), mechs in groups.items():
        hm = [humans[(mech, cohort, endpoint)] for mech in mechs]
        human_z: dict[str, float] = {}
        if all(mech in e1.TRAINING_REGIME for mech in mechs):
            for i, a in enumerate(mechs):
                for b in mechs[i + 1 :]:
                    human_z[f"{a} vs {b}"] = metrics.rank_sum_z(
                        _endpoint(training.where(mechanism=a, cohort=cohort), endpoint),
                        _endpoint(training.where(mechanism=b, cohort=cohort), endpoint),
                    )
        group_records.append({
            "cohort": cohort, "endpoint": endpoint, "mechanisms": mechs, "human_means": hm,
            "human_rank_sum_z_all_pairs": human_z,
            "note": "human ordering by game means of this cohort; pairs not in the "
                    "published list are included with their rank-sum z (may be non-significant)",
        })  # fmt: skip
        for model in MODELS:
            sm = [sims.get((model, e1.regime_of(mech), endpoint)) for mech in mechs]
            tau = None
            if all(v is not None for v in sm):
                tau = e1.kendall_tau_b(hm, [float(v) for v in sm if v is not None])
            tau_rows.append([
                cohort, endpoint, model, " | ".join(mechs), _round_list(hm),
                None if any(v is None for v in sm)
                else _round_list([float(v) for v in sm if v is not None]),
                tau,
            ])  # fmt: skip

    excluded = [p.name for p in pairs if e1.match_pair(p, "", {}, accepted).status == "excluded"]
    exp4 = [p.name for p in pairs if p.exp4_game is not None]
    max_scorable = max(s.scorable for s in scores)
    max_in_sample = max(s.scorable - s.heldout_scorable for s in scores)
    run_id = args.run_id
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": "citizens-e1-ranking-v1",
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": {
            "repository": REPOSITORY,
            "git_commit": commit,
            "git_dirty": False,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "environment": environment_info(),
        "model": model_block(),
        "simulator": None,
        "random_seed": None,
        "artifacts": [],
        "metrics": {},
        "limitations": [
            "Exploratory (E1): held-out human outcomes were already seen; not part of any "
            "frozen protocol; it cannot change a C2 label or a C4 verdict.",
            f"Few pairs: at most {max_scorable} scorable published pairs per model "
            f"({max_in_sample} of them in-sample, between training mechanisms the models "
            "were fitted on); every published pair is significant (|z| >= 1.96), so "
            "'significant only' equals 'all'.",
            "The human side is a rank-sum direction; the simulated side compares means "
            "(1,000 simulated games per model and regime; about 40 games for B4).",
            "Pairs involving M1 are excluded: the M1 mechanism surrogate was not accepted "
            "on human rows (closed_loop_tier R-M1 = false), so no closed-loop M1 "
            "simulation is valid. Exp 4 pairs are not scorable (game-specific lengths).",
        ],
        "validation": {
            "checks": [c.to_json() for c in checks],
            "validated_by": f"{GENERATOR} consistency checks (pass: observed <= threshold); "
            "status stays computed_unvalidated (exploratory)",
            "validated_at": timestamp,
        },
        "citizens": {
            "stage": "e1_ranking",
            "exploratory": EXPLORATORY,
            "frozen_protocol": None,
            "can_change_c4_verdict": False,
            "c2_config_sha256": config.config_sha256(),
            "inputs": records,
            "dataset": {"training": data.processed_sidecar(args.processed_dir, data.TRAINING_FILE)},
            "held_out_access": "no held-out game is loaded; held-out human per-experiment "
            "means are read from the evaluation run's outcomes_diagnostic.json",
            "surrogate_accepted": accepted,
            "models": {
                "simulated": list(SIM_MODELS),
                "b4": list(B4_MODELS.values()),
                "controls": list(CONTROLS),
            },
            "matching": [
                "Training mechanisms (Equal, Mixed, Proportional Baseline) -> the evaluation "
                "run's in-sample simulation regimes Equal, Mixed, Proportional; held-out "
                "Interpolating -> R-I, M2 -> R-M2 (accepted surrogate), M1 -> R-M1.",
                "Cohort: the simulator is cohort-agnostic (40 rounds, same start, seeds per "
                "model and regime only; checked: per-experiment diagnostic sim means equal "
                "the pooled ones), so each 40-round cohort (Exps 1-3) uses its regime's "
                "simulation; e.g. Exp 2 Interpolating vs Proportional = R-I vs Proportional, "
                "Exp 3 M2 vs Interpolating = R-M2 vs R-I.",
                "Metric: the pair's endpoint (surplus per player-round, offer Gini) with the "
                "same notebook definitions on both sides; depletion round and players "
                "active appear in no published pair.",
                "B4 (plain, calibrated) exists only for the gating regimes R-I, R-M1, R-M2: "
                "only pairs between two of them are scorable (Exp 3 M2 vs Interpolating). "
                "The pre-freeze B4 pilot and calibration runs (Proportional) are not used: "
                "dirty trees, different sampling (calibration compositions), pre-freeze.",
                "Human magnitudes: game means of the cohort (training games for training "
                "mechanisms; the evaluation run's per-experiment rows for held-out ones).",
            ],
            "exclusions": [
                f"Excluded ({len(excluded)} pairs: {', '.join(excluded)}): every pair with "
                "RL Agent (M1), because the M1 surrogate failed human acceptance.",
                f"Not scorable ({len(exp4)} pairs: {', '.join(exp4)}): Exp 4 M2 vs "
                "Proportional per game; no stored simulation of Proportional under Exp 4's "
                "recorded lengths, and the M2 Exp 4 diagnostic pools games 1-3.",
                "Not scorable for B4: every pair with a training mechanism (no B4 "
                "evaluation under Equal, Mixed or Proportional).",
            ],
            "definitions": {
                "agreement": "sim mean under the human-higher mechanism > under the "
                "human-lower one (strict; ties counted, never agreements)",
                "significant": f"published |z| >= {e1.Z_CRITICAL}",
                "kendall_tau": "tau-b between human and simulated means of the mechanisms "
                "of one (experiment, endpoint) with >= 3 mechanisms in non-excluded pairs",
            },
            "ranking_groups": group_records,
        },
    }
    for s in scores:
        slug = s.model.lower().replace("-", "_")
        manifest["metrics"][f"scorable_pairs_{slug}"] = {
            "value": s.scorable, "unit": "count",
            "definition": f"published pairs scored for {s.model}"}  # fmt: skip
        if s.rate is not None:
            manifest["metrics"][f"agreement_rate_{slug}"] = {
                "value": s.rate, "unit": "dimensionless",
                "definition": f"sign agreements / scorable pairs, {s.model}"}  # fmt: skip
    files = [
        ("sample_table", "e1_pairs.json", _table(run_id, PAIR_COLUMNS, pair_rows)),
        ("sample_table", "e1_scores.json", _table(run_id, SCORE_COLUMNS, score_rows)),
        ("sample_table", "e1_tau.json", _table(run_id, TAU_COLUMNS, tau_rows)),
    ]
    summary = [
        f"{s.model:14s} scorable {s.scorable:2d} agree {s.agreements:2d} ties {s.ties} "
        f"rate {s.rate} sig {s.significant_agreements}/{s.significant_scorable} "
        f"held-out {s.heldout_agreements}/{s.heldout_scorable}"
        for s in scores
    ] + [f"tau {r[0]} {r[1]} {r[2]}: {r[6]}" for r in tau_rows]
    return manifest, files, summary


def _table(
    run_id: str, columns: Sequence[tuple[str, str | None, str]], rows: list[list[Any]]
) -> bytes:
    return dumps_json(
        {
            "schema_version": 1,
            "run_id": run_id,
            "columns": [{"name": n, "unit": u, "definition": d} for n, u, d in columns],
            "rows": rows,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.e1_stage")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    for dest in INPUT_RUNS:
        parser.add_argument(f"--{dest.replace('_', '-')}", dest=dest, type=Path, default=None)
    args = parser.parse_args(argv)
    for dest, (run_id, _) in INPUT_RUNS.items():
        if getattr(args, dest) is None:
            setattr(args, dest, args.runs_dir / run_id)
    validate_run_id(args.run_id)
    out_dir = args.runs_dir / args.run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    commit, dirty = git_source()
    if dirty:
        raise SystemExit(
            "the working tree has uncommitted or untracked changes; E1 runs are made from a "
            "clean tree only (git_dirty = false)"
        )
    manifest, files, summary = build(args, commit)
    write_run_directory(out_dir, manifest, files)
    for line in summary:
        _log(line)
    print(f"{out_dir}: status {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
