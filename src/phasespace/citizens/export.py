"""Export the C1 data audit of Model Citizens as an immutable run (schema v1).

Usage::

    uv run python -m phasespace.citizens.export [--runs-dir runs] [--run-id <run_id>]

Reads the parsed dataset (`phasespace.citizens.data`), runs every C1 check
(`phasespace.citizens.validation`) and writes `<runs-dir>/<run-id>/` with
`manifest.json` and four sample tables:

- `game_counts.json`: games per (mechanism, cohort, Exp 4 game) and split; rows for
  training games only;
- `round_lengths.json`: the distribution of game lengths of the training games;
- `identity_checks.json`: residual summaries of every identity check, including the
  held-out human games (residual statistics and pass/fail counts only);
- `z_values.json`: the published Exp 1 training-mechanism z-values and their
  reproduction.

Gate scope (D33): the run's `validation.checks` are the G1a checks (training
populations, mechanism-offer reproduction including the held-out Interpolating rows as
an identity, and the four training-only Exp 1 z-values). The held-out pool-transition
identity is the deferred G1b item `heldout_pool_anomaly`, recorded with its documented
bound in `citizens.gate.deferred`. The run is `validated` only if every G1a check passes
and every deferred item lies within its bound, otherwise `computed_unvalidated`; see
the Model Citizens addendum of `docs/data-contract.md`. Held-out human games enter only
through `data.heldout_identity_residuals` and `data.heldout_game_counts`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.citizens import data, validation
from phasespace.citizens.game import (
    DEPLETION_THRESHOLD,
    EXCLUSION_THRESHOLD,
    GROWTH_FACTOR,
    MAX_POOL,
    N_PLAYERS,
    N_ROUNDS,
    POOL_FLOOR,
)
from phasespace.citizens.mechanisms import (
    DEFAULT_OFFER_RULE,
    INTERPOLATING_DECIMALS,
    K_INTERPOLATING,
    RECORDED_DECIMALS,
    RECORDED_WITHHELD,
    W_EQUAL,
    W_MIXED,
    W_PROPORTIONAL,
)
from phasespace.citizens.records import Games
from phasespace.citizens.source import (
    SOURCE_CITATION,
    SOURCE_LICENSE,
    SOURCE_MD5_BASE64,
    SOURCE_SIZE_BYTES,
    SOURCE_URL,
)
from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    run_status,
    validate_run_id,
    write_run_directory,
)

SCHEMA_VERSION: Final = 1
STUDY_ID: Final = "model-citizens"
EXPERIMENT_ID: Final = "citizens-c1-data-audit-v1"
DEFAULT_RUN_ID: Final = "citizens-c1-data-audit-v1"
DEFAULT_RUNS_DIR: Final = Path("runs")
GATE: Final = "G1a"
COIN: Final = "coin"
GENERATOR: Final = "phasespace.citizens.export"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"
NOTEBOOK_URL: Final = (
    "https://github.com/google-deepmind/sustainable_behavior/blob/main/notebooks/"
    "sustainable_behavior.ipynb"
)

DESIGN_GAMES: Final = {
    ("Equal Baseline", "Exp 1", 0): 40,
    ("Mixed Baseline", "Exp 1", 0): 40,
    ("Proportional Baseline", "Exp 1", 0): 40,
    ("RL Agent (M1)", "Exp 1", 0): 40,
    ("Proportional Baseline", "Exp 2", 0): 40,
    ("Interpolating Baseline", "Exp 2", 0): 40,
    ("RL Agent (M1)", "Exp 2", 0): 40,
    ("Interpolating Baseline", "Exp 3", 0): 80,
    ("RL Agent (M2)", "Exp 3", 0): 80,
    **{("Proportional Baseline", "Exp 4", k): 40 for k in (1, 2, 3)},
    **{("RL Agent (M2)", "Exp 4", k): 40 for k in (1, 2, 3)},
}
"""Analysed human groups per mechanism and experiment (Koster et al. 2025, Results and
Methods: Exps 1-2 40 groups per mechanism, Exp 3 80, Exp 4 three consecutive games)."""


@dataclass(frozen=True, slots=True)
class Audit:
    """Everything the C1 data-audit run records."""

    training: Games
    training_sidecar: dict[str, Any]
    heldout_sidecar: dict[str, Any]
    heldout_counts: dict[tuple[str, str, int], int]
    identity: list[validation.IdentityResult]
    z_values: list[validation.ZComparison]
    diagnostics: dict[str, Any]


def compute_audit(processed_dir: Path = data.DEFAULT_PROCESSED_DIR) -> Audit:
    training = data.load_training(processed_dir)
    human = training.select(training.is_human)
    bc = training.select(~training.is_human)
    identity: list[validation.IdentityResult] = []
    for population, games in (("training_human", human), ("bc", bc)):
        identity.append(validation.pool_transition(games, population))
        identity.extend(validation.offer_reproduction(games, population))
        identity.extend(validation.record_consistency(games, population))
    identity.append(validation.pool_record_consistency(human, "training_human"))
    identity.extend(data.heldout_identity_residuals(processed_dir))
    diagnostics: dict[str, Any] = {
        "pool_regimes_training_human": validation.pool_regimes(human),
        "pool_regimes_bc": validation.pool_regimes(bc),
        "legacy_remainder_dropped_launch_ids": sorted(validation.LEGACY_REMAINDER_DROPPED),
        "interpolating_weight_laws_bc": validation.interpolating_alternatives(bc),
        "withheld_alternatives_training": validation.withheld_alternatives(training),
        "post_depletion_training_human": validation.post_depletion(human),
        "depletion_absorbing": validation.depletion_absorbing(training),
        "zero_previous_sum_training_human": validation.zero_previous_sum(human),
        "human_flags_training": validation.human_flags(human),
    }
    return Audit(
        training=training,
        training_sidecar=data.processed_sidecar(processed_dir, data.TRAINING_FILE),
        heldout_sidecar=data.processed_sidecar(processed_dir, data.HELDOUT_FILE),
        heldout_counts=data.heldout_game_counts(processed_dir),
        identity=identity,
        z_values=validation.published_z(training),
        diagnostics=diagnostics,
    )


# --- tables ---------------------------------------------------------------------------


def _table(run_id: str, columns: list[tuple[str, str | None, str]], rows: list[list[Any]]) -> bytes:
    return dumps_json(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "columns": [{"name": n, "unit": u, "definition": d} for n, u, d in columns],
            "rows": rows,
        }
    )


def _training_groups(training: Games) -> dict[tuple[str, str, int], Games]:
    keys = sorted(
        set(
            zip(
                training.mechanism.tolist(),
                training.cohort.tolist(),
                training.exp4_game.tolist(),
                strict=True,
            )
        )
    )
    return {(m, c, k): training.where(mechanism=m, cohort=c, exp4_game=k) for (m, c, k) in keys}


def game_counts_table(audit: Audit, run_id: str) -> bytes:
    rows: list[list[Any]] = []
    for (m, c, k), g in _training_groups(audit.training).items():
        rows.append([m, c, k, "training", g.n_games, int(g.n_rounds.sum())])
    for (m, c, k), n in audit.heldout_counts.items():
        rows.append([m, c, k, "heldout", n, None])
    columns: list[tuple[str, str | None, str]] = [
        ("mechanism", None, "base mechanism (mech_name_by_player without ' Game k' and cohort)"),
        ("cohort", None, "Exp 1..Exp 4 (human groups) or BC 1/BC 2 (simulated clone games)"),
        ("exp4_game", "count", "1..3 for the consecutive Exp 4 games, 0 otherwise"),
        ("split", None, "training, or heldout (human Interpolating/M1/M2 games)"),
        ("games", "count", "games = distinct (mech_name_by_player, launch_id)"),
        ("rows", "count", "game rounds (CSV rows); null for held-out games (not reported)"),
    ]
    return _table(run_id, columns, rows)


def round_lengths_table(audit: Audit, run_id: str) -> bytes:
    rows: list[list[Any]] = []
    for (m, c, k), g in _training_groups(audit.training).items():
        lengths, counts = np.unique(g.n_rounds, return_counts=True)
        for n, count in zip(lengths.tolist(), counts.tolist(), strict=True):
            rows.append([m, c, k, int(n), int(count)])
    columns: list[tuple[str, str | None, str]] = [
        ("mechanism", None, "base mechanism"),
        ("cohort", None, "cohort"),
        ("exp4_game", "count", "1..3 for Exp 4, else 0"),
        ("n_rounds", "count", "rounds recorded for the game (rows, round_id 0..n-1)"),
        ("games", "count", "training games with this length"),
    ]
    return _table(run_id, columns, rows)


def identity_table(audit: Audit, run_id: str) -> bytes:
    rows: list[list[Any]] = [
        [
            r.name,
            r.population,
            r.entries,
            None if not np.isfinite(r.max_abs_residual) else r.max_abs_residual,
            r.threshold,
            r.n_fail,
            "true" if r.passed else "false",
            json.dumps(dict(r.details), sort_keys=True),
        ]
        for r in audit.identity
    ]
    columns: list[tuple[str, str | None, str]] = [
        ("check", None, "identity checked (phasespace.citizens.validation)"),
        ("population", None, "training_human, bc, or heldout_human[:mechanism:cohort]"),
        ("entries", "count", "rows (pool) or player-entries (offers, records) checked"),
        ("max_abs_residual", COIN, "largest |recorded - rule|; null if nothing was checked"),
        ("threshold", COIN, "tolerance from the CSV's print precision"),
        ("n_fail", "count", "entries outside the threshold under every accepted rule"),
        ("passed", None, "entries > 0 and n_fail == 0"),
        ("details", None, "JSON object of further counts and residual maxima"),
    ]
    return _table(run_id, columns, rows)


def z_table(audit: Audit, run_id: str) -> bytes:
    rows: list[list[Any]] = [
        [
            z.name,
            z.endpoint,
            z.cohort,
            z.higher,
            z.lower,
            z.n_higher,
            z.n_lower,
            z.published,
            z.reproduced,
            z.abs_difference,
        ]
        for z in audit.z_values
    ]
    columns: list[tuple[str, str | None, str]] = [
        ("comparison", None, "published comparison (Exp 1, training mechanisms only)"),
        ("endpoint", None, "surplus or gini, notebook definitions (phasespace.citizens.metrics)"),
        ("cohort", None, "human experiment"),
        ("higher", None, "mechanism reported higher"),
        ("lower", None, "mechanism reported lower"),
        ("n_higher", "count", "games (groups) of the higher mechanism"),
        ("n_lower", "count", "games (groups) of the lower mechanism"),
        ("published", "dimensionless", "|z| printed in Koster et al. 2025 (two decimals)"),
        ("reproduced", "dimensionless", "scipy.stats.ranksums(higher, lower).statistic"),
        ("abs_difference", "dimensionless", "|reproduced - published|"),
    ]
    return _table(run_id, columns, rows)


# --- checks and manifest ---------------------------------------------------------------


def design_count_difference(audit: Audit) -> int:
    """Sum over human (mechanism, cohort, game) cells of |games - published design|."""
    counts = dict(audit.heldout_counts)
    for (m, c, k), g in _training_groups(audit.training).items():
        if c.startswith("Exp"):
            counts[(m, c, k)] = g.n_games
    keys = set(counts) | set(DESIGN_GAMES)
    return sum(abs(counts.get(key, 0) - DESIGN_GAMES.get(key, 0)) for key in keys)


def _is_heldout_pool(result: validation.IdentityResult) -> bool:
    return result.name == "pool_transition" and result.population.startswith(
        data.HELDOUT_POPULATION
    )


def heldout_pool_anomaly(audit: Audit) -> dict[str, Any]:
    """The deferred G1b item: held-out pool-transition failures against their bound.

    Reads only failure counts and the residual maximum of the held-out pool identity.
    """
    pool = [r for r in audit.identity if _is_heldout_pool(r)]
    allowed = (
        data.HELDOUT_POPULATION,
        f"{data.HELDOUT_POPULATION}:{validation.HELDOUT_POOL_ANOMALY_POPULATION}",
    )
    outside = sorted(r.population for r in pool if not r.passed and r.population not in allowed)
    total = next(r for r in pool if r.population == data.HELDOUT_POPULATION)
    games_fail = int(total.details["games_fail"])
    within = (
        not outside
        and total.n_fail <= validation.HELDOUT_POOL_ANOMALY_MAX_ROWS
        and games_fail <= validation.HELDOUT_POOL_ANOMALY_MAX_GAMES
        and total.max_abs_residual <= validation.HELDOUT_POOL_ANOMALY_MAX_RESIDUAL
    )
    return {
        "name": "heldout_pool_anomaly",
        "gate": "G1b",
        "definition": "Held-out human pool rows outside POOL_TOLERANCE of the full rule. "
        "Documented and bounded before diagnosis (D33); root-caused in G1b after the "
        "validation protocol freeze. Not an accepted rule.",
        "observed": {
            "failing_rows": total.n_fail,
            "failing_games": games_fail,
            "max_abs_residual_coin": total.max_abs_residual,
            "failing_populations_outside_bound": outside,
        },
        "bound": {
            "population": validation.HELDOUT_POOL_ANOMALY_POPULATION,
            "max_rows": validation.HELDOUT_POOL_ANOMALY_MAX_ROWS,
            "max_games": validation.HELDOUT_POOL_ANOMALY_MAX_GAMES,
            "max_abs_residual_coin": validation.HELDOUT_POOL_ANOMALY_MAX_RESIDUAL,
        },
        "within_bound": within,
    }


def gate_status(run_checks: Sequence[Check], deferred: Sequence[dict[str, Any]]) -> str:
    """'validated' iff every G1a check passes and every deferred item is within its bound."""
    if run_status(run_checks) == "validated" and all(d["within_bound"] for d in deferred):
        return "validated"
    return "computed_unvalidated"


def checks(audit: Audit) -> list[Check]:
    """The G1a checks (held-out pool transitions are the deferred G1b item instead)."""
    parse = audit.training_sidecar["audit"]
    out = [
        Check(
            "source_md5_mismatch",
            0.0 if audit.training_sidecar["source_md5_base64"] == SOURCE_MD5_BASE64 else 1.0,
            0.0,
            "1 if the parsed CSV's base64 MD5 differs from the published GCS hash, else 0",
        ),
        Check(
            "game_identity_violations",
            float(parse["duplicate_game_keys"] + parse["non_contiguous_games"]),
            0.0,
            "(mech_name_by_player, launch_id) keys seen in two separate row blocks plus games "
            "whose round_id is not exactly 0..n-1 in file order",
        ),
        Check(
            "design_game_counts_abs_difference",
            float(design_count_difference(audit)),
            0.0,
            "sum over human (mechanism, cohort, Exp 4 game) cells of |games in the CSV - "
            "groups reported by Koster et al. 2025| (Exps 1-2: 40, Exp 3: 80, Exp 4: 40 "
            "per game)",
        ),
    ]
    for r in audit.identity:
        if r.entries == 0 or _is_heldout_pool(r):
            continue
        slug = r.population.lower().replace(" ", "_")
        for char in "():":
            slug = slug.replace(char, "_" if char == ":" else "")
        out.append(
            Check(
                f"{r.name}_{slug}",
                r.max_abs_residual,
                r.threshold,
                f"max |recorded - rule| over {r.entries} entries of {r.population} "
                f"({r.n_fail} outside the threshold); see identity_checks.json",
            )
        )
    for z in audit.z_values:
        out.append(
            Check(
                f"z_{z.name}",
                z.abs_difference,
                validation.Z_TOLERANCE,
                f"|ranksums({z.higher}, {z.lower}) {z.endpoint} z in {z.cohort} - published "
                f"{z.published}| (published to two decimals)",
            )
        )
    return out


def _metric(value: float, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


RULES: Final = (
    {
        "name": "game_identity",
        "rule": "A game is (mech_name_by_player, launch_id); its rows are consecutive with "
        "round_id 0..n-1. Exp 4 launch_ids repeat across a group's Game 1-3.",
    },
    {
        "name": "pool_update",
        "rule": "R' = max(0.01, min(200, R - sum e + 1.4 sum c)). The 0.01 floor is not in "
        "the paper; every recorded pool is >= 0.01.",
    },
    {
        "name": "legacy_remainder_dropped",
        "rule": "A frozen list of early Exp 1 Equal/Proportional training games "
        "(diagnostics.legacy_remainder_dropped_launch_ids, model."
        "legacy_remainder_dropped_games; found in training data only) "
        "drops the undistributed remainder R - sum e (0.0008-0.0012 coins): R' = max(0.01, "
        "min(200, 1.4 sum c)) in every row of those games (the two rules coincide at the "
        "floor and the cap). Every other game follows the full rule; rules are assigned "
        "per game, never chosen per row. No held-out game carries a listed launch_id.",
    },
    {
        "name": "deployed_offers",
        "rule": "Deployed baseline offers are round(share_i * (R - 0.001), 4) in float32, "
        "share_i = w/4 + (1 - w) c_i / sum c; about 0.001 coins stays in the pool each "
        "round independent of R. Withholding 0, 0.0005, 0.0015 or 0.002 does not "
        "reproduce the offers (diagnostics.withheld_alternatives_training). The paper's "
        "idealised rule (whole pool) differs by <= 0.0011. Simulations (C2-C4) use the "
        "deployed rule (model.offer_rule).",
    },
    {
        "name": "zero_previous_sum",
        "rule": "In the human training games sum c_{t-1} = 0 occurs only with the pool at "
        "its 0.01 floor (diagnostics.zero_previous_sum_training_human); the offer is "
        "then the equal split (0.0022 each). Implemented as an equal split.",
    },
    {
        "name": "interpolating_weight",
        "rule": "w = round((R/200)^22, 1), w in {0, 0.1, ..., 1}; reproduces every BC and "
        "held-out human Interpolating offer. The unquantised weight, floor rounding, "
        "k = 21, 23 or e^3.1 do not (see diagnostics).",
    },
    {
        "name": "distribute_pool_ratio",
        "rule": "1.0 in every human training row, empty in BC rows: it never varies, so "
        "its effect is not identifiable; the 0.001 withholding is absolute, not a ratio.",
    },
    {
        "name": "depletion",
        "rule": "Depleted = pool before allocation < 1. In the human training games it is "
        "absorbing (diagnostics.depletion_absorbing): offers are then < 1, which forces "
        "c = 0, so the pool stays at the 0.01 floor. BC clones reciprocate continuous "
        "amounts even on offers < 1, so a few BC games recover.",
    },
    {
        "name": "post_depletion_rows",
        "rule": "Training human games (Exp 1, 2 and 4 baselines) keep recording rows "
        "after depletion until the game ends (40 rows in Exps 1-2, the lottery length in "
        "Exp 4): pool 0.01, offers 0.0022, c = 0, game_has_no_pool True "
        "(diagnostics.post_depletion_training_human). The notebook's metrics include them; "
        "so do ours.",
    },
    {
        "name": "game_length",
        "rule": "Exps 1-3 and BC games have 40 rounds; Exp 4 games 25-49. The notebook's "
        "depletion round defaults to 40 for an undepleted game whatever its length.",
    },
    {
        "name": "filters",
        "rule": "The notebook applies no filters; every human training row has "
        "four_usable_players, completed and first_x_games True.",
    },
    {
        "name": "heldout_pool_anomaly",
        "rule": "Residual-only finding, deferred to G1b (citizens.gate.deferred): held-out "
        "pool rows outside tolerance of the full rule occur only in "
        f"{validation.HELDOUT_POOL_ANOMALY_POPULATION.replace(':', ' ')} (at most "
        f"{validation.HELDOUT_POOL_ANOMALY_MAX_ROWS} rows in "
        f"{validation.HELDOUT_POOL_ANOMALY_MAX_GAMES} games, |residual| <= "
        f"{validation.HELDOUT_POOL_ANOMALY_MAX_RESIDUAL:.4f} coins). Unexplained; not "
        "accepted as a rule; nothing else about those rows is inspected before the "
        "validation protocol freeze.",
    },
)


def build_documents(
    audit: Audit,
    run_id: str,
    *,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
) -> tuple[dict[str, Any], list[tuple[str, str, bytes]]]:
    """Return (manifest without artifacts, [(kind, path, bytes)])."""
    timestamp = created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_checks = checks(audit)
    training = audit.training
    human = training.is_human
    parse = audit.training_sidecar["audit"]
    limitations = [
        "Data audit of a published dataset: 'validated' means the recorded identity and "
        "reproduction checks passed; it is not a new behavioural result.",
        "Held-out human games (Interpolating, M1, M2 in Exps 1-4) were touched only by the "
        "pool-transition and offer-reproduction identities; only residual statistics, "
        "pass/fail counts and game counts are reported. No behavioural quantity of a "
        "held-out game is computed and their game lengths are not tabulated (the identity "
        "table's entry counts are their row totals per mechanism and cohort; Exp 4 lengths "
        "follow the exogenous continuation lottery).",
        "Gate scope G1a only (D33): 'validated' covers the training-population checks, "
        "the offer identities (including held-out Interpolating rows) and the four "
        "training-only Exp 1 z-values. Published z-values involving held-out mechanisms "
        "and the root cause of the held-out pool anomaly (citizens.gate.deferred) are G1b, "
        "after the validation protocol freeze; G1 as a whole is not yet passed.",
        "M1/M2 offers cannot be reproduced (the policies' weights are not released); for "
        "their rows only the pool transition is checked.",
        "Two pool rules are accepted, assigned per game: remainder-dropped for the "
        f"{len(validation.LEGACY_REMAINDER_DROPPED)} listed early Exp 1 games, full for every "
        "other game; simulations use the full rule.",
    ]
    if git_dirty:
        limitations.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    deferred = [heldout_pool_anomaly(audit)]
    metrics = {
        "csv_rows": _metric(parse["total_rows"], "count", "data rows of the CSV"),
        "games": _metric(parse["games"], "count", "games in the CSV (all cohorts)"),
        "training_human_games": _metric(
            int(np.count_nonzero(human)), "count", "human Equal/Mixed/Proportional games"
        ),
        "training_human_rows": _metric(
            int(training.n_rounds[human].sum()), "count", "rows of the human training games"
        ),
        "bc_games": _metric(int(np.count_nonzero(~human)), "count", "BC 1 and BC 2 games"),
        "heldout_human_games": _metric(
            sum(audit.heldout_counts.values()), "count", "human Interpolating/M1/M2 games"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": gate_status(run_checks, deferred),
        "created_at": timestamp,
        "source": {
            "repository": REPOSITORY,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "environment": environment_info(),
        "model": {
            "name": "koster2025-common-pool",
            "units": COIN,
            "players": N_PLAYERS,
            "max_pool": MAX_POOL,
            "growth_factor": GROWTH_FACTOR,
            "pool_floor": POOL_FLOOR,
            "depletion_threshold": DEPLETION_THRESHOLD,
            "exclusion_threshold": EXCLUSION_THRESHOLD,
            "rounds_exps_1_3": N_ROUNDS,
            "baseline_weights": {
                "equal": W_EQUAL,
                "mixed": W_MIXED,
                "proportional": W_PROPORTIONAL,
            },
            "interpolating_k": K_INTERPOLATING,
            "interpolating_weight_decimals": INTERPOLATING_DECIMALS,
            "recorded_offer_withheld": RECORDED_WITHHELD,
            "recorded_offer_decimals": RECORDED_DECIMALS,
            "offer_rule": DEFAULT_OFFER_RULE,
            "legacy_remainder_dropped_games": len(validation.LEGACY_REMAINDER_DROPPED),
        },
        "simulator": None,
        "random_seed": None,
        "artifacts": [],
        "metrics": metrics,
        "limitations": limitations,
        "validation": {
            "checks": [c.to_json() for c in run_checks],
            "validated_by": f"{GENERATOR} automated checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
        "citizens": {
            "stage": "c1_data_audit",
            "gate": {
                "name": GATE,
                "scope": "training-population identities (human Equal/Mixed/Proportional "
                "and all BC games), mechanism-offer reproduction including held-out "
                "Interpolating rows as an identity, the four training-only Exp 1 z-values",
                "status_semantics": "validated = every check in validation.checks (the G1a "
                "scope) passed and every deferred item is within its bound; it says "
                "nothing more about deferred items",
                "deferred": deferred,
                "deferred_to_g1b": [
                    "published z-values involving held-out mechanisms",
                    "root cause of the held-out pool anomaly",
                ],
            },
            "dataset": {
                "url": SOURCE_URL,
                "size_bytes": SOURCE_SIZE_BYTES,
                "md5_base64": SOURCE_MD5_BASE64,
                "sha256": audit.training_sidecar["source_sha256"],
                "license": SOURCE_LICENSE,
                "citation": SOURCE_CITATION,
                "notebook": NOTEBOOK_URL,
            },
            "parser_version": audit.training_sidecar["parser_version"],
            "processed": {
                "training_sha256": audit.training_sidecar["file_sha256"],
                "heldout_sha256": audit.heldout_sidecar["file_sha256"],
            },
            "tolerances": {
                "pool": validation.POOL_TOLERANCE,
                "offer": validation.OFFER_TOLERANCE,
                "offer_exact": validation.EXACT_OFFER,
                "record": validation.RECORD_TOLERANCE,
                "z": validation.Z_TOLERANCE,
            },
            "validation_protocol": {
                "path": data.VALIDATION_PROTOCOL_PATH.as_posix(),
                "held_out_access": "identity residuals and game counts only",
            },
            "rules": list(RULES),
            "diagnostics": audit.diagnostics,
        },
    }
    files = [
        ("sample_table", "game_counts.json", game_counts_table(audit, run_id)),
        ("sample_table", "round_lengths.json", round_lengths_table(audit, run_id)),
        ("sample_table", "identity_checks.json", identity_table(audit, run_id)),
        ("sample_table", "z_values.json", z_table(audit, run_id)),
    ]
    return manifest, files


def write_run(
    out_dir: Path,
    audit: Audit,
    *,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
) -> dict[str, Any]:
    """Create `out_dir` (must not exist) atomically; never overwrites an existing run."""
    run_id = out_dir.name
    validate_run_id(run_id)
    manifest, files = build_documents(
        audit, run_id, git_commit=git_commit, git_dirty=git_dirty, created_at=created_at
    )
    return write_run_directory(out_dir, manifest, files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.export")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    args = parser.parse_args(argv)
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    audit = compute_audit(args.processed_dir)
    commit, dirty = git_source()
    manifest = write_run(
        out_dir, audit, git_commit=commit, git_dirty=dirty, created_at=datetime.now(UTC)
    )
    failed = [c["name"] for c in manifest["validation"]["checks"] if not c["passed"]]
    print(f"{out_dir}: status {manifest['status']}, git_dirty {dirty}")
    for name in failed:
        print(f"  failed check: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
