"""Post-freeze stages of Model Citizens C2 (protocol-validation §10.3, §8, §9).

Every command refuses unless `protocol-validation.md` reads `frozen`, is tracked and
committed (`data.require_frozen_protocol`) and ``--confirm-frozen`` is passed; the
guard runs before any data are loaded. Every command also refuses a dirty working tree,
so each run records ``git_dirty = false`` with its commit, the protocol SHA-256 and the
C2 `config_sha256`.

Commands (in protocol order)::

    python -m phasespace.citizens.heldout g1b --confirm-frozen --run-id citizens-c1-g1b-v1
    python -m phasespace.citizens.heldout evaluate --confirm-frozen \\
        --fitted-run runs/<fit> --surrogate-run runs/<surrogates> --g1b-run runs/<g1b> \\
        --run-id citizens-c2-heldout-eval-v1
    python -m phasespace.citizens.heldout b4 --confirm-frozen --variant plain \\
        --mechanism Interpolating --surrogate-run runs/<surrogates> --run-id <id>
    python -m phasespace.citizens.heldout b4-evaluate --confirm-frozen \\
        --evaluation-run runs/<evaluation> --b4-run runs/<b4 run> [--b4-run ...] \\
        --run-id citizens-c2-b4-eval-v1
    python -m phasespace.citizens.heldout b4-weights --calibration-run runs/<calibration>

- **g1b** (§10.3 step 8): the published Exp 1-4 rank-sum z-values that involve
  held-out mechanisms (games, or Exp 4 games k, as units; `scipy.stats.ranksums`), each
  within half a unit of its last printed digit; and a row-level report of the M1 Exp 1
  pool rows that miss the full pool rule, for root-causing, with the number of games
  whose depletion round would change if each recorded next pool were replaced by the
  full rule's one-step prediction.
- **evaluate** (§6.4, §8, §9): surrogate acceptance on human mechanism rows; the
  criterion for B0, M0, P0, B1, B2, B3 and B23 on the gating regimes R-I, R-M1, R-M2;
  per-experiment and M2 Exp 4 diagnostics; the in-sample check; the G2 decision.
- **b4** (§6.3, §7): one resumable B4 run of 40 games (or a pre-declared reduction)
  for one (variant, mechanism), with the pinned Claude Code CLI version (A26).
  B4-calibrated draws each player's type i.i.d. from the weights chosen on the
  verified calibration games (`choose_type_weights`).
- **b4-evaluate** (A24): criterion (a) and the must-beat-marginal rule (A22) for the B4
  runs, against the human reference CIs and unit metrics recorded by the evaluation run;
  discard fraction and B4 gating.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
from scipy import stats

from phasespace.citizens import config, data, evaluate, metrics, validation
from phasespace.citizens.game import N_PLAYERS, N_ROUNDS
from phasespace.citizens.llm.backend import cli_version_number
from phasespace.citizens.records import TRAINING_MECHANISMS, Games
from phasespace.citizens.simulate import EXACT_MECHANISMS, Mechanism
from phasespace.citizens.split import FIT, human_training_games, make_split
from phasespace.gravity.export import (
    Check,
    dumps_json,
    git_source,
    run_status,
    validate_run_id,
    write_run_directory,
)

type FloatArray = npt.NDArray[np.float64]

GENERATOR: Final = "phasespace.citizens.heldout"
DEFAULT_RUNS_DIR: Final = Path("runs")
INTERPOLATING: Final = "Interpolating Baseline"
M1: Final = "RL Agent (M1)"
M2: Final = "RL Agent (M2)"
REGIME_OF_MECHANISM: Final = {INTERPOLATING: "R-I", M1: "R-M1", M2: "R-M2"}
TRAINABLE: Final = ("B1", "B2", "B3", "B23")
SURROGATE_REGIMES: Final = {"R-M1": "M1", "R-M2": "M2"}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class DirtyTreeError(RuntimeError):
    """A post-freeze stage was started from a working tree with uncommitted changes."""


def require_post_freeze(confirm_frozen: bool) -> tuple[data.FrozenProtocol, str]:
    """The freeze guard, then a clean tree; returns the protocol record and the commit."""
    protocol = data.require_frozen_protocol(
        data._repo_root(),  # pyright: ignore[reportPrivateUsage]
        data.VALIDATION_PROTOCOL_PATH,
        confirm_frozen,
    )
    commit, dirty = git_source()
    if dirty:
        raise DirtyTreeError(
            "the working tree has uncommitted or untracked changes; post-freeze evidence "
            "runs are made from a clean tree only (git_dirty = false)"
        )
    return protocol, commit


def _protocol_block(protocol: data.FrozenProtocol, access: str) -> dict[str, Any]:
    return {
        "path": protocol.path,
        "version": config.PROTOCOL_VERSION,
        "status": protocol.status,
        "sha256": protocol.sha256,
        "held_out_access": access,
    }


def _metric(value: float | int | None, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


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


def _base_manifest(
    run_id: str,
    experiment_id: str,
    stage: str,
    protocol: data.FrozenProtocol,
    commit: str,
    *,
    simulator: dict[str, Any] | None,
    random_seed: int,
    access: str,
) -> dict[str, Any]:
    from phasespace.citizens import fit  # manifest helpers shared with the C2 fit runs

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "study_id": fit.STUDY_ID,
        "experiment_id": experiment_id,
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": {
            "repository": fit.REPOSITORY,
            "git_commit": commit,
            "git_dirty": False,
            "generator": GENERATOR,
            "generator_version": fit.version("phasespace"),
        },
        "environment": fit._environment(),  # pyright: ignore[reportPrivateUsage]
        "model": fit.model_block(),
        "simulator": simulator,
        "random_seed": random_seed,
        "artifacts": [],
        "metrics": {},
        "limitations": [],
        "validation": {
            "checks": [],
            "validated_by": f"{GENERATOR} automated checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
        "citizens": {
            "stage": stage,
            "config_sha256": config.config_sha256(),
            "config": config.config_dict(),
            "validation_protocol": _protocol_block(protocol, access),
        },
    }


def _new_run_dir(runs_dir: Path, run_id: str) -> Path:
    validate_run_id(run_id)
    out = runs_dir / run_id
    if out.exists():
        raise SystemExit(f"{out} already exists; runs are never overwritten")
    return out


def _manifest_sha256(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text("utf-8"))


# --- G1b ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublishedZ:
    name: str
    endpoint: str  # "surplus" | "gini"
    cohort: str
    higher: str  # mechanism reported higher
    lower: str
    published: float  # |z| as printed
    decimals: int  # digits printed after the point
    exp4_game: int | None = None  # Exp 4: game 1, 2 or 3 of each group

    @property
    def tolerance(self) -> float:
        """Half a unit of the last printed digit (2 decimals: 0.005; 1 decimal: 0.05)."""
        return 0.5 * 10.0**-self.decimals


PUBLISHED_HELDOUT_Z: Final = (
    PublishedZ(
        "exp1_surplus_m1_gt_proportional", "surplus", "Exp 1", M1, "Proportional Baseline", 3.25, 2
    ),
    PublishedZ("exp1_surplus_m1_gt_mixed", "surplus", "Exp 1", M1, "Mixed Baseline", 4.38, 2),
    PublishedZ("exp1_surplus_m1_gt_equal", "surplus", "Exp 1", M1, "Equal Baseline", 6.31, 2),
    PublishedZ(
        "exp1_gini_proportional_gt_m1", "gini", "Exp 1", "Proportional Baseline", M1, 2.78, 2
    ),
    PublishedZ(
        "exp2_surplus_m1_gt_proportional", "surplus", "Exp 2", M1, "Proportional Baseline", 2.9, 1
    ),
    PublishedZ(
        "exp2_surplus_interpolating_gt_proportional",
        "surplus",
        "Exp 2",
        INTERPOLATING,
        "Proportional Baseline",
        2.52,
        2,
    ),
    PublishedZ(
        "exp2_gini_proportional_gt_interpolating",
        "gini",
        "Exp 2",
        "Proportional Baseline",
        INTERPOLATING,
        4.17,
        2,
    ),
    PublishedZ("exp2_gini_m1_gt_interpolating", "gini", "Exp 2", M1, INTERPOLATING, 3.67, 2),
    PublishedZ("exp3_surplus_m2_gt_interpolating", "surplus", "Exp 3", M2, INTERPOLATING, 2.35, 2),
    PublishedZ("exp3_gini_m2_gt_interpolating", "gini", "Exp 3", M2, INTERPOLATING, 5.64, 2),
    PublishedZ(
        "exp4_game1_surplus_m2_gt_proportional",
        "surplus",
        "Exp 4",
        M2,
        "Proportional Baseline",
        2.22,
        2,
        1,
    ),
    PublishedZ(
        "exp4_game2_surplus_m2_gt_proportional",
        "surplus",
        "Exp 4",
        M2,
        "Proportional Baseline",
        4.99,
        2,
        2,
    ),
    PublishedZ(
        "exp4_game3_surplus_m2_gt_proportional",
        "surplus",
        "Exp 4",
        M2,
        "Proportional Baseline",
        6.08,
        2,
        3,
    ),
)
"""Koster et al. 2025, Results (Exps 1-4): the surplus and Gini rank-sum z-values that
involve a held-out mechanism (protocol-validation §10.3 step 8a, as amended)."""


def _endpoint(games: Games, endpoint: str) -> FloatArray:
    if endpoint == "surplus":
        return metrics.surplus(games.rewards, games.n_rounds)
    if endpoint == "gini":
        return metrics.reward_gini(games.rewards, games.n_rounds)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def _human(training: Games, heldout: Games, mechanism: str, cohort: str, game: int | None) -> Games:
    source = training if mechanism in TRAINING_MECHANISMS else heldout
    out = source.where(mechanism=mechanism, cohort=cohort, exp4_game=game)
    if out.n_games == 0 or not out.is_human.all():
        raise ValueError(f"no human games for {mechanism} / {cohort} / game {game}")
    return out


@dataclass(frozen=True, slots=True)
class ZResult:
    spec: PublishedZ
    reproduced: float
    n_higher: int
    n_lower: int

    @property
    def abs_difference(self) -> float:
        return abs(self.reproduced - self.spec.published)

    @property
    def passed(self) -> bool:
        return self.abs_difference <= self.spec.tolerance


def heldout_z(training: Games, heldout: Games) -> list[ZResult]:
    out: list[ZResult] = []
    for z in PUBLISHED_HELDOUT_Z:
        hi = _human(training, heldout, z.higher, z.cohort, z.exp4_game)
        lo = _human(training, heldout, z.lower, z.cohort, z.exp4_game)
        value = metrics.rank_sum_z(_endpoint(hi, z.endpoint), _endpoint(lo, z.endpoint))
        out.append(ZResult(z, value, hi.n_games, lo.n_games))
    return out


ANOMALY_BOUND: Final = {"rows": 23, "games": 5, "max_abs_residual": 0.0100 + 1e-4}
"""D33: only RL Agent (M1) Exp 1, at most 23 rows in at most 5 games, |residual| <= 0.0101."""


def pool_anomaly(heldout: Games) -> tuple[list[list[Any]], dict[str, float]]:
    """Rows of held-out games whose recorded next pool misses the full rule (> 1e-4).

    Every held-out mechanism and cohort is scanned (D33 bounds the misses to M1 Exp 1).
    The depletion count replaces each recorded next pool by the full rule's one-step
    prediction from the same row's recorded R, e and c (offers are not recomputed:
    M1's policy is not available) and counts games whose depletion round changes.
    """
    record, known = validation._next_pool_record(heldout)  # pyright: ignore[reportPrivateUsage]
    full, _ = validation.pool_rules(heldout)
    raw = validation.full_raw(heldout)
    residual = np.where(known, record - full, 0.0)
    miss = known & (np.abs(residual) > validation.POOL_TOLERANCE)
    rows: list[list[Any]] = []
    for g, t in zip(*np.nonzero(miss), strict=True):
        e = heldout.offers[g, t]
        c = heldout.actions[g, t]
        rows.append(
            [
                str(heldout.mechanism[g]),
                str(heldout.cohort[g]),
                str(heldout.launch_id[g]),
                int(t),
                int(heldout.n_rounds[g]),
                float(heldout.pool[g, t]),
                float(e.sum()),
                float(c.sum()),
                float(raw[g, t]),
                float(full[g, t]),
                float(record[g, t]),
                float(residual[g, t]),
                json.dumps([float(x) for x in e]),
                json.dumps([float(x) for x in c]),
                int(heldout.players_kept_active[g, t]),
            ]
        )
    # Depletion under the one-step full-rule substitution.
    n = heldout.n_rounds
    pool_alt = heldout.pool.copy()
    replace = np.zeros_like(known)
    replace[:, 1:] = miss[:, :-1]
    shifted = np.full_like(full, np.nan)
    shifted[:, 1:] = full[:, :-1]
    pool_alt = np.where(replace, shifted, pool_alt)
    before = metrics.depletion_round(heldout.pool, n)
    after = metrics.depletion_round(pool_alt, n)
    summary = {
        "rows": float(miss.sum()),
        "games": float(miss.any(axis=1).sum()),
        "max_abs_residual": float(np.abs(residual[miss]).max()) if miss.any() else 0.0,
        "games_outside_m1_exp1": float(
            (miss.any(axis=1) & ~((heldout.mechanism == M1) & (heldout.cohort == "Exp 1"))).sum()
        ),
        "games_depletion_round_changes": float((before != after).sum()),
    }
    return rows, summary


Z_COLUMNS: Final = [
    ("name", None, "comparison (Koster et al. 2025, Results)"),
    ("endpoint", None, "surplus or gini (notebook definitions, games as units)"),
    ("cohort", None, "experiment"),
    ("exp4_game", "count", "Exp 4: game 1, 2 or 3 of each group; null otherwise"),
    ("higher", None, "mechanism reported higher"),
    ("lower", None, "mechanism reported lower"),
    ("n_higher", "count", "games"),
    ("n_lower", "count", "games"),
    ("published", "dimensionless", "|z| as printed"),
    ("reproduced", "dimensionless", "scipy.stats.ranksums(higher, lower).statistic"),
    ("tolerance", "dimensionless", "half a unit of the last printed digit"),
    ("passed", None, "|reproduced - published| <= tolerance"),
]

ANOMALY_COLUMNS: Final = [
    ("mechanism", None, "held-out mechanism"),
    ("cohort", None, "experiment"),
    ("launch_id", None, "game"),
    ("round", "count", "0-based round"),
    ("n_rounds", "count", "recorded rounds of the game"),
    ("pool", "coin", "R before allocation"),
    ("sum_offers", "coin", "sum e"),
    ("sum_actions", "coin", "sum c"),
    ("raw", "coin", "R - sum e + 1.4 sum c (no cap, no floor)"),
    ("full_rule", "coin", "max(0.01, min(200, raw))"),
    ("recorded_next_pool", "coin", "next_environment_state.pool"),
    ("residual", "coin", "recorded_next_pool - full_rule"),
    ("offers", "coin", "JSON list of the 4 offers"),
    ("actions", "coin", "JSON list of the 4 reciprocations"),
    ("players_kept_active", "count", "CSV players_kept_active of the row"),
]


def g1b_command(args: argparse.Namespace) -> int:
    protocol, commit = require_post_freeze(bool(args.confirm_frozen))
    out_dir = _new_run_dir(args.runs_dir, args.run_id)
    held = data.load_heldout(confirm_frozen=True, processed_dir=args.processed_dir)
    training = data.load_training(args.processed_dir)
    zs = heldout_z(training, held.games)
    rows, summary = pool_anomaly(held.games)
    run_id = out_dir.name
    manifest = _base_manifest(
        run_id,
        "citizens-c1-g1b-v1",
        "c1_g1b",
        held.protocol,
        commit,
        simulator=None,
        random_seed=0,
        access="data.load_heldout(confirm_frozen=True)",
    )
    del protocol
    checks = [
        Check(
            f"z_{z.spec.name}_abs_difference",
            z.abs_difference,
            z.spec.tolerance,
            f"|ranksums z - published {z.spec.published}| ({z.n_higher} vs {z.n_lower} games)",
        )
        for z in zs
    ]
    checks += [
        Check("pool_anomaly_rows_over_bound", max(0.0, summary["rows"] - 23), 0.0,
              "pool rows missing the full rule beyond the D33 bound of 23 (0 = within)"),
        Check("pool_anomaly_games_over_bound", max(0.0, summary["games"] - 5), 0.0,
              "games with such rows beyond the D33 bound of 5 (0 = within)"),
        Check("pool_anomaly_max_abs_residual", summary["max_abs_residual"],
              ANOMALY_BOUND["max_abs_residual"], "max |recorded - full rule| over those rows"),
        Check("pool_anomaly_games_outside_m1_exp1", summary["games_outside_m1_exp1"], 0.0,
              "games outside RL Agent (M1) Exp 1 with such rows"),
    ]  # fmt: skip
    manifest["validation"]["checks"] = [c.to_json() for c in checks]
    manifest["status"] = run_status(checks)
    manifest["metrics"] = {
        f"anomaly_{k}": _metric(v, "coin" if "residual" in k else "count", k.replace("_", " "))
        for k, v in summary.items()
    }
    manifest["limitations"] = [
        "The anomaly table lists the rows for root-causing; the diagnosis itself is the "
        "owner's reading (§10.3 step 8b). Human metrics always use the recorded rows and "
        "the simulator keeps the full rule, whatever the cause (§10.3).",
        "The depletion count substitutes the full rule's one-step prediction for each "
        "missing row's next pool without recomputing later offers (M1 is not released).",
        "Published z-values printed to one decimal (Exp 2 M1 > Proportional, 2.9) carry a "
        "tolerance of 0.05, those printed to two decimals 0.005.",
    ]
    manifest["citizens"]["dataset"] = {
        "training": data.processed_sidecar(args.processed_dir, data.TRAINING_FILE),
        "heldout": data.processed_sidecar(args.processed_dir, data.HELDOUT_FILE),
    }
    z_rows = [
        [
            z.spec.name, z.spec.endpoint, z.spec.cohort, z.spec.exp4_game, z.spec.higher,
            z.spec.lower, z.n_higher, z.n_lower, z.spec.published, z.reproduced,
            z.spec.tolerance, "true" if z.passed else "false",
        ]
        for z in zs
    ]  # fmt: skip
    files = [
        ("sample_table", "z_values.json", _table(run_id, Z_COLUMNS, z_rows)),
        ("sample_table", "pool_anomaly_rows.json", _table(run_id, ANOMALY_COLUMNS, rows)),
    ]
    write_run_directory(out_dir, manifest, files)
    for z in zs:
        print(f"{z.spec.name:<46} published {z.spec.published:5.2f} reproduced "
              f"{z.reproduced:7.4f} {'pass' if z.passed else 'FAIL'}")  # fmt: skip
    print(json.dumps(summary, indent=1))
    print(f"{out_dir}: status {manifest['status']}")
    return 0


# --- surrogate acceptance and the evaluation ------------------------------------------


@dataclass(frozen=True, slots=True)
class Inputs:
    """The frozen inputs of the evaluation, each verified against its manifest."""

    fitted: Any  # fit.Fitted
    mechanisms: dict[str, Any]  # "M1"/"M2" -> surrogate.SurrogateMechanism (BC-accepted)
    bc_accepted: dict[str, bool]
    g1b_failed_regimes: frozenset[str]
    records: dict[str, dict[str, str]]  # input run -> {run_id, manifest_sha256}
    surrogate_reproduction: dict[str, Any] = field(default_factory=dict[str, Any])


def surrogate_reproduction_problems(manifest: Mapping[str, Any]) -> list[str]:
    """§10.2 / A25: the post-freeze surrogate run must reproduce the pre-freeze fit bit
    for bit: `--reproduce` was used (reference_parameters_sha256 set for M1 and M2),
    every `surrogate_*_differs_from_reference` check is 0 and passed, and the run's
    status is `validated` (every BC check passed). Returns the problems (empty = ok)."""
    problems: list[str] = []
    reference = manifest.get("citizens", {}).get("reference_parameters_sha256")
    if not isinstance(reference, dict) or set(reference) != {"M1", "M2"}:  # pyright: ignore[reportUnknownArgumentType]
        problems.append("surrogate run: not made with --reproduce (no reference parameters)")
    checks = {c["name"]: c for c in manifest.get("validation", {}).get("checks", [])}
    for key in ("m1", "m2"):
        c = checks.get(f"surrogate_{key}_differs_from_reference")
        if c is None:
            problems.append(f"surrogate run: no reproduction check for {key.upper()}")
        elif c.get("observed") != 0.0 or not c.get("passed"):
            problems.append(f"surrogate run: {key.upper()} parameters differ from the reference")
    if manifest.get("status") != "validated":
        problems.append(f"surrogate run: status {manifest.get('status')!r}, not 'validated'")
    return problems


def load_inputs(
    fitted_run: Path,
    surrogate_run: Path,
    g1b_run: Path,
    protocol: data.FrozenProtocol,
    *,
    override_surrogate_reproduction: str | None = None,
) -> Inputs:
    """Load and check the fitted-model, surrogate and G1b runs (refuses on any mismatch).

    The surrogate run must have reproduced the pre-freeze fit bit for bit
    (`surrogate_reproduction_problems`) unless the owner overrides it with a reason
    (`override_surrogate_reproduction`), which is recorded in the evaluation manifest.
    """
    from phasespace.citizens import fit, surrogate

    problems: list[str] = []
    fit_m = _read_manifest(fitted_run)
    sur_m = _read_manifest(surrogate_run)
    g1b_m = _read_manifest(g1b_run)
    for label, m, stage in (
        ("fitted run", fit_m, "c2_fit"),
        ("surrogate run", sur_m, "c2_surrogate_fit"),
        ("G1b run", g1b_m, "c1_g1b"),
    ):
        c = m.get("citizens", {})
        if c.get("stage") != stage:
            problems.append(f"{label}: stage {c.get('stage')!r}, expected {stage!r}")
        vp = c.get("validation_protocol", {})
        if vp.get("sha256") != protocol.sha256 or vp.get("status") != data.FROZEN:
            problems.append(f"{label}: not made under the frozen protocol {protocol.sha256}")
        if c.get("config_sha256") != config.config_sha256():
            problems.append(f"{label}: config_sha256 differs from the current configuration")
        if m["source"].get("git_dirty"):
            problems.append(f"{label}: made from a dirty tree")
    if not fit_m.get("citizens", {}).get("fit_settings", {}).get("protocol_settings"):
        problems.append("fitted run: reduced fit settings, not the protocol's")
    reproduction = surrogate_reproduction_problems(sur_m)
    if reproduction and not (override_surrogate_reproduction or "").strip():
        problems.extend(reproduction)
    if problems:
        raise ValueError("evaluation inputs refused:\n- " + "\n- ".join(problems))
    fitted = fit.load_fitted(fitted_run)
    mechanisms: dict[str, Any] = {}
    accepted: dict[str, bool] = {}
    for key in ("M1", "M2"):
        entry = sur_m["citizens"]["surrogates"][key]
        accepted[key] = bool(entry["bc_validation_fidelity"]["accepted"])
        if accepted[key]:
            mechanisms[key] = surrogate.load_mechanism(surrogate_run, key)
    failed: set[str] = set()
    z_rows = json.loads((g1b_run / "z_values.json").read_text("utf-8"))["rows"]
    for row in z_rows:
        if row[-1] != "true":
            for mech in (row[4], row[5]):
                if mech in REGIME_OF_MECHANISM:
                    failed.add(REGIME_OF_MECHANISM[mech])
    records = {
        name: {"run_id": path.name, "manifest_sha256": _manifest_sha256(path)}
        for name, path in (
            ("fitted_run", fitted_run),
            ("surrogate_run", surrogate_run),
            ("g1b_run", g1b_run),
        )
    }
    reproduction_record = {
        "problems": reproduction,
        "owner_override_reason": override_surrogate_reproduction if reproduction else None,
    }
    return Inputs(fitted, mechanisms, accepted, frozenset(failed), records, reproduction_record)


VALIDATED_CLOSED: Final = "validated (closed-loop)"
VALIDATED_ONE_STEP: Final = "validated (one-step)"
NOT_VALIDATED: Final = "not validated"
NOT_BEATING_M0_SIM: Final = "not validated (mean normalised W1 not below M0-sim)"


def beats_m0_sim(model_w1: float | None, m0_sim_w1: float | None) -> bool:
    """A22: the model's mean normalised W1 is strictly lower than M0-sim's."""
    if model_w1 is None or m0_sim_w1 is None:
        return False
    return model_w1 < m0_sim_w1


def regime_label(
    result: evaluate.ModelResult,
    closed_loop: bool,
    *,
    m0_sim_passes_a: bool = False,
    m0_sim_w1: float | None = None,
) -> str:
    """§8: closed-loop needs (a) and (b); with a rejected surrogate only (b) counts.

    Must beat marginal (A22): where M0-sim passes (a) (`m0_sim_passes_a`), a
    closed-loop validation also needs the model's mean normalised W1 over the 4 metrics
    to be strictly lower than M0-sim's (`m0_sim_w1`).
    """
    if result.pass_likelihood is None:
        return "control"
    if closed_loop:
        if not result.validated:
            return NOT_VALIDATED
        if m0_sim_passes_a and not beats_m0_sim(result.mean_normalised_w1, m0_sim_w1):
            return NOT_BEATING_M0_SIM
        return VALIDATED_CLOSED
    return VALIDATED_ONE_STEP if result.pass_likelihood else NOT_VALIDATED


def b0_control_passes(b0: evaluate.ModelResult, closed_loop: bool) -> bool:
    """§9 negative control (A23): B0 passes the criterion that decides validation in the
    regime, i.e. (a) where closed-loop results count and (b) where only the one-step
    likelihood counts (a rejected surrogate). B0 never needs both."""
    return bool(b0.pass_outcomes) if closed_loop else bool(b0.pass_likelihood)


def g2_decision(
    labels: Mapping[str, Mapping[str, str]],
    counting: Mapping[str, bool],
    b0_passes: Mapping[str, bool],
) -> dict[str, Any]:
    """§9 with O3 and O4: the G2 verdict from the per (regime, model) labels.

    `labels[regime][model]`; `counting[regime]` is False where the regime's human
    reference is unverified (G1b); `b0_passes[regime]` is `b0_control_passes` of the B0
    control there. B0 passing a regime makes the criterion too weak there: no label from
    that regime counts, and G2 cannot pass (§9). A one-step label counts only with a
    closed-loop validation of the same model on another counting regime (O3). G2 needs
    two trainable classes, at least one not B1, validated on one common counting regime
    (O4).
    """
    b0 = {r: bool(b0_passes.get(r, False)) for r in labels}
    usable = {r: counting.get(r, False) and not b0[r] for r in labels}
    closed = {
        m: {r for r in labels if usable[r] and labels[r].get(m) == VALIDATED_CLOSED}
        for m in TRAINABLE
    }
    counted: dict[str, list[str]] = {}
    for r in labels:
        counted[r] = []
        if not usable[r]:
            continue
        for m in TRAINABLE:
            label = labels[r].get(m)
            if label == VALIDATED_CLOSED or (label == VALIDATED_ONE_STEP and closed[m] - {r}):
                counted[r].append(m)
    common = {r: ms for r, ms in counted.items() if len(ms) >= 2 and any(m != "B1" for m in ms)}
    passed = bool(common) and not any(b0.values())
    return {
        "g2_passed": passed,
        "b0_passes": b0,
        "regime_counts": usable,
        "counted_classes": counted,
        "regimes_meeting_o4": sorted(common),
        "rule": "G2 passes iff B0 fails every gating regime ((a) where closed-loop, (b) "
        "where one-step; A23) and some counting regime has >= 2 trainable classes "
        "(>= 1 not B1) counted there (§9, O3, O4); closed-loop labels include the A22 "
        "must-beat-marginal rule",
    }


def w1_comparison(
    result: evaluate.RegimeResult, closed_loop: bool
) -> tuple[bool, float | None, dict[str, Any]]:
    """A22 inputs and record of one regime: (m0_sim_passes_a, M0-sim mean normalised W1,
    the comparison block for the G2 record)."""
    m0 = next((m for m in result.models if m.model == "M0-sim"), None)
    passes_a = bool(m0 is not None and m0.pass_outcomes)
    m0_w1 = None if m0 is None else m0.mean_normalised_w1
    models = {
        m.model: {
            "mean_normalised_w1": m.mean_normalised_w1,
            "beats_m0_sim": beats_m0_sim(m.mean_normalised_w1, m0_w1),
        }
        for m in result.models
        if m.outcomes is not None and m.model != "M0-sim"
    }
    block = {
        "m0_sim_passes_a": passes_a,
        "m0_sim_mean_normalised_w1": m0_w1,
        "rule_applies": closed_loop and passes_a,
        "w1_scale": result.w1_scale,
        "models": models,
    }
    return passes_a, m0_w1, block


def _sim_layout_exp4(
    human: Games, sets_per_human: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """M2 Exp 4: per human set (sorted by launch_id) `sets_per_human` simulated sets of
    its games' recorded lengths (game 1, 2, 3); returns (lengths, unit index) per game."""
    lengths: list[int] = []
    units: list[int] = []
    unit = 0
    for key in sorted(set(human.launch_id.tolist())):
        group = human.select(human.launch_id == key)
        order = np.argsort(group.exp4_game)
        group_lengths = [int(x) for x in group.n_rounds[order]]
        for _ in range(sets_per_human):
            lengths.extend(group_lengths)
            units.extend([unit] * len(group_lengths))
            unit += 1
    return np.array(lengths, dtype=np.int64), np.array(units, dtype=np.int64)


HUMAN_UNITS_FILE: Final = "human_units_heldout.json"
HUMAN_UNIT_COLUMNS: Final = [
    ("regime", None, "gating regime"),
    ("cohort", None, "experiment (bootstrap stratum)"),
    ("launch_id", None, "human group"),
    *[(k, None, f"unit {k} (notebook definition; players active = #{{e >= 1}}, A30)")
      for k in config.METRICS],
]  # fmt: skip
"""Human unit-level metrics of each gating regime: the distribution the A22 W1 uses,
read by `b4-evaluate` so B4 is compared with exactly the evaluation run's reference."""


def human_unit_rows(heldout_games: Games) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for regime, (mechanism, cohorts, _) in evaluate.GATING_REGIMES.items():
        human = evaluate.regime_games(heldout_games, mechanism, cohorts)
        units = evaluate.units_of(human)
        per_game = evaluate.human_game_metrics(human)
        values = {k: evaluate.unit_means(per_game[k], units) for k in config.METRICS}
        for u, (cohort, launch_id) in enumerate(units.keys):
            rows.append([regime, cohort, launch_id, *(float(values[k][u]) for k in config.METRICS)])
    return rows


def evaluate_command(args: argparse.Namespace) -> int:
    protocol, commit = require_post_freeze(bool(args.confirm_frozen))
    for flag in ("fitted_run", "surrogate_run", "g1b_run", "run_id"):
        if getattr(args, flag) is None:
            raise SystemExit(f"--{flag.replace('_', '-')} is required")
    out_dir = _new_run_dir(args.runs_dir, args.run_id)
    inputs = load_inputs(
        args.fitted_run,
        args.surrogate_run,
        args.g1b_run,
        protocol,
        override_surrogate_reproduction=args.owner_override_surrogate_reproduction,
    )
    held = data.load_heldout(confirm_frozen=True, processed_dir=args.processed_dir)
    training = data.load_training(args.processed_dir)
    from phasespace.citizens import surrogate

    # (b) surrogate acceptance on human mechanism rows (every experiment of M1, M2).
    accepted_mechanisms = cast(
        "dict[surrogate.MechanismKey, surrogate.SurrogateMechanism]", inputs.mechanisms
    )
    human_fid = surrogate.human_fidelity(
        accepted_mechanisms, confirm_frozen=True, processed_dir=args.processed_dir
    )
    accepted_human = {
        key: inputs.bc_accepted[key]
        and all(f.accepted for (k, _), f in human_fid.by_experiment.items() if k == key)
        for key in ("M1", "M2")
    }
    lik = inputs.fitted.likelihood_models()
    sims = inputs.fitted.sim_models()

    def mech(sim_regime: str) -> Mechanism[Any] | None:
        if sim_regime == "Interpolating":
            return EXACT_MECHANISMS["Interpolating"]
        return inputs.mechanisms.get(sim_regime.split()[0])

    def run(regime: str, sim_regime: str, human: Games, **kw: Any) -> evaluate.RegimeResult:
        _log(f"{regime}: {human.n_games} human games")
        result, _ = evaluate.evaluate_regime(
            regime,
            sim_regime,
            human,
            likelihood_models=lik,
            sim_models=sims,
            mechanism=mech(sim_regime),
            **kw,
        )
        return result

    gating: list[evaluate.RegimeResult] = []
    diagnostic: list[evaluate.RegimeResult] = []
    for regime, (mechanism, cohorts, sim_regime) in evaluate.GATING_REGIMES.items():
        gating.append(
            run(regime, sim_regime, evaluate.regime_games(held.games, mechanism, cohorts))
        )
        for cohort in cohorts:
            human = evaluate.regime_games(held.games, mechanism, (cohort,))
            diagnostic.append(run(f"{regime} {cohort}", sim_regime, human))
    exp4 = evaluate.regime_games(held.games, M2, ("Exp 4",))
    lengths, units = _sim_layout_exp4(exp4, config.M2_EXP4_SETS_PER_HUMAN_SET)
    diagnostic.append(
        run("M2 Exp 4", "M2 Exp 4", exp4, n_sim=int(lengths.size), n_rounds=lengths, sim_unit=units)
    )
    in_sample: list[evaluate.RegimeResult] = []
    human_train = human_training_games(training)
    for regime, (mechanism, cohorts) in evaluate.TRAINING_REGIMES.items():
        human = evaluate.regime_games(human_train, mechanism, cohorts)
        _log(f"in-sample {regime}: {human.n_games} games")
        result, _ = evaluate.evaluate_regime(
            regime, regime, human, likelihood_models=lik, sim_models=sims,
            mechanism=EXACT_MECHANISMS[regime],
        )  # fmt: skip
        in_sample.append(result)

    closed_loop = {
        "R-I": True,
        "R-M1": accepted_human["M1"],
        "R-M2": accepted_human["M2"],
    }
    labels: dict[str, dict[str, str]] = {}
    b0_passes: dict[str, bool] = {}
    comparison: dict[str, Any] = {}
    for r in gating:
        tier = closed_loop[r.regime]
        passes_a, m0_w1, comparison[r.regime] = w1_comparison(r, tier)
        labels[r.regime] = {
            m.model: regime_label(m, tier, m0_sim_passes_a=passes_a, m0_sim_w1=m0_w1)
            for m in r.models
        }
        b0 = next(m for m in r.models if m.model == "B0")
        b0_passes[r.regime] = b0_control_passes(b0, tier)
    counting = {r: r not in inputs.g1b_failed_regimes for r in labels}
    decision = g2_decision(labels, counting, b0_passes)
    decision["m0_sim_passes_a"] = {r: c["m0_sim_passes_a"] for r, c in comparison.items()}
    decision["w1_comparison"] = comparison
    reference = {
        r.regime: {
            k: {"mean": e.value, "lo": e.lo, "hi": e.hi, "tau": config.TOLERANCE[k],
                "w1_scale": r.w1_scale[k]}
            for k, e in r.human.items()
        }
        for r in gating
    }  # fmt: skip

    run_id = out_dir.name
    simulated = sum(r.feasible_games for r in gating + diagnostic + in_sample)
    manifest = _base_manifest(
        run_id,
        "citizens-c2-heldout-eval-v1",
        "c2_heldout_eval",
        held.protocol,
        commit,
        simulator={
            "name": "phasespace.citizens.simulate",
            "pool_rule": "full: R' = max(0.01, min(200, R - sum e + 1.4 sum c))",
            "offer_rule": "deployed",
            "rounds": N_ROUNDS,
            "games_per_model_mechanism": config.SIM_GAMES,
            "seed_rule": "50000 + 100 m + r (protocol 7)",
            "feasibility_atol": config.SIM_FEASIBILITY_ATOL,
        },
        random_seed=config.BOOTSTRAP_SEED,
        access="data.load_heldout(confirm_frozen=True)",
    )
    del protocol
    checks = [
        Check("infeasible_simulated_games", 0.0, 0.0,
              f"simulated games violating protocol 5 (a violation aborts the run, K2); "
              f"{simulated} games simulated"),
        Check("input_checksums_verified", 0.0, 0.0,
              "fitted-model and surrogate artifacts loaded by SHA-256 (a mismatch aborts)"),
        Check("g1b_failed_regimes", float(len(inputs.g1b_failed_regimes)), 0.0,
              "gating regimes whose published z-values were not reproduced (G1b)"),
    ]  # fmt: skip
    manifest["validation"]["checks"] = [c.to_json() for c in checks]
    manifest["status"] = run_status(checks)
    manifest["citizens"]["inputs"] = inputs.records
    manifest["citizens"]["surrogate_reproduction"] = inputs.surrogate_reproduction
    manifest["citizens"]["dataset"] = {
        "training": data.processed_sidecar(args.processed_dir, data.TRAINING_FILE),
        "heldout": data.processed_sidecar(args.processed_dir, data.HELDOUT_FILE),
    }
    manifest["citizens"]["surrogates"] = {
        "bc_accepted": inputs.bc_accepted,
        "human_accepted": accepted_human,
        "closed_loop_tier": closed_loop,
    }
    manifest["citizens"]["labels"] = labels
    manifest["citizens"]["g2"] = decision
    manifest["citizens"]["human_reference"] = reference
    manifest["citizens"]["closed_loop_tier"] = closed_loop
    manifest["citizens"]["declared_handling"] = {
        "excluded_decisions": {r.regime: r.excluded_decisions for r in gating},
        "floored_decisions": {
            r.regime: {m.model: m.floored_decisions for m in r.models} for r in gating
        },
        "rule": "A29: anomalous recorded decisions excluded from the likelihood; scored "
        "decisions with P < 1e-12 scored at log(1e-12); both counted",
    }
    manifest["metrics"]["g2_passed"] = _metric(
        int(decision["g2_passed"]), "count", "1 if G2 passes by §9 / O3 / O4, else 0"
    )
    manifest["limitations"] = [
        "B4 (LLM citizens) results are separate runs (stage c2_b4_heldout); they never "
        "decide G2 (O4).",
        "M2 Exp 4 and the per-experiment rows are diagnostics, never gating (§8).",
    ]
    if inputs.surrogate_reproduction.get("owner_override_reason"):
        manifest["limitations"].append(
            "The surrogate run did not reproduce the pre-freeze fit bit for bit; the owner "
            "accepted it: " + str(inputs.surrogate_reproduction["owner_override_reason"])
        )
    fid_rows = [
        [key, cohort, f.player_rounds, f.median_abs, f.p90_abs, f.mean_abs,
         f.exclusion_agreement, f.recorded_exclusion_rate, "true" if f.accepted else "false"]
        for (key, cohort), f in sorted(human_fid.by_experiment.items())
    ]  # fmt: skip
    fid_cols = [
        ("mechanism", None, "M1 or M2"),
        ("cohort", None, "experiment"),
        ("player_rounds", "count", "player-rounds of rounds with R >= 1"),
        ("median_abs", "coin", "median |e_hat - e|, teacher-forced"),
        ("p90_abs", "coin", "90th percentile |e_hat - e|"),
        ("mean_abs", "coin", "mean |e_hat - e|"),
        ("exclusion_agreement", "dimensionless", "fraction with 1[e_hat<1] = 1[e<1]"),
        ("recorded_exclusion_rate", "dimensionless", "fraction of recorded e < 1"),
        ("accepted", None, "median <= 1, p90 <= 5, agreement >= 0.90 (§6.4)"),
    ]
    files = [
        ("sample_table", "surrogate_fidelity_human.json", _table(run_id, fid_cols, fid_rows)),
        ("sample_table", HUMAN_UNITS_FILE,
         _table(run_id, HUMAN_UNIT_COLUMNS, human_unit_rows(held.games))),
    ]  # fmt: skip
    for name, results in (
        ("heldout", gating),
        ("diagnostic", diagnostic),
        ("in_sample", in_sample),
    ):
        files.append(
            ("sample_table", f"criterion_{name}.json",
             _table(run_id, evaluate.CRITERION_COLUMNS, evaluate.criterion_rows(results)))
        )  # fmt: skip
        files.append(
            ("sample_table", f"outcomes_{name}.json",
             _table(run_id, evaluate.OUTCOME_COLUMNS, evaluate.outcome_rows(results)))
        )  # fmt: skip
    write_run_directory(out_dir, manifest, files)
    print(evaluate.summary(gating))
    print(json.dumps({"labels": labels, "g2": decision}, indent=1))
    print(f"{out_dir}: status {manifest['status']}")
    return 0


# --- B4 held-out runs -------------------------------------------------------------------


def human_type_targets(training: Games) -> tuple[dict[str, float], dict[str, float], int]:
    """h_m, s_m: mean and between-group SD (ddof 1) of the fit-split Exp 1 + Exp 2
    Proportional human games (§6.3), per §8 metric; and the number of games."""
    human = human_training_games(training)
    split = make_split(training)
    fit_games = human.select(split.mask(human, FIT))
    games = evaluate.regime_games(fit_games, "Proportional Baseline", ("Exp 1", "Exp 2"))
    per_game = evaluate.human_game_metrics(games)
    h = {k: float(per_game[k].mean()) for k in config.METRICS}
    s = {k: float(per_game[k].std(ddof=1)) for k in config.METRICS}
    return h, s, games.n_games


def simplex_grid(step: float = config.B4_GRID_STEP) -> list[tuple[float, float, float]]:
    """The weights (w1, w2, w3) >= 0 summing to 1 on a grid of `step` (66 for 0.1)."""
    n = round(1.0 / step)
    return [(a / n, b / n, (n - a - b) / n) for a in range(n + 1) for b in range(n + 1 - a)]


def composition_probability(counts: tuple[int, int, int], w: Sequence[float]) -> float:
    """Multinomial(counts; 4, w)."""
    coef = math.factorial(N_PLAYERS) / math.prod(math.factorial(k) for k in counts)
    return coef * math.prod(p**k for p, k in zip(w, counts, strict=True))


def choose_type_weights(
    composition_means: Mapping[tuple[int, int, int], Mapping[str, float]],
    h: Mapping[str, float],
    s: Mapping[str, float],
    step: float = config.B4_GRID_STEP,
) -> tuple[tuple[float, float, float], float]:
    """§6.3: w on the simplex grid minimising sum_m ((pred_m - h_m) / s_m)^2, with
    pred_m = sum_comp Multinomial(comp; 4, w) mean_m(comp); ties (1e-12) go to the w
    nearest uniform, then the first in grid order. Returns (w, objective)."""
    best: tuple[float, float, float] | None = None
    best_key = (math.inf, math.inf)
    for w in simplex_grid(step):
        loss = 0.0
        for m in config.METRICS:
            pred = sum(
                composition_probability(comp, w) * means[m]
                for comp, means in composition_means.items()
            )
            loss += ((pred - h[m]) / s[m]) ** 2
        dist = math.dist(w, (1 / 3, 1 / 3, 1 / 3))
        key = (round(loss, 12), dist)
        if key < best_key:
            best, best_key = w, key
    assert best is not None
    return best, best_key[0]


def verified_artifact(run_dir: Path, name: str) -> bytes:
    """The bytes of artifact `name` of a run, after checking its SHA-256 in the manifest."""
    entry = next(
        (a for a in _read_manifest(run_dir).get("artifacts", []) if a["path"] == name), None
    )
    if entry is None:
        raise ValueError(f"{run_dir}: {name} is not listed in the manifest's artifacts")
    payload = (run_dir / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"{run_dir / name}: SHA-256 differs from the manifest")
    return payload


def calibration_composition_means(run_dir: Path) -> dict[tuple[int, int, int], dict[str, float]]:
    """Mean §8 metrics of the complete calibration games per type composition.

    `games.json` is verified against the run manifest's SHA-256, and every one of the 15
    compositions must have exactly 4 complete games (§6.3; A31)."""
    games = json.loads(verified_artifact(run_dir, "games.json").decode("utf-8"))
    cols = [c["name"] for c in games["columns"]]
    by: dict[tuple[int, int, int], list[dict[str, float]]] = {}
    for row in games["rows"]:
        r = dict(zip(cols, row, strict=True))
        if r["status"] != "complete":
            continue
        types: list[str] = json.loads(r["types"])
        comp = (types.count("T1"), types.count("T2"), types.count("T3"))
        by.setdefault(comp, []).append(
            {
                "surplus": float(r["surplus"]),
                "gini": float(r["gini"]),
                "depletion_round": float(r["depletion_round"]),
                "players_active": float(r["players_active_end"]),
            }
        )
    expected = {(a, b, 4 - a - b) for a in range(5) for b in range(5 - a)}
    if set(by) != expected:
        raise ValueError("the calibration run does not cover all 15 type compositions")
    per = config.B4_CALIBRATION_GAMES_PER_COMPOSITION
    wrong = sorted(comp for comp, rows in by.items() if len(rows) != per)
    if wrong:
        raise ValueError(
            f"the calibration run needs exactly {per} complete games per composition; "
            f"compositions {wrong} differ"
        )
    return {
        comp: {m: float(np.mean([g[m] for g in rows])) for m in config.METRICS}
        for comp, rows in by.items()
    }


def type_weights(calibration_run: Path, processed_dir: Path) -> dict[str, Any]:
    h, s, n = human_type_targets(data.load_training(processed_dir))
    means = calibration_composition_means(calibration_run)
    w, loss = choose_type_weights(means, h, s)
    return {
        "weights": list(w),
        "objective": loss,
        "human_targets": {"mean": h, "sd": s, "games": n},
        "calibration_run": {
            "run_id": calibration_run.name,
            "manifest_sha256": _manifest_sha256(calibration_run),
        },
    }


CALIBRATION_IDENTITY: Final = ("prompt_version", "system_prompt", "rules_sha256", "type_texts")


def check_calibration_identity(calibration_run: Path, backend: Mapping[str, Any]) -> None:
    """Refuse calibration games played with other prompts, model, effort or CLI flags."""
    from phasespace.citizens.llm import prompts, runner

    cfg = _read_manifest(calibration_run)["citizens"]["config"]
    current = runner.RunConfig(
        run_id="x", label="x", variant="calibrated", mechanism="x", target_games=0,
        n_rounds=0, workers=0, max_retries=0, discard_limit=0, seed=0, backend={},
    ).to_json()  # fmt: skip
    problems = [k for k in CALIBRATION_IDENTITY if cfg.get(k) != current[k]]
    if cfg.get("backend", {}).get("argv") != backend["argv"]:
        problems.append("backend.argv")
    if cli_version_number(cfg.get("backend", {}).get("cli_version")) != config.B4_CLI_VERSION:
        problems.append("backend.cli_version")
    if cfg.get("prompt_version") != prompts.PROMPT_VERSION:
        problems.append("prompt_version")
    if problems:
        raise SystemExit(
            f"{calibration_run}: calibration games differ from the frozen B4 set-up in "
            f"{sorted(set(problems))}; re-run the calibration (§10.1 step 4)"
        )


@dataclass(frozen=True, slots=True)
class SingleGameMechanism:
    """A batched `simulate.Mechanism` driven one game at a time (`runner.GameMechanism`)."""

    batched: Mechanism[Any]

    @property
    def name(self) -> str:
        return self.batched.name

    def start(self) -> object:
        return self.batched.initial_state(1)

    def offers(
        self,
        state: object,
        pool: float,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
    ) -> tuple[FloatArray, object]:
        e, new = self.batched.offers(
            state,
            np.array([pool], dtype=np.float64),
            None if prev_offers is None else np.asarray(prev_offers, dtype=np.float64)[None],
            None
            if prev_reciprocations is None
            else np.asarray(prev_reciprocations, dtype=np.float64)[None],
            0,
        )
        return np.asarray(e, dtype=np.float64)[0], new


B4_MECHANISMS: Final = ("Interpolating", "M1", "M2")


def allowed_b4_games(mechanism: str) -> tuple[int, ...]:
    """§7 / A31: 40 games, or a pre-declared reduction (30; 20 for M1/M2 only)."""
    full, reduced, m1m2 = config.B4_EVAL_GAMES, *config.B4_EVAL_GAMES_REDUCED
    return (full, reduced) if mechanism == "Interpolating" else (full, reduced, m1m2)


def require_cli_version(info: Mapping[str, Any]) -> None:
    """A26: the installed Claude Code CLI must be the pinned version."""
    found = cli_version_number(info.get("cli_version"))
    if found != config.B4_CLI_VERSION:
        raise SystemExit(
            f"claude --version reports {info.get('cli_version')!r}; the frozen protocol "
            f"pins Claude Code {config.B4_CLI_VERSION} (A26)"
        )


def b4_command(args: argparse.Namespace) -> int:
    from phasespace.citizens import surrogate
    from phasespace.citizens.llm import runner
    from phasespace.citizens.llm.backend import ClaudeCLI

    protocol, commit = require_post_freeze(bool(args.confirm_frozen))
    validate_run_id(args.run_id)
    if (args.runs_dir / args.run_id).exists():
        raise SystemExit(f"{args.runs_dir / args.run_id} already exists")
    key: str = args.mechanism
    if int(args.games) not in allowed_b4_games(key):
        raise SystemExit(
            f"--games {args.games}: the protocol allows {allowed_b4_games(key)} for {key} (§7)"
        )
    require_cli_version(ClaudeCLI().describe())
    surrogate_record: dict[str, Any] | None = None
    if key == "Interpolating":
        batched: Mechanism[Any] = EXACT_MECHANISMS["Interpolating"]
    else:
        if args.surrogate_run is None:
            raise SystemExit("--surrogate-run is required for M1/M2")
        sur_m = _read_manifest(args.surrogate_run)
        if sur_m["citizens"]["validation_protocol"].get("sha256") != protocol.sha256:
            raise SystemExit("the surrogate run was not made under the frozen protocol")
        if not sur_m["citizens"]["surrogates"][key]["bc_validation_fidelity"]["accepted"]:
            raise SystemExit(f"the {key} surrogate failed the BC check: B4 is not run (§7)")
        batched = surrogate.load_mechanism(args.surrogate_run, cast(surrogate.MechanismKey, key))
        surrogate_record = {
            "run_id": args.surrogate_run.name,
            "manifest_sha256": _manifest_sha256(args.surrogate_run),
        }
    mechanism = SingleGameMechanism(batched)
    weights_record: dict[str, Any] | None = None
    if args.variant == "calibrated":
        if args.calibration_run is None:
            raise SystemExit("--calibration-run is required for B4-calibrated")
        check_calibration_identity(args.calibration_run, ClaudeCLI().describe())
        weights_record = type_weights(args.calibration_run, args.processed_dir)
        w = weights_record["weights"]

        def spec(k: int) -> runner.GameSpec:
            return runner.GameSpec(k, N_ROUNDS, runner.evaluation_types(w, k))
    else:

        def spec(k: int) -> runner.GameSpec:
            return runner.GameSpec(k, N_ROUNDS, (None, None, None, None))

    backend = ClaudeCLI()
    games = int(args.games)
    run_config = runner.RunConfig(
        run_id=args.run_id,
        label="heldout",
        variant=args.variant,
        mechanism=mechanism.name,
        target_games=games,
        n_rounds=N_ROUNDS,
        workers=int(args.workers),
        max_retries=runner.MAX_RETRIES,
        discard_limit=runner.discard_limit(games),
        seed=runner.B4_SEED,
        backend=backend.describe(),
    )
    store = runner.prepare_work_dir(args.runs_dir / runner.WORK_DIR / args.run_id, run_config)
    start = time.perf_counter()
    try:
        outcomes = asyncio.run(
            runner.run_games(
                run_config, spec, mechanism, backend, store, log=lambda s: print(s, flush=True)
            )
        )
    except runner.InfrastructureError as exc:
        print(f"stopped: {exc}", flush=True)
        return 3
    extra = {
        "frozen_protocol_sha256": protocol.sha256,
        "config_sha256": config.config_sha256(),
        "git_commit_at_start": commit,
        "mechanism_key": key,
        "surrogate_run": surrogate_record,
        "type_weights": weights_record,
    }
    manifest = runner.finalize_games_run(
        args.runs_dir, run_config, outcomes, wall_s=time.perf_counter() - start, extra=extra
    )
    print(json.dumps(manifest["citizens"]["call_statistics"], indent=1))
    print(f"{args.runs_dir / args.run_id}: {manifest['status']}")
    return 0


# --- B4 evaluation (criterion (a) and A22 against the evaluation run) ------------------

B4_REGIMES: Final = {"Interpolating": "R-I", "M1": "R-M1", "M2": "R-M2"}


@dataclass(frozen=True, slots=True)
class B4MetricRow:
    metric: str
    b4: evaluate.Estimate  # mean over complete games, game-bootstrap CI
    human_lo: float
    human_hi: float
    tau: float
    in_range: bool
    w1: float
    w1_normalised: float


@dataclass(frozen=True, slots=True)
class B4Result:
    variant: str
    regime: str
    complete: int
    discarded: int
    rows: tuple[B4MetricRow, ...]
    m0_sim_passes_a: bool
    m0_sim_mean_normalised_w1: float | None
    closed_loop: bool

    @property
    def discard_fraction(self) -> float:
        played = self.complete + self.discarded
        return self.discarded / played if played else 1.0

    @property
    def gating(self) -> bool:
        """§6.3 / §7 / A24: <= 10 % of the regime's games discarded and >= 20 complete."""
        return (
            self.discard_fraction <= config.B4_DISCARD_LIMIT
            and self.complete >= config.B4_MIN_GATING_GAMES
        )

    @property
    def metrics_in_range(self) -> int:
        return sum(r.in_range for r in self.rows)

    @property
    def pass_outcomes(self) -> bool:
        return self.metrics_in_range >= config.OUTCOME_METRICS_REQUIRED

    @property
    def mean_normalised_w1(self) -> float:
        return float(np.mean([r.w1_normalised for r in self.rows]))

    @property
    def beats_m0_sim(self) -> bool:
        return beats_m0_sim(self.mean_normalised_w1, self.m0_sim_mean_normalised_w1)

    @property
    def label(self) -> str:
        if not self.closed_loop:
            return "diagnostic (surrogate rejected on human rows)"
        if not self.pass_outcomes:
            return NOT_VALIDATED
        if self.m0_sim_passes_a and not self.beats_m0_sim:
            return NOT_BEATING_M0_SIM
        return VALIDATED_CLOSED

    @property
    def counts(self) -> bool:
        """Validated and gating: B4 is then a validated class there (never G2 alone, O4)."""
        return self.label == VALIDATED_CLOSED and self.gating

    def to_json(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "regime": self.regime,
            "games_complete": self.complete,
            "games_discarded": self.discarded,
            "discard_fraction": self.discard_fraction,
            "b4_gating": self.gating,
            "metrics_in_range": self.metrics_in_range,
            "pass_outcomes": self.pass_outcomes,
            "mean_normalised_w1": self.mean_normalised_w1,
            "m0_sim_passes_a": self.m0_sim_passes_a,
            "m0_sim_mean_normalised_w1": self.m0_sim_mean_normalised_w1,
            "beats_m0_sim": self.beats_m0_sim,
            "closed_loop": self.closed_loop,
            "label": self.label,
            "counts_as_validated": self.counts,
        }


def evaluate_b4(
    variant: str,
    regime: str,
    game_metrics: Mapping[str, npt.ArrayLike],
    discarded: int,
    human_units: Mapping[str, npt.ArrayLike],
    reference: Mapping[str, Mapping[str, float]],
    *,
    m0_sim_passes_a: bool,
    m0_sim_w1: float | None,
    closed_loop: bool,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.BOOTSTRAP_SEED,
) -> B4Result:
    """§8 criterion (a) and the A22 rule for one B4 (variant, regime).

    `game_metrics[k]`: the complete B4 games' metric k; `human_units[k]`: the regime's
    human unit values and `reference[k]` its CI (`lo`, `hi`), `tau` and `w1_scale`, all
    from the evaluation run. The B4 mean's CI resamples games (B = 10,000, a fresh
    default_rng(9001), one resample matrix for the 4 metrics).
    """
    values = {k: np.asarray(game_metrics[k], dtype=np.float64) for k in config.METRICS}
    n = values[config.METRICS[0]].shape[0]
    if n == 0:
        raise ValueError(f"B4 {variant} {regime}: no complete game")
    index = evaluate.bootstrap_indices(np.full(n, "b4"), n_resamples, np.random.default_rng(seed))
    rows: list[B4MetricRow] = []
    for k in config.METRICS:
        ref = reference[k]
        est = evaluate.mean_ci(values[k], index)
        w1 = float(stats.wasserstein_distance(values[k], np.asarray(human_units[k], dtype=float)))
        rows.append(
            B4MetricRow(
                metric=k,
                b4=est,
                human_lo=float(ref["lo"]),
                human_hi=float(ref["hi"]),
                tau=float(ref["tau"]),
                in_range=float(ref["lo"]) - float(ref["tau"])
                <= est.value
                <= float(ref["hi"]) + float(ref["tau"]),
                w1=w1,
                w1_normalised=w1 / float(ref["w1_scale"]),
            )
        )
    return B4Result(
        variant, regime, n, discarded, tuple(rows), m0_sim_passes_a, m0_sim_w1, closed_loop
    )


def b4_game_metrics(run_dir: Path) -> tuple[dict[str, FloatArray], int]:
    """Complete games' §8 metrics and the number of discarded games of a B4 run
    (games.json verified against the manifest)."""
    table = json.loads(verified_artifact(run_dir, "games.json").decode("utf-8"))
    cols = [c["name"] for c in table["columns"]]
    rows = [dict(zip(cols, r, strict=True)) for r in table["rows"]]
    complete = [r for r in rows if r["status"] == "complete"]
    source = {"players_active": "players_active_end"}
    out = {
        k: np.array([float(r[source.get(k, k)]) for r in complete], dtype=np.float64)
        for k in config.METRICS
    }
    return out, sum(1 for r in rows if r["status"] == "discarded")


def _human_units_by_regime(eval_run: Path) -> dict[str, dict[str, FloatArray]]:
    table = json.loads(verified_artifact(eval_run, HUMAN_UNITS_FILE).decode("utf-8"))
    cols = [c["name"] for c in table["columns"]]
    out: dict[str, dict[str, list[float]]] = {}
    for row in table["rows"]:
        r = dict(zip(cols, row, strict=True))
        regime = out.setdefault(str(r["regime"]), {k: [] for k in config.METRICS})
        for k in config.METRICS:
            regime[k].append(float(r[k]))
    return {g: {k: np.array(v) for k, v in m.items()} for g, m in out.items()}


def _input_problems(label: str, m: Mapping[str, Any], stage: str, protocol_sha: str) -> list[str]:
    c = m.get("citizens", {})
    problems: list[str] = []
    if c.get("stage") != stage:
        problems.append(f"{label}: stage {c.get('stage')!r}, expected {stage!r}")
    sha = c.get("frozen_protocol_sha256") or c.get("validation_protocol", {}).get("sha256")
    if sha != protocol_sha:
        problems.append(f"{label}: not made under the frozen protocol {protocol_sha}")
    if c.get("config_sha256") != config.config_sha256():
        problems.append(f"{label}: config_sha256 differs from the current configuration")
    return problems


def b4_evaluate_command(args: argparse.Namespace) -> int:
    """A24: B4-plain and B4-calibrated held-out runs against the evaluation run."""
    protocol, commit = require_post_freeze(bool(args.confirm_frozen))
    if args.evaluation_run is None or not args.b4_run:
        raise SystemExit("--evaluation-run and at least one --b4-run are required")
    out_dir = _new_run_dir(args.runs_dir, args.run_id)
    ev_m = _read_manifest(args.evaluation_run)
    problems = _input_problems("evaluation run", ev_m, "c2_heldout_eval", protocol.sha256)
    if ev_m.get("source", {}).get("git_dirty"):
        problems.append("evaluation run: made from a dirty tree")
    b4_manifests = {path: _read_manifest(path) for path in args.b4_run}
    for path, m in b4_manifests.items():
        problems += _input_problems(f"B4 run {path.name}", m, "c2_b4_heldout", protocol.sha256)
    if problems:
        raise ValueError("B4 evaluation inputs refused:\n- " + "\n- ".join(problems))
    human = _human_units_by_regime(args.evaluation_run)
    reference = ev_m["citizens"]["human_reference"]
    comparison = ev_m["citizens"]["g2"]["w1_comparison"]
    tier = ev_m["citizens"]["closed_loop_tier"]
    results: list[B4Result] = []
    seen: set[tuple[str, str]] = set()
    for path, m in b4_manifests.items():
        variant = str(m["citizens"]["config"]["variant"])
        regime = B4_REGIMES[str(m["citizens"]["mechanism_key"])]
        if (variant, regime) in seen:
            raise ValueError(f"two B4 runs for ({variant}, {regime})")
        seen.add((variant, regime))
        values, discarded = b4_game_metrics(path)
        results.append(
            evaluate_b4(
                variant,
                regime,
                values,
                discarded,
                human[regime],
                reference[regime],
                m0_sim_passes_a=bool(comparison[regime]["m0_sim_passes_a"]),
                m0_sim_w1=comparison[regime]["m0_sim_mean_normalised_w1"],
                closed_loop=bool(tier[regime]),
            )
        )
    run_id = out_dir.name
    manifest = _base_manifest(
        run_id,
        "citizens-c2-b4-eval-v1",
        "c2_b4_eval",
        protocol,
        commit,
        simulator=None,
        random_seed=config.BOOTSTRAP_SEED,
        access="none (human unit metrics from the evaluation run)",
    )
    checks = [
        Check("input_checksums_verified", 0.0, 0.0,
              "evaluation-run human units and B4 games.json loaded by SHA-256 (a mismatch "
              "aborts)"),
    ]  # fmt: skip
    manifest["validation"]["checks"] = [c.to_json() for c in checks]
    manifest["status"] = run_status(checks)
    manifest["citizens"]["inputs"] = {
        "evaluation_run": {
            "run_id": args.evaluation_run.name,
            "manifest_sha256": _manifest_sha256(args.evaluation_run),
        },
        "b4_runs": [
            {"run_id": p.name, "manifest_sha256": _manifest_sha256(p)} for p in b4_manifests
        ],
    }
    manifest["citizens"]["b4"] = [r.to_json() for r in results]
    manifest["limitations"] = [
        "B4 has about 40 simulated games per regime against 1,000 for M0-sim; the W1 of a "
        "small sample is biased upward, so the A22 comparison is conservative for B4.",
        "B4 never decides G2 (O4); a validated and gating B4 label is reported as such.",
    ]
    rows = [
        [r.variant, r.regime, row.metric, row.b4.value, row.b4.lo, row.b4.hi, row.human_lo,
         row.human_hi, row.tau, "true" if row.in_range else "false", row.w1, row.w1_normalised]
        for r in results
        for row in r.rows
    ]  # fmt: skip
    columns = [
        ("variant", None, "plain or calibrated"),
        ("regime", None, "gating regime"),
        ("metric", None, "§8 metric"),
        ("b4_mean", None, "mean over complete B4 games"),
        ("b4_ci_lo", None, "2.5 % percentile of the game bootstrap (B=10000, seed 9001)"),
        ("b4_ci_hi", None, "97.5 % percentile"),
        ("human_ci_lo", None, "human CI from the evaluation run"),
        ("human_ci_hi", None, "human CI from the evaluation run"),
        ("tolerance", None, "tau"),
        ("in_range", None, "lo - tau <= b4_mean <= hi + tau"),
        ("w1", None, "W1 between B4 games and human units"),
        ("w1_normalised", "dimensionless", "w1 / the evaluation run's w1_scale (A22)"),
    ]
    files = [("sample_table", "b4_outcomes.json", _table(run_id, columns, rows))]
    write_run_directory(out_dir, manifest, files)
    print(json.dumps(manifest["citizens"]["b4"], indent=1))
    print(f"{out_dir}: status {manifest['status']}")
    return 0


def b4_weights_command(args: argparse.Namespace) -> int:
    """Print the B4-calibrated weights (training data and calibration games only)."""
    print(json.dumps(type_weights(args.calibration_run, args.processed_dir), indent=1))
    return 0


# --- command line ---------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.heldout")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    g1b = sub.add_parser("g1b", help="G1b: held-out z-values and the pool anomaly report")
    g1b.add_argument("--confirm-frozen", action="store_true")
    g1b.add_argument("--run-id", default="citizens-c1-g1b-v1")
    ev = sub.add_parser("evaluate", help="the frozen held-out evaluation (evidence run)")
    ev.add_argument("--confirm-frozen", action="store_true")
    ev.add_argument("--fitted-run", type=Path)
    ev.add_argument("--surrogate-run", type=Path)
    ev.add_argument("--g1b-run", type=Path)
    ev.add_argument("--run-id", default="citizens-c2-heldout-eval-v1")
    ev.add_argument(
        "--owner-override-surrogate-reproduction",
        metavar="REASON",
        default=None,
        help="the owner's recorded reason to accept a surrogate run that did not reproduce "
        "the pre-freeze fit bit for bit (A25)",
    )
    b4 = sub.add_parser("b4", help="one B4 held-out run (resumable, long)")
    b4.add_argument("--confirm-frozen", action="store_true")
    b4.add_argument("--variant", choices=("plain", "calibrated"), required=True)
    b4.add_argument("--mechanism", choices=B4_MECHANISMS, required=True)
    b4.add_argument("--surrogate-run", type=Path)
    b4.add_argument("--calibration-run", type=Path)
    b4.add_argument("--games", type=int, default=config.B4_EVAL_GAMES)
    b4.add_argument("--workers", type=int, default=8)
    b4.add_argument("--run-id", required=True)
    be = sub.add_parser("b4-evaluate", help="B4 runs against the evaluation run (A24)")
    be.add_argument("--confirm-frozen", action="store_true")
    be.add_argument("--evaluation-run", type=Path)
    be.add_argument("--b4-run", type=Path, action="append", default=[])
    be.add_argument("--run-id", default="citizens-c2-b4-eval-v1")
    bw = sub.add_parser("b4-weights", help="print the B4-calibrated type weights")
    bw.add_argument("--calibration-run", type=Path, required=True)
    args = parser.parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "g1b": g1b_command,
        "evaluate": evaluate_command,
        "b4": b4_command,
        "b4-evaluate": b4_evaluate_command,
        "b4-weights": b4_weights_command,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
