"""Fit the C2 citizen models on the training fit split and write immutable runs.

Commands::

    uv run python -m phasespace.citizens.fit fit   --run-id <id>   # fitted-model run
    uv run python -m phasespace.citizens.fit smoke --run-id <id>   # fit + training dry run

`fit` fits M0, P0, B1 (4 members), B2, B3 and B23 exactly as protocol 6 prescribes (the
fit split only, validation split for B1's early stopping and for reporting; no refit on
fit + validation) and writes the fitted parameters, the B1 checkpoints (.npz) and the
fit/validation-split likelihoods. `smoke` (pre-freeze, protocol 10.1) does the same, then
reloads the models from the serialised artifacts and runs the full evaluation machinery
of protocol 8 on the *training* regimes (Equal, Mixed, Proportional): (i) a dry run of
the criterion on the validation split's human groups and (ii) the in-sample check on all
40-round training groups, with S = 1,000 simulated games per (model, mechanism). It also
times S = 1,000 games under the exact Interpolating rule (timing and feasibility only; no
simulated outcome of a held-out mechanism is computed or recorded), and projects the
non-LLM compute of the whole protocol against its 12 h ceiling.

Only training data are read (`data.load_training`); no held-out human game is touched.
Runs record the SHA-256 of the validation protocol text they ran under together with its
header status (``draft`` before the freeze), and the C2 `config_sha256`. A run made
before the freeze is always `computed_unvalidated`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.citizens import config, data, evaluate, validation
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
from phasespace.citizens.models import b2_condcoop as b2
from phasespace.citizens.models import b3_ewa as b3
from phasespace.citizens.models.b1_gru import B1GRUEnsemble, GRUWeights
from phasespace.citizens.models.base import (
    CitizenModel,
    Streams,
    streams_from_games,
    teacher_forced_log_prob,
)
from phasespace.citizens.models.baselines import (
    B0Rational,
    M0Marginal,
    P0Persistence,
    fit_m0,
    fit_p0,
)
from phasespace.citizens.records import Games
from phasespace.citizens.simulate import EXACT_MECHANISMS, simulate
from phasespace.citizens.split import FIT, VALIDATION, Split, Unit, human_training_games, make_split
from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    validate_run_id,
    write_run_directory,
)

SCHEMA_VERSION: Final = 1
STUDY_ID: Final = "model-citizens"
GENERATOR: Final = "phasespace.citizens.fit"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"
DEFAULT_RUNS_DIR: Final = Path("runs")
DEFAULT_FIT_RUN_ID: Final = "citizens-c2-fit-v1-dev"
DEFAULT_SMOKE_RUN_ID: Final = "citizens-c2-smoke-v1-dev"
MODELS_FILE: Final = "models.json"
SPLIT_FILE: Final = "split.json"
LIKELIHOOD_FILE: Final = "likelihoods.json"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- fitting --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitSettings:
    b1_seeds: tuple[int, ...] = config.B1_SEEDS
    b1_max_updates: int = config.B1_MAX_UPDATES
    b2_seeds: tuple[int, ...] = config.B2_SEEDS
    b3_seeds: tuple[int, ...] = config.B3_SEEDS
    b23_seeds: tuple[int, ...] = config.B23_SEEDS

    @property
    def is_protocol(self) -> bool:
        return self == FitSettings()


@dataclass(frozen=True, slots=True)
class B1Member:
    seed: int
    weights: GRUWeights
    summary: dict[str, Any]  # best_update, best_val_nll, updates, stopped_early, seconds, history


@dataclass(slots=True)
class Fitted:
    """Everything fitted on the fit split (parameters, checkpoints, every start)."""

    split: Split
    m0_bins: np.ndarray[Any, np.dtype[np.float64]]
    p0_q: float
    ewa_p0: np.ndarray[Any, np.dtype[np.float64]]
    b1: list[B1Member]
    b2_starts: list[dict[str, Any]]
    b3_starts: list[dict[str, Any]]
    b23_starts: list[dict[str, Any]]
    settings: FitSettings
    seconds: dict[str, float] = field(default_factory=dict[str, float])

    @staticmethod
    def best(starts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """The start with the lowest fit-split NLL (ties: first seed)."""
        return min(starts, key=lambda s: (s["mean_nll"], s["seed"]))

    def likelihood_models(self) -> dict[str, CitizenModel[Any]]:
        m0 = M0Marginal(self.m0_bins)
        return {
            "B0": B0Rational(),
            "M0": m0,
            "P0": P0Persistence(self.m0_bins, self.p0_q),
            "B1": B1GRUEnsemble([m.weights for m in self.b1], [m.seed for m in self.b1]),
            "B2": b2.B2ConditionalCooperation(self.best(self.b2_starts)["theta"]),
            "B3": b3.EWAModel(self.best(self.b3_starts)["params"], self.ewa_p0, hybrid=False),
            "B23": b3.EWAModel(self.best(self.b23_starts)["params"], self.ewa_p0, hybrid=True),
        }

    def sim_models(self) -> dict[str, CitizenModel[Any]]:
        lik = self.likelihood_models()
        return {
            "B0": lik["B0"],
            "M0-sim": lik["M0"],
            "B1": lik["B1"],
            "B2": lik["B2"],
            "B3": lik["B3"],
            "B23": lik["B23"],
        }


def training_streams(training: Games, split: Split) -> tuple[Games, Games]:
    human = human_training_games(training)
    return human.select(split.mask(human, FIT)), human.select(split.mask(human, VALIDATION))


def fit_all(training: Games, settings: FitSettings | None = None) -> Fitted:
    """Fit every model on the fit split (protocol 6); timings in `seconds`."""
    from phasespace.citizens.models import b1_train  # JAX is imported only to train B1

    settings = FitSettings() if settings is None else settings
    seconds: dict[str, float] = {}
    split = make_split(training)
    fit_games, val_games = training_streams(training, split)
    fit_s, val_s = streams_from_games(fit_games), streams_from_games(val_games)

    start = time.perf_counter()
    m0_bins = fit_m0(fit_s)
    q, _ = fit_p0(m0_bins, fit_s)
    seconds["M0+P0"] = time.perf_counter() - start
    _log(f"M0/P0 fitted (q = {q:.4f})")

    start = time.perf_counter()
    d2 = b2.FitData.from_streams(fit_s)
    b2_starts = [asdict(r) for r in b2.fit(d2, settings.b2_seeds)]
    seconds["B2"] = time.perf_counter() - start
    _log(f"B2: {len(b2_starts)} starts in {seconds['B2']:.0f} s")

    start = time.perf_counter()
    ewa_p0 = b3.round1_levels(fit_s)
    d3 = b3.FitData.from_streams(fit_s, ewa_p0)
    b3_starts = [asdict(r) for r in b3.fit(d3, hybrid=False, seeds=settings.b3_seeds)]
    seconds["B3"] = time.perf_counter() - start
    _log(f"B3: {len(b3_starts)} starts in {seconds['B3']:.0f} s")
    start = time.perf_counter()
    b23_starts = [asdict(r) for r in b3.fit(d3, hybrid=True, seeds=settings.b23_seeds)]
    seconds["B23"] = time.perf_counter() - start
    _log(f"B23: {len(b23_starts)} starts in {seconds['B23']:.0f} s")

    start = time.perf_counter()
    fit_seq = b1_train.SequenceData.from_streams(fit_s)
    val_seq = b1_train.SequenceData.from_streams(val_s)
    members: list[B1Member] = []
    for seed in settings.b1_seeds:
        r = b1_train.train_member(fit_seq, val_seq, seed, max_updates=settings.b1_max_updates)
        summary = {
            "best_update": r.best_update,
            "best_val_nll": r.best_val_nll,
            "updates": r.updates,
            "stopped_early": r.stopped_early,
            "seconds": r.seconds,
            "history": r.history,
        }
        members.append(B1Member(seed=seed, weights=r.weights, summary=summary))
        _log(
            f"B1 seed {seed}: best val NLL {r.best_val_nll:.4f} at update {r.best_update} "
            f"({r.updates} updates, {r.seconds:.0f} s)"
        )
    seconds["B1"] = time.perf_counter() - start
    return Fitted(
        split=split,
        m0_bins=m0_bins,
        p0_q=q,
        ewa_p0=ewa_p0,
        b1=members,
        b2_starts=b2_starts,
        b3_starts=b3_starts,
        b23_starts=b23_starts,
        settings=settings,
        seconds=seconds,
    )


# --- serialisation ---------------------------------------------------------------------------


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


def member_file(seed: int) -> str:
    return f"b1-member-{seed}.npz"


def models_document(fitted: Fitted) -> dict[str, Any]:
    def best_block(starts: list[dict[str, Any]], key: str, names: Sequence[str]) -> dict[str, Any]:
        best = Fitted.best(starts)
        return {
            "best_seed": best["seed"],
            "best_mean_nll_fit": best["mean_nll"],
            key: best[key],
            "names": list(names),
            "starts": starts,
        }

    return {
        "settings": asdict(fitted.settings),
        "protocol_settings": fitted.settings.is_protocol,
        "b0": {"eps": config.B0_EPS, "closed_loop": "c = 0"},
        "m0": {"p_bins": fitted.m0_bins.tolist(), "smoothing": config.M0_SMOOTHING},
        "p0": {"q": fitted.p0_q},
        "ewa_p0_levels": fitted.ewa_p0.tolist(),
        "b1": {
            "members": [
                {"seed": m.seed, "file": member_file(m.seed), **m.summary} for m in fitted.b1
            ],
            "likelihood": "mean of the members' integer pmfs",
            "closed_loop": "one member per simulated player, uniform with replacement",
        },
        "b2": best_block(fitted.b2_starts, "theta", b2.PARAM_NAMES),
        "b3": best_block(fitted.b3_starts, "params", b3.B3_NAMES),
        "b23": best_block(fitted.b23_starts, "params", b3.B23_NAMES),
        "seconds": fitted.seconds,
    }


SPLIT_COLUMNS: Final = [
    ("mechanism", None, "stratum mechanism"),
    ("cohort", None, "stratum cohort (Exp 4: a 3-game group is one unit)"),
    ("launch_id", None, "unit key (Exp 4: shared by the group's Games 1-3)"),
    ("split", None, "fit or validation (protocol 6.1, default_rng(20261020))"),
]


def fitted_files(fitted: Fitted, run_id: str) -> list[tuple[str, str, bytes]]:
    files: list[tuple[str, str, bytes]] = [
        ("sample_table", SPLIT_FILE, _table(run_id, SPLIT_COLUMNS, fitted.split.to_rows())),
        ("fitted_parameters", MODELS_FILE, dumps_json(models_document(fitted))),
    ]
    for m in fitted.b1:
        files.append(("checkpoint", member_file(m.seed), m.weights.to_npz_bytes()))
    return files


def fitted_from_files(files: Mapping[str, bytes]) -> Fitted:
    """Rebuild `Fitted` from a run's artifact bytes (split table, models.json, checkpoints)."""
    doc = json.loads(files[MODELS_FILE])
    split_rows = json.loads(files[SPLIT_FILE])["rows"]
    split = Split(units=tuple(Unit(m, c, k, s) for m, c, k, s in split_rows))
    members = [
        B1Member(
            seed=int(m["seed"]),
            weights=GRUWeights.from_npz_bytes(files[m["file"]]),
            summary={k: v for k, v in m.items() if k not in ("seed", "file")},
        )
        for m in doc["b1"]["members"]
    ]
    s = doc["settings"]
    return Fitted(
        split=split,
        m0_bins=np.asarray(doc["m0"]["p_bins"], dtype=np.float64),
        p0_q=float(doc["p0"]["q"]),
        ewa_p0=np.asarray(doc["ewa_p0_levels"], dtype=np.float64),
        b1=members,
        b2_starts=doc["b2"]["starts"],
        b3_starts=doc["b3"]["starts"],
        b23_starts=doc["b23"]["starts"],
        settings=FitSettings(
            b1_seeds=tuple(s["b1_seeds"]),
            b1_max_updates=int(s["b1_max_updates"]),
            b2_seeds=tuple(s["b2_seeds"]),
            b3_seeds=tuple(s["b3_seeds"]),
            b23_seeds=tuple(s["b23_seeds"]),
        ),
        seconds=dict(doc["seconds"]),
    )


def load_fitted(run_dir: Path) -> Fitted:
    """Load a fitted-model run, verifying every artifact's SHA-256 against its manifest."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    files: dict[str, bytes] = {}
    for artifact in manifest["artifacts"]:
        payload = (run_dir / artifact["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise ValueError(f"{run_dir / artifact['path']}: SHA-256 differs from the manifest")
        files[artifact["path"]] = payload
    return fitted_from_files(files)


# --- likelihood tables ---------------------------------------------------------------------


def split_likelihoods(
    models: Mapping[str, CitizenModel[Any]], streams: Mapping[str, Streams]
) -> list[list[Any]]:
    """Rows (model, split, mean log P, Delta-ell vs M0, decisions) per model and split."""
    rows: list[list[Any]] = []
    for split_name, s in streams.items():
        n = int(s.scored.sum())
        means = {
            name: float(np.nansum(teacher_forced_log_prob(m, s)) / n) for name, m in models.items()
        }
        for name, value in means.items():
            rows.append([name, split_name, value, value - means["M0"], n])
    return rows


LIKELIHOOD_COLUMNS: Final = [
    ("model", None, "model"),
    ("split", None, "fit or validation split of the human training games"),
    ("log_lik", "nat", "mean log P(c) per non-forced decision (teacher-forced)"),
    ("delta_ell", "nat", "log_lik - log_lik(M0) on the same split"),
    ("decisions", "count", "non-forced decisions of the split"),
]


# --- manifest ----------------------------------------------------------------------------------


def protocol_record() -> dict[str, Any]:
    """Path, SHA-256 and header status of the validation protocol text (draft or frozen)."""
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    content = path.read_bytes()
    return {
        "path": data.VALIDATION_PROTOCOL_PATH.as_posix(),
        "version": config.PROTOCOL_VERSION,
        "status": data.header_status(content.decode("utf-8")),
        "sha256": hashlib.sha256(content).hexdigest(),
        "held_out_access": "none (training data only; data.load_heldout is never called)",
    }


def model_block() -> dict[str, Any]:
    return {
        "name": "koster2025-common-pool",
        "units": "coin",
        "players": N_PLAYERS,
        "max_pool": MAX_POOL,
        "growth_factor": GROWTH_FACTOR,
        "pool_floor": POOL_FLOOR,
        "depletion_threshold": DEPLETION_THRESHOLD,
        "exclusion_threshold": EXCLUSION_THRESHOLD,
        "rounds_exps_1_3": N_ROUNDS,
        "baseline_weights": {"equal": W_EQUAL, "mixed": W_MIXED, "proportional": W_PROPORTIONAL},
        "interpolating_k": K_INTERPOLATING,
        "interpolating_weight_decimals": INTERPOLATING_DECIMALS,
        "recorded_offer_withheld": RECORDED_WITHHELD,
        "recorded_offer_decimals": RECORDED_DECIMALS,
        "offer_rule": DEFAULT_OFFER_RULE,
        "legacy_remainder_dropped_games": len(validation.LEGACY_REMAINDER_DROPPED),
    }


def simulator_block(games_per_cell: int) -> dict[str, Any]:
    return {
        "name": "phasespace.citizens.simulate",
        "pool_rule": "full: R' = max(0.01, min(200, R - sum e + 1.4 sum c))",
        "offer_rule": DEFAULT_OFFER_RULE,
        "rounds": config.SIM_ROUNDS,
        "games_per_model_mechanism": games_per_cell,
        "seed_rule": "50000 + 100 m + r (protocol 7)",
        "feasibility_atol": config.SIM_FEASIBILITY_ATOL,
    }


def _environment() -> dict[str, str]:
    env = environment_info()
    for package in ("jax", "jaxlib", "equinox", "optax"):
        env[package] = version(package)
    return env


def _split_summary(fitted: Fitted, fit_s: Streams, val_s: Streams) -> dict[str, Any]:
    units = fitted.split.units
    return {
        "seed": config.SPLIT_SEED,
        "units_fit": sum(u.split == FIT for u in units),
        "units_validation": sum(u.split == VALIDATION for u in units),
        "games_fit": int(np.unique(fit_s.game).size),
        "games_validation": int(np.unique(val_s.game).size),
        "decisions_fit": int(fit_s.scored.sum()),
        "decisions_validation": int(val_s.scored.sum()),
        "stream_order": "draws per stratum in config.SPLIT_STRATA order, one generator",
    }


def base_manifest(
    run_id: str,
    stage: str,
    *,
    created_at: datetime,
    git_commit: str,
    git_dirty: bool,
    training_sidecar: Mapping[str, Any],
    simulator: dict[str, Any] | None,
) -> dict[str, Any]:
    timestamp = created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": f"citizens-c2-{stage}-v1",
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": {
            "repository": REPOSITORY,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "environment": _environment(),
        "model": model_block(),
        "simulator": simulator,
        "random_seed": config.SPLIT_SEED,
        "artifacts": [],
        "metrics": {},
        "limitations": [],
        "validation": {
            "checks": [],
            "validated_by": f"{GENERATOR} automated checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
        "citizens": {
            "stage": f"c2_{stage}",
            "config_sha256": config.config_sha256(),
            "config": config.config_dict(),
            "validation_protocol": protocol_record(),
            "dataset": {
                "source_sha256": training_sidecar["source_sha256"],
                "parser_version": training_sidecar["parser_version"],
                "processed_training_sha256": training_sidecar["file_sha256"],
            },
            "seeds": {
                "split": config.SPLIT_SEED,
                "b1": list(config.B1_SEEDS),
                "b2": list(config.B2_SEEDS),
                "b3": list(config.B3_SEEDS),
                "b23": list(config.B23_SEEDS),
                "bootstrap": config.BOOTSTRAP_SEED,
                "simulation": "config.sim_seed(model, regime) = 50000 + 100 m + r",
            },
        },
    }


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


PRE_FREEZE_LIMITATIONS: Final = [
    "Pre-freeze run: the validation protocol is a draft (citizens.validation_protocol.status); "
    "the status is computed_unvalidated by rule, whatever the checks say.",
    "Training data only: held-out human games (Interpolating, M1, M2) were not read.",
]


def k1_check(likelihood_rows: list[list[Any]]) -> Check:
    val = {r[0]: r[2] for r in likelihood_rows if r[1] == VALIDATION}
    return Check(
        "k1_b1_validation_nll_minus_m0",
        float(-val["B1"] + val["M0"]),
        0.0,
        "validation-split mean NLL of B1 minus that of M0 (stop rule K1: B1 must beat M0; "
        "pass: observed <= 0)",
    )


def split_check(fitted: Fitted) -> Check:
    units = fitted.split.units
    fit_n = sum(u.split == FIT for u in units)
    val_n = sum(u.split == VALIDATION for u in units)
    expected = len(config.SPLIT_STRATA)
    diff = abs(fit_n - expected * config.FIT_UNITS_PER_STRATUM) + abs(
        val_n - expected * config.VAL_UNITS_PER_STRATUM
    )
    return Check(
        "split_unit_count_difference",
        float(diff),
        0.0,
        "|fit units - 160| + |validation units - 40| (protocol 6.1)",
    )


def write_fit_run(
    out_dir: Path, fitted: Fitted, training: Games, sidecar: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = out_dir.name
    validate_run_id(run_id)
    commit, dirty = git_source()
    fit_games, val_games = training_streams(training, fitted.split)
    fit_s, val_s = streams_from_games(fit_games), streams_from_games(val_games)
    lik_rows = split_likelihoods(fitted.likelihood_models(), {FIT: fit_s, VALIDATION: val_s})
    manifest = base_manifest(
        run_id,
        "fit",
        created_at=datetime.now(UTC),
        git_commit=commit,
        git_dirty=dirty,
        training_sidecar=sidecar,
        simulator=None,
    )
    _fill_fit_sections(manifest, fitted, lik_rows, fit_s, val_s)
    files = [
        *fitted_files(fitted, run_id),
        ("sample_table", LIKELIHOOD_FILE, _table(run_id, LIKELIHOOD_COLUMNS, lik_rows)),
    ]
    return write_run_directory(out_dir, manifest, files)


def _fill_fit_sections(
    manifest: dict[str, Any],
    fitted: Fitted,
    lik_rows: list[list[Any]],
    fit_s: Streams,
    val_s: Streams,
) -> None:
    checks = [split_check(fitted), k1_check(lik_rows)]
    manifest["validation"]["checks"] = [c.to_json() for c in checks]
    manifest["citizens"]["split"] = _split_summary(fitted, fit_s, val_s)
    manifest["citizens"]["fit_settings"] = {
        **asdict(fitted.settings),
        "protocol_settings": fitted.settings.is_protocol,
        "lbfgsb_options": {
            "maxiter": config.LBFGSB_MAXITER,
            "maxfun": config.LBFGSB_MAXFUN,
            "ftol": config.LBFGSB_FTOL,
            "gtol": config.LBFGSB_GTOL,
            "gradient": "2-point finite differences",
        },
        "b1_early_stopping_score": config.B1_EARLY_STOPPING_SCORE,
        "b1_batching": config.B1_BATCHING,
    }
    for row in lik_rows:
        name, split_name, value, delta, _ = row
        key = f"{name.lower()}_{split_name}_log_lik"
        manifest["metrics"][key] = _metric(
            value, "nat", f"{name} mean log P per decision, {split_name} split"
        )
        if name != "M0":
            manifest["metrics"][f"{name.lower()}_{split_name}_delta_ell"] = _metric(
                delta, "nat", f"{name} log_lik - M0 log_lik, {split_name} split"
            )
    for name, sec in fitted.seconds.items():
        manifest["metrics"][f"fit_seconds_{name.lower().replace('+', '_')}"] = _metric(
            sec, "s", f"wall time to fit {name}"
        )
    frozen = manifest["citizens"]["validation_protocol"]["status"] == data.FROZEN
    limitations = [PRE_FREEZE_LIMITATIONS[1]] if frozen else list(PRE_FREEZE_LIMITATIONS)
    if not fitted.settings.is_protocol:
        limitations.append(
            "Reduced fit settings (citizens.fit_settings.protocol_settings = false): not the "
            "protocol's fitted models."
        )
    if manifest["source"]["git_dirty"]:
        limitations.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    manifest["limitations"] = limitations


# --- smoke ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmokeOutputs:
    dry_run: list[evaluate.RegimeResult]
    in_sample: list[evaluate.RegimeResult]
    sim_seconds: dict[str, dict[str, float]]  # model -> mechanism -> s per S games
    interpolating_games: int
    projection: dict[str, float]


def run_training_evaluations(
    fitted: Fitted, training: Games, *, n_sim: int, n_resamples: int
) -> SmokeOutputs:
    lik = fitted.likelihood_models()
    sims = fitted.sim_models()
    human = human_training_games(training)
    val_mask = fitted.split.mask(human, VALIDATION)
    dry: list[evaluate.RegimeResult] = []
    ins: list[evaluate.RegimeResult] = []
    sim_seconds: dict[str, dict[str, float]] = {name: {} for name in sims}
    for regime, (mechanism, cohorts) in evaluate.TRAINING_REGIMES.items():
        mech = EXACT_MECHANISMS[regime]
        all_games = evaluate.regime_games(human, mechanism, cohorts)
        val_games = evaluate.regime_games(human.select(val_mask), mechanism, cohorts)
        _log(f"{regime}: dry run on {val_games.n_games} validation games")
        result, _ = evaluate.evaluate_regime(
            regime,
            regime,
            val_games,
            likelihood_models=lik,
            sim_models=sims,
            mechanism=mech,
            n_sim=n_sim,
            n_resamples=n_resamples,
        )
        dry.append(result)
        for m in result.models:
            if m.sim_seconds is not None:
                sim_seconds[m.model][regime] = m.sim_seconds
        _log(f"{regime}: in-sample check on {all_games.n_games} games")
        result_in, _ = evaluate.evaluate_regime(
            regime,
            regime,
            all_games,
            likelihood_models=lik,
            sim_models=sims,
            mechanism=mech,
            n_sim=n_sim,
            n_resamples=n_resamples,
        )
        ins.append(result_in)
    # Interpolating: timing and feasibility only (simulate raises on an infeasible game).
    n_interp = 0
    for name, model in sims.items():
        rng = np.random.default_rng(config.sim_seed(name, "Interpolating"))
        start = time.perf_counter()
        games = simulate(model, EXACT_MECHANISMS["Interpolating"], n_sim, rng)
        sim_seconds[name]["Interpolating"] = time.perf_counter() - start
        n_interp += games.n_games
        del games
    return SmokeOutputs(
        dry_run=dry,
        in_sample=ins,
        sim_seconds=sim_seconds,
        interpolating_games=n_interp,
        projection=project_compute(fitted, sim_seconds, n_sim),
    )


def project_compute(
    fitted: Fitted, sim_seconds: Mapping[str, Mapping[str, float]], n_sim: int
) -> dict[str, float]:
    """Projected non-LLM wall time of the full protocol (hours), from measured costs.

    Simulation: per model, the mean measured seconds per game times the protocol's
    games: S = 1000 for each of Interpolating, M1, M2, Equal, Mixed, Proportional and
    3 x 1000 for M2 Exp 4 (1000 sets of 3 games of ~40 rounds) = 9000 games. The M1/M2
    surrogate's per-round cost is not measured here and is not included. Fitting: B2/B3/B23
    as measured (all starts); B1 as measured (early stopping) and as the worst case of
    4 x 20,000 updates at the measured seconds per update. Surrogates (not built here):
    estimated as 12 x 20,000 updates at the B1 seconds per update (an upper bound for the
    smaller GRU 32 network, protocol 6.4: <= 2 x 6 x 20,000 updates).
    """
    games_per_model = 9 * config.SIM_GAMES
    sim_hours = 0.0
    for per_mech in sim_seconds.values():
        if per_mech:
            per_game = float(np.mean(list(per_mech.values()))) / n_sim
            sim_hours += per_game * games_per_model / 3600.0
    updates = sum(int(m.summary["updates"]) for m in fitted.b1)
    b1_seconds = sum(float(m.summary["seconds"]) for m in fitted.b1)
    per_update = b1_seconds / max(updates, 1)
    b1_worst = per_update * len(config.B1_SEEDS) * config.B1_MAX_UPDATES / 3600.0
    other_fit = sum(v for k, v in fitted.seconds.items() if k != "B1") / 3600.0
    surrogate = per_update * 12 * config.SURROGATE_MAX_UPDATES / 3600.0
    measured_fit = sum(fitted.seconds.values()) / 3600.0
    return {
        "fit_measured_hours": measured_fit,
        "b1_seconds_per_update": per_update,
        "b1_worst_case_hours": b1_worst,
        "surrogate_estimate_hours": surrogate,
        "simulation_hours": sim_hours,
        "total_projected_hours": other_fit + b1_worst + surrogate + sim_hours,
        "ceiling_hours": config.COMPUTE_CEILING_HOURS,
    }


COMPUTE_DEFINITIONS: Final = {
    "fit_measured_hours": "measured wall time of all fits in this run",
    "b1_seconds_per_update": "measured B1 training seconds per update (incl. evaluations)",
    "b1_worst_case_hours": "4 members x 20,000 updates at the measured seconds per update",
    "surrogate_estimate_hours": "12 x 20,000 surrogate updates at the B1 seconds per update "
    "(upper-bound estimate; surrogates are not built in this run)",
    "simulation_hours": "per model, measured seconds per simulated game x 9,000 games "
    "(S = 1000 for 6 mechanisms + 3,000 M2 Exp 4 games); surrogate cost per round excluded",
    "total_projected_hours": "fits other than B1 (measured) + B1 worst case + surrogate "
    "estimate + simulation",
    "ceiling_hours": "protocol 7 non-LLM compute ceiling",
}


def write_smoke_run(
    out_dir: Path,
    fitted: Fitted,
    training: Games,
    sidecar: Mapping[str, Any],
    *,
    n_sim: int,
    n_resamples: int,
) -> dict[str, Any]:
    run_id = out_dir.name
    validate_run_id(run_id)
    commit, dirty = git_source()
    files = fitted_files(fitted, run_id)
    # Evaluate the models as reloaded from the serialised artifacts (checks the round trip).
    reloaded = fitted_from_files({path: payload for _, path, payload in files})
    fit_games, val_games = training_streams(training, reloaded.split)
    fit_s, val_s = streams_from_games(fit_games), streams_from_games(val_games)
    lik_rows = split_likelihoods(reloaded.likelihood_models(), {FIT: fit_s, VALIDATION: val_s})
    smoke = run_training_evaluations(reloaded, training, n_sim=n_sim, n_resamples=n_resamples)
    manifest = base_manifest(
        run_id,
        "smoke",
        created_at=datetime.now(UTC),
        git_commit=commit,
        git_dirty=dirty,
        training_sidecar=sidecar,
        simulator=simulator_block(n_sim),
    )
    _fill_fit_sections(manifest, fitted, lik_rows, fit_s, val_s)
    simulated = sum(r.feasible_games for r in smoke.dry_run + smoke.in_sample)
    manifest["validation"]["checks"].append(
        Check(
            "infeasible_simulated_games",
            0.0,
            0.0,
            f"simulated games violating protocol 5 (any violation raises and aborts the "
            f"run, K2); {simulated + smoke.interpolating_games} games simulated",
        ).to_json()
    )
    b0_rejected = all(
        m.validated is False for r in smoke.dry_run for m in r.models if m.model == "B0"
    )
    manifest["validation"]["checks"].append(
        Check(
            "b0_not_rejected_training_dry_run",
            0.0 if b0_rejected else 1.0,
            0.0,
            "1 if the rational control B0 passes the criterion on any training regime of the "
            "validation-split dry run, else 0 (sanity check, not a gating result)",
        ).to_json()
    )
    for name, value in smoke.projection.items():
        unit = "s" if name.endswith("per_update") else "h"
        manifest["metrics"][f"compute_{name}"] = _metric(value, unit, COMPUTE_DEFINITIONS[name])
    for model, per in smoke.sim_seconds.items():
        for mech, sec in per.items():
            manifest["metrics"][f"sim_seconds_{model.lower()}_{mech.lower()}"] = _metric(
                sec, "s", f"wall time of {n_sim} simulated {mech} games of {model}"
            )
    manifest["limitations"].extend(
        [
            "Smoke / dry run of the criterion on TRAINING regimes only (Equal, Mixed, "
            "Proportional): a sanity check of the machinery, not evidence of validity. "
            "The dry run uses the validation split's 40-round human groups (Exp 1 and 2; "
            "Exp 4 sets are not 40-round games); the in-sample table uses every 40-round "
            "training group, including fit-split ones.",
            "Interpolating games were simulated for timing and feasibility only; no "
            "simulated outcome under a held-out mechanism is computed or recorded.",
            "The compute projection excludes the M1/M2 surrogate's per-round simulation "
            "cost and estimates surrogate training from the B1 per-update cost.",
        ]
    )
    manifest["citizens"]["smoke"] = {
        "n_sim": n_sim,
        "bootstrap_resamples": n_resamples,
        "dry_run": "training regimes, validation-split human groups",
        "in_sample": "training regimes, all 40-round human training groups",
        "interpolating_timing_games": smoke.interpolating_games,
    }
    files.extend(
        [
            ("sample_table", LIKELIHOOD_FILE, _table(run_id, LIKELIHOOD_COLUMNS, lik_rows)),
            (
                "sample_table",
                "criterion_dry_run.json",
                _table(run_id, evaluate.CRITERION_COLUMNS, evaluate.criterion_rows(smoke.dry_run)),
            ),
            (
                "sample_table",
                "outcomes_dry_run.json",
                _table(run_id, evaluate.OUTCOME_COLUMNS, evaluate.outcome_rows(smoke.dry_run)),
            ),
            (
                "sample_table",
                "criterion_in_sample.json",
                _table(
                    run_id, evaluate.CRITERION_COLUMNS, evaluate.criterion_rows(smoke.in_sample)
                ),
            ),
            (
                "sample_table",
                "outcomes_in_sample.json",
                _table(run_id, evaluate.OUTCOME_COLUMNS, evaluate.outcome_rows(smoke.in_sample)),
            ),
        ]
    )
    written = write_run_directory(out_dir, manifest, files)
    print("Dry run (training regimes, validation split):")
    print(evaluate.summary(smoke.dry_run))
    print("In-sample check (all 40-round training groups):")
    print(evaluate.summary(smoke.in_sample))
    print(json.dumps(smoke.projection, indent=2))
    return written


# --- command line ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.fit")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=data.DEFAULT_PROCESSED_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, default in (("fit", DEFAULT_FIT_RUN_ID), ("smoke", DEFAULT_SMOKE_RUN_ID)):
        p = sub.add_parser(name)
        p.add_argument("--run-id", default=default)
        p.add_argument("--b1-max-updates", type=int, default=config.B1_MAX_UPDATES)
        p.add_argument(
            "--starts", type=int, default=None, help="use only the first N seeds of each MLE fit"
        )
        p.add_argument("--b1-members", type=int, default=None, help="first N B1 seeds only")
        if name == "smoke":
            p.add_argument("--sim-games", type=int, default=config.SIM_GAMES)
            p.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    run_id: str = args.run_id
    validate_run_id(run_id)
    out_dir: Path = args.runs_dir / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    starts: int | None = args.starts
    members: int | None = args.b1_members
    settings = FitSettings(
        b1_seeds=config.B1_SEEDS[:members] if members else config.B1_SEEDS,
        b1_max_updates=int(args.b1_max_updates),
        b2_seeds=config.B2_SEEDS[:starts] if starts else config.B2_SEEDS,
        b3_seeds=config.B3_SEEDS[:starts] if starts else config.B3_SEEDS,
        b23_seeds=config.B23_SEEDS[:starts] if starts else config.B23_SEEDS,
    )
    training = data.load_training(args.processed_dir)
    sidecar = data.processed_sidecar(args.processed_dir, data.TRAINING_FILE)
    fitted = fit_all(training, settings)
    if args.command == "fit":
        manifest = write_fit_run(out_dir, fitted, training, sidecar)
    else:
        manifest = write_smoke_run(
            out_dir,
            fitted,
            training,
            sidecar,
            n_sim=int(args.sim_games),
            n_resamples=int(args.bootstrap),
        )
    print(f"{out_dir}: status {manifest['status']}, config_sha256 {config.config_sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
