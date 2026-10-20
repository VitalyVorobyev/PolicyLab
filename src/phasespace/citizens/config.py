"""C2 configuration: every constant of the validation protocol, in one place (pure).

Source: `docs/studies/model-citizens/protocol-validation.md` (v1, Appendix A and the
sections cited below). `tests/citizens/test_citizens_config.py` parses Appendix A and
checks that it agrees with this module (freeze checklist item 2); `config_sha256` is the
SHA-256 of the canonical JSON of `config_dict()` and is recorded in every C2 manifest.

Constants marked *implementation choice* are not fixed by the protocol text; they are
the closest faithful choice and are recorded as pre-freeze amendments (protocol §14).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from phasespace.citizens import game

PROTOCOL_VERSION: Final = "v1"

# --- split (protocol 6.1) ---------------------------------------------------------------
SPLIT_SEED: Final = 20261020
BC_SPLIT_SEED: Final = 20261021
FIT_UNITS_PER_STRATUM: Final = 32
VAL_UNITS_PER_STRATUM: Final = 8
SPLIT_STRATA: Final = (
    ("Equal Baseline", "Exp 1"),
    ("Mixed Baseline", "Exp 1"),
    ("Proportional Baseline", "Exp 1"),
    ("Proportional Baseline", "Exp 2"),
    ("Proportional Baseline", "Exp 4"),
)
"""Strata in the order of protocol 6.1; one generator draws one permutation per stratum
in this order (*implementation choice*: the text does not fix the order of the draws)."""
BC_FIT_GAMES: Final = 410
BC_VAL_GAMES: Final = 102

# --- likelihood (protocol 4, 6.2, 6.3) --------------------------------------------------
N_FLOOR_BINS: Final = 10
N_LEVELS: Final = 11
B0_EPS: Final = 0.05
M0_SMOOTHING: Final = 0.5
EWA_ROUND1_SMOOTHING: Final = 0.5
"""p0 of B3/B23: round-1 level counts of the fit split + 0.5 (protocol 6.3)."""

# --- B1 GRU clone (protocol 6.3, Koster et al. 2025 Table S1) ---------------------------
B1_INPUTS: Final = 9
B1_ENCODER: Final = (16, 32)
B1_GRU: Final = 64
B1_PROJECTION: Final = (32, 16)
B1_ACTIVATION: Final = "tanh"
B1_BINS: Final = N_FLOOR_BINS
B1_LEARNING_RATE: Final = 5e-4
B1_BATCH: Final = 64
B1_EVAL_EVERY: Final = 250
B1_PATIENCE: Final = 8
B1_MAX_UPDATES: Final = 20_000
B1_SEEDS: Final = (101, 102, 103, 104)
B1_INPUT_SCALE: Final = 200.0
B1_EARLY_STOPPING_SCORE: Final = "integer_nll"
"""*Implementation choice*: the 'validation NLL' of early stopping is the per-decision
integer-pmf NLL of protocol 4 (the score Delta-ell and K1 use), not the bin cross-entropy."""
B1_BATCHING: Final = "epoch_permutation"
"""*Implementation choice*: each epoch is a permutation of the fit-split sequences drawn
with numpy default_rng(seed); batches are consecutive slices of 64 (the last partial
batch of an epoch is dropped)."""

# --- B2 conditional cooperation ----------------------------------------------------------
B2_PARAMS: Final = 12
B2_STARTS: Final = 10
B2_SEEDS: Final = tuple(range(201, 211))
B2_START_SD: Final = 0.5
B2_MAX_X1: Final = 4.0

# --- B3 EWA and B23 hybrid ---------------------------------------------------------------
B3_PARAMS: Final = 6
B3_STARTS: Final = 10
B3_SEEDS: Final = tuple(range(301, 311))
B3_BOUNDS: Final = {
    "phi": (0.0, 1.0),
    "rho": (0.0, 1.0),
    "delta": (0.0, 1.0),
    "lambda": (0.01, 50.0),
    "n0": (0.0, 20.0),
    "theta": (0.0, 5.0),
}
B23_PARAMS: Final = 7
B23_SEEDS: Final = tuple(range(311, 321))
B23_KAPPA_BOUNDS: Final = (0.0, 50.0)
EWA_PAYOFF_SCALE: Final = 50.0

# --- optimiser (implementation choice: the protocol names L-BFGS-B only) -----------------
LBFGSB_MAXITER: Final = 2000
LBFGSB_MAXFUN: Final = 100_000
LBFGSB_FTOL: Final = 1e-12
LBFGSB_GTOL: Final = 1e-7
"""scipy.optimize.minimize(method="L-BFGS-B") options; gradients by 2-point finite
differences (scipy default); objective = mean NLL per non-forced decision."""

# --- surrogates (protocol 6.4; built in phasespace.citizens.surrogate) -------------------
SURROGATE_FC: Final = 32
SURROGATE_GRU: Final = 32
SURROGATE_LEARNING_RATE: Final = 1e-3
SURROGATE_BATCH: Final = 64
SURROGATE_EVAL_EVERY: Final = 250
SURROGATE_PATIENCE: Final = 8
SURROGATE_MAX_UPDATES: Final = 20_000
SURROGATE_SEEDS_M1: Final = (401, 402, 403)
SURROGATE_SEEDS_M2: Final = (411, 412, 413)
SURROGATE_STEP2_WIDTH: Final = 64
SURROGATE_STEP2_SEEDS_M1: Final = (404, 405, 406)
SURROGATE_STEP2_SEEDS_M2: Final = (414, 415, 416)
SURROGATE_ACCEPT_MEDIAN_ABS: Final = 1.0
SURROGATE_ACCEPT_P90_ABS: Final = 5.0
SURROGATE_ACCEPT_EXCLUSION: Final = 0.90

# --- simulation (protocol 7) -------------------------------------------------------------
SIM_GAMES: Final = 1000
SIM_ROUNDS: Final = game.N_ROUNDS
M2_EXP4_SETS_PER_HUMAN_SET: Final = 25
SIM_SEED_BASE: Final = 50_000
SIM_MODEL_CODES: Final = {"B0": 0, "M0-sim": 1, "B1": 2, "B2": 3, "B3": 4, "B23": 5}
SIM_REGIME_CODES: Final = {
    "Interpolating": 1,
    "M1": 2,
    "M2": 3,
    "M2 Exp 4": 4,
    "Equal": 5,
    "Mixed": 6,
    "Proportional": 7,
}
SIM_FEASIBILITY_ATOL: Final = 1e-9
"""Protocol 5: sum(e) <= R + 1e-9."""

# --- evaluation (protocol 8) -------------------------------------------------------------
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 9001
CI_LEVEL: Final = 0.95
TOLERANCE: Final = {
    "surplus": 0.1,
    "gini": 0.005,
    "depletion_round": 0.5,
    "players_active": 0.1,
}
METRICS: Final = ("surplus", "gini", "depletion_round", "players_active")
OUTCOME_METRICS_REQUIRED: Final = 3
W1_SPLITS: Final = 1000
PMF_FLOOR: Final = 1e-12
"""Amendment A29: a scored decision with P(c) < 1e-12 (including P = 0) is scored at
log(1e-12) and counted, instead of stopping the evaluation."""

# --- B4 (protocol 6.3; built in phasespace.citizens.llm) ---------------------------------
B4_TYPES: Final = 3
B4_CALIBRATION_GAMES: Final = 60
B4_CALIBRATION_GAMES_PER_COMPOSITION: Final = 4
B4_COMPOSITIONS: Final = 15
B4_GRID_STEP: Final = 0.1
B4_EVAL_GAMES: Final = 40
B4_RETRIES: Final = 2
B4_TIMEOUT_S: Final = 120
B4_DISCARD_LIMIT: Final = 0.10
B4_CALL_CEILING: Final = 55_000
B4_SEED: Final = 7001
B4_MODEL: Final = "claude-haiku-5-5"
"""Protocol 13 O6: the owner's choice, passed as `claude -p --model`."""
B4_EFFORT: Final = "medium"
"""`claude -p --effort`, pinned explicitly. It is the default effort of claude-haiku-5-5
in Claude Code 2.1.304 (built-in model catalog and the served catalog cache both give
default_effort "medium"), so the pre-freeze pilot (run without the flag) stays
representative; pinning protects the protocol against a changed CLI default."""

B4_CLI_VERSION: Final = "2.1.304"
"""Claude Code CLI version (`claude --version`) every B4 run records; the held-out B4
stage refuses another version (amendment A26)."""
B4_EVAL_GAMES_REDUCED: Final = (30, 20)
"""Protocol 7 pre-declared reductions: 30 games per (variant, mechanism), then 20 for
M1/M2 only; `heldout b4 --games` accepts only these and B4_EVAL_GAMES (amendment A31)."""
B4_MIN_GATING_GAMES: Final = 20
"""Protocol 7: below 20 complete games a regime's B4 result is exploratory."""
B4_INFRA_RETRIES: Final = 6
B4_INFRA_BACKOFF_S: Final = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
"""Amendment A27: infrastructure errors (timeout, non-zero exit, CLI/API error, rate
limit, missing result, model mismatch) are retried up to 6 times per decision with these
waits; they are not invalid answers and never discard a game."""
B4_ENV_ALLOWLIST: Final = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "TERM")
"""Amendment A28: the only caller variables passed to `claude -p` (names recorded per
run, never values). The CLI authenticates through the macOS keychain with these alone
(checked 2026-10-20)."""

# --- compute (protocol 7) ----------------------------------------------------------------
COMPUTE_CEILING_HOURS: Final = 12.0


def sim_seed(model: str, regime: str) -> int:
    """Simulation seed 50000 + 100 m + r (protocol 7)."""
    return SIM_SEED_BASE + 100 * SIM_MODEL_CODES[model] + SIM_REGIME_CODES[regime]


def config_dict() -> dict[str, Any]:
    """Every constant above as plain JSON values (tuples become lists)."""
    out: dict[str, Any] = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_"):
            out[name.lower()] = json.loads(json.dumps(value))
    return out


def config_sha256() -> str:
    """SHA-256 of the canonical JSON of `config_dict()` (sorted keys, no whitespace)."""
    payload = json.dumps(config_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(config_sha256())
