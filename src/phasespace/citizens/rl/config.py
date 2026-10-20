"""C3 configuration: every constant of the planner pipeline, in one place (pure).

Pre-protocol (C3 pilot). The PPO hyperparameters were chosen on the B1 training
environment only, from the deterministic learning curve on 1,024 games of a separate
key (777), never on the evaluation games or another model; C4's protocol freezes them.
Two settings were tried, seed 1: (a) 256 games per iteration, Adam 3e-4, 4 minibatches:
740 coins after 1M steps, still rising slowly; (b) 1,024 games, Adam 1e-3, 8 minibatches
(kept): 1,306 coins after 2.5M steps and a plateau near 1,530 from about 10M steps
(Interpolating k = 22 scored 1,270 on the same games). Everything else is the common
PPO default and was not tuned. `config_sha256` (SHA-256 of the canonical
JSON of `config_dict()`) is recorded in every C3 manifest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

PILOT_VERSION: Final = "c3-pilot-v1"

# --- games and random numbers ------------------------------------------------------------
ROUNDS: Final = 40
EVAL_GAMES: Final = 1000
"""Evaluation games per (mechanism, citizen model); the same citizen draws for every
mechanism (common random numbers)."""
EVAL_KEY_SEED: Final = 31_000
TUNE_GAMES: Final = 1000
"""Games per grid point of the k* re-tuning (separate draws from the evaluation games, so
the selected k* is evaluated on fresh games)."""
TUNE_KEY_SEED: Final = 31_001
MONITOR_GAMES: Final = 512
"""Games of the learning curve's deterministic evaluation (training environment)."""
MONITOR_KEY_SEED: Final = 31_002
PILOT_TRAIN_SEEDS: Final = (1, 2, 3)

# --- baselines ---------------------------------------------------------------------------
FIXED_WEIGHTS: Final = {"Equal": 1.0, "Mixed": 0.5, "Proportional": 0.0}
K_PUBLISHED: Final = 22.0
LN_K_MIN: Final = -5.0
LN_K_MAX: Final = 5.0
LN_K_STEP: Final = 0.1
K_TIE_RULE: Final = "smallest ln k among grid points with the maximal mean surplus"

# --- PPO (chosen on B1; recorded per run) --------------------------------------------------
PPO_N_GAMES: Final = 1024
PPO_N_ROUNDS: Final = ROUNDS
PPO_EPOCHS: Final = 4
PPO_MINIBATCHES: Final = 8
PPO_LEARNING_RATE: Final = 1e-3
PPO_CLIP: Final = 0.2
PPO_GAMMA: Final = 1.0
PPO_GAE_LAMBDA: Final = 0.95
PPO_VALUE_COEF: Final = 0.5
PPO_ENTROPY_COEF: Final = 0.0
PPO_MAX_GRAD_NORM: Final = 0.5
PPO_REWARD_SCALE: Final = 1.0 / 200.0
PPO_HIDDEN: Final = 32
PPO_INIT_LOG_STD: Final = -0.5
PILOT_ITERATIONS: Final = 500
"""PPO iterations of a pilot run (500 x 1,024 x 40 = 20.5M planner decisions)."""
PPO_CHUNK: Final = 10
"""Iterations per jitted chunk (one learning-curve point per chunk)."""

# --- G3 ------------------------------------------------------------------------------------
G3_MODEL: Final = "B1"
G3_HOURS_PER_RUN: Final = 24.0
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 9001
CI_LEVEL: Final = 0.95

# --- port checks recorded in the baselines run --------------------------------------------
CHECK_PMF_ATOL: Final = 1e-6
CHECK_SIM_GAMES: Final = 1000
CHECK_SIM_SEED: Final = 31_100
CHECK_Z_MAX: Final = 4.5
CHECK_MECHANISMS: Final = ("Equal", "Proportional", "Interpolating")


def ln_k_grid() -> list[float]:
    n = round((LN_K_MAX - LN_K_MIN) / LN_K_STEP)
    return [round(LN_K_MIN + i * LN_K_STEP, 10) for i in range(n + 1)]


def config_dict() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_"):
            out[name.lower()] = json.loads(json.dumps(value))
    return out


def config_sha256() -> str:
    payload = json.dumps(config_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
