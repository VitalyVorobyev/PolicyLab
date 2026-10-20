"""C4 constants (pure): pilots, cross-play evaluation, diagnostics, distillation, and the
protocol constants of Appendix B of `docs/studies/model-citizens/protocol.md` (seeds,
iterations, delta, families, thresholds, B4 rows and ceilings).

The protocol text is authoritative; `tests/citizens/test_citizens_c4_protocol.py` renders
Appendices A and B from this module and the C3 configuration and compares them line by
line, so neither side changes alone. `config_sha256` (SHA-256 of the canonical JSON of
`config_dict()`) is recorded in every C4 manifest, next to the C3 configuration hash.

**Random-number separation.** Every key here differs from the C3 keys (evaluation 31,000,
tuning 31,001, monitoring 31,002, sampled noise 31,003, port checks 31,100+) and from
every PPO training seed (small integers; the pilots' seeds 901-903 are reserved and never
reused in C4). `tests/citizens/test_citizens_c4.py` checks the separation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

C4_VERSION: Final = "c4-v1"

# --- PPO feasibility pilots (review item 2) -----------------------------------------------
PILOT_SEEDS: Final = {"B2": 901, "B3": 902, "B23": 903}
"""One pilot training run per non-B1 model; these seeds are never reused in C4."""
PILOT_ITERATIONS: Final = 1000
"""Twice the C3 pilot length, to see where the learning curve flattens on each model."""
PILOT_SAMPLED_NOISE_KEY_SEED: Final = 41_010
"""Planner-noise key of a pilot's sampled-policy score on the monitoring games."""
PLATEAU_SMOOTH_POINTS: Final = 5
"""Trailing moving average (curve points of 10 iterations) of the monitoring return."""
PLATEAU_FINAL_POINTS: Final = 10
"""The final level F is the mean monitoring return of the last points."""
PLATEAU_TOL_FRACTION: Final = 0.02
"""Plateau: first iteration from which the smoothed curve stays >= F - tol (F - I) to the
end, I the first point."""

# --- cross-play evaluation (C4b; nothing is evaluated before the protocol freeze) ----------
EVAL_KEY_SEED: Final = 41_000
"""Citizen key of the C4 evaluation games (common random numbers for every policy)."""
EVAL_GAMES: Final = 1000
SAMPLED_NOISE_KEY_SEED: Final = 41_001
"""Planner-noise key of the sampled-policy evaluation."""
FIXED_RULES: Final = ("Equal", "Mixed", "Proportional")
K_PUBLISHED_RULE: Final = "Interpolating k=22"
K_STAR_RULE: Final = "Interpolating k*"
"""Each training model's re-tuned k* is a row of its own ("Interpolating k*[B2]")."""
LAST_ROUNDS: Final = (38, 39)
"""0-based rounds 39 and 40: the endgame kept share (review item 4)."""
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 9101
CI_LEVEL: Final = 0.95

# --- extrapolation diagnostics (review item 1) -------------------------------------------
OOD_MODELS: Final = ("B2", "B3", "B23")
OOD_FEATURES: Final = {
    "B2": ("x1", "x2", "x3", "x4"),
    "B3": ("own_offer", "own_prev_offer"),
    "B23": ("own_offer", "own_prev_offer", "x3"),
}
"""Continuous inputs each model reads (B2 features x1..x4 of `b2_condcoop`; EWA reads the
own offer and the own previous offer, B23 also x3). Range = [min, max] over the scored
fit-split decisions of the C2 fit (training data only)."""
OOD_GRID_BINS: Final = 20
"""Joint (x1 / 4, x4) occupancy grid: a decision is off-support if its cell holds no
scored fit-split decision."""

# --- no-retention ablation (review item 1) -----------------------------------------------
ABLATION_FLAG: Final = "no_retention"
"""PPOConfig.retention = False: 4-way softmax over the players, sum e = R - 0.001."""

# --- distillation (C4c) -------------------------------------------------------------------
DISTILL_KEY_SEED: Final = 41_002
"""Citizen key of the games whose planner trajectories are distilled."""
DISTILL_GAMES: Final = 1000
DISTILL_W_BINS: Final = 10
"""Equal-width bins of R / 200 for the binned w(R) and kept share."""
DISTILL_TREE_DEPTH: Final = 3
DISTILL_TREE_MIN_LEAF: Final = 200
DISTILL_FEATURES: Final = (
    "pool_frac",
    "own_prev_fraction",
    "others_prev_fraction",
    "own_prev_share",
)

# --- planner versus B4 (evaluation only) ---------------------------------------------------
B4_PLANNER_GAMES: Final = 40
B4_CALLS_PER_GAME_CALIBRATION: Final = 106.11666666666666
"""Measured in citizens-c2-b4-calibration-v2 (Proportional, 60 games, retries included)."""
B4_RETRY_FRACTION_CALIBRATION: Final = 0.020025632809996797
C4_PROTOCOL_PATH: Final = "docs/studies/model-citizens/protocol.md"

# --- protocol constants (Appendix B of the C4 protocol; P8) --------------------------------
# `tests/citizens/test_citizens_c4_protocol.py` renders these and compares them with the
# protocol text line by line.
TRAIN_ITERATIONS: Final = 1000
"""PPO iterations of every C4 planner row (the training signature)."""
MAIN_SEEDS: Final = (1001, 1002, 1003, 1004, 1005)
"""Training seeds of the main rows Planner[B1], Planner[B2], Planner[B3], Planner[B23]."""
NO_RETENTION_SEEDS: Final = (2001, 2002, 2003)
"""Training seeds of the no-retention ablation rows Planner-noret[X] (S1)."""
MIXTURE_MODEL: Final = "Mix"
"""Training-model name of the robust mixture planner (S2; owner decision O1 = yes)."""
MIXTURE_SEEDS: Final = (3001, 3002, 3003)
"""Training seeds of Planner[Mix]: each training game draws its citizen model uniformly
from B1, B2, B3, B23 (keys derived from the training seed)."""
SECONDARY_REDUCED_SEEDS: Final = 2
"""Pre-declared non-LLM reduction (protocol §7): a secondary row may run its first 2
declared seeds instead of 3; main rows are never reduced."""
DELTA: Final = 16.0
"""Minimum effect: coins of total surplus per game (0.1 coin per player-round)."""
ALPHA: Final = 0.05
"""Family-wise error rate; Holm within each family, separately for each test."""
FAMILY_SIZES: Final = {
    "H1-reversal": 12,
    "H1-transfer": 12,
    "H2a-loss": 3,
    "H2a-noninferior": 3,
    "H2b-loss": 3,
    "own-model-limited": 4,
    "own-model-gain": 4,
    "H2-B4": 3,
    "S1-reversal": 12,
    "S1-transfer": 12,
    "S2-reversal": 4,
    "S2-transfer": 4,
}
"""Cells per Holm family (H2-B4: per family and per validated B4 variant)."""
CONVERGENCE_MAX_PLATEAU_ITERATION: Final = 900
CONVERGENCE_MAX_TAIL_SLOPE: Final = 16.0
"""Coins per 100 iterations (= delta)."""
EXPLOIT_TV_RATIO: Final = 1.5
"""Possible exploitation: B1 mean member TV above 1.5 x that of k = 22 in the column."""
EXPLOIT_SUPPORT_MODELS: Final = ("B2",)
EXPLOIT_SUPPORT_MARGIN: Final = 0.05
"""(B2 only) range or grid fraction above k = 22's by more than 0.05 (absolute)."""
AGREEMENT_MIN_ROWS: Final = 200
"""w(R) bins compared need >= 200 informative rows for every seed of both models."""
AGREEMENT_L1_FLOOR: Final = 0.10
AGREEMENT_WITHIN_FACTOR: Final = 2.0
"""Same w(R) if L1 <= max(0.10, 2 x the larger within-model L1)."""
FIDELITY_MIN_R2: Final = 0.5
"""A planner whose distilled w-rule R^2 is below 0.5 is low fidelity."""
RETENTION_L1_RATIO: Final = 0.5
"""S1: attributable to retention if the no-retention between-model w(R) L1 is below half
the main planners' (or by the H1-reversal rule)."""
B4_VARIANTS: Final = ("plain", "calibrated")
"""Owner decision O2 = (ii): every B4 row on both variants."""
B4_PLANNER_SEED: Final = 1001
B4_COMPARATOR_MODEL: Final = "B1"
B4_COMPARATOR_K: Final = 9.025013499434122
"""k*[B1] = exp(2.2) from citizens-c3-baselines-v1 (the `c4 b4` rule row checks it)."""
B4_GAMES_ALLOWED: Final = (40, 30)
"""40 games per (row, variant); 30 is the first pre-declared LLM reduction."""
B4_MIN_COMPLETE_GAMES: Final = 20
"""Below 20 complete games a B4 row is exploratory."""
B4_GATING_REGIMES: Final = ("R-I", "R-M2")
"""A B4 variant is a validated evaluation population if C2 counts it validated (closed
loop) and gating on R-I or R-M2 (§9.4)."""
CEILING_NON_LLM_HOURS: Final = 40.0
CEILING_LLM_CALLS: Final = 66_000
"""C4 LLM calls, both B4 variants (O2 = (ii)); counted apart from C2's 55,000."""


def config_dict() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and not name.startswith("_"):
            out[name.lower()] = json.loads(json.dumps(value))
    return out


def config_sha256() -> str:
    payload = json.dumps(config_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
