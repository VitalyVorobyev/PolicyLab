"""C1 validation of the reimplemented game against the recorded data (pure).

Every check compares the *recorded* columns of the published CSV with an independent
rule (the paper's game equations and mechanisms, plus the conventions established in
the C1 audit); nothing is compared with output of the same code path.

Tolerances come from how the CSV stores numbers:

- Human and BC values are float32 numbers written in full (shortest repr or 17
  digits); Exp 4 rows are written with 8 significant digits. Pool values are < 256,
  where a float32 half-ulp is 7.6e-6, and the pool update takes about 10 float32
  operations, so `POOL_TOLERANCE` = 1e-4 bounds the arithmetic and print error with a
  margin (observed maximum 4.3e-5).
- Recorded offers are rounded to 4 decimals after float32 arithmetic; a different
  float32 operation order can move a value across a rounding boundary, so an offer is
  reproduced when it is within one unit of the 4th decimal plus the float32
  representation error of values < 256: `OFFER_TOLERANCE` = 1.5e-4.

Held-out human games reach this module only through
`phasespace.citizens.data.heldout_identity_residuals`, which calls `pool_transition`,
`offer_reproduction` (without diagnostics), `action_anomalies` and
`players_active_identity` only; their results are residual statistics and pass/fail
counts.

**Two pool rules, assigned per game in advance.** Seven early Exp 1 training games
(`LEGACY_REMAINDER_DROPPED`, found in the training data only) were run by a software
version that discarded the undistributed remainder R - sum e; every other game follows
the full rule. Each game is checked against its own rule only; no row may pick
whichever rule fits better.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from phasespace.citizens import metrics
from phasespace.citizens.game import GROWTH_FACTOR, MAX_POOL, POOL_FLOOR, next_pool
from phasespace.citizens.mechanisms import (
    BASELINE_WEIGHTS,
    K_INTERPOLATING,
    RECORDED_WITHHELD,
    deployed_offers,
    interpolating_weight,
    paper_offers,
)
from phasespace.citizens.records import TRAINING_MECHANISMS, Games

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

POOL_TOLERANCE: Final = 1e-4
OFFER_TOLERANCE: Final = 1e-4 + 5e-5
EXACT_OFFER: Final = 2e-5
"""An offer residual this small is an exact reproduction up to float32 storage."""
RECORD_TOLERANCE: Final = 1e-4
"""Reward = offer - action, prev_reciprocations = previous actions (float32 storage)."""
Z_TOLERANCE: Final = 0.005
"""Published z-values are given to two decimals."""

REPRODUCED_MECHANISMS: Final = (*BASELINE_WEIGHTS, "Interpolating Baseline")

LEGACY_REMAINDER_DROPPED: Final = frozenset(
    {"14606420", "18817699", "21063212", "56254954", "70560629", "89143435", "93922021"}
)
"""launch_ids of the human games whose pool update drops the remainder R - sum e.

Frozen from the training data (C1 audit, D33): 7 early Exp 1 games (3 Equal, 4
Proportional) in which the full rule fails in early rounds and the remainder-dropped
rule holds in every row. They apply to human games only (BC games
have their own launch_id space); no held-out human game carries one of these ids
(`pool_transition` details: games_legacy_rule = 0 for every held-out population).
"""
HELDOUT_POOL_ANOMALY_POPULATION: Final = "RL Agent (M1):Exp 1"
"""Deferred G1b item (D33): the only held-out (mechanism:cohort) with failing pool rows."""
HELDOUT_POOL_ANOMALY_MAX_ROWS: Final = 23
HELDOUT_POOL_ANOMALY_MAX_GAMES: Final = 5
HELDOUT_POOL_ANOMALY_MAX_RESIDUAL: Final = 0.0100 + POOL_TOLERANCE
"""Documented bound of the anomaly (rows ~0.01 coins below the rule, plus print tolerance).

A regression bound recorded before diagnosis, not an accepted rule: G1b root-causes the
rows after the validation protocol freeze.
"""
WITHHELD_ALTERNATIVES: Final = (0.0, 0.0005, RECORDED_WITHHELD, 0.0015, 0.002)
"""Withheld amounts compared by `withheld_alternatives` (only 0.001 reproduces the data)."""


@dataclass(frozen=True, slots=True)
class IdentityResult:
    """Residual summary of one identity check over one population of rows."""

    name: str
    population: str
    entries: int  # rows (pool) or player-entries (offers, records) checked
    max_abs_residual: float  # of the accepted rule per entry; NaN if nothing was checked
    threshold: float
    n_fail: int  # entries outside the threshold under every accepted rule
    details: Mapping[str, int | float] = field(default_factory=dict[str, int | float])

    @property
    def passed(self) -> bool:
        return self.entries > 0 and self.n_fail == 0 and math.isfinite(self.max_abs_residual)


def _summary(
    name: str,
    population: str,
    residual: FloatArray,
    threshold: float,
    details: Mapping[str, int | float] | None = None,
) -> IdentityResult:
    finite = np.isfinite(residual)
    return IdentityResult(
        name=name,
        population=population,
        entries=int(residual.size),
        max_abs_residual=float(np.abs(residual).max())
        if residual.size and finite.all()
        else math.nan,
        threshold=threshold,
        n_fail=int(np.count_nonzero(~finite | (np.abs(residual) > threshold))),
        details=dict(details or {}),
    )


# --- pool transition ------------------------------------------------------------------


def _next_pool_record(games: Games) -> tuple[FloatArray, BoolArray]:
    """(G, T) the recorded pool after round t and where it is known.

    Human rows carry `next_environment_state.pool`; BC rows have only the next row's
    pool before allocation, so a BC game's last round has no recorded successor.
    """
    mask = games.round_mask()
    following = np.full_like(games.pool, np.nan)
    following[:, :-1] = games.pool[:, 1:]
    has_following = mask.copy()
    has_following[np.arange(games.n_games), games.n_rounds - 1] = False
    human = games.is_human[:, None]
    record = np.where(human, games.next_pool, following)
    known = np.where(human, mask, has_following)
    return record, known


def full_raw(games: Games) -> FloatArray:
    """R - sum e + 1.4 sum c from the recorded columns (no cap, no floor)."""
    return games.pool - games.offers.sum(axis=-1) + GROWTH_FACTOR * games.actions.sum(axis=-1)


def pool_rules(games: Games) -> tuple[FloatArray, FloatArray]:
    """(full, remainder_dropped) predictions of the next pool from recorded R, e, c.

    full: max(0.01, min(200, R - sum e + 1.4 sum c)), the paper's update with the
    observed 0.01 floor. remainder_dropped: the same with the undistributed remainder
    R - sum e discarded, max(0.01, min(200, 1.4 sum c)), which an early version of the
    human game software applied (the `LEGACY_REMAINDER_DROPPED` games). The two agree
    at the 0.01 floor and the 200 cap.
    """
    full = next_pool(games.pool, games.offers, games.actions)
    nothing = np.zeros_like(games.offers)
    dropped = next_pool(np.zeros_like(games.pool), nothing, games.actions)
    return full, dropped


def legacy_games(games: Games) -> BoolArray:
    """(G,) human games listed in `LEGACY_REMAINDER_DROPPED`."""
    return games.is_human & np.isin(games.launch_id, sorted(LEGACY_REMAINDER_DROPPED))


def pool_transition(games: Games, population: str) -> IdentityResult:
    """Recorded next pool vs each game's pre-assigned rule, over every recorded successor.

    Games in `LEGACY_REMAINDER_DROPPED` are checked against the remainder-dropped rule
    in every row, all other games against the full rule; nothing is chosen per row.
    Counts: rows and games under each rule, legacy rows where the full rule fails (so
    the legacy rule is needed) and the legacy games with at least one such row, and the
    games with at least one failing row.
    """
    record, known = _next_pool_record(games)
    full, dropped = pool_rules(games)
    legacy = legacy_games(games)[:, None] & known
    prediction = np.where(legacy, dropped, full)
    residual_grid = np.where(known, record - prediction, 0.0)
    failing = known & (np.abs(residual_grid) > POOL_TOLERANCE)
    legacy_needed = legacy & (np.abs(record - full) > POOL_TOLERANCE)
    details: dict[str, int | float] = {
        "rows_full_rule": int(np.count_nonzero(known & ~legacy)),
        "rows_legacy_rule": int(np.count_nonzero(legacy)),
        "games_legacy_rule": int(np.count_nonzero(legacy.any(axis=1))),
        "legacy_rows_full_rule_fails": int(np.count_nonzero(legacy_needed)),
        "legacy_games_full_rule_fails": int(np.count_nonzero(legacy_needed.any(axis=1))),
        "games_fail": int(np.count_nonzero(failing.any(axis=1))),
    }
    return _summary("pool_transition", population, residual_grid[known], POOL_TOLERANCE, details)


def pool_record_consistency(games: Games, population: str) -> IdentityResult:
    """Human rows: next_environment_state.pool of round t equals round t+1's pool."""
    mask = games.round_mask()
    inner = mask.copy()
    inner[np.arange(games.n_games), games.n_rounds - 1] = False
    inner &= games.is_human[:, None]
    residual = (games.next_pool[:, :-1] - games.pool[:, 1:])[inner[:, :-1]]
    return _summary("pool_record_consistency", population, residual, 0.0)


def pool_regimes(games: Games) -> dict[str, int]:
    """Training-only diagnostics: rows where the 0.01 floor or the 200 cap binds."""
    _, known = _next_pool_record(games)
    raw = full_raw(games)
    return {
        "rows_floor_binding": int(np.count_nonzero(known & (raw < POOL_FLOOR))),
        "rows_cap_binding": int(np.count_nonzero(known & (raw > MAX_POOL))),
    }


# --- mechanism offers -----------------------------------------------------------------


def _weights(games: Games, mechanism: str, *, quantize: bool = True) -> FloatArray:
    if mechanism == "Interpolating Baseline":
        return interpolating_weight(games.pool, K_INTERPOLATING, quantize=quantize)
    return np.full_like(games.pool, BASELINE_WEIGHTS[mechanism])


def _prev(games: Games) -> FloatArray:
    """Recorded previous reciprocations with round 0 set to zeros (equal split)."""
    prev = games.prev_reciprocations.copy()
    prev[:, 0, :] = 0.0
    return prev


def offer_reproduction(
    games: Games, population: str, *, diagnostics: bool = True
) -> list[IdentityResult]:
    """Recorded offers vs `deployed_offers`, one result per reproducible mechanism.

    Uses each row's recorded pool and `mechanism_observation.prev_reciprocations`;
    round 0 is the equal split. Counts: exact entries (<= 2e-5) and, only with
    `diagnostics` (training games), rows whose previous reciprocations sum to zero and
    the largest residual of the paper's idealised rule (whole pool, unquantised weight).
    Both depend on behaviour, so they are never computed for held-out games.
    """
    results: list[IdentityResult] = []
    for mechanism in REPRODUCED_MECHANISMS:
        subset = games.select(games.mechanism == mechanism)
        if subset.n_games == 0:
            continue
        mask = subset.round_mask()
        prev = _prev(subset)
        model = deployed_offers(subset.pool, prev, _weights(subset, mechanism))
        residual = (subset.offers - model)[mask]
        details: dict[str, int | float] = {
            "exact_entries": int(np.count_nonzero(np.abs(residual) <= EXACT_OFFER)),
        }
        if diagnostics:
            paper = paper_offers(subset.pool, prev, _weights(subset, mechanism, quantize=False))
            zero_sum = mask & (prev.sum(axis=-1) == 0.0)
            zero_sum[:, 0] = False
            details["rows_zero_previous_sum"] = int(np.count_nonzero(zero_sum))
            details["max_abs_paper_rule_residual"] = float(
                np.abs(subset.offers - paper)[mask].max()
            )
        slug = mechanism.split()[0].lower()
        results.append(
            _summary(f"offers_{slug}", population, residual.ravel(), OFFER_TOLERANCE, details)
        )
    return results


def interpolating_alternatives(games: Games) -> dict[str, int]:
    """Training (BC) diagnostic: offer mismatches (> OFFER_TOLERANCE) under other weight laws."""
    subset = games.select(games.mechanism == "Interpolating Baseline")
    mask = subset.round_mask()
    prev = _prev(subset)
    x = subset.pool / MAX_POOL
    laws = {
        "k22_round1": np.round(x**22.0, 1),
        "k22_unquantized": x**22.0,
        "k22_floor1": np.floor(x**22.0 * 10.0) / 10.0,
        "k21_round1": np.round(x**21.0, 1),
        "k23_round1": np.round(x**23.0, 1),
        "kexp3.1_round1": np.round(x ** math.exp(3.1), 1),
    }
    out: dict[str, int] = {"entries": int(np.count_nonzero(mask) * 4)}
    for name, w in laws.items():
        model = deployed_offers(subset.pool, prev, w)
        out[name] = int(np.count_nonzero(np.abs(subset.offers - model)[mask] > OFFER_TOLERANCE))
    return out


def withheld_alternatives(games: Games) -> dict[str, dict[str, int]]:
    """Training diagnostic: offer reproduction under other withheld amounts.

    For each amount in `WITHHELD_ALTERNATIVES`, over every row of the training games of
    the reproducible mechanisms: entries, exact entries (<= EXACT_OFFER) and entries
    outside OFFER_TOLERANCE. Pass training games only (the result counts behaviour-
    dependent rows of every mechanism).
    """
    out: dict[str, dict[str, int]] = {}
    for withheld in WITHHELD_ALTERNATIVES:
        entries = exact = outside = 0
        for mechanism in REPRODUCED_MECHANISMS:
            subset = games.select(games.mechanism == mechanism)
            if subset.n_games == 0:
                continue
            mask = subset.round_mask()
            model = deployed_offers(
                subset.pool, _prev(subset), _weights(subset, mechanism), withheld=withheld
            )
            residual = np.abs(subset.offers - model)[mask]
            entries += int(residual.size)
            exact += int(np.count_nonzero(residual <= EXACT_OFFER))
            outside += int(np.count_nonzero(residual > OFFER_TOLERANCE))
        out[f"{withheld:g}"] = {"entries": entries, "exact": exact, "outside_tolerance": outside}
    return out


# --- record consistency (training only) ------------------------------------------------


def record_consistency(games: Games, population: str) -> list[IdentityResult]:
    """Internal identities of the recorded rows (training games only).

    - reward_identity: player_reward = offer - action;
    - previous_reciprocations: prev_reciprocations(t) = actions(t-1);
    - mechanism_reward: mechanism_reward = sum of player rewards;
    - offer_validity: offers >= 0 and sum <= pool (violation size, 0 if valid);
    - action_validity: 0 <= c <= e; for humans c integral and c <= floor(e) (so an
      offer below 1 forces c = 0).
    """
    mask = games.round_mask()
    e, c, r = games.offers, games.actions, games.rewards
    entries = mask[..., None].repeat(4, axis=-1)
    reward_res = (r - (e - c))[entries]
    inner = mask.copy()
    inner[:, 0] = False
    prev_res = (games.prev_reciprocations[:, 1:] - c[:, :-1])[inner[:, 1:][..., None].repeat(4, -1)]
    mech_res = (games.mechanism_reward - r.sum(axis=-1))[mask]
    over = np.maximum(0.0, e.sum(axis=-1) - games.pool)[mask]
    negative = np.maximum(0.0, -e)[entries]
    offer_violation = np.concatenate([over, negative])
    human = games.is_human[:, None, None] & entries
    bc = ~games.is_human[:, None, None] & entries
    bound = np.where(human, np.floor(e), e)
    action_violation = np.maximum(np.maximum(0.0, -c), np.maximum(0.0, c - bound))
    non_integral = human & (c != np.floor(c))
    action_violation = np.where(non_integral, np.inf, action_violation)[entries]
    bc_sub_one = bc & (e < 1.0) & (c > 0.0)
    return [
        _summary("reward_identity", population, reward_res, RECORD_TOLERANCE),
        _summary("previous_reciprocations", population, prev_res, RECORD_TOLERANCE),
        _summary("mechanism_reward", population, mech_res, RECORD_TOLERANCE),
        _summary("offer_validity", population, offer_violation, RECORD_TOLERANCE),
        _summary(
            "action_validity",
            population,
            action_violation,
            0.0,
            {"bc_positive_actions_on_offers_below_1": int(np.count_nonzero(bc_sub_one))},
        ),
    ]


ACTION_ANOMALIES: Final = "action_anomalies"
PLAYERS_ACTIVE_IDENTITY: Final = "players_active_identity"
COUNT_ONLY_CHECKS: Final = (ACTION_ANOMALIES, PLAYERS_ACTIVE_IDENTITY)
"""C2 declared-handling counts (protocol-validation A29, A30): reported, never C1 gates."""


def action_anomalies(games: Games, population: str) -> IdentityResult:
    """Counts of recorded player-entries a C2 likelihood cannot score as recorded.

    Categories (an entry can fall in more than one; `n_fail` counts the union):
    non-integer c; c > floor(e) with e >= 1; c > 0 with e < 1 (an excluded player who
    reciprocated); c < 0. The residual of an entry is its indicator (0 or 1), so the
    result carries counts only, never a behavioural value (amendment A29).
    """
    entries = games.round_mask()[..., None].repeat(4, axis=-1)
    e, c = games.offers, games.actions
    non_integer = entries & (c != np.floor(c))
    above = entries & (e >= 1.0) & (c > np.floor(e))
    excluded_positive = entries & (e < 1.0) & (c > 0.0)
    negative = entries & (c < 0.0)
    anomalous = non_integer | above | excluded_positive | negative
    return _summary(
        ACTION_ANOMALIES,
        population,
        anomalous[entries].astype(np.float64),
        0.0,
        {
            "non_integer_c": int(np.count_nonzero(non_integer)),
            "c_above_floor_e": int(np.count_nonzero(above)),
            "positive_c_on_offer_below_1": int(np.count_nonzero(excluded_positive)),
            "negative_c": int(np.count_nonzero(negative)),
            "games_with_anomalies": int(np.count_nonzero(anomalous.any(axis=(1, 2)))),
        },
    )


def players_active_identity(games: Games, population: str) -> IdentityResult:
    """Rows whose `players_kept_active` differs from #{i : e_i >= 1} (indicator residuals).

    Counts only (amendment A30): the residual of a row is 1 if the two differ, else 0.
    """
    mask = games.round_mask()
    offered = metrics.players_offered_at_least_one(games.offers)
    mismatch = mask & (games.players_kept_active != offered)
    return _summary(
        PLAYERS_ACTIVE_IDENTITY,
        population,
        mismatch[mask].astype(np.float64),
        0.0,
        {"games_with_mismatch": int(np.count_nonzero(mismatch.any(axis=1)))},
    )


def depletion_absorbing(games: Games) -> dict[str, int]:
    """Training diagnostic: games whose pool rises back to >= 1 after falling below 1."""
    mask = games.round_mask()
    below = mask & (games.pool < 1.0)
    length = int(games.pool.shape[1])
    first: npt.NDArray[np.int64] = np.where(below.any(axis=1), below.argmax(axis=1), length)
    rounds = np.arange(length, dtype=np.int64)
    later: BoolArray = mask & (rounds[None, :] > first[:, None])
    recovered: BoolArray = (later & (games.pool >= 1.0)).any(axis=1)
    human = games.is_human
    return {
        "human_games_depleted": int(np.count_nonzero(below.any(axis=1) & human)),
        "human_games_recovered": int(np.count_nonzero(recovered & human)),
        "bc_games_depleted": int(np.count_nonzero(below.any(axis=1) & ~human)),
        "bc_games_recovered": int(np.count_nonzero(recovered & ~human)),
    }


# --- published statistics -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ZComparison:
    """A published Wilcoxon rank-sum z-value and its reproduction."""

    name: str
    endpoint: str  # "surplus" | "gini"
    cohort: str
    higher: str  # mechanism the paper reports as higher
    lower: str
    published: float  # |z| as printed in the paper
    reproduced: float  # ranksums(higher, lower).statistic
    n_higher: int
    n_lower: int

    @property
    def abs_difference(self) -> float:
        return abs(self.reproduced - self.published)

    @property
    def passed(self) -> bool:
        return self.abs_difference <= Z_TOLERANCE


PUBLISHED_TRAINING_Z: Final = (
    ("exp1_surplus_mixed_gt_equal", "surplus", "Exp 1", "Mixed Baseline", "Equal Baseline", 4.58),
    (
        "exp1_surplus_proportional_gt_equal",
        "surplus",
        "Exp 1",
        "Proportional Baseline",
        "Equal Baseline",
        5.28,
    ),
    (
        "exp1_gini_proportional_gt_mixed",
        "gini",
        "Exp 1",
        "Proportional Baseline",
        "Mixed Baseline",
        5.89,
    ),
    (
        "exp1_gini_proportional_gt_equal",
        "gini",
        "Exp 1",
        "Proportional Baseline",
        "Equal Baseline",
        4.76,
    ),
)
"""Koster et al. 2025, Results (Exp 1); training-mechanism comparisons only."""


def _endpoint(games: Games, endpoint: str) -> FloatArray:
    if endpoint == "surplus":
        return metrics.surplus(games.rewards, games.n_rounds)
    if endpoint == "gini":
        return metrics.reward_gini(games.rewards, games.n_rounds)
    raise ValueError(f"unknown endpoint {endpoint!r}")


def published_z(training: Games) -> list[ZComparison]:
    """Reproduce the published training-mechanism z-values (games as units)."""
    out: list[ZComparison] = []
    for name, endpoint, cohort, higher, lower, published in PUBLISHED_TRAINING_Z:
        if higher not in TRAINING_MECHANISMS or lower not in TRAINING_MECHANISMS:
            raise ValueError("held-out comparisons wait for the validation protocol freeze")
        hi = training.where(mechanism=higher, cohort=cohort)
        lo = training.where(mechanism=lower, cohort=cohort)
        if not (hi.is_human.all() and lo.is_human.all()):
            raise ValueError("published comparisons are over human games")
        out.append(
            ZComparison(
                name=name,
                endpoint=endpoint,
                cohort=cohort,
                higher=higher,
                lower=lower,
                published=published,
                reproduced=metrics.rank_sum_z(_endpoint(hi, endpoint), _endpoint(lo, endpoint)),
                n_higher=hi.n_games,
                n_lower=lo.n_games,
            )
        )
    return out


# --- training-only diagnostics of the edge-case rules -----------------------------------


def zero_previous_sum(games: Games) -> dict[str, int]:
    """Rows (t >= 1) whose previous reciprocations sum to zero, and how they are split."""
    mask = games.round_mask()
    mask[:, 0] = False
    zero = mask & (games.prev_reciprocations.sum(axis=-1) == 0.0)
    off_floor = zero & (np.abs(games.pool - POOL_FLOOR) > POOL_TOLERANCE)
    spread = games.offers.max(axis=-1) - games.offers.min(axis=-1)
    return {
        "rows": int(np.count_nonzero(zero)),
        "rows_pool_not_at_floor": int(np.count_nonzero(off_floor)),
        "rows_equal_split": int(np.count_nonzero(zero & (spread <= EXACT_OFFER))),
    }


def post_depletion(games: Games) -> dict[str, int]:
    """Training human diagnostic: what the rows after depletion (pool < 1) look like.

    Counts depleted games, depleted Exps 1-2 games with other than 40 rows, rows from the
    first depleted round on, and those rows whose pool is not the 0.01 floor, whose
    offers are not 0.0022, whose reciprocations are not 0, or whose `game_has_no_pool`
    is not True.
    """
    mask = games.round_mask()
    below = mask & (games.pool < 1.0)
    depleted = below.any(axis=1)
    length = int(games.pool.shape[1])
    first: npt.NDArray[np.int64] = np.where(depleted, below.argmax(axis=1), length)
    post: BoolArray = mask & (np.arange(length)[None, :] >= first[:, None])
    exps_1_2 = np.isin(games.cohort, ("Exp 1", "Exp 2"))
    return {
        "depleted_games": int(np.count_nonzero(depleted)),
        "depleted_exps_1_2_games_not_40_rows": int(
            np.count_nonzero(depleted & exps_1_2 & (games.n_rounds != 40))
        ),
        "post_depletion_rows": int(np.count_nonzero(post)),
        "rows_pool_not_floor": int(
            np.count_nonzero(post & (np.abs(games.pool - POOL_FLOOR) > POOL_TOLERANCE))
        ),
        "rows_offer_not_0p0022": int(
            np.count_nonzero(post & (np.abs(games.offers - 0.0022) > EXACT_OFFER).any(axis=-1))
        ),
        "rows_reciprocation_not_0": int(
            np.count_nonzero(post & (games.actions != 0.0).any(axis=-1))
        ),
        "rows_game_has_no_pool_not_true": int(
            np.count_nonzero(post & (games.game_has_no_pool != 1))
        ),
    }


def human_flags(games: Games) -> dict[str, int]:
    """Human rows whose usability flags are not True, and `game_has_no_pool` vs R < 1."""
    mask = games.round_mask() & games.is_human[:, None]
    no_pool = games.game_has_no_pool == 1
    return {
        "rows": int(np.count_nonzero(mask)),
        "rows_four_usable_players_not_true": int(
            np.count_nonzero(mask & (games.four_usable_players != 1))
        ),
        "rows_completed_not_true": int(np.count_nonzero(mask & (games.completed != 1))),
        "rows_first_x_games_not_true": int(np.count_nonzero(mask & (games.first_x_games != 1))),
        "rows_game_has_no_pool_differs_from_pool_below_1": int(
            np.count_nonzero(mask & (no_pool != (games.pool < 1.0)))
        ),
        "rows_distribute_pool_ratio_not_1": int(
            np.count_nonzero(mask & ~(games.distribute_pool_ratio == 1.0))
        ),
    }
