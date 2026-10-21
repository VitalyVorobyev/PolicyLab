"""Model Citizens E1 (exploratory): do the simulators rank mechanisms as humans did?

A small policy-ranking validity check of the citizen models used as judges in C4. The
human side is the published within-experiment rank-sum comparisons of Koster et al.
2025 that the project reproduces (C1 audit: four training-only Exp 1 z-values; G1b: the
thirteen involving held-out mechanisms). Each comparison says which of two mechanisms
gave the higher group-level metric (surplus per player-round or offer Gini) in one
experiment. The simulator side is each citizen model's closed-loop simulated mean of the
same metric under each of the two mechanisms, read from stored runs.

E1 is **exploratory**: the held-out human outcomes were seen before it was specified, it
is not part of any frozen protocol, and it can never change a C2 label or a C4 verdict.

Pure functions only (no I/O); the stage is `phasespace.citizens.e1_stage`.

Definitions:

- **Sign agreement** of a scorable pair: the simulated mean under the human-higher
  mechanism is strictly greater than under the human-lower one. A tie (equal simulated
  means, as for a model whose behaviour ignores the mechanism) is not an agreement and is
  counted separately.
- **Significant pair**: published |z| >= 1.96 (two-sided 5 %).
- **Kendall tau-b** between the human and simulated orderings of the mechanisms of one
  (experiment, metric), only where at least three mechanisms are compared there.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

Z_CRITICAL: Final = 1.96
"""Two-sided 5 % threshold on the published |z|."""

TRAINING_REGIME: Final = {
    "Equal Baseline": "Equal",
    "Mixed Baseline": "Mixed",
    "Proportional Baseline": "Proportional",
}
"""Training mechanisms -> in-sample simulation regime (40 rounds; cohort-agnostic)."""
HELDOUT_REGIME: Final = {
    "Interpolating Baseline": "R-I",
    "RL Agent (M1)": "R-M1",
    "RL Agent (M2)": "R-M2",
}
"""Held-out mechanisms -> gating simulation regime (40 rounds; cohort-agnostic)."""
SURROGATE_REGIMES: Final = ("R-M1", "R-M2")
"""Regimes simulated through a learned mechanism surrogate: usable only if accepted."""

Status = Literal["scored", "excluded", "not_scorable"]


@dataclass(frozen=True, slots=True)
class HumanPair:
    """One published within-experiment comparison: `higher` > `lower` on `endpoint`."""

    name: str
    endpoint: str  # "surplus" | "gini"
    cohort: str  # "Exp 1" .. "Exp 4"
    exp4_game: int | None  # Exp 4: game 1, 2 or 3 of each group
    higher: str
    lower: str
    published: float  # |z| as printed
    reproduced: float  # our rank-sum z (higher vs lower)
    source_run: str

    @property
    def significant(self) -> bool:
        return abs(self.published) >= Z_CRITICAL

    @property
    def heldout(self) -> bool:
        """True when either mechanism is held out (not a training mechanism)."""
        return self.higher not in TRAINING_REGIME or self.lower not in TRAINING_REGIME


def pairs_from_table(table: Mapping[str, Any]) -> list[HumanPair]:
    """Parse a z-value sample table (C1 audit or G1b layout) into pairs.

    The pair name is the `comparison` (audit) or `name` (G1b) column; `exp4_game` is
    read when present. Every row must report a positive reproduced z (higher vs lower).
    """
    cols = [str(c["name"]) for c in table["columns"]]
    name_col = "comparison" if "comparison" in cols else "name"
    out: list[HumanPair] = []
    for row in table["rows"]:
        r = dict(zip(cols, row, strict=True))
        game = r.get("exp4_game")
        pair = HumanPair(
            name=str(r[name_col]),
            endpoint=str(r["endpoint"]),
            cohort=str(r["cohort"]),
            exp4_game=None if game is None else int(game),
            higher=str(r["higher"]),
            lower=str(r["lower"]),
            published=float(r["published"]),
            reproduced=float(r["reproduced"]),
            source_run=str(table["run_id"]),
        )
        if not pair.reproduced > 0.0:
            raise ValueError(f"{pair.name}: reproduced z {pair.reproduced} is not positive")
        out.append(pair)
    return out


def regime_of(mechanism: str) -> str:
    if mechanism in TRAINING_REGIME:
        return TRAINING_REGIME[mechanism]
    if mechanism in HELDOUT_REGIME:
        return HELDOUT_REGIME[mechanism]
    raise ValueError(f"unknown mechanism {mechanism!r}")


@dataclass(frozen=True, slots=True)
class Match:
    """A pair scored (or not) for one model."""

    pair: HumanPair
    model: str
    status: Status
    reason: str
    regime_higher: str
    regime_lower: str
    sim_higher: float | None = None
    sim_lower: float | None = None

    @property
    def sim_difference(self) -> float | None:
        if self.sim_higher is None or self.sim_lower is None:
            return None
        return self.sim_higher - self.sim_lower

    @property
    def agrees(self) -> bool | None:
        """Sign agreement of a scored pair; None if the pair is not scored."""
        if self.status != "scored":
            return None
        return sign_agrees(_req(self.sim_higher), _req(self.sim_lower))

    @property
    def tie(self) -> bool:
        return self.status == "scored" and self.sim_higher == self.sim_lower


def _req(x: float | None) -> float:
    if x is None:
        raise ValueError("a scored match has both simulated values")
    return x


def sign_agrees(sim_higher: float, sim_lower: float) -> bool:
    """The simulated difference has the human sign (strictly; a tie is no agreement)."""
    return sim_higher > sim_lower


def match_pair(
    pair: HumanPair,
    model: str,
    simulated: Mapping[tuple[str, str, str], float],
    surrogate_accepted: Mapping[str, bool],
) -> Match:
    """Match a human pair to a model's simulated means `simulated[(model, regime, metric)]`.

    Order of rules: (1) a pair involving a surrogate regime whose surrogate was not
    accepted on human rows (`surrogate_accepted[regime]` false or missing) is
    **excluded**; (2) an Exp 4 pair is **not scorable** (no stored simulation matches
    Exp 4's recorded game lengths per game 1-3 under both mechanisms); (3) a pair whose
    two simulated means are not both available is **not scorable**; else **scored**.
    The simulator is cohort-agnostic (40 rounds, the same initial state, seeds per
    model and regime only), so the regime simulation stands for every 40-round cohort.
    """
    rh, rl = regime_of(pair.higher), regime_of(pair.lower)
    rejected = [r for r in (rh, rl) if r in SURROGATE_REGIMES and not surrogate_accepted.get(r)]
    if rejected:
        reason = f"{', '.join(rejected)}: mechanism surrogate not accepted on human rows"
        return Match(pair, model, "excluded", reason, rh, rl)
    if pair.exp4_game is not None:
        reason = "Exp 4 game-specific lengths: no stored simulation of both mechanisms per game"
        return Match(pair, model, "not_scorable", reason, rh, rl)
    hi = simulated.get((model, rh, pair.endpoint))
    lo = simulated.get((model, rl, pair.endpoint))
    if hi is None or lo is None:
        missing = [r for r, v in ((rh, hi), (rl, lo)) if v is None]
        reason = f"no stored simulation for {model} under {', '.join(missing)}"
        return Match(pair, model, "not_scorable", reason, rh, rl)
    return Match(pair, model, "scored", "", rh, rl, hi, lo)


@dataclass(frozen=True, slots=True)
class Score:
    """Agreement counts of one model over a set of matches."""

    model: str
    pairs: int
    excluded: int
    not_scorable: int
    scorable: int
    agreements: int
    ties: int
    significant_scorable: int
    significant_agreements: int
    heldout_scorable: int
    heldout_agreements: int

    @property
    def rate(self) -> float | None:
        return _rate(self.agreements, self.scorable)

    @property
    def significant_rate(self) -> float | None:
        return _rate(self.significant_agreements, self.significant_scorable)

    @property
    def heldout_rate(self) -> float | None:
        return _rate(self.heldout_agreements, self.heldout_scorable)


def _rate(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def score(model: str, matches: Iterable[Match]) -> Score:
    ms = [m for m in matches if m.model == model]
    scored = [m for m in ms if m.status == "scored"]
    sig = [m for m in scored if m.pair.significant]
    held = [m for m in scored if m.pair.heldout]
    return Score(
        model=model,
        pairs=len(ms),
        excluded=sum(m.status == "excluded" for m in ms),
        not_scorable=sum(m.status == "not_scorable" for m in ms),
        scorable=len(scored),
        agreements=sum(bool(m.agrees) for m in scored),
        ties=sum(m.tie for m in scored),
        significant_scorable=len(sig),
        significant_agreements=sum(bool(m.agrees) for m in sig),
        heldout_scorable=len(held),
        heldout_agreements=sum(bool(m.agrees) for m in held),
    )


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b of paired values; None when either sequence is constant or n < 2.

    tau_b = (C - D) / sqrt((n0 - n1) (n0 - n2)), with C, D the concordant and discordant
    pairs, n0 = n (n - 1) / 2, n1 and n2 the pairs tied in x and in y.
    """
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    n = len(x)
    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                tied_x += 1
            if dy == 0:
                tied_y += 1
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    n0 = n * (n - 1) // 2
    denominator = math.sqrt(float((n0 - tied_x) * (n0 - tied_y)))
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def ranking_groups(
    pairs: Iterable[HumanPair], usable: Iterable[str]
) -> dict[tuple[str, str], list[str]]:
    """(cohort, endpoint) -> the sorted mechanisms compared there in at least one pair whose
    both mechanisms are `usable`, kept only where at least three mechanisms are compared.
    Exp 4 pairs (game-specific) never form a group."""
    ok = set(usable)
    groups: dict[tuple[str, str], set[str]] = {}
    for p in pairs:
        if p.exp4_game is not None or p.higher not in ok or p.lower not in ok:
            continue
        groups.setdefault((p.cohort, p.endpoint), set()).update((p.higher, p.lower))
    return {k: sorted(v) for k, v in sorted(groups.items()) if len(v) >= 3}
