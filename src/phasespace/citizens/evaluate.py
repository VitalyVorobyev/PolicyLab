"""C2 evaluation: human reference CIs, closed-loop outcomes, likelihood gain, criterion.

Protocol 8. For a regime (a mechanism and its human groups):

- **Units.** A unit is a human group: one game in Exps 1-3, and the three games of an
  Exp 4 group (same `launch_id`), whose unit metric is the mean over its games. Strata are
  the experiments (cohorts) of a pooled regime.
- **Human reference.** The mean over units of each metric and its 95 % percentile
  bootstrap CI: B = 10,000 resamples of units with replacement, stratified by experiment
  (each stratum resampled to its own size), generator default_rng(9001). The same
  resample matrix serves every metric and every model's likelihood gain of the regime.
- **Outcomes (a).** Per metric, the simulated mean over S games must lie in
  [lo - tau, hi + tau]; (a) holds when >= 3 of the 4 metrics do.
- **Likelihood (b).** l_m = mean log P_m(c) over the regime's non-forced decisions
  (teacher-forced on each recorded history) and Delta-l_m = l_m - l_M0. Its bootstrap
  resamples units (same matrix) and recomputes the decision-weighted ratio
  sum_u S_u / sum_u N_u, with S_u the unit's sum of log P_m - log P_M0 and N_u its number
  of non-forced decisions. (b) holds when Delta-l > 0 and the 2.5 % percentile > 0.
- **Validated (closed-loop)** = (a) and (b). B0 and M0-sim are controls; M0-sim has
  Delta-l = 0 by definition and is reported for (a) only. Where M0-sim passes (a), a
  model must also have a strictly lower mean normalised W1 than M0-sim ("must beat
  marginal", amendment A22; applied by `phasespace.citizens.heldout.regime_label`).
- **Distributional diagnostic**: W1 between the simulated game-level and the human
  unit-level distribution of each metric, with the human-vs-human reference W1 between
  random halves of the units (1,000 splits, seed 9001). The normalised W1 of a metric is
  W1 divided by the human between-unit SD (ddof 1), or by the tolerance tau when that SD
  is 0 (`w1_scale`); its mean over the 4 metrics is the A22 statistic.
- **Declared data handling (A29).** Recorded decisions that are not an integer
  0 <= c <= floor(e) are excluded from the likelihood and counted; a scored decision
  with P(c) < 1e-12 is scored at log(1e-12) and counted.
- **Players active (A30)** is #{i : e_i >= 1} in the last round, for humans as for
  simulated games.

Held-out regimes are evaluated only by `phasespace.citizens.heldout` (`evaluate`),
whose guard (`require_post_freeze`) refuses unless the validation protocol is frozen,
committed and the tree clean, before `data.load_heldout` is called.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy import stats

from phasespace.citizens import config, metrics
from phasespace.citizens.models.base import (
    CitizenModel,
    Streams,
    streams_from_games,
    teacher_forced_log_prob,
)
from phasespace.citizens.records import Games
from phasespace.citizens.simulate import (
    Mechanism,
    SimulatedGames,
    game_metrics,
    simulate,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type StrArray = npt.NDArray[np.str_]

TRAINING_REGIMES: Final = {
    "Equal": ("Equal Baseline", ("Exp 1",)),
    "Mixed": ("Mixed Baseline", ("Exp 1",)),
    "Proportional": ("Proportional Baseline", ("Exp 1", "Exp 2")),
}
"""In-sample regimes (protocol 8): training mechanisms, 40-round human games."""
GATING_REGIMES: Final = {
    "R-I": ("Interpolating Baseline", ("Exp 2", "Exp 3"), "Interpolating"),
    "R-M1": ("RL Agent (M1)", ("Exp 1", "Exp 2"), "M1"),
    "R-M2": ("RL Agent (M2)", ("Exp 3",), "M2"),
}
"""Gating regimes: (mechanism, experiments pooled, simulation regime key)."""
SIM_MODELS: Final = ("B0", "M0-sim", "B1", "B2", "B3", "B23")
LIKELIHOOD_MODELS: Final = ("B0", "M0", "P0", "B1", "B2", "B3", "B23")
CONTROLS: Final = ("B0", "M0", "M0-sim", "P0")


# --- units and bootstrap -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Units:
    """Human units of a regime: per-game unit index, unit strata and keys."""

    game_unit: IntArray  # (G,)
    stratum: StrArray  # (U,)
    keys: tuple[tuple[str, str], ...]  # (cohort, launch_id)

    @property
    def n_units(self) -> int:
        return len(self.keys)


def units_of(games: Games) -> Units:
    """Group games into units by (cohort, launch_id): Exp 4 sets share their launch_id."""
    keys = list(zip(games.cohort.tolist(), games.launch_id.tolist(), strict=True))
    unique = sorted(set(keys))
    index = {k: i for i, k in enumerate(unique)}
    return Units(
        game_unit=np.array([index[k] for k in keys], dtype=np.int64),
        stratum=np.array([k[0] for k in unique], dtype=np.str_),
        keys=tuple(unique),
    )


def unit_means(values: npt.ArrayLike, units: Units) -> FloatArray:
    """(U,) mean over each unit's games of a per-game value."""
    v = np.asarray(values, dtype=np.float64)
    sums = np.bincount(units.game_unit, weights=v, minlength=units.n_units)
    counts = np.bincount(units.game_unit, minlength=units.n_units)
    return sums / counts


def bootstrap_indices(
    strata: npt.ArrayLike, n_resamples: int, rng: np.random.Generator
) -> IntArray:
    """(B, U) unit indices: each stratum (sorted) resampled with replacement to its size."""
    s = np.asarray(strata)
    columns: list[IntArray] = []
    for label in sorted(set(s.tolist())):
        members = np.flatnonzero(s == label)
        draws = rng.integers(0, members.shape[0], size=(n_resamples, members.shape[0]))
        columns.append(members[draws])
    return np.concatenate(columns, axis=1)


def percentile_ci(samples: FloatArray, level: float = config.CI_LEVEL) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    lo: float
    hi: float


def mean_ci(values: npt.ArrayLike, index: IntArray) -> Estimate:
    """Mean over units and its percentile bootstrap CI under the resample matrix."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = percentile_ci(v[index].mean(axis=1))
    return Estimate(float(v.mean()), lo, hi)


def ratio_ci(numerator: npt.ArrayLike, denominator: npt.ArrayLike, index: IntArray) -> Estimate:
    """sum(num) / sum(den) over units and its bootstrap CI (decision-weighted mean)."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    lo, hi = percentile_ci(num[index].sum(axis=1) / den[index].sum(axis=1))
    return Estimate(float(num.sum() / den.sum()), lo, hi)


# --- human reference and likelihood -------------------------------------------------------


def human_game_metrics(games: Games) -> dict[str, FloatArray]:
    """Per-game metrics from all recorded rows (notebook definitions); players active is
    #{i : e_i >= 1} in the last round (A30), as for simulated games."""
    active = metrics.players_offered_at_least_one(games.offers)
    return game_metrics(games.rewards, games.pool, active, games.n_rounds)


def w1_scale(unit_values: npt.ArrayLike, tau: float) -> float:
    """A22 normaliser of a metric's W1: the human between-unit SD (ddof 1), or tau if 0."""
    v = np.asarray(unit_values, dtype=np.float64)
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return sd if sd > 0.0 else tau


@dataclass(frozen=True, slots=True)
class LikelihoodSums:
    """Per-unit sums of log P(c) over non-forced decisions, and decision counts."""

    log_prob: FloatArray  # (U,)
    decisions: FloatArray  # (U,)
    floored: int = 0  # scored decisions with P < PMF_FLOOR, scored at log(PMF_FLOOR)

    @property
    def mean(self) -> float:
        return float(self.log_prob.sum() / self.decisions.sum())


def likelihood_sums(model: CitizenModel[Any], streams: Streams, units: Units) -> LikelihoodSums:
    """Per-unit log-likelihood sums; P(c) < 1e-12 is floored and counted (A29)."""
    lp = teacher_forced_log_prob(model, streams)
    scored = streams.scored
    log_floor = float(np.log(config.PMF_FLOOR))
    low = scored & ~(lp >= log_floor)  # NaN (never on scored rows) also counts
    lp = np.where(low, log_floor, lp)
    unit = units.game_unit[streams.game]
    per_stream = np.where(scored, lp, 0.0).sum(axis=1)
    if not np.all(np.isfinite(per_stream)):
        raise ValueError(f"{model.name}: a scored decision has a non-finite log probability")
    return LikelihoodSums(
        log_prob=np.bincount(unit, weights=per_stream, minlength=units.n_units),
        decisions=np.bincount(
            unit, weights=scored.sum(axis=1).astype(np.float64), minlength=units.n_units
        ),
        floored=int(np.count_nonzero(low)),
    )


# --- regime evaluation -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    metric: str
    human: Estimate
    tolerance: float
    sim_mean: float
    sim_sd: float
    in_range: bool
    w1: float
    w1_human_halves: float  # median W1 between random halves of the human units
    w1_normalised: float = math.nan  # w1 / w1_scale (A22)


@dataclass(frozen=True, slots=True)
class ModelResult:
    model: str
    outcomes: tuple[OutcomeRow, ...] | None  # None: not simulated (P0, M0 likelihood)
    log_lik: float | None  # mean log P per decision; None for M0-sim
    delta: Estimate | None  # vs M0; None for M0 and M0-sim
    sim_seconds: float | None
    sim_games: int | None
    floored_decisions: int = 0  # A29: scored decisions with P < 1e-12

    @property
    def mean_normalised_w1(self) -> float | None:
        """A22: mean over the 4 metrics of W1 / scale; None if not simulated."""
        if self.outcomes is None:
            return None
        return float(np.mean([o.w1_normalised for o in self.outcomes]))

    @property
    def metrics_in_range(self) -> int | None:
        return None if self.outcomes is None else sum(o.in_range for o in self.outcomes)

    @property
    def pass_outcomes(self) -> bool | None:
        n = self.metrics_in_range
        return None if n is None else n >= config.OUTCOME_METRICS_REQUIRED

    @property
    def pass_likelihood(self) -> bool | None:
        if self.delta is None:
            return None
        return self.delta.value > 0.0 and self.delta.lo > 0.0

    @property
    def validated(self) -> bool | None:
        """(a) and (b); None for models without both parts (M0, M0-sim, P0)."""
        a, b = self.pass_outcomes, self.pass_likelihood
        return None if a is None or b is None else a and b


@dataclass(frozen=True, slots=True)
class RegimeResult:
    regime: str
    sim_regime: str
    n_units: int
    n_games: int
    n_decisions: int
    strata: dict[str, int]
    human: dict[str, Estimate]
    models: tuple[ModelResult, ...]
    feasible_games: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)
    w1_scale: dict[str, float] = field(default_factory=dict[str, float])
    excluded_decisions: int = 0  # A29: anomalous recorded decisions, not scored


def _w1_halves(values: FloatArray, rng: np.random.Generator, splits: int) -> float:
    n = values.shape[0]
    out = np.empty(splits)
    for k in range(splits):
        perm = rng.permutation(n)
        out[k] = stats.wasserstein_distance(values[perm[: n // 2]], values[perm[n // 2 :]])
    return float(np.median(out))


def evaluate_regime(
    regime: str,
    sim_regime: str,
    human: Games,
    *,
    likelihood_models: Mapping[str, CitizenModel[Any]],
    sim_models: Mapping[str, CitizenModel[Any]],
    mechanism: Mechanism[Any] | None,
    n_sim: int = config.SIM_GAMES,
    n_rounds: int | npt.ArrayLike = config.SIM_ROUNDS,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.BOOTSTRAP_SEED,
    w1_splits: int = config.W1_SPLITS,
    sim_unit: npt.ArrayLike | None = None,
) -> tuple[RegimeResult, dict[str, SimulatedGames]]:
    """Criterion of protocol 8 for every model on one regime.

    `likelihood_models` must contain "M0" (the reference); `sim_models` are simulated
    with seed `config.sim_seed(name, sim_regime)` when `mechanism` is given (otherwise
    the regime is one-step only and outcomes are None). `sim_unit` (one index per
    simulated game) groups simulated games into units whose metric is the mean over
    their games, as human Exp 4 sets are (M2 Exp 4: 3 games per simulated set); by
    default every simulated game is its own unit.
    """
    if "M0" not in likelihood_models:
        raise ValueError("the likelihood models need the M0 reference")
    units = units_of(human)
    index = bootstrap_indices(units.stratum, n_resamples, np.random.default_rng(seed))
    per_game = human_game_metrics(human)
    unit_metric = {k: unit_means(v, units) for k, v in per_game.items()}
    reference = {k: mean_ci(v, index) for k, v in unit_metric.items()}
    w1_rng = np.random.default_rng(seed)
    halves = {k: _w1_halves(unit_metric[k], w1_rng, w1_splits) for k in config.METRICS}
    scale = {k: w1_scale(unit_metric[k], config.TOLERANCE[k]) for k in config.METRICS}
    streams = streams_from_games(human)
    sums = {name: likelihood_sums(m, streams, units) for name, m in likelihood_models.items()}
    m0 = sums["M0"]
    simulated: dict[str, SimulatedGames] = {}

    def run(name: str) -> tuple[tuple[OutcomeRow, ...] | None, float | None]:
        if mechanism is None:
            return None, None
        rng = np.random.default_rng(config.sim_seed(name, sim_regime))
        start = time.perf_counter()
        games = simulate(sim_models[name], mechanism, n_sim, rng, n_rounds=n_rounds)
        seconds = time.perf_counter() - start
        simulated[name] = games
        sim_metric = games.game_metrics()
        if sim_unit is not None:
            index = np.asarray(sim_unit, dtype=np.int64)
            if index.shape != (n_sim,):
                raise ValueError("sim_unit needs one unit index per simulated game")
            counts = np.bincount(index)
            sim_metric = {k: np.bincount(index, weights=v) / counts for k, v in sim_metric.items()}
        rows: list[OutcomeRow] = []
        for metric in config.METRICS:
            est, tau = reference[metric], config.TOLERANCE[metric]
            mean = float(sim_metric[metric].mean())
            w1 = float(stats.wasserstein_distance(sim_metric[metric], unit_metric[metric]))
            rows.append(
                OutcomeRow(
                    metric=metric,
                    human=est,
                    tolerance=tau,
                    sim_mean=mean,
                    sim_sd=float(sim_metric[metric].std(ddof=1)),
                    in_range=est.lo - tau <= mean <= est.hi + tau,
                    w1=w1,
                    w1_human_halves=halves[metric],
                    w1_normalised=w1 / scale[metric],
                )
            )
        return tuple(rows), seconds

    results: list[ModelResult] = []
    for name, lik in sums.items():
        if name == "M0":
            results.append(ModelResult(name, None, lik.mean, None, None, None, lik.floored))
            continue
        delta = ratio_ci(lik.log_prob - m0.log_prob, m0.decisions, index)
        outcomes, seconds = run(name) if name in sim_models else (None, None)
        n = n_sim if outcomes is not None else None
        results.append(ModelResult(name, outcomes, lik.mean, delta, seconds, n, lik.floored))
    for name in sim_models:
        if name not in sums:
            outcomes, seconds = run(name)
            n = n_sim if outcomes is not None else None
            results.append(ModelResult(name, outcomes, None, None, seconds, n))
    strata = {s: int((units.stratum == s).sum()) for s in sorted(set(units.stratum.tolist()))}
    return (
        RegimeResult(
            regime=regime,
            sim_regime=sim_regime,
            n_units=units.n_units,
            n_games=human.n_games,
            n_decisions=int(m0.decisions.sum()),
            strata=strata,
            human=reference,
            models=tuple(results),
            feasible_games=sum(g.n_games for g in simulated.values()),
            w1_scale=scale,
            excluded_decisions=int(np.count_nonzero(streams.excluded)),
        ),
        simulated,
    )


# --- tables --------------------------------------------------------------------------------


def outcome_rows(results: Sequence[RegimeResult]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for r in results:
        for m in r.models:
            for o in m.outcomes or ():
                rows.append(
                    [
                        r.regime,
                        m.model,
                        o.metric,
                        o.human.value,
                        o.human.lo,
                        o.human.hi,
                        o.tolerance,
                        o.sim_mean,
                        o.sim_sd,
                        "true" if o.in_range else "false",
                        o.w1,
                        o.w1_human_halves,
                        r.w1_scale.get(o.metric),
                        o.w1_normalised,
                    ]
                )
    return rows


OUTCOME_COLUMNS: Final = [
    ("regime", None, "regime (training regimes: Equal, Mixed, Proportional)"),
    ("model", None, "simulated model (B0 and M0-sim are controls)"),
    ("metric", None, "surplus, gini, depletion_round or players_active (notebook definitions)"),
    ("human_mean", None, "mean over human units"),
    ("human_ci_lo", None, "2.5 % percentile of the stratified unit bootstrap (B=10000, 9001)"),
    ("human_ci_hi", None, "97.5 % percentile"),
    ("tolerance", None, "resolution tolerance tau of the metric"),
    ("sim_mean", None, "mean over S simulated games"),
    ("sim_sd", None, "SD over the simulated games"),
    ("in_range", None, "lo - tau <= sim_mean <= hi + tau"),
    ("w1", None, "Wasserstein-1 between simulated games and human units (diagnostic)"),
    ("w1_human_halves", None, "median W1 between random halves of the human units"),
    ("w1_scale", None, "human between-unit SD (ddof 1), or tau if that SD is 0 (A22)"),
    ("w1_normalised", "dimensionless", "w1 / w1_scale (A22)"),
]


def criterion_rows(results: Sequence[RegimeResult]) -> list[list[Any]]:
    rows: list[list[Any]] = []

    def flag(v: bool | None) -> str | None:
        return None if v is None else ("true" if v else "false")

    for r in results:
        for m in r.models:
            d = m.delta
            rows.append(
                [
                    r.regime,
                    m.model,
                    m.metrics_in_range,
                    flag(m.pass_outcomes),
                    m.log_lik,
                    None if d is None else d.value,
                    None if d is None else d.lo,
                    None if d is None else d.hi,
                    flag(m.pass_likelihood),
                    flag(m.validated),
                    r.n_units,
                    r.n_decisions,
                    m.mean_normalised_w1,
                    m.floored_decisions,
                    r.excluded_decisions,
                ]
            )
    return rows


CRITERION_COLUMNS: Final = [
    ("regime", None, "regime"),
    ("model", None, "model (B0, M0, M0-sim, P0 are controls or diagnostics)"),
    ("metrics_in_range", "count", "metrics whose simulated mean is in the widened human CI"),
    ("pass_outcomes", None, "(a): metrics_in_range >= 3"),
    ("log_lik", "nat", "mean log P(c) per non-forced decision (teacher-forced)"),
    ("delta_ell", "nat", "log_lik - log_lik(M0)"),
    ("delta_ci_lo", "nat", "2.5 % percentile of the unit bootstrap of delta_ell"),
    ("delta_ci_hi", "nat", "97.5 % percentile"),
    ("pass_likelihood", None, "(b): delta_ell > 0 and delta_ci_lo > 0"),
    ("validated", None, "(a) and (b); null where a part is undefined"),
    ("units", "count", "human units of the regime"),
    ("decisions", "count", "non-forced human decisions of the regime (scored)"),
    ("mean_normalised_w1", "dimensionless", "mean over the 4 metrics of w1_normalised (A22)"),
    ("floored_decisions", "count", "scored decisions with P(c) < 1e-12, scored at 1e-12 (A29)"),
    ("excluded_decisions", "count", "anomalous recorded decisions excluded from scoring (A29)"),
]


def summary(results: Sequence[RegimeResult]) -> str:
    """A plain-text criterion table."""
    lines = [
        f"{'regime':<14}{'model':<8}{'in range':>9}{'(a)':>6}{'Δℓ':>9}"
        f"{'CI lo':>9}{'CI hi':>9}{'(b)':>6}{'valid':>7}"
    ]
    for row in criterion_rows(results):
        regime, model, n_in, a, _, dl, lo, hi, b, v = row[:10]

        def num(x: float | None) -> str:
            return f"{x:9.4f}" if x is not None else f"{'-':>9}"

        lines.append(
            f"{regime:<14}{model:<8}{n_in if n_in is not None else '-':>9}{a or '-':>6}"
            f"{num(dl)}{num(lo)}{num(hi)}{b or '-':>6}{v or '-':>7}"
        )
    return "\n".join(lines)


# --- regimes from games ----------------------------------------------------------------------


def regime_games(games: Games, mechanism: str, cohorts: Sequence[str]) -> Games:
    """Human games of `mechanism` in `cohorts` (40-round experiments only)."""
    mask = games.is_human & (games.mechanism == mechanism) & np.isin(games.cohort, cohorts)
    return games.select(mask)


def main(argv: Sequence[str] | None = None) -> int:
    """`heldout ...` is the evaluation stage of `phasespace.citizens.heldout` (same flags)."""
    from phasespace.citizens import heldout  # the run writer; imports this module

    args = list(argv) if argv is not None else None
    if args is None:
        import sys

        args = sys.argv[1:]
    if not args or args[0] != "heldout":
        raise SystemExit("usage: python -m phasespace.citizens.evaluate heldout [flags]")
    return heldout.main(["evaluate", *args[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
