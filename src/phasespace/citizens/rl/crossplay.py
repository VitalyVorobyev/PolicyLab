"""C4 cross-play evaluation and its extrapolation diagnostics (pure; deterministic given keys).

**Matrix.** Rows are policies: planners pooled per (training model, action head) with one
entry per training seed, and the deployed rules (Equal, Mixed, Proportional,
Interpolating k = 22 and every training model's re-tuned k*). Columns are citizen
models. Every policy faces the same citizen draws in a column (one citizen key: common
random numbers), so differences are paired by game. Planners are evaluated
deterministically (shares = softmax(mu)) and with their sampling policy (one planner
noise key); the rules are deterministic.

**Diagnostics** (review item 1; reported, not gating):

- the per-round kept share 1 - sum e / R (mean over games and seeds), including the
  last two rounds (the endgame, review item 4);
- the B1 ensemble's teacher-forced member disagreement (mean pairwise total variation,
  `citizens.b1_member_disagreement`) on every trajectory;
- for B2, B3 and B23, the fraction of non-forced decisions whose inputs fall outside the
  support of the model's training data: outside the [min, max] range of any input the
  model reads (`c4_config.OOD_FEATURES`), and outside the occupied cells of a joint
  (x1 / 4, x4) grid. The support comes from the scored fit-split decisions the C2 fit
  used (`training_support`); held-out rows are never read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import equinox as eqx
import jax
import numpy as np
from jax import Array

from phasespace.citizens.game import EXCLUSION_THRESHOLD
from phasespace.citizens.models import b2_condcoop as b2
from phasespace.citizens.models.base import Streams, streams_from_arrays
from phasespace.citizens.rl import c4_config, checks, ppo, stage, twin
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl.planner import Planner, sampling_step

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
type Mode = Literal["deterministic", "sampled"]

MODES: Final[tuple[Mode, Mode]] = ("deterministic", "sampled")


# --- policies --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """One row of the matrix: a pooled planner (one entry per seed) or a deployed rule."""

    name: str
    kind: Literal["planner", "rule"]
    training_model: str | None = None
    retention: bool | None = None
    seeds: tuple[int, ...] = ()
    planners: tuple[Planner, ...] = field(default=(), repr=False)
    rule_kind: int | None = None  # 0: constant weight w, 1: Interpolating k
    rule_param: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "training_model": self.training_model,
            "retention": self.retention,
            "seeds": list(self.seeds),
            "rule": None
            if self.rule_kind is None
            else {"kind": ("weight", "interpolating_k")[self.rule_kind], "param": self.rule_param},
        }


def planner_name(model: str, retention: bool) -> str:
    return f"Planner[{model}]" if retention else f"Planner-noret[{model}]"


def k_star_name(model: str) -> str:
    return f"{c4_config.K_STAR_RULE}[{model}]"


def planner_policy(
    model: str, retention: bool, seeds: Sequence[int], planners: Sequence[Planner]
) -> Policy:
    if len(seeds) != len(planners) or not seeds:
        raise ValueError("one planner per seed, at least one")
    if any(p.retention != retention for p in planners):
        raise ValueError("planners of one row share the action head")
    return Policy(
        planner_name(model, retention), "planner", model, retention, tuple(seeds), tuple(planners)
    )


def rule_policies(k_star: Mapping[str, float]) -> list[Policy]:
    """Equal, Mixed, Proportional, Interpolating k = 22 and each model's k* (deployed rule)."""
    out = [
        Policy(name, "rule", rule_kind=0, rule_param=c3config.FIXED_WEIGHTS[name])
        for name in c4_config.FIXED_RULES
    ]
    out.append(
        Policy(c4_config.K_PUBLISHED_RULE, "rule", rule_kind=1, rule_param=c3config.K_PUBLISHED)
    )
    out += [
        Policy(k_star_name(m), "rule", training_model=m, rule_kind=1, rule_param=float(k))
        for m, k in k_star.items()
    ]
    return out


# --- extrapolation: support of the training data ----------------------------------------------


def decision_inputs(streams: Streams) -> dict[str, FloatArray]:
    """(B, T) inputs of every decision; NaN where an input does not apply.

    x1..x4 are B2's features (`b2_condcoop.stream_features`); x2 applies when the own
    previous offer was >= 1, x3 when the others' previous offers sum to >= 1 (both from
    round 2); `own_offer` is e_{i,t}; `own_prev_offer` is e_{i,t-1} when it was >= 1.
    """
    x = b2.stream_features(streams)
    b, t_max = streams.n_streams, streams.n_rounds_max
    later = np.broadcast_to(np.arange(t_max)[None, :] > 0, (b, t_max))
    prev = np.concatenate([np.zeros((b, 1, 4)), streams.offers[:, :-1]], axis=1)
    own_ok = later & (prev[:, :, 0] >= EXCLUSION_THRESHOLD)
    others_ok = later & (prev[:, :, 1:].sum(axis=2) >= EXCLUSION_THRESHOLD)
    nan = np.full((b, t_max), np.nan)
    return {
        "x1": x[:, :, 0],
        "x2": np.where(own_ok, x[:, :, 1], nan),
        "x3": np.where(others_ok, x[:, :, 2], nan),
        "x4": x[:, :, 3],
        "own_offer": streams.own_offer,
        "own_prev_offer": np.where(own_ok, prev[:, :, 0], nan),
    }


def _grid_cells(x1: FloatArray, x4: FloatArray) -> tuple[FloatArray, FloatArray]:
    n = c4_config.OOD_GRID_BINS
    i = np.clip(np.floor(x1 / 4.0 * n), 0, n - 1)
    j = np.clip(np.floor(x4 * n), 0, n - 1)
    return i, j


@dataclass(frozen=True)
class TrainingSupport:
    """Per-model input ranges and the joint (x1 / 4, x4) occupancy of the training data."""

    model: str
    features: tuple[str, ...]
    lo: dict[str, float]
    hi: dict[str, float]
    grid: FloatArray  # (bins, bins) decision counts
    decisions: int

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "features": list(self.features),
            "range": {f: [self.lo[f], self.hi[f]] for f in self.features},
            "grid_bins": int(self.grid.shape[0]),
            "grid_occupied_cells": int((self.grid > 0).sum()),
            "decisions": self.decisions,
        }


def training_support(streams: Streams, model: str) -> TrainingSupport:
    """The support of `model`'s inputs over the scored decisions of `streams` (the C2 fit
    split's human training games; the caller must pass training rows only)."""
    features = c4_config.OOD_FEATURES[model]
    inputs = decision_inputs(streams)
    scored = streams.scored
    lo = {f: float(np.nanmin(inputs[f][scored])) for f in features}
    hi = {f: float(np.nanmax(inputs[f][scored])) for f in features}
    n = c4_config.OOD_GRID_BINS
    i, j = _grid_cells(inputs["x1"][scored], inputs["x4"][scored])
    grid = np.zeros((n, n))
    np.add.at(grid, (i.astype(np.int64), j.astype(np.int64)), 1.0)
    return TrainingSupport(model, features, lo, hi, grid, int(scored.sum()))


def trajectory_streams(traj: rc.Trajectory) -> Streams:
    """The trajectory's games as streams (own first), every round recorded."""
    offers = np.asarray(traj.offers)
    return streams_from_arrays(
        offers,
        np.asarray(traj.actions),
        np.asarray(traj.pool),
        np.full(offers.shape[0], offers.shape[1]),
    )


def ood_fractions(support: TrainingSupport, streams: Streams) -> dict[str, Any]:
    """Fractions of the non-forced decisions (e >= 1) outside the training support."""
    inputs = decision_inputs(streams)
    scored = streams.scored
    n = int(scored.sum())
    per_feature: dict[str, float] = {}
    outside_any = np.zeros_like(scored)
    for f in support.features:
        v = inputs[f]
        out = np.isfinite(v) & ((v < support.lo[f]) | (v > support.hi[f])) & scored
        per_feature[f] = float(out.sum() / n) if n else 0.0
        outside_any |= out
    i, j = _grid_cells(inputs["x1"], inputs["x4"])
    off_grid = (support.grid[i.astype(np.int64), j.astype(np.int64)] == 0.0) & scored
    return {
        "decisions": n,
        "outside_range_any": float(outside_any.sum() / n) if n else 0.0,
        "outside_range_per_feature": per_feature,
        "off_grid": float(off_grid.sum() / n) if n else 0.0,
    }


# --- evaluation ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostics:
    """Optional diagnostic inputs: the B1 port (disagreement) and per-model supports."""

    b1: rc.B1Port | None = None
    supports: Mapping[str, TrainingSupport] = field(default_factory=dict[str, TrainingSupport])


@dataclass
class Series:
    """One evaluated (policy, column, mode, seed): per-game totals and diagnostics."""

    policy: str
    column: str
    mode: Mode
    seed: int | None
    total_surplus: FloatArray
    gini: FloatArray
    kept_share: FloatArray  # (T,)
    decisions_per_game: float
    b1_disagreement: dict[str, float] | None
    ood: dict[str, Any] | None


_member_tv = eqx.filter_jit(rc.b1_member_disagreement)


def b1_disagreement(port: rc.B1Port, traj: rc.Trajectory) -> dict[str, float]:
    """Mean and 90th percentile of the B1 members' mean pairwise TV over the trajectory's
    non-forced decisions (as `stage.disagreement`, one compiled function for all calls)."""
    tv = np.asarray(_member_tv(port, traj))
    values = tv[np.isfinite(tv)]
    if values.size == 0:
        return {"mean_tv": float("nan"), "p90_tv": float("nan"), "decisions": 0}
    return {
        "mean_tv": float(values.mean()),
        "p90_tv": float(np.quantile(values, 0.9)),
        "decisions": int(values.size),
    }


def _series(
    policy: str,
    column: str,
    mode: Mode,
    seed: int | None,
    traj: rc.Trajectory,
    diagnostics: Diagnostics,
) -> Series:
    m = checks.trajectory_metrics(traj)
    support = diagnostics.supports.get(column)
    return Series(
        policy=policy,
        column=column,
        mode=mode,
        seed=seed,
        total_surplus=m["total_surplus"],
        gini=m["gini"],
        kept_share=stage.kept_share(traj),
        decisions_per_game=float((np.asarray(traj.offers) >= 1.0).sum(axis=(1, 2)).mean()),
        b1_disagreement=None if diagnostics.b1 is None else b1_disagreement(diagnostics.b1, traj),
        ood=None if support is None else ood_fractions(support, trajectory_streams(traj)),
    )


def evaluate_column(
    policies: Sequence[Policy],
    column: str,
    port: rc.JaxCitizen,
    citizen_key: Array,
    noise_key: Array,
    n_games: int,
    diagnostics: Diagnostics,
) -> list[Series]:
    """Every policy against one citizen model on the same `n_games` games."""
    twin.require_x64()
    rounds = c3config.ROUNDS

    @eqx.filter_jit
    def deterministic(p: Planner) -> rc.Trajectory:
        return ppo.evaluate_planner(p, port, citizen_key, n_games, rounds)[0]

    @eqx.filter_jit
    def sampled(p: Planner) -> rc.Trajectory:
        return rc.rollout(
            port,
            sampling_step(p),
            p.initial_state(n_games),
            citizen_key,
            noise_key,
            n_games,
            rounds,
        )[0]

    rule = stage.weight_runner(port, n_games)
    out: list[Series] = []
    for policy in policies:
        if policy.kind == "planner":
            for seed, planner in zip(policy.seeds, policy.planners, strict=True):
                out.append(
                    _series(
                        policy.name,
                        column,
                        "deterministic",
                        seed,
                        deterministic(planner),
                        diagnostics,
                    )
                )
                out.append(
                    _series(policy.name, column, "sampled", seed, sampled(planner), diagnostics)
                )
        else:
            if policy.rule_kind is None or policy.rule_param is None:
                raise ValueError(f"{policy.name}: a rule needs its kind and parameter")
            traj = rule(twin.f64(policy.rule_kind), twin.f64(policy.rule_param), citizen_key)
            out.append(_series(policy.name, column, "deterministic", None, traj, diagnostics))
    return out


# --- statistics -----------------------------------------------------------------------------------


def group_bootstrap(
    a: FloatArray, b: FloatArray, n_resamples: int, seed: int, level: float = c4_config.CI_LEVEL
) -> tuple[float, float, float]:
    """Mean of a (Sa seeds x J games) minus b (Sb x J), paired by game, with a percentile
    CI: each resample draws Sa seeds of a, Sb seeds of b and J games with replacement
    (the same games for both)."""
    rng = np.random.default_rng(seed)
    sa, j = a.shape
    sb, jb = b.shape
    if j != jb:
        raise ValueError("a and b must share the games")
    stats = np.empty(n_resamples)
    for r in range(n_resamples):
        ia = rng.integers(0, sa, sa)
        ib = rng.integers(0, sb, sb)
        g = rng.integers(0, j, j)
        stats[r] = a[np.ix_(ia, g)].mean() - b[np.ix_(ib, g)].mean()
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def stack_totals(series: Sequence[Series], policy: str, column: str, mode: Mode) -> FloatArray:
    """(S, J) per-game totals of one policy, column and mode (seeds in input order)."""
    rows = [
        s.total_surplus
        for s in series
        if s.policy == policy and s.column == column and s.mode == mode
    ]
    if not rows:
        raise KeyError((policy, column, mode))
    return np.stack(rows)


def paired_differences(
    policies: Sequence[Policy],
    columns: Sequence[str],
    series: Sequence[Series],
    n_resamples: int = c4_config.BOOTSTRAP_RESAMPLES,
    seed: int = c4_config.BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Every planner row (both modes) minus every rule and every other planner row (same
    mode), per column: paired mean, per-seed differences and the bootstrap CI."""
    planners = [p for p in policies if p.kind == "planner"]
    rules = [p for p in policies if p.kind == "rule"]
    out: list[dict[str, Any]] = []
    for column in columns:
        for p in planners:
            for mode in MODES:
                a = stack_totals(series, p.name, column, mode)
                others: list[tuple[str, Mode]] = []
                for r in rules:
                    others.append((r.name, "deterministic"))
                for q in planners:
                    if q.name != p.name:
                        others.append((q.name, mode))
                for other, other_mode in others:
                    b = stack_totals(series, other, column, other_mode)
                    mean, lo, hi = group_bootstrap(a, b, n_resamples, seed)
                    out.append(
                        {
                            "column": column,
                            "policy": p.name,
                            "mode": mode,
                            "versus": other,
                            "versus_mode": other_mode,
                            "difference_mean": mean,
                            "difference_ci": [lo, hi],
                            "per_seed_difference": [
                                float(x) for x in (a - b.mean(axis=0)).mean(axis=1)
                            ],
                        }
                    )
    return out


def summarise(
    policies: Sequence[Policy], columns: Sequence[str], series: Sequence[Series]
) -> dict[str, Any]:
    """The matrix: per policy and column, per mode, the mean, per-seed means and the
    diagnostics (kept share per round and in the last rounds, B1 disagreement, OOD)."""
    last = list(c4_config.LAST_ROUNDS)
    cells: dict[str, dict[str, Any]] = {}
    for p in policies:
        row: dict[str, Any] = {}
        for column in columns:
            cell: dict[str, Any] = {}
            for mode in MODES:
                chosen = [
                    s
                    for s in series
                    if s.policy == p.name and s.column == column and s.mode == mode
                ]
                if not chosen:
                    cell[mode] = None
                    continue
                totals = np.stack([s.total_surplus for s in chosen])
                kept = np.mean([s.kept_share for s in chosen], axis=0)
                cell[mode] = {
                    "total_surplus_mean": float(totals.mean()),
                    "total_surplus_per_seed": [float(x) for x in totals.mean(axis=1)],
                    "seeds": [s.seed for s in chosen],
                    "gini_mean": float(np.mean([s.gini.mean() for s in chosen])),
                    "decisions_per_game_mean": float(
                        np.mean([s.decisions_per_game for s in chosen])
                    ),
                    "kept_share_per_round": kept.tolist(),
                    "kept_share_last_rounds": [float(kept[t]) for t in last],
                    "b1_disagreement": [s.b1_disagreement for s in chosen],
                    "ood": [s.ood for s in chosen],
                }
            row[column] = cell
        cells[p.name] = row
    return {
        "columns": list(columns),
        "policies": [p.describe() for p in policies],
        "cells": cells,
    }


def series_document(series: Sequence[Series]) -> list[dict[str, Any]]:
    """Per-game totals and Gini of every series (the evaluation's raw numbers)."""
    return [
        {
            "policy": s.policy,
            "column": s.column,
            "mode": s.mode,
            "seed": s.seed,
            "total_surplus": [float(x) for x in s.total_surplus],
            "gini": [float(x) for x in s.gini],
        }
        for s in series
    ]


def eval_keys() -> tuple[Array, Array]:
    """The C4 evaluation games' citizen key and the sampled policy's noise key."""
    return jax.random.key(c4_config.EVAL_KEY_SEED), jax.random.key(c4_config.SAMPLED_NOISE_KEY_SEED)
