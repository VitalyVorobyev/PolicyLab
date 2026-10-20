"""C4c distillation: closed-form rules fitted to a planner's trajectories (pure).

Every player-round of a trajectory is one row: the planner's offer share s_i = e_i / R,
the pool fraction R / 200 and the previous round's reciprocations.

**w(R) family** (the paper's Interpolating form, with a free weight and a kept share):

    s_i = (1 - kappa(R)) [w(R) / 4 + (1 - w(R)) q_i],   q_i = c_{i,t-1} / sum_j c_{j,t-1}

(q_i = 1/4 in round 1 or when nobody returned anything), with w and kappa piecewise
constant on `DISTILL_W_BINS` equal-width bins of R / 200. kappa(bin) is the
coin-weighted kept share of the bin, sum (R - sum e) / sum R over its rounds with
R >= 1 (protocol §6.3; a mean of ratios would read the 0.001 withheld at R = 0.01 as
10 %); w(bin) is the least-squares weight of the normalised shares
p_i = s_i / (1 - kappa_round) on the rows where q is informative:
w = sum (p - q)(1/4 - q) / sum (1/4 - q)^2, clipped to [0, 1]. Empty bins take the
nearest non-empty bin's value. The one-parameter special case w = (R / 200)^k is fitted
by grid search over the paper's ln k grid (with the same binned kappa).

**Decision tree**: a CART regression tree (depth `DISTILL_TREE_DEPTH`, at least
`DISTILL_TREE_MIN_LEAF` rows per leaf, exact squared-error splits) of s_i on
(R / 200, own previous fraction c/e, the others' previous fraction, own previous share
e_{i,t-1} / R_{t-1}), rounds 2..40; round 1 gets the mean round-1 share.

Both rules can be played (`w_rule_step`, `tree_rule_step`) to measure the surplus a
distilled rule earns next to the planner on the same games. The distilled k is played
through the deployed rule (`citizens.weight_rule_step(citizens.interpolating(k))`:
w = round((R / 200)^k, 1), float32 offers, sum e = R - 0.001), like k* and k = 22.

**Recovery test** (`recovery`): the distiller run on trajectories of a deployed,
quantised Interpolating rule of known k; the recovered k and the per-bin error of the
fitted w against the true quantised weight (its least-squares average in the bin) are
the method's bias.

A fit statistic that is undefined (R^2 when the target has no variance) is null, with
the reason, never NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phasespace.citizens.game import EXCLUSION_THRESHOLD, MAX_POOL, N_PLAYERS
from phasespace.citizens.mechanisms import interpolating_weight
from phasespace.citizens.rl import c4_config, twin
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import config as c3config

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

EPS: Final = 1e-12


@dataclass(frozen=True)
class Rows:
    """Flattened player-rounds (G * T * 4) of a trajectory."""

    pool_frac: FloatArray
    pool: FloatArray  # R in coins
    share: FloatArray
    q: FloatArray
    informative: BoolArray  # t > 0 and sum c_{t-1} > 0
    kept: FloatArray  # 1 - sum e / R of the row's round
    later: BoolArray  # t > 0
    features: FloatArray  # (N, 4) tree inputs (`c4_config.DISTILL_FEATURES`)


def rows_from_arrays(offers: FloatArray, actions: FloatArray, pool: FloatArray) -> Rows:
    """Rows of (G, T, 4) offers and actions and (G, T) pools."""
    g, t_max, n = offers.shape
    share = offers / pool[..., None]
    prev_c = np.concatenate([np.zeros((g, 1, n)), actions[:, :-1]], axis=1)
    prev_e = np.concatenate([np.zeros((g, 1, n)), offers[:, :-1]], axis=1)
    prev_pool = np.concatenate([pool[:, :1], pool[:, :-1]], axis=1)
    later = np.broadcast_to((np.arange(t_max) > 0)[None, :, None], offers.shape)
    total = prev_c.sum(axis=-1, keepdims=True)
    informative = later & (total > 0.0)
    q = np.where(informative, prev_c / np.where(total > 0.0, total, 1.0), 0.25)
    own_ok = prev_e >= EXCLUSION_THRESHOLD
    own_frac = np.where(own_ok, prev_c / np.where(own_ok, prev_e, 1.0), 0.0)
    others_e = prev_e.sum(axis=-1, keepdims=True) - prev_e
    others_c = total - prev_c
    others_ok = others_e >= EXCLUSION_THRESHOLD
    others_frac = np.where(others_ok, others_c / np.where(others_ok, others_e, 1.0), 0.0)
    own_share = np.where(later, prev_e / prev_pool[..., None], 0.0)
    pool_frac = np.broadcast_to((pool / MAX_POOL)[..., None], offers.shape)
    kept = np.broadcast_to((1.0 - offers.sum(axis=-1) / pool)[..., None], offers.shape)
    features = np.stack([pool_frac, own_frac, others_frac, own_share], axis=-1)
    return Rows(
        pool_frac=pool_frac.reshape(-1),
        pool=np.broadcast_to(pool[..., None], offers.shape).reshape(-1),
        share=share.reshape(-1),
        q=q.reshape(-1),
        informative=informative.reshape(-1),
        kept=kept.reshape(-1),
        later=later.reshape(-1),
        features=features.reshape(-1, len(c4_config.DISTILL_FEATURES)),
    )


def trajectory_rows(traj: rc.Trajectory) -> Rows:
    return rows_from_arrays(
        np.asarray(traj.offers), np.asarray(traj.actions), np.asarray(traj.pool)
    )


R2_NULL_REASON: Final = "R^2 undefined: the target has no variance"


def r_squared(y: FloatArray, prediction: FloatArray) -> float | None:
    """1 - SSE / SST; None (undefined) when SST = 0 or there are no rows."""
    if y.size == 0:
        return None
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - prediction) ** 2).sum()) / sst if sst > 0.0 else None


def r2_record(value: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {"r2": value}
    if value is None:
        out["r2_null_reason"] = R2_NULL_REASON
    return out


# --- w(R) family -----------------------------------------------------------------------------


def bin_index(pool_frac: FloatArray, n_bins: int) -> IntArray:
    return np.clip(np.floor(pool_frac * n_bins), 0, n_bins - 1).astype(np.int64)


def _fill_nearest(values: FloatArray, filled: BoolArray, default: float) -> FloatArray:
    """Empty bins take the nearest filled bin's value (the lower one on ties)."""
    if not filled.any():
        return np.full_like(values, default)
    idx = np.flatnonzero(filled)
    out = values.copy()
    for b in np.flatnonzero(~filled):
        out[b] = values[idx[np.argmin(np.abs(idx - b))]]
    return out


@dataclass(frozen=True)
class WRule:
    w: FloatArray  # (bins,)
    kept: FloatArray  # (bins,)
    w_rows: IntArray  # informative rows per bin
    rows: IntArray  # rows per bin

    def predict(self, rows: Rows) -> FloatArray:
        b = bin_index(rows.pool_frac, self.w.size)
        w, kappa = self.w[b], self.kept[b]
        return (1.0 - kappa) * (w / N_PLAYERS + (1.0 - w) * rows.q)

    def to_json(self) -> dict[str, Any]:
        return {
            "bins": int(self.w.size),
            "w": self.w.tolist(),
            "kept_share": self.kept.tolist(),
            "informative_rows": self.w_rows.tolist(),
            "rows": self.rows.tolist(),
        }


KEPT_MIN_POOL: Final = 1.0
"""Rows with R >= 1 coin enter the coin-weighted kept share (protocol §6.3)."""


def coin_weighted_kept(rows: Rows, n_bins: int) -> tuple[FloatArray, BoolArray]:
    """(bins,) sum (R - sum e) / sum R over each bin's rows with R >= 1, and which bins
    have such rows (the player rows of a round repeat its R and kept coins, so the ratio
    is the rounds' coin-weighted kept share)."""
    b = bin_index(rows.pool_frac, n_bins)
    use = rows.pool >= KEPT_MIN_POOL
    kept_coins = np.bincount(b[use], weights=(rows.kept * rows.pool)[use], minlength=n_bins)
    pool_coins = np.bincount(b[use], weights=rows.pool[use], minlength=n_bins)
    has = pool_coins > 0.0
    return np.where(has, kept_coins / np.where(has, pool_coins, 1.0), 0.0), has


def fit_w_rule(rows: Rows, n_bins: int = c4_config.DISTILL_W_BINS) -> WRule:
    b = bin_index(rows.pool_frac, n_bins)
    counts = np.bincount(b, minlength=n_bins)
    kept_raw, has = coin_weighted_kept(rows, n_bins)
    kept = _fill_nearest(kept_raw, has, 0.0)
    distributed = 1.0 - rows.kept
    use = rows.informative & (distributed > EPS)
    p = np.where(use, rows.share / np.where(use, distributed, 1.0), 0.0)
    a = 0.25 - rows.q
    num = np.bincount(b[use], weights=((p - rows.q) * a)[use], minlength=n_bins)
    den = np.bincount(b[use], weights=(a * a)[use], minlength=n_bins)
    w_rows = np.bincount(b[use], minlength=n_bins)
    ok = den > EPS
    w_raw = np.clip(np.where(ok, num / np.where(ok, den, 1.0), 1.0), 0.0, 1.0)
    w = _fill_nearest(w_raw, ok, 1.0)
    return WRule(w=w, kept=kept, w_rows=w_rows.astype(np.int64), rows=counts.astype(np.int64))


def fit_interpolating_k(rows: Rows, kept_bins: FloatArray) -> dict[str, Any]:
    """Grid search of w = (R / 200)^k over the paper's ln k grid, with the binned kept
    share; the smallest ln k among the minimal squared errors."""
    b = bin_index(rows.pool_frac, kept_bins.size)
    kappa = kept_bins[b]
    best: tuple[float, float] | None = None
    for ln_k in c3config.ln_k_grid():
        w = rows.pool_frac ** math.exp(ln_k)
        pred = (1.0 - kappa) * (w / N_PLAYERS + (1.0 - w) * rows.q)
        sse = float(((rows.share - pred) ** 2).sum())
        if best is None or sse < best[1]:
            best = (ln_k, sse)
    assert best is not None
    ln_k = best[0]
    w = rows.pool_frac ** math.exp(ln_k)
    pred = (1.0 - kappa) * (w / N_PLAYERS + (1.0 - w) * rows.q)
    return {"ln_k": ln_k, "k": math.exp(ln_k), **r2_record(r_squared(rows.share, pred))}


# --- decision tree ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tree:
    """A binary regression tree in arrays; leaves have feature -1."""

    feature: IntArray
    threshold: FloatArray
    left: IntArray
    right: IntArray
    value: FloatArray
    depth: int

    def predict(self, x: FloatArray) -> FloatArray:
        node = np.zeros(x.shape[0], dtype=np.int64)
        for _ in range(self.depth):
            f = self.feature[node]
            go_left = x[np.arange(x.shape[0]), np.maximum(f, 0)] <= self.threshold[node]
            child = np.where(go_left, self.left[node], self.right[node])
            node = np.where(f < 0, node, child)
        return self.value[node]

    def to_json(self, names: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            {
                "node": i,
                "feature": None if self.feature[i] < 0 else names[int(self.feature[i])],
                "threshold": None if self.feature[i] < 0 else float(self.threshold[i]),
                "left": None if self.feature[i] < 0 else int(self.left[i]),
                "right": None if self.feature[i] < 0 else int(self.right[i]),
                "value": float(self.value[i]),
            }
            for i in range(self.feature.size)
        ]


def best_split(xf: FloatArray, y: FloatArray, min_leaf: int) -> tuple[float, float] | None:
    """(SSE, threshold) of the best split x <= threshold of one feature (exact CART: every
    midpoint between consecutive distinct values with >= `min_leaf` rows on each side)."""
    order = np.argsort(xf, kind="stable")
    xs, ys = xf[order], y[order]
    n = ys.size
    csum, csq = np.cumsum(ys), float((ys * ys).sum())
    n_left = np.arange(1, n)  # split after position i - 1: left = first i rows
    valid = (xs[1:] > xs[:-1]) & (n_left >= min_leaf) & (n - n_left >= min_leaf)
    if not valid.any():
        return None
    s_left = csum[:-1]
    s_right = csum[-1] - s_left
    sse = csq - s_left**2 / n_left - s_right**2 / (n - n_left)
    sse = np.where(valid, sse, np.inf)
    i = int(np.argmin(sse))
    return float(sse[i]), float(0.5 * (xs[i] + xs[i + 1]))


def fit_tree(
    x: FloatArray,
    y: FloatArray,
    depth: int = c4_config.DISTILL_TREE_DEPTH,
    min_leaf: int = c4_config.DISTILL_TREE_MIN_LEAF,
) -> Tree:
    """Greedy CART regression tree (squared error, exact thresholds; `best_split`)."""
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []

    def new_node(idx: IntArray) -> int:
        feature.append(-1)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(float(y[idx].mean()) if idx.size else 0.0)
        return len(feature) - 1

    def grow(idx: IntArray, d: int) -> int:
        node = new_node(idx)
        if d >= depth or idx.size < 2 * min_leaf:
            return node
        yi = y[idx]
        total, total_sq = float(yi.sum()), float((yi * yi).sum())
        base = total_sq - total * total / idx.size
        best: tuple[float, int, float] | None = None
        for f in range(x.shape[1]):
            split = best_split(x[idx, f], yi, min_leaf)
            if split is not None and split[0] < base - EPS and (best is None or split[0] < best[0]):
                best = (split[0], f, split[1])
        if best is None:
            return node
        _, f, thr = best
        mask = x[idx, f] <= thr
        feature[node], threshold[node] = f, thr
        left[node] = grow(idx[mask], d + 1)
        right[node] = grow(idx[~mask], d + 1)
        return node

    grow(np.arange(y.size, dtype=np.int64), 0)
    return Tree(
        feature=np.asarray(feature, dtype=np.int64),
        threshold=np.asarray(threshold, dtype=np.float64),
        left=np.asarray(left, dtype=np.int64),
        right=np.asarray(right, dtype=np.int64),
        value=np.asarray(value, dtype=np.float64),
        depth=depth,
    )


# --- distillation of one trajectory ----------------------------------------------------------------


@dataclass(frozen=True)
class Distilled:
    w_rule: WRule
    w_rule_r2: float | None
    interpolating: dict[str, Any]
    tree: Tree
    first_share: float
    tree_r2: float | None
    tree_r2_later: float | None
    rows: int

    def to_json(self) -> dict[str, Any]:
        later = r2_record(self.tree_r2_later)
        return {
            "rows": self.rows,
            "w_rule": {
                **self.w_rule.to_json(),
                "kept_share_definition": "coin-weighted: sum (R - sum e) / sum R over the "
                "bin's rounds with R >= 1",
                **r2_record(self.w_rule_r2),
            },
            "interpolating_k": self.interpolating,
            "tree": {
                "features": list(c4_config.DISTILL_FEATURES),
                "depth": self.tree.depth,
                "min_leaf": c4_config.DISTILL_TREE_MIN_LEAF,
                "first_round_share": self.first_share,
                "nodes": self.tree.to_json(c4_config.DISTILL_FEATURES),
                **r2_record(self.tree_r2),
                "r2_rounds_2_to_40": later["r2"],
                **(
                    {"r2_rounds_2_to_40_null_reason": R2_NULL_REASON} if later["r2"] is None else {}
                ),
            },
        }


def distill(rows: Rows) -> Distilled:
    w_rule = fit_w_rule(rows)
    interp = fit_interpolating_k(rows, w_rule.kept)
    later = rows.later
    tree = fit_tree(rows.features[later], rows.share[later])
    first = float(rows.share[~later].mean()) if (~later).any() else 0.25
    tree_pred = np.where(later, tree.predict(rows.features), first)
    return Distilled(
        w_rule=w_rule,
        w_rule_r2=r_squared(rows.share, w_rule.predict(rows)),
        interpolating=interp,
        tree=tree,
        first_share=first,
        tree_r2=r_squared(rows.share, tree_pred),
        tree_r2_later=r_squared(rows.share[later], tree.predict(rows.features[later])),
        rows=int(rows.share.size),
    )


# --- recovery test (method bias) ---------------------------------------------------------------------


def true_binned_w(rows: Rows, k: float, n_bins: int = c4_config.DISTILL_W_BINS) -> FloatArray:
    """(bins,) the least-squares average, over each bin's informative rows, of the true
    deployed weight round((R / 200)^k, 1) (`mechanisms.interpolating_weight`), with the
    same row weights (1/4 - q)^2 as the w fit; NaN for a bin without informative rows."""
    b = bin_index(rows.pool_frac, n_bins)
    use = rows.informative
    a2 = (0.25 - rows.q) ** 2
    w_true = interpolating_weight(rows.pool, k)
    num = np.bincount(b[use], weights=(w_true * a2)[use], minlength=n_bins)
    den = np.bincount(b[use], weights=a2[use], minlength=n_bins)
    return np.where(den > EPS, num / np.where(den > EPS, den, 1.0), np.nan)


def recovery(rows: Rows, k_true: float, n_bins: int = c4_config.DISTILL_W_BINS) -> dict[str, Any]:
    """The distiller (w(R) bins and the one-parameter k) on trajectories of the deployed,
    quantised Interpolating rule with exponent `k_true`: the recovered k and ln k, the
    ln k bias, and per bin the fitted w against the true quantised weight."""
    rule = fit_w_rule(rows, n_bins)
    fitted_k = fit_interpolating_k(rows, rule.kept)
    true_w = true_binned_w(rows, k_true, n_bins)
    ok = np.isfinite(true_w) & (rule.w_rows > 0)
    error = np.where(ok, rule.w - np.where(ok, true_w, 0.0), np.nan)
    abs_error = np.abs(error[ok])
    return {
        "k_true": k_true,
        "ln_k_true": math.log(k_true),
        "k_recovered": fitted_k["k"],
        "ln_k_recovered": fitted_k["ln_k"],
        "ln_k_bias": fitted_k["ln_k"] - math.log(k_true),
        **r2_record(fitted_k["r2"]),
        "w_fitted": rule.w.tolist(),
        "w_true_quantised": [float(x) if np.isfinite(x) else None for x in true_w],
        "w_error": [float(x) if np.isfinite(x) else None for x in error],
        "w_abs_error_max": float(abs_error.max()) if abs_error.size else None,
        "w_abs_error_mean": float(abs_error.mean()) if abs_error.size else None,
        "informative_rows": rule.w_rows.tolist(),
        "note": "deployed rule: w = round((R/200)^k, 1), float32 offers, sum e = R - 0.001; "
        "w_true_quantised is the least-squares average of the true weight in the bin (null "
        "without informative rows)",
    }


# --- distilled rules as mechanisms -------------------------------------------------------------------


def _bin(pool: Array, n_bins: int) -> Array:
    return jnp.clip(jnp.floor(pool / MAX_POOL * n_bins), 0, n_bins - 1).astype(jnp.int32)


def _proportional_target(obs: rc.MechObs) -> Array:
    total = obs.prev_reciprocations.sum(axis=-1, keepdims=True)
    informative = (obs.t > 0) & (total > 0.0)
    return jnp.where(
        informative, obs.prev_reciprocations / jnp.where(total > 0.0, total, 1.0), 0.25
    )


def w_rule_step(rule: WRule) -> rc.MechStep:
    """The binned w(R) rule (idealised offers, no rounding): (G, 4) offers."""
    w_table = twin.f64(rule.w)
    kept_table = twin.f64(rule.kept)

    def step(state: Any, obs: rc.MechObs, key: Array) -> tuple[Array, Any, Any]:
        del key
        b = _bin(obs.pool, rule.w.size)
        w, kappa = w_table[b][:, None], kept_table[b][:, None]
        q = _proportional_target(obs)
        offers = (obs.pool * (1.0 - kappa[:, 0]))[:, None] * (w / N_PLAYERS + (1.0 - w) * q)
        return offers, state, None

    return step


def interpolating_step(k: float) -> rc.MechStep:
    """Paper-form rule with the continuous weight w = (R / 200)^k and no kept share
    (idealised; the C4 comparisons play a distilled k through the deployed rule)."""

    def step(state: Any, obs: rc.MechObs, key: Array) -> tuple[Array, Any, Any]:
        del key
        w = (obs.pool / MAX_POOL) ** k
        return twin.paper_offers(obs.pool, obs.prev_reciprocations, w, obs.t == 0), state, None

    return step


def tree_features(obs: rc.MechObs, prev_pool: Array) -> Array:
    """(G, 4, 4) tree inputs of every player from the observation and the previous pool."""
    prev_e, prev_c = obs.prev_offers, obs.prev_reciprocations
    own_ok = prev_e >= EXCLUSION_THRESHOLD
    own_frac = jnp.where(own_ok, prev_c / jnp.where(own_ok, prev_e, 1.0), 0.0)
    others_e = prev_e.sum(axis=-1, keepdims=True) - prev_e
    others_c = prev_c.sum(axis=-1, keepdims=True) - prev_c
    others_ok = others_e >= EXCLUSION_THRESHOLD
    others_frac = jnp.where(others_ok, others_c / jnp.where(others_ok, others_e, 1.0), 0.0)
    own_share = prev_e / prev_pool[:, None]
    pool_frac = jnp.broadcast_to((obs.pool / MAX_POOL)[:, None], prev_e.shape)
    return jnp.stack([pool_frac, own_frac, others_frac, own_share], axis=-1)


def tree_initial_state(n_games: int) -> Array:
    """The tree rule's state, the previous round's pool (unused in round 1)."""
    return twin.full((n_games,), MAX_POOL)


def tree_rule_step(tree: Tree, first_share: float) -> rc.MechStep:
    """The tree rule (state: the previous pool, `tree_initial_state`). Predicted shares
    are clipped at 0 and scaled down to sum 1 when they exceed it."""
    feature = twin.asarray(tree.feature)
    threshold = twin.f64(tree.threshold)
    left, right = twin.asarray(tree.left), twin.asarray(tree.right)
    value = twin.f64(tree.value)

    def predict(x: Array) -> Array:
        node = twin.zeros(x.shape[:-1], dtype=feature.dtype)
        for _ in range(tree.depth):
            f = feature[node]
            xf = jnp.take_along_axis(x, jnp.maximum(f, 0)[..., None], axis=-1)[..., 0]
            child = jnp.where(xf <= threshold[node], left[node], right[node])
            node = jnp.where(f < 0, node, child)
        return value[node]

    def step(prev_pool: Array, obs: rc.MechObs, key: Array) -> tuple[Array, Any, Any]:
        del key
        s = jnp.maximum(predict(tree_features(obs, prev_pool)), 0.0)
        s = jnp.where(obs.t == 0, first_share, s)
        total = s.sum(axis=-1, keepdims=True)
        s = jnp.where(total > 1.0, s / jnp.where(total > 1.0, total, 1.0), s)
        return obs.pool[:, None] * s, obs.pool, None

    return step


def total_surplus(
    step: rc.MechStep, state: Any, port: rc.JaxCitizen, key: Array, n_games: int
) -> FloatArray:
    """(G,) total surplus of a mechanism on the games of `key`."""
    from phasespace.citizens.rl.checks import trajectory_metrics

    traj, _ = rc.rollout(port, step, state, key, jax.random.key(0), n_games)
    return trajectory_metrics(traj)["total_surplus"]
