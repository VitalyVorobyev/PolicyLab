"""C4 preparation: configuration, pilots' curve summary, the no-retention planner, cross-play,
extrapolation diagnostics, distillation and the planner-versus-B4 path.

Toy and analytic citizens only: no fitted model, no human data, no LLM call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phasespace.citizens import game
from phasespace.citizens.llm import prompts, runner
from phasespace.citizens.llm.backend import CallResult, ClaudeCLI
from phasespace.citizens.mechanisms import RECORDED_WITHHELD, baseline_offers
from phasespace.citizens.models import b1_gru
from phasespace.citizens.models.base import K_MAX, streams_from_arrays
from phasespace.citizens.rl import (
    b4play,
    c4,
    c4_config,
    checks,
    crossplay,
    distill,
    ppo,
    stage,
    twin,
)
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import config as c3config
from phasespace.citizens.rl.planner import (
    Planner,
    deserialise,
    deterministic_step,
    player_features,
    sampling_step,
    serialise,
)

ROUNDS = c3config.ROUNDS


# --- toy citizens ----------------------------------------------------------------------------------


class UniformReturn(eqx.Module):
    """Returns an integer drawn uniformly from 0..floor(e) (0 when e < 1)."""

    name: str = eqx.field(static=True, default="uniform-return")

    def init(self, key: Array, n_streams: int) -> None:
        del key, n_streams

    def step(self, state: None, obs: rc.Obs) -> tuple[Array, None]:
        n = twin.max_reciprocation(obs.offers[:, 0])
        c = twin.arange(K_MAX, dtype=jnp.float64)[None, :]
        mass = (c <= n[:, None]).astype(jnp.float64)
        return mass / mass.sum(axis=1, keepdims=True), state


class HalfOfFloor(rc.DeterministicCitizen):
    """Returns floor(floor(e) / 2): what `FakeBackend(0.5)` answers to a B4 prompt."""

    name: str = eqx.field(static=True, default="half-of-floor")

    def react(self, own_offer: Array) -> Array:
        return jnp.floor(twin.max_reciprocation(own_offer) * 0.5)


@pytest.fixture(autouse=True)
def x64_mode() -> Iterator[None]:
    with twin.x64():
        yield


def _planner(seed: int, retention: bool = True) -> Planner:
    return Planner(8, -0.5, jax.random.key(seed), retention=retention)


# --- configuration ------------------------------------------------------------------------------------


def test_c4_keys_differ_from_every_training_monitoring_and_tuning_seed() -> None:
    c3_keys = {
        c3config.EVAL_KEY_SEED,
        c3config.TUNE_KEY_SEED,
        c3config.MONITOR_KEY_SEED,
        c3config.SAMPLED_EVAL_KEY_SEED,
        *(c3config.CHECK_SIM_SEED + i for i in range(len(c3config.CHECK_MECHANISMS))),
        *c3config.PILOT_TRAIN_SEEDS,
    }
    c4_keys = [
        c4_config.EVAL_KEY_SEED,
        c4_config.SAMPLED_NOISE_KEY_SEED,
        c4_config.DISTILL_KEY_SEED,
        c4_config.PILOT_SAMPLED_NOISE_KEY_SEED,
    ]
    assert len(set(c4_keys)) == len(c4_keys)
    assert not set(c4_keys) & c3_keys
    assert not set(c4_keys) & set(c4_config.PILOT_SEEDS.values())
    assert set(c4_config.PILOT_SEEDS.values()) == {901, 902, 903}
    assert c4.reserved_seeds() >= set(c4_keys) | {901, 902, 903}
    assert len(c4_config.config_sha256()) == 64


def test_plateau_of_a_step_curve() -> None:
    """Analytic: 0 before iteration 300, 100 from it. F = 100, I = 0, threshold 98; the
    5-point trailing mean first reaches 98 when the five points 300..340 are all 100."""
    its = list(range(10, 1010, 10))
    values = [0.0 if i < 300 else 100.0 for i in its]
    out = c4.plateau_iteration(its, values)
    assert out["plateau_iteration"] == 340
    assert out["final_level"] == 100.0 and out["threshold"] == pytest.approx(98.0)
    assert out["tail_slope_per_100_iterations"] == pytest.approx(0.0, abs=1e-9)


def test_plateau_slope_of_a_linear_curve() -> None:
    its = list(range(10, 510, 10))
    out = c4.plateau_iteration(its, [2.0 * i for i in its])
    assert out["tail_slope_per_100_iterations"] == pytest.approx(200.0)
    assert out["plateau_iteration"] > 400  # a rising curve has no early plateau


# --- the no-retention ablation ------------------------------------------------------------------------


def test_no_retention_planner_offers_sum_to_the_deployed_total() -> None:
    planner = _planner(3, retention=False)
    assert planner.log_std.shape == (4,)
    g = 6
    obs = rc.MechObs(
        pool=twin.f64(np.linspace(0.01, 200.0, g)),
        prev_offers=twin.f64(np.random.default_rng(1).uniform(0, 50, (g, 4))),
        prev_reciprocations=twin.f64(np.random.default_rng(2).integers(0, 10, (g, 4))),
        t=twin.asarray(3),
    )
    state = planner.initial_state(g)
    e_det, _, s_det = deterministic_step(planner)(state, obs, jax.random.key(0))
    e_smp, _, _ = sampling_step(planner)(state, obs, jax.random.key(1))
    expected = np.asarray(obs.pool) - RECORDED_WITHHELD
    assert np.allclose(np.asarray(e_det).sum(axis=1), expected, rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(e_smp).sum(axis=1), expected, rtol=0, atol=1e-12)
    # independent reference: (R - 0.001) * softmax(mu[:4]) computed in NumPy
    _, mu = planner.policy_step(state[0], player_features(obs))
    m = np.asarray(mu)
    soft = np.exp(m - m.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    assert np.allclose(np.asarray(e_det), expected[:, None] * soft, rtol=1e-12, atol=1e-12)
    kept = np.asarray(s_det)[:, 4]
    assert np.allclose(kept, RECORDED_WITHHELD / np.asarray(obs.pool), rtol=1e-12)
    assert np.allclose(np.asarray(s_det).sum(axis=1), 1.0, atol=1e-12)


def test_default_planner_is_unchanged_by_the_ablation_flag() -> None:
    a = _planner(7)
    b = Planner(8, -0.5, jax.random.key(7))
    leaves_a = jax.tree_util.tree_leaves(ppo.array_part(a))
    leaves_b = jax.tree_util.tree_leaves(ppo.array_part(b))
    assert a.retention and a.log_std.shape == (5,)
    assert all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(leaves_a, leaves_b, strict=True)
    )
    noret = _planner(7, retention=False)
    shared = jax.tree_util.tree_leaves(noret.policy)
    assert all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(shared, jax.tree_util.tree_leaves(a.policy), strict=True)
    )


def test_no_retention_planner_round_trips_and_trains() -> None:
    cfg = ppo.PPOConfig(n_games=16, minibatches=2, hidden=8, retention=False)
    planner, curve = ppo.train(rc.FixedFraction(0.5), cfg, seed=4, n_iterations=2, chunk=1)
    assert not planner.retention and len(curve) == 2
    again = deserialise(serialise(planner), 8, retention=False)
    leaves = zip(
        jax.tree_util.tree_leaves(ppo.array_part(planner)),
        jax.tree_util.tree_leaves(ppo.array_part(again)),
        strict=True,
    )
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in leaves)
    assert cfg.to_json()["retention"] is False and ppo.PPOConfig().retention is True


# --- poolability and grouping ----------------------------------------------------------------------------


def _train_manifest(run_id: str, model: str, seed: int, retention: bool = True) -> dict[str, Any]:
    ppo_record = {**ppo.PPOConfig().to_json(), "retention": retention}
    return {
        "run_id": run_id,
        "source": {"git_dirty": False, "git_commit": "c" * 40},
        "citizens": {
            "stage": "c4_train",
            "fit_run": {"manifest_sha256": "f" * 64},
            "seeds": {"eval_key": c3config.EVAL_KEY_SEED},
            "training": {
                "model": model,
                "seed": seed,
                "ppo": ppo_record,
                "overrides": {} if retention else {"retention": False},
                "iterations": 500,
            },
        },
    }


def _base_manifest() -> dict[str, Any]:
    return {
        "run_id": "base",
        "source": {"git_dirty": False},
        "citizens": {
            "fit_run": {"manifest_sha256": "f" * 64},
            "seeds": {"eval_key": c3config.EVAL_KEY_SEED},
            "k_star": {m: {"k": 9.0, "ln_k": 2.2} for m in rc.TRAINABLE_MODELS},
        },
    }


def test_training_runs_are_grouped_by_model_and_action_head() -> None:
    trains = [
        _train_manifest("b2-1", "B2", 11),
        _train_manifest("b1-1", "B1", 11),
        _train_manifest("b1-2", "B1", 12),
        _train_manifest("b1-n", "B1", 11, retention=False),
    ]
    groups = c4.group_training_runs(_base_manifest(), trains)
    assert list(groups) == [("B1", True), ("B1", False), ("B2", True)]
    assert groups[("B1", True)] == [1, 2]


def _set_stage(m: dict[str, Any]) -> None:
    m["citizens"]["stage"] = "c4_pilot"


def _set_seed(m: dict[str, Any]) -> None:
    m["citizens"]["training"]["seed"] = 901


def _set_dirty(m: dict[str, Any]) -> None:
    m["source"]["git_dirty"] = True


CHANGES: list[tuple[Callable[[dict[str, Any]], None], str]] = [
    (_set_stage, "never C4 inputs"),
    (_set_seed, "reserved"),
    (_set_dirty, "dirty"),
]


@pytest.mark.parametrize(("change", "message"), CHANGES)
def test_crossplay_refuses_pilots_reserved_seeds_and_dirty_runs(
    change: Callable[[dict[str, Any]], None], message: str
) -> None:
    bad = _train_manifest("b1-2", "B1", 12)
    change(bad)
    with pytest.raises(stage.RunMismatchError, match=message):
        c4.group_training_runs(_base_manifest(), [_train_manifest("b1-1", "B1", 11), bad])


def test_check_poolable_any_model() -> None:
    trains = [_train_manifest("b3-1", "B3", 1), _train_manifest("b3-2", "B3", 2)]
    stage.check_poolable(_base_manifest(), trains, model=None)
    with pytest.raises(stage.RunMismatchError, match="not B1"):
        stage.check_poolable(_base_manifest(), trains)


def test_planners_are_loaded_from_run_directories(tmp_path: Path) -> None:
    from phasespace.gravity.export import write_run_directory

    dirs: list[Path] = []
    docs: list[dict[str, Any]] = []
    for seed, retention in ((21, True), (22, True), (21, False)):
        run_id = f"t-{seed}-{int(retention)}"
        manifest = _train_manifest(run_id, "B2", seed, retention)
        manifest["citizens"]["training"]["ppo"]["hidden"] = 8
        planner = _planner(seed, retention)
        written = write_run_directory(
            tmp_path / run_id,
            manifest,
            [("model_parameters", stage.PLANNER_FILE, serialise(planner))],
        )
        dirs.append(tmp_path / run_id)
        docs.append(written)
    policies = c4.load_policies(_base_manifest(), dirs, docs)
    names = [p.name for p in policies]
    assert names[:2] == ["Planner[B2]", "Planner-noret[B2]"]
    assert policies[0].seeds == (21, 22) and policies[1].seeds == (21,)
    assert names[2:] == [
        "Equal",
        "Mixed",
        "Proportional",
        "Interpolating k=22",
        *(f"Interpolating k*[{m}]" for m in rc.TRAINABLE_MODELS),
    ]
    reloaded = policies[1].planners[0]
    assert not reloaded.retention


# --- cross-play on toy citizens -----------------------------------------------------------------------------


def _numpy_equal_rule_total(fraction: float, rounds: int = ROUNDS) -> float:
    """Independent reference: deployed Equal (NumPy `baseline_offers`) against c = f e."""
    pool, prev = game.MAX_POOL, None
    total = 0.0
    for _ in range(rounds):
        e = np.asarray(baseline_offers(pool, prev, 1.0), dtype=np.float64)
        c = fraction * e
        total += float((e - c).sum())
        pool = max(
            game.POOL_FLOOR, min(game.MAX_POOL, pool - e.sum() + game.GROWTH_FACTOR * c.sum())
        )
        prev = c
    return total


def test_crossplay_matrix_on_toy_citizens() -> None:
    policies = [
        crossplay.planner_policy("T1", True, [1, 2], [_planner(1), _planner(2)]),
        crossplay.planner_policy("T1", False, [1], [_planner(1, retention=False)]),
        *crossplay.rule_policies({"T1": 3.0}),
    ]
    columns = {"F": rc.FixedFraction(0.5), "U": UniformReturn()}
    key, noise = jax.random.key(5), jax.random.key(6)
    series: list[crossplay.Series] = []
    for name, port in columns.items():
        series += crossplay.evaluate_column(
            policies, name, port, key, noise, 8, crossplay.Diagnostics()
        )
    # 2 columns x (2 + 1 planner seeds x 2 modes + 5 rules)
    assert len(series) == 2 * (3 * 2 + 5)
    equal = crossplay.stack_totals(series, "Equal", "F", "deterministic")
    assert np.allclose(equal, _numpy_equal_rule_total(0.5), rtol=0, atol=1e-6)
    # common random numbers: a planner's deterministic series equals an independent rollout
    traj, _ = ppo.evaluate_planner(_planner(2), columns["U"], key, 8, ROUNDS)
    direct = checks.trajectory_metrics(traj)["total_surplus"]
    got = crossplay.stack_totals(series, "Planner[T1]", "U", "deterministic")[1]
    assert np.array_equal(got, direct)
    noret = next(s for s in series if s.policy == "Planner-noret[T1]" and s.column == "U")
    pools = np.asarray(
        ppo.evaluate_planner(_planner(1, False), columns["U"], key, 8, ROUNDS)[0].pool
    )
    assert np.allclose(noret.kept_share, (RECORDED_WITHHELD / pools).mean(axis=0), rtol=1e-9)
    summary = crossplay.summarise(policies, list(columns), series)
    cell = summary["cells"]["Planner[T1]"]["U"]["deterministic"]
    assert cell["seeds"] == [1, 2] and len(cell["kept_share_per_round"]) == ROUNDS
    assert summary["cells"]["Equal"]["F"]["sampled"] is None
    diffs = crossplay.paired_differences(policies, list(columns), series, n_resamples=50, seed=1)
    self_vs_rule = [
        d
        for d in diffs
        if d["policy"] == "Planner[T1]" and d["versus"] == "Equal" and d["column"] == "F"
    ]
    assert len(self_vs_rule) == 2  # deterministic and sampled
    for d in self_vs_rule:
        a = crossplay.stack_totals(series, "Planner[T1]", "F", d["mode"])
        assert d["difference_mean"] == pytest.approx(float(a.mean() - equal.mean()))
        assert len(d["per_seed_difference"]) == 2


def test_group_bootstrap_of_a_constant_shift_is_exact() -> None:
    """Identical seeds in both groups, shifted by 5: every resample's difference is 5."""
    row = np.random.default_rng(3).normal(1000.0, 30.0, (1, 50))
    same = np.repeat(row, 3, axis=0)
    mean, lo, hi = crossplay.group_bootstrap(same + 5.0, same[:2], 200, 4)
    assert mean == pytest.approx(5.0) and lo == pytest.approx(5.0) and hi == pytest.approx(5.0)


def test_group_bootstrap_covers_a_known_difference() -> None:
    rng = np.random.default_rng(5)
    games = rng.normal(0.0, 5.0, 300)
    a = games[None, :] + 2.0 + rng.normal(0.0, 1.0, (4, 300))
    b = games[None, :] + rng.normal(0.0, 1.0, (3, 300))
    _, lo, hi = crossplay.group_bootstrap(a, b, 1000, 6)
    assert lo < 2.0 < hi and hi - lo < 1.0


# --- extrapolation diagnostics -------------------------------------------------------------------------------


def _games(offers: np.ndarray, pool: float = 100.0) -> Any:
    g, t = offers.shape[:2]
    actions = np.floor(offers * 0.5)
    return streams_from_arrays(offers, actions, np.full((g, t), pool), np.full(g, t))


def test_training_support_and_out_of_range_fractions() -> None:
    rng = np.random.default_rng(0)
    train = _games(rng.uniform(1.0, 10.0, (20, 5, 4)))
    support = crossplay.training_support(train, "B3")
    assert support.lo["own_offer"] >= 1.0 and support.hi["own_offer"] <= 10.0
    # the training data are inside their own support
    inside = crossplay.ood_fractions(support, train)
    assert inside["outside_range_any"] == 0.0 and inside["off_grid"] == 0.0
    # one decision of 2 games x 5 rounds x 4 players = 40 at e = 50, in the last round
    # (so no later round sees it as the previous offer): exactly 1/40 outside
    offers = rng.uniform(2.0, 9.0, (2, 5, 4))
    offers[0, 4, 1] = 50.0
    out = crossplay.ood_fractions(support, _games(offers))
    assert out["decisions"] == 40
    assert out["outside_range_per_feature"]["own_offer"] == pytest.approx(1 / 40)
    assert out["outside_range_per_feature"]["own_prev_offer"] == 0.0
    assert out["outside_range_any"] == pytest.approx(1 / 40)


def test_decision_inputs_mark_inapplicable_inputs() -> None:
    offers = np.full((1, 3, 4), 5.0)
    offers[0, 0, 0] = 0.5  # player 0 excluded in round 1
    inputs = crossplay.decision_inputs(_games(offers))
    # stream 0 is player 0: own previous offer < 1 in round 2 -> x2, own_prev NaN
    assert np.isnan(inputs["x2"][0, 1]) and np.isnan(inputs["own_prev_offer"][0, 1])
    assert np.isnan(inputs["x3"][:, 0]).all()  # round 1: nothing before
    assert inputs["x4"][0, 0] == pytest.approx(0.5)


def _random_b1(members: int, identical: bool) -> rc.B1Port:
    shapes = b1_gru.weight_shapes()
    rng = np.random.default_rng(9)
    first = {n: rng.normal(0.0, 0.3, s) for n, s in shapes.items()}
    stacked = {
        n: np.stack(
            [
                first[n] if identical or m == 0 else rng.normal(0.0, 0.3, shapes[n])
                for m in range(members)
            ]
        )
        for n in shapes
    }
    return rc.B1Port(weights={n: twin.f64(v) for n, v in stacked.items()})


def test_b1_disagreement_is_zero_for_identical_members() -> None:
    traj, _ = ppo.evaluate_planner(_planner(1), UniformReturn(), jax.random.key(2), 4, ROUNDS)
    same = crossplay.b1_disagreement(_random_b1(3, identical=True), traj)
    differ = crossplay.b1_disagreement(_random_b1(3, identical=False), traj)
    assert same["mean_tv"] == pytest.approx(0.0, abs=1e-12) and same["decisions"] > 0
    assert differ["mean_tv"] > 1e-3


# --- distillation ----------------------------------------------------------------------------------------------


def _interpolating_games(k: float, n_games: int = 400) -> rc.Trajectory:
    traj, _ = rc.rollout(
        UniformReturn(),
        distill.interpolating_step(k),
        None,
        jax.random.key(11),
        jax.random.key(0),
        n_games,
    )
    return traj


def test_distillation_recovers_a_known_interpolating_rule() -> None:
    ln_k = 0.7  # a point of the paper's grid
    k = math.exp(ln_k)
    traj = _interpolating_games(k)
    rows = distill.trajectory_rows(traj)
    result = distill.distill(rows)
    assert result.interpolating["ln_k"] == pytest.approx(ln_k)
    assert result.interpolating["r2"] > 1.0 - 1e-12
    assert np.abs(result.w_rule.kept).max() < 1e-12  # the rule keeps nothing
    # binned w is the least-squares average of (R / 200)^k within each bin
    b = distill.bin_index(rows.pool_frac, result.w_rule.w.size)
    for j in range(result.w_rule.w.size):
        sel = (b == j) & rows.informative
        if sel.sum() < 400:
            continue
        a = 0.25 - rows.q[sel]
        true = float(((rows.pool_frac[sel] ** k) * a * a).sum() / (a * a).sum())
        assert result.w_rule.w[j] == pytest.approx(true, abs=1e-9)
    assert result.w_rule_r2 > 0.99
    # the distilled one-parameter rule earns exactly the rule's surplus on the same games
    key = jax.random.key(11)
    rule = distill.total_surplus(distill.interpolating_step(k), None, UniformReturn(), key, 400)
    fitted = distill.total_surplus(
        distill.interpolating_step(result.interpolating["k"]), None, UniformReturn(), key, 400
    )
    assert np.array_equal(rule, fitted)
    binned = distill.total_surplus(
        distill.w_rule_step(result.w_rule), None, UniformReturn(), key, 400
    )
    assert abs(binned.mean() - rule.mean()) < 0.02 * rule.mean()


def test_tree_recovers_a_step_function_and_jax_matches_numpy() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 1.0, (2000, 4))
    y = np.where(x[:, 2] > 0.5, 0.3, 0.1)
    tree = distill.fit_tree(x, y, depth=2, min_leaf=50)
    assert tree.feature[0] == 2 and 0.45 < tree.threshold[0] < 0.55
    assert np.allclose(tree.predict(x), y)
    # the JAX rule evaluates the same tree on the same features
    g = 5
    obs = rc.MechObs(
        pool=twin.f64(rng.uniform(10.0, 200.0, g)),
        prev_offers=twin.f64(rng.uniform(1.0, 40.0, (g, 4))),
        prev_reciprocations=twin.f64(rng.integers(0, 5, (g, 4))),
        t=twin.asarray(2),
    )
    prev_pool = twin.f64(rng.uniform(10.0, 200.0, g))
    feats = np.asarray(distill.tree_features(obs, prev_pool)).reshape(-1, 4)
    step = distill.tree_rule_step(tree, 0.25)
    offers, _, _ = step(prev_pool, obs, jax.random.key(0))
    shares = tree.predict(feats).reshape(g, 4)
    shares = np.where(
        shares.sum(1, keepdims=True) > 1, shares / shares.sum(1, keepdims=True), shares
    )
    assert np.allclose(np.asarray(offers), np.asarray(obs.pool)[:, None] * shares, rtol=1e-12)


def test_tree_features_agree_with_the_row_features() -> None:
    traj = _interpolating_games(2.0, n_games=3)
    rows = distill.trajectory_rows(traj)
    offers, actions, pool = (np.asarray(a) for a in (traj.offers, traj.actions, traj.pool))
    t = 5
    obs = rc.MechObs(
        pool=twin.f64(pool[:, t]),
        prev_offers=twin.f64(offers[:, t - 1]),
        prev_reciprocations=twin.f64(actions[:, t - 1]),
        t=twin.asarray(t),
    )
    jax_feats = np.asarray(distill.tree_features(obs, twin.f64(pool[:, t - 1])))
    row_feats = rows.features.reshape(3, ROUNDS, 4, 4)[:, t]
    assert np.allclose(jax_feats, row_feats, rtol=1e-12, atol=1e-15)


# --- planner versus B4 --------------------------------------------------------------------------------------------


FROZEN_SYSTEM_PROMPT = (
    "You are a participant in an online decision-making study. Read the instructions and "
    "answer exactly in the requested format."
)
FROZEN_RULES_SHA256 = "a9d5de8ad52bafd8d5696d02060ba1eab708bf2784715ee0aa74224838c6808c"
FROZEN_ARGV = [
    "claude", "-p", "--model", "claude-haiku-5-5", "--effort", "medium", "--output-format",
    "json", "--system-prompt", FROZEN_SYSTEM_PROMPT, "--tools", "", "--setting-sources", "",
    "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
    "--safe-mode", "--max-turns", "1",
]  # fmt: skip
"""The calibration-v2 run's recorded configuration (citizens-c2-b4-calibration-v2)."""
FROZEN_TYPE_TEXTS = {
    "T1": "About you: you are the kind of person who keeps most of the coins offered to you "
    "for yourself.",
    "T2": "About you: you are the kind of person who returns about as much as the other "
    "players returned in the previous round.",
    "T3": "About you: you are the kind of person who returns coins generously to keep the "
    "pool healthy.",
}
FROZEN_PROMPT_SHA256 = "e0e628cdd49eb6ea6e4983b0498c42bbc2631ed2f7b595ebd96eed402f0bfed3"
"""SHA-256 of `_fixed_prompt()` rendered by `prompts.py` at d249ef0 (the commit the B4
held-out runs execute from)."""


def _fixed_prompt() -> str:
    past = (
        prompts.PastRound(200.0, (50.0, 50.0, 50.0, 49.99), (25, 20, 30, 0)),
        prompts.PastRound(180.5, (60.25, 40.0, 0.5, 79.25), (30, 0, 0, 79)),
    )
    obs = prompts.Observation(2, 150.75, (12.5, 0.25, 100.0, 38.0), past)
    return prompts.render_prompt(obs, "T2")


def test_b4_prompt_and_flags_are_the_frozen_b4_prompt_v2() -> None:
    assert prompts.PROMPT_VERSION == "b4-prompt-v2"
    assert prompts.SYSTEM_PROMPT == FROZEN_SYSTEM_PROMPT
    assert hashlib.sha256(prompts.RULES.encode("utf-8")).hexdigest() == FROZEN_RULES_SHA256
    assert ClaudeCLI().argv() == FROZEN_ARGV
    assert {str(k): v for k, v in prompts.TYPE_TEXTS.items()} == FROZEN_TYPE_TEXTS
    assert hashlib.sha256(_fixed_prompt().encode("utf-8")).hexdigest() == FROZEN_PROMPT_SHA256
    config = runner.RunConfig(
        run_id="x", label="c4-planner", variant="plain", mechanism="m", target_games=1,
        n_rounds=40, workers=1, max_retries=2, discard_limit=1, seed=1, backend={},
    )  # fmt: skip
    assert config.prompt_version == "b4-prompt-v2"
    assert config.rules_sha256 == FROZEN_RULES_SHA256


class FakeBackend:
    """Answers floor(max_return / 2) to every prompt (no LLM, no subprocess)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> CallResult:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        n = int(re.findall(r"from 0 to (\d+)", prompt)[-1])
        return CallResult(
            "{}", "", 0, 0.01, False, json.dumps({"return": n // 2}), False, 0.0, ["fake"]
        )

    def describe(self) -> dict[str, Any]:
        return {"kind": "fake"}


@pytest.mark.parametrize("retention", [True, False])
def test_planner_offers_in_llm_games_equal_the_jax_rollout(retention: bool) -> None:
    planner = _planner(13, retention)
    mechanism = b4play.PlannerGameMechanism("Planner[toy]", planner)
    backend = FakeBackend()
    spec = runner.GameSpec(0, ROUNDS, (None, None, None, None))

    async def go() -> runner.GameOutcome:
        return await runner.play_game(
            spec, mechanism, backend, asyncio.Semaphore(4), lambda r: None
        )

    outcome = asyncio.run(go())
    assert outcome.status == "complete" and len(outcome.rounds) == ROUNDS
    traj, _ = ppo.evaluate_planner(planner, HalfOfFloor(), jax.random.key(0), 1, ROUNDS)
    offers = np.array([r.offers for r in outcome.rounds])
    assert np.allclose(offers, np.asarray(traj.offers)[0], rtol=0, atol=1e-9)
    returns = np.array([r.returns for r in outcome.rounds], dtype=np.float64)
    assert np.array_equal(returns, np.asarray(traj.actions)[0])
    assert len(backend.prompts) == int((offers >= 1.0).sum())
    first = runner.observation(0, 0, 200.0, offers[0], [])
    assert backend.prompts[0] == prompts.render_prompt(first, None)


def test_planner_b4_run_with_a_fake_backend(tmp_path: Path) -> None:
    mechanism = b4play.PlannerGameMechanism("Planner[toy]", _planner(2))

    def spec(k: int) -> runner.GameSpec:
        return runner.GameSpec(k, ROUNDS, (None, None, None, None))

    config, outcomes = b4play.run_planner_games(
        mechanism, FakeBackend(), tmp_path, "c4-b4-toy", 2, spec, variant="plain", workers=2
    )
    assert [o.status for o in outcomes] == ["complete", "complete"]
    assert config.prompt_version == "b4-prompt-v2" and config.mechanism == "Planner[toy]"


def test_b4_call_estimate() -> None:
    default = b4play.calls_estimate()
    assert default["calls"] == pytest.approx(106.11666666666666 * 40)
    measured = b4play.calls_estimate(150.0)
    assert measured["calls_per_game"] == pytest.approx(150.0 * (1 + 0.020025632809996797))
    assert measured["calls_worst_case"] == pytest.approx(160 * 1.020025632809996797 * 40)


def test_c4_evidence_commands_refuse_before_the_freeze(tmp_path: Path) -> None:
    for command in ("crossplay", "distill"):
        with pytest.raises(Exception, match="confirm"):
            c4.main(
                [
                    "--runs-dir",
                    str(tmp_path),
                    command,
                    "--train-run",
                    str(tmp_path),
                    "--run-id",
                    "x",
                ]
            )
    with pytest.raises(Exception, match=r"does not exist|status"):
        c4.main(
            [
                "--runs-dir", str(tmp_path), "b4", "--confirm-frozen", "--train-run",
                str(tmp_path), "--variant", "plain", "--run-id", "x",
            ]
        )  # fmt: skip


def test_c4_manifest_labels() -> None:
    m = stage.base_manifest("citizens-c4-x", "train", None, 11, milestone="c4")
    assert m["experiment_id"] == "citizens-c4-train-v1"
    assert m["citizens"]["stage"] == "c4_train"
    assert m["citizens"]["c4_config_sha256"] == c4_config.config_sha256()
    c3 = stage.base_manifest("citizens-c3-x", "train", None, 11)
    assert c3["citizens"]["stage"] == "c3_train" and "c4_config_sha256" not in c3["citizens"]
    with pytest.raises(ValueError, match="milestone"):
        stage.base_manifest("x", "train", None, 1, milestone="c5")
    pilot = c4.c4_manifest("citizens-c4-pilot-b2-s901", "pilot", None, 901)
    assert pilot["citizens"]["stage"] == "c4_pilot" and "pilot" in pilot["limitations"][0]
