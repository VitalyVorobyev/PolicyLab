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

from phasespace.citizens import data, game
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


# --- C4 training requests, the shared signature and grouping (P1) -----------------------------------


def test_c4_training_request_accepts_only_declared_rows_and_seeds() -> None:
    ok = stage.c4_train_run_id
    stage.check_c4_training_request(ok("B2", 1003, True), "B2", 1003, True, 1000, {})
    stage.check_c4_training_request(
        ok("B23", 2002, False), "B23", 2002, False, 1000, {"retention": False}
    )
    stage.check_c4_training_request(ok("Mix", 3001, True), "Mix", 3001, True, 1000, {})
    stage.check_c4_training_request(ok("B1", 1001, True, 2), "B1", 1001, True, 1000, {})
    assert ok("B1", 1001, True) == "citizens-c4-train-b1-s1001-v1"
    assert ok("B3", 2001, False) == "citizens-c4-train-b3-noret-s2001-v1"
    assert ok("Mix", 3002, True) == "citizens-c4-train-mix-s3002-v1"
    refused: list[tuple[str, int, bool, int, dict[str, Any], str]] = [
        ("B2", 5, True, 1000, {}, "not declared"),
        ("B2", 901, True, 1000, {}, "not declared"),
        ("B2", 1, True, 1000, {}, "not declared"),
        ("B2", 2001, True, 1000, {}, "not declared"),  # an ablation seed on a main row
        ("B2", 1001, False, 1000, {"retention": False}, "not declared"),
        ("Mix", 3001, False, 1000, {"retention": False}, "no planner row"),
        ("B9", 1001, True, 1000, {}, "no planner row"),
        ("B2", 1001, True, 500, {}, "1000 iterations"),
        ("B2", 1001, True, 1000, {"learning_rate": 3e-4}, "no PPO overrides"),
    ]
    for model, seed, retention, iterations, overrides, message in refused:
        run_id = ok(model, seed, retention) if model != "B9" else "citizens-c4-train-b9-s1001-v1"
        with pytest.raises(SystemExit, match=message):
            stage.check_c4_training_request(run_id, model, seed, retention, iterations, overrides)
    with pytest.raises(SystemExit, match="run ids"):
        stage.check_c4_training_request("citizens-c4-b2-1001", "B2", 1001, True, 1000, {})


def _train_args(tmp_path: Path, model: str, seed: int, *extra: str) -> list[str]:
    run_id = stage.c4_train_run_id(model, seed, "--no-retention" not in extra)
    return [
        "--runs-dir", str(tmp_path), "train", "--milestone", "c4", "--run-id", run_id,
        "--model", model, "--seed", str(seed), "--iterations", "1000",
        "--fit-run", str(tmp_path / "no-fit-run"), *extra,
    ]  # fmt: skip


def test_c4_training_refuses_undeclared_seeds_and_the_unfrozen_protocol(tmp_path: Path) -> None:
    """Both refusals happen before any input is read or any training step is taken
    (the fit run does not exist)."""
    with pytest.raises(SystemExit, match="not declared"):
        stage.main(_train_args(tmp_path, "B2", 7, "--confirm-frozen"))
    with pytest.raises(SystemExit, match="not declared"):
        stage.main(_train_args(tmp_path, "B1", 1001, "--no-retention", "--confirm-frozen"))
    # a declared seed: refused because the C4 protocol is not frozen (draft) or not confirmed
    with pytest.raises(data.ProtocolNotFrozenError, match="confirm"):
        stage.main(_train_args(tmp_path, "B2", 1001))
    with pytest.raises(data.ProtocolNotFrozenError, match="header status"):
        stage.main(_train_args(tmp_path, "B2", 1001, "--confirm-frozen"))
    with pytest.raises(SystemExit, match="C4 row"):
        stage.main(
            [
                "--runs-dir", str(tmp_path), "train", "--run-id", "citizens-c3-mix", "--model",
                "Mix", "--seed", "5", "--iterations", "10",
            ]
        )  # fmt: skip


def _train_manifest(
    run_id: str, model: str, seed: int, retention: bool = True, iterations: int = 1000
) -> dict[str, Any]:
    ppo_record = {**ppo.PPOConfig().to_json(), "retention": retention}
    return {
        "run_id": run_id,
        "source": {"git_dirty": False, "git_commit": "c" * 40},
        "citizens": {
            "stage": "c4_train",
            "c3_config_sha256": c3config.config_sha256(),
            "c4_config_sha256": c4_config.config_sha256(),
            "fit_run": {"manifest_sha256": "f" * 64},
            "seeds": {"eval_key": c3config.EVAL_KEY_SEED},
            "training": {
                "model": model,
                "seed": seed,
                "ppo": ppo_record,
                "overrides": {} if retention else {"retention": False},
                "iterations": iterations,
            },
        },
    }


def _all_c4_manifests() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (model, retention), seeds in c4.expected_rows().items():
        for seed in seeds:
            out.append(
                _train_manifest(
                    stage.c4_train_run_id(model, seed, retention), model, seed, retention
                )
            )
    return out


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


def test_expected_rows_are_the_protocol_rows() -> None:
    rows = c4.expected_rows()
    assert len(rows) == 9
    assert sum(len(s) for s in rows.values()) == 35  # 20 main + 12 ablation + 3 mixture
    assert rows[("Mix", True)] == (3001, 3002, 3003)
    assert rows[("B3", False)] == (2001, 2002, 2003)


def test_training_runs_are_grouped_by_row() -> None:
    trains = _all_c4_manifests()
    groups = c4.group_training_runs(_base_manifest(), trains)
    assert list(groups)[:3] == [("B1", True), ("B1", False), ("B2", True)]
    assert list(groups)[-1] == ("Mix", True)
    assert [trains[i]["citizens"]["training"]["seed"] for i in groups[("B1", True)]] == list(
        c4_config.MAIN_SEEDS
    )


def _set_stage(ms: list[dict[str, Any]]) -> None:
    ms[3]["citizens"]["stage"] = "c4_pilot"


def _set_c3(ms: list[dict[str, Any]]) -> None:
    ms[3]["citizens"]["stage"] = "c3_train"


def _set_reserved(ms: list[dict[str, Any]]) -> None:
    ms[3]["citizens"]["training"]["seed"] = 901


def _set_c3_seed(ms: list[dict[str, Any]]) -> None:
    ms[3]["citizens"]["training"]["seed"] = 2


def _set_undeclared(ms: list[dict[str, Any]]) -> None:
    ms[3]["citizens"]["training"]["seed"] = 1006


def _set_dirty(ms: list[dict[str, Any]]) -> None:
    ms[3]["source"]["git_dirty"] = True


def _set_iterations_one_row(ms: list[dict[str, Any]]) -> None:
    for m in ms:
        if m["citizens"]["training"]["model"] == "B3":
            m["citizens"]["training"]["iterations"] = 500


def _set_iterations_all(ms: list[dict[str, Any]]) -> None:
    for m in ms:
        m["citizens"]["training"]["iterations"] = 500


def _set_commit(ms: list[dict[str, Any]]) -> None:
    ms[-1]["source"]["git_commit"] = "d" * 40


def _set_ppo(ms: list[dict[str, Any]]) -> None:
    ms[-1]["citizens"]["training"]["ppo"]["learning_rate"] = 3e-4


def _set_override(ms: list[dict[str, Any]]) -> None:
    ms[-1]["citizens"]["training"]["overrides"]["clip"] = 0.1


def _set_fit(ms: list[dict[str, Any]]) -> None:
    ms[-1]["citizens"]["fit_run"]["manifest_sha256"] = "e" * 64


def _drop_seed(ms: list[dict[str, Any]]) -> None:
    del ms[2]


def _drop_row(ms: list[dict[str, Any]]) -> None:
    ms[:] = [m for m in ms if m["citizens"]["training"]["model"] != "Mix"]


def _duplicate(ms: list[dict[str, Any]]) -> None:
    ms.append(json.loads(json.dumps(ms[0])))


def _reduce_one_noret_row(ms: list[dict[str, Any]]) -> None:
    ms[:] = [m for m in ms if m["run_id"] != "citizens-c4-train-b2-noret-s2003-v1"]


CHANGES: list[tuple[Callable[[list[dict[str, Any]]], None], str]] = [
    (_set_stage, "never C4 inputs"),
    (_set_c3, "never C4 inputs"),
    (_set_reserved, "reserved"),
    (_set_c3_seed, "reserved"),
    (_set_undeclared, "not declared"),
    (_set_dirty, "dirty"),
    (_set_iterations_one_row, r"signature differs .*iterations"),
    (_set_iterations_all, "1000"),
    (_set_commit, r"signature differs .*git_commit"),
    (_set_ppo, r"signature differs .*ppo"),
    (_set_override, r"signature differs .*overrides"),
    (_set_fit, r"signature differs .*fit_run"),
    (_drop_seed, "seeds"),
    (_drop_row, "missing"),
    (_duplicate, "twice"),
    (_reduce_one_noret_row, "different seed counts"),
]


@pytest.mark.parametrize(("change", "message"), CHANGES)
def test_crossplay_refuses_runs_outside_the_shared_signature_and_declared_seeds(
    change: Callable[[list[dict[str, Any]]], None], message: str
) -> None:
    trains = _all_c4_manifests()
    change(trains)
    with pytest.raises(stage.RunMismatchError, match=message):
        c4.group_training_runs(_base_manifest(), trains)


def test_secondary_rows_may_use_the_pre_declared_reduction() -> None:
    reduced = [
        m for m in _all_c4_manifests() if m["citizens"]["training"]["seed"] not in (2003, 3003)
    ]
    groups = c4.group_training_runs(_base_manifest(), reduced)
    assert len(groups[("B1", False)]) == 2 and len(groups[("Mix", True)]) == 2
    main_reduced = [m for m in _all_c4_manifests() if m["citizens"]["training"]["seed"] != 1005]
    with pytest.raises(stage.RunMismatchError, match="seeds"):
        c4.group_training_runs(_base_manifest(), main_reduced)


def test_base_run_must_match() -> None:
    base = _base_manifest()
    base["citizens"]["fit_run"]["manifest_sha256"] = "0" * 64
    with pytest.raises(stage.RunMismatchError, match="fit runs"):
        c4.group_training_runs(base, _all_c4_manifests())
    dirty = _base_manifest()
    dirty["source"]["git_dirty"] = True
    with pytest.raises(stage.RunMismatchError, match="dirty"):
        c4.group_training_runs(dirty, _all_c4_manifests())


def test_check_poolable_any_model() -> None:
    trains = [_train_manifest("b3-1", "B3", 1), _train_manifest("b3-2", "B3", 2)]
    stage.check_poolable(_base_manifest(), trains, model=None)
    with pytest.raises(stage.RunMismatchError, match="not B1"):
        stage.check_poolable(_base_manifest(), trains)


def test_planners_are_loaded_from_run_directories(tmp_path: Path) -> None:
    from phasespace.gravity.export import write_run_directory

    dirs: list[Path] = []
    docs: list[dict[str, Any]] = []
    for manifest in _all_c4_manifests():
        manifest["citizens"]["training"]["ppo"]["hidden"] = 8
        run_id = manifest["run_id"]
        retention = bool(manifest["citizens"]["training"]["ppo"]["retention"])
        planner = _planner(int(manifest["citizens"]["training"]["seed"]), retention)
        written = write_run_directory(
            tmp_path / run_id,
            manifest,
            [("model_parameters", stage.PLANNER_FILE, serialise(planner))],
        )
        dirs.append(tmp_path / run_id)
        docs.append(written)
    policies = c4.load_policies(_base_manifest(), dirs, docs)
    names = [p.name for p in policies]
    assert names[:9] == [
        "Planner[B1]", "Planner-noret[B1]", "Planner[B2]", "Planner-noret[B2]", "Planner[B3]",
        "Planner-noret[B3]", "Planner[B23]", "Planner-noret[B23]", "Planner[Mix]",
    ]  # fmt: skip
    assert policies[0].seeds == c4_config.MAIN_SEEDS and policies[1].seeds == (2001, 2002, 2003)
    assert names[9:] == [
        "Equal",
        "Mixed",
        "Proportional",
        "Interpolating k=22",
        *(f"Interpolating k*[{m}]" for m in rc.TRAINABLE_MODELS),
    ]
    assert not policies[1].planners[0].retention and policies[8].training_model == "Mix"


# --- the mixture citizen (P7) ------------------------------------------------------------------------------


class ConstantFraction(eqx.Module):
    """A pmf citizen returning floor(f floor(e)) with probability 1 (0 when e < 1)."""

    fraction: float = eqx.field(static=True)
    name: str = eqx.field(static=True, default="constant-fraction")

    def init(self, key: Array, n_streams: int) -> None:
        del key, n_streams

    def step(self, state: None, obs: rc.Obs) -> tuple[Array, None]:
        c = jnp.floor(self.fraction * twin.max_reciprocation(obs.offers[:, 0]))
        return jax.nn.one_hot(c.astype(jnp.int32), K_MAX, dtype=jnp.float64), state


def _toy_ports() -> dict[str, Any]:
    return {
        "A": UniformReturn(),
        "B": ConstantFraction(0.0),
        "C": ConstantFraction(0.5),
        "D": ConstantFraction(1.0),
    }


def test_mixture_games_are_played_exactly_as_under_their_drawn_model() -> None:
    """Independent reference: each model alone, on the same citizen key (same uniforms;
    the toy components have no latent draw), equals the mixture's games of that model."""
    ports = _toy_ports()
    names = ("A", "B", "C", "D")
    mix = rc.MixtureCitizen.of(ports, names)
    key, g = jax.random.key(21), 64
    mech = rc.weight_rule_step(rc.interpolating(22.0))
    traj, _ = rc.rollout(mix, mech, None, key, jax.random.key(0), g)
    k_init, _ = rc.citizen_draws(key, g, ROUNDS)
    drawn = np.asarray(mix.draw_models(k_init, g))
    assert set(drawn.tolist()) == {0, 1, 2, 3}
    for m, name in enumerate(names):
        alone, _ = rc.rollout(ports[name], mech, None, key, jax.random.key(0), g)
        sel = drawn == m
        for field_name in ("offers", "actions", "pool"):
            got = np.asarray(getattr(traj, field_name))[sel]
            want = np.asarray(getattr(alone, field_name))[sel]
            assert np.array_equal(got, want), (name, field_name)
    # the drawn model is the same for the 4 players of a game
    state = mix.init(k_init, 4 * g)
    assert np.array_equal(np.asarray(state.model), np.repeat(drawn, 4))


def test_mixture_draw_is_uniform_and_a_function_of_the_key() -> None:
    mix = rc.MixtureCitizen.of(_toy_ports(), ("A", "B", "C", "D"))
    n = 40_000
    drawn = np.asarray(mix.draw_models(jax.random.key(3), n))
    counts = np.bincount(drawn, minlength=4)
    # binomial SD sqrt(n p (1 - p)) = 86.6; 5 SD
    assert np.all(np.abs(counts - n / 4) < 5 * math.sqrt(n * 0.25 * 0.75))
    again = np.asarray(mix.draw_models(jax.random.key(3), n))
    assert np.array_equal(drawn, again)
    assert not np.array_equal(drawn, np.asarray(mix.draw_models(jax.random.key(4), n)))
    with pytest.raises(TypeError):
        rc.MixtureCitizen.of({"F": rc.FixedFraction(0.5)}, ("F",))


def test_mixture_planner_trains_and_evaluates_against_the_mixture() -> None:
    toy = _toy_ports()
    ports: dict[str, Any] = dict(
        zip(rc.TRAINABLE_MODELS, (toy["A"], toy["B"], toy["C"], toy["D"]), strict=True)
    )
    mixture = stage.training_citizen(ports, "Mix")
    assert isinstance(mixture, rc.MixtureCitizen) and mixture.names == rc.TRAINABLE_MODELS
    assert stage.training_citizen(ports, "B2") is ports["B2"]
    cfg = ppo.PPOConfig(n_games=16, minibatches=2, hidden=8)
    planner, curve = ppo.train(mixture, cfg, seed=4, n_iterations=2, chunk=1)
    assert len(curve) == 2 and np.isfinite(curve[-1].episode_surplus)
    traj, _ = ppo.evaluate_planner(planner, mixture, jax.random.key(8), 8, ROUNDS)
    assert np.all(np.isfinite(np.asarray(traj.pool)))
    assert stage.scored_models("Mix") == rc.TRAINABLE_MODELS
    assert stage.scored_models("B2") == ("B2",)


# --- coin-weighted kept share (P2) ----------------------------------------------------------------------------


def _traj(offers: np.ndarray, pool: np.ndarray) -> rc.Trajectory:
    return rc.Trajectory(
        offers=twin.f64(offers), actions=twin.f64(np.zeros_like(offers)), pool=twin.f64(pool)
    )


def test_coin_weighted_kept_share_by_hand() -> None:
    """Round 0: games R = 100 keeping 10 and R = 50 keeping 0 -> 10 / 150 (the mean of
    ratios would be 0.05). Round 1: R = 0.01 keeping 0.001 (excluded, R < 1) and R = 80
    keeping 0 -> 0 (the mean of ratios would read 5 %). Round 2: both games below 1 coin
    -> null, with the reason."""
    pool = np.array([[100.0, 0.01, 0.5], [50.0, 80.0, 0.2]])
    kept = np.array([[10.0, 0.001, 0.1], [0.0, 0.0, 0.0]])
    offers = np.repeat(((pool - kept) / 4.0)[..., None], 4, axis=-1)
    k, r = stage.kept_coins(_traj(offers, pool))
    assert np.allclose(k, [10.0, 0.0, 0.0]) and np.allclose(r, [150.0, 80.0, 0.0])
    out = stage.coin_weighted_kept_share(k, r, rounds=(0, 1))
    assert out["per_round"][0] == pytest.approx(10.0 / 150.0)
    assert out["per_round"][1] == pytest.approx(0.0, abs=1e-12)
    assert out["per_round"][2] is None and "rounds (0-based) [2]" in out["null_reason"]
    assert out["pooled"] == pytest.approx(10.0 / 230.0)
    assert out["endgame"] == pytest.approx(10.0 / 230.0)
    legacy = stage.kept_share_mean_of_ratios(_traj(offers, pool))
    assert legacy[0] == pytest.approx(0.05) and legacy[1] == pytest.approx(0.05)


# --- input support labels and null diagnostics (P5, NaN) -------------------------------------------------------


def test_input_support_is_labelled_per_model() -> None:
    rng = np.random.default_rng(0)
    train = _games(rng.uniform(1.0, 10.0, (20, 5, 4)))
    b2_out = crossplay.ood_fractions(crossplay.training_support(train, "B2"), train)
    b3_out = crossplay.ood_fractions(crossplay.training_support(train, "B3"), train)
    assert b2_out["kind"] == "input_out_of_range" and "model extrapolation" in b2_out["label"]
    assert b3_out["kind"] == "state_novelty" and "not model extrapolation" in b3_out["label"]
    b23 = crossplay.training_support(train, "B23")
    assert crossplay.ood_fractions(b23, train)["kind"] == "state_novelty"
    forced = _games(np.full((2, 3, 4), 0.5))
    empty = crossplay.ood_fractions(crossplay.training_support(train, "B3"), forced)
    assert empty["decisions"] == 0 and empty["off_grid"] is None and "null_reason" in empty


def test_b1_disagreement_without_decisions_is_null() -> None:
    offers = np.full((2, ROUNDS, 4), 0.5)
    traj = rc.Trajectory(
        offers=twin.f64(offers), actions=twin.f64(np.zeros_like(offers)),
        pool=twin.f64(np.full((2, ROUNDS), 2.0)),
    )  # fmt: skip
    out = crossplay.b1_disagreement(_random_b1(2, identical=False), traj)
    assert out["mean_tv"] is None and out["decisions"] == 0 and "null_reason" in out


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
    # coin-weighted: sum of the 0.001 withheld over games with R >= 1 / sum of those R
    use = pools >= 1.0
    n_use = use.sum(axis=0)
    want = np.where(n_use > 0, RECORDED_WITHHELD * n_use, np.nan) / np.where(
        n_use > 0, np.where(use, pools, 0.0).sum(axis=0), 1.0
    )
    got = noret.kept_share()["per_round"]
    assert [v is None for v in got] == (n_use == 0).tolist()
    assert np.allclose(np.asarray(got, dtype=np.float64), want, rtol=1e-9, equal_nan=True)
    summary = crossplay.summarise(policies, list(columns), series)
    cell = summary["cells"]["Planner[T1]"]["U"]["deterministic"]
    kept = cell["kept_share_coin_weighted"]
    assert cell["seeds"] == [1, 2] and len(kept["per_round"]) == ROUNDS
    seeds = [s for s in series if s.policy == "Planner[T1]" and s.column == "U"]
    seeds = [s for s in seeds if s.mode == "deterministic"]
    pooled = sum(float(s.kept_coins.sum()) for s in seeds) / sum(
        float(s.pool_coins.sum()) for s in seeds
    )
    assert kept["pooled"] == pytest.approx(pooled)
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
