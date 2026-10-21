"""Model Citizens web demo export: the reproduction check, the traces against the game rule,
the human-game selection rule and the copied summaries.

Toy citizens and synthetic games only: no fitted model, no human data, no recorded run.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phasespace.citizens import web_export as we
from phasespace.citizens.models.base import K_MAX
from phasespace.citizens.records import Games
from phasespace.citizens.rl import checks, crossplay, twin
from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl.planner import Planner
from phasespace.gravity.export import write_run_directory

# The game rule, written out from Koster et al. (2025) and D33, independently of the code:
# pool cap 200, growth 1.4 of returned coins, floor 0.01.
CAP, GROWTH, FLOOR = 200.0, 1.4, 0.01
N_GAMES = 8


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


@pytest.fixture(autouse=True)
def x64_mode() -> Iterator[None]:
    with twin.x64():
        yield


def _policies() -> list[crossplay.Policy]:
    planners = [Planner(8, -0.5, jax.random.key(s)) for s in (1, 2)]
    return [
        crossplay.planner_policy("T1", True, [1, 2], planners),
        *crossplay.rule_policies({"T1": 3.0}),
    ]


def _columns() -> dict[str, rc.JaxCitizen]:
    return {"F": rc.FixedFraction(0.5), "U": UniformReturn()}


def _recorded_and_traced() -> tuple[
    list[dict[str, Any]], list[tuple[str, str, int | None, rc.Trajectory]]
]:
    key, noise = jax.random.key(5), jax.random.key(6)
    series: list[crossplay.Series] = []
    traced: list[tuple[str, str, int | None, rc.Trajectory]] = []
    for name, port in _columns().items():
        series += crossplay.evaluate_column(
            _policies(), name, port, key, noise, N_GAMES, crossplay.Diagnostics()
        )
        for policy, seed, traj in crossplay.deterministic_trajectories(
            _policies(), port, key, N_GAMES
        ):
            traced.append((policy, name, seed, traj))
    return crossplay.series_document(series), traced


# --- reproduction check ---------------------------------------------------------------------------


def test_traced_rollouts_reproduce_the_recorded_series_and_a_perturbed_total_is_caught() -> None:
    recorded, traced = _recorded_and_traced()
    by_key = {
        (s["policy"], s["column"], s["seed"]): s["total_surplus"]
        for s in recorded
        if s["mode"] == "deterministic"
    }
    assert len(traced) == len(by_key) == 2 * (2 + 5)
    for policy, column, seed, traj in traced:
        totals = checks.trajectory_metrics(traj)["total_surplus"]
        assert we.check_reproduction(totals, by_key[(policy, column, seed)], policy) == 0.0
    policy, column, seed, traj = traced[0]
    totals = checks.trajectory_metrics(traj)["total_surplus"]
    perturbed = list(by_key[(policy, column, seed)])
    perturbed[3] += 1e-6
    with pytest.raises(we.ReproductionError):
        we.check_reproduction(totals, perturbed, policy)
    with pytest.raises(we.ReproductionError):
        we.check_reproduction(totals, perturbed[:-1], policy)  # a missing game


# --- traces against the game rule ---------------------------------------------------------------------


def _independent_violations(record: dict[str, Any], tol: float) -> list[str]:
    """Rule violations of an exported (rounded) game, checked with plain Python."""
    out: list[str] = []
    pool, offers, returns, kept = (
        record["pool"],
        record["offers"],
        record["returns"],
        record["kept"],
    )
    assert len(pool) == record["rounds"] + 1
    running = 0.0
    for t in range(record["rounds"]):
        e, c, k = offers[t], returns[t], kept[t]
        expected = max(FLOOR, min(CAP, pool[t] - sum(e) + GROWTH * sum(c)))
        if abs(pool[t + 1] - expected) > tol:
            out.append(f"pool update in round {t}")
        if sum(e) > pool[t] + 0.02:
            out.append(f"offers above the pool in round {t}")
        for i in range(4):
            if abs(k[i] - (e[i] - c[i])) > 0.011:
                out.append(f"kept != offer - returned, round {t}, player {i}")
            if c[i] < 0 or c[i] > e[i] + 1e-9:
                out.append(f"return outside [0, offer], round {t}, player {i}")
        running += sum(e) - sum(c)
        if abs(record["cumulative_surplus"][t] - running) > 0.01 * (t + 1) * 4:
            out.append(f"running surplus in round {t}")
    return out


def test_exported_traces_obey_the_game_rule() -> None:
    _, traced = _recorded_and_traced()
    for policy, column, _seed, traj in traced:
        offers = np.asarray(traj.offers)
        returns = np.asarray(traj.actions)
        pools = np.asarray(traj.pool)
        finals = np.asarray(
            twin.next_pool(traj.pool[:, -1], traj.offers[:, -1], traj.actions[:, -1])
        )
        for g in range(N_GAMES):
            # unrounded: the rule holds to float precision
            res = we.trace_residuals(pools[g], offers[g], returns[g], float(finals[g]))
            assert res["pool_update"] <= 1e-9, (policy, column, g)
            assert res["offers_above_pool"] <= 1e-9 and res["returns_outside_offer"] == 0.0
            # the consecutive recorded pools themselves follow the rule
            for t in range(pools.shape[1] - 1):
                want = max(
                    FLOOR, min(CAP, pools[g, t] - offers[g, t].sum() + GROWTH * returns[g, t].sum())
                )
                assert pools[g, t + 1] == pytest.approx(want, abs=1e-9)
            # rounded export: within rounding of the rule
            record = we.game_record(pools[g], offers[g], returns[g], float(finals[g]))
            assert _independent_violations(record, tol=0.06) == [], (policy, column, g)
            assert record["total_surplus"] == pytest.approx(
                float((offers[g] - returns[g]).sum()), abs=0.005
            )


def test_trace_residuals_catch_a_broken_pool_update() -> None:
    pool = np.array([200.0, 160.0])
    offers = np.array([[10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0]])
    returns = np.array([[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    # 200 - 40 = 160; 160 - 20 + 28 = 168
    assert we.trace_residuals(pool, offers, returns, 168.0)["pool_update"] == pytest.approx(0.0)
    assert we.trace_residuals(pool, offers, returns, 170.0)["pool_update"] == pytest.approx(2.0)
    bad = we.trace_residuals(np.array([200.0, 10.0]), offers, returns, 9.0)
    assert bad["offers_above_pool"] == pytest.approx(10.0)


def test_game_record_by_hand() -> None:
    pool = np.array([200.0, 179.0])
    offers = np.array([[50.0, 50.0, 50.0, 49.999], [44.446, 44.446, 44.446, 44.446]])
    returns = np.array([[40.0, 30.0, 20.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
    rec = we.game_record(pool, offers, returns, 1.224)
    assert rec["rounds"] == 2
    assert rec["pool"] == [200.0, 179.0, 1.22]
    assert rec["kept"][0] == [10.0, 20.0, 30.0, 40.0]  # 49.999 - 10 = 39.999 -> 40.0
    assert rec["kept"][1] == [44.45] * 4
    assert rec["cumulative_surplus"] == [100.0, 277.78]  # 99.999; 99.999 + 177.784
    assert rec["total_surplus"] == 277.78


# --- human games -------------------------------------------------------------------------------------------


def _games(
    mechanism: list[str],
    cohort: list[str],
    launch: list[str],
    rounds: list[int],
    present: list[bool],
) -> Games:
    g, t = len(launch), 41
    per_game: dict[str, Any] = {
        "mechanism": np.array(mechanism),
        "cohort": np.array(cohort),
        "exp4_game": np.zeros(g, dtype=np.int64),
        "launch_id": np.array(launch),
        "n_rounds": np.array(rounds, dtype=np.int64),
        "first_row": np.arange(g, dtype=np.int64) * 100,
    }
    four = {"offers", "actions", "rewards", "prev_reciprocations", "prev_offers",
            "next_reciprocations", "player_present", "player_has_no_choice"}  # fmt: skip
    kw: dict[str, Any] = {
        f.name: per_game[f.name]
        if f.name in per_game
        else np.zeros((g, t, 4) if f.name in four else (g, t))
        for f in dataclasses.fields(Games)
    }
    fp = np.ones((g, t), dtype=np.int8)
    for i, ok in enumerate(present):
        if not ok:
            fp[i, 7] = 0
    kw["four_players_present"] = fp
    return Games(**kw)


def test_human_games_follow_the_declared_rule() -> None:
    games = _games(
        mechanism=["Equal Baseline"] * 6 + ["Mixed Baseline"],
        cohort=["Exp 1", "Exp 4", "Exp 1", "Exp 1", "Exp 2", "Exp 1", "Exp 1"],
        launch=["300", "100", "200", "150", "250", "120", "050"],
        rounds=[40, 40, 40, 39, 40, 40, 40],
        present=[True, True, True, True, True, False, True],
    )
    # Exp 4 (100), 39 rounds (150), a missing player (120) and another mechanism (050) are out;
    # the eligible 200, 250, 300 sort by launch id.
    assert we.select_human_games(games, "Equal Baseline") == [2, 4]
    assert we.select_human_games(games, "Equal Baseline", n=5) == [2, 4, 0]


# --- copied summaries --------------------------------------------------------------------------------------


def test_crossplay_summary_copies_the_recorded_numbers() -> None:
    columns = ["B1"]
    cp = {
        "columns": columns,
        "policies": [
            {"name": "Interpolating k*[B1]", "rule": {"kind": "interpolating_k", "param": 9.0}},
        ],
        "cells": {
            "P": {
                "B1": {
                    "deterministic": {
                        "total_surplus_mean": 900.0,
                        "total_surplus_per_seed": [890.0, 910.0],
                        "seeds": [1, 2],
                        "gini_mean": 0.2,
                    }
                }
            },
            "Equal": {
                "B1": {
                    "deterministic": {
                        "total_surplus_mean": 600.0,
                        "total_surplus_per_seed": [600.0],
                        "seeds": [None],
                        "gini_mean": 0.1,
                    }
                }
            },
            "Interpolating k*[B1]": {
                "B1": {
                    "deterministic": {
                        "total_surplus_mean": 1000.0,
                        "total_surplus_per_seed": [1000.0],
                        "seeds": [None],
                        "gini_mean": 0.1,
                    }
                }
            },
            "Interpolating k=22": {
                "B1": {
                    "deterministic": {
                        "total_surplus_mean": 980.0,
                        "total_surplus_per_seed": [980.0],
                        "seeds": [None],
                        "gini_mean": 0.1,
                    }
                }
            },
        },
        "paired_differences": [
            {
                "policy": "P",
                "column": "B1",
                "mode": "deterministic",
                "versus": "Interpolating k*[B1]",
                "difference_mean": -100.5,
                "difference_ci": [-120.0, -80.0],
                "seeds_positive": 0,
                "seeds_negative": 2,
            },
            {
                "policy": "P",
                "column": "B1",
                "mode": "sampled",
                "versus": "Interpolating k*[B1]",
                "difference_mean": 7.0,
                "difference_ci": [0.0, 9.0],
                "seeds_positive": 2,
                "seeds_negative": 0,
            },
        ],
    }
    out = we.crossplay_summary(cp, ["P", "Equal"])
    p, equal = out["cells"]
    assert p["vs_k_star"]["difference"] == -100.5 and p["vs_k_star"]["ci"] == [-120.0, -80.0]
    assert p["vs_k_star"]["seeds_negative"] == 2
    assert p["vs_k22"]["difference"] == pytest.approx(-80.0) and p["vs_k22"]["ci"] is None
    assert equal["vs_k_star"]["difference"] == pytest.approx(-400.0)
    assert equal["vs_k_star"]["relative"] == pytest.approx(-0.4)
    assert out["k_star"] == {"B1": 9.0}


def test_reference_weight_is_the_deployed_interpolating_rule() -> None:
    fractions = np.array([0.0, 0.5, 0.9, 0.97, 1.0])
    # round((R/200)^22, 1): 0.9^22 = 0.098 -> 0.1; 0.97^22 = 0.512 -> 0.5
    assert we.interpolating_weight(fractions, 22.0).tolist() == [0.0, 0.0, 0.1, 0.5, 1.0]


def test_robustness_is_the_worst_share_of_each_column_s_own_planner() -> None:
    def cell(policy: str, column: str, mean: float) -> dict[str, Any]:
        return {"policy": policy, "column": column, "total_surplus_mean": mean}

    cells = [
        cell("Planner[A]", "A", 1000.0),
        cell("Planner[A]", "B", 300.0),
        cell("Planner[B]", "A", 500.0),
        cell("Planner[B]", "B", 600.0),
        cell("Rule", "A", 800.0),
        cell("Rule", "B", 540.0),
    ]
    out = we.robustness(cells, ["Planner[A]", "Planner[B]", "Rule"], ["A", "B"])["policies"]
    assert out["Planner[A]"]["worst_column"] == "B"
    assert out["Planner[A]"]["worst_ratio"] == pytest.approx(0.5)
    assert out["Planner[B]"]["worst_ratio"] == pytest.approx(0.5)
    assert out["Rule"]["ratios"] == pytest.approx({"A": 0.8, "B": 0.9})
    assert out["Rule"]["worst_column"] == "A"


def test_optional_cell_metrics_are_copied_from_a_supplementary_run(tmp_path: Path) -> None:
    doc = {
        "cells": {
            "Equal": {
                "B1": {
                    "deterministic": {
                        "depletion_round_mean": 14.5,
                        "players_active_mean": 1.25,
                        "total_surplus_mean": 700.0,
                    },
                    "sampled": None,
                }
            },
            "Other": {"B1": {"deterministic": {"depletion_round_mean": 3.0}}},
        }
    }
    payload = (json.dumps(doc) + "\n").encode()
    run = tmp_path / "metrics-run"
    write_run_directory(run, {"run_id": "metrics-run"}, [("summary", "cells.json", payload)])
    out = we.crossplay_metrics(run, ["Equal", "Planner[B1]"])
    assert out == {"Equal": {"B1": {"depletion_round_mean": 14.5, "players_active_mean": 1.25}}}
