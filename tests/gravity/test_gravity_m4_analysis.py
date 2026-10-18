"""M4 post-freeze analysis (freeze checklist steps 4, 5, 6, 8) on synthetic stored runs.

A module-scoped synthetic world (see `m4_stored`) holds every run the analysis reads:
ground truth, 32 tuning runs, the pilot runs of the selected configurations, 70 main
runs, CC* and their robustness runs, all written by the production writers under the
frozen protocol. The robustness rows are designed so that B, the selected
configurations, n, P1, the contrasts and the criteria have known answers, which the tests
recompute independently (direct arithmetic, `ConstraintSet.satisfied_by` loops, scipy).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from m4_protocol import protocol_file
from m4_stored import (
    FAKE_COMMIT,
    GT_ENTRIES,
    NFEV_CORRECTED,
    NFEV_FAILED,
    NFEV_FREE,
    WORLD_LOG,
    feasible,
    write_groundtruth,
    write_method,
    write_robustness,
)
from m4_toy import toy_evaluate
from scipy.stats import norm, rankdata, spearmanr

from phasespace.gravity.m4 import cli
from phasespace.gravity.m4.analysis import (
    EliteDraws,
    EvidenceError,
    FeasiblePool,
    ProjectionInputs,
    StoredRun,
    achieved_power,
    average_ranks,
    dispersion,
    evidence_problems,
    lsoda_validation,
    main_seeds,
    projection,
    projection_after_pilot,
    run_analysis,
    run_calibration,
    run_pilot_analysis,
    run_selection,
    select_configurations,
)
from phasespace.gravity.m4.catalog import build_catalog
from phasespace.gravity.m4.config import (
    METHODS,
    PROTOCOL,
    ProtocolNotFrozenError,
    config_sha256,
)
from phasespace.gravity.m4.evaluation import Evaluation, cell_of, to_unit
from phasespace.gravity.m4.robustness import DrawResult, Execution, held_out_sets, tuning_sets
from phasespace.gravity.m4.runs import robustness_run_id

# --- pure functions ---------------------------------------------------------------------------


def _draw(uncorrected: bool, norm_m_s: float, executed: bool | None) -> DrawResult:
    nan2 = (math.nan, math.nan)
    return DrawResult(
        uncorrected_feasible=uncorrected,
        commanded_m_s=(norm_m_s, 0.0),
        executions=(Execution("gates_primary", (norm_m_s, 0.0), executed),),
        converged=False,
        diverged=executed is None,
        commanded_feasible=None,
        iterations=3,
        initial_residual_km=nan2,
        propagations=4,
        nfev=0,
    )


def test_stored_survival_matches_the_evaluator_rule() -> None:
    # The reference is DrawResult.survives (D25), applied to the values the robustness
    # command stores (|dv| in km/s, null for an execution not propagated).
    draws = [
        _draw(u, n, x)
        for u in (False, True)
        for n in (0.0, 0.5, 1.0, 2.999, 3.0, 3.001, 10.0, 29.9, 30.0, 64.0)
        for x in (False, True)
    ] + [_draw(False, 150.0, None), _draw(True, 150.0, None)]
    stored = EliteDraws(
        cell=(0, 0),
        index=0,
        singular=False,
        uncorrected=tuple(d.uncorrected_feasible for d in draws),
        commanded_km_s=tuple(d.commanded_m_s_norm / 1000.0 for d in draws),
        executed={"gates_primary": tuple(d.executions[0].feasible for d in draws)},
        converged=tuple(d.converged for d in draws),
        diverged=tuple(d.diverged for d in draws),
        nfev=tuple(0 for _ in draws),
    )
    for b in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0):
        assert stored.successes(b) == sum(d.survives(b) for d in draws)
        assert stored.survival(b) == pytest.approx(sum(d.survives(b) for d in draws) / len(draws))
    with pytest.raises(ValueError, match="defined"):
        stored.successes(101.0)


def test_feasible_pool_satisfiability_matches_the_constraint_predicate() -> None:
    rng = np.random.default_rng(5)
    evals: list[Evaluation] = []
    for _ in range(6):
        u0, u2 = float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.0, 0.99))
        evals.append(feasible(u0, u2, float(rng.uniform(-9.0, -4.0))))
    evals.append(toy_evaluate((0.05, 0.5, 0.5)))  # infeasible rows are ignored
    pool = FeasiblePool()
    pool.add(evals)
    sets = held_out_sets(confirm_frozen=True) + tuning_sets()
    expected = [any(c.satisfied_by(e) for e in evals) for c in sets]
    assert pool.satisfiable(sets) == expected
    assert 0 < sum(expected) < len(sets)  # the check is not vacuous
    assert FeasiblePool().satisfiable(sets[:3]) == [False] * 3


def test_average_ranks_match_scipy() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        values = [Fraction(int(v), 4) for v in rng.integers(0, 5, size=7)]
        expected = rankdata([-float(v) for v in values], method="average")
        assert [float(r) for r in average_ranks(values)] == expected.tolist()


def test_selection_rule_rank_sum_then_coverage_then_index() -> None:
    f = Fraction

    def scores(per_config: dict[int, tuple[Fraction, Fraction, Fraction]], method: str):
        return {(method, k, s): v for k, v in per_config.items() for s in (9101, 9102)}

    base = {(m, 0, s): (f(1), f(1), f(1)) for m in METHODS for s in (9101, 9102)}
    # MS: c1 better on P1, tied on P2 -> c1. MBH: rank sums all 4 -> coverage picks c2.
    # DE: everything tied -> the lower index c0. T: c0 best P1 but worst P2 (sum 1 + 3),
    # c1 second on both (2 + 2), c2 worst P1, best P2 (3 + 1): a three-way tie on 4, the
    # coverage of c0 and c1 tie, so the lower index c0 wins.
    all_scores = {
        **base,
        **scores({0: (f(1), f(1), f(1)), 1: (f(2), f(1), f(1))}, "MS"),
        **scores({0: (f(3), f(1), f(1)), 1: (f(2), f(2), f(2)), 2: (f(1), f(3), f(3))}, "MBH"),
        **scores({k: (f(1), f(1), f(1)) for k in range(3)}, "DE"),
        **scores({0: (f(3), f(1), f(2)), 1: (f(2), f(2), f(2)), 2: (f(1), f(3), f(1))}, "T"),
        **scores({0: (f(1), f(1), f(1)), 1: (f(1), f(1), f(1)), 2: (f(5), f(5), f(0))}, "T2"),
    }
    selected, rows = select_configurations(all_scores)
    assert selected == {"D": 0, "MS": 1, "MBH": 2, "DE": 0, "CC": 0, "T": 0, "T2": 2}
    mbh = [r for r in rows if r.method == "MBH"]
    assert [r.rank_sum for r in mbh] == [4, 4, 4]
    ms = [r for r in rows if r.method == "MS"]
    assert [r.rank_p2 for r in ms] == [Fraction(3, 2), Fraction(3, 2)]
    # Per-seed values are averaged over the tuning seeds: MS c0 scores 4 and 0 on P1, a
    # mean of 2 like c1, so everything ties and the lower index wins.
    split = {**all_scores, ("MS", 0, 9101): (f(4), f(1), f(1)), ("MS", 0, 9102): (f(0), f(1), f(1))}
    assert select_configurations(split)[0]["MS"] == 0


def test_dispersion_and_seed_count_follow_the_section_8_rule() -> None:
    seeds: tuple[int, ...] = (9101, 9102, 9201, 9202, 9203)
    zeros: dict[int, float] = dict.fromkeys(seeds, 0.0)
    p1: dict[str, dict[int, float]] = {m: dict(zeros) for m in METHODS}
    p2: dict[str, dict[int, float]] = {m: dict(zeros) for m in METHODS}
    # T - MS on P1 = (0, 0, 0, 0, 4.25): SD = 4.25 / sqrt(5); ratio = SD / 2.
    p1["T"] = {**zeros, 9203: 4.25}
    p1["MS"] = {**zeros, 9203: 0.0}
    p1["D"] = dict(p1["T"])  # T - D = 0
    p1["MBH"] = dict(p1["T"])
    p1["DE"] = dict(p1["T"])
    p1["CC"] = dict(p1["T"])
    rows = dispersion(p1, p2)
    assert [(x, ep) for x, ep, _, _ in rows] == [
        (x, ep) for x in ("D", "MS", "MBH", "DE", "CC") for ep in ("P1", "P2")
    ]
    sd = {(x, ep): s for x, ep, s, _ in rows}
    assert sd[("MS", "P1")] == pytest.approx(4.25 / math.sqrt(5.0), rel=1e-12)
    assert sd[("D", "P1")] == 0.0 and sd[("MS", "P2")] == 0.0
    from phasespace.gravity.m4.stats import seed_count

    ratios = [s / d for _, _, s, d in rows]
    # ceil(13.315201 * (4.25 / sqrt(5) / 2)^2) = ceil(12.0253...) = 13.
    assert seed_count(ratios) == (13, 13)
    assert math.ceil(13.315201 * (4.25 / math.sqrt(5.0) / 2.0) ** 2) == 13
    worst = max(ratios)
    assert achieved_power(16, worst) == pytest.approx(norm.cdf(4.0 / worst - 2.807), rel=1e-12)
    assert achieved_power(10, 0.0) == 1.0


def test_projection_reproduces_the_section_7_4_table() -> None:
    items = projection(ProjectionInputs())
    # The table: tuning robustness ~116 000, pilot ~76 000, main ~860 000, total ~4.22 M.
    assert items["tuning_robustness"] == pytest.approx(32 * 30 * (1.2 + 2.52 + 32 * 3.65))
    assert round(items["tuning_robustness"], -3) == 116_000
    assert round(items["pilot_robustness"], -3) == 76_000
    assert round(items["main_robustness"], -4) == 860_000
    assert items["main_search"] == 1_146_880 and items["pilot_search"] == 344_064
    assert items["tuning_search"] == 524_288
    assert round(items["total"], -4) == 4_220_000
    # "Each main seed beyond 10 adds ~0.20 M PE ... n = 16 gives ~5.4 M PE."
    more = projection(ProjectionInputs(main_seeds=11))
    assert round(more["total"] - items["total"], -4) == 200_000
    assert round(projection(ProjectionInputs(main_seeds=16))["total"], -5) == 5_400_000


def test_projection_after_pilot_applies_the_reductions_in_order() -> None:
    scenarios, applied, k4 = projection_after_pilot(3.65, 10)
    assert applied == [] and not k4 and len(scenarios) == 1
    assert scenarios[0][1]["total"] == pytest.approx(projection(ProjectionInputs())["total"])
    # A draw 3x dearer at n = 16 exceeds 7.8 M; K = 32 alone is not enough here.
    scenarios, applied, k4 = projection_after_pilot(3 * 3.65, 16)
    totals = [s[1]["total"] for s in scenarios]
    assert totals[0] > 7_800_000 and applied[0] == "k_50_to_32"
    assert totals == sorted(totals, reverse=True)
    assert k4 == (totals[-1] > 7_800_000)
    _, applied_huge, k4_huge = projection_after_pilot(100.0, 16)
    assert applied_huge == ["k_50_to_32", "drop_t2"] and k4_huge


def test_main_seeds_follow_the_count_rule() -> None:
    assert main_seeds(10) == tuple(range(1001, 1011))
    assert main_seeds(13) == (*range(1001, 1011), 1011, 1012, 1013)
    with pytest.raises(ValueError):
        main_seeds(17)


def test_evidence_problems_name_every_violation(tmp_path: Path) -> None:
    rid = write_method(tmp_path, stage="tune", method="D", config_index=0, seed=9101)
    manifest = json.loads((tmp_path / rid / "manifest.json").read_text())
    run = StoredRun(rid, tmp_path / rid, manifest, "x")
    assert evidence_problems(run) == []
    bad = json.loads(json.dumps(manifest))
    bad["source"]["git_dirty"] = True
    bad["m4"]["protocol_status"] = "draft"
    bad["m4"]["config_sha256"] = "0" * 64
    problems = evidence_problems(StoredRun(rid, tmp_path / rid, bad, "x"))
    assert len(problems) == 3
    assert any("git_dirty" in p for p in problems)
    assert any("draft" in p for p in problems)
    assert any("config_sha256" in p for p in problems)


# --- the synthetic world ------------------------------------------------------------------------

TUNE_FREE: dict[tuple[str, int], int] = {
    ("D", 0): 4,
    ("MS", 0): 4,
    ("MS", 1): 5,
    ("MBH", 0): 4,
    ("MBH", 1): 4,
    ("MBH", 2): 4,
    ("DE", 0): 4,
    ("DE", 1): 4,
    ("DE", 2): 6,
    ("CC", 0): 4,
    ("T", 0): 4,
    ("T", 1): 6,
    ("T", 2): 5,
    ("T2", 0): 4,
    ("T2", 1): 4,
    ("T2", 2): 5,
}
"""T_E draws feasible without correction per elite; 8 more succeed from B = 2.5 m/s on."""
TUNE_CORRECTED = 8
EXPECTED_CONFIGS = {"D": 0, "MS": 1, "MBH": 0, "DE": 2, "CC": 0, "T": 1, "T2": 2}
MAIN_FREE: dict[str, Callable[[int], int]] = {
    "D": lambda i: 5,
    "MS": lambda i: 10 + i,
    "MBH": lambda i: 12 + i,
    "DE": lambda i: 10 + i,
    "CC": lambda i: 10 + i,
    "T": lambda i: 10 + i,
    "T2": lambda i: 11 + i,
}
"""H_E draws (K = 50) feasible without correction for the first elite of main seed i;
elite r adds r, and 5 more draws succeed from B = 2.5 m/s on (primary set only)."""
MAIN_CORRECTED = 5
CCSTAR_FREE = 8
ELITES = 7
FAILING_GAMMA = -7.0
"""The fake LSODA validator fails elites with gamma_S < -7 deg (2 of the 7 cells)."""


def fake_validator(elite: Evaluation) -> tuple[bool, tuple[str, ...]]:
    ok = elite.return_flight_path_angle_deg >= FAILING_GAMMA
    return ok, () if ok else ("fake_lsoda_check",)


@dataclass(frozen=True)
class World:
    runs: Path
    calibration: dict[str, Any]
    selection: dict[str, Any]
    pilot: dict[str, Any]
    analysis: dict[str, Any]


def _main_spec(method: str, i: int) -> Callable[[int], tuple[int, int]]:
    return lambda r: (MAIN_FREE[method](i) + r, MAIN_CORRECTED)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    runs = tmp_path_factory.mktemp("runs")
    common: dict[str, Any] = {"git_commit": FAKE_COMMIT, "git_dirty": False}
    write_groundtruth(runs, GT_ENTRIES)
    for (method, k), free in TUNE_FREE.items():
        for seed in PROTOCOL.seeds.tuning:
            rid = write_method(runs, stage="tune", method=method, config_index=k, seed=seed)
            write_robustness(runs, rid, "T_E", lambda r, f=free: (f, TUNE_CORRECTED))
    calibration = run_calibration(runs, **common)
    selection = run_selection(runs, **common)
    configs: dict[str, int] = selection["m4"]["selection"]["configs"]
    for method in METHODS:
        free = TUNE_FREE[(method, configs[method])]
        for seed in PROTOCOL.seeds.pilot:
            rid = write_method(
                runs, stage="pilot", method=method, config_index=configs[method], seed=seed
            )
            write_robustness(runs, rid, "T_E", lambda r, f=free: (f, TUNE_CORRECTED))
    pilot = run_pilot_analysis(runs, **common)
    for method in METHODS:
        for i, seed in enumerate(main_seeds(pilot["m4"]["pilot"]["seed_count"])):
            rid = write_method(
                runs, stage="main", method=method, config_index=configs[method], seed=seed
            )
            first = (method, i) == ("D", 0)  # one re-check failure among 8 re-checks
            for errors in ("H_E_nominal", "H_E_wide"):
                write_robustness(
                    runs,
                    rid,
                    errors,
                    _main_spec(method, i),
                    rechecks=8 if first else None,
                    recheck_failures=1 if first else 0,
                )
    cc = write_method(
        runs,
        stage="reference",
        method="CCstar",
        config_index=0,
        seed=0,
        run_id="gravity-m4-ccstar-v1",
    )
    for errors in ("H_E_nominal", "H_E_wide"):
        write_robustness(runs, cc, errors, lambda r: (CCSTAR_FREE, MAIN_CORRECTED))
    analysis = run_analysis(runs, confirm_frozen=True, validate=fake_validator, **common)
    return World(runs, calibration, selection, pilot, analysis)


def _table(world: World, run_id: str, path: str) -> list[dict[str, Any]]:
    table = json.loads((world.runs / run_id / path).read_text())
    names = [c["name"] for c in table["columns"]]
    return [dict(zip(names, row, strict=True)) for row in table["rows"]]


def _results(world: World, **match: Any) -> list[dict[str, Any]]:
    rows = _table(world, "gravity-m4-analysis-v1", "results.json")
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def _one(world: World, **match: Any) -> dict[str, Any]:
    (row,) = _results(world, **match)
    return row


def test_calibration_picks_the_smallest_b_in_band(world: World) -> None:
    m4 = world.calibration["m4"]
    # D elites: 4 of 32 draws free (s = 0.125 at B = 1), 12 of 32 from B = 2.5 (0.375).
    assert m4["calibration"]["budget_m_s"] == 3.0
    assert m4["calibration"]["medians"] == {"1": 0.125, "3": 0.375, "10": 0.375, "30": 0.375}
    assert m4["stage"] == "calibration" and m4["protocol_status"] == "frozen"
    assert m4["config_sha256"] == config_sha256()
    assert world.calibration["source"]["git_dirty"] is False
    assert {i["run_id"] for i in m4["inputs"]} == {
        "gravity-m4-tune-d-c0-s9101-v1",
        "gravity-m4-tune-d-c0-s9102-v1",
        robustness_run_id("gravity-m4-tune-d-c0-s9101-v1", "T_E"),
        robustness_run_id("gravity-m4-tune-d-c0-s9102-v1", "T_E"),
    }
    rows = _table(world, "gravity-m4-calibration-v1", "calibration.json")
    assert [r["selected"] for r in rows] == [0, 1, 0, 0]
    assert all(r["elites"] == 2 * ELITES for r in rows)


def test_selection_picks_the_designed_configurations(world: World) -> None:
    block = world.selection["m4"]["selection"]
    assert block["configs"] == EXPECTED_CONFIGS
    assert block["configs_argument"] == "MS=1,MBH=0,DE=2,T=1,T2=2"
    assert block["budget_m_s"] == 3.0
    # Satisfiable tuning sets: reference loop over GT (a)-(c) and the tuning runs' finds.
    pool = [e.evaluation for e in GT_ENTRIES] + [e for e in WORLD_LOG if e.feasible]
    expected = [[c.family, c.index] for c in tuning_sets() if any(c.satisfied_by(e) for e in pool)]
    assert block["satisfiable_tuning_sets"] == expected and 0 < len(expected) < 30
    scores = _table(world, "gravity-m4-selection-v1", "tuning_scores.json")
    assert len(scores) == 32
    ms1 = next(r for r in scores if (r["method"], r["config_index"]) == ("MS", 1))
    assert ms1["p1"] == pytest.approx(ELITES * (5 + 8) / 32)  # P1 at B = 3 m/s
    # Same catalog everywhere: G_tune = 7 GT cells + (4, 9) from the runs; 7 of 8 covered.
    assert ms1["coverage_g_tune"] == pytest.approx(7 / 8)
    elites = list(build_catalog(WORLD_LOG).values())
    kept = [c for c in tuning_sets() if [c.family, c.index] in expected]
    hits = sum(any(c.satisfied_by(WORLD_LOG[el.index]) for el in elites) for c in kept)
    assert ms1["p2"] == pytest.approx(hits / len(kept))


def test_pilot_analysis_records_n_k3_and_the_projection(world: World) -> None:
    pilot = world.pilot["m4"]["pilot"]
    # Every selected configuration has the same P1 on all 5 seeds: no dispersion, n = 10.
    assert pilot["seed_count"] == 10 and pilot["seed_count_unclamped"] == 0
    assert pilot["main_seeds"] == list(range(1001, 1011))
    assert pilot["k3_triggered"] is False
    assert pilot["pilot_median_survival"]["DE"] == pytest.approx((6 + 8) / 32)
    # PE per draw: every T_E draw of the tuning and pilot stages, read from the raw tables.
    nfev = draws = 0
    for path in world.runs.glob("gravity-m4-*-robustness-t-e/robustness.json"):
        table = json.loads(path.read_text())
        nfev += sum(row[-1] for row in table["rows"])
        draws += len(table["rows"])
    assert draws == (32 + 21) * ELITES * 32
    assert pilot["pe_per_draw_t_e"] == pytest.approx(nfev / draws / 4106, rel=1e-12)
    assert pilot["k4_triggered"] is False and pilot["reductions"] == []
    expected = projection_after_pilot(nfev / draws / 4106, 10)[0][-1][1]["total"]
    assert pilot["projection_total_pe"] == pytest.approx(expected)
    scores = _table(world, "gravity-m4-pilot-analysis-v1", "pilot_scores.json")
    assert len(scores) == 7 * 5


def _p1(method: str, i: int, budget: float = 3.0) -> float:
    corrected = MAIN_CORRECTED if budget >= 2.5 else 0
    return sum(MAIN_FREE[method](i) + r + corrected for r in range(ELITES)) / 50


def test_analysis_per_run_metrics(world: World) -> None:
    rows = _table(world, "gravity-m4-analysis-v1", "per_run.json")
    assert len(rows) == 7 * 10 + 1
    gt = [e.evaluation for e in GT_ENTRIES] + [e for e in WORLD_LOG if e.feasible]
    g20 = {cell_of(e, 20) for e in gt}
    cat20 = {cell_of(e, 20) for e in WORLD_LOG if e.feasible}
    sets = held_out_sets(confirm_frozen=True)
    kept = [c for c in sets if any(c.satisfied_by(e) for e in gt)]
    elites = [WORLD_LOG[el.index] for el in build_catalog(WORLD_LOG).values()]
    p2 = sum(any(c.satisfied_by(e) for e in elites) for c in kept) / len(kept)
    for row in rows:
        if row["method"] == "CCstar":
            assert row["seed"] is None
            assert row["p1"] == pytest.approx(ELITES * (CCSTAR_FREE + MAIN_CORRECTED) / 50)
            continue
        i = row["seed"] - 1001
        method = row["method"]
        assert row["p1"] == pytest.approx(_p1(method, i))
        assert row["p1_b0"] == pytest.approx(_p1(method, i, 0.0))
        assert row["p1_b1"] == pytest.approx(_p1(method, i, 1.0))
        assert row["p1_b30"] == pytest.approx(_p1(method, i, 30.0))
        assert row["p1_stress"] == pytest.approx(_p1(method, i, 0.0))  # stress: no corrected
        assert row["p1_wide"] == pytest.approx(row["p1"])
        assert row["mean_s"] == pytest.approx(row["p1"] / ELITES)
        assert row["min_s"] == pytest.approx((MAIN_FREE[method](i) + MAIN_CORRECTED) / 50)
        assert row["coverage"] == pytest.approx(7 / 8)
        assert row["coverage_20"] == pytest.approx(len(cat20 & g20) / len(g20))
        assert row["p2"] == pytest.approx(p2)
        assert row["elites"] == ELITES and row["diversity_classes"] == 1
        assert row["quality_mean_e"] == pytest.approx(np.mean([0.0, 0.1, 0.3, 0.5, 0.5, 1.5, 1.0]))
        assert row["feasibility_yield"] == pytest.approx(9 / 16)
        assert row["first_feasible_index"] == 1 and row["gate_failures"] == 0
        assert row["commanded_fraction"] == pytest.approx(MAIN_CORRECTED * ELITES / (50 * ELITES))
        assert row["median_commanded_km_s"] == 0.0
    anytime = _table(world, "gravity-m4-analysis-v1", "anytime.json")
    t = [r for r in anytime if r["method"] == "T" and r["seed"] == 1001]
    assert [r["checkpoint"] for r in t] == [2, 4, 8, 16]
    # The first 2 propagations hold one feasible point, cell (3, 2): 1 of 8 cells.
    assert t[0]["coverage"] == pytest.approx(1 / 8) and t[-1]["coverage"] == pytest.approx(7 / 8)


def test_analysis_primary_contrasts_and_holm(world: World) -> None:
    rows = {(r["metric"], r["baseline"]): r for r in _results(world, section="primary")}
    assert len(rows) == 10
    d = [_p1("T", i) - _p1("D", i) for i in range(10)]
    t_d = rows[("P1", "D")]
    assert t_d["value"] == pytest.approx(np.mean(d))
    # All ten differences positive: only the identity and the all-minus flip reach |mean|.
    assert t_d["p_value"] == pytest.approx(2 / 1024)
    assert t_d["p_holm"] == pytest.approx(10 * 2 / 1024) and t_d["significant"] == 1
    idx = np.random.default_rng(4242).integers(0, 10, size=(10_000, 10))
    lo, hi = np.quantile(np.asarray(d)[idx].mean(axis=1), [0.025, 0.975])
    assert (t_d["ci_low"], t_d["ci_high"]) == (pytest.approx(lo), pytest.approx(hi))
    for x in ("MS", "DE", "CC"):  # identical P1: every flip ties
        assert rows[("P1", x)]["p_value"] == 1.0 and rows[("P1", x)]["value"] == 0.0
    assert rows[("P1", "MBH")]["value"] == pytest.approx(-ELITES * 2 / 50)
    assert rows[("P1", "MBH")]["significant"] == 0  # significant, but in favour of MBH
    assert all(rows[("P2", x)]["value"] == 0.0 for x in ("D", "MS", "MBH", "DE", "CC"))
    status = {r["baseline"]: r["note"] for r in _results(world, section="h1")}
    assert status == {
        "D": "not_supported",  # better on P1 only
        "MS": "refuted",
        "MBH": "refuted",
        "DE": "refuted",
        "CC": "refuted",
    }


def test_analysis_criteria_ground_truth_and_lsoda(world: World) -> None:
    counts = {r["statistic"]: r["value"] for r in _results(world, metric="cells_10")}
    assert counts == {"prefix": 4, "map": 5, "a_c": 7, "a_d": 8}
    k = {(r["metric"], r["baseline"]): r["value"] for r in _results(world, section="criterion")}
    assert k[("K7", None)] == 1  # 8 > 1.1 x 4
    assert k[("K5", "CC")] == 1 and k[("K5", "D")] == 0  # CC equals T on everything
    assert k[("K6", None)] == 0  # T beats D on P1
    assert k[("K2", None)] == 0
    assert k[("K2_recheck_fraction", None)] == pytest.approx(1 / (69 * ELITES + 8))
    assert world.analysis["m4"]["analysis"]["criteria"]["K5"] == {"D": False, "CC": True}
    pooled = 7 * 10 * ELITES
    sample = sorted(
        np.random.default_rng(7301)
        .choice(pooled, size=math.ceil(0.05 * pooled), replace=False)
        .tolist()
    )
    lsoda = _table(world, "gravity-m4-analysis-v1", "lsoda.json")
    assert [r["pool_index"] for r in lsoda] == sample
    # Pooled order (method, seed, cell): pool index i is cell rank i % 7 of method i // 70.
    failing = {5, 6}  # cells (7, 6) and (9, 8) have gamma_S -8 and -7.5 deg
    assert [r["passed"] for r in lsoda] == [int(i % ELITES not in failing) for i in sample]
    assert [r["method"] for r in lsoda] == [METHODS[i // 70] for i in sample]
    assert world.analysis["metrics"]["lsoda_failures"]["value"] == sum(
        i % ELITES in failing for i in sample
    )


def test_analysis_h2_and_secondary(world: World) -> None:
    h2 = _one(world, section="h2", metric="spearman_rho_s_margin")
    s: list[float] = []
    margins: list[float] = []
    for i in range(10):
        s += [(MAIN_FREE["T"](i) + 0 + 5) / 50, (MAIN_FREE["T"](i) + 2 + 5) / 50]
        margins += [1.5, 0.2]  # cells (1, 4) and (3, 2), ranks 0 and 2
    assert h2["value"] == pytest.approx(spearmanr(s, margins).statistic)
    assert h2["note"] == "20 elites" and 0.0 < h2["p_value"] <= 1.0
    t2 = _one(world, section="secondary", metric="p1", arm="T2", baseline="T")
    assert t2["value"] == pytest.approx(ELITES / 50) and t2["p_value"] == pytest.approx(2 / 1024)
    assert t2["p_holm"] is None
    stress = _one(world, section="secondary", metric="p1_stress_minus_p1", arm="T")
    assert stress["value"] == pytest.approx(-ELITES * MAIN_CORRECTED / 50)
    cc = _one(world, section="cc_star", metric="p1", baseline="D")
    assert cc["value"] == pytest.approx(ELITES * (CCSTAR_FREE + 5) / 50 - _p1("D", 0))
    worst = _one(world, section="summary", metric="p1", arm="T", statistic="worst_seed")
    assert worst["value"] == pytest.approx(_p1("T", 0)) and worst["note"] == "seed 1001"


def test_analysis_never_overwrites_and_needs_confirmation(world: World) -> None:
    with pytest.raises(FileExistsError):
        run_calibration(world.runs, git_commit=FAKE_COMMIT, git_dirty=False)
    with pytest.raises(FileExistsError):
        run_analysis(world.runs, git_commit=FAKE_COMMIT, git_dirty=False, confirm_frozen=True)
    with pytest.raises(ProtocolNotFrozenError):
        run_analysis(
            world.runs,
            git_commit=FAKE_COMMIT,
            git_dirty=False,
            confirm_frozen=False,
            run_id="other",
        )


def test_cli_runs_a_step_and_the_guard_holds(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--runs-dir", str(world.runs), "calibrate", "--run-id", "cal-cli"]
    assert cli.main([*argv, "--confirm-frozen"]) == 0
    assert "B = 3 m/s" in capsys.readouterr().out
    written = json.loads((world.runs / "cal-cli" / "manifest.json").read_text())
    assert written["m4"]["calibration"]["budget_m_s"] == 3.0
    for command in ("calibrate", "select", "pilot-analysis", "analysis"):
        base = ["--runs-dir", str(world.runs), command, "--run-id", f"refused-{command}"]
        with pytest.raises(SystemExit):
            cli.main(base)
        assert "confirm-frozen" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            cli.main([*base, "--confirm-frozen"], protocol_path=protocol_file(tmp_path, "draft"))
        assert "draft" in capsys.readouterr().err
        assert not (world.runs / f"refused-{command}").exists()
    missing = ["--runs-dir", str(tmp_path / "empty"), "select", "--confirm-frozen"]
    with pytest.raises(SystemExit):
        cli.main(missing)
    assert "no such run" in capsys.readouterr().err


def test_dirty_or_inconsistent_inputs_are_refused(tmp_path: Path) -> None:
    for seed in PROTOCOL.seeds.tuning:
        rid = write_method(
            tmp_path, stage="tune", method="D", config_index=0, seed=seed, git_dirty=seed == 9101
        )
        write_robustness(tmp_path, rid, "T_E", lambda r: (4, 8))
    with pytest.raises(EvidenceError, match="git_dirty"):
        run_calibration(tmp_path, git_commit=FAKE_COMMIT, git_dirty=False)
    # A robustness run whose elite is not a catalog candidate of its source is refused.
    other = tmp_path / "other"
    for seed in PROTOCOL.seeds.tuning:
        rid = write_method(other, stage="tune", method="D", config_index=0, seed=seed)
        shifted = [WORLD_LOG[0], *WORLD_LOG[2:]] if seed == 9102 else list(WORLD_LOG)
        write_robustness(other, rid, "T_E", lambda r: (4, 8), log=shifted)
    with pytest.raises(EvidenceError, match="not a feasible propagation"):
        run_calibration(other, git_commit=FAKE_COMMIT, git_dirty=False)
    assert not (other / "gravity-m4-calibration-v1").exists()


def test_lsoda_validation_passes_c1() -> None:
    from phasespace.gravity.m4.evaluation import evaluate_unit

    c1 = evaluate_unit(to_unit((229.14224099316039, 3.1567704135102614, 0.0)))
    passed, failed = lsoda_validation(c1)
    assert passed and failed == ()


def test_held_out_robustness_needs_the_frozen_protocol(world: World, tmp_path: Path) -> None:
    from phasespace.gravity.m4.analysis import load_robustness, load_stage_run

    src = load_stage_run(world.runs, "main", "T", 1, 1001)
    draft = protocol_file(tmp_path, "draft")
    with pytest.raises(ProtocolNotFrozenError, match="draft"):
        load_robustness(
            world.runs,
            src,
            src.run.run_id,
            "H_E_nominal",
            confirm_frozen=True,
            protocol_path=draft,
        )
    with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
        load_robustness(world.runs, src, src.run.run_id, "H_E_nominal")
    rob = load_robustness(world.runs, src, src.run.run_id, "H_E_nominal", confirm_frozen=True)
    assert len(rob.elites) == ELITES and rob.errors.draws == 50
    assert rob.p1(3.0) == Fraction(round(_p1("T", 0) * 50), 50)
    assert NFEV_FREE != NFEV_CORRECTED != NFEV_FAILED
