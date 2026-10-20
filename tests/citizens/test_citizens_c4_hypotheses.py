"""C4 hypothesis stage (protocol §9): p-values, Holm, the seed-level tests, families and
verdicts on synthetic matrices with known answers. No simulation, no fitted model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy import stats

from phasespace.citizens.rl import c4_config, crossplay, hypotheses
from phasespace.citizens.rl.citizens import TRAINABLE_MODELS

DELTA = c4_config.DELTA
GAMES = 200
RESAMPLES = 1000


# --- Holm and the p-values ----------------------------------------------------------------------


def test_holm_against_a_hand_computed_example() -> None:
    """m = 4, alpha = 0.05; sorted p: 0.005 <= 0.0125 (reject), 0.01 <= 0.05/3 = 0.0167
    (reject), 0.03 > 0.025 (stop: 0.03 and 0.04 are not rejected). Adjusted: 4 x 0.005 =
    0.02, max(0.02, 3 x 0.01) = 0.03, max(0.03, 2 x 0.03) = 0.06, max(0.06, 0.04) = 0.06."""
    out = hypotheses.holm([0.01, 0.04, 0.03, 0.005], 0.05)
    assert out["reject"] == [True, False, False, True]
    assert out["adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_rejects_all_and_never_a_null() -> None:
    assert hypotheses.holm([0.01, 0.02, 0.012], 0.05)["reject"] == [True, True, True]
    # 0.02 > 0.05 / 3 once a null counts in m: nothing is rejected
    out = hypotheses.holm([0.02, None, 0.03], 0.05)
    assert out["reject"] == [False, False, False] and out["adjusted"][1] is None
    assert hypotheses.holm([0.001, None], 0.05)["reject"] == [True, False]


def test_bootstrap_p_values_of_a_point_mass() -> None:
    """Every resample equals 5: p(D > 4) = 1 / (B + 1), p(D > 6) = 1, p(D < 6) = 1/(B + 1)."""
    point = np.full(99, 5.0)
    assert hypotheses.bootstrap_p(point, 4.0, "greater") == pytest.approx(0.01)
    assert hypotheses.bootstrap_p(point, 6.0, "greater") == pytest.approx(1.0)
    assert hypotheses.bootstrap_p(point, 6.0, "less") == pytest.approx(0.01)
    assert hypotheses.bootstrap_p(point, 5.0, "less") == pytest.approx(1.0)


def test_crossed_bootstrap_of_a_constant_shift_gives_the_point_mass() -> None:
    row = np.random.default_rng(3).normal(1000.0, 30.0, (1, 50))
    same = np.repeat(row, 3, axis=0)
    stats_ = crossplay.crossed_bootstrap_stats(same + 20.0, same[:1], 200, 4)
    assert np.allclose(stats_, 20.0)
    assert hypotheses.bootstrap_p(stats_, DELTA, "greater") == pytest.approx(1 / 201)


CLAIMS: tuple[crossplay.Claim, ...] = ("less", "greater")


def test_seed_level_t_against_scipy() -> None:
    rng = np.random.default_rng(1)
    games = rng.normal(1000.0, 100.0, 300)
    a = games[None, :] + rng.normal(30.0, 10.0, (5, 1)) + rng.normal(0.0, 5.0, (5, 300))
    rule = games[None, :]
    d = (a - rule).mean(axis=1)
    for claim in CLAIMS:
        got = crossplay.seed_level_t(a, rule, DELTA, claim)
        want = stats.ttest_1samp(d, popmean=DELTA, alternative=claim)
        assert got["p"] == pytest.approx(float(want.pvalue), rel=1e-10)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        assert got["df"] == 4
    b = games[None, :] + rng.normal(0.0, 20.0, (5, 1))
    for claim in CLAIMS:
        got = crossplay.seed_level_t(a, b, -DELTA, claim)
        want = stats.ttest_ind(
            a.mean(axis=1) + DELTA, b.mean(axis=1), equal_var=False, alternative=claim
        )
        assert got["p"] == pytest.approx(float(want.pvalue), rel=1e-10)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    interval = crossplay.seed_level_interval(a, rule)["interval"]
    # independent reference: the t quantile from SciPy, the mean and SE by hand
    q = float(stats.t.ppf(0.975, 4))
    se = float(d.std(ddof=1) / np.sqrt(5))
    assert interval == pytest.approx([d.mean() - q * se, d.mean() + q * se], rel=1e-12)


def test_seed_level_t_edge_cases() -> None:
    games = np.arange(900.0, 920.0)  # integers: every d_s is exactly 30
    same = np.repeat(games[None, :], 3, axis=0)
    exact = crossplay.seed_level_t(same + 30.0, games[None, :], DELTA, "greater")
    assert exact["p"] == 0.0 and exact["se"] == 0.0
    assert crossplay.seed_level_t(same + 30.0, games[None, :], DELTA, "less")["p"] == 1.0
    single = crossplay.seed_level_t(same[:1] + 30.0, games[None, :], DELTA, "greater")
    assert single["p"] is None and "fewer than 2" in single["null_reason"]


def test_a_cell_is_confirmed_only_if_both_tests_reject() -> None:
    """One seed: the bootstrap rejects (a 100-coin gain with game noise 1) but the
    seed-level test cannot exist, so the cell is not confirmed; without the seed-level
    test (the B4 design) the bootstrap alone confirms it."""
    rng = np.random.default_rng(2)
    games = rng.normal(1000.0, 100.0, GAMES)
    a = (games + 100.0 + rng.normal(0.0, 1.0, GAMES))[None, :]
    cell = hypotheses.Cell("P@Y vs Q", "P", "Y", "Q", a, games[None, :])
    cache = hypotheses.BootstrapCache(RESAMPLES, 5)
    both = hypotheses.run_family("f", [cell], "greater", DELTA, cache)
    assert both["cells"][0]["reject_bootstrap"] and not both["cells"][0]["reject_t"]
    assert both["confirmed"] == []
    alone = hypotheses.run_family("f", [cell], "greater", DELTA, cache, seed_level=False)
    assert alone["confirmed"] == ["P@Y vs Q"]
    held = hypotheses.run_family(
        "f", [cell], "greater", DELTA, cache, seed_level=False, exploratory={"P@Y vs Q": "few"}
    )
    assert held["confirmed"] == [] and held["cells"][0]["exploratory"] == "few"


# --- synthetic matrices --------------------------------------------------------------------------

type Effect = Callable[[str, str], float]


def _series(
    effect: Effect, noret_effect: Effect | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    """Every column has common game values g_j (the k*[Y] rule's totals); a planner of
    row X with seed s scores g_j + effect(X, Y) + a seed offset (SD 3) + game noise (SD 2)."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for y in TRAINABLE_MODELS:
        g = rng.normal(1000.0, 200.0, GAMES)
        for name in (crossplay.k_star_name(y), c4_config.K_PUBLISHED_RULE):
            out.append(
                {"policy": name, "column": y, "mode": "deterministic", "seed": None,
                 "total_surplus": g.tolist()}
            )  # fmt: skip
        noret = noret_effect or effect
        rows: list[tuple[str, tuple[int, ...], float]] = [
            (crossplay.planner_name(x, True), c4_config.MAIN_SEEDS, effect(x, y))
            for x in TRAINABLE_MODELS
        ]
        rows += [
            (crossplay.planner_name(x, False), c4_config.NO_RETENTION_SEEDS, noret(x, y))
            for x in TRAINABLE_MODELS
        ]
        rows.append(("Planner[Mix]", c4_config.MIXTURE_SEEDS, effect("Mix", y)))
        for name, seeds, shift in rows:
            for mode in crossplay.MODES:
                for s in seeds:
                    v = g + shift + rng.normal(0.0, 3.0) + rng.normal(0.0, 2.0, GAMES)
                    out.append(
                        {"policy": name, "column": y, "mode": mode, "seed": s,
                         "total_surplus": v.tolist()}
                    )  # fmt: skip
    return out


def _summary(tv: Callable[[str, str], float] | None = None) -> dict[str, Any]:
    """Diagnostics: B1 TV 0.2 everywhere (k = 22: 0.2) unless `tv` says otherwise; input
    support 0.01 everywhere."""
    names = [
        *(crossplay.planner_name(x, r) for x in TRAINABLE_MODELS for r in (True, False)),
        "Planner[Mix]",
        c4_config.K_PUBLISHED_RULE,
    ]
    cells: dict[str, Any] = {}
    for name in names:
        cells[name] = {}
        for y in TRAINABLE_MODELS:
            value = 0.2 if tv is None else tv(name, y)
            kind = "input_out_of_range" if y == "B2" else "state_novelty"
            support = None if y == "B1" else {
                "kind": kind, "label": kind, "outside_range_any": 0.01, "off_grid": 0.01,
            }  # fmt: skip
            entry = {"b1_disagreement": [{"mean_tv": value}], "input_support": [support]}
            cells[name][y] = {"deterministic": entry, "sampled": entry}
    return {"cells": cells}


def _analyse(series: list[dict[str, Any]], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    return hypotheses.analyse(
        series, summary or _summary(), None, {}, n_resamples=RESAMPLES, seed=11
    )


def test_transfer_everywhere_supports_h1_and_refutes_h2() -> None:
    out = _analyse(_series(lambda x, y: 100.0))
    det = out["modes"]["deterministic"]["main"]
    assert out["verdicts"]["H1"] == "supported"
    assert out["verdicts"]["H2"] == "refuted"
    assert det["families"]["H1-reversal"]["confirmed"] == []
    assert len(det["families"]["H1-transfer"]["confirmed"]) == 12
    assert det["H2b"]["confirmed_losses"] == []
    assert all(r["own_model_gain"] and not r["optimiser_limited"] for r in det["rows"].values())
    assert not any(r["possible_exploitation"] for r in det["rows"].values())
    assert out["sampled_policy_disagreement"] == {}
    cell = det["families"]["H1-transfer"]["cells"][0]
    assert cell["seeds_above_theta"] == 5 and cell["seeds_below_theta"] == 0
    # S2: the mixture row transfers too
    s2 = out["modes"]["deterministic"]["S2"]
    assert s2["verdict"]["verdict"] == "supported"


def test_one_reversal_refutes_h1_and_supports_h2() -> None:
    def effect(x: str, y: str) -> float:
        return -100.0 if (x, y) == ("B2", "B1") else 100.0

    out = _analyse(_series(effect))
    det = out["modes"]["deterministic"]["main"]
    assert out["verdicts"]["H1"] == "refuted"
    assert det["H1"]["confirmed_reversals"] == ["Planner[B2]@B1 vs Interpolating k*[B1]"]
    assert out["verdicts"]["H2"] == "supported"
    assert det["H2"]["confirmed_losses"] == ["Planner[B2]@B1 vs Interpolating k*[B1]"]
    # H2b: Planner[B2]@B1 is 200 coins below Planner[B1]@B1
    assert out["verdicts"]["H2b_confirmed_losses"] == ["Planner[B2]@B1 vs Planner[B1]"]
    regret = next(r for r in det["regret"] if r["other"] == "Planner[B2]" and r["column"] == "B1")
    assert regret["regret"] == pytest.approx(200.0, abs=10.0)
    assert regret["ci"][0] < regret["regret"] < regret["ci"][1]


def test_effects_within_delta_leave_h1_not_refuted_and_h2_inconclusive() -> None:
    """Planners equal k* except on B1, where the H2 planners sit at -16 (the boundary):
    neither a transfer nor a reversal is confirmed, and on B1 neither the loss nor the
    non-inferiority."""

    def effect(x: str, y: str) -> float:
        return -DELTA if (y == "B1" and x != "B1") else 0.0

    out = _analyse(_series(effect))
    det = out["modes"]["deterministic"]["main"]
    assert out["verdicts"]["H1"] == "not refuted"
    assert len(det["H1"]["transfer_not_shown"]) == 12
    assert out["verdicts"]["H2"] == "inconclusive"
    assert det["H2"]["confirmed_losses"] == [] and det["H2"]["confirmed_noninferior"] == []


def test_optimiser_limited_and_possible_exploitation_labels() -> None:
    def effect(x: str, y: str) -> float:
        return -100.0 if (x, y) == ("B3", "B3") else 100.0

    def tv(name: str, column: str) -> float:
        return 0.4 if (name, column) == ("Planner[B2]", "B2") else 0.2

    out = _analyse(_series(effect), _summary(tv))
    det = out["modes"]["deterministic"]["main"]
    assert det["rows"]["Planner[B3]"]["optimiser_limited"]
    assert det["rows"]["Planner[B2]"]["possible_exploitation"]  # TV ratio 2 > 1.5
    assert not det["rows"]["Planner[B1]"]["possible_exploitation"]
    assert det["flags"]["Planner[B2]@B2"]["tv_ratio"] == pytest.approx(2.0)


def test_support_flag_enters_only_in_the_b2_column() -> None:
    summary = _summary()
    for column, name in (("B2", "Planner[B1]"), ("B3", "Planner[B1]")):
        summary["cells"][name][column]["deterministic"]["input_support"] = [
            {"kind": "k", "label": "l", "outside_range_any": 0.2, "off_grid": 0.01}
        ]
    b2 = hypotheses.exploitation_flags(summary, "Planner[B1]", "B2")
    b3 = hypotheses.exploitation_flags(summary, "Planner[B1]", "B3")
    assert b2["support_flag"] and b2["flagged"]
    assert not b3["support_flag"] and not b3["flagged"]
    assert b3["input_support"]["outside_range_any"] == pytest.approx(0.2)


def test_s1_and_retention_attribution() -> None:
    def main(x: str, y: str) -> float:
        return -100.0 if (x, y) == ("B2", "B1") else 100.0

    out = _analyse(_series(main, noret_effect=lambda x, y: 100.0))
    s1 = out["modes"]["deterministic"]["S1"]
    assert s1["families"]["S1-reversal"]["confirmed"] == []
    assert s1["H1"]["verdict"] == "supported"
    att = hypotheses.retention_attribution(1, 0, 0.3, 0.2)
    assert att["attributable_to_retention"] and att["by_h1_reversal"] and not att["by_w_l1"]
    assert hypotheses.retention_attribution(0, 0, 0.3, 0.1)["by_w_l1"]
    assert not hypotheses.retention_attribution(0, 0, 0.3, 0.2)["attributable_to_retention"]
    assert not hypotheses.retention_attribution(1, 1, None, None)["attributable_to_retention"]


def test_convergence_labels() -> None:
    labels = hypotheses.convergence_labels(
        {
            "a": {"plateau_iteration": 900, "tail_slope_per_100_iterations": 16.0},
            "b": {"plateau_iteration": 910, "tail_slope_per_100_iterations": 0.0},
            "c": {"plateau_iteration": 500, "tail_slope_per_100_iterations": 16.5},
        }
    )
    assert [labels[k]["not_converged"] for k in "abc"] == [False, True, True]


# --- mechanism agreement --------------------------------------------------------------------------


def _distilled(model: str, seed: int, w: list[float], r2: float = 0.9) -> dict[str, Any]:
    return {
        "policy": crossplay.planner_name(model, True),
        "seed": seed,
        "w_rule": {
            "w": w,
            "kept_share": [0.0] * len(w),
            "informative_rows": [500] * (len(w) - 1) + [10],
            "r2": r2,
        },
        "interpolating_k": {"k": 9.0, "r2": 0.8},
        "tree": {"r2": 0.95},
        "total_surplus_mean": {"k_star": 9.0, "recovery_of_planner_gain": {"value": 0.5}},
        "kept_share_coin_weighted": {"endgame": 0.1},
        "method_bias_ln_k": {"value": 0.05},
    }


def test_mechanism_agreement_same_different_and_mixed() -> None:
    base = [0.1 * i for i in range(10)]
    same = [
        _distilled(m, s, [v + 0.01 * s for v in base]) for m in TRAINABLE_MODELS for s in (1, 2)
    ]
    out = hypotheses.mechanism_agreement(same, True)
    assert out["summary"] == "same mechanism"
    pair = out["pairs"][0]
    assert pair["bins"] == list(range(9))  # the last bin has 10 rows < 200
    assert pair["l1"] == pytest.approx(0.0, abs=1e-12)
    assert pair["within"]["B1"] == pytest.approx(0.01)
    assert pair["threshold"] == pytest.approx(0.10)
    shifted = [
        _distilled(m, s, [v + (0.5 if m == "B3" else 0.0) for v in base])
        for m in TRAINABLE_MODELS
        for s in (1, 2)
    ]
    diff = hypotheses.mechanism_agreement(shifted, True)
    assert diff["summary"] == "different mechanisms"
    labels = {tuple(p["pair"]): p["label"] for p in diff["pairs"]}
    assert labels[("B1", "B3")] == "different" and labels[("B1", "B2")] == "same w(R)"
    low = [
        _distilled(m, s, list(base), r2=0.3 if m == "B3" else 0.9)
        for m in TRAINABLE_MODELS
        for s in (1, 2)
    ]
    assert hypotheses.mechanism_agreement(low, True)["summary"] == "mixed"
    assert hypotheses.mechanism_agreement(low, True)["fidelity"]["B3"]["low_fidelity"]


# --- H2-B4 -----------------------------------------------------------------------------------------


def _b4_row(name: str, mean: float, n: int = 40, seed: int = 0) -> hypotheses.B4Row:
    rng = np.random.default_rng(seed)
    total = rng.normal(mean, 50.0, n)
    ones = np.ones(n)
    return hypotheses.B4Row(name, total, 0.1 * ones, 30.0 * ones, 3.0 * ones, 0)


def _b4_rows(b2_mean: float, n_b3: int = 40) -> dict[str, hypotheses.B4Row]:
    k = crossplay.k_star_name("B1")
    return {
        k: _b4_row(k, 1200.0, seed=1),
        "Planner[B1]": _b4_row("Planner[B1]", 1200.0, seed=2),
        "Planner[B2]": _b4_row("Planner[B2]", b2_mean, seed=3),
        "Planner[B3]": _b4_row("Planner[B3]", 1200.0, n=n_b3, seed=4),
        "Planner[B23]": _b4_row("Planner[B23]", 1200.0, seed=5),
    }


def test_h2_b4_confirmatory_only_when_validated() -> None:
    cache = hypotheses.BootstrapCache(RESAMPLES, 3)
    out = hypotheses.b4_hypotheses(_b4_rows(800.0), True, "plain", cache)
    assert out["H2-B4"]["verdict"] == "supported"
    assert out["families"]["B4-loss"]["confirmed"] == [
        "Planner[B2]@B4-plain vs Interpolating k*[B1]"
    ]
    assert out["families"]["B4-b"]["confirmed"] == ["Planner[B2]@B4-plain vs Planner[B1]"]
    descriptive = hypotheses.b4_hypotheses(_b4_rows(800.0), False, "calibrated", cache)
    assert "families" not in descriptive
    assert descriptive["label"] == "B4-calibrated not validated in closed loop (C2)"
    assert descriptive["differences_vs_k_star"]["Planner[B2]"] == pytest.approx(
        float(
            _b4_rows(800.0)["Planner[B2]"].total_surplus.mean()
            - _b4_rows(800.0)[crossplay.k_star_name("B1")].total_surplus.mean()
        )
    )


def test_h2_b4_exploratory_rows_and_missing_rows() -> None:
    cache = hypotheses.BootstrapCache(RESAMPLES, 3)
    rows = _b4_rows(1200.0, n_b3=15)
    rows["Planner[B3]"] = _b4_row("Planner[B3]", 800.0, n=15, seed=4)
    out = hypotheses.b4_hypotheses(rows, True, "plain", cache)
    assert out["descriptive"]["Planner[B3]"]["exploratory"]
    assert out["families"]["B4-loss"]["confirmed"] == []  # B3's loss cell is exploratory
    del rows["Planner[B1]"]
    with pytest.raises(ValueError, match="missing"):
        hypotheses.b4_hypotheses(rows, True, "plain", cache)


def test_b4_variant_validation_reads_the_c2_labels() -> None:
    ev = [
        {"variant": "plain", "regime": "R-M1", "counts_as_validated": True},
        {"variant": "plain", "regime": "R-I", "counts_as_validated": False},
        {"variant": "calibrated", "regime": "R-M2", "counts_as_validated": True},
    ]
    assert not hypotheses.b4_variant_validated(ev, "plain")  # R-M1 cannot validate B4
    assert hypotheses.b4_variant_validated(ev, "calibrated")
