"""E1 (exploratory mechanism-ranking check): pure parts against hand-computed references.

Sign agreement and Kendall tau-b are checked on small hand-worked examples (and tau-b
also against `scipy.stats.kendalltau` as an independent implementation); the pair
matcher on a hand-written mini z-value fixture in both table layouts (C1 audit and
G1b), including the exclusion of M1 pairs when the M1 surrogate was not accepted.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from scipy import stats

from phasespace.citizens import e1, e1_stage

EQ, MX, PR = "Equal Baseline", "Mixed Baseline", "Proportional Baseline"
INTERP, M1, M2 = "Interpolating Baseline", "RL Agent (M1)", "RL Agent (M2)"


def _audit_table() -> dict[str, Any]:
    cols = ["comparison", "endpoint", "cohort", "higher", "lower", "n_higher", "n_lower",
            "published", "reproduced", "abs_difference"]  # fmt: skip
    return {
        "run_id": "audit-x",
        "columns": [{"name": c} for c in cols],
        "rows": [
            ["a_surplus_mixed_gt_equal", "surplus", "Exp 1", MX, EQ, 40, 40, 4.58, 4.58, 0.0],
            ["a_gini_prop_gt_mixed", "gini", "Exp 1", PR, MX, 40, 40, 1.50, 1.49, 0.01],
        ],
    }


def _g1b_table() -> dict[str, Any]:
    cols = ["name", "endpoint", "cohort", "exp4_game", "higher", "lower", "n_higher",
            "n_lower", "published", "reproduced", "tolerance", "passed"]  # fmt: skip
    return {
        "run_id": "g1b-x",
        "columns": [{"name": c} for c in cols],
        "rows": [
            ["g_surplus_m1_gt_prop", "surplus", "Exp 1", None, M1, PR, 40, 40, 3.25, 3.25,
             0.005, "true"],
            ["g_surplus_m2_gt_interp", "surplus", "Exp 3", None, M2, INTERP, 80, 80, 2.35,
             2.35, 0.005, "true"],
            ["g_surplus_m2_gt_prop_game2", "surplus", "Exp 4", 2, M2, PR, 40, 40, 4.99, 4.99,
             0.005, "true"],
        ],
    }  # fmt: skip


def _pairs() -> list[e1.HumanPair]:
    return e1.pairs_from_table(_audit_table()) + e1.pairs_from_table(_g1b_table())


# Hand-written simulated means: model X agrees on the surplus pairs and disagrees on the
# Gini one; a constant model C ties everywhere; R-M1 values exist for X but must be ignored
# while the M1 surrogate is not accepted.
SIM = {
    ("X", "Equal", "surplus"): 2.0,
    ("X", "Mixed", "surplus"): 3.0,
    ("X", "Mixed", "gini"): 0.20,
    ("X", "Proportional", "gini"): 0.10,
    ("X", "Proportional", "surplus"): 4.0,
    ("X", "R-M1", "surplus"): 9.0,
    ("X", "R-I", "surplus"): 5.0,
    ("X", "R-M2", "surplus"): 6.0,
    ("C", "Equal", "surplus"): 1.0,
    ("C", "Mixed", "surplus"): 1.0,
}
ACCEPTED = {"R-M1": False, "R-M2": True}


# --- sign agreement and tau ---------------------------------------------------------------


def test_sign_agreement_is_strict() -> None:
    assert e1.sign_agrees(2.0, 1.0)
    assert not e1.sign_agrees(1.0, 2.0)
    assert not e1.sign_agrees(1.0, 1.0)  # a tie is no agreement


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 1.0),
        ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], -1.0),
        # pairs (1,2) and (1,3) concordant, (2,3) discordant: (2 - 1) / 3
        ([1.0, 2.0, 3.0], [1.0, 3.0, 2.0], 1.0 / 3.0),
        # one pair tied in y: C = 2, D = 0, n0 = 3, n2 = 1 -> 2 / sqrt(3 * 2)
        ([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], 2.0 / math.sqrt(6.0)),
        # 4 items, one swap of adjacent ranks: C = 5, D = 1 -> 4 / 6
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0], 4.0 / 6.0),
    ],
)
def test_kendall_tau_b_hand_examples(x: list[float], y: list[float], expected: float) -> None:
    got = e1.kendall_tau_b(x, y)
    assert got is not None
    assert got == pytest.approx(expected, abs=1e-12)
    assert got == pytest.approx(float(stats.kendalltau(x, y).statistic), abs=1e-12)


def test_kendall_tau_b_undefined_for_constant_input() -> None:
    assert e1.kendall_tau_b([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None
    assert e1.kendall_tau_b([1.0], [1.0]) is None
    with pytest.raises(ValueError):
        e1.kendall_tau_b([1.0, 2.0], [1.0])


# --- pair parsing and matching ---------------------------------------------------------------


def test_pairs_from_both_table_layouts() -> None:
    pairs = _pairs()
    assert [p.name for p in pairs] == [
        "a_surplus_mixed_gt_equal", "a_gini_prop_gt_mixed", "g_surplus_m1_gt_prop",
        "g_surplus_m2_gt_interp", "g_surplus_m2_gt_prop_game2",
    ]  # fmt: skip
    assert [p.source_run for p in pairs] == ["audit-x"] * 2 + ["g1b-x"] * 3
    assert [p.exp4_game for p in pairs] == [None, None, None, None, 2]
    assert [p.significant for p in pairs] == [True, False, True, True, True]  # 1.50 < 1.96
    assert [p.heldout for p in pairs] == [False, False, True, True, True]


def test_pairs_from_table_rejects_a_non_positive_reproduced_z() -> None:
    table = _audit_table()
    table["rows"][0][8] = -4.58
    with pytest.raises(ValueError, match="not positive"):
        e1.pairs_from_table(table)


def test_matcher_statuses_and_agreement() -> None:
    m = {p.name: e1.match_pair(p, "X", SIM, ACCEPTED) for p in _pairs()}
    a = m["a_surplus_mixed_gt_equal"]
    assert (a.status, a.regime_higher, a.regime_lower) == ("scored", "Mixed", "Equal")
    assert (a.sim_higher, a.sim_lower, a.sim_difference, a.agrees) == (3.0, 2.0, 1.0, True)
    g = m["a_gini_prop_gt_mixed"]
    assert g.status == "scored" and g.agrees is False
    assert g.sim_difference == pytest.approx(-0.10)
    h = m["g_surplus_m2_gt_interp"]
    assert (h.regime_higher, h.regime_lower, h.agrees) == ("R-M2", "R-I", True)
    assert m["g_surplus_m2_gt_prop_game2"].status == "not_scorable"
    assert m["g_surplus_m2_gt_prop_game2"].agrees is None


def test_m1_pairs_are_excluded_while_the_m1_surrogate_is_rejected() -> None:
    pair = next(p for p in _pairs() if p.name == "g_surplus_m1_gt_prop")
    rejected = e1.match_pair(pair, "X", SIM, ACCEPTED)
    assert rejected.status == "excluded" and "R-M1" in rejected.reason
    assert rejected.sim_higher is None and rejected.agrees is None
    missing = e1.match_pair(pair, "X", SIM, {"R-M2": True})  # no acceptance recorded
    assert missing.status == "excluded"
    # The exclusion follows the acceptance, not the mechanism name: accepted -> scored.
    accepted = e1.match_pair(pair, "X", SIM, {"R-M1": True, "R-M2": True})
    assert accepted.status == "scored" and accepted.sim_difference == 9.0 - 4.0


def test_missing_simulation_is_not_scorable() -> None:
    pair = next(p for p in _pairs() if p.name == "g_surplus_m2_gt_interp")
    m = e1.match_pair(pair, "C", SIM, ACCEPTED)
    assert m.status == "not_scorable" and "R-M2" in m.reason and "R-I" in m.reason


def test_score_counts_by_hand() -> None:
    pairs = _pairs()
    matches = [e1.match_pair(p, model, SIM, ACCEPTED) for p in pairs for model in ("X", "C")]
    x = e1.score("X", matches)
    # scored: mixed>equal (agree), prop>mixed gini (disagree, not significant),
    # m2>interp (agree); excluded: M1; not scorable: Exp 4.
    assert (x.pairs, x.excluded, x.not_scorable, x.scorable) == (5, 1, 1, 3)
    assert (x.agreements, x.ties, x.rate) == (2, 0, pytest.approx(2 / 3))
    assert (x.significant_scorable, x.significant_agreements, x.significant_rate) == (2, 2, 1.0)
    assert (x.heldout_scorable, x.heldout_agreements, x.heldout_rate) == (1, 1, 1.0)
    c = e1.score("C", matches)
    # Only mixed>equal is scorable for C, and it is a tie: no agreement.
    assert (c.scorable, c.agreements, c.ties, c.rate) == (1, 0, 1, 0.0)
    assert c.heldout_rate is None


def test_ranking_groups_need_three_usable_mechanisms() -> None:
    pairs = _pairs()
    usable_without_m1 = [EQ, MX, PR, INTERP, M2]
    assert e1.ranking_groups(pairs, usable_without_m1) == {}  # Exp 1 surplus: Equal, Mixed
    groups = e1.ranking_groups(pairs, [*usable_without_m1, M1])
    assert groups == {("Exp 1", "surplus"): sorted([EQ, MX, M1, PR])}


# --- stage helpers ------------------------------------------------------------------------


def test_simulated_means_reads_both_sources() -> None:
    in_sample = [
        {"regime": "Equal", "model": "B1", "metric": "gini", "sim_mean": 0.1},
        {"regime": "Equal", "model": "M0", "metric": "gini", "sim_mean": None},
    ]
    heldout = [{"regime": "R-I", "model": "B1", "metric": "gini", "sim_mean": 0.2}]
    b4 = [{"variant": "plain", "regime": "R-I", "metric": "gini", "b4_mean": 0.3},
          {"variant": "calibrated", "regime": "R-M2", "metric": "gini", "b4_mean": 0.4}]  # fmt: skip
    assert e1_stage.simulated_means(in_sample, heldout, b4) == {
        ("B1", "Equal", "gini"): 0.1,
        ("B1", "R-I", "gini"): 0.2,
        ("B4-plain", "R-I", "gini"): 0.3,
        ("B4-calibrated", "R-M2", "gini"): 0.4,
    }
