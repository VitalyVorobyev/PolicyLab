"""Evaluation machinery: bootstrap against closed forms, criterion logic, held-out guard,
the declared handling of anomalous rows (A29) and the A22 W1 normaliser."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from phasespace.citizens import data, evaluate
from phasespace.citizens.evaluate import Estimate, ModelResult, OutcomeRow
from phasespace.citizens.models.base import (
    K_MAX,
    CitizenModel,
    Observation,
    anomalous_decisions,
    streams_from_arrays,
)


def test_bootstrap_of_a_binary_mean_follows_the_binomial_closed_form() -> None:
    # 20 units, 10 ones: a bootstrap mean is K / 20 with K ~ Binomial(20, 1/2).
    values = np.array([1.0] * 10 + [0.0] * 10)
    index = evaluate.bootstrap_indices(np.full(20, "s"), 10_000, np.random.default_rng(9001))
    boot = values[index].mean(axis=1)
    # Moments: mean 1/2, variance p(1 - p)/n = 1/80 (Monte Carlo SE of the variance ~1.4e-4).
    assert boot.mean() == pytest.approx(0.5, abs=0.005)
    assert boot.var() == pytest.approx(0.25 / 20, abs=6e-4)
    # Percentiles: P(K <= 5) = 0.0207 < 0.025 < P(K <= 6) = 0.0577, so the 2.5 % point is
    # 6/20; symmetrically the 97.5 % point is 14/20 (each ~3 MC SDs from a CDF jump).
    est = evaluate.mean_ci(values, index)
    assert stats.binom.ppf(0.025, 20, 0.5) == 6 and stats.binom.ppf(0.975, 20, 0.5) == 14
    assert (est.value, est.lo, est.hi) == (0.5, 6 / 20, 14 / 20)


def test_stratified_bootstrap_keeps_stratum_sizes() -> None:
    strata = np.array(["Exp 3"] * 5 + ["Exp 2"] * 3)
    index = evaluate.bootstrap_indices(strata, 500, np.random.default_rng(1))
    assert index.shape == (500, 8)
    assert np.all(np.isin(index[:, :3], [5, 6, 7]))  # sorted strata: Exp 2 first
    assert np.all(np.isin(index[:, 3:], [0, 1, 2, 3, 4]))
    # A constant within each stratum: every resample has the same pooled mean.
    values = np.array([1.0] * 5 + [3.0] * 3)
    est = evaluate.mean_ci(values, index)
    assert est.lo == est.hi == est.value == pytest.approx(14 / 8)


def test_ratio_estimator_weights_units_by_decisions() -> None:
    num = np.array([1.0, 3.0])
    den = np.array([1.0, 3.0])
    index = np.array([[0, 0], [1, 1], [0, 1]])
    est = evaluate.ratio_ci(num, den, index)
    assert est.value == 1.0 and est.lo == 1.0 and est.hi == 1.0
    est = evaluate.ratio_ci(np.array([2.0, -3.0]), np.array([1.0, 3.0]), index)
    assert est.value == pytest.approx(-1 / 4)
    assert (est.lo, est.hi) == pytest.approx(tuple(np.quantile([2.0, -1.0, -0.25], [0.025, 0.975])))


def _outcome(metric: str, mean: float, lo: float, hi: float, tau: float, sim: float) -> OutcomeRow:
    return OutcomeRow(
        metric, Estimate(mean, lo, hi), tau, sim, 0.0, lo - tau <= sim <= hi + tau, 0.0, 0.0
    )


def test_criterion_needs_three_of_four_metrics_and_a_positive_ci() -> None:
    rows = (
        _outcome("surplus", 10.0, 9.0, 11.0, 0.1, 11.05),  # inside the widened CI
        _outcome("gini", 0.1, 0.09, 0.11, 0.005, 0.2),  # outside
        _outcome("depletion_round", 40.0, 40.0, 40.0, 0.5, 39.6),  # degenerate CI, inside tau
        _outcome("players_active", 3.0, 2.8, 3.2, 0.1, 3.3),  # boundary hi + tau: inside
    )
    good = ModelResult("B2", rows, -2.5, Estimate(0.1, 0.02, 0.2), 1.0, 1000)
    assert good.metrics_in_range == 3 and good.pass_outcomes and good.pass_likelihood
    assert good.validated is True
    weak = ModelResult("B3", rows, -2.5, Estimate(0.1, -0.01, 0.2), 1.0, 1000)
    assert weak.pass_likelihood is False and weak.validated is False
    two = (*rows[:2], _outcome("depletion_round", 40, 40, 40, 0.5, 39.4), rows[3])
    assert ModelResult("B1", two, -2.5, Estimate(0.1, 0.02, 0.2), 1.0, 1000).validated is False
    assert ModelResult("M0-sim", rows, None, None, 1.0, 1000).validated is None


def test_heldout_evaluation_refuses_before_the_freeze() -> None:
    with pytest.raises(data.ProtocolNotFrozenError):
        evaluate.main(["heldout", "--fitted-run", "runs/none"])


def _protocol_is_draft() -> bool:
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    try:
        return data.header_status(path.read_text(encoding="utf-8")) != data.FROZEN
    except data.ProtocolNotFrozenError:
        return True


@pytest.mark.skipif(not _protocol_is_draft(), reason="protocol frozen: refusal no longer applies")
def test_heldout_evaluation_refuses_a_draft_protocol_even_when_confirmed() -> None:
    with pytest.raises(data.ProtocolNotFrozenError, match="draft"):
        evaluate.main(["heldout", "--confirm-frozen", "--fitted-run", "runs/none"])


def test_heldout_stage_helper_is_gone() -> None:
    # Held-out regimes are evaluated only through phasespace.citizens.heldout (A20/A25).
    assert not hasattr(evaluate, "heldout_stage")


class _Fixed(CitizenModel[None]):
    """P(c) = 1/2 on c in {0, 1} of every non-forced decision, 0 elsewhere."""

    name = "fixed"

    def initial_state(self, n_streams: int, rng: np.random.Generator | None) -> None:
        return None

    def step(self, state: None, obs: Observation) -> tuple[np.ndarray, None]:
        pmf = np.zeros((obs.n_streams, K_MAX))
        pmf[:, :2] = 0.5
        pmf[obs.own_offer < 1.0] = 0.0
        pmf[obs.own_offer < 1.0, 0] = 1.0
        return pmf, None

    def describe(self) -> dict[str, object]:
        return {}


def test_zero_probability_decisions_are_floored_and_counted() -> None:
    # One game, one round: actions (0, 1, 3, 2) on offers 5; c = 3 and c = 2 get P = 0.
    offers = np.full((1, 1, 4), 5.0)
    actions = np.array([[[0.0, 1.0, 3.0, 2.0]]])
    s = streams_from_arrays(offers, actions, np.array([[100.0]]), np.array([1]))
    units = evaluate.Units(np.array([0]), np.array(["s"]), (("Exp 1", "g"),))
    sums = evaluate.likelihood_sums(_Fixed(), s, units)
    assert sums.floored == 2
    assert sums.decisions.tolist() == [4.0]
    assert sums.log_prob[0] == pytest.approx(2 * math.log(0.5) + 2 * math.log(1e-12))


def test_anomalous_recorded_decisions_are_excluded_not_fatal() -> None:
    # Player 0: non-integer c; player 1: c > floor(e); player 2: c > 0 on e < 1 (forced
    # anyway); player 3: valid. Only player 3's decision is scored.
    offers = np.array([[[5.0, 2.5, 0.5, 5.0]]])
    actions = np.array([[[1.5, 3.0, 1.0, 1.0]]])
    s = streams_from_arrays(offers, actions, np.array([[100.0]]), np.array([1]))
    s = replace(s, excluded=anomalous_decisions(s))
    assert s.excluded[:, 0].tolist() == [True, True, True, False]
    assert s.scored[:, 0].tolist() == [False, False, False, True]
    units = evaluate.Units(np.array([0]), np.array(["s"]), (("Exp 1", "g"),))
    sums = evaluate.likelihood_sums(_Fixed(), s, units)
    assert sums.decisions.tolist() == [1.0] and sums.floored == 0
    assert sums.log_prob[0] == pytest.approx(math.log(0.5))


def test_w1_scale_is_the_between_unit_sd_or_tau() -> None:
    assert evaluate.w1_scale([1.0, 3.0], 0.1) == pytest.approx(math.sqrt(2.0))
    assert evaluate.w1_scale([40.0, 40.0, 40.0], 0.5) == 0.5
    row = _outcome("gini", 0.1, 0.09, 0.11, 0.005, 0.2)
    rows = tuple(replace(row, w1_normalised=v) for v in (0.2, 0.4, 0.6, 0.8))
    assert ModelResult("B2", rows, -2.5, None, 1.0, 10).mean_normalised_w1 == pytest.approx(0.5)
