"""C2 citizen models against hand-computed references (no data needed).

References are independent of the implementation: integer bins counted by hand,
pmfs from the protocol formulas evaluated with plain Python arithmetic (the
Beta-binomial through its closed form for N = 2), EWA attractions updated by hand,
and the GRU through Equinox's own cell.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasespace.citizens.models import b2_condcoop as b2
from phasespace.citizens.models import b3_ewa as b3
from phasespace.citizens.models.b1_gru import (
    B1GRUEnsemble,
    GRUWeights,
    forward_step,
    random_weights,
)
from phasespace.citizens.models.base import (
    K_MAX,
    Binning,
    Observation,
    bin_index,
    bins_to_integer_pmf,
    integer_log_prob,
    rotate,
    sample_pmf,
    streams_from_arrays,
    teacher_forced_log_prob,
)
from phasespace.citizens.models.baselines import (
    B0Rational,
    M0Marginal,
    P0Persistence,
    fit_m0,
    fit_p0,
    p0_mean_nll,
)


def _obs(offers: list[float], prev: list[float], pool: float, t: int) -> Observation:
    return Observation(
        offers=np.array([offers]),
        prev_reciprocations=np.array([prev]),
        pool=np.array([pool]),
        round_index=t,
    )


# --- bin maps ---------------------------------------------------------------------------


def test_floor_bins_of_e13_counted_by_hand() -> None:
    # q = 10c/13 for c = 0..13: 0, .77, 1.54, 2.31, 3.08, 3.85, 4.62, 5.38, 6.15, 6.92,
    # 7.69, 8.46, 9.23, 10 -> floor bins (last bin closed):
    expected = [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9]
    assert bin_index(np.arange(14), 13.0, "floor10").tolist() == expected
    pmf = bins_to_integer_pmf(np.full((1, 10), 0.1), np.array([13.0]), "floor10")[0]
    counts = {b: expected.count(b) for b in range(10)}
    assert pmf[:14] == pytest.approx([0.1 / counts[b] for b in expected], abs=1e-15)
    assert pmf[14:].sum() == 0.0


def test_levels_of_e13_round_half_up_counted_by_hand() -> None:
    # q + 1/2 = 10c/13 + 0.5: .5, 1.27, 2.04, 2.81, 3.58, 4.35, 5.12, 5.88, 6.65, 7.42,
    # 8.19, 8.96, 9.73, 10.5
    expected = [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10]
    assert bin_index(np.arange(14), 13.0, "level11").tolist() == expected
    assert bin_index(1, 4.0, "level11").item() == 3  # q = 2.5 exactly: half rounds up
    assert bin_index(3, 10.0, "floor10").item() == 3  # q = 3 exactly: not bin 2
    assert bin_index(10, 10.0, "floor10").item() == 9  # c = e: the closed last bin


def test_bins_without_integers_are_renormalised_away() -> None:
    # e = 5: c = 0..5 -> bins 0, 2, 4, 6, 8, 9; p_b = (b + 1) / 55.
    p = (np.arange(10) + 1.0) / 55.0
    pmf = bins_to_integer_pmf(p[None, :], np.array([5.0]), "floor10")[0]
    assert pmf[:6] == pytest.approx(np.array([1, 3, 5, 7, 9, 10]) / 35.0, abs=1e-15)
    # e = 1.5: c = 0 -> level 0, c = 1 -> q = 6.67 -> level 7; uniform levels -> 1/2 each.
    pmf = bins_to_integer_pmf(np.full((1, 11), 1 / 11), np.array([1.5]), "level11")[0]
    assert pmf[:2] == pytest.approx([0.5, 0.5], abs=1e-15)


def test_integer_pmf_matches_a_manual_integration_and_sums_to_one() -> None:
    rng = np.random.default_rng(5)
    offers = np.array([1.0, 1.7, 2.5, 9.99, 10.0, 37.3, 49.9997, 112.4, 200.0])
    cases: tuple[tuple[Binning, int], ...] = (("floor10", 10), ("level11", 11))
    for binning, nb in cases:
        p = rng.dirichlet(np.ones(nb), size=offers.shape[0])
        pmf = bins_to_integer_pmf(p, offers, binning)
        assert pmf.sum(axis=1) == pytest.approx(np.ones(offers.shape[0]), abs=1e-12)
        for row, e in enumerate(offers.tolist()):
            n = math.floor(e)
            bins: list[int] = []
            for c in range(n + 1):
                q = 10.0 * c / e
                bins.append(min(9, math.floor(q)) if nb == 10 else math.floor(q + 0.5))
            z = sum(p[row, b] for b in set(bins))
            manual = [p[row, b] / bins.count(b) / z for b in bins]
            assert pmf[row, : n + 1] == pytest.approx(manual, rel=1e-12)
            lp = integer_log_prob(np.log(p[row : row + 1]), [n // 2], [e], binning)
            assert lp[0] == pytest.approx(math.log(manual[n // 2]), rel=1e-12)


def test_forced_decisions_are_point_masses_at_zero() -> None:
    pmf = bins_to_integer_pmf(np.full((2, 10), 0.1), np.array([0.0022, 0.99]), "floor10")
    assert pmf[:, 0].tolist() == [1.0, 1.0]
    assert pmf[:, 1:].sum() == 0.0


# --- B0, M0, P0 ----------------------------------------------------------------------------


def test_b0_pmf_and_closed_loop() -> None:
    model = B0Rational()
    pmf, _ = model.step(None, _obs([3.7, 1, 1, 1], [0, 0, 0, 0], 50.0, 0))
    assert pmf[0, :4] == pytest.approx([0.95 + 0.05 / 4, 0.0125, 0.0125, 0.0125], abs=1e-15)
    assert pmf[0, 4:].sum() == 0.0
    forced, _ = model.step(None, _obs([0.5, 1, 1, 1], [0, 0, 0, 0], 50.0, 0))
    assert forced[0, 0] == 1.0
    draw = model.sample(pmf, _obs([3.7, 1, 1, 1], [0, 0, 0, 0], 50.0, 0), np.random.default_rng(0))
    assert draw.tolist() == [0]


def _toy_streams() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One 3-round game: offers 10 each (round 2: player 3 excluded), fixed actions."""
    offers = np.array(
        [[[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 0.5]]]
    )
    actions = np.array([[[0.0, 3.0, 10.0, 5.0], [0.0, 4.0, 9.0, 5.0], [1.0, 3.0, 10.0, 0.0]]])
    pool = np.array([[100.0, 80.0, 70.0]])
    return offers, actions, pool, np.array([3])


def test_m0_frequencies_counted_by_hand() -> None:
    s = streams_from_arrays(*_toy_streams())
    # 11 scored decisions (one forced). Floor bins at e = 10: c -> bin c, c = 10 -> 9.
    # actions 0,3,10,5 | 0,4,9,5 | 1,3,10 -> bins 0,3,9,5,0,4,9,5,1,3,9.
    counts = np.array([2, 1, 0, 2, 1, 2, 0, 0, 0, 3], dtype=np.float64)
    assert fit_m0(s) == pytest.approx((counts + 0.5) / 16.0, abs=1e-15)
    m0 = M0Marginal((counts + 0.5) / 16.0)
    lp = teacher_forced_log_prob(m0, s)
    # e = 10: every bin holds one integer except bin 9 ({9, 10}).
    p = (counts + 0.5) / 16.0
    assert lp[1, 0] == pytest.approx(math.log(p[3]), rel=1e-12)  # player 1 round 0: c = 3
    assert lp[2, 0] == pytest.approx(math.log(p[9] / 2.0), rel=1e-12)  # c = 10
    assert math.isnan(lp[3, 2])  # forced (e = 0.5)


def test_p0_mixture_by_hand_and_vectorised_nll() -> None:
    s = streams_from_arrays(*_toy_streams())
    p = np.full(10, 0.1)
    q = 0.4
    model = P0Persistence(p, q)
    lp = teacher_forced_log_prob(model, s)
    # Player 1, round 1: previous c = 3 at e = 10 -> bin 3; now c = 4 -> bin 4:
    # p_4 = (1 - q) 0.1, all bins available and the mixture sums to 1.
    assert lp[1, 1] == pytest.approx(math.log(0.6 * 0.1), rel=1e-12)
    # Player 2, round 1: previous c = 10 -> bin 9; now c = 9 -> bin 9 (two integers).
    assert lp[2, 1] == pytest.approx(math.log((0.4 + 0.06) / 2.0), rel=1e-12)
    # Round 0 has no previous bin: M0 (uniform) -> 0.1 per integer except bin 9.
    assert lp[0, 0] == pytest.approx(math.log(0.1), rel=1e-12)
    nll = -np.nanmean(lp)
    assert p0_mean_nll(q, p, s) == pytest.approx(nll, rel=1e-12)
    q_hat, best = fit_p0(p, s)
    assert 0.0 <= q_hat <= 1.0 and best <= nll + 1e-12


# --- B2 --------------------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_b2_features_by_hand() -> None:
    x = b2.features(
        np.array([10.0]),
        np.array([100.0]),
        3,
        np.array([[8.0, 10.0, 10.0, 0.5]]),
        np.array([[4.0, 5.0, 2.0, 0.0]]),
    )
    assert x[0] == pytest.approx([0.4, 0.5, 7.0 / 20.5, 0.5, 0.0, 0.0], abs=1e-15)
    first = b2.features(np.array([50.0]), np.array([200.0]), 0, None, np.zeros((1, 4)))
    assert first[0] == pytest.approx([1.0, 0.0, 0.0, 1.0, 1.0, 0.0], abs=1e-15)
    excluded = b2.features(
        np.array([150.0]), np.array([160.0]), 2, np.array([[0.5, 30, 20, 10]]), np.zeros((1, 4))
    )
    assert excluded[0, 0] == 3.75 and excluded[0, 1] == 0.0 and excluded[0, 5] == 1.0


def test_b2_pmf_by_hand_for_n4() -> None:
    theta = np.array([-0.5, 1.0, 0.7, -1.2, 0.3, 0.2, 0.5, 0.8, -0.4, 0.1, 0.0, 1.1])
    x = np.array([[0.4, 0.5, 0.3, 0.5, 0.0, 0.0]])
    pi0 = _sigmoid(-0.5 + 1.0 * 0.5 + 0.7 * 0.3)
    pif = _sigmoid(-1.2)
    mu = _sigmoid(0.3 + 0.2 * 0.4 + 0.5 * 0.5 + 0.8 * 0.3 - 0.4 * 0.5)
    phi = math.exp(1.1)
    a, b = mu * phi, (1 - mu) * phi
    den = (a + b) * (a + b + 1)
    bb = [b * (b + 1) / den, 2 * a * b / den, a * (a + 1) / den]  # BetaBinomial(k; 2, a, b)
    inner = (1 - pi0) * (1 - pif)
    expected = [pi0, inner * bb[0], inner * bb[1], inner * bb[2], (1 - pi0) * pif]
    pmf = b2.pmf(theta, x, np.array([4.6]))  # n = floor(4.6) = 4
    assert pmf[0, :5] == pytest.approx(expected, rel=1e-12)
    assert pmf[0].sum() == pytest.approx(1.0, abs=1e-12)
    one = b2.pmf(theta, x, np.array([1.2]))
    assert one[0, :2] == pytest.approx([pi0, 1 - pi0], rel=1e-12)
    two = b2.pmf(theta, x, np.array([2.0]))
    assert two[0, :3] == pytest.approx([pi0, inner, (1 - pi0) * pif], rel=1e-12)
    lp = b2.log_prob(theta, np.repeat(x, 5, axis=0), np.arange(5.0), np.full(5, 4.0))
    assert np.exp(lp) == pytest.approx(expected, rel=1e-12)


def test_b2_vectorised_nll_equals_the_step_interface() -> None:
    s = streams_from_arrays(*_toy_streams())
    theta = np.linspace(-0.6, 0.6, 12)
    model = b2.B2ConditionalCooperation(theta)
    generic = -np.nanmean(teacher_forced_log_prob(model, s))
    assert b2.mean_nll(theta, b2.FitData.from_streams(s)) == pytest.approx(generic, rel=1e-12)


# --- B3 / B23 ---------------------------------------------------------------------------------


def test_ewa_two_rounds_by_hand() -> None:
    phi, rho, delta, lam, n0, theta = 0.5, 0.8, 0.3, 2.0, 2.0, 1.0
    p0 = np.full(11, 1 / 11)
    model = b3.EWAModel([phi, rho, delta, lam, n0, theta], p0, hybrid=False)
    state = model.initial_state(1, None)
    pmf0, state = model.step(state, _obs([20, 20, 20, 20], [0, 0, 0, 0], 120.0, 0))
    # Round 0: softmax(lambda A0) = p0 (uniform): e = 20, levels hold c with
    # round(c / 2) = j: level 0 {0}, level 10 {19, 20}, others {2j - 1, 2j}.
    assert pmf0[0, 0] == pytest.approx(1 / 11, rel=1e-12)
    assert pmf0[0, 20] == pytest.approx(1 / 22, rel=1e-12)
    # Player returned c = 10 of 20 -> level 5; next offer 30.
    a0 = math.log(1 / 11) / lam
    n1 = rho * n0 + 1.0
    a1: list[float] = []
    for j in range(11):
        chosen = 1.0 if j == 5 else 0.0
        payoff = (delta + (1 - delta) * chosen) * 20 * (1 - j / 10) / 50 + chosen * theta * 30 / 50
        a1.append((phi * n0 * a0 + payoff) / n1)
    w = [math.exp(lam * a) for a in a1]
    p1 = [v / sum(w) for v in w]
    pmf1, state = model.step(state, _obs([30, 20, 20, 20], [10, 5, 5, 5], 140.0, 1))
    # e = 30: level j holds round(c / 3) = j: {0, 1}, {29, 30} for 0 and 10, else 3 each.
    for c in range(31):
        j = math.floor(c / 3 + 0.5)
        m = 2 if j in (0, 10) else 3
        assert pmf1[0, c] == pytest.approx(p1[j] / m, rel=1e-12)
    assert state.attractions[0] == pytest.approx(a1, rel=1e-12)
    assert state.experience[0] == pytest.approx(n1, rel=1e-15)
    # Round 1 (non-forced) is reinforced at round 2 even though round 2 is an exclusion
    # (theta term with e = 0.5); the forced round 2 itself is not reinforced at round 3.
    _, state2 = model.step(state, _obs([0.5, 30, 30, 30], [12, 5, 5, 5], 90.0, 2))
    n2 = rho * n1 + 1.0
    a2: list[float] = []
    for j in range(11):
        chosen = 1.0 if j == 4 else 0.0  # c = 12 of 30 -> level 4
        payoff = (delta + (1 - delta) * chosen) * 30 * (1 - j / 10) / 50 + chosen * theta * 0.5 / 50
        a2.append((phi * n1 * a1[j] + payoff) / n2)
    assert state2.attractions[0] == pytest.approx(a2, rel=1e-12)
    _, state3 = model.step(state2, _obs([25, 25, 25, 25], [0, 5, 5, 5], 100.0, 3))
    assert state3.attractions[0].tolist() == state2.attractions[0].tolist()
    assert state3.experience[0] == state2.experience[0]


def test_b23_conditional_pull_by_hand() -> None:
    params = [0.5, 0.8, 0.3, 2.0, 2.0, 1.0, 4.0]
    p0 = np.linspace(1, 2, 11) / np.linspace(1, 2, 11).sum()
    model = b3.EWAModel(params, p0, hybrid=True)
    state = model.initial_state(1, None)
    pmf0, state = model.step(state, _obs([20, 10, 10, 20], [0, 0, 0, 0], 120.0, 0))
    # Round 0: no pull (t = 0), p = p0.
    assert pmf0[0, 0] == pytest.approx(p0[0], rel=1e-12)
    pmf1, state = model.step(state, _obs([30, 10, 10, 10], [10, 2, 6, 10], 140.0, 1))
    x3 = (2 + 6 + 10) / (10 + 10 + 20)
    logits = [2.0 * a - 4.0 * (j / 10 - x3) ** 2 for j, a in enumerate(state.attractions[0])]
    w = [math.exp(v) for v in logits]
    p = [v / sum(w) for v in w]
    assert pmf1[0, 15] == pytest.approx(p[5] / 3, rel=1e-12)  # round(15/3) = 5, three integers


def test_ewa_vectorised_nll_equals_the_step_interface() -> None:
    s = streams_from_arrays(*_toy_streams())
    p0 = b3.round1_levels(s)
    for hybrid, params in (
        (False, [0.6, 0.9, 0.2, 3.0, 1.5, 0.7]),
        (True, [0.6, 0.9, 0.2, 3.0, 1.5, 0.7, 2.5]),
    ):
        model = b3.EWAModel(params, p0, hybrid=hybrid)
        generic = -np.nanmean(teacher_forced_log_prob(model, s))
        data = b3.FitData.from_streams(s, p0)
        assert b3.mean_nll(params, data, hybrid) == pytest.approx(generic, rel=1e-12)


# --- B1 -----------------------------------------------------------------------------------------


def test_numpy_gru_matches_equinox() -> None:
    import jax
    import jax.numpy as jnp

    from phasespace.citizens.models.b1_train import GRUClone, to_weights

    model = GRUClone(jax.random.PRNGKey(7))
    weights = to_weights(model)
    rng = np.random.default_rng(1)
    xs = rng.uniform(0.0, 1.0, (6, 9))
    reference = np.asarray(model(jnp.asarray(xs, dtype=jnp.float32)), dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
    hidden = np.zeros((1, 64))
    for t in range(6):
        hidden, logits = forward_step(weights, hidden, xs[t : t + 1])
        assert logits[0] == pytest.approx(reference[t], abs=2e-5)


def test_b1_ensemble_mean_and_closed_loop_member() -> None:
    rng = np.random.default_rng(3)
    members = [random_weights(rng), random_weights(rng)]
    model = B1GRUEnsemble(members)
    obs = _obs([30, 10, 10, 10], [0, 0, 0, 0], 120.0, 0)
    pmf, _ = model.step(model.initial_state(1, None), obs)
    single = [
        B1GRUEnsemble([w]).step(B1GRUEnsemble([w]).initial_state(1, None), obs)[0] for w in members
    ]
    assert pmf == pytest.approx((single[0] + single[1]) / 2.0, abs=1e-15)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    state = model.initial_state(8, np.random.default_rng(0))
    assert state.member is not None and set(state.member.tolist()) <= {0, 1}
    many = Observation(
        offers=np.repeat(obs.offers, 8, axis=0),
        prev_reciprocations=np.zeros((8, 4)),
        pool=np.full(8, 120.0),
        round_index=0,
    )
    closed, _ = model.step(state, many)
    for row, m in enumerate(state.member.tolist()):
        assert closed[row] == pytest.approx(single[m][0], abs=1e-15)


def test_gru_weights_round_trip_through_npz_bytes() -> None:
    rng = np.random.default_rng(2)
    w = random_weights(rng)
    as32 = GRUWeights({k: v.astype(np.float32).astype(np.float64) for k, v in w.arrays.items()})
    payload = as32.to_npz_bytes()
    assert payload == as32.to_npz_bytes()  # deterministic bytes (stable checksums)
    back = GRUWeights.from_npz_bytes(payload)
    for name in as32.arrays:
        assert np.array_equal(back[name], as32[name])


# --- streams and sampling ------------------------------------------------------------------------


def test_streams_rotate_own_first() -> None:
    offers, actions, pool, n = _toy_streams()
    s = streams_from_arrays(offers + np.arange(4.0), actions, pool, n)
    # Stream 2 (player 2) sees offers (e2, e3, e0, e1) and previous actions likewise.
    assert s.offers[2, 0].tolist() == [12.0, 13.0, 10.0, 11.0]
    assert s.prev_reciprocations[2, 1].tolist() == [10.0, 5.0, 0.0, 3.0]
    assert s.prev_reciprocations[2, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert rotate(np.array([1.0, 2.0, 3.0, 4.0]))[3].tolist() == [4.0, 1.0, 2.0, 3.0]
    assert s.actions[3].tolist() == [5.0, 5.0, 0.0]


def test_sampling_frequencies_follow_the_pmf() -> None:
    pmf = np.zeros((20000, K_MAX))
    pmf[:, :3] = [0.2, 0.5, 0.3]
    draws = sample_pmf(pmf, np.random.default_rng(11))
    freq = np.bincount(draws, minlength=3) / draws.shape[0]
    assert freq[:3] == pytest.approx([0.2, 0.5, 0.3], abs=0.015)  # ~4.5 binomial SDs
    assert draws.max() <= 2


def test_jax_integer_nll_equals_the_numpy_integer_pmf_nll() -> None:
    # B1 early stopping scores with the JAX integer_nll_sum (float32); the evaluation
    # with the NumPy integer pmf of protocol 4 (float64). Same model, random games.
    import jax

    from phasespace.citizens.models.b1_train import (
        GRUClone,
        SequenceData,
        integer_nll_sum,
        to_weights,
    )

    rng = np.random.default_rng(42)
    g, t = 3, 7
    offers = rng.uniform(0.2, 60.0, (g, t, 4))
    offers[0, 2, 1] = 0.5  # a forced decision
    offers[1, 3, :] = [1.0, 2.0, 10.0, 13.0]  # small offers and exact bin edges
    actions = np.where(
        offers >= 1.0, np.floor(rng.uniform(0, 1, offers.shape) * (np.floor(offers) + 1)), 0.0
    )
    pool = rng.uniform(1.0, 200.0, (g, t))
    s = streams_from_arrays(offers, actions, pool, np.array([7, 5, 6]))
    for key in (3, 11):
        model = GRUClone(jax.random.PRNGKey(key))
        total, count = integer_nll_sum(model, SequenceData.from_streams(s).device())
        lp = teacher_forced_log_prob(B1GRUEnsemble([to_weights(model)]), s)
        assert int(count) == int(s.scored.sum())
        assert float(total) == pytest.approx(-float(np.nansum(lp)), rel=2e-5)
