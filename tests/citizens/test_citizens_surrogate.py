"""M1/M2 mechanism surrogate: shapes, simplex, equivariance, recovery of a known rule.

Independent references: hand-computed features and fidelity statistics; the exact
Proportional rule (`mechanisms.paper_offers`) as a synthetic mechanism the surrogate must
recover; permutation symmetry of the architecture; the dataset's own BC game counts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import numpy as np
import numpy.typing as npt
import pytest

from phasespace.citizens import data, game, simulate
from phasespace.citizens import mechanisms as mech
from phasespace.citizens import surrogate as sur

type FloatArray = npt.NDArray[np.float64]

TRAINING_NPZ = data.processed_path(data.DEFAULT_PROCESSED_DIR, data.TRAINING_FILE)
needs_data = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / TRAINING_NPZ).exists(),
    reason="processed dataset absent (run `just citizens-data`)",
)


def test_features_by_hand() -> None:
    x = sur.features(100.0, [10.0, 20.0, 30.0, 40.0], [2.0, 4.0, 6.0, 0.0])
    assert x.shape == (4, 5)
    assert x.dtype == np.float32
    # player 1: (e, c, R, mean e, mean c) = (20, 4, 100, 25, 3) / 200
    np.testing.assert_allclose(x[1], np.array([20, 4, 100, 25, 3]) / 200.0, rtol=1e-6)


def test_target_shares_clip_and_sum() -> None:
    # Offers summing to slightly more than R (float32 software) -> pool share clipped to 0.
    y = sur.target_shares(100.0, [25.0, 25.0, 25.0, 25.00001])
    assert y.shape == (5,)
    assert float(y[4]) == 0.0
    np.testing.assert_allclose(y.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(sur.target_shares(200.0, [50, 30, 20, 0]), [0.25, 0.15, 0.1, 0, 0.5])


def test_fidelity_statistics_by_hand() -> None:
    recorded = np.zeros((1, 2, 4))
    recorded[0, 0] = [10.0, 20.0, 0.5, 3.0]
    recorded[0, 1] = [99.0, 99.0, 99.0, 99.0]  # masked round
    predicted = recorded.copy()
    predicted[0, 0] = [11.0, 18.0, 1.5, 3.0]  # errors 1, 2, 1, 0
    f = sur.fidelity(predicted, recorded, np.array([[True, False]]))
    assert f.player_rounds == 4
    assert f.median_abs == pytest.approx(1.0)
    assert f.p90_abs == pytest.approx(np.percentile([1.0, 2.0, 1.0, 0.0], 90))
    assert f.mean_abs == pytest.approx(1.0)
    assert f.exclusion_agreement == pytest.approx(0.75)  # player 2: 0.5 excluded, 1.5 not
    assert f.recorded_exclusion_rate == pytest.approx(0.25)
    assert not f.accepted  # exclusion agreement 0.75 < 0.90


def _random_inputs(g: int, t: int, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    pool = rng.uniform(0.01, 200.0, size=(g, t))
    prev_e = rng.uniform(0.0, 50.0, size=(g, t, 4))
    prev_c = prev_e * rng.uniform(0.0, 1.0, size=(g, t, 4))
    prev_e[:, 0] = prev_c[:, 0] = 0.0
    return pool, prev_e, prev_c


def _seqs(pool: FloatArray, prev_e: FloatArray, prev_c: FloatArray) -> sur.SequenceData:
    offers = np.full(prev_e.shape, 1.0)
    return sur.SequenceData(
        inputs=sur.features(pool, prev_e, prev_c),
        pool=pool,
        offers=offers,
        targets=sur.target_shares(pool, offers),
        mask=np.ones(pool.shape, dtype=np.bool_),
        launch_id=np.array([str(i) for i in range(pool.shape[0])]),
    )


def test_offers_are_feasible_shares() -> None:
    model = sur.SurrogateNet(sur.BASE_WIDTH, jax.random.key(0))
    pool, prev_e, prev_c = _random_inputs(5, 7, 1)
    seqs = _seqs(pool, prev_e, prev_c)
    logits = np.asarray(sur._logits_jit(model, sur.to_jax(seqs.inputs)))  # pyright: ignore[reportPrivateUsage]
    assert logits.shape == (5, 7, 5)
    shares = sur.shares_from_logits(logits)
    assert np.all(shares >= 0.0)
    np.testing.assert_allclose(shares.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
    offers = sur.predict_offers(model, seqs)
    assert offers.shape == (5, 7, 4)
    for g in range(5):
        for t in range(7):
            assert game.offers_valid(float(pool[g, t]), offers[g, t])


def test_permutation_equivariance() -> None:
    """Permuting the players' inputs permutes their offers and keeps the pool share."""
    model = sur.SurrogateNet(sur.BASE_WIDTH, jax.random.key(3))
    pool, prev_e, prev_c = _random_inputs(3, 6, 2)
    perm = np.array([2, 0, 3, 1])
    base = sur.predict_offers(model, _seqs(pool, prev_e, prev_c))
    permuted = sur.predict_offers(model, _seqs(pool, prev_e[..., perm], prev_c[..., perm]))
    np.testing.assert_allclose(permuted, base[..., perm], rtol=1e-5, atol=1e-5)


def test_mechanism_steps_match_teacher_forcing_and_serialisation() -> None:
    model = sur.SurrogateNet(sur.BASE_WIDTH, jax.random.key(5))
    restored = sur.deserialise(sur.serialise(model), sur.BASE_WIDTH)
    # Type-level check: the surrogate plugs into closed-loop simulation.
    mechanism: simulate.Mechanism[sur.SurrogateState] = sur.SurrogateMechanism("test", restored)
    pool, prev_e, prev_c = _random_inputs(4, 5, 3)
    batch = sur.predict_offers(model, _seqs(pool, prev_e, prev_c))
    state = mechanism.initial_state(4)
    for t in range(5):
        first = t == 0
        offers, state = mechanism.offers(
            state, pool[:, t], None if first else prev_e[:, t], None if first else prev_c[:, t], t
        )
        np.testing.assert_allclose(offers, batch[:, t], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="both"):
        mechanism.offers(mechanism.initial_state(4), pool[:, 0], prev_e[:, 1], None, 1)


def _proportional_games(n: int, t: int, seed: int) -> sur.SequenceData:
    """Synthetic games under the exact Proportional rule, random reciprocation fractions."""
    rng = np.random.default_rng(seed)
    pool = np.zeros((n, t))
    offers = np.zeros((n, t, 4))
    prev_e = np.zeros((n, t, 4))
    prev_c = np.zeros((n, t, 4))
    for g in range(n):
        r, e_prev, c_prev = game.MAX_POOL, np.zeros(4), np.zeros(4)
        for k in range(t):
            e = mech.paper_offers(r, None if k == 0 else c_prev, mech.W_PROPORTIONAL)
            c = e * rng.uniform(0.2, 0.9, size=4)
            pool[g, k], offers[g, k], prev_e[g, k], prev_c[g, k] = r, e, e_prev, c_prev
            r, e_prev, c_prev = float(game.next_pool(r, e, c)), e, c
    return sur.SequenceData(
        inputs=sur.features(pool, prev_e, prev_c),
        pool=pool,
        offers=offers,
        targets=sur.target_shares(pool, offers),
        mask=pool >= game.DEPLETION_THRESHOLD,
        launch_id=np.array([str(i) for i in range(n)]),
    )


def test_surrogate_recovers_the_exact_proportional_rule() -> None:
    fit_set, val_set = _proportional_games(256, 10, 1), _proportional_games(64, 10, 2)
    config = sur.TrainConfig(max_updates=1500, eval_every=250)
    result = sur.fit_seed(fit_set, val_set, 7, config)
    f = sur.fidelity(sur.predict_offers(result.model, val_set), val_set.offers, val_set.mask)
    # Reference: the exact rule's offers. Thresholds are 2-3x looser than observed
    # (median 0.17, p90 0.52 coins) to stay robust across platforms.
    assert f.median_abs <= 0.5
    assert f.p90_abs <= 1.5
    assert f.exclusion_agreement >= 0.95


def test_fit_is_deterministic_given_the_seed() -> None:
    fit_set, val_set = _proportional_games(16, 4, 3), _proportional_games(8, 4, 4)
    config = sur.TrainConfig(max_updates=20, eval_every=10, batch_games=8)
    a = sur.fit_seed(fit_set, val_set, 11, config)
    b = sur.fit_seed(fit_set, val_set, 11, config)
    assert sur.serialise(a.model) == sur.serialise(b.model)
    assert a.curve == b.curve
    c = sur.fit_seed(fit_set, val_set, 12, config)
    assert sur.serialise(c.model) != sur.serialise(a.model)


def test_surrogate_mechanism_is_frozen() -> None:
    mechanism = sur.SurrogateMechanism("x", sur.SurrogateNet(8, jax.random.key(0)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        mechanism.name = "y"  # pyright: ignore[reportAttributeAccessIssue]


def test_human_fidelity_refuses_before_the_freeze() -> None:
    with pytest.raises(data.ProtocolNotFrozenError):
        sur.human_fidelity({}, confirm_frozen=False)


@needs_data
def test_bc_split_uses_only_bc_mechanism_games() -> None:
    training = data.load_training()
    for key, spec in sur.MECHANISM_SPECS.items():
        games = sur.mechanism_games(training, key)
        assert games.n_games == 512
        assert set(games.cohort.tolist()) == {spec.cohort}
        assert set(games.mechanism.tolist()) == {spec.mechanism}
        fit, val = sur.bc_split(games)
        assert (fit.n_games, val.n_games) == (410, 102)
        assert not set(fit.launch_id.tolist()) & set(val.launch_id.tolist())
        again_fit, _ = sur.bc_split(games)
        assert fit.launch_id.tolist() == again_fit.launch_id.tolist()
    human = training.select(training.is_human)
    with pytest.raises(ValueError, match="BC"):
        sur.bc_split(human.where(mechanism="Proportional Baseline", cohort="Exp 1"))
