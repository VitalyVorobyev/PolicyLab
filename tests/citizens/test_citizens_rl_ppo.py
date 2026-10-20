"""PPO planner correctness against independent, analytically known optima.

1. **Toy with an interior optimum** (`QuadraticToy`, one round): citizen i returns
   c = min(e, e^2 / (2a)), so its surplus e - e^2/(2a) peaks at e = a; the best planner
   offers a to every player (shares a/200, keeps 1 - 4a/200) for a total of 2a.
2. **Fixed reciprocation fraction f** (`FixedFraction`, 40 rounds, continuous c = f e,
   integer rounding aside). With s_t = sum_i e_{i,t} <= R_t:
   R_{t+1} = min(200, R_t - (1 - 1.4 f) s_t) and the round's surplus is (1 - f) s_t.
   - f > 1/1.4: every coin offered grows the pool, so R stays at the cap when s = R;
     each round's surplus is at most (1 - f) R_t <= 200 (1 - f), hence the optimum
     40 * 200 * (1 - f), reached by offering the whole pool every round.
   - f < 1/1.4: the pool never grows and R_{t+1} >= 1.4 f R_t, so the final pool is at
     least 200 (1.4 f)^40 and sum_t s_t = (200 - R_40) / (1 - 1.4 f) is at most
     200 (1 - (1.4 f)^40) / (1 - 1.4 f); the optimum (1 - f) times that is again
     reached by offering the whole pool every round (the 0.01 floor never binds for
     f = 0.6: 200 * 0.84^40 = 0.19).
   The logistic-normal planner cannot offer exactly the whole pool (softmax shares are
   positive), so it can only approach these values.
3. **Determinism**: the same seed gives bit-identical parameters.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasespace.citizens.rl import citizens as rc
from phasespace.citizens.rl import ppo, twin
from phasespace.citizens.rl.planner import Planner, deterministic_step


@pytest.fixture(autouse=True)
def x64_mode() -> Iterator[None]:
    with twin.x64():
        yield


def optimum_fixed_fraction(f: float, rounds: int = 40) -> float:
    if 1.4 * f >= 1.0:
        return rounds * 200.0 * (1.0 - f)
    g = 1.4 * f
    return (1.0 - f) * 200.0 * (1.0 - g**rounds) / (1.0 - g)


def total_surplus(
    planner: Planner, citizen: rc.JaxCitizen, rounds: int
) -> tuple[float, np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    traj, shares = rc.rollout(
        citizen,
        deterministic_step(planner),
        planner.initial_state(8),
        jax.random.key(1),
        jax.random.key(2),
        8,
        rounds,
    )
    return float(ppo.round_surplus(traj).sum(axis=1).mean()), np.asarray(shares, dtype=np.float64)


def test_optimum_formula_by_brute_force_simulation() -> None:
    """The closed forms equal the simulated give-everything policy (independent check)."""
    for f in (0.6, 0.8):
        pool, total = 200.0, 0.0
        for _ in range(40):
            total += (1.0 - f) * pool
            pool = max(0.01, min(200.0, pool - (1.0 - 1.4 * f) * pool))
        assert total == pytest.approx(optimum_fixed_fraction(f), rel=1e-12)


def test_ppo_finds_interior_optimum_of_toy() -> None:
    a = 30.0
    cfg = ppo.PPOConfig(n_games=256, n_rounds=1, learning_rate=3e-3, minibatches=4)
    citizen = rc.QuadraticToy(a=a)
    planner, _ = ppo.train(citizen, cfg, seed=3, n_iterations=150, chunk=50)
    value, shares = total_surplus(planner, citizen, 1)
    print(f"toy: planner {value:.3f} of optimum {2.0 * a}; offers {200.0 * shares[0, 0, :4]}")
    assert value >= 0.98 * 2.0 * a
    np.testing.assert_allclose(shares[0, :, :4], a / 200.0, atol=0.02)


@pytest.mark.parametrize("f", [0.6, 0.8])
def test_ppo_approaches_fixed_fraction_optimum(f: float) -> None:
    cfg = ppo.PPOConfig(n_games=128, learning_rate=3e-3, minibatches=2)
    citizen = rc.FixedFraction(fraction=f)
    planner, _ = ppo.train(citizen, cfg, seed=4, n_iterations=60, chunk=30)
    value, shares = total_surplus(planner, citizen, 40)
    optimum = optimum_fixed_fraction(f)
    assert value <= optimum * (1.0 + 1e-12)
    assert value >= 0.97 * optimum, (value, optimum)
    print(
        f"f={f}: planner {value:.2f} of optimum {optimum:.2f}; kept share {shares[:, :, 4].mean():.4f}"
    )


def test_training_is_deterministic_for_a_seed() -> None:
    cfg = ppo.PPOConfig(n_games=32, n_rounds=5, minibatches=2, epochs=2)
    citizen = rc.FixedFraction(fraction=0.5)

    def leaves(seed: int) -> list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
        planner, _ = ppo.train(citizen, cfg, seed=seed, n_iterations=4, chunk=2)
        arrays = ppo.array_part(planner)
        return [np.asarray(x) for x in jax.tree_util.tree_leaves(arrays)]

    first, second, other = leaves(9), leaves(9), leaves(10)
    assert all(np.array_equal(x, y) for x, y in zip(first, second, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(first, other, strict=True))


def test_planner_is_permutation_equivariant() -> None:
    planner = Planner(16, -0.5, jax.random.key(5))
    x = jax.random.uniform(jax.random.key(6), (3, 4, 3), dtype=jnp.float64)
    perm = twin.asarray([2, 0, 3, 1])
    h0, _ = planner.initial_state(3)
    _, mu = planner.policy_step(h0, x)
    _, mu_p = planner.policy_step(h0, x[:, perm])
    np.testing.assert_allclose(np.asarray(mu_p[:, :4]), np.asarray(mu[:, perm]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(mu_p[:, 4]), np.asarray(mu[:, 4]), atol=1e-12)


@pytest.mark.parametrize("citizen", [rc.FixedFraction(fraction=0.6), rc.QuadraticToy(a=30.0)])
def test_first_minibatch_ratio_is_one(citizen: rc.JaxCitizen) -> None:
    """Before any update, replaying the first minibatch of the first epoch gives the
    log-probabilities recorded during the rollout, so every PPO ratio is 1."""
    cfg = ppo.PPOConfig(n_games=64, n_rounds=40, minibatches=4)
    planner = Planner(cfg.hidden, cfg.init_log_std, jax.random.key(11))
    k_citizen, k_mech, k_perm = jax.random.split(jax.random.key(12), 3)
    batch, _ = ppo.collect(planner, citizen, cfg, k_citizen, k_mech)
    idx = ppo.minibatch_indices(cfg, jax.random.split(k_perm, cfg.epochs)[0])[0]
    first = ppo.Batch(*(a[:, idx] for a in batch))
    logp, _ = ppo.replay(planner, first)
    ratio = np.exp(np.asarray(logp) - np.asarray(first.log_prob))
    deviation = float(np.abs(ratio - 1.0).max())
    print(f"max |ratio - 1| = {deviation:.3e}")
    assert deviation <= 1e-12
