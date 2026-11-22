"""Bit-budget invariant, round-trip, and pedagogical-selection tests for records."""

from __future__ import annotations

import numpy as np

from phasespace.library.belief import (
    bayes_update_batch,
    kl_divergence_nats,
    map_levels,
    uniform_belief,
)
from phasespace.library.config import DEFAULT_CONFIG
from phasespace.library.records import (
    BitRecord,
    decode_examples,
    decode_rule,
    decode_rule_confidence,
    encode_examples_pedagogical,
    encode_examples_random,
    encode_placebo_random,
    encode_rule,
    encode_rule_confidence,
)
from phasespace.library.world import observation_matrix


def test_all_encoders_respect_exact_bit_budget_across_sweep() -> None:
    obs = observation_matrix(DEFAULT_CONFIG.n_levels, DEFAULT_CONFIG.noise_beta)
    history = [(0, 0), (0, 0), (2, 3), (2, 3), (4, 1), (5, 2), (7, 0), (7, 0), (7, 0)]
    belief = bayes_update_batch(uniform_belief(), history, obs)
    rng = np.random.default_rng(42)

    for budget in DEFAULT_CONFIG.bit_budgets:
        recs: list[BitRecord] = [
            encode_rule(belief, budget, DEFAULT_CONFIG),
            encode_rule_confidence(belief, budget, DEFAULT_CONFIG),
            encode_examples_random(history, budget, rng, DEFAULT_CONFIG),
            encode_examples_pedagogical(belief, history, budget, DEFAULT_CONFIG, obs),
            encode_placebo_random(budget, rng, DEFAULT_CONFIG),
        ]
        for rec in recs:
            assert 0 <= rec.n_bits <= budget
            assert 0 <= rec.payload_int < (1 << rec.n_bits)
            if budget == 0:
                assert rec.n_bits == 0
                assert rec.payload_int == 0


def test_zero_budget_decodes_to_uniform_prior() -> None:
    obs = observation_matrix(DEFAULT_CONFIG.n_levels, DEFAULT_CONFIG.noise_beta)
    belief = bayes_update_batch(uniform_belief(), [(1, 2), (1, 2), (4, 0)], obs)
    u = uniform_belief()

    np.testing.assert_allclose(decode_rule(encode_rule(belief, 0)), u, atol=1e-15)
    np.testing.assert_allclose(
        decode_rule_confidence(encode_rule_confidence(belief, 0)), u, atol=1e-15
    )
    dec_ex, pairs = decode_examples(
        encode_examples_pedagogical(belief, [(1, 2)], 0, obs_matrix=obs)
    )
    assert pairs == []
    np.testing.assert_allclose(dec_ex, u, atol=1e-15)


def test_rule_and_confidence_round_trip() -> None:
    obs = observation_matrix(DEFAULT_CONFIG.n_levels, DEFAULT_CONFIG.noise_beta)
    belief = bayes_update_batch(
        uniform_belief(),
        [(2, 0), (2, 0), (2, 0), (5, 3), (5, 3), (1, 1)],
        obs,
    )
    # Dense rule at B = 20 (>= 19):
    rec_dense = encode_rule(belief, 20)
    assert rec_dense.n_bits == 19
    dec_dense = decode_rule(rec_dense)
    assert float(dec_dense[2, 0]) == DEFAULT_CONFIG.rule_trust
    assert float(dec_dense[5, 3]) == DEFAULT_CONFIG.rule_trust
    assert float(dec_dense[1, 1]) == DEFAULT_CONFIG.rule_trust
    np.testing.assert_allclose(dec_dense[0], np.full(4, 0.25), atol=1e-15)

    # Sparse rule at B = 10 -> top 2 lowest-entropy terrains (2 and 5, ignoring weaker 1):
    rec_sparse = encode_rule(belief, 10)
    assert rec_sparse.n_bits == 10
    dec_sparse = decode_rule(rec_sparse)
    assert float(dec_sparse[2, 0]) == DEFAULT_CONFIG.rule_trust
    assert float(dec_sparse[5, 3]) == DEFAULT_CONFIG.rule_trust
    np.testing.assert_allclose(dec_sparse[1], np.full(4, 0.25), atol=1e-15)

    # Dense confidence rule at B = 40 (>= 32) preserves confidence gradient:
    rec_conf = encode_rule_confidence(belief, 40)
    assert rec_conf.n_bits == 32
    dec_conf = decode_rule_confidence(rec_conf)
    assert float(dec_conf[2, 0]) > float(dec_conf[1, 1])
    assert map_levels(dec_conf)[2] == 0
    assert map_levels(dec_conf)[5] == 3

    # Sparse confidence rule at B = 15 (7 bits/entry -> 2 entries):
    rec_conf_sparse = encode_rule_confidence(belief, 15)
    assert rec_conf_sparse.n_bits == 14
    dec_conf_sparse = decode_rule_confidence(rec_conf_sparse)
    assert float(dec_conf_sparse[2, 0]) > 0.7
    assert float(dec_conf_sparse[5, 3]) > 0.7


def test_pedagogical_examples_beat_random_on_kl() -> None:
    obs = observation_matrix(DEFAULT_CONFIG.n_levels, DEFAULT_CONFIG.noise_beta)
    # Sender strongly knows T2=L0 and T5=L3 from 4 clean hits each, amidst noisy T0 distractors:
    pool = (
        [(2, 0)] * 4
        + [(5, 3)] * 4
        + [(0, 1), (0, 2), (1, 0), (1, 3), (3, 1), (3, 2), (4, 1), (4, 2)]
    )
    sender = bayes_update_batch(uniform_belief(), pool, obs)
    ped_rec = encode_examples_pedagogical(sender, pool, budget_bits=20, obs_matrix=obs)
    ped_belief, _ = decode_examples(ped_rec, obs_matrix=obs)
    kl_ped = kl_divergence_nats(sender, ped_belief)

    rng = np.random.default_rng(2026)
    kl_rands: list[float] = []
    for _ in range(30):
        rand_rec = encode_examples_random(pool, budget_bits=20, rng=rng)
        rand_belief, _ = decode_examples(rand_rec, obs_matrix=obs)
        kl_rands.append(kl_divergence_nats(sender, rand_belief))

    assert kl_ped < float(np.mean(kl_rands))
