"""Generational simulation loop for The Last Library."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from phasespace.library.belief import (
    bayes_update,
    belief_entropy_bits,
    expected_terrain_costs,
    map_levels,
    uniform_belief,
)
from phasespace.library.config import DEFAULT_CONFIG, LibraryConfig
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
from phasespace.library.world import (
    FloatArray,
    GridLayout,
    LatentRule,
    evaluate_path_expected_cost,
    expected_level_costs,
    generate_layout,
    mutate_rule,
    observation_matrix,
    oracle_cost,
    sample_rule,
    shortest_path,
    true_terrain_expected_costs,
)

ARMS: Final = (
    "oracle",
    "none",
    "unlimited",
    "single_lifetime",
    "example_random",
    "example_pedagogical",
    "rule",
    "rule_confidence",
    "placebo_random",
    "placebo_shuffled",
)


@dataclass(frozen=True, slots=True)
class LineageFixture:
    """Common-random-numbers layouts and latent rules for one lineage."""

    seed: int
    rule_initial: LatentRule
    rule_after: LatentRule
    train_layouts: tuple[tuple[GridLayout, ...], ...]
    heldout_layouts: tuple[tuple[GridLayout, ...], ...]
    unrelated_rule: LatentRule
    unrelated_train_layouts: tuple[tuple[GridLayout, ...], ...]


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Per-generation evaluation and channel diagnostics."""

    generation: int
    heldout_regret: float
    train_regret: float
    bits_used: int
    posterior_entropy_bits: float
    map_accuracy: float
    record_summary: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LineageResult:
    """Full trajectory of one lineage under one arm and bit budget."""

    arm: str
    budget_bits: int
    rule_changed: bool
    generations: tuple[GenerationMetrics, ...]


def make_lineage_fixture(
    seed: int, rule_changed: bool, cfg: LibraryConfig = DEFAULT_CONFIG
) -> LineageFixture:
    """Create shared rules and layouts for one lineage under `seed`."""
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    rule_initial = sample_rule(world_rng, cfg)
    rule_after = mutate_rule(rule_initial, world_rng, cfg) if rule_changed else rule_initial
    train_layouts = tuple(
        tuple(generate_layout(world_rng, cfg) for _ in range(cfg.episodes_per_generation))
        for _ in range(cfg.n_generations)
    )
    heldout_layouts = tuple(
        tuple(generate_layout(world_rng, cfg) for _ in range(cfg.heldout_layouts_per_gen))
        for _ in range(cfg.n_generations)
    )
    unrelated_rule = sample_rule(world_rng, cfg)
    unrelated_train = tuple(
        tuple(generate_layout(world_rng, cfg) for _ in range(cfg.episodes_per_generation))
        for _ in range(cfg.n_generations)
    )
    return LineageFixture(
        seed=seed,
        rule_initial=rule_initial,
        rule_after=rule_after,
        train_layouts=train_layouts,
        heldout_layouts=heldout_layouts,
        unrelated_rule=unrelated_rule,
        unrelated_train_layouts=unrelated_train,
    )


def run_lifetime(
    prior_belief: FloatArray,
    rule: LatentRule,
    train_layouts: Sequence[GridLayout],
    rng: np.random.Generator,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    obs_matrix: FloatArray | None = None,
    exp_level_costs_arr: FloatArray | None = None,
) -> tuple[FloatArray, list[tuple[int, int]], float]:
    """Run one generation's Thompson-sampling episodes, returning `(belief, obs, mean_regret)`."""
    if obs_matrix is None:
        obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    if exp_level_costs_arr is None:
        exp_level_costs_arr = expected_level_costs(cfg.level_costs, obs_matrix)
    true_costs = true_terrain_expected_costs(rule, exp_level_costs_arr)

    belief = prior_belief.copy()
    observations: list[tuple[int, int]] = []
    regrets: list[float] = []

    for layout in train_layouts:
        sampled_levels = np.array(
            [int(rng.choice(cfg.n_levels, p=belief[t])) for t in range(cfg.n_terrains)],
            dtype=np.int64,
        )
        sampled_costs: FloatArray = exp_level_costs_arr[sampled_levels]
        path, _ = shortest_path(layout, sampled_costs)
        opt = oracle_cost(layout, rule, cfg, exp_level_costs_arr)
        exp_cost = evaluate_path_expected_cost(layout, path, true_costs)
        regrets.append(max(0.0, exp_cost - opt))

        for r, c in path[1:]:
            t = int(layout.terrains[r, c])
            if t < 0:
                continue
            y = int(rng.choice(cfg.n_levels, p=obs_matrix[rule.levels[t]]))
            observations.append((t, y))
            belief = bayes_update(belief, t, y, obs_matrix)

    return belief, observations, float(np.mean(regrets))


def evaluate_heldout(
    belief: FloatArray,
    rule: LatentRule,
    heldout_layouts: Sequence[GridLayout],
    cfg: LibraryConfig = DEFAULT_CONFIG,
    obs_matrix: FloatArray | None = None,
    exp_level_costs_arr: FloatArray | None = None,
) -> float:
    """Mean held-out regret of greedy planning under posterior expected terrain costs."""
    if obs_matrix is None:
        obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    if exp_level_costs_arr is None:
        exp_level_costs_arr = expected_level_costs(cfg.level_costs, obs_matrix)
    true_costs = true_terrain_expected_costs(rule, exp_level_costs_arr)
    est_costs = expected_terrain_costs(belief, exp_level_costs_arr)

    regrets: list[float] = []
    for layout in heldout_layouts:
        path, _ = shortest_path(layout, est_costs)
        opt = oracle_cost(layout, rule, cfg, exp_level_costs_arr)
        exp_cost = evaluate_path_expected_cost(layout, path, true_costs)
        regrets.append(max(0.0, exp_cost - opt))
    return float(np.mean(regrets))


def _unrelated_rule_records(
    fixture: LineageFixture,
    budget_bits: int,
    cfg: LibraryConfig,
    obs_matrix: FloatArray,
    exp_costs: FloatArray,
) -> list[BitRecord]:
    decoy_records: list[BitRecord] = []
    belief = uniform_belief(cfg.n_terrains, cfg.n_levels)
    for g in range(cfg.n_generations):
        decoy_rng = np.random.default_rng(np.random.SeedSequence([fixture.seed, 2, g]))
        belief, _, _ = run_lifetime(
            belief,
            fixture.unrelated_rule,
            fixture.unrelated_train_layouts[g],
            decoy_rng,
            cfg,
            obs_matrix,
            exp_costs,
        )
        decoy_records.append(encode_rule(belief, budget_bits, cfg))
    return decoy_records


def _encode_outgoing(
    arm: str,
    belief: FloatArray,
    obs_pool: Sequence[tuple[int, int]],
    budget_bits: int,
    gen_rng: np.random.Generator,
    cfg: LibraryConfig,
    obs_matrix: FloatArray,
    decoy_record: BitRecord | None,
) -> BitRecord | None:
    if arm == "rule":
        return encode_rule(belief, budget_bits, cfg)
    if arm == "rule_confidence":
        return encode_rule_confidence(belief, budget_bits, cfg)
    if arm == "example_random":
        return encode_examples_random(obs_pool, budget_bits, gen_rng, cfg)
    if arm == "example_pedagogical":
        return encode_examples_pedagogical(belief, obs_pool, budget_bits, cfg, obs_matrix)
    if arm == "placebo_random":
        return encode_placebo_random(budget_bits, gen_rng, cfg)
    if arm == "placebo_shuffled":
        return decoy_record
    return None


def _decode_incoming(
    arm: str,
    record: BitRecord | None,
    carried_belief: FloatArray | None,
    cfg: LibraryConfig,
    obs_matrix: FloatArray,
) -> tuple[FloatArray, list[tuple[int, int]], int, tuple[str, ...]]:
    if arm in ("unlimited", "single_lifetime") and carried_belief is not None:
        return carried_belief.copy(), [], 0, (arm,)
    if record is None:
        return uniform_belief(cfg.n_terrains, cfg.n_levels), [], 0, ()
    if arm in ("rule", "placebo_random", "placebo_shuffled"):
        return decode_rule(record, cfg), [], record.n_bits, record.summary
    if arm == "rule_confidence":
        return decode_rule_confidence(record, cfg), [], record.n_bits, record.summary
    if arm in ("example_random", "example_pedagogical"):
        decoded_belief, pairs = decode_examples(record, cfg, obs_matrix)
        return decoded_belief, pairs, record.n_bits, record.summary
    return uniform_belief(cfg.n_terrains, cfg.n_levels), [], 0, ()


def run_lineage(
    arm: str,
    budget_bits: int,
    fixture: LineageFixture,
    rule_changed: bool,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    obs_matrix: FloatArray | None = None,
    exp_level_costs_arr: FloatArray | None = None,
) -> LineageResult:
    """Simulate a single multi-generation lineage on `fixture` for `(arm, budget_bits)`."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    if budget_bits < 0:
        raise ValueError(f"budget_bits must be >= 0, got {budget_bits}")
    if obs_matrix is None:
        obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    if exp_level_costs_arr is None:
        exp_level_costs_arr = expected_level_costs(cfg.level_costs, obs_matrix)

    decoy_records = (
        _unrelated_rule_records(fixture, budget_bits, cfg, obs_matrix, exp_level_costs_arr)
        if arm == "placebo_shuffled"
        else None
    )

    metrics: list[GenerationMetrics] = []
    incoming_record: BitRecord | None = None
    carried_belief: FloatArray | None = None

    for g in range(cfg.n_generations):
        active_rule = (
            fixture.rule_after
            if (rule_changed and g >= cfg.change_generation)
            else fixture.rule_initial
        )
        if arm == "oracle":
            metrics.append(
                GenerationMetrics(
                    generation=g,
                    heldout_regret=0.0,
                    train_regret=0.0,
                    bits_used=0,
                    posterior_entropy_bits=0.0,
                    map_accuracy=1.0,
                    record_summary=(),
                )
            )
            continue

        gen_rng = np.random.default_rng(np.random.SeedSequence([fixture.seed, 1, g]))
        prior, inherited_obs, bits_used, summary = _decode_incoming(
            arm, incoming_record, carried_belief, cfg, obs_matrix
        )
        posterior, new_obs, train_regret = run_lifetime(
            prior,
            active_rule,
            fixture.train_layouts[g],
            gen_rng,
            cfg,
            obs_matrix,
            exp_level_costs_arr,
        )
        heldout_regret = evaluate_heldout(
            posterior,
            active_rule,
            fixture.heldout_layouts[g],
            cfg,
            obs_matrix,
            exp_level_costs_arr,
        )
        maps = map_levels(posterior)
        acc = sum(m == t for m, t in zip(maps, active_rule.levels, strict=True)) / float(
            cfg.n_terrains
        )
        metrics.append(
            GenerationMetrics(
                generation=g,
                heldout_regret=heldout_regret,
                train_regret=train_regret,
                bits_used=bits_used,
                posterior_entropy_bits=belief_entropy_bits(posterior),
                map_accuracy=acc,
                record_summary=summary,
            )
        )

        if arm in ("unlimited", "single_lifetime"):
            carried_belief = posterior.copy()
            incoming_record = None
        else:
            carried_belief = None
            obs_pool = inherited_obs + new_obs
            decoy_rec = decoy_records[g] if decoy_records is not None else None
            incoming_record = _encode_outgoing(
                arm, posterior, obs_pool, budget_bits, gen_rng, cfg, obs_matrix, decoy_rec
            )

    return LineageResult(
        arm=arm,
        budget_bits=budget_bits,
        rule_changed=rule_changed,
        generations=tuple(metrics),
    )
