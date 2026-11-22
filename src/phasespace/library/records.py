"""Bit-budgeted record encoders and decoders for rule, confidence-rule, example, and placebo arms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from phasespace.library.belief import bayes_update_batch, terrain_entropy_bits, uniform_belief
from phasespace.library.config import DEFAULT_CONFIG, LibraryConfig
from phasespace.library.world import FloatArray, observation_matrix

RULE_DENSE_BITS: Final = 19  # 5^8 = 390,625 < 2^19 = 524,288
RULE_SPARSE_ENTRY_BITS: Final = 5  # 3 bits terrain + 2 bits level
RULE_CONF_DENSE_BITS: Final = 32  # 8 * 4 bits (2 bits level + 2 bits confidence bin)
RULE_CONF_SPARSE_ENTRY_BITS: Final = 7  # 3 bits terrain + 2 bits level + 2 bits confidence bin
EXAMPLE_ENTRY_BITS: Final = 5  # 3 bits terrain + 2 bits observed level


@dataclass(frozen=True, slots=True)
class BitRecord:
    """An exact bounded bit-string record passed between generations."""

    encoding: str
    budget_bits: int
    n_bits: int
    payload_int: int
    summary: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.budget_bits < 0:
            raise ValueError(f"budget_bits must be >= 0, got {self.budget_bits}")
        if not (0 <= self.n_bits <= self.budget_bits):
            raise ValueError(f"n_bits={self.n_bits} exceeds budget_bits={self.budget_bits}")
        if self.payload_int < 0 or self.payload_int >= (1 << self.n_bits):
            raise ValueError(f"payload_int={self.payload_int} does not fit in n_bits={self.n_bits}")


def _confidence_bin(p_max: float, thresholds: tuple[float, ...]) -> int:
    if p_max <= thresholds[0]:
        return 0
    if p_max <= thresholds[1]:
        return 1
    if p_max <= thresholds[2]:
        return 2
    return 3


def _prior_from_level_trust(level: int, trust: float, n_levels: int) -> FloatArray:
    rest = (1.0 - trust) / float(n_levels - 1)
    row = np.full(n_levels, rest, dtype=np.float64)
    row[level] = trust
    return row


def encode_rule(
    belief: FloatArray, budget_bits: int, cfg: LibraryConfig = DEFAULT_CONFIG
) -> BitRecord:
    """Encode MAP terrain rules under `budget_bits` (`19`-bit dense base-5 or `5`-bit sparse)."""
    if budget_bits < 0:
        raise ValueError(f"budget_bits must be >= 0, got {budget_bits}")
    if budget_bits >= RULE_DENSE_BITS:
        payload = 0
        mult = 1
        summary: list[str] = []
        base_thresh = 1.0 / cfg.n_levels + 1e-9
        for t in range(cfg.n_terrains):
            p_max = float(belief[t].max())
            if p_max > base_thresh:
                level = int(np.argmax(belief[t]))
                digit = 1 + level
                summary.append(f"T{t}=L{level}")
            else:
                digit = 0
            payload += digit * mult
            mult *= cfg.n_levels + 1
        return BitRecord(
            encoding="rule",
            budget_bits=budget_bits,
            n_bits=RULE_DENSE_BITS,
            payload_int=payload,
            summary=tuple(summary),
        )

    max_entries = budget_bits // RULE_SPARSE_ENTRY_BITS
    if max_entries <= 0:
        return BitRecord(
            encoding="rule", budget_bits=budget_bits, n_bits=0, payload_int=0, summary=()
        )
    entropies = terrain_entropy_bits(belief)
    base_thresh = 1.0 / cfg.n_levels + 1e-9
    informed = [t for t in range(cfg.n_terrains) if float(belief[t].max()) > base_thresh]
    informed.sort(key=lambda t: (float(entropies[t]), t))
    chosen = informed[:max_entries]
    payload = 0
    summary = []
    for i, t in enumerate(chosen):
        level = int(np.argmax(belief[t]))
        entry = (t << 2) | level
        payload |= entry << (i * RULE_SPARSE_ENTRY_BITS)
        summary.append(f"T{t}=L{level}")
    return BitRecord(
        encoding="rule",
        budget_bits=budget_bits,
        n_bits=len(chosen) * RULE_SPARSE_ENTRY_BITS,
        payload_int=payload,
        summary=tuple(summary),
    )


def decode_rule(record: BitRecord, cfg: LibraryConfig = DEFAULT_CONFIG) -> FloatArray:
    """Decode a `rule` (or `placebo_*`) bit record into a prior belief array."""
    belief = uniform_belief(cfg.n_terrains, cfg.n_levels)
    if record.n_bits == 0:
        return belief
    if record.n_bits == RULE_DENSE_BITS:
        rem = record.payload_int
        base = cfg.n_levels + 1
        for t in range(cfg.n_terrains):
            digit = rem % base
            rem //= base
            if digit > 0:
                belief[t] = _prior_from_level_trust(digit - 1, cfg.rule_trust, cfg.n_levels)
        return belief
    if record.n_bits % RULE_SPARSE_ENTRY_BITS != 0:
        raise ValueError(f"Invalid sparse rule bit length {record.n_bits}")
    n_entries = record.n_bits // RULE_SPARSE_ENTRY_BITS
    for i in range(n_entries):
        entry = (record.payload_int >> (i * RULE_SPARSE_ENTRY_BITS)) & 0x1F
        t = (entry >> 2) & 0x7
        level = entry & 0x3
        belief[t] = _prior_from_level_trust(level, cfg.rule_trust, cfg.n_levels)
    return belief


def decode_rule_hedged(record: BitRecord, cfg: LibraryConfig = DEFAULT_CONFIG) -> FloatArray:
    """Decode a `rule` record with receiver hazard hedging toward uniform."""
    rule_prior = decode_rule(record, cfg)
    uni = uniform_belief(cfg.n_terrains, cfg.n_levels)
    return (1.0 - cfg.hedge_discount) * rule_prior + cfg.hedge_discount * uni


def encode_rule_confidence(
    belief: FloatArray, budget_bits: int, cfg: LibraryConfig = DEFAULT_CONFIG
) -> BitRecord:
    """Encode `(level, confidence_bin)` rules (`32`-bit dense or `7`-bit sparse)."""
    if budget_bits < 0:
        raise ValueError(f"budget_bits must be >= 0, got {budget_bits}")
    if budget_bits >= RULE_CONF_DENSE_BITS:
        payload = 0
        summary: list[str] = []
        for t in range(cfg.n_terrains):
            p_max = float(belief[t].max())
            conf_bin = _confidence_bin(p_max, cfg.confidence_thresholds)
            level = int(np.argmax(belief[t])) if conf_bin > 0 else 0
            nibble = (level << 2) | conf_bin
            payload |= nibble << (4 * t)
            if conf_bin > 0:
                summary.append(f"T{t}=L{level}@c{conf_bin}")
        return BitRecord(
            encoding="rule_confidence",
            budget_bits=budget_bits,
            n_bits=RULE_CONF_DENSE_BITS,
            payload_int=payload,
            summary=tuple(summary),
        )

    max_entries = budget_bits // RULE_CONF_SPARSE_ENTRY_BITS
    if max_entries <= 0:
        return BitRecord(
            encoding="rule_confidence", budget_bits=budget_bits, n_bits=0, payload_int=0, summary=()
        )
    entropies = terrain_entropy_bits(belief)
    informed = [
        t
        for t in range(cfg.n_terrains)
        if _confidence_bin(float(belief[t].max()), cfg.confidence_thresholds) > 0
    ]
    informed.sort(key=lambda t: (float(entropies[t]), t))
    chosen = informed[:max_entries]
    payload = 0
    summary = []
    for i, t in enumerate(chosen):
        conf_bin = _confidence_bin(float(belief[t].max()), cfg.confidence_thresholds)
        level = int(np.argmax(belief[t]))
        entry = (t << 4) | (level << 2) | conf_bin
        payload |= entry << (i * RULE_CONF_SPARSE_ENTRY_BITS)
        summary.append(f"T{t}=L{level}@c{conf_bin}")
    return BitRecord(
        encoding="rule_confidence",
        budget_bits=budget_bits,
        n_bits=len(chosen) * RULE_CONF_SPARSE_ENTRY_BITS,
        payload_int=payload,
        summary=tuple(summary),
    )


def decode_rule_confidence(record: BitRecord, cfg: LibraryConfig = DEFAULT_CONFIG) -> FloatArray:
    """Decode a `rule_confidence` bit record using `cfg.confidence_trusts`."""
    belief = uniform_belief(cfg.n_terrains, cfg.n_levels)
    if record.n_bits == 0:
        return belief
    if record.n_bits == RULE_CONF_DENSE_BITS:
        for t in range(cfg.n_terrains):
            nibble = (record.payload_int >> (4 * t)) & 0xF
            level = (nibble >> 2) & 0x3
            conf_bin = nibble & 0x3
            if conf_bin > 0:
                belief[t] = _prior_from_level_trust(
                    level, cfg.confidence_trusts[conf_bin], cfg.n_levels
                )
        return belief
    if record.n_bits % RULE_CONF_SPARSE_ENTRY_BITS != 0:
        raise ValueError(f"Invalid sparse confidence-rule bit length {record.n_bits}")
    n_entries = record.n_bits // RULE_CONF_SPARSE_ENTRY_BITS
    for i in range(n_entries):
        entry = (record.payload_int >> (i * RULE_CONF_SPARSE_ENTRY_BITS)) & 0x7F
        t = (entry >> 4) & 0x7
        level = (entry >> 2) & 0x3
        conf_bin = entry & 0x3
        if conf_bin > 0:
            belief[t] = _prior_from_level_trust(
                level, cfg.confidence_trusts[conf_bin], cfg.n_levels
            )
    return belief


def _pack_examples(encoding: str, chosen: Sequence[tuple[int, int]], budget_bits: int) -> BitRecord:
    payload = 0
    summary: list[str] = []
    for i, (t, y) in enumerate(chosen):
        entry = ((t & 0x7) << 2) | (y & 0x3)
        payload |= entry << (i * EXAMPLE_ENTRY_BITS)
        summary.append(f"T{t}->y{y}")
    return BitRecord(
        encoding=encoding,
        budget_bits=budget_bits,
        n_bits=len(chosen) * EXAMPLE_ENTRY_BITS,
        payload_int=payload,
        summary=tuple(summary),
    )


def encode_examples_random(
    observations: Sequence[tuple[int, int]],
    budget_bits: int,
    rng: np.random.Generator,
    cfg: LibraryConfig = DEFAULT_CONFIG,
) -> BitRecord:
    """Randomly subsample up to `budget_bits // 5` observations `(terrain_id, observed_y)`."""
    if budget_bits < 0:
        raise ValueError(f"budget_bits must be >= 0, got {budget_bits}")
    max_examples = budget_bits // EXAMPLE_ENTRY_BITS
    if max_examples <= 0 or not observations:
        return BitRecord(
            encoding="example_random",
            budget_bits=budget_bits,
            n_bits=0,
            payload_int=0,
            summary=(),
        )
    take = min(max_examples, len(observations))
    indices = sorted(int(i) for i in rng.choice(len(observations), size=take, replace=False))
    chosen = [observations[i] for i in indices]
    return _pack_examples("example_random", chosen, budget_bits)


def _row_kl_nats(p_row: FloatArray, q_row: FloatArray) -> float:
    q_safe = np.maximum(q_row, 1e-15)
    terms = np.where(p_row > 0.0, p_row * np.log(np.maximum(p_row, 1e-15) / q_safe), 0.0)
    return float(np.sum(terms))


def encode_examples_pedagogical(
    sender_belief: FloatArray,
    observations: Sequence[tuple[int, int]],
    budget_bits: int,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    obs_matrix: FloatArray | None = None,
) -> BitRecord:
    """Greedily select up to `budget_bits // 5` observations minimizing `KL(sender || receiver)`."""
    if budget_bits < 0:
        raise ValueError(f"budget_bits must be >= 0, got {budget_bits}")
    max_examples = budget_bits // EXAMPLE_ENTRY_BITS
    if max_examples <= 0 or not observations:
        return BitRecord(
            encoding="example_pedagogical",
            budget_bits=budget_bits,
            n_bits=0,
            payload_int=0,
            summary=(),
        )
    if obs_matrix is None:
        obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)

    pool: Counter[tuple[int, int]] = Counter(observations)
    candidates = sorted(pool.keys())
    receiver = uniform_belief(cfg.n_terrains, cfg.n_levels)
    cur_kl = [_row_kl_nats(sender_belief[t], receiver[t]) for t in range(cfg.n_terrains)]
    chosen: list[tuple[int, int]] = []

    for _ in range(min(max_examples, len(observations))):
        best_pair: tuple[int, int] | None = None
        best_gain = 1e-12
        best_new_row: FloatArray | None = None
        best_new_kl = 0.0

        for t, y in candidates:
            if pool[(t, y)] <= 0:
                continue
            cand_row = receiver[t] * obs_matrix[:, y]
            total = float(cand_row.sum())
            if total <= 0.0:
                continue
            cand_row = cand_row / total
            new_kl = _row_kl_nats(sender_belief[t], cand_row)
            gain = cur_kl[t] - new_kl
            if gain > best_gain:
                best_gain = gain
                best_pair = (t, y)
                best_new_row = cand_row
                best_new_kl = new_kl

        if best_pair is None or best_new_row is None:
            break
        t_best, _ = best_pair
        chosen.append(best_pair)
        pool[best_pair] -= 1
        receiver[t_best] = best_new_row
        cur_kl[t_best] = best_new_kl

    return _pack_examples("example_pedagogical", chosen, budget_bits)


def decode_examples(
    record: BitRecord, cfg: LibraryConfig = DEFAULT_CONFIG, obs_matrix: FloatArray | None = None
) -> tuple[FloatArray, list[tuple[int, int]]]:
    """Decode an example record into `(posterior_belief, unpacked_observations)`."""
    if obs_matrix is None:
        obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    if record.n_bits == 0:
        return uniform_belief(cfg.n_terrains, cfg.n_levels), []
    if record.n_bits % EXAMPLE_ENTRY_BITS != 0:
        raise ValueError(f"Invalid example bit length {record.n_bits}")
    n_entries = record.n_bits // EXAMPLE_ENTRY_BITS
    pairs: list[tuple[int, int]] = []
    for i in range(n_entries):
        entry = (record.payload_int >> (i * EXAMPLE_ENTRY_BITS)) & 0x1F
        t = (entry >> 2) & 0x7
        y = entry & 0x3
        pairs.append((t, y))
    belief = bayes_update_batch(uniform_belief(cfg.n_terrains, cfg.n_levels), pairs, obs_matrix)
    return belief, pairs


def encode_placebo_random(
    budget_bits: int, rng: np.random.Generator, cfg: LibraryConfig = DEFAULT_CONFIG
) -> BitRecord:
    """Sample uniformly random valid `rule`-formatted bits within `budget_bits`."""
    if budget_bits < 0:
        raise ValueError(f"budget_bits must be >= 0, got {budget_bits}")
    if budget_bits >= RULE_DENSE_BITS:
        max_val = (cfg.n_levels + 1) ** cfg.n_terrains
        payload = int(rng.integers(0, max_val))
        return BitRecord(
            encoding="placebo_random",
            budget_bits=budget_bits,
            n_bits=RULE_DENSE_BITS,
            payload_int=payload,
            summary=("random-dense",),
        )
    max_entries = budget_bits // RULE_SPARSE_ENTRY_BITS
    if max_entries <= 0:
        return BitRecord(
            encoding="placebo_random", budget_bits=budget_bits, n_bits=0, payload_int=0, summary=()
        )
    terrains = rng.choice(cfg.n_terrains, size=max_entries, replace=False)
    payload = 0
    summary: list[str] = []
    for i, t_raw in enumerate(terrains):
        t = int(t_raw)
        level = int(rng.integers(0, cfg.n_levels))
        entry = (t << 2) | level
        payload |= entry << (i * RULE_SPARSE_ENTRY_BITS)
        summary.append(f"T{t}=L{level}")
    return BitRecord(
        encoding="placebo_random",
        budget_bits=budget_bits,
        n_bits=max_entries * RULE_SPARSE_ENTRY_BITS,
        payload_int=payload,
        summary=tuple(summary),
    )
