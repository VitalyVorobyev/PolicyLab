"""Exact tabular Bayesian belief updates, entropies, and KL divergence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from phasespace.library.world import FloatArray


def uniform_belief(n_terrains: int = 8, n_levels: int = 4) -> FloatArray:
    """Uniform factorized prior `(n_terrains, n_levels)`."""
    return np.full((n_terrains, n_levels), 1.0 / n_levels, dtype=np.float64)


def bayes_update(
    belief: FloatArray, terrain_id: int, observed_y: int, obs_matrix: FloatArray
) -> FloatArray:
    """Return a copy of `belief` updated with single observation `(terrain_id, observed_y)`."""
    updated = belief.copy()
    row = updated[terrain_id] * obs_matrix[:, observed_y]
    total = float(row.sum())
    if total <= 0.0:
        raise ValueError(f"Zero posterior normalizer for terrain={terrain_id}, y={observed_y}")
    updated[terrain_id] = row / total
    return updated


def bayes_update_batch(
    belief: FloatArray, observations: Sequence[tuple[int, int]], obs_matrix: FloatArray
) -> FloatArray:
    """Return a copy of `belief` updated over a sequence of `(terrain_id, observed_y)` pairs."""
    updated = belief.copy()
    for terrain_id, observed_y in observations:
        row = updated[terrain_id] * obs_matrix[:, observed_y]
        total = float(row.sum())
        if total <= 0.0:
            raise ValueError(f"Zero posterior normalizer for terrain={terrain_id}, y={observed_y}")
        updated[terrain_id] = row / total
    return updated


def terrain_entropy_bits(belief: FloatArray) -> FloatArray:
    """Per-terrain Shannon entropy in bits."""
    safe = np.where(belief > 0.0, belief, 1.0)
    log_p = np.where(belief > 0.0, np.log2(safe), 0.0)
    out: FloatArray = -np.sum(belief * log_p, axis=1)
    return np.maximum(out, 0.0)


def belief_entropy_bits(belief: FloatArray) -> float:
    """Total Shannon entropy across all terrains in bits."""
    return float(np.sum(terrain_entropy_bits(belief)))


def kl_divergence_nats(p: FloatArray, q: FloatArray) -> float:
    """Sum of row-wise `KL(p[t] || q[t])` in nats."""
    q_safe = np.maximum(q, 1e-15)
    ratio = np.where(p > 0.0, np.log(np.maximum(p, 1e-15) / q_safe), 0.0)
    return float(np.sum(np.where(p > 0.0, p * ratio, 0.0)))


def expected_terrain_costs(belief: FloatArray, exp_level_costs: FloatArray) -> FloatArray:
    """Expected cost per terrain type under `belief`."""
    out: FloatArray = belief @ exp_level_costs
    return out


def map_levels(belief: FloatArray) -> tuple[int, ...]:
    """Deterministic argmax cost level for each terrain type."""
    return tuple(int(x) for x in np.argmax(belief, axis=1))
