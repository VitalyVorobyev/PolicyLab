"""Configuration and deterministic hash for The Last Library (D38)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class LibraryConfig:
    """Model, environment, and channel parameters for The Last Library."""

    grid_size: int = 9
    n_terrains: int = 8
    n_levels: int = 4
    active_terrains_per_layout: int = 4
    n_voronoi_patches: int = 8
    level_costs: tuple[float, ...] = (1.0, 3.0, 7.0, 16.0)
    noise_beta: float = 1.0
    rule_trust: float = 0.88
    hedge_discount: float = 0.30
    confidence_trusts: tuple[float, ...] = (0.25, 0.55, 0.78, 0.92)
    confidence_thresholds: tuple[float, ...] = (0.35, 0.65, 0.85)
    episodes_per_generation: int = 2
    heldout_layouts_per_gen: int = 10
    n_generations: int = 10
    change_generation: int = 5
    bit_budgets: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 40, 60)

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.grid_size}")
        if self.n_terrains != 8:
            raise ValueError(f"n_terrains must be 8 for fixed 3-bit packing, got {self.n_terrains}")
        if self.n_levels != 4:
            raise ValueError(f"n_levels must be 4 for fixed 2-bit packing, got {self.n_levels}")
        if len(self.level_costs) != self.n_levels:
            raise ValueError("level_costs length must equal n_levels")
        if any(c <= 0.0 for c in self.level_costs):
            raise ValueError("level_costs must all be strictly positive")
        if self.noise_beta <= 0.0:
            raise ValueError(f"noise_beta must be positive, got {self.noise_beta}")
        if not (0 < self.active_terrains_per_layout <= self.n_terrains):
            raise ValueError("active_terrains_per_layout must be in [1, n_terrains]")
        if self.n_voronoi_patches < self.active_terrains_per_layout:
            raise ValueError("n_voronoi_patches must be >= active_terrains_per_layout")
        if not (0.25 < self.rule_trust < 1.0):
            raise ValueError(f"rule_trust must be in (0.25, 1.0), got {self.rule_trust}")
        if not (0.0 < self.hedge_discount < 1.0):
            raise ValueError(f"hedge_discount must be in (0.0, 1.0), got {self.hedge_discount}")
        if len(self.confidence_trusts) != 4:
            raise ValueError("confidence_trusts must have 4 entries")
        if len(self.confidence_thresholds) != 3:
            raise ValueError("confidence_thresholds must have 3 entries")
        if not (
            0.25
            < self.confidence_thresholds[0]
            < self.confidence_thresholds[1]
            < self.confidence_thresholds[2]
            < 1.0
        ):
            raise ValueError("confidence_thresholds must strictly increase inside (0.25, 1.0)")
        if self.episodes_per_generation < 1 or self.heldout_layouts_per_gen < 1:
            raise ValueError("episodes and heldout counts must be >= 1")
        if not (1 <= self.change_generation < self.n_generations):
            raise ValueError(
                "change_generation must satisfy 1 <= change_generation < n_generations"
            )
        if any(b < 0 for b in self.bit_budgets):
            raise ValueError("bit_budgets must be non-negative")


DEFAULT_CONFIG: Final = LibraryConfig()


def config_to_dict(cfg: LibraryConfig) -> dict[str, Any]:
    """Serialize `cfg` to JSON-compatible primitives in sorted key order."""
    return {
        "active_terrains_per_layout": cfg.active_terrains_per_layout,
        "bit_budgets": list(cfg.bit_budgets),
        "change_generation": cfg.change_generation,
        "confidence_thresholds": list(cfg.confidence_thresholds),
        "confidence_trusts": list(cfg.confidence_trusts),
        "episodes_per_generation": cfg.episodes_per_generation,
        "grid_size": cfg.grid_size,
        "hedge_discount": cfg.hedge_discount,
        "heldout_layouts_per_gen": cfg.heldout_layouts_per_gen,
        "level_costs": list(cfg.level_costs),
        "n_generations": cfg.n_generations,
        "n_levels": cfg.n_levels,
        "n_terrains": cfg.n_terrains,
        "n_voronoi_patches": cfg.n_voronoi_patches,
        "noise_beta": cfg.noise_beta,
        "rule_trust": cfg.rule_trust,
    }


def config_sha256(cfg: LibraryConfig = DEFAULT_CONFIG) -> str:
    """SHA-256 hex digest of the canonical configuration JSON."""
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
