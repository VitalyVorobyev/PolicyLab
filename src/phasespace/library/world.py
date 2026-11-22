"""Pure 2-D Voronoi terrain gridworld, latent cost rules, and exact Dijkstra planner."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phasespace.library.config import DEFAULT_CONFIG, LibraryConfig

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

NEIGHBOR_STEPS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEUTRAL_CELL_COST = 1.0


def observation_matrix(n_levels: int = 4, beta: float = 1.0) -> FloatArray:
    """Noisy channel `P(observed y | true level l) ∝ exp(-beta |l - y|)`, row-normalized."""
    idx = np.arange(n_levels, dtype=np.float64)
    dist = np.abs(idx[:, None] - idx[None, :])
    raw = np.exp(-beta * dist)
    return raw / raw.sum(axis=1, keepdims=True)


def expected_level_costs(level_costs: Sequence[float], obs_matrix: FloatArray) -> FloatArray:
    """Expected step cost for each true latent level `l`: `(obs_matrix @ level_costs)[l]`."""
    costs = np.asarray(level_costs, dtype=np.float64)
    if costs.shape != (obs_matrix.shape[1],):
        raise ValueError(
            f"level_costs length {costs.shape} does not match obs_matrix cols {obs_matrix.shape[1]}"
        )
    out: FloatArray = obs_matrix @ costs
    return out


@dataclass(frozen=True, slots=True)
class LatentRule:
    """Discrete latent cost level `0..n_levels-1` assigned to each terrain type."""

    levels: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.levels) == 0:
            raise ValueError("LatentRule.levels must be non-empty")


def sample_rule(rng: np.random.Generator, cfg: LibraryConfig = DEFAULT_CONFIG) -> LatentRule:
    """Sample a uniformly random latent cost level for every terrain type."""
    drawn = rng.integers(0, cfg.n_levels, size=cfg.n_terrains)
    return LatentRule(levels=tuple(int(x) for x in drawn))


def mutate_rule(
    rule: LatentRule, rng: np.random.Generator, cfg: LibraryConfig = DEFAULT_CONFIG
) -> LatentRule:
    """Return a strict derangement of `rule` where every terrain switches to another level."""
    if len(rule.levels) != cfg.n_terrains:
        raise ValueError(f"Expected {cfg.n_terrains} terrain levels, got {len(rule.levels)}")
    new_levels: list[int] = []
    for old in rule.levels:
        draw = int(rng.integers(0, cfg.n_levels - 1))
        new_levels.append(draw if draw < old else draw + 1)
    return LatentRule(levels=tuple(new_levels))


def true_terrain_expected_costs(rule: LatentRule, exp_level_costs: FloatArray) -> FloatArray:
    """Expected step cost per terrain type under `rule`."""
    idx = np.asarray(rule.levels, dtype=np.int64)
    out: FloatArray = exp_level_costs[idx]
    return out


@dataclass(frozen=True, slots=True)
class GridLayout:
    """Square navigation layout; `(0, 0)` and `(size - 1, size - 1)` are neutral (`-1`)."""

    size: int
    terrains: IntArray

    def __post_init__(self) -> None:
        if self.terrains.shape != (self.size, self.size):
            raise ValueError(
                f"Expected terrain array of shape ({self.size}, {self.size}), got {self.terrains.shape}"
            )
        if int(self.terrains[0, 0]) != -1 or int(self.terrains[self.size - 1, self.size - 1]) != -1:
            raise ValueError("Start (0, 0) and goal (size - 1, size - 1) must be neutral (-1)")


def generate_layout(rng: np.random.Generator, cfg: LibraryConfig = DEFAULT_CONFIG) -> GridLayout:
    """Generate a Voronoi terrain grid with start `(0, 0)` and goal `(size - 1, size - 1)`."""
    n = cfg.grid_size
    active = rng.choice(cfg.n_terrains, size=cfg.active_terrains_per_layout, replace=False)
    centers = rng.uniform(0.0, float(n), size=(cfg.n_voronoi_patches, 2))
    patch_terrains = np.array(
        [int(active[i % len(active)]) for i in range(cfg.n_voronoi_patches)], dtype=np.int64
    )
    coords = np.arange(n, dtype=np.float64) + 0.5
    rr, cc = np.meshgrid(coords, coords, indexing="ij")
    d2 = (rr[:, :, None] - centers[None, None, :, 0]) ** 2 + (
        cc[:, :, None] - centers[None, None, :, 1]
    ) ** 2
    nearest = np.argmin(d2, axis=2)
    grid: IntArray = patch_terrains[nearest].astype(np.int64)
    grid[0, 0] = -1
    grid[n - 1, n - 1] = -1
    return GridLayout(size=n, terrains=grid)


def _cell_cost(grid: IntArray, r: int, c: int, terrain_costs: FloatArray) -> float:
    t = int(grid[r, c])
    return NEUTRAL_CELL_COST if t < 0 else float(terrain_costs[t])


def shortest_path(
    layout: GridLayout, terrain_costs: FloatArray
) -> tuple[list[tuple[int, int]], float]:
    """Exact 4-connected Dijkstra shortest path from `(0, 0)` to `(size - 1, size - 1)`."""
    n = layout.size
    goal = (n - 1, n - 1)
    dist = np.full((n, n), math.inf, dtype=np.float64)
    dist[0, 0] = 0.0
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, int, int]] = [(0.0, 0, 0)]

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > float(dist[r, c]):
            continue
        if (r, c) == goal:
            break
        for dr, dc in NEIGHBOR_STEPS:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            nd = d + _cell_cost(layout.terrains, nr, nc, terrain_costs)
            if nd < float(dist[nr, nc]):
                dist[nr, nc] = nd
                parent[(nr, nc)] = (r, c)
                heapq.heappush(heap, (nd, nr, nc))

    path: list[tuple[int, int]] = [goal]
    cur = goal
    while cur != (0, 0):
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path, float(dist[n - 1, n - 1])


def evaluate_path_expected_cost(
    layout: GridLayout, path: Sequence[tuple[int, int]], true_terrain_costs: FloatArray
) -> float:
    """Expected cost incurred by walking `path` under `true_terrain_costs`."""
    if not path:
        raise ValueError("path must be non-empty")
    return sum(_cell_cost(layout.terrains, r, c, true_terrain_costs) for r, c in path[1:])


def oracle_cost(
    layout: GridLayout,
    rule: LatentRule,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    exp_level_costs: FloatArray | None = None,
) -> float:
    """Minimum expected path cost achievable with exact knowledge of `rule`."""
    if exp_level_costs is None:
        exp_level_costs = expected_level_costs(
            cfg.level_costs, observation_matrix(cfg.n_levels, cfg.noise_beta)
        )
    true_costs = true_terrain_expected_costs(rule, exp_level_costs)
    _, cost = shortest_path(layout, true_costs)
    return cost
