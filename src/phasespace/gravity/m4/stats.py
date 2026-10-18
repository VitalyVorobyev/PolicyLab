"""Statistical analysis of protocol sections 8 and 11.

- Paired differences T - X by seed (`paired_differences`).
- Exact two-sided sign-flip permutation test over all 2^n sign vectors
  (`sign_flip_pvalue`): p = #{s : |mean(s * d)| >= |mean(d)|} / 2^n, with a relative
  tolerance of 1e-12 so that ties in floating point count as ties.
- Holm-Bonferroni step-down adjustment across a family (`holm`).
- 95 % percentile bootstrap CI of the mean paired difference over seeds (10 000
  resamples, seed 4242) (`bootstrap_ci`).
- The pilot's seed-count rule n = clamp(ceil((z_a + z_b)^2 max (sigma_d / delta)^2),
  10, 16) (`seed_count`).
- H2: Spearman correlation with a seed-stratified permutation test (`spearman`,
  `stratified_permutation_pvalue`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.m4.config import PROTOCOL, Statistics

_STATS = PROTOCOL.statistics


def paired_differences(
    treatment: Mapping[int, float], baseline: Mapping[int, float]
) -> tuple[list[int], FloatArray]:
    """(seeds, T - X) over the seeds both arms have, in increasing seed order."""
    if set(treatment) != set(baseline):
        raise ValueError("paired analysis needs the same seeds in both arms")
    seeds = sorted(treatment)
    return seeds, np.array([treatment[s] - baseline[s] for s in seeds], dtype=np.float64)


def sign_flip_pvalue(differences: Sequence[float] | FloatArray, max_n: int = 20) -> float:
    """Exact two-sided sign-flip test of mean(d) = 0 (all 2^n sign vectors)."""
    d = np.asarray(differences, dtype=np.float64)
    n = d.size
    if n == 0:
        return math.nan
    if n > max_n:
        raise ValueError(f"exact enumeration limited to n <= {max_n}")
    observed = abs(float(d.mean()))
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)), dtype=np.float64)
    means = np.abs(signs @ d) / n
    tolerance = 1e-12 * max(1.0, observed)
    return float(np.count_nonzero(means >= observed - tolerance)) / signs.shape[0]


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(
    differences: Sequence[float] | FloatArray, stats: Statistics = _STATS
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over seeds (resampling seeds with replacement)."""
    d = np.asarray(differences, dtype=np.float64)
    if d.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(stats.bootstrap_seed)
    idx = rng.integers(0, d.size, size=(stats.bootstrap_resamples, d.size))
    means = d[idx].mean(axis=1)
    tail = (1.0 - stats.confidence) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class Contrast:
    name: str
    seeds: tuple[int, ...]
    mean_difference: float
    ci: tuple[float, float]
    p_value: float
    p_holm: float
    significant: bool


def primary_contrasts(
    contrasts: Mapping[str, tuple[Mapping[int, float], Mapping[int, float]]],
    stats: Statistics = _STATS,
) -> list[Contrast]:
    """Paired T - X contrasts with sign-flip p-values, Holm across the given family."""
    rows: list[tuple[str, list[int], FloatArray, float]] = []
    for name, (t, x) in contrasts.items():
        seeds, d = paired_differences(t, x)
        rows.append((name, seeds, d, sign_flip_pvalue(d)))
    adjusted = holm([r[3] for r in rows])
    return [
        Contrast(
            name=name,
            seeds=tuple(seeds),
            mean_difference=float(d.mean()),
            ci=bootstrap_ci(d, stats),
            p_value=p,
            p_holm=q,
            significant=q <= stats.alpha,
        )
        for (name, seeds, d, p), q in zip(rows, adjusted, strict=True)
    ]


def seed_count(sigma_over_delta: Sequence[float], stats: Statistics = _STATS) -> tuple[int, int]:
    """(n, unclamped n) from the pilot SDs of paired differences divided by the effects."""
    worst = max(sigma_over_delta)
    raw = math.ceil((stats.z_alpha + stats.z_power) ** 2 * worst**2)
    lo, hi = stats.seed_count_bounds
    return min(hi, max(lo, raw)), raw


def summary(values: Sequence[float]) -> dict[str, float]:
    """mean, median, SD (n - 1), min and max."""
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "sd": float(a.std(ddof=1)) if a.size > 1 else math.nan,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    return float(np.corrcoef(rx, ry)[0, 1])


def stratified_permutation_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    strata: Sequence[int],
    *,
    resamples: int = 10_000,
    seed: int = _STATS.bootstrap_seed,
) -> tuple[float, float]:
    """(pooled Spearman rho, two-sided p) permuting y within each stratum (seed)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    sa = np.asarray(strata)
    rho = spearman(xa.tolist(), ya.tolist())
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(sa == s) for s in np.unique(sa)]
    count = 0
    for _ in range(resamples):
        yp = ya.copy()
        for g in groups:
            yp[g] = ya[rng.permutation(g)]
        if abs(spearman(xa.tolist(), yp.tolist())) >= abs(rho) - 1e-12:
            count += 1
    return rho, (count + 1) / (resamples + 1)
