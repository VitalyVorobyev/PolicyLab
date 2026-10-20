"""C4 hypothesis stage (pure): the tests, families and verdicts of protocol §9.

Inputs are recorded numbers only: the per-game totals of the cross-play run (one array
per policy, column, mode and seed; common random numbers within a column), its
diagnostics, the distillation run, the training runs' plateau summaries and, for H2-B4,
the B4 games. Nothing is simulated here.

**Tests of one cell** (§9.2). For a planner row P (S seeds) and a comparator Q in one
column, D = mean(P) - mean(Q):

1. the crossed bootstrap (`crossplay.crossed_bootstrap_stats`; B resamples of seeds of P,
   seeds of Q and games, seed `c4_config.BOOTSTRAP_SEED`), with the one-sided p-values
   p(D < theta) = (1 + #{D* >= theta}) / (B + 1) and p(D > theta) = (1 + #{D* <= theta}) /
   (B + 1);
2. the seed-level t-test (`crossplay.seed_level_t`: one-sample t on the per-seed
   differences against a rule, Welch's t between two planner rows), one-sided at theta;
3. the per-seed signs: how many d_s lie below and above theta.

**Families.** Holm's step-down procedure (`holm`) is applied within each family,
separately to the bootstrap and to the t p-values, at family-wise alpha. A cell's claim is
**confirmed** only if both tests reject under Holm (B4: the unpaired game bootstrap
only, one seed per row).

**Verdicts** (§9.3-9.7), exactly as written in the protocol:

- H1 refuted if any H1-reversal cell is confirmed; supported if all 12 H1-transfer cells
  are confirmed; otherwise not refuted (the confirmed and unconfirmed cells listed);
- H2 supported if any H2a-loss cell is confirmed; refuted if all three H2a-noninferior
  cells are confirmed; otherwise inconclusive; H2b reported with its own confirmed cells;
- own-model adequacy: a row is optimiser-limited if its own-model loss is confirmed; a
  confirmed own-model gain is labelled possible exploitation if the §9.5 flags hold;
- every H1 cell confirmed in either direction is annotated "on novel states" when its
  §9.5 flags hold;
- S1 (no-retention rows) and S2 (mixture row): the same families, labelled secondary;
- H2-B4 per B4 variant: confirmatory only if C2 validated that variant (§9.4).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from phasespace.citizens.rl import c4_config, crossplay
from phasespace.citizens.rl.citizens import TRAINABLE_MODELS
from phasespace.citizens.rl.crossplay import Claim

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

DELTA: Final = c4_config.DELTA
K22: Final = c4_config.K_PUBLISHED_RULE
H2_MODELS: Final = tuple(m for m in TRAINABLE_MODELS if m != "B1")
"""The closed-loop-rejected training models whose planners H2 puts on B1."""
NOT_VALIDATED_B4: Final = "B4-{variant} not validated in closed loop (C2)"


# --- one-sided p-values and Holm -------------------------------------------------------------------


def bootstrap_p(stats: FloatArray, theta: float, claim: Claim) -> float:
    """One-sided bootstrap p-value of the claim D < theta ("less") or D > theta
    ("greater") from the resampled differences (protocol §9.2)."""
    b = stats.size
    count = int((stats >= theta).sum()) if claim == "less" else int((stats <= theta).sum())
    return (1.0 + count) / (b + 1.0)


def holm(pvalues: Sequence[float | None], alpha: float = c4_config.ALPHA) -> dict[str, Any]:
    """Holm's step-down procedure: sort p ascending and reject the i-th (1-based) while
    p_(i) <= alpha / (m - i + 1); stop at the first failure. A null p is never rejected
    (it counts in m). Also the Holm-adjusted p-values, min(1, max_{j <= i} (m - j + 1)
    p_(j)), in the input order (null stays null)."""
    m = len(pvalues)
    order = sorted(
        (i for i in range(m) if pvalues[i] is not None),
        key=lambda i: (float(pvalues[i] or 0.0), i),
    )
    reject = [False] * m
    adjusted: list[float | None] = [None] * m
    stopped = False
    running = 0.0
    for rank, i in enumerate(order):
        p = float(pvalues[i] or 0.0)
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
        if not stopped and p <= alpha / (m - rank):
            reject[i] = True
        else:
            stopped = True
    return {"reject": reject, "adjusted": adjusted, "m": m, "alpha": alpha}


# --- cells and families ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One comparison: row (S, J) minus comparator (S', J) on the same J games."""

    name: str
    row: str
    column: str
    comparator: str
    a: FloatArray
    b: FloatArray
    mode: str = "deterministic"


class BootstrapCache:
    """Crossed-bootstrap differences per cell (the same draws serve both claims)."""

    def __init__(self, n_resamples: int, seed: int) -> None:
        self.n_resamples = n_resamples
        self.seed = seed
        self._stats: dict[tuple[str, str, bool], FloatArray] = {}

    def stats(self, cell: Cell, paired: bool = True) -> FloatArray:
        key = (cell.name, cell.mode, paired)
        if key not in self._stats:
            if paired:
                self._stats[key] = crossplay.crossed_bootstrap_stats(
                    cell.a, cell.b, self.n_resamples, self.seed
                )
            else:
                self._stats[key] = unpaired_bootstrap_stats(
                    cell.a[0], cell.b[0], self.n_resamples, self.seed
                )
        return self._stats[key]


def unpaired_bootstrap_stats(
    a: FloatArray, b: FloatArray, n_resamples: int, seed: int
) -> FloatArray:
    """(B,) resampled mean(a) - mean(b) with games resampled within each row
    independently (B4 rows cannot be paired: unseeded sampling, §9.4)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples)
    for r in range(n_resamples):
        out[r] = (
            a[rng.integers(0, a.size, a.size)].mean() - b[rng.integers(0, b.size, b.size)].mean()
        )
    return out


def run_family(
    name: str,
    cells: Sequence[Cell],
    claim: Claim,
    theta: float,
    cache: BootstrapCache,
    *,
    alpha: float = c4_config.ALPHA,
    seed_level: bool = True,
    paired: bool = True,
    exploratory: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Both tests on every cell, Holm within the family for each test, and the confirmed
    cells (both reject; the bootstrap alone when `seed_level` is False). A cell named in
    `exploratory` (with its reason) counts in m but is never confirmed."""
    exploratory = exploratory or {}
    out_cells: list[dict[str, Any]] = []
    p_boot: list[float | None] = []
    p_t: list[float | None] = []
    for cell in cells:
        stats = cache.stats(cell, paired)
        d = crossplay.per_seed_differences(cell.a, cell.b) if paired else None
        t = crossplay.seed_level_t(cell.a, cell.b, theta, claim) if seed_level else None
        p_boot.append(bootstrap_p(stats, theta, claim))
        p_t.append(None if t is None else t["p"])
        record: dict[str, Any] = {
            "cell": cell.name,
            "row": cell.row,
            "column": cell.column,
            "comparator": cell.comparator,
            "difference": float(cell.a.mean() - cell.b.mean()),
            "p_bootstrap": p_boot[-1],
            "seed_level_t": t,
        }
        if d is not None:
            record["per_seed_difference"] = [float(x) for x in d]
            record["seeds_below_theta"] = int((d < theta).sum())
            record["seeds_above_theta"] = int((d > theta).sum())
        if cell.name in exploratory:
            record["exploratory"] = exploratory[cell.name]
        out_cells.append(record)
    hb = holm(p_boot, alpha)
    ht = holm(p_t, alpha) if seed_level else None
    confirmed: list[str] = []
    for i, record in enumerate(out_cells):
        record["p_bootstrap_holm"] = hb["adjusted"][i]
        record["reject_bootstrap"] = hb["reject"][i]
        if ht is not None:
            record["p_t_holm"] = ht["adjusted"][i]
            record["reject_t"] = ht["reject"][i]
        both = hb["reject"][i] and (ht is None or ht["reject"][i])
        record["confirmed"] = bool(both and record["cell"] not in exploratory)
        if record["confirmed"]:
            confirmed.append(record["cell"])
    return {
        "family": name,
        "claim": f"D {'<' if claim == 'less' else '>'} {theta:g}",
        "theta": theta,
        "m": len(cells),
        "alpha": alpha,
        "confirmation": "crossed bootstrap and seed-level t both reject under Holm"
        if seed_level
        else "unpaired game bootstrap rejects under Holm (one seed per row)",
        "cells": out_cells,
        "confirmed": confirmed,
    }


# --- the matrix ------------------------------------------------------------------------------------


class Matrix:
    """Per-game totals of the cross-play run: (S, J) per (policy, column, mode), seeds
    in ascending order; rules are deterministic only."""

    def __init__(self, series: Sequence[Mapping[str, Any]]) -> None:
        rows: dict[tuple[str, str, str], list[tuple[int, list[float]]]] = {}
        for s in series:
            key = (str(s["policy"]), str(s["column"]), str(s["mode"]))
            seed = -1 if s["seed"] is None else int(s["seed"])
            rows.setdefault(key, []).append((seed, list(s["total_surplus"])))
        self._data = {
            k: np.asarray([v for _, v in sorted(entries)], dtype=np.float64)
            for k, entries in rows.items()
        }

    def get(self, policy: str, column: str, mode: str) -> FloatArray:
        key = (policy, column, mode)
        if key not in self._data and mode != "deterministic":
            key = (policy, column, "deterministic")  # a rule has one (deterministic) mode
        if key not in self._data:
            raise KeyError((policy, column, mode))
        return self._data[key]

    def has(self, policy: str) -> bool:
        return any(k[0] == policy for k in self._data)


def planner(model: str, retention: bool = True) -> str:
    return crossplay.planner_name(model, retention)


def k_star(model: str) -> str:
    return crossplay.k_star_name(model)


def _cell(matrix: Matrix, row: str, column: str, comparator: str, mode: str) -> Cell:
    comparator_mode = mode if comparator.startswith("Planner") else "deterministic"
    return Cell(
        name=f"{row}@{column} vs {comparator}",
        row=row,
        column=column,
        comparator=comparator,
        a=matrix.get(row, column, mode),
        b=matrix.get(comparator, column, comparator_mode),
        mode=mode,
    )


def off_diagonal(models: Sequence[str]) -> list[tuple[str, str]]:
    return [(x, y) for x in models for y in models if x != y]


# --- §9.5 flags ------------------------------------------------------------------------------------


def _mean_or_none(values: Sequence[Any]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return float(np.mean(kept)) if kept else None


def exploitation_flags(
    summary: Mapping[str, Any], policy: str, column: str, mode: str = "deterministic"
) -> dict[str, Any]:
    """The §9.5 flags of one cell from the cross-play summary: the B1 mean member TV
    ratio against k = 22 in the same column, and (B2 column only) the input-support
    fractions above k = 22's. Null ratios carry a reason; B3/B23 support numbers are
    state novelty and reported without entering the flag."""
    cells = summary["cells"]
    cell = cells[policy][column][mode] or cells[policy][column]["deterministic"]
    ref = cells[K22][column]["deterministic"]
    tv = _mean_or_none([d["mean_tv"] for d in cell["b1_disagreement"] if d is not None])
    tv_ref = _mean_or_none([d["mean_tv"] for d in ref["b1_disagreement"] if d is not None])
    out: dict[str, Any] = {"mean_tv": tv, "mean_tv_k22": tv_ref}
    if tv is None or tv_ref is None or tv_ref == 0.0:
        out["tv_ratio"] = None
        out["tv_ratio_null_reason"] = "no decisions or zero k = 22 disagreement"
        tv_flag = False
    else:
        out["tv_ratio"] = tv / tv_ref
        tv_flag = out["tv_ratio"] > c4_config.EXPLOIT_TV_RATIO
    support = [s for s in cell.get("input_support", []) if s is not None]
    support_ref = [s for s in ref.get("input_support", []) if s is not None]
    support_flag = False
    if support and support_ref:
        rng = _mean_or_none([s["outside_range_any"] for s in support])
        grid = _mean_or_none([s["off_grid"] for s in support])
        rng_ref = _mean_or_none([s["outside_range_any"] for s in support_ref])
        grid_ref = _mean_or_none([s["off_grid"] for s in support_ref])
        out["input_support"] = {
            "kind": support[0]["kind"],
            "label": support[0]["label"],
            "outside_range_any": rng,
            "off_grid": grid,
            "outside_range_any_k22": rng_ref,
            "off_grid_k22": grid_ref,
        }
        if column in c4_config.EXPLOIT_SUPPORT_MODELS:
            margin = c4_config.EXPLOIT_SUPPORT_MARGIN
            support_flag = (rng is not None and rng_ref is not None and rng - rng_ref > margin) or (
                grid is not None and grid_ref is not None and grid - grid_ref > margin
            )
    out["tv_flag"] = tv_flag
    out["support_flag"] = support_flag
    out["flagged"] = tv_flag or support_flag
    return out


# --- verdicts --------------------------------------------------------------------------------------


def h1_verdict(reversal: Mapping[str, Any], transfer: Mapping[str, Any]) -> dict[str, Any]:
    all_cells = [c["cell"] for c in transfer["cells"]]
    if reversal["confirmed"]:
        verdict = "refuted"
    elif len(transfer["confirmed"]) == transfer["m"]:
        verdict = "supported"
    else:
        verdict = "not refuted"
    return {
        "verdict": verdict,
        "confirmed_reversals": list(reversal["confirmed"]),
        "confirmed_transfers": list(transfer["confirmed"]),
        "transfer_not_shown": [c for c in all_cells if c not in transfer["confirmed"]],
        "rule": "refuted if any reversal cell is confirmed; supported if all transfer cells "
        "are confirmed; otherwise not refuted",
    }


def h2_verdict(loss: Mapping[str, Any], noninferior: Mapping[str, Any]) -> dict[str, Any]:
    if loss["confirmed"]:
        verdict = "supported"
    elif len(noninferior["confirmed"]) == noninferior["m"]:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "confirmed_losses": list(loss["confirmed"]),
        "confirmed_noninferior": list(noninferior["confirmed"]),
        "rule": "supported (one-step validation was not enough) if any loss cell is "
        "confirmed; refuted if all non-inferiority cells are confirmed; otherwise "
        "inconclusive",
    }


def regret(
    matrix: Matrix, own: str, other: str, column: str, mode: str, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Descriptive regret own@column - other@column with a crossed-bootstrap 95 % CI."""
    a, b = matrix.get(own, column, mode), matrix.get(other, column, mode)
    mean, lo, hi = crossplay.group_bootstrap(a, b, n_resamples, seed)
    return {
        "own": own,
        "other": other,
        "column": column,
        "mode": mode,
        "regret": mean,
        "ci": [lo, hi],
    }


def convergence_labels(plateaus: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """§6.1: a run is "not converged" if its monitoring plateau iteration is above 900
    or its tail slope above 16 coins per 100 iterations."""
    out: dict[str, Any] = {}
    for run_id, p in plateaus.items():
        late = p["plateau_iteration"] > c4_config.CONVERGENCE_MAX_PLATEAU_ITERATION
        steep = p["tail_slope_per_100_iterations"] > c4_config.CONVERGENCE_MAX_TAIL_SLOPE
        out[run_id] = {**p, "not_converged": bool(late or steep)}
    return out


def main_families(
    matrix: Matrix,
    summary: Mapping[str, Any],
    mode: str,
    cache: BootstrapCache,
) -> dict[str, Any]:
    """H1, H2a, H2b and own-model adequacy of the main rows in one mode (§9.3)."""
    models = TRAINABLE_MODELS
    pairs = off_diagonal(models)
    h1_cells = [_cell(matrix, planner(x), y, k_star(y), mode) for x, y in pairs]
    families = {
        "H1-reversal": run_family("H1-reversal", h1_cells, "less", -DELTA, cache),
        "H1-transfer": run_family("H1-transfer", h1_cells, "greater", DELTA, cache),
    }
    h2_cells = [_cell(matrix, planner(x), "B1", k_star("B1"), mode) for x in H2_MODELS]
    families["H2a-loss"] = run_family("H2a-loss", h2_cells, "less", -DELTA, cache)
    families["H2a-noninferior"] = run_family("H2a-noninferior", h2_cells, "greater", -DELTA, cache)
    h2b_cells = [_cell(matrix, planner(x), "B1", planner("B1"), mode) for x in H2_MODELS]
    families["H2b-loss"] = run_family("H2b-loss", h2b_cells, "less", -DELTA, cache)
    own_cells = [_cell(matrix, planner(x), x, k_star(x), mode) for x in models]
    families["own-model-limited"] = run_family(
        "own-model-limited", own_cells, "less", -DELTA, cache
    )
    families["own-model-gain"] = run_family("own-model-gain", own_cells, "greater", DELTA, cache)
    flags = {
        f"{planner(x)}@{y}": exploitation_flags(summary, planner(x), y, mode)
        for x in models
        for y in models
    }
    confirmed_h1 = set(families["H1-reversal"]["confirmed"]) | set(
        families["H1-transfer"]["confirmed"]
    )
    annotations = {
        c.name: "on novel states"
        for c, (x, y) in zip(h1_cells, pairs, strict=True)
        if c.name in confirmed_h1 and flags[f"{planner(x)}@{y}"]["flagged"]
    }
    rows: dict[str, Any] = {}
    for x, cell in zip(models, own_cells, strict=True):
        gain = cell.name in families["own-model-gain"]["confirmed"]
        rows[planner(x)] = {
            "optimiser_limited": cell.name in families["own-model-limited"]["confirmed"],
            "own_model_gain": gain,
            "possible_exploitation": bool(gain and flags[f"{planner(x)}@{x}"]["flagged"]),
        }
    return {
        "families": families,
        "H1": h1_verdict(families["H1-reversal"], families["H1-transfer"]),
        "H2": h2_verdict(families["H2a-loss"], families["H2a-noninferior"]),
        "H2b": {"confirmed_losses": list(families["H2b-loss"]["confirmed"])},
        "rows": rows,
        "flags": flags,
        "annotations": annotations,
        "regret": [
            regret(matrix, planner(y), planner(x), y, mode, cache.n_resamples, cache.seed)
            for x, y in pairs
        ],
    }


def secondary_s1(matrix: Matrix, mode: str, cache: BootstrapCache) -> dict[str, Any]:
    """S1: the H1 families for the no-retention rows (secondary)."""
    pairs = off_diagonal(TRAINABLE_MODELS)
    cells = [_cell(matrix, planner(x, False), y, k_star(y), mode) for x, y in pairs]
    reversal = run_family("S1-reversal", cells, "less", -DELTA, cache)
    transfer = run_family("S1-transfer", cells, "greater", DELTA, cache)
    return {
        "label": "secondary (3 seeds; seed-level test with df = 2 is weak)",
        "families": {"S1-reversal": reversal, "S1-transfer": transfer},
        "H1": h1_verdict(reversal, transfer),
        "entropy_confound": "entropy_coef = 0, so no entropy bonus enters the loss; the "
        "4-way head's Gaussian entropy at equal sigma is about 20 % smaller than the 5-way's",
    }


def secondary_s2(matrix: Matrix, mode: str, cache: BootstrapCache) -> dict[str, Any]:
    """S2: Planner[Mix] against k*[Y] in every column, and its regret against Planner[Y]."""
    mix = planner(c4_config.MIXTURE_MODEL)
    cells = [_cell(matrix, mix, y, k_star(y), mode) for y in TRAINABLE_MODELS]
    reversal = run_family("S2-reversal", cells, "less", -DELTA, cache)
    transfer = run_family("S2-transfer", cells, "greater", DELTA, cache)
    return {
        "label": "secondary (robust mixture planner, 3 seeds)",
        "families": {"S2-reversal": reversal, "S2-transfer": transfer},
        "verdict": h1_verdict(reversal, transfer),
        "regret": [
            regret(matrix, planner(y), mix, y, mode, cache.n_resamples, cache.seed)
            for y in TRAINABLE_MODELS
        ],
    }


# --- §9.6 mechanism agreement --------------------------------------------------------------------


def _l1(u: FloatArray, v: FloatArray) -> float:
    return float(np.abs(u - v).mean())


def _within(seeds: FloatArray) -> float | None:
    """Mean pairwise L1 among one model's seeds (None with fewer than 2 seeds)."""
    pairs = list(itertools.combinations(range(seeds.shape[0]), 2))
    if not pairs:
        return None
    return float(np.mean([_l1(seeds[i], seeds[j]) for i, j in pairs]))


def mechanism_agreement(planners: Sequence[Mapping[str, Any]], retention: bool) -> dict[str, Any]:
    """§9.6 among the four planner rows of one action head, from the distillation
    run's planner records (`c4.distill_planner`)."""
    names = {planner(m, retention): m for m in TRAINABLE_MODELS}
    by_model: dict[str, list[Mapping[str, Any]]] = {m: [] for m in TRAINABLE_MODELS}
    for rec in planners:
        if rec["policy"] in names:
            by_model[names[rec["policy"]]].append(rec)
    w = {m: np.asarray([r["w_rule"]["w"] for r in recs]) for m, recs in by_model.items() if recs}
    kappa = {
        m: np.asarray([r["w_rule"]["kept_share"] for r in recs])
        for m, recs in by_model.items()
        if recs
    }
    rows = {
        m: np.asarray([r["w_rule"]["informative_rows"] for r in recs])
        for m, recs in by_model.items()
        if recs
    }
    fidelity: dict[str, Any] = {}
    for m, recs in by_model.items():
        r2s = [r["w_rule"]["r2"] for r in recs]
        low = any(v is None or v < c4_config.FIDELITY_MIN_R2 for v in r2s)
        fidelity[m] = {
            "w_rule_r2": r2s,
            "interpolating_k_r2": [r["interpolating_k"]["r2"] for r in recs],
            "tree_r2": [r["tree"]["r2"] for r in recs],
            "low_fidelity": low or not recs,
        }
    pairs_out: list[dict[str, Any]] = []
    for x, y in itertools.combinations(TRAINABLE_MODELS, 2):
        if x not in w or y not in w:
            pairs_out.append({"pair": [x, y], "label": None, "null_reason": "missing row"})
            continue
        common = (
            np.minimum(rows[x].min(axis=0), rows[y].min(axis=0)) >= c4_config.AGREEMENT_MIN_ROWS
        )
        if not common.any():
            pairs_out.append(
                {"pair": [x, y], "label": None, "null_reason": "no bin with enough rows"}
            )
            continue
        l1 = _l1(w[x][:, common].mean(axis=0), w[y][:, common].mean(axis=0))
        wx, wy = _within(w[x][:, common]), _within(w[y][:, common])
        ref = max([v for v in (wx, wy) if v is not None], default=0.0)
        threshold = max(c4_config.AGREEMENT_L1_FLOOR, c4_config.AGREEMENT_WITHIN_FACTOR * ref)
        pairs_out.append(
            {
                "pair": [x, y],
                "bins": [int(b) for b in np.flatnonzero(common)],
                "l1": l1,
                "within": {x: wx, y: wy},
                "threshold": threshold,
                "label": "same w(R)" if l1 <= threshold else "different",
                "kappa_l1": _l1(kappa[x][:, common].mean(axis=0), kappa[y][:, common].mean(axis=0)),
            }
        )
    adequate = {m for m, f in fidelity.items() if not f["low_fidelity"]}
    labelled = [p for p in pairs_out if p["label"] is not None]
    if (
        len(labelled) == len(pairs_out)
        and all(p["label"] == "same w(R)" for p in labelled)
        and adequate == set(TRAINABLE_MODELS)
    ):
        sentence = "same mechanism"
    elif any(p["label"] == "different" and set(p["pair"]) <= adequate for p in labelled):
        sentence = "different mechanisms"
    else:
        sentence = "mixed"
    between = [p["l1"] for p in labelled]
    return {
        "action_head": "5-way (retention)" if retention else "4-way (no retention)",
        "pairs": pairs_out,
        "fidelity": fidelity,
        "mean_between_model_l1": float(np.mean(between)) if between else None,
        "summary": sentence,
        "distilled_k": {
            m: {
                "per_seed": [r["interpolating_k"]["k"] for r in recs],
                "k_star": recs[0]["total_surplus_mean"]["k_star"] if recs else None,
                "method_bias_ln_k": recs[0]["method_bias_ln_k"]["value"] if recs else None,
            }
            for m, recs in by_model.items()
        },
        "endgame_kept_share": {
            m: _mean_or_none([r["kept_share_coin_weighted"]["endgame"] for r in recs])
            for m, recs in by_model.items()
        },
        "recovery_of_planner_gain": {
            m: [r["total_surplus_mean"]["recovery_of_planner_gain"]["value"] for r in recs]
            for m, recs in by_model.items()
        },
        "note": "distilled-rule surplus is in-sample (the distillation games)",
    }


def distilled_crossplay_matrix(cross: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seed means of the distilled rules relative to k*[column] (§9.6 item 5)."""
    cells: dict[str, dict[str, Any]] = {}
    for policy in sorted({c["policy"] for c in cross}):
        for column in TRAINABLE_MODELS:
            chosen = [c for c in cross if c["policy"] == policy and c["column"] == column]
            if not chosen:
                continue
            cells.setdefault(policy, {})[column] = {
                "w_rule_minus_k_star": float(np.mean([c["w_rule_minus_k_star"] for c in chosen])),
                "distilled_k_minus_k_star": float(
                    np.mean([c["distilled_k_minus_k_star"] for c in chosen])
                ),
                "seeds": [c["seed"] for c in chosen],
            }
    return cells


def retention_attribution(
    main_reversals: int, noret_reversals: int, main_l1: float | None, noret_l1: float | None
) -> dict[str, Any]:
    """S1: disagreement is attributable to retention if the main rows have at least one
    confirmed H1 reversal and the no-retention rows none, or the no-retention planners'
    mean between-model w(R) L1 is below half the main planners'."""
    by_reversal = main_reversals >= 1 and noret_reversals == 0
    by_l1 = (
        main_l1 is not None
        and noret_l1 is not None
        and noret_l1 < c4_config.RETENTION_L1_RATIO * main_l1
    )
    return {
        "attributable_to_retention": bool(by_reversal or by_l1),
        "by_h1_reversal": by_reversal,
        "by_w_l1": by_l1,
        "main_confirmed_reversals": main_reversals,
        "noret_confirmed_reversals": noret_reversals,
        "main_mean_between_model_l1": main_l1,
        "noret_mean_between_model_l1": noret_l1,
    }


# --- H2-B4 ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class B4Row:
    """One B4 row's complete games (§8 metrics; total surplus = 160 x surplus)."""

    name: str
    total_surplus: FloatArray
    gini: FloatArray
    depletion_round: FloatArray
    players_active: FloatArray
    discarded: int


def _game_ci(x: FloatArray, n_resamples: int, seed: int) -> list[float] | None:
    if x.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n_resamples)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def b4_hypotheses(
    rows: Mapping[str, B4Row], validated: bool, variant: str, cache: BootstrapCache
) -> dict[str, Any]:
    """H2-B4 for one variant (§9.4): confirmatory families when the variant is a
    validated evaluation population, descriptive numbers always. A row with fewer than
    `B4_MIN_COMPLETE_GAMES` complete games is exploratory (never confirmed)."""
    descriptive = {
        name: {
            "games_complete": int(r.total_surplus.size),
            "games_discarded": r.discarded,
            "exploratory": r.total_surplus.size < c4_config.B4_MIN_COMPLETE_GAMES,
            **{
                metric: {
                    "mean": float(values.mean()) if values.size else None,
                    "ci": _game_ci(values, cache.n_resamples, cache.seed),
                }
                for metric, values in (
                    ("total_surplus", r.total_surplus),
                    ("gini", r.gini),
                    ("depletion_round", r.depletion_round),
                    ("players_active", r.players_active),
                )
            },
        }
        for name, r in rows.items()
    }
    out: dict[str, Any] = {
        "variant": variant,
        "validated_evaluation_population": validated,
        "descriptive": descriptive,
        "conditional_on": f"planner seed {c4_config.B4_PLANNER_SEED}",
    }
    comparator = k_star(c4_config.B4_COMPARATOR_MODEL)
    if not validated:
        out["label"] = NOT_VALIDATED_B4.format(variant=variant)
        if comparator in rows:
            out["differences_vs_k_star"] = {
                name: float(r.total_surplus.mean() - rows[comparator].total_surplus.mean())
                for name, r in rows.items()
                if name != comparator
                and r.total_surplus.size
                and rows[comparator].total_surplus.size
            }
        return out

    needed = [comparator, *(planner(m) for m in TRAINABLE_MODELS)]
    missing = [name for name in needed if name not in rows]
    if missing:
        raise ValueError(f"B4-{variant}: confirmatory rows missing: {missing}")

    def cell(x: str, comp: str) -> Cell:
        return Cell(
            name=f"{x}@B4-{variant} vs {comp}",
            row=x,
            column=f"B4-{variant}",
            comparator=comp,
            a=rows[x].total_surplus[None, :],
            b=rows[comp].total_surplus[None, :],
            mode="deterministic",
        )

    exploratory = {
        f"{x}@B4-{variant} vs {comp}": "a row has fewer than 20 complete games"
        for x in (planner(m) for m in H2_MODELS)
        for comp in (comparator, planner("B1"))
        if x in rows
        and comp in rows
        and min(rows[x].total_surplus.size, rows[comp].total_surplus.size)
        < c4_config.B4_MIN_COMPLETE_GAMES
    }
    loss_cells = [cell(planner(m), comparator) for m in H2_MODELS]
    b_cells = [cell(planner(m), planner("B1")) for m in H2_MODELS]
    kw: dict[str, Any] = {"seed_level": False, "paired": False, "exploratory": exploratory}
    loss = run_family("B4-loss", loss_cells, "less", -DELTA, cache, **kw)
    noninferior = run_family("B4-noninferior", loss_cells, "greater", -DELTA, cache, **kw)
    b = run_family("B4-b", b_cells, "less", -DELTA, cache, **kw)
    out["families"] = {"B4-loss": loss, "B4-noninferior": noninferior, "B4-b": b}
    out["H2-B4"] = {**h2_verdict(loss, noninferior), "confirmatory": True}
    out["power_note"] = (
        "40 games per row: only large effects can be confirmed; an unconfirmed cell is not "
        "evidence of no effect"
    )
    return out


def b4_variant_validated(b4_eval: Sequence[Mapping[str, Any]], variant: str) -> bool:
    """§9.4: validated (closed loop) and gating on R-I or R-M2 in C2's b4-evaluate."""
    return any(
        r["variant"] == variant
        and r["regime"] in c4_config.B4_GATING_REGIMES
        and bool(r["counts_as_validated"])
        for r in b4_eval
    )


# --- everything --------------------------------------------------------------------------------------


def analyse(
    series: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    distill_doc: Mapping[str, Any] | None,
    plateaus: Mapping[str, Mapping[str, Any]],
    *,
    n_resamples: int = c4_config.BOOTSTRAP_RESAMPLES,
    seed: int = c4_config.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The §9 analysis of the main and secondary rows (both modes; deterministic
    confirmatory, sampled secondary), mechanism agreement and S1's attribution."""
    matrix = Matrix(series)
    cache = BootstrapCache(n_resamples, seed)
    modes: dict[str, Any] = {}
    for mode in crossplay.MODES:
        block: dict[str, Any] = {"main": main_families(matrix, summary, mode, cache)}
        if matrix.has(planner("B1", False)):
            block["S1"] = secondary_s1(matrix, mode, cache)
        if matrix.has(planner(c4_config.MIXTURE_MODEL)):
            block["S2"] = secondary_s2(matrix, mode, cache)
        modes[mode] = block
    det, smp = modes["deterministic"]["main"], modes["sampled"]["main"]
    out: dict[str, Any] = {
        "delta": DELTA,
        "alpha": c4_config.ALPHA,
        "bootstrap": {"resamples": n_resamples, "seed": seed},
        "confirmatory_mode": "deterministic",
        "modes": modes,
        "verdicts": {
            "H1": det["H1"]["verdict"],
            "H2": det["H2"]["verdict"],
            "H2b_confirmed_losses": det["H2b"]["confirmed_losses"],
        },
        "sampled_policy_disagreement": {
            name: {"deterministic": det[name]["verdict"], "sampled": smp[name]["verdict"]}
            for name in ("H1", "H2")
            if det[name]["verdict"] != smp[name]["verdict"]
        },
        "convergence": convergence_labels(plateaus),
    }
    if distill_doc is not None:
        main = mechanism_agreement(distill_doc["planners"], True)
        noret = mechanism_agreement(distill_doc["planners"], False)
        out["mechanism_agreement"] = {
            "main": main,
            "no_retention": noret,
            "distilled_crossplay": distilled_crossplay_matrix(distill_doc["distilled_crossplay"]),
        }
        if "S1" in modes["deterministic"]:
            s1 = modes["deterministic"]["S1"]
            out["S1_attribution"] = retention_attribution(
                len(det["families"]["H1-reversal"]["confirmed"]),
                len(s1["families"]["S1-reversal"]["confirmed"]),
                main["mean_between_model_l1"],
                noret["mean_between_model_l1"],
            )
    return out
