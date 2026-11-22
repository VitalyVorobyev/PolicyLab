"""Stage L2 frozen confirmatory comparison runner for The Last Library (D39)."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import stats

from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    run_status,
    validate_run_id,
    write_run_directory,
)
from phasespace.library.belief import expected_terrain_costs
from phasespace.library.config import DEFAULT_CONFIG, LibraryConfig, config_sha256, config_to_dict
from phasespace.library.feasibility import (
    F1_MIN_GAP,
    F2_MIN_POST_GEN2_DROP,
    F3_MAX_PLACEBO_ADVANTAGE,
    ArmSummary,
    arm_summary_from_arrays,
)
from phasespace.library.simulate import (
    LineageFixture,
    make_lineage_fixture,
    run_lineage,
    run_lineage_with_posteriors,
)
from phasespace.library.world import (
    FloatArray,
    expected_level_costs,
    observation_matrix,
    shortest_path,
)

SCHEMA_VERSION: Final = 1
STUDY_ID: Final = "last-library"
EXPERIMENT_ID: Final = "library-confirmatory-v1"
DEFAULT_RUN_ID: Final = "library-main-v1"
DEFAULT_RUNS_DIR: Final = Path("runs")
DEFAULT_PROTOCOL_PATH: Final = Path("docs/studies/last-library/protocol.md")
GENERATOR: Final = "phasespace.library.experiment"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"

STATIONARY_SEED_START: Final = 300_000
RULE_CHANGE_SEED_START: Final = 400_000
DEFAULT_N_LINEAGES: Final = 200
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 9201
MARGIN_DELTA: Final = 0.50
ALPHA: Final = 0.05

FIXED_ARMS: Final[tuple[str, ...]] = ("oracle", "none", "unlimited", "single_lifetime")
BUDGETED_ARMS: Final[tuple[str, ...]] = (
    "example_random",
    "example_pedagogical",
    "rule",
    "rule_confidence",
    "rule_hedged",
    "placebo_random",
    "placebo_shuffled",
)
SAMPLE_ARMS: Final[tuple[str, ...]] = (
    "oracle",
    "none",
    "unlimited",
    "example_random",
    "example_pedagogical",
    "rule",
    "rule_confidence",
    "rule_hedged",
)
SAMPLE_LINEAGE_INDICES: Final[tuple[int, ...]] = (0, 1, 2)
SAMPLE_BUDGETS: Final[tuple[int, ...]] = (10, 20, 40)

type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

_STATUS_RE: Final = re.compile(
    r"^\s*-\s+\*\*Status:\*\*\s+\*\*(?P<status>[a-zA-Z0-9_-]+)\*\*", re.M
)
_HASH_RE: Final = re.compile(
    r"^\s*-\s+\*\*Config hash \(`config_sha256`\):\*\*\s+`(?P<hash>[0-9a-f]{64})`",
    re.M,
)


@dataclass(frozen=True, slots=True)
class _LineageTask:
    """Picklable work unit evaluating all `(arm, budget)` specs for one lineage."""

    scenario_idx: int
    lineage_idx: int
    seed: int
    rule_changed: bool
    specs: tuple[tuple[str, int], ...]
    cfg: LibraryConfig


def spec_key(arm: str, budget_bits: int) -> str:
    """Canonical JSON key identifying `(arm, budget_bits)`."""
    return f"{arm}_b{budget_bits}"


def all_specs(cfg: LibraryConfig = DEFAULT_CONFIG) -> tuple[tuple[str, int], ...]:
    """Return ordered `(arm, budget_bits)` specifications for a confirmatory sweep."""
    fixed = tuple((arm, 0) for arm in FIXED_ARMS)
    swept = tuple((arm, b) for arm in BUDGETED_ARMS for b in cfg.bit_budgets)
    return fixed + swept


def parse_protocol_header(protocol_path: Path = DEFAULT_PROTOCOL_PATH) -> tuple[str, str]:
    """Parse `(status, config_sha256)` from `docs/studies/last-library/protocol.md`."""
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    text = protocol_path.read_text("utf-8")
    status_match = _STATUS_RE.search(text)
    hash_match = _HASH_RE.search(text)
    if status_match is None:
        raise ValueError(f"Protocol status header missing in {protocol_path}")
    if hash_match is None:
        raise ValueError(f"Protocol config_sha256 header missing in {protocol_path}")
    return status_match.group("status").lower(), hash_match.group("hash").lower()


def require_frozen_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    *,
    confirm_frozen: bool = False,
    allow_dirty: bool = False,
) -> tuple[str, str, bool]:
    """Enforce `--confirm-frozen`, frozen protocol header, matching config hash, and clean git."""
    if not confirm_frozen:
        raise RuntimeError(
            "Confirmatory run refused without --confirm-frozen (freeze check per protocol v1)."
        )
    status, proto_hash = parse_protocol_header(protocol_path)
    if status != "frozen":
        raise RuntimeError(
            f"Refusing confirmatory run: {protocol_path} status is {status!r}, expected 'frozen'."
        )
    code_hash = config_sha256(cfg)
    if proto_hash != code_hash:
        raise RuntimeError(
            f"Config hash mismatch between {protocol_path} ({proto_hash}) and code ({code_hash})."
        )
    git_commit, git_dirty = git_source()
    if git_dirty and not allow_dirty:
        raise RuntimeError(
            "Confirmatory run refused on a dirty git working tree; commit Stage L2 changes first."
        )
    return proto_hash, git_commit, git_dirty


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-Bonferroni step-down adjusted p-values preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid p-value {p!r}; expected finite float in [0, 1]")

    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * p_values[idx]
        running_max = min(1.0, max(running_max, scaled))
        adjusted[idx] = running_max
    return adjusted


def paired_contrast(
    differences: Sequence[float] | FloatArray,
    *,
    contrast_id: str,
    family: str,
    label: str,
    higher_expected_arm: str,
    lower_expected_arm: str,
    delta: float = MARGIN_DELTA,
    boot_indices: IntArray | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute one-sided paired Student t-test (`mean > delta`) and percentile bootstrap 95% CI."""
    arr = np.asarray(differences, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"Contrast {contrast_id}: expected 1D array of >= 2 differences")
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    if sd == 0.0:
        if mean > delta:
            t_stat = float("inf")
            p_raw = 0.0
        elif mean < delta:
            t_stat = float("-inf")
            p_raw = 1.0
        else:
            t_stat = 0.0
            p_raw = 0.5
        cohens_dz = 0.0
    else:
        t_stat = (mean - delta) / se
        p_raw = float(stats.t.sf(t_stat, df=df))
        cohens_dz = mean / sd

    if boot_indices is None:
        rng = np.random.default_rng(bootstrap_seed)
        idx: IntArray = rng.integers(0, n, size=(n_resamples, n), dtype=np.int64)
    else:
        idx = boot_indices
    boot_means = arr[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "contrast_id": contrast_id,
        "family": family,
        "label": label,
        "higher_expected_arm": higher_expected_arm,
        "lower_expected_arm": lower_expected_arm,
        "n_lineages": n,
        "delta": delta,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t_stat": float(t_stat),
        "df": df,
        "p_raw": p_raw,
        "cohens_dz": cohens_dz,
    }


def evaluate_family(
    family_id: str,
    title: str,
    contrasts_raw: Sequence[dict[str, Any]],
    *,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Apply Holm-Bonferroni correction across a hypothesis family and score support."""
    adjusted = holm_adjust([float(c["p_raw"]) for c in contrasts_raw])
    contrasts: list[dict[str, Any]] = []
    for c, p_holm in zip(contrasts_raw, adjusted, strict=True):
        passed = math.isfinite(p_holm) and p_holm < alpha
        contrasts.append({**c, "p_holm": p_holm, "alpha": alpha, "passed": passed})
    n_passed = sum(1 for c in contrasts if c["passed"])
    return {
        "family_id": family_id,
        "title": title,
        "n_contrasts": len(contrasts),
        "alpha": alpha,
        "n_passed": n_passed,
        "all_passed": n_passed == len(contrasts),
        "contrasts": contrasts,
    }


def evaluate_hypotheses(
    stationary_heldout: dict[tuple[str, int], FloatArray],
    change_heldout: dict[tuple[str, int], FloatArray],
    *,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    delta: float = MARGIN_DELTA,
    alpha: float = ALPHA,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Evaluate all pre-registered H1, H2a, H2b, H3a, H3b, and H4 families from paired tensors."""
    sample_arr = next(iter(stationary_heldout.values()))
    n_lin = sample_arr.shape[0]
    rng = np.random.default_rng(bootstrap_seed)
    boot_indices: IntArray = rng.integers(0, n_lin, size=(n_resamples, n_lin), dtype=np.int64)

    g_final = cfg.n_generations - 1
    g_change = cfg.change_generation

    h1_raw = [
        paired_contrast(
            stationary_heldout[("example_pedagogical", b)][:, g_final]
            - stationary_heldout[("rule", b)][:, g_final],
            contrast_id=f"H1_b{b}",
            family="H1",
            label=f"Stationary gen {g_final}: example_pedagogical@{b} - rule@{b}",
            higher_expected_arm=f"example_pedagogical_b{b}",
            lower_expected_arm=f"rule_b{b}",
            delta=delta,
            boot_indices=boot_indices,
        )
        for b in (10, 15, 20, 40)
    ]

    h2a_raw = [
        paired_contrast(
            change_heldout[("rule", b)][:, g_change]
            - change_heldout[("example_pedagogical", b)][:, g_change],
            contrast_id=f"H2a_b{b}",
            family="H2a",
            label=f"Rule-change gen {g_change}: rule@{b} - example_pedagogical@{b}",
            higher_expected_arm=f"rule_b{b}",
            lower_expected_arm=f"example_pedagogical_b{b}",
            delta=delta,
            boot_indices=boot_indices,
        )
        for b in (20, 40)
    ]

    h2b_raw = [
        paired_contrast(
            change_heldout[("rule", b)][:, g_change] - change_heldout[("none", 0)][:, g_change],
            contrast_id=f"H2b_b{b}",
            family="H2b",
            label=f"Rule-change gen {g_change}: rule@{b} - none",
            higher_expected_arm=f"rule_b{b}",
            lower_expected_arm="none_b0",
            delta=delta,
            boot_indices=boot_indices,
        )
        for b in (20, 40)
    ]

    h3a_raw = [
        paired_contrast(
            change_heldout[("rule", 40)][:, g_change]
            - change_heldout[("rule_confidence", 40)][:, g_change],
            contrast_id="H3a_b40",
            family="H3a",
            label=f"Rule-change gen {g_change}: rule@40 - rule_confidence@40",
            higher_expected_arm="rule_b40",
            lower_expected_arm="rule_confidence_b40",
            delta=delta,
            boot_indices=boot_indices,
        )
    ]

    h3b_raw = [
        paired_contrast(
            change_heldout[("rule", b)][:, g_change]
            - change_heldout[("rule_hedged", b)][:, g_change],
            contrast_id=f"H3b_change_b{b}",
            family="H3b",
            label=f"Rule-change gen {g_change}: rule@{b} - rule_hedged@{b}",
            higher_expected_arm=f"rule_b{b}",
            lower_expected_arm=f"rule_hedged_b{b}",
            delta=delta,
            boot_indices=boot_indices,
        )
        for b in (20, 40)
    ] + [
        paired_contrast(
            stationary_heldout[("example_pedagogical", b)][:, g_final]
            - stationary_heldout[("rule_hedged", b)][:, g_final],
            contrast_id=f"H3b_stationary_b{b}",
            family="H3b",
            label=f"Stationary gen {g_final}: example_pedagogical@{b} - rule_hedged@{b}",
            higher_expected_arm=f"example_pedagogical_b{b}",
            lower_expected_arm=f"rule_hedged_b{b}",
            delta=delta,
            boot_indices=boot_indices,
        )
        for b in (20, 40)
    ]

    h4_raw = [
        paired_contrast(
            stationary_heldout[("unlimited", 0)][:, 1]
            - stationary_heldout[("unlimited", 0)][:, g_final],
            contrast_id="H4_unlimited",
            family="H4",
            label=f"Stationary unlimited cumulative drop (gen 1 - gen {g_final})",
            higher_expected_arm="unlimited_gen1",
            lower_expected_arm=f"unlimited_gen{g_final}",
            delta=delta,
            boot_indices=boot_indices,
        ),
        paired_contrast(
            stationary_heldout[("rule", 20)][:, 1] - stationary_heldout[("rule", 20)][:, g_final],
            contrast_id="H4_rule_b20",
            family="H4",
            label=f"Stationary rule@20 cumulative drop (gen 1 - gen {g_final})",
            higher_expected_arm="rule_b20_gen1",
            lower_expected_arm=f"rule_b20_gen{g_final}",
            delta=delta,
            boot_indices=boot_indices,
        ),
        paired_contrast(
            stationary_heldout[("example_pedagogical", 20)][:, 1]
            - stationary_heldout[("example_pedagogical", 20)][:, g_final],
            contrast_id="H4_example_pedagogical_b20",
            family="H4",
            label=f"Stationary example_pedagogical@20 cumulative drop (gen 1 - gen {g_final})",
            higher_expected_arm="example_pedagogical_b20_gen1",
            lower_expected_arm=f"example_pedagogical_b20_gen{g_final}",
            delta=delta,
            boot_indices=boot_indices,
        ),
    ]

    return {
        "H1": evaluate_family(
            "H1",
            "Stationary efficiency: rule beats pedagogical examples at gen 9",
            h1_raw,
            alpha=alpha,
        ),
        "H2a": evaluate_family(
            "H2a",
            "Rule-change overshoot: rule suffers higher regret than examples at gen 5",
            h2a_raw,
            alpha=alpha,
        ),
        "H2b": evaluate_family(
            "H2b",
            "Rule-change overshoot: rule suffers higher regret than none at gen 5",
            h2b_raw,
            alpha=alpha,
        ),
        "H3a": evaluate_family(
            "H3a",
            "Sender confidence calibration at B=40 vs rigid rule at change gen 5",
            h3a_raw,
            alpha=alpha,
        ),
        "H3b": evaluate_family(
            "H3b",
            "Receiver hazard hedging mitigates gen 5 overshoot and keeps gen 9 edge",
            h3b_raw,
            alpha=alpha,
        ),
        "H4": evaluate_family(
            "H4",
            "Q3 secondary: cumulative multi-generation improvement (gen 1 -> gen 9)",
            h4_raw,
            alpha=alpha,
        ),
    }


def _run_lineage_worker(
    task: _LineageTask,
) -> tuple[int, int, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    cfg = task.cfg
    obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    exp_costs = expected_level_costs(cfg.level_costs, obs_matrix)
    fixture = make_lineage_fixture(task.seed, rule_changed=task.rule_changed, cfg=cfg)

    n_specs = len(task.specs)
    n_gen = cfg.n_generations
    heldout = np.zeros((n_specs, n_gen), dtype=np.float64)
    train = np.zeros((n_specs, n_gen), dtype=np.float64)
    entropy = np.zeros((n_specs, n_gen), dtype=np.float64)
    accuracy = np.zeros((n_specs, n_gen), dtype=np.float64)
    bits = np.zeros((n_specs, n_gen), dtype=np.float64)

    for s_idx, (arm, budget_bits) in enumerate(task.specs):
        res = run_lineage(
            arm=arm,
            budget_bits=budget_bits,
            fixture=fixture,
            rule_changed=task.rule_changed,
            cfg=cfg,
            obs_matrix=obs_matrix,
            exp_level_costs_arr=exp_costs,
        )
        for gm in res.generations:
            g = gm.generation
            heldout[s_idx, g] = gm.heldout_regret
            train[s_idx, g] = gm.train_regret
            entropy[s_idx, g] = gm.posterior_entropy_bits
            accuracy[s_idx, g] = gm.map_accuracy
            bits[s_idx, g] = float(gm.bits_used)

    return task.scenario_idx, task.lineage_idx, heldout, train, entropy, accuracy, bits


def _build_budget_curves(
    stationary_summaries: dict[str, ArmSummary],
    change_summaries: dict[str, ArmSummary],
    cfg: LibraryConfig,
) -> dict[str, Any]:
    g_final = cfg.n_generations - 1
    g_change = cfg.change_generation
    arm_curves: dict[str, list[dict[str, float | int]]] = {}
    for arm in BUDGETED_ARMS:
        points: list[dict[str, float | int]] = []
        for b in cfg.bit_budgets:
            key = spec_key(arm, b)
            st = stationary_summaries[key]
            rc = change_summaries[key]
            points.append(
                {
                    "budget_bits": b,
                    "stationary_final_mean": st.heldout_regret_mean_by_gen[g_final],
                    "stationary_final_se": st.heldout_regret_se_by_gen[g_final],
                    "stationary_bits_mean": st.bits_used_mean_by_gen[g_final],
                    "change_at_split_mean": rc.heldout_regret_mean_by_gen[g_change],
                    "change_at_split_se": rc.heldout_regret_se_by_gen[g_change],
                    "change_final_mean": rc.heldout_regret_mean_by_gen[g_final],
                    "change_final_se": rc.heldout_regret_se_by_gen[g_final],
                }
            )
        arm_curves[arm] = points

    baselines: dict[str, dict[str, float]] = {}
    for arm in FIXED_ARMS:
        key = spec_key(arm, 0)
        st = stationary_summaries[key]
        rc = change_summaries[key]
        baselines[arm] = {
            "stationary_final_mean": st.heldout_regret_mean_by_gen[g_final],
            "stationary_final_se": st.heldout_regret_se_by_gen[g_final],
            "change_at_split_mean": rc.heldout_regret_mean_by_gen[g_change],
            "change_at_split_se": rc.heldout_regret_se_by_gen[g_change],
            "change_final_mean": rc.heldout_regret_mean_by_gen[g_final],
            "change_final_se": rc.heldout_regret_se_by_gen[g_final],
        }
    return {"budgets": list(cfg.bit_budgets), "arms": arm_curves, "baselines": baselines}


def build_validation_checks(
    stationary_heldout: dict[tuple[str, int], FloatArray],
    change_heldout: dict[tuple[str, int], FloatArray],
    stationary_bits: dict[tuple[str, int], FloatArray],
    change_bits: dict[tuple[str, int], FloatArray],
    *,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    protocol_hash_matches: bool = True,
) -> tuple[Check, ...]:
    """Compute automated schema-v1 sanity checks (`observed <= threshold` indicates pass)."""
    oracle_max = max(
        float(stationary_heldout[("oracle", 0)].max()),
        float(change_heldout[("oracle", 0)].max()),
    )
    single_unlim_diff = max(
        float(
            np.abs(
                stationary_heldout[("single_lifetime", 0)] - stationary_heldout[("unlimited", 0)]
            ).max()
        ),
        float(
            np.abs(change_heldout[("single_lifetime", 0)] - change_heldout[("unlimited", 0)]).max()
        ),
    )
    none_gen0 = float(stationary_heldout[("none", 0)][:, 0].mean())
    f1_deficit = max(0.0, F1_MIN_GAP - none_gen0)

    unlim_curve = stationary_heldout[("unlimited", 0)].mean(axis=0)
    unlim_drop = float(unlim_curve[1] - unlim_curve[-1])
    f2_deficit = max(0.0, F2_MIN_POST_GEN2_DROP - unlim_drop)

    none_post_gen0 = float(stationary_heldout[("none", 0)][:, 1:].mean())
    p_rand_20 = float(stationary_heldout[("placebo_random", 20)][:, 1:].mean())
    p_shuf_20 = float(stationary_heldout[("placebo_shuffled", 20)][:, 1:].mean())
    f3_advantage = max(0.0, none_post_gen0 - min(p_rand_20, p_shuf_20))

    max_budget_violation = 0.0
    for arm, b in all_specs(cfg):
        st_v = float(np.maximum(0.0, stationary_bits[(arm, b)] - float(b)).max())
        rc_v = float(np.maximum(0.0, change_bits[(arm, b)] - float(b)).max())
        max_budget_violation = max(max_budget_violation, st_v, rc_v)

    return (
        Check(
            name="oracle_zero_regret_max",
            observed=oracle_max,
            threshold=1e-12,
            definition="max held-out regret across oracle lineages and generations",
        ),
        Check(
            name="single_lifetime_unlimited_abs_diff_max",
            observed=single_unlim_diff,
            threshold=1e-12,
            definition="max abs regret difference between single_lifetime and unlimited",
        ),
        Check(
            name="none_gen0_regret_deficit",
            observed=f1_deficit,
            threshold=0.0,
            definition=f"max(0, {F1_MIN_GAP} - none_gen0_stationary_heldout_regret)",
        ),
        Check(
            name="unlimited_post_gen2_drop_deficit",
            observed=f2_deficit,
            threshold=0.0,
            definition=f"max(0, {F2_MIN_POST_GEN2_DROP} - (unlimited_gen1 - unlimited_gen_final))",
        ),
        Check(
            name="placebo_advantage_over_none_max",
            observed=f3_advantage,
            threshold=F3_MAX_PLACEBO_ADVANTAGE,
            definition="max(0, none_post_gen0 - min(placebo_random@20, placebo_shuffled@20))",
        ),
        Check(
            name="bit_budget_violation_max",
            observed=max_budget_violation,
            threshold=0.0,
            definition="max bits_used overshoot above configured budget_bits across all runs",
        ),
        Check(
            name="protocol_config_sha256_mismatch",
            observed=0.0 if protocol_hash_matches else 1.0,
            threshold=0.0,
            definition="0 if frozen protocol.md config_sha256 matches code config_sha256()",
        ),
    )


def run_experiment(
    *,
    n_lineages: int = DEFAULT_N_LINEAGES,
    stationary_seed_start: int = STATIONARY_SEED_START,
    rule_change_seed_start: int = RULE_CHANGE_SEED_START,
    workers: int = 1,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    protocol_hash_matches: bool = True,
) -> tuple[dict[str, Any], tuple[Check, ...]]:
    """Run all specs across stationary and rule-change lineages and evaluate hypotheses."""
    if n_lineages < 2:
        raise ValueError(f"n_lineages must be >= 2, got {n_lineages}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    specs = all_specs(cfg)
    n_specs = len(specs)
    n_gen = cfg.n_generations

    tasks: list[_LineageTask] = []
    for scenario_idx, (seed_start, rule_changed) in enumerate(
        ((stationary_seed_start, False), (rule_change_seed_start, True))
    ):
        for lin_idx in range(n_lineages):
            tasks.append(
                _LineageTask(
                    scenario_idx=scenario_idx,
                    lineage_idx=lin_idx,
                    seed=seed_start + lin_idx,
                    rule_changed=rule_changed,
                    specs=specs,
                    cfg=cfg,
                )
            )

    heldout = np.zeros((2, n_specs, n_lineages, n_gen), dtype=np.float64)
    train = np.zeros((2, n_specs, n_lineages, n_gen), dtype=np.float64)
    entropy = np.zeros((2, n_specs, n_lineages, n_gen), dtype=np.float64)
    accuracy = np.zeros((2, n_specs, n_lineages, n_gen), dtype=np.float64)
    bits = np.zeros((2, n_specs, n_lineages, n_gen), dtype=np.float64)

    if workers == 1:
        worker_results = [_run_lineage_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            worker_results = list(pool.map(_run_lineage_worker, tasks))

    for sc_idx, lin_idx, h_arr, t_arr, e_arr, a_arr, b_arr in worker_results:
        heldout[sc_idx, :, lin_idx, :] = h_arr
        train[sc_idx, :, lin_idx, :] = t_arr
        entropy[sc_idx, :, lin_idx, :] = e_arr
        accuracy[sc_idx, :, lin_idx, :] = a_arr
        bits[sc_idx, :, lin_idx, :] = b_arr

    stationary_heldout_map: dict[tuple[str, int], FloatArray] = {}
    change_heldout_map: dict[tuple[str, int], FloatArray] = {}
    stationary_bits_map: dict[tuple[str, int], FloatArray] = {}
    change_bits_map: dict[tuple[str, int], FloatArray] = {}
    stationary_summaries: dict[str, ArmSummary] = {}
    change_summaries: dict[str, ArmSummary] = {}

    for s_idx, (arm, budget_bits) in enumerate(specs):
        key = spec_key(arm, budget_bits)
        stationary_heldout_map[(arm, budget_bits)] = heldout[0, s_idx]
        change_heldout_map[(arm, budget_bits)] = heldout[1, s_idx]
        stationary_bits_map[(arm, budget_bits)] = bits[0, s_idx]
        change_bits_map[(arm, budget_bits)] = bits[1, s_idx]
        stationary_summaries[key] = arm_summary_from_arrays(
            arm=arm,
            budget_bits=budget_bits,
            rule_changed=False,
            heldout=heldout[0, s_idx],
            train=train[0, s_idx],
            entropy=entropy[0, s_idx],
            accuracy=accuracy[0, s_idx],
            bits=bits[0, s_idx],
        )
        change_summaries[key] = arm_summary_from_arrays(
            arm=arm,
            budget_bits=budget_bits,
            rule_changed=True,
            heldout=heldout[1, s_idx],
            train=train[1, s_idx],
            entropy=entropy[1, s_idx],
            accuracy=accuracy[1, s_idx],
            bits=bits[1, s_idx],
        )

    hypotheses = evaluate_hypotheses(stationary_heldout_map, change_heldout_map, cfg=cfg)
    checks = build_validation_checks(
        stationary_heldout_map,
        change_heldout_map,
        stationary_bits_map,
        change_bits_map,
        cfg=cfg,
        protocol_hash_matches=protocol_hash_matches,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "n_lineages": n_lineages,
        "seeds": {
            "stationary_start": stationary_seed_start,
            "stationary_end_inclusive": stationary_seed_start + n_lineages - 1,
            "rule_change_start": rule_change_seed_start,
            "rule_change_end_inclusive": rule_change_seed_start + n_lineages - 1,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "config": config_to_dict(cfg),
        "config_sha256": config_sha256(cfg),
        "hypotheses": hypotheses,
        "budget_curves": _build_budget_curves(stationary_summaries, change_summaries, cfg),
        "stationary": {k: v.to_json() for k, v in stationary_summaries.items()},
        "rule_change": {k: v.to_json() for k, v in change_summaries.items()},
    }
    return payload, checks


def _extract_scenario_samples(
    fixtures: Sequence[tuple[int, LineageFixture]],
    *,
    rule_changed: bool,
    cfg: LibraryConfig,
    obs_matrix: FloatArray,
    exp_costs: FloatArray,
) -> list[dict[str, Any]]:
    key_gens = sorted({0, 1, cfg.change_generation, cfg.n_generations - 1})
    sample_budgets = tuple(b for b in SAMPLE_BUDGETS if b in cfg.bit_budgets)
    sample_specs: list[tuple[str, int]] = [
        (arm, 0) for arm in ("oracle", "none", "unlimited") if arm in SAMPLE_ARMS
    ]
    sample_specs.extend(
        (arm, b)
        for arm in SAMPLE_ARMS
        if arm not in ("oracle", "none", "unlimited")
        for b in sample_budgets
    )

    entries: list[dict[str, Any]] = []
    for lineage_idx, fixture in fixtures:
        layouts_by_gen: list[dict[str, Any]] = []
        for g in key_gens:
            h0 = fixture.heldout_layouts[g][0]
            layouts_by_gen.append(
                {
                    "generation": g,
                    "heldout_preview": {
                        "start": [0, 0],
                        "goal": [h0.size - 1, h0.size - 1],
                        "terrains": [[int(cell) for cell in row] for row in h0.terrains],
                    },
                }
            )

        arm_traces: dict[str, dict[str, Any]] = {}
        for arm, budget_bits in sample_specs:
            res, posteriors = run_lineage_with_posteriors(
                arm=arm,
                budget_bits=budget_bits,
                fixture=fixture,
                rule_changed=rule_changed,
                cfg=cfg,
                obs_matrix=obs_matrix,
                exp_level_costs_arr=exp_costs,
            )
            planned_paths: dict[str, list[list[int]]] = {}
            for g in key_gens:
                est_costs = expected_terrain_costs(posteriors[g], exp_costs)
                path, _ = shortest_path(fixture.heldout_layouts[g][0], est_costs)
                planned_paths[str(g)] = [[int(r), int(c)] for r, c in path]

            arm_traces[spec_key(arm, budget_bits)] = {
                "arm": arm,
                "budget_bits": budget_bits,
                "planned_paths_by_key_gen": planned_paths,
                "generations": [
                    {
                        "generation": gm.generation,
                        "heldout_regret": gm.heldout_regret,
                        "train_regret": gm.train_regret,
                        "bits_used": gm.bits_used,
                        "posterior_entropy_bits": gm.posterior_entropy_bits,
                        "map_accuracy": gm.map_accuracy,
                        "record_summary": list(gm.record_summary),
                    }
                    for gm in res.generations
                ],
            }

        entries.append(
            {
                "lineage_index": lineage_idx,
                "seed": fixture.seed,
                "rule_changed": rule_changed,
                "rule_initial": [int(x) for x in fixture.rule_initial.levels],
                "rule_after": [int(x) for x in fixture.rule_after.levels],
                "layouts_by_key_gen": layouts_by_gen,
                "arms": arm_traces,
            }
        )
    return entries


def build_samples_payload(
    *,
    run_id: str,
    n_lineages: int = DEFAULT_N_LINEAGES,
    stationary_seed_start: int = STATIONARY_SEED_START,
    rule_change_seed_start: int = RULE_CHANGE_SEED_START,
    cfg: LibraryConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Export fixed sample lineages `(0, 1, 2)` with grids, paths, and records for UI inspection."""
    indices = [idx for idx in SAMPLE_LINEAGE_INDICES if idx < n_lineages]
    obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    exp_costs = expected_level_costs(cfg.level_costs, obs_matrix)

    st_fixtures = [
        (idx, make_lineage_fixture(stationary_seed_start + idx, rule_changed=False, cfg=cfg))
        for idx in indices
    ]
    rc_fixtures = [
        (idx, make_lineage_fixture(rule_change_seed_start + idx, rule_changed=True, cfg=cfg))
        for idx in indices
    ]
    key_gens = sorted({0, 1, cfg.change_generation, cfg.n_generations - 1})

    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "sample_indices": indices,
        "sample_budgets": [b for b in SAMPLE_BUDGETS if b in cfg.bit_budgets],
        "key_generations": key_gens,
        "config": {
            "grid_size": cfg.grid_size,
            "level_costs": list(cfg.level_costs),
            "noise_beta": cfg.noise_beta,
            "n_generations": cfg.n_generations,
            "change_generation": cfg.change_generation,
            "hedge_discount": cfg.hedge_discount,
        },
        "stationary": _extract_scenario_samples(
            st_fixtures, rule_changed=False, cfg=cfg, obs_matrix=obs_matrix, exp_costs=exp_costs
        ),
        "rule_change": _extract_scenario_samples(
            rc_fixtures, rule_changed=True, cfg=cfg, obs_matrix=obs_matrix, exp_costs=exp_costs
        ),
    }


def _contrast_by_id(hypotheses: dict[str, dict[str, Any]], family: str, contrast_id: str) -> float:
    for item in hypotheses[family]["contrasts"]:
        if item["contrast_id"] == contrast_id:
            return float(item["mean_diff"])
    raise KeyError(f"Missing contrast {contrast_id} in family {family}")


def export_confirmatory_run(
    runs_dir: Path,
    run_id: str,
    *,
    confirm_frozen: bool,
    allow_dirty_for_test: bool = False,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    n_lineages: int = DEFAULT_N_LINEAGES,
    stationary_seed_start: int = STATIONARY_SEED_START,
    rule_change_seed_start: int = RULE_CHANGE_SEED_START,
    workers: int = 8,
    cfg: LibraryConfig = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify frozen protocol, execute confirmatory study, and export immutable run artifacts."""
    validate_run_id(run_id)
    out_dir = runs_dir / run_id
    if out_dir.exists():
        raise FileExistsError(
            f"Run directory {out_dir} already exists; runs are immutable and never overwritten."
        )

    proto_hash, git_commit, git_dirty = require_frozen_protocol(
        protocol_path=protocol_path,
        cfg=cfg,
        confirm_frozen=confirm_frozen,
        allow_dirty=allow_dirty_for_test,
    )

    results_payload, checks = run_experiment(
        n_lineages=n_lineages,
        stationary_seed_start=stationary_seed_start,
        rule_change_seed_start=rule_change_seed_start,
        workers=workers,
        cfg=cfg,
        protocol_hash_matches=(proto_hash == config_sha256(cfg)),
    )
    results_with_id: dict[str, Any] = {"run_id": run_id, **results_payload}
    samples_payload = build_samples_payload(
        run_id=run_id,
        n_lineages=n_lineages,
        stationary_seed_start=stationary_seed_start,
        rule_change_seed_start=rule_change_seed_start,
        cfg=cfg,
    )

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    limitations = [
        "Finite 9x9 Voronoi terrain navigation gridworld under Bayesian path-cost inference.",
        "Confirmatory hedge_discount=0.30 is a fixed pre-registered constant rather than meta-learned online.",
    ]
    if git_dirty:
        limitations.append(
            "Generated with allow_dirty_for_test=True on an uncommitted tree (source.git_dirty = true)."
        )

    hyp = results_payload["hypotheses"]
    metrics_block: dict[str, Any] = {}
    metric_specs = (
        (
            "h1_rule_vs_ped_b20_diff",
            "H1",
            "H1_b20",
            "stationary gen-9 example_pedagogical@20 - rule@20",
        ),
        (
            "h2a_rule_vs_ped_b20_change_diff",
            "H2a",
            "H2a_b20",
            "change gen-5 rule@20 - example_pedagogical@20",
        ),
        ("h2b_rule_vs_none_b20_change_diff", "H2b", "H2b_b20", "change gen-5 rule@20 - none"),
        (
            "h3a_conf_vs_rule_b40_change_diff",
            "H3a",
            "H3a_b40",
            "change gen-5 rule@40 - rule_confidence@40",
        ),
        (
            "h3b_hedged_vs_rule_b20_change_diff",
            "H3b",
            "H3b_change_b20",
            "change gen-5 rule@20 - rule_hedged@20",
        ),
        (
            "h3b_hedged_vs_ped_b20_stat_diff",
            "H3b",
            "H3b_stationary_b20",
            "stationary gen-9 example_pedagogical@20 - rule_hedged@20",
        ),
        (
            "h4_rule_b20_gen1_to_9_drop",
            "H4",
            "H4_rule_b20",
            "stationary rule@20 gen-1 minus gen-9 regret drop",
        ),
    )
    for metric_name, fam, cid, desc in metric_specs:
        if fam in hyp and any(item["contrast_id"] == cid for item in hyp[fam]["contrasts"]):
            metrics_block[metric_name] = {
                "value": _contrast_by_id(hyp, fam, cid),
                "unit": "dimensionless",
                "definition": desc,
            }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": run_status(checks),
        "created_at": timestamp,
        "source": {
            "repository": REPOSITORY,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "protocol": {
            "version": "v1",
            "status": "frozen",
            "path": str(protocol_path),
            "config_sha256": proto_hash,
            "confirmed_frozen": confirm_frozen,
        },
        "environment": environment_info(),
        "model": {
            "name": "voronoi-terrain-navigation",
            "config_sha256": config_sha256(cfg),
            **config_to_dict(cfg),
        },
        "random_seed": stationary_seed_start,
        "artifacts": [],
        "metrics": metrics_block,
        "limitations": limitations,
        "validation": {
            "checks": [c.to_json() for c in checks],
            "validated_by": f"{GENERATOR} automated sanity checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
    }

    written = write_run_directory(
        out_dir,
        manifest,
        [
            ("confirmatory_results", "results.json", dumps_json(results_with_id)),
            ("sample_lineages", "samples.json", dumps_json(samples_payload)),
        ],
    )
    return written, results_with_id, samples_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m phasespace.library.experiment")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument("--allow-dirty-for-test", action="store_true")
    parser.add_argument("--lineages", type=int, default=DEFAULT_N_LINEAGES)
    parser.add_argument("--stationary-seed-start", type=int, default=STATIONARY_SEED_START)
    parser.add_argument("--rule-change-seed-start", type=int, default=RULE_CHANGE_SEED_START)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, results, _ = export_confirmatory_run(
        args.runs_dir,
        args.run_id,
        confirm_frozen=args.confirm_frozen,
        allow_dirty_for_test=args.allow_dirty_for_test,
        protocol_path=args.protocol_path,
        n_lineages=args.lineages,
        stationary_seed_start=args.stationary_seed_start,
        rule_change_seed_start=args.rule_change_seed_start,
        workers=args.workers,
    )
    print(f"Wrote {args.runs_dir / args.run_id} (status={manifest['status']})")
    for chk in manifest["validation"]["checks"]:
        tag = "PASS" if chk["passed"] else "FAIL"
        print(f"  [{tag}] {chk['name']}: observed={chk['observed']:.4g} <= {chk['threshold']}")
    for fam_id, fam in results["hypotheses"].items():
        print(f"  [{fam_id}] {fam['n_passed']}/{fam['n_contrasts']} passed — {fam['title']}")
        for c in fam["contrasts"]:
            verdict = "PASS" if c["passed"] else "FAIL"
            print(
                f"      {verdict} {c['contrast_id']}: diff={c['mean_diff']:+.3f} "
                f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}], "
                f"p_raw={c['p_raw']:.3e}, p_holm={c['p_holm']:.3e}"
            )
    return 0 if manifest["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
