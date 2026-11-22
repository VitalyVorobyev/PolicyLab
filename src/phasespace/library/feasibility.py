"""Pre-protocol Stage L1 feasibility gate for The Last Library (D38)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.gravity.export import (
    Check,
    dumps_json,
    environment_info,
    git_source,
    run_status,
    validate_run_id,
    write_run_directory,
)
from phasespace.library.config import DEFAULT_CONFIG, LibraryConfig, config_sha256, config_to_dict
from phasespace.library.simulate import LineageFixture, make_lineage_fixture, run_lineage
from phasespace.library.world import FloatArray, expected_level_costs, observation_matrix

SCHEMA_VERSION: Final = 1
STUDY_ID: Final = "last-library"
EXPERIMENT_ID: Final = "library-feasibility-v1"
DEFAULT_RUN_ID: Final = "library-feasibility-dev1"
DEFAULT_RUNS_DIR: Final = Path("runs")
GENERATOR: Final = "phasespace.library.feasibility"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"

F1_MIN_GAP: Final = 5.0
F2_MIN_POST_GEN2_DROP: Final = 1.0
F3_MAX_PLACEBO_ADVANTAGE: Final = 0.5


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Aggregated generational curves across lineages for one arm and bit budget."""

    arm: str
    budget_bits: int
    rule_changed: bool
    n_lineages: int
    heldout_regret_mean_by_gen: list[float]
    heldout_regret_se_by_gen: list[float]
    train_regret_mean_by_gen: list[float]
    entropy_bits_mean_by_gen: list[float]
    map_accuracy_mean_by_gen: list[float]
    bits_used_mean_by_gen: list[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "budget_bits": self.budget_bits,
            "rule_changed": self.rule_changed,
            "n_lineages": self.n_lineages,
            "heldout_regret_mean_by_gen": self.heldout_regret_mean_by_gen,
            "heldout_regret_se_by_gen": self.heldout_regret_se_by_gen,
            "train_regret_mean_by_gen": self.train_regret_mean_by_gen,
            "entropy_bits_mean_by_gen": self.entropy_bits_mean_by_gen,
            "map_accuracy_mean_by_gen": self.map_accuracy_mean_by_gen,
            "bits_used_mean_by_gen": self.bits_used_mean_by_gen,
        }


def arm_summary_from_arrays(
    arm: str,
    budget_bits: int,
    rule_changed: bool,
    heldout: FloatArray,
    train: FloatArray,
    entropy: FloatArray,
    accuracy: FloatArray,
    bits: FloatArray,
) -> ArmSummary:
    """Aggregate `(n_lineages, n_generations)` lineage metric matrices into `ArmSummary`."""
    n_lin, n_gen = heldout.shape
    se_list: list[float] = (
        [float(x) for x in (heldout.std(axis=0, ddof=1) / np.sqrt(n_lin))]
        if n_lin > 1
        else [0.0] * n_gen
    )
    return ArmSummary(
        arm=arm,
        budget_bits=budget_bits,
        rule_changed=rule_changed,
        n_lineages=n_lin,
        heldout_regret_mean_by_gen=[float(x) for x in heldout.mean(axis=0)],
        heldout_regret_se_by_gen=se_list,
        train_regret_mean_by_gen=[float(x) for x in train.mean(axis=0)],
        entropy_bits_mean_by_gen=[float(x) for x in entropy.mean(axis=0)],
        map_accuracy_mean_by_gen=[float(x) for x in accuracy.mean(axis=0)],
        bits_used_mean_by_gen=[float(x) for x in bits.mean(axis=0)],
    )


def _summarize_arm(
    arm: str,
    budget_bits: int,
    fixtures: Sequence[LineageFixture],
    rule_changed: bool,
    cfg: LibraryConfig,
) -> ArmSummary:
    obs_matrix = observation_matrix(cfg.n_levels, cfg.noise_beta)
    exp_costs = expected_level_costs(cfg.level_costs, obs_matrix)
    n_lin = len(fixtures)
    n_gen = cfg.n_generations

    heldout = np.zeros((n_lin, n_gen), dtype=np.float64)
    train = np.zeros((n_lin, n_gen), dtype=np.float64)
    entropy = np.zeros((n_lin, n_gen), dtype=np.float64)
    accuracy = np.zeros((n_lin, n_gen), dtype=np.float64)
    bits = np.zeros((n_lin, n_gen), dtype=np.float64)

    for i, fix in enumerate(fixtures):
        res = run_lineage(
            arm=arm,
            budget_bits=budget_bits,
            fixture=fix,
            rule_changed=rule_changed,
            cfg=cfg,
            obs_matrix=obs_matrix,
            exp_level_costs_arr=exp_costs,
        )
        for gm in res.generations:
            g = gm.generation
            heldout[i, g] = gm.heldout_regret
            train[i, g] = gm.train_regret
            entropy[i, g] = gm.posterior_entropy_bits
            accuracy[i, g] = gm.map_accuracy
            bits[i, g] = float(gm.bits_used)

    return arm_summary_from_arrays(
        arm=arm,
        budget_bits=budget_bits,
        rule_changed=rule_changed,
        heldout=heldout,
        train=train,
        entropy=entropy,
        accuracy=accuracy,
        bits=bits,
    )


def run_feasibility(
    n_lineages: int = 100, seed: int = 20261122, cfg: LibraryConfig = DEFAULT_CONFIG
) -> tuple[dict[str, Any], tuple[Check, ...]]:
    """Run the pre-protocol F1-F3 feasibility check plus stationary and rule-change diagnostics."""
    if n_lineages < 2:
        raise ValueError(f"n_lineages must be >= 2, got {n_lineages}")
    stationary_fixtures = [
        make_lineage_fixture(seed + i, rule_changed=False, cfg=cfg) for i in range(n_lineages)
    ]
    change_fixtures = [
        make_lineage_fixture(seed + 100_000 + i, rule_changed=True, cfg=cfg)
        for i in range(n_lineages)
    ]

    stationary_specs: tuple[tuple[str, int], ...] = (
        ("none", 0),
        ("unlimited", 0),
        ("single_lifetime", 0),
        ("example_random", 10),
        ("example_pedagogical", 10),
        ("rule", 10),
        ("rule_confidence", 10),
        ("placebo_random", 10),
        ("placebo_shuffled", 10),
        ("example_random", 20),
        ("example_pedagogical", 20),
        ("rule", 20),
        ("rule_confidence", 20),
        ("placebo_random", 20),
        ("placebo_shuffled", 20),
        ("example_pedagogical", 40),
        ("rule", 40),
        ("rule_confidence", 40),
    )
    change_specs: tuple[tuple[str, int], ...] = (
        ("none", 0),
        ("unlimited", 0),
        ("example_random", 20),
        ("example_pedagogical", 20),
        ("rule", 20),
        ("rule_confidence", 20),
        ("example_pedagogical", 40),
        ("rule", 40),
        ("rule_confidence", 40),
    )

    stationary_arms = {
        f"{arm}_b{b}": _summarize_arm(arm, b, stationary_fixtures, rule_changed=False, cfg=cfg)
        for arm, b in stationary_specs
    }
    change_arms = {
        f"{arm}_b{b}": _summarize_arm(arm, b, change_fixtures, rule_changed=True, cfg=cfg)
        for arm, b in change_specs
    }

    none_curve = stationary_arms["none_b0"].heldout_regret_mean_by_gen
    unlim_curve = stationary_arms["unlimited_b0"].heldout_regret_mean_by_gen
    pr_curve = stationary_arms["placebo_random_b20"].heldout_regret_mean_by_gen
    ps_curve = stationary_arms["placebo_shuffled_b20"].heldout_regret_mean_by_gen

    none_gen0 = none_curve[0]
    f1_deficit = max(0.0, F1_MIN_GAP - none_gen0)

    unlim_drop_after_gen2 = unlim_curve[1] - unlim_curve[-1]
    f2_deficit = max(0.0, F2_MIN_POST_GEN2_DROP - unlim_drop_after_gen2)

    none_post = sum(none_curve[1:]) / float(len(none_curve) - 1)
    pr_post = sum(pr_curve[1:]) / float(len(pr_curve) - 1)
    ps_post = sum(ps_curve[1:]) / float(len(ps_curve) - 1)
    placebo_advantage = max(0.0, none_post - min(pr_post, ps_post))

    checks = (
        Check(
            "f1_from_scratch_gap_deficit",
            f1_deficit,
            0.0,
            f"max(0, {F1_MIN_GAP} - mean gen-0 held-out regret of 'none') [gen0={none_gen0:.4f}]",
        ),
        Check(
            "f2_unlimited_post_gen2_drop_deficit",
            f2_deficit,
            0.0,
            f"max(0, {F2_MIN_POST_GEN2_DROP} - (unlimited gen-1 - gen-{cfg.n_generations - 1})) "
            f"[drop={unlim_drop_after_gen2:.4f}]",
        ),
        Check(
            "f3_placebo_advantage_over_none",
            placebo_advantage,
            F3_MAX_PLACEBO_ADVANTAGE,
            f"max(0, none mean regret (gens 1..{cfg.n_generations - 1}) - min(placebo_random, "
            f"placebo_shuffled) at B=20) [none={none_post:.4f}, rand={pr_post:.4f}, shuf={ps_post:.4f}]",
        ),
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha256(cfg),
        "config": config_to_dict(cfg),
        "n_lineages": n_lineages,
        "seed": seed,
        "gate_measurements": {
            "none_gen0_heldout_regret": none_gen0,
            "unlimited_gen1_heldout_regret": float(unlim_curve[1]),
            "unlimited_final_heldout_regret": float(unlim_curve[-1]),
            "unlimited_post_gen2_drop": unlim_drop_after_gen2,
            "none_post_gen0_mean_regret": none_post,
            "placebo_random_b20_post_gen0_mean_regret": pr_post,
            "placebo_shuffled_b20_post_gen0_mean_regret": ps_post,
        },
        "stationary": {k: v.to_json() for k, v in stationary_arms.items()},
        "rule_change": {k: v.to_json() for k, v in change_arms.items()},
    }
    return payload, checks


def export_feasibility_run(
    runs_dir: Path,
    run_id: str,
    *,
    n_lineages: int = 100,
    seed: int = 20261122,
    cfg: LibraryConfig = DEFAULT_CONFIG,
    created_at: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the feasibility suite and write an immutable run directory under `runs_dir/run_id`."""
    validate_run_id(run_id)
    out_dir = runs_dir / run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")

    payload, checks = run_feasibility(n_lineages=n_lineages, seed=seed, cfg=cfg)
    payload_with_id = {"run_id": run_id, **payload}
    git_commit, git_dirty = git_source()
    now = (datetime.now(UTC) if created_at is None else created_at).astimezone(UTC)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    limitations = [
        "Pre-protocol Stage L1 feasibility run; results establish task signal and placebo controls "
        "only and are not frozen confirmatory hypothesis tests.",
        "Latent rules use uniform discrete priors over K=8 terrain types and L=4 cost levels.",
    ]
    if git_dirty:
        limitations.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )

    gm = payload["gate_measurements"]
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
        "environment": environment_info(),
        "model": {
            "name": "voronoi-terrain-navigation",
            "config_sha256": config_sha256(cfg),
            **config_to_dict(cfg),
        },
        "random_seed": seed,
        "artifacts": [],
        "metrics": {
            "none_gen0_heldout_regret": {
                "value": gm["none_gen0_heldout_regret"],
                "unit": "dimensionless",
                "definition": "mean generation-0 held-out expected path regret of 'none'",
            },
            "unlimited_post_gen2_drop": {
                "value": gm["unlimited_post_gen2_drop"],
                "unit": "dimensionless",
                "definition": "mean 'unlimited' held-out regret reduction from gen 1 to final gen",
            },
            "placebo_random_b20_mean_regret": {
                "value": gm["placebo_random_b20_post_gen0_mean_regret"],
                "unit": "dimensionless",
                "definition": "mean held-out regret across generations 1..T-1 for random placebo B=20",
            },
            "placebo_shuffled_b20_mean_regret": {
                "value": gm["placebo_shuffled_b20_post_gen0_mean_regret"],
                "unit": "dimensionless",
                "definition": "mean held-out regret across generations 1..T-1 for shuffled placebo B=20",
            },
        },
        "limitations": limitations,
        "validation": {
            "checks": [c.to_json() for c in checks],
            "validated_by": f"{GENERATOR} automated feasibility gate checks (pass: observed <= threshold)",
            "validated_at": timestamp,
        },
    }
    written = write_run_directory(
        out_dir,
        manifest,
        [("feasibility_summary", "feasibility.json", dumps_json(payload_with_id))],
    )
    return written, payload_with_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m phasespace.library.feasibility")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--lineages", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261122)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, payload = export_feasibility_run(
        args.runs_dir, args.run_id, n_lineages=args.lineages, seed=args.seed
    )
    print(f"Wrote {args.runs_dir / args.run_id} (status={manifest['status']})")
    for chk in manifest["validation"]["checks"]:
        tag = "PASS" if chk["passed"] else "FAIL"
        print(f"  [{tag}] {chk['name']}: observed={chk['observed']:.4f} <= {chk['threshold']}")
    st = payload["stationary"]
    rc = payload["rule_change"]
    c = int(payload["config"]["change_generation"])
    print(
        "Stationary gen 9 (final) held-out regret: "
        f"none={st['none_b0']['heldout_regret_mean_by_gen'][-1]:.2f}, "
        f"unlimited={st['unlimited_b0']['heldout_regret_mean_by_gen'][-1]:.2f}, "
        f"ex_ped@20={st['example_pedagogical_b20']['heldout_regret_mean_by_gen'][-1]:.2f}, "
        f"rule@20={st['rule_b20']['heldout_regret_mean_by_gen'][-1]:.2f}, "
        f"rule_conf@20={st['rule_confidence_b20']['heldout_regret_mean_by_gen'][-1]:.2f}, "
        f"rule_conf@40={st['rule_confidence_b40']['heldout_regret_mean_by_gen'][-1]:.2f}"
    )
    print(
        f"Rule-change gen {c} (first post-change) held-out regret: "
        f"none={rc['none_b0']['heldout_regret_mean_by_gen'][c]:.2f}, "
        f"ex_ped@20={rc['example_pedagogical_b20']['heldout_regret_mean_by_gen'][c]:.2f}, "
        f"rule@20={rc['rule_b20']['heldout_regret_mean_by_gen'][c]:.2f}, "
        f"rule_conf@20={rc['rule_confidence_b20']['heldout_regret_mean_by_gen'][c]:.2f}, "
        f"rule_conf@40={rc['rule_confidence_b40']['heldout_regret_mean_by_gen'][c]:.2f}"
    )
    return 0 if manifest["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
