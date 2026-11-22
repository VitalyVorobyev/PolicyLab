"""Protocol guard, statistical inference, worker parity, and export tests for experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import phasespace.library.experiment as experiment_mod
from phasespace.library.config import config_sha256
from phasespace.library.experiment import (
    export_confirmatory_run,
    holm_adjust,
    paired_contrast,
    run_experiment,
)
from phasespace.library.experiment import (
    main as experiment_main,
)


def test_holm_adjust_matches_hand_calculation() -> None:
    # Step-down monotonic max check:
    # Sorted: p_0=0.01 (*3 -> 0.03), p_2=0.03 (*2 -> 0.06), p_1=0.04 (*1 -> 0.04 => max=0.06)
    adj1 = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj1, [0.03, 0.06, 0.06], atol=1e-15)

    # Sorted: 0.001 (*3 -> 0.003), 0.02 (*2 -> 0.04), 0.025 (*1 -> 0.025 => max=0.04)
    adj2 = holm_adjust([0.025, 0.001, 0.02])
    np.testing.assert_allclose(adj2, [0.04, 0.003, 0.04], atol=1e-15)

    assert holm_adjust([]) == []
    assert holm_adjust([0.8, 0.9]) == [1.0, 1.0]

    with pytest.raises(ValueError, match="Invalid p-value"):
        holm_adjust([-0.01, 0.5])


def test_paired_contrast_matches_scipy_ttest() -> None:
    rng = np.random.default_rng(12345)
    diffs = rng.normal(loc=1.8, scale=0.9, size=40)
    delta = 0.50

    res = paired_contrast(
        diffs,
        contrast_id="test_c",
        family="test",
        label="test contrast",
        higher_expected_arm="A",
        lower_expected_arm="B",
        delta=delta,
        n_resamples=2000,
        bootstrap_seed=9201,
    )
    scipy_res = stats.ttest_1samp(diffs, popmean=delta, alternative="greater")
    assert res["df"] == len(diffs) - 1
    assert res["t_stat"] == pytest.approx(float(scipy_res.statistic), abs=1e-12)
    assert res["p_raw"] == pytest.approx(float(scipy_res.pvalue), abs=1e-12)
    assert res["ci95_low"] < res["mean_diff"] < res["ci95_high"]

    # Deterministic given identical bootstrap seed:
    res_repeat = paired_contrast(
        diffs,
        contrast_id="test_c",
        family="test",
        label="test contrast",
        higher_expected_arm="A",
        lower_expected_arm="B",
        delta=delta,
        n_resamples=2000,
        bootstrap_seed=9201,
    )
    assert res["ci95_low"] == res_repeat["ci95_low"]
    assert res["ci95_high"] == res_repeat["ci95_high"]

    # Zero-variance degenerate array branch:
    zero_var = paired_contrast(
        [1.0, 1.0, 1.0],
        contrast_id="const",
        family="test",
        label="const",
        higher_expected_arm="A",
        lower_expected_arm="B",
        delta=0.5,
        n_resamples=100,
        bootstrap_seed=1,
    )
    assert zero_var["p_raw"] == 0.0
    assert zero_var["ci95_low"] == 1.0 and zero_var["ci95_high"] == 1.0


def test_run_experiment_workers_bit_identical() -> None:
    res1, checks1 = run_experiment(n_lineages=4, workers=1)
    res2, checks2 = run_experiment(n_lineages=4, workers=2)
    assert res1 == res2
    assert [c.to_json() for c in checks1] == [c.to_json() for c in checks2]


def test_protocol_guards_and_confirmatory_smoke_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proto_text = Path("docs/studies/last-library/protocol.md").read_text("utf-8")

    # 1. Refuses without --confirm-frozen:
    with pytest.raises(RuntimeError, match="confirm-frozen"):
        export_confirmatory_run(
            tmp_path,
            "smoke-no-confirm",
            confirm_frozen=False,
            allow_dirty_for_test=True,
            n_lineages=4,
            workers=1,
        )

    # 1b. Refuses dirty working tree when allow_dirty_for_test=False:
    monkeypatch.setattr(experiment_mod, "git_source", lambda: ("deadbeef", True))
    with pytest.raises(RuntimeError, match="dirty git working tree"):
        export_confirmatory_run(
            tmp_path,
            "smoke-dirty",
            confirm_frozen=True,
            allow_dirty_for_test=False,
            n_lineages=4,
            workers=1,
        )
    monkeypatch.undo()

    # 2. Refuses if protocol status is draft:
    draft_proto = tmp_path / "draft_protocol.md"
    draft_proto.write_text(proto_text.replace("**frozen**", "**draft**", 1), "utf-8")
    with pytest.raises(RuntimeError, match="expected 'frozen'"):
        export_confirmatory_run(
            tmp_path,
            "smoke-draft",
            confirm_frozen=True,
            allow_dirty_for_test=True,
            protocol_path=draft_proto,
            n_lineages=4,
            workers=1,
        )

    # 3. Refuses if config_sha256 mismatches:
    bad_hash_proto = tmp_path / "bad_hash_protocol.md"
    bad_hash_proto.write_text(proto_text.replace(config_sha256(), "0" * 64, 1), "utf-8")
    with pytest.raises(RuntimeError, match="Config hash mismatch"):
        export_confirmatory_run(
            tmp_path,
            "smoke-bad-hash",
            confirm_frozen=True,
            allow_dirty_for_test=True,
            protocol_path=bad_hash_proto,
            n_lineages=4,
            workers=1,
        )

    # 4. Smoke run with --confirm-frozen --allow-dirty-for-test exports complete verified artifacts:
    run_id = "library-smoke-v1"
    exit_code = experiment_main(
        [
            "--runs-dir",
            str(tmp_path),
            "--run-id",
            run_id,
            "--confirm-frozen",
            "--allow-dirty-for-test",
            "--lineages",
            "6",
            "--workers",
            "2",
        ]
    )
    assert exit_code in (0, 1)

    out_dir = tmp_path / run_id
    manifest = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["study_id"] == "last-library"
    assert manifest["protocol"]["status"] == "frozen"
    assert manifest["protocol"]["config_sha256"] == config_sha256()

    artifact_paths = {a["path"]: a["sha256"] for a in manifest["artifacts"]}
    assert set(artifact_paths) == {"results.json", "samples.json"}
    for rel_path, expected_sha in artifact_paths.items():
        actual_sha = hashlib.sha256((out_dir / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha

    results = json.loads((out_dir / "results.json").read_text("utf-8"))
    assert set(results["hypotheses"]) == {"H1", "H2a", "H2b", "H3a", "H3b", "H4"}
    assert "rule_hedged" in results["budget_curves"]["arms"]

    samples = json.loads((out_dir / "samples.json").read_text("utf-8"))
    assert samples["sample_indices"] == [0, 1, 2]
    assert len(samples["stationary"]) == 3
    first_sample = samples["stationary"][0]
    assert len(first_sample["layouts_by_key_gen"]) == 4
    assert "rule_hedged_b20" in first_sample["arms"]
    assert "0" in first_sample["arms"]["rule_hedged_b20"]["planned_paths_by_key_gen"]

    # 5. Refuses overwriting existing run directory:
    with pytest.raises(FileExistsError, match="never overwritten"):
        export_confirmatory_run(
            tmp_path,
            run_id,
            confirm_frozen=True,
            allow_dirty_for_test=True,
            n_lineages=6,
            workers=1,
        )
