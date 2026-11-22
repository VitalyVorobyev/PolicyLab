"""Simulation determinism, oracle/unlimited/single-lifetime invariants, and feasibility CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phasespace.library.config import DEFAULT_CONFIG
from phasespace.library.feasibility import main as feasibility_main
from phasespace.library.simulate import make_lineage_fixture, run_lineage


def test_oracle_zero_regret_and_determinism() -> None:
    fix = make_lineage_fixture(seed=1001, rule_changed=False)
    oracle = run_lineage("oracle", 0, fix, rule_changed=False)
    assert all(gm.heldout_regret == 0.0 and gm.train_regret == 0.0 for gm in oracle.generations)
    assert all(gm.map_accuracy == 1.0 for gm in oracle.generations)

    r1 = run_lineage("rule", 20, fix, rule_changed=False)
    r2 = run_lineage("rule", 20, fix, rule_changed=False)
    assert r1 == r2
    assert r1.generations[0].bits_used == 0
    assert all(gm.bits_used == 19 for gm in r1.generations[1:])

    r_hedged = run_lineage("rule_hedged", 20, fix, rule_changed=False)
    assert r_hedged.generations[0] == r1.generations[0]
    assert [gm.bits_used for gm in r_hedged.generations] == [gm.bits_used for gm in r1.generations]


def test_unlimited_matches_single_lifetime_and_beats_none() -> None:
    n_lin = 20
    fixtures = [make_lineage_fixture(5000 + i, rule_changed=False) for i in range(n_lin)]
    none_final: list[float] = []
    unlim_final: list[float] = []

    for fix in fixtures:
        r_none = run_lineage("none", 0, fix, rule_changed=False)
        r_unlim = run_lineage("unlimited", 0, fix, rule_changed=False)
        r_single = run_lineage("single_lifetime", 0, fix, rule_changed=False)
        assert r_unlim.generations[-1].heldout_regret == pytest.approx(
            r_single.generations[-1].heldout_regret, abs=1e-12
        )
        none_final.append(r_none.generations[-1].heldout_regret)
        unlim_final.append(r_unlim.generations[-1].heldout_regret)

    assert sum(unlim_final) / n_lin < sum(none_final) / n_lin
    # Confirm generation 0 aligns between 'none' and 'unlimited':
    r_none0 = run_lineage("none", 0, fixtures[0], rule_changed=False).generations[0]
    r_unlim0 = run_lineage("unlimited", 0, fixtures[0], rule_changed=False).generations[0]
    assert r_none0.heldout_regret == pytest.approx(r_unlim0.heldout_regret, abs=1e-12)


def test_feasibility_cli_exports_verified_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    run_id = "library-feasibility-smoke-v1"
    # Note: with n_lineages=8, CLI writes the run regardless of gate exit code 0/1; check artifacts.
    feasibility_main(
        ["--runs-dir", str(tmp_path), "--run-id", run_id, "--lineages", "8", "--seed", "7"]
    )
    out_dir = tmp_path / run_id
    manifest = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["study_id"] == "last-library"
    assert manifest["run_id"] == run_id
    assert len(manifest["artifacts"]) == 1
    art = manifest["artifacts"][0]
    assert art["path"] == "feasibility.json"
    actual_sha = hashlib.sha256((out_dir / "feasibility.json").read_bytes()).hexdigest()
    assert actual_sha == art["sha256"]
    payload = json.loads((out_dir / "feasibility.json").read_text("utf-8"))
    assert payload["config"]["grid_size"] == DEFAULT_CONFIG.grid_size

    with pytest.raises(FileExistsError, match="never overwritten"):
        feasibility_main(
            ["--runs-dir", str(tmp_path), "--run-id", run_id, "--lineages", "8", "--seed", "7"]
        )
