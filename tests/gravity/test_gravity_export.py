"""Export of the schema-v1 reference run (docs/data-contract.md)."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from jpl_catalog import load_catalog

from phasespace.gravity import export
from phasespace.gravity.export import (
    DEFAULT_RUN_ID,
    Check,
    ReferenceComputation,
    compute_reference,
    main,
    run_status,
    write_run,
)

CATALOG = load_catalog()
CREATED_AT = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)
FAKE_COMMIT = "0" * 40


@pytest.fixture(scope="module")
def computation() -> ReferenceComputation:
    return compute_reference()


def _write(path: Path, computation: ReferenceComputation) -> dict[str, Any]:
    return write_run(
        path, computation, git_commit=FAKE_COMMIT, git_dirty=False, created_at=CREATED_AT
    )


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_manifest_schema_fields(tmp_path: Path, computation: ReferenceComputation) -> None:
    out = tmp_path / "gravity-test-run"
    _write(out, computation)
    manifest = _load(out / "manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "gravity-test-run"
    assert manifest["study_id"] == "gravity-atlas"
    assert manifest["experiment_id"]
    assert manifest["status"] == "validated"
    assert manifest["created_at"] == "2026-10-18T12:00:00Z"
    assert manifest["random_seed"] is None
    source = manifest["source"]
    assert source["git_commit"] == FAKE_COMMIT
    assert source["git_dirty"] is False
    assert source["generator"] == "phasespace.gravity.export"
    model = manifest["model"]
    assert model["name"] == "planar-cr3bp"
    assert model["coordinate_frame"] == "barycentric_synodic"
    assert model["mu"] == CATALOG.mu
    assert model["length_scale_km"] == CATALOG.lunit_km
    assert model["time_scale_seconds"] == CATALOG.tunit_s
    assert model["primaries"]["earth"]["x"] == -CATALOG.mu
    assert "planar" in model["assumptions"]
    assert manifest["solver"]["name"] == "DOP853"
    assert manifest["solver"]["rtol"] > 0 and manifest["solver"]["atol"] > 0
    assert manifest["limitations"]
    for metric in manifest["metrics"].values():
        assert set(metric) == {"value", "unit", "definition"}
    validation = manifest["validation"]
    assert validation["validated_at"] == "2026-10-18T12:00:00Z"
    assert validation["validated_by"]
    assert validation["reference"]["url"] == CATALOG.source_url
    assert validation["checks"]
    for check in validation["checks"]:
        assert set(check) == {"name", "observed", "threshold", "passed", "definition"}
        assert check["passed"] is True
        assert check["observed"] <= check["threshold"]
        assert check["definition"]


EXPECTED_CHECKS = {
    "periodicity_max_abs_state_error_one_period",
    "jacobi_max_abs_drift_exported_samples",
    "jacobi_max_abs_drift_3_periods",
    "tight_tolerance_max_abs_state_difference_samples",
    "independent_integrator_max_abs_state_difference_samples",
    "jpl_catalog_abs_diff_vy0",
    "jpl_catalog_abs_diff_period",
    "jpl_catalog_abs_diff_jacobi",
    "jpl_catalog_rel_diff_stability_index",
    "non_horizon_terminations_or_collision_events",
    "non_finite_sample_values",
    "non_increasing_sample_times",
}


def test_manifest_matches_v1_contract(tmp_path: Path, computation: ReferenceComputation) -> None:
    # Key sets of docs/data-contract.md (all required fields present, nothing undocumented).
    out = tmp_path / "gravity-test-run"
    _write(out, computation)
    manifest = _load(out / "manifest.json")
    assert set(manifest) == {
        "schema_version",
        "run_id",
        "study_id",
        "experiment_id",
        "status",
        "created_at",
        "source",
        "environment",
        "model",
        "solver",
        "random_seed",
        "artifacts",
        "metrics",
        "limitations",
        "validation",
    }
    assert set(manifest["source"]) == {
        "repository",
        "git_commit",
        "git_dirty",
        "generator",
        "generator_version",
    }
    environment = manifest["environment"]
    assert set(environment) == {
        "python",
        "python_implementation",
        "numpy",
        "scipy",
        "platform",
        "machine",
    }
    assert all(isinstance(v, str) and v for v in environment.values())
    model = manifest["model"]
    assert model["length_scale_convention"] == "jpl_periodic_orbits_catalog"
    assert model["velocity_scale_km_s"] == CATALOG.lunit_km / CATALOG.tunit_s
    assert model["primaries"]["moon"] == {
        "x": 1.0 - CATALOG.mu,
        "y": 0.0,
        "radius_km": CATALOG.radius_secondary_km,
    }
    solver = manifest["solver"]
    assert solver["max_evaluations"] == 1_000_000
    assert solver["collision_events"] == {
        "earth_radius_km": 6371.0,
        "moon_radius_km": CATALOG.radius_secondary_km,
        "function": "distance_minus_radius",
        "direction": "decreasing",
        "terminal": True,
        "grazing_detection": "periapsis_below_radius",
        "start_inside": "not_propagated",
    }
    assert [c["role"] for c in solver["comparisons"]] == [
        "tight_tolerance",
        "independent_integrator",
    ]
    for block in (*solver["comparisons"], solver["monodromy"], solver["differential_correction"]):
        assert block["name"] in {"DOP853", "LSODA", "Radau", "RK45"}
        assert block["rtol"] > 0.0 and block["atol"] > 0.0
    convergence = solver["differential_correction"]["convergence"]
    assert (
        convergence["half_period_abs_vx"]
        < solver["differential_correction"]["half_period_abs_vx_tolerance"]
    )
    assert convergence["iterations_final_member"] >= 1
    assert {c["name"] for c in manifest["validation"]["checks"]} == EXPECTED_CHECKS
    reference = manifest["validation"]["reference"]
    assert reference["stability_index"] == next(
        o.stability for o in CATALOG.orbits if o.row_index == 2600
    )
    units = {"normalized_cr3bp", "km", "km/s", "day", "hour", "count", "dimensionless"}
    for name, metric in manifest["metrics"].items():
        assert set(metric) == {"value", "unit", "definition"}
        assert re.fullmatch(r"[a-z0-9_]+", name)
        assert metric["unit"] in units, name
        if metric["unit"] == "km":
            assert f"length_scale_km = {CATALOG.lunit_km} km" in metric["definition"], name
        if metric["unit"] in {"km", "km/s"}:
            assert "jpl_periodic_orbits_catalog" in metric["definition"], name


def test_monodromy_metrics_match_catalog_stability(computation: ReferenceComputation) -> None:
    # Independent reference: the catalog's nu gives lambda_max = nu + sqrt(nu^2 - 1).
    nu = next(o.stability for o in CATALOG.orbits if o.row_index == 2600)
    lambda_max = computation.metrics["monodromy_lambda_max"]["value"]
    assert lambda_max == pytest.approx(nu + np.sqrt(nu * nu - 1.0), rel=1e-9)
    assert lambda_max == pytest.approx(1808.93, abs=0.01)
    assert computation.metrics["monodromy_stability_index"]["value"] == pytest.approx(nu, rel=1e-9)


def test_artifact_hash_and_trajectory_invariants(
    tmp_path: Path, computation: ReferenceComputation
) -> None:
    out = tmp_path / "gravity-test-run"
    manifest = _write(out, computation)
    (artifact,) = manifest["artifacts"]
    assert artifact["kind"] == "trajectory"
    assert artifact["path"] == "trajectory.json"
    data = (out / "trajectory.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    assert _load(out / "manifest.json")["artifacts"] == manifest["artifacts"]

    trajectory = _load(out / "trajectory.json")
    assert trajectory["schema_version"] == 1
    assert trajectory["run_id"] == "gravity-test-run"
    assert trajectory["frame"] == "barycentric_synodic"
    assert trajectory["time_unit"] == "normalized_cr3bp"
    assert trajectory["state_order"] == ["t", "x", "y", "vx", "vy"]
    samples = np.array(trajectory["samples"], dtype=np.float64)
    assert samples.ndim == 2 and samples.shape[1] == 5
    assert 300 <= samples.shape[0] <= 2001
    assert np.all(np.isfinite(samples))
    t = samples[:, 0]
    assert np.all(np.diff(t) > 0.0)
    step = manifest["metrics"]["sample_step_normalized"]["value"]
    np.testing.assert_allclose(np.diff(t), step, rtol=0.0, atol=1e-14)

    # Independent physics checks on the exported numbers, using catalog values only.
    x, y, vx, vy = samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4]
    mu = CATALOG.mu
    r1 = np.hypot(x + mu, y)
    r2 = np.hypot(x - 1 + mu, y)
    jacobi = x**2 + y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - vx**2 - vy**2
    reference = next(o for o in CATALOG.orbits if o.row_index == 2600)
    assert np.max(np.abs(jacobi - reference.jacobi)) < 1e-10
    assert t[-1] == pytest.approx(reference.period, abs=1e-10)
    np.testing.assert_allclose(samples[0, 1:], [reference.x0, 0.0, 0.0, reference.vy0], atol=1e-10)
    np.testing.assert_allclose(samples[-1, 1:], samples[0, 1:], atol=1e-9)


def test_refuses_to_overwrite(tmp_path: Path, computation: ReferenceComputation) -> None:
    out = tmp_path / "gravity-test-run"
    _write(out, computation)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _write(out, computation)
    assert (out / "manifest.json").read_bytes() == before
    empty = tmp_path / "existing-empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        _write(empty, computation)


def test_rejects_unsafe_run_id(tmp_path: Path, computation: ReferenceComputation) -> None:
    with pytest.raises(ValueError):
        _write(tmp_path / "Bad Run", computation)


def test_status_requires_all_checks_to_pass() -> None:
    ok = Check("a", 1e-12, 1e-10, "test")
    bad = Check("b", 1e-8, 1e-10, "test")
    nan = Check("c", float("nan"), 1e-10, "test")
    assert run_status([ok]) == "validated"
    assert run_status([ok, bad]) == "computed_unvalidated"
    assert run_status([ok, nan]) == "computed_unvalidated"
    assert run_status([]) == "computed_unvalidated"
    assert nan.to_json()["observed"] is None
    assert nan.to_json()["passed"] is False
    assert ok.to_json()["observed"] == 1e-12


def test_write_is_atomic_and_leaves_no_staging(
    tmp_path: Path, computation: ReferenceComputation, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_rename(src: object, dst: object) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr(export.os, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated failure"):
        _write(tmp_path / "gravity-test-run", computation)
    assert list(tmp_path.iterdir()) == []


def test_refusal_leaves_no_staging(tmp_path: Path, computation: ReferenceComputation) -> None:
    existing = tmp_path / "gravity-test-run"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _write(existing, computation)
    assert [p.name for p in tmp_path.iterdir()] == ["gravity-test-run"]
    assert list(existing.iterdir()) == []


def test_cli_writes_run_and_refuses_existing(tmp_path: Path) -> None:
    assert main(["--runs-dir", str(tmp_path), "--run-id", "cli-run"]) == 0
    out = tmp_path / "cli-run"
    manifest = _load(out / "manifest.json")
    assert manifest["status"] == "validated"
    assert manifest["run_id"] == "cli-run"
    assert _load(out / "trajectory.json")["run_id"] == "cli-run"
    assert len(manifest["source"]["git_commit"]) == 40
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cli-run"]
    with pytest.raises(SystemExit):
        main(["--runs-dir", str(tmp_path), "--run-id", "cli-run"])
    with pytest.raises(SystemExit):
        main(["--runs-dir", str(tmp_path), "--run-id", "../escape"])


def test_default_run_id_is_v2() -> None:
    # v1 predates the numerics-review fixes and must not be published.
    assert DEFAULT_RUN_ID == "gravity-l1-lyapunov-reference-v2"
