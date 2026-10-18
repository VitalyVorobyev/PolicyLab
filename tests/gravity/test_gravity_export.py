"""Export of the schema-v1 reference run (docs/data-contract.md)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from jpl_catalog import load_catalog

from phasespace.gravity.export import (
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
        assert set(check) == {"name", "observed", "threshold", "passed"}
        assert check["passed"] is True
        assert check["observed"] <= check["threshold"]


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
    ok = Check("a", 1e-12, 1e-10)
    bad = Check("b", 1e-8, 1e-10)
    nan = Check("c", float("nan"), 1e-10)
    assert run_status([ok]) == "validated"
    assert run_status([ok, bad]) == "computed_unvalidated"
    assert run_status([ok, nan]) == "computed_unvalidated"
    assert run_status([]) == "computed_unvalidated"


def test_cli_writes_run_and_refuses_existing(tmp_path: Path) -> None:
    out = tmp_path / "cli-run"
    assert main(["--out", str(out)]) == 0
    manifest = _load(out / "manifest.json")
    assert manifest["status"] == "validated"
    assert len(manifest["source"]["git_commit"]) == 40
    with pytest.raises(SystemExit):
        main(["--out", str(out)])
