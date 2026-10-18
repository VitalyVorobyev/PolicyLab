"""M3 pipeline smoke tests: map designs, table round trip, search budget, validation, runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from phasespace.gravity.free_return import (
    DEFAULT_DEFINITION,
    FreeReturnDefinition,
    FreeReturnRecord,
    Outcome,
    evaluate,
)
from phasespace.gravity.free_return_export import (
    build_candidate_documents,
    check_map_compatible,
    load_map_records,
    run_map,
)
from phasespace.gravity.free_return_map import (
    COLUMNS,
    GridDesign,
    SobolDesign,
    dumps_table,
    evaluate_points,
    record_from_row,
    record_row,
    records_table,
)
from phasespace.gravity.free_return_search import (
    SearchSettings,
    StartResult,
    distinct_converged,
    refine,
    select_starts,
)
from phasespace.gravity.free_return_validation import validate_candidate

FAKE_COMMIT = "0" * 40
# Small grid around the pilot free-return region: feasible, collision and no-return cells.
SMOKE_GRID = GridDesign(
    phase_min_deg=226.0,
    phase_step_deg=1.0,
    n_phase=4,
    delta_v_min_km_s=3.16,
    delta_v_step_km_s=0.005,
    n_delta_v=4,
)


@pytest.fixture(scope="module")
def smoke_records() -> list[FreeReturnRecord]:
    return evaluate_points(SMOKE_GRID.points(), workers=1)


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_grid_points_are_phase_major_and_cover_the_s1_box() -> None:
    points = SMOKE_GRID.points()
    assert len(points) == 16
    assert points[:2] == [(226.0, 3.16, 0.0), (226.0, 3.165, 0.0)]
    assert points[4] == (227.0, 3.16, 0.0)
    s1 = GridDesign()
    assert s1.size == 90_720
    full = s1.points()
    assert full[0] == (0.0, 3.0, 0.0)
    assert full[-1] == (359.5, 3.25, 0.0)


def test_sobol_points_are_seeded_and_in_bounds() -> None:
    a = SobolDesign(seed=7, log2_samples=6).points()
    assert a == SobolDesign(seed=7, log2_samples=6).points()
    assert a != SobolDesign(seed=8, log2_samples=6).points()
    arr = np.array(a)
    assert arr.shape == (64, 3)
    assert np.all((arr[:, 0] >= 0.0) & (arr[:, 0] < 360.0))
    assert np.all((arr[:, 1] >= 3.0) & (arr[:, 1] <= 3.25))
    assert np.all((arr[:, 2] >= -10.0) & (arr[:, 2] <= 10.0))


def test_map_is_independent_of_worker_count_and_round_trips(
    smoke_records: list[FreeReturnRecord],
) -> None:
    serial = smoke_records
    parallel = evaluate_points(SMOKE_GRID.points(), workers=2, chunksize=3)
    assert [record_row(r) for r in serial] == [record_row(r) for r in parallel]
    outcomes = {r.outcome for r in serial}
    assert Outcome.FEASIBLE in outcomes and len(outcomes) >= 2

    table = json.loads(dumps_table(records_table("x", serial)))
    assert [c["name"] for c in table["columns"]] == [c.name for c in COLUMNS]
    back = [record_from_row(row) for row in table["rows"]]
    assert [record_row(r) for r in back] == [record_row(r) for r in serial]


def test_map_run_is_written_once_with_checksum(tmp_path: Path) -> None:
    design = replace(SMOKE_GRID, n_phase=2, n_delta_v=2)
    out = tmp_path / "gravity-free-return-map-test"
    manifest, records = run_map(out, design, workers=1, git_commit=FAKE_COMMIT, git_dirty=True)
    on_disk = _load(out / "manifest.json")
    assert on_disk == manifest
    (artifact,) = on_disk["artifacts"]
    assert artifact["kind"] == "sample_table"
    data = (out / artifact["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    assert on_disk["random_seed"] is None
    assert on_disk["status"] == "computed_unvalidated"
    assert on_disk["solver"]["return_shell"]["radius_km"] == pytest.approx(6371.0 + 121.92)
    assert on_disk["metrics"]["sample_count"]["value"] == 4
    assert any("git_dirty" in s for s in on_disk["limitations"])
    loaded, digest = load_map_records(out)
    assert digest == artifact["sha256"]
    assert [record_row(r) for r in loaded] == [record_row(r) for r in records]
    with pytest.raises(FileExistsError):
        run_map(out, design, workers=1, git_commit=FAKE_COMMIT, git_dirty=True)


def test_select_starts_orders_feasible_by_residual(
    smoke_records: list[FreeReturnRecord],
) -> None:
    starts = select_starts(smoke_records, 3)
    assert starts and all(s.feasible for s in starts)
    norms = [
        math.hypot(
            (s.perilune_altitude_km - 1000.0) / 1000.0, (s.vacuum_perigee_altitude_km - 38.0) / 10.0
        )
        for s in starts
    ]
    assert norms == sorted(norms)


def test_refine_respects_the_propagation_cap() -> None:
    start = evaluate(227.63, 3.1628)
    result = refine(start, settings=SearchSettings(max_propagations_per_start=5))
    assert result.propagations == 5
    assert not result.converged
    assert "cap" in result.message


def test_distinct_converged_removes_duplicates_and_sorts_by_delta_v() -> None:
    base = evaluate(227.63, 3.1628)

    def fake(phase: float, dv: float, converged: bool) -> StartResult:
        final = replace(base, phase_deg=phase, delta_v_km_s=dv)
        return StartResult(base, final, (0.0, 0.0), 10, converged, "fake")

    results = [
        fake(229.0, 3.157, True),
        fake(229.0 + 1e-6, 3.157 + 1e-9, True),
        fake(225.8, 3.150, True),
        fake(200.0, 3.100, False),
    ]
    kept = distinct_converged(results)
    assert [(k.final.phase_deg, k.final.delta_v_km_s) for k in kept] == [
        (225.8, 3.150),
        (229.0, 3.157),
    ]


def test_validation_passes_on_the_pilot_sample_and_documents_build() -> None:
    validation = validate_candidate(227.63, 3.1628)
    failed = [c.name for c in validation.checks if not c.passed]
    assert failed == []
    assert validation.passed
    samples = validation.samples
    assert np.all(np.diff(samples[:, 0]) > 0.0)
    assert samples[-1, 0] == validation.run.t_final
    assert samples.shape[0] > 2000

    start = validation.records[0]
    search = StartResult(start, start, (0.0, 0.0), 0, True, "constructed")
    manifest, trajectory = build_candidate_documents(
        "gravity-free-return-test",
        validation,
        search,
        definition=DEFAULT_DEFINITION,
        map_run_id="map",
        map_sha256="0" * 64,
        search_run_id="search",
        git_commit=FAKE_COMMIT,
        git_dirty=False,
        created_at=datetime(2026, 10, 18, tzinfo=UTC),
    )
    assert manifest["status"] == "validated"
    assert manifest["termination"] == "earth_return_shell"
    assert manifest["metrics"]["departure_phase_deg"]["unit"] == "deg"
    doc = json.loads(trajectory)
    assert [e["name"] for e in doc["events"]] == [
        "departure",
        "perilune",
        "return_shell_crossing",
    ]
    assert doc["samples"][-1][0] == doc["events"][-1]["t"]


def test_validation_flags_an_infeasible_point() -> None:
    validation = validate_candidate(225.0, 3.15)
    assert not validation.passed
    assert validation.checks[0].observed == 3.0


def test_documents_build_for_a_failed_validation() -> None:
    # Regression: a candidate that fails validation must still export (status
    # computed_unvalidated) without NaN events or metrics, and without a shell-crossing
    # event when the run ended on the horizon.
    validation = validate_candidate(225.0, 3.15)
    start = validation.records[0]
    search = StartResult(start, start, None, 0, False, "constructed")
    manifest, trajectory = build_candidate_documents(
        "gravity-free-return-test",
        validation,
        search,
        definition=DEFAULT_DEFINITION,
        map_run_id="map",
        map_sha256="0" * 64,
        search_run_id="search",
        git_commit=FAKE_COMMIT,
        git_dirty=False,
        created_at=datetime(2026, 10, 18, tzinfo=UTC),
    )
    assert manifest["status"] == "computed_unvalidated"
    assert manifest["termination"] == "horizon"
    names = [e["name"] for e in json.loads(trajectory)["events"]]
    assert "return_shell_crossing" not in names
    assert names[0] == "departure"
    assert all(math.isfinite(m["value"]) for m in manifest["metrics"].values())


def test_candidates_require_a_compatible_2d_map(tmp_path: Path) -> None:
    out = tmp_path / "gravity-free-return-sobol-test"
    run_map(
        out, SobolDesign(seed=1, log2_samples=1), workers=1, git_commit=FAKE_COMMIT, git_dirty=False
    )
    with pytest.raises(ValueError, match="not a"):
        check_map_compatible(out, DEFAULT_DEFINITION)
    grid = tmp_path / "gravity-free-return-grid-test"
    run_map(
        grid,
        replace(SMOKE_GRID, n_phase=1, n_delta_v=1),
        workers=1,
        git_commit=FAKE_COMMIT,
        git_dirty=False,
    )
    check_map_compatible(grid, DEFAULT_DEFINITION)
    with pytest.raises(ValueError):
        check_map_compatible(grid, FreeReturnDefinition(t_max_days=12.0))
