"""Derived web outcome grid: lossless round trip, provenance and refusal cases (synthetic map)."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from phasespace.gravity.export import write_run_directory
from phasespace.gravity.free_return import DEFAULT_DEFINITION, FreeReturnRecord, Outcome
from phasespace.gravity.free_return_export import (
    OUTCOME_GRID_FILE,
    build_map_documents,
    main,
    run_outcome_grid,
)
from phasespace.gravity.free_return_grid import (
    OUTCOMES,
    decode_codes,
    decode_outcomes,
    encode_codes,
    outcome_grid,
)
from phasespace.gravity.free_return_map import GridDesign, dumps_table, records_table
from phasespace.gravity.propagation import Termination

FAKE_COMMIT = "1" * 40
DESIGN = GridDesign(
    phase_min_deg=10.0,
    phase_step_deg=0.5,
    n_phase=3,
    delta_v_min_km_s=3.1,
    delta_v_step_km_s=0.002,
    n_delta_v=4,
)
# Hand-written outcome pattern (phase-major): every outcome appears at least once.
PATTERN: tuple[Outcome, ...] = (
    Outcome.NO_RETURN_WITHIN_T_MAX,
    Outcome.FEASIBLE,
    Outcome.MOON_COLLISION,
    Outcome.PERILUNE_OUT_OF_BAND,
    Outcome.RETURN_BEFORE_LUNAR_ENCOUNTER,
    Outcome.FEASIBLE,
    Outcome.NUMERICAL_FAILURE,
    Outcome.INVALID_START,
    Outcome.EARTH_COLLISION,
    Outcome.NO_RETURN_WITHIN_T_MAX,
    Outcome.FEASIBLE,
    Outcome.NO_RETURN_WITHIN_T_MAX,
)


def _record(k: int, point: tuple[float, float, float], outcome: Outcome) -> FreeReturnRecord:
    feasible = outcome is Outcome.FEASIBLE
    nan = math.nan
    return FreeReturnRecord(
        phase_deg=point[0],
        delta_v_km_s=point[1],
        impulse_angle_deg=point[2],
        outcome=outcome,
        termination=Termination.EARTH_RETURN_SHELL if feasible else Termination.HORIZON,
        t_final_days=6.0 + k if feasible else 10.0,
        perilune_altitude_km=500.0 + 100.0 * k,
        lunar_encounter_time_days=3.0 + 0.01 * k,
        lunar_passage_sense=-1 if k % 2 else 1,
        moon_periapses_in_band=1,
        earth_periapses_after_encounter=0,
        return_time_days=6.0 + k if feasible else nan,
        return_speed_km_s=11.0 if feasible else nan,
        return_flight_path_angle_deg=-6.5 if feasible else nan,
        return_vacuum_perigee_altitude_km=38.0 if feasible else nan,
        vacuum_perigee_altitude_km=38.0 if feasible else nan,
        perigee_source="return_shell" if feasible else "none",
        jacobi_constant=1.9,
        jacobi_max_abs_drift_steps=1e-10,
        nfev=4000 + k,
    )


def _records() -> list[FreeReturnRecord]:
    return [_record(k, p, o) for k, (p, o) in enumerate(zip(DESIGN.points(), PATTERN, strict=True))]


def _write_map(runs_dir: Path, run_id: str = "gravity-map-synthetic") -> Path:
    manifest, table = build_map_documents(
        run_id,
        DESIGN,
        _records(),
        definition=DEFAULT_DEFINITION,
        workers=1,
        wall_time_s=1.0,
        git_commit=FAKE_COMMIT,
        git_dirty=False,
        created_at=datetime(2026, 10, 18, tzinfo=UTC),
    )
    write_run_directory(runs_dir / run_id, manifest, [("sample_table", "samples.json", table)])
    return runs_dir / run_id


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_codes_round_trip_and_reject_wrong_sizes() -> None:
    raw = bytes([0, 1, 2, 7, 255, 3])
    assert decode_codes(encode_codes(raw), 6) == raw
    with pytest.raises(ValueError, match="expected 7 codes"):
        decode_codes(encode_codes(raw), 7)
    with pytest.raises(ValueError, match="base64"):
        decode_codes("not base64!", 6)


def test_decoded_outcomes_equal_the_source_outcomes() -> None:
    table = json.loads(dumps_table(records_table("src", _records())))
    grid = outcome_grid("web", DESIGN, table)
    assert decode_outcomes(grid) == [o.value for o in PATTERN]
    assert grid["outcomes"] == [o.value for o in OUTCOMES]
    raw = base64.b64decode(grid["codes"])
    assert len(raw) == DESIGN.n_phase * DESIGN.n_delta_v
    # Feasible cells: indices in phase-major order and the source values verbatim.
    # PATTERN has feasible cells at k = 1, 5, 10, i.e. (i_phase, i_delta_v) = k divmod 4.
    assert [row[:2] for row in grid["feasible"]["rows"]] == [[0, 1], [1, 1], [2, 2]]
    names = [c["name"] for c in grid["feasible"]["columns"]]
    for row, k in zip(grid["feasible"]["rows"], (1, 5, 10), strict=True):
        values = dict(zip(names, row, strict=True))
        assert values["perilune_altitude_km"] == 500.0 + 100.0 * k
        assert values["lunar_encounter_time_days"] == 3.0 + 0.01 * k
        assert values["return_time_days"] == 6.0 + k
        assert values["lunar_passage_sense"] == (-1 if k % 2 else 1)


def test_grid_refuses_rows_that_are_not_the_design_points() -> None:
    table = json.loads(dumps_table(records_table("src", _records())))
    swapped = dict(table, rows=[table["rows"][1], table["rows"][0], *table["rows"][2:]])
    with pytest.raises(ValueError, match="expected grid point"):
        outcome_grid("web", DESIGN, swapped)
    with pytest.raises(ValueError, match="rows"):
        outcome_grid("web", DESIGN, dict(table, rows=table["rows"][:-1]))


def test_web_map_run_is_derived_with_provenance_and_never_overwritten(tmp_path: Path) -> None:
    source = _write_map(tmp_path)
    source_manifest = _load(source / "manifest.json")
    manifest = run_outcome_grid(
        tmp_path, source.name, "gravity-map-synthetic-web", git_commit=FAKE_COMMIT, git_dirty=True
    )
    out = tmp_path / "gravity-map-synthetic-web"
    assert _load(out / "manifest.json") == manifest
    (artifact,) = manifest["artifacts"]
    assert artifact["kind"] == "outcome_grid" and artifact["path"] == OUTCOME_GRID_FILE
    data = (out / OUTCOME_GRID_FILE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    derived = manifest["derived_from"]
    assert derived["run_id"] == source.name
    assert derived["artifact_sha256"] == source_manifest["artifacts"][0]["sha256"]
    assert derived["git_commit"] == FAKE_COMMIT and derived["git_dirty"] is False
    assert manifest["status"] == "computed_unvalidated"
    assert manifest["model"] == source_manifest["model"]
    assert manifest["solver"] == source_manifest["solver"]
    assert "wall_time_hours" not in manifest["metrics"]
    assert manifest["metrics"]["outcome_feasible_count"]["value"] == 3
    assert any("git_dirty" in s for s in manifest["limitations"])
    grid = json.loads(data)
    assert grid["run_id"] == "gravity-map-synthetic-web"
    assert decode_outcomes(grid) == [o.value for o in PATTERN]
    with pytest.raises(FileExistsError):
        run_outcome_grid(
            tmp_path,
            source.name,
            "gravity-map-synthetic-web",
            git_commit=FAKE_COMMIT,
            git_dirty=True,
        )


def test_web_map_refuses_a_source_whose_checksum_does_not_match(tmp_path: Path) -> None:
    source = _write_map(tmp_path)
    samples = source / "samples.json"
    samples.write_bytes(
        samples.read_bytes().replace(b"no_return_within_t_max", b"moon_collision", 1)
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_outcome_grid(tmp_path, source.name, "web", git_commit=FAKE_COMMIT, git_dirty=False)
    assert not (tmp_path / "web").exists()


def test_web_map_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write_map(tmp_path)
    code = main(["--runs-dir", str(tmp_path), "web-map", "--map-run", source.name, "--run-id", "w"])
    assert code == 0
    assert (tmp_path / "w" / OUTCOME_GRID_FILE).exists()
    assert "outcome_grid.json" in capsys.readouterr().out
