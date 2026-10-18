"""M4 configuration, freeze guard, catalogs, ground truth and run artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from m4_protocol import FROZEN_CONFIG_SHA256, FROZEN_GIT_COMMIT, protocol_file
from m4_toy import surface, toy_evaluate

from phasespace.gravity.free_return import Outcome, evaluate
from phasespace.gravity.m4 import cli
from phasespace.gravity.m4.catalog import (
    Recheck,
    build_catalog,
    checkpoint_catalogs,
    lsoda_sample,
    recheck,
    rechecked_catalog,
)
from phasespace.gravity.m4.config import (
    PROTOCOL,
    PROTOCOL_PATH,
    ProtocolConfig,
    ProtocolNotFrozenError,
    checkpoints_for,
    config_sha256,
    protocol_status,
    require_frozen,
)
from phasespace.gravity.m4.evaluation import (
    Evaluation,
    Evaluator,
    LogEntry,
    evaluate_unit,
    to_physical,
    to_unit,
)
from phasespace.gravity.m4.groundtruth import (
    compare_with_m3,
    compute_ground_truth,
    next_level_centres,
    refinement_box,
    saturated,
)
from phasespace.gravity.m4.methods import sobol_prefix
from phasespace.gravity.m4.runs import build_manifest, evaluation_from_row, evaluation_row

FAKE_COMMIT = "0" * 40


def _feasible(u0: float, u2: float, gamma: float) -> Evaluation:
    e = toy_evaluate((u0, surface(u0, u2), u2))
    assert e.feasible
    return replace(e, return_flight_path_angle_deg=gamma)


# --- configuration and the freeze guard -------------------------------------------------------


def test_real_protocol_is_frozen_with_the_recorded_config_hash() -> None:
    assert PROTOCOL_PATH.is_file()
    assert protocol_status() == "frozen"
    assert config_sha256() == FROZEN_CONFIG_SHA256
    header = PROTOCOL_PATH.read_text(encoding="utf-8").split("## 2.")[0]
    assert f"| Config hash at freeze | `{FROZEN_CONFIG_SHA256}`" in header
    assert f"| Git commit at freeze | `{FROZEN_GIT_COMMIT}`" in header
    require_frozen(True)
    with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
        require_frozen(False)


def test_the_guard_refuses_a_draft_and_requires_confirmation(tmp_path: Path) -> None:
    draft = protocol_file(tmp_path, "draft")
    assert protocol_status(draft) == "draft"
    for confirm in (True, False):
        with pytest.raises(ProtocolNotFrozenError, match="draft"):
            require_frozen(confirm, draft)
    frozen = protocol_file(tmp_path, "frozen")
    assert protocol_status(frozen) == "frozen"
    require_frozen(True, frozen)
    with pytest.raises(ProtocolNotFrozenError, match="confirm-frozen"):
        require_frozen(False, frozen)
    (tmp_path / "bad.md").write_text("no header\n")
    with pytest.raises(ValueError, match="Status"):
        protocol_status(tmp_path / "bad.md")


def test_config_hash_is_stable_and_sensitive() -> None:
    digest = config_sha256()
    assert len(digest) == 64 and digest == config_sha256()
    assert config_sha256(ProtocolConfig(run_budget=2**15)) != digest
    assert PROTOCOL.run_budget == 2**14
    assert PROTOCOL.checkpoints == checkpoints_for(2**14) == (2048, 4096, 8192, 16384)


GUARDED_ARGV: tuple[tuple[str, ...], ...] = (
    ("search", "--stage", "main", "--method", "MS", "--seed", "1001"),
    ("stage", "--stage", "pilot"),
    ("robustness", "--run-id", "x", "--errors", "H_E_nominal"),
    ("groundtruth",),
    ("ccstar",),
)


def test_guarded_commands_refuse_while_draft(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    draft = protocol_file(tmp_path, "draft")
    runs = tmp_path / "runs"
    for argv in GUARDED_ARGV:
        with pytest.raises(SystemExit):
            cli.main(["--runs-dir", str(runs), *argv, "--confirm-frozen"], protocol_path=draft)
        assert "draft" in capsys.readouterr().err
    assert not runs.exists()


def test_guarded_commands_require_confirmation_when_frozen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    for path in (protocol_file(tmp_path, "frozen"), PROTOCOL_PATH):
        for argv in GUARDED_ARGV:
            with pytest.raises(SystemExit):
                cli.main(["--runs-dir", str(runs), *argv], protocol_path=path)
            assert "confirm-frozen" in capsys.readouterr().err
    assert not runs.exists()


def test_config_command_prints_the_hash(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["config"]) == 0
    out = capsys.readouterr().out
    assert FROZEN_CONFIG_SHA256 in out and "status: frozen" in out
    assert cli.main(["config", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_budget"] == 2**14


def test_stage_tasks() -> None:
    tune = cli.stage_tasks("tune", {}, (9101, 9102))
    per_method = {
        m: sum(t[0] == m for t in tune) for m in ("D", "MS", "MBH", "DE", "CC", "T", "T2")
    }
    # D and CC run once per tuning seed (B calibration, pilot dispersion); 16 x 2 runs.
    assert per_method == {"D": 2, "MS": 4, "MBH": 6, "DE": 6, "CC": 2, "T": 6, "T2": 6}
    assert len(tune) == 32
    with pytest.raises(ValueError, match="select"):
        cli.stage_tasks("main", {"MS": 0}, (1001,))
    main = cli.stage_tasks("main", {"MS": 1, "MBH": 0, "DE": 2, "T": 1, "T2": 0}, (1001, 1002))
    assert len(main) == 14 and ("DE", 2, 1002) in main and ("D", 0, 1001) in main


# --- catalogs ---------------------------------------------------------------------------------


def test_catalog_picks_lowest_e_with_ties_by_index_and_margins() -> None:
    a = _feasible(0.5, 0.40, -7.0)  # e = 0.5
    b = _feasible(0.5, 0.401, -6.0)  # same cell, e = 0.5 (tie, later)
    c = _feasible(0.5, 0.402, -6.75)  # e = 0.25 (better, latest)
    infeasible = toy_evaluate((0.05, 0.5, 0.5))
    log = [infeasible, a, b, c]
    cat = build_catalog(log)
    (elite,) = cat.values()
    assert elite.index == 3 and elite.objective_deg == pytest.approx(0.25)
    assert elite.margin_deg == pytest.approx(0.25)
    early = build_catalog(log, limit=3)
    assert next(iter(early.values())).index == 1  # tie broken by the earlier index
    snaps = checkpoint_catalogs(log, (1, 4))
    assert snaps[1] == {} and len(snaps[4]) == 1


def test_rechecked_catalog_replaces_a_failing_elite() -> None:
    log = [_feasible(0.5, 0.4, g) for g in (-6.5, -6.6, -6.7, -6.8, -6.9, -7.0)]

    def check(i: int) -> Recheck:
        return Recheck(i, i >= 2, True, True, 0.0, 0.0, 0.0)

    cat, made = rechecked_catalog(log, check)
    (elite,) = cat.values()
    assert elite.index == 2 and [r.index for r in made] == [0, 1, 2]
    _, none = rechecked_catalog(log, lambda i: Recheck(i, False, True, True, 0.0, 0.0, 0.0))
    assert [r.index for r in none] == [0, 1, 2, 3]  # the elite plus 3 replacements


def test_c1_passes_the_tight_recheck() -> None:
    e = evaluate_unit(to_unit((229.14224099316039, 3.1567704135102614, 0.0)))
    rc = recheck(e)
    assert rc.passed and rc.perilune_diff_km < 0.010 and rc.gamma_diff_deg < 1e-3


def test_lsoda_sample() -> None:
    assert lsoda_sample(0) == []
    s = lsoda_sample(100)
    assert len(s) == 5 and s == sorted(set(s)) and s == lsoda_sample(100)
    assert len(lsoda_sample(30)) == 2


# --- ground truth -------------------------------------------------------------------------


def test_refinement_boxes() -> None:
    h = 2.0 ** (-20.0 / 3.0)
    centre = (0.5, 0.5, 0.5)
    for level in (1, 2, 3):
        pts = refinement_box(centre, level)
        assert len(pts) == 32
        half = h / 2 ** (level - 1)
        assert all(abs(p[i] - 0.5) <= half for p in pts for i in range(3))
    shifted = refinement_box((0.6, 0.5, 0.5), 1)
    base = refinement_box(centre, 1)
    assert [p[0] - 0.1 for p in shifted] == pytest.approx([p[0] for p in base])
    corner = refinement_box((0.0, 1.0, 0.5), 1)
    assert all(p[0] >= 0.0 and p[1] <= 1.0 for p in corner)
    with pytest.raises(ValueError, match="level"):
        refinement_box(centre, 4)


def test_ground_truth_levels_and_saturation() -> None:
    ev = Evaluator(10**6, toy_evaluate)
    points = sobol_prefix(20261018, 2048)
    cells, map_evals, refined = compute_ground_truth(ev.evaluate_batch, points)
    assert len(map_evals) == 2048
    assert cells.by_source["map"] >= cells.by_source["map_prefix"]
    centres = sum(e.feasible for e in map_evals)
    assert sum(e.tag == "gt_level1" for e in ev.log) <= 32 * centres
    assert all(e.feasible for e in refined)
    assert len(cells.union()) >= len(cells.union("map"))
    assert saturated(20, 22) and not saturated(20, 23)
    known = cells.union("map")
    new = next_level_centres(refined, known, 10)
    assert all(e not in known for e in new) or new == []


def test_k1_comparison_with_m3_records() -> None:
    # The M4 evaluation reproduces the M3 map code (free_return.evaluate) exactly.
    points = sobol_prefix(20261018, 3)
    m4 = [evaluate_unit(u) for u in points]
    m3 = [evaluate(*to_physical(u)) for u in points]
    assert compare_with_m3(m4, m3) == []
    shifted = replace(m3[0], phase_deg=m3[0].phase_deg + 1e-9)
    assert [m.field for m in compare_with_m3(m4[:1], [shifted])] == ["phase_deg"]
    other = Outcome.FEASIBLE if m3[1].outcome is not Outcome.FEASIBLE else Outcome.MOON_COLLISION
    wrong = replace(m3[1], outcome=other)
    assert [m.field for m in compare_with_m3(m4[1:2], [wrong])] == ["outcome"]
    with pytest.raises(ValueError, match="fewer"):
        compare_with_m3(m4[:1], m3)


# --- run artifacts --------------------------------------------------------------------------


def test_evaluation_rows_round_trip() -> None:
    for e in (_feasible(0.5, 0.4, -6.0), toy_evaluate((0.05, 0.5, 0.5))):
        entry = LogEntry(7, e, "local", 0.125)
        row = evaluation_row(entry)
        json.dumps(row, allow_nan=False)
        back = evaluation_from_row(json.loads(json.dumps(row)))
        assert evaluation_row(back) == row


def test_smoke_run_writes_a_manifest_and_never_overwrites(tmp_path: Path) -> None:
    manifest = cli.run_smoke(
        tmp_path,
        "smoke-a",
        budget=2,
        seed=9001,
        workers=1,
        probe_draws=0,
        git_commit=FAKE_COMMIT,
        git_dirty=True,
    )
    m4: dict[str, Any] = manifest["m4"]
    assert m4["stage"] == "smoke" and m4["protocol_status"] == protocol_status() == "frozen"
    assert m4["config_sha256"] == config_sha256() == FROZEN_CONFIG_SHA256
    assert [b["name"] for b in m4["methods"]] == ["D", "MS", "MBH", "DE", "CC", "T", "T2", "CCstar"]
    assert all(b["propagations"] <= 2 for b in m4["methods"])
    assert manifest["random_seed"] == 9001
    assert any(
        "not comparative" in s.lower() or "NOT comparative" in s for s in manifest["limitations"]
    )
    written = json.loads((tmp_path / "smoke-a" / "manifest.json").read_text())
    assert {a["path"] for a in written["artifacts"]} >= {"evaluations-d.json", "catalog-t2.json"}
    with pytest.raises(FileExistsError):
        cli.run_smoke(
            tmp_path,
            "smoke-a",
            budget=2,
            seed=9001,
            workers=1,
            probe_draws=0,
            git_commit=FAKE_COMMIT,
            git_dirty=True,
        )
    with pytest.raises(ValueError, match="budget"):
        cli.run_smoke(
            tmp_path,
            "smoke-b",
            budget=512,
            seed=9001,
            workers=1,
            probe_draws=0,
            git_commit=FAKE_COMMIT,
            git_dirty=True,
        )
    with pytest.raises(ValueError, match="protocol seed"):
        cli.run_smoke(
            tmp_path,
            "smoke-c",
            budget=2,
            seed=1001,
            workers=1,
            probe_draws=0,
            git_commit=FAKE_COMMIT,
            git_dirty=True,
        )


def test_manifests_record_the_protocol_status_they_were_made_under(tmp_path: Path) -> None:
    def manifest(path: Path) -> dict[str, Any]:
        return build_manifest(
            "status-check",
            stage="smoke",
            experiment_id="x",
            seed=None,
            methods=[],
            metrics={},
            limitations=[],
            git_commit=FAKE_COMMIT,
            git_dirty=False,
            created_at=datetime(2026, 10, 18, tzinfo=UTC),
            protocol_path=path,
        )

    def not_frozen_note(m: dict[str, Any]) -> bool:
        return any("not frozen" in s for s in m["limitations"])

    real = manifest(PROTOCOL_PATH)
    assert real["m4"]["protocol_status"] == "frozen" and not not_frozen_note(real)
    assert real["m4"]["config_sha256"] == FROZEN_CONFIG_SHA256
    draft = manifest(protocol_file(tmp_path, "draft"))
    assert draft["m4"]["protocol_status"] == "draft" and not_frozen_note(draft)
