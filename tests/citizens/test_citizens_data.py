"""Parser, storage, held-out guard and verified download on small synthetic inputs."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phasespace.citizens import data, source

PLAYERS = range(4)


def _row(label: str, launch: str, round_id: int, pool: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mech_name_by_player": label,
        "launch_id": launch,
        "round_id": round_id,
        "mechanism_observation.pool": pool,
        "next_environment_state.pool": "",
        "mechanism_observation.prev_reciprocations": "[0. 0. 0. 0.]",
        "mechanism_observation.prev_offers": "[49.9997 49.9997 49.9997 49.9997]",
        "next_environment_state.reciprocations": "[1.0, 2.0, 3.0, 4.0]",
        "mechanism_reward": 1.0,
        "offer_gini": 0.0,
        "players_kept_active": 4,
        "mechanism_parameters_uniform_offer_ratio": "",
        "mechanism_parameters_distribute_pool_ratio": "",
        "four_usable_players": "True",
        "completed": "True",
        "first_x_games": "",
        "four_players_present": "False",
        "game_has_no_pool": "False",
    }
    for i in PLAYERS:
        row[f"offer_{i}"] = 10.0 + i
        row[f"player_action_{i}"] = float(i)
        row[f"player_reward_{i}"] = 10.0
        row[f"player_{i}_present"] = "True"
        row[f"player_{i}_has_no_choice"] = "False"
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["", *rows[0].keys()])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"": index, **row})
    return path


def _synthetic(tmp_path: Path) -> Path:
    rows = [
        _row("Equal Baseline Exp 1", "11", 0, 200.0),
        _row("Equal Baseline Exp 1", "11", 1, 150.0),
        _row("RL Agent (M1) Exp 1", "12", 0, 200.0),
        _row("RL Agent (M1) Exp 1", "12", 1, 120.0),
        _row("Interpolating Baseline BC 1", "11", 0, 200.0),
        _row("RL Agent (M2) Game 2 Exp 4", "13", 0, 200.0),
        _row("Proportional Baseline Game 2 Exp 4", "13", 0, 200.0),
    ]
    return _write_csv(tmp_path / "synthetic.csv", rows)


def test_parse_label() -> None:
    label = data.parse_label("Proportional Baseline Game 3 Exp 4")
    assert (label.mechanism, label.cohort, label.exp4_game) == ("Proportional Baseline", "Exp 4", 3)
    assert not label.is_heldout
    assert data.parse_label("RL Agent (M2) Game 1 Exp 4").is_heldout
    assert data.parse_label("Interpolating Baseline Exp 3").is_heldout
    assert data.parse_label("RL Agent (M1) Exp 2").is_heldout
    assert not data.parse_label("Interpolating Baseline BC 2").is_heldout
    assert not data.parse_label("RL Agent (M1) BC 1").is_heldout
    assert not data.parse_label("Mixed Baseline Exp 1").is_heldout
    for bad in ("Mixed Baseline Exp 9", "Unknown Rule Exp 1", "Mixed Baseline Game 1 Exp 1"):
        with pytest.raises(data.DataFormatError):
            data.parse_label(bad)
    with pytest.raises(data.DataFormatError, match="without 'Game k'"):
        data.parse_label("RL Agent (M2) Exp 4")


def test_parse_array_formats() -> None:
    assert data.parse_array("[49.9997 49.9997 49.9997 49.9997]") == [49.9997] * 4
    assert data.parse_array("[27.9998, 0.0, 1.5, 2]") == [27.9998, 0.0, 1.5, 2.0]
    assert data.parse_array("[[ 0. 25. 25. 30.]]") == [0.0, 25.0, 25.0, 30.0]
    assert all(np.isnan(data.parse_array("")))
    with pytest.raises(data.DataFormatError):
        data.parse_array("[1. 2. 3.]")


def test_parse_csv_splits_heldout_games(tmp_path: Path) -> None:
    training, heldout, audit = data.parse_csv(_synthetic(tmp_path))
    assert (audit.total_rows, audit.games) == (7, 5)
    assert audit.launch_ids_in_several_labels == 2  # "11" and "13"
    assert sorted(training.mechanism.tolist()) == [
        "Equal Baseline",
        "Interpolating Baseline",
        "Proportional Baseline",
    ]
    assert sorted(heldout.mechanism.tolist()) == ["RL Agent (M1)", "RL Agent (M2)"]
    assert heldout.exp4_game.tolist() == [0, 2]
    equal = training.where(mechanism="Equal Baseline")
    assert equal.n_rounds.tolist() == [2]
    assert equal.pool[0, :3].tolist()[:2] == [200.0, 150.0] and np.isnan(equal.pool[0, 2])
    np.testing.assert_array_equal(equal.offers[0, 0], [10, 11, 12, 13])
    np.testing.assert_array_equal(equal.next_reciprocations[0, 1], [1, 2, 3, 4])
    assert equal.first_x_games[0, 0] == -1 and equal.four_usable_players[0, 0] == 1
    assert equal.four_players_present[0, 0] == 0 and equal.first_row.tolist() == [0]


def test_parse_csv_rejects_split_or_non_contiguous_games(tmp_path: Path) -> None:
    split = [
        _row("Equal Baseline Exp 1", "11", 0, 200.0),
        _row("Mixed Baseline Exp 1", "12", 0, 200.0),
        _row("Equal Baseline Exp 1", "11", 1, 150.0),
    ]
    with pytest.raises(data.DataFormatError, match="contiguity"):
        data.parse_csv(_write_csv(tmp_path / "split.csv", split))
    gap = [_row("Equal Baseline Exp 1", "11", 0, 200.0), _row("Equal Baseline Exp 1", "11", 2, 1)]
    with pytest.raises(data.DataFormatError, match="contiguity"):
        data.parse_csv(_write_csv(tmp_path / "gap.csv", gap))


def _store(tmp_path: Path) -> Path:
    training, heldout, _ = data.parse_csv(_synthetic(tmp_path))
    processed = tmp_path / "processed"
    data.save_games(training, data.processed_path(processed, data.TRAINING_FILE), {"split": "t"})
    data.save_games(heldout, data.processed_path(processed, data.HELDOUT_FILE), {"split": "h"})
    return processed


def test_storage_round_trip_is_byte_stable_and_never_overwrites(tmp_path: Path) -> None:
    processed = _store(tmp_path)
    loaded = data.load_training(processed)
    assert loaded.n_games == 3
    path = data.processed_path(processed, data.TRAINING_FILE)
    with pytest.raises(FileExistsError):
        data.save_games(loaded, path, {})
    again = tmp_path / "again" / "t.npz"
    data.save_games(loaded, again, {})
    assert (
        hashlib.sha256(again.read_bytes()).hexdigest()
        == data.processed_sidecar(processed, data.TRAINING_FILE)["file_sha256"]
    )


def test_tampered_processed_file_is_rejected(tmp_path: Path) -> None:
    processed = _store(tmp_path)
    path = data.processed_path(processed, data.TRAINING_FILE)
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(data.DataFormatError, match="SHA-256"):
        data.load_training(processed)


def test_training_loader_refuses_heldout_games(tmp_path: Path) -> None:
    _, heldout, _ = data.parse_csv(_synthetic(tmp_path))
    processed = tmp_path / "bad"
    data.save_games(heldout, data.processed_path(processed, data.TRAINING_FILE), {})
    with pytest.raises(data.DataFormatError, match="held-out"):
        data.load_training(processed)


def test_header_status_reads_only_the_header() -> None:
    assert data.header_status("# Protocol\n\nStatus: frozen\n\n## 1. Scope\n") == "frozen"
    assert data.header_status("# P\n**Status:** `draft` (v0)\n## 1\n") == "draft"
    assert data.header_status("# P\n| Field | Value |\n|---|---|\n| Status | **`frozen`** |\n")
    with pytest.raises(data.ProtocolNotFrozenError, match="found 0"):
        data.header_status("# P\n\n## 1. Scope\n\nStatus: frozen\n")  # not in the header
    with pytest.raises(data.ProtocolNotFrozenError, match="found 2"):
        data.header_status("# P\nStatus: draft\nStatus: frozen\n")


def test_repo_root_is_the_repository() -> None:
    root = data._repo_root()  # pyright: ignore[reportPrivateUsage]
    assert (root / "pyproject.toml").is_file()
    assert Path(__file__).resolve().is_relative_to(root)


def _repo_protocol_declares_frozen() -> bool:
    """Whether the project's own validation protocol header says `frozen` (test gating)."""
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    if not path.is_file():
        return False
    try:
        return data.header_status(path.read_text(encoding="utf-8")) == data.FROZEN
    except data.ProtocolNotFrozenError:
        return False


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.org",
            *args,
        ],
        check=True,
        capture_output=True,
    )


def _protocol_repo(tmp_path: Path, status: str) -> tuple[Path, Path]:
    """A throwaway git repository (under tmp_path, never the project) holding a protocol."""
    repo = tmp_path / "repo"
    relative = data.VALIDATION_PROTOCOL_PATH
    (repo / relative).parent.mkdir(parents=True)
    (repo / relative).write_text(
        f"# Validation protocol\n\nStatus: {status}\n\n## 1. Scope\n", encoding="utf-8"
    )
    _git(repo, "init", "-q")
    return repo, relative


def test_freeze_conditions_on_a_throwaway_repository(tmp_path: Path) -> None:
    repo, relative = _protocol_repo(tmp_path, "frozen")
    with pytest.raises(data.ProtocolNotFrozenError, match="confirm_frozen"):
        data.require_frozen_protocol(repo, relative, False)
    with pytest.raises(data.ProtocolNotFrozenError, match="not tracked"):
        data.require_frozen_protocol(repo, relative, True)
    _git(repo, "add", relative.as_posix())
    _git(repo, "commit", "-q", "-m", "freeze")
    protocol = data.require_frozen_protocol(repo, relative, True)
    assert protocol.sha256 == hashlib.sha256((repo / relative).read_bytes()).hexdigest()
    assert (protocol.path, protocol.status) == (relative.as_posix(), "frozen")
    with (repo / relative).open("a", encoding="utf-8") as fh:
        fh.write("edited after the freeze\n")
    with pytest.raises(data.ProtocolNotFrozenError, match="uncommitted"):
        data.require_frozen_protocol(repo, relative, True)
    draft_repo, _ = _protocol_repo(tmp_path / "draft", "draft")
    _git(draft_repo, "add", relative.as_posix())
    _git(draft_repo, "commit", "-q", "-m", "draft")
    with pytest.raises(data.ProtocolNotFrozenError, match="'draft'"):
        data.require_frozen_protocol(draft_repo, relative, True)
    with pytest.raises(data.ProtocolNotFrozenError, match="does not exist"):
        data.require_frozen_protocol(tmp_path / "empty", relative, True)


@pytest.mark.skipif(
    _repo_protocol_declares_frozen(),
    reason="the validation protocol is frozen; this checks the pre-freeze repository state",
)
def test_load_heldout_refuses_before_the_repository_protocol_is_frozen(tmp_path: Path) -> None:
    processed = _store(tmp_path)
    with pytest.raises(data.ProtocolNotFrozenError, match="confirm_frozen"):
        data.load_heldout(confirm_frozen=False, processed_dir=processed)
    # Missing, draft, untracked or edited: every pre-freeze state of the repository's own
    # protocol is refused.
    with pytest.raises(data.ProtocolNotFrozenError):
        data.load_heldout(confirm_frozen=True, processed_dir=processed)


def test_load_heldout_under_a_frozen_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = _store(tmp_path)
    repo, relative = _protocol_repo(tmp_path, "frozen")
    _git(repo, "add", relative.as_posix())
    _git(repo, "commit", "-q", "-m", "freeze")
    # TEST-ONLY REDIRECTION: point the repository resolver at the throwaway repository so
    # the success path runs without a frozen protocol in the project itself.
    monkeypatch.setattr(data, "_repo_root", lambda: repo)
    result = data.load_heldout(confirm_frozen=True, processed_dir=processed)
    assert sorted(result.games.mechanism.tolist()) == ["RL Agent (M1)", "RL Agent (M2)"]
    assert result.protocol.sha256 == hashlib.sha256((repo / relative).read_bytes()).hexdigest()


def test_private_reader_refuses_heldout_games_without_the_token(tmp_path: Path) -> None:
    processed = _store(tmp_path)
    heldout_path = data.processed_path(processed, data.HELDOUT_FILE)
    with pytest.raises(data.DataFormatError, match="held-out"):
        data._read_games(heldout_path)  # pyright: ignore[reportPrivateUsage]
    # Renaming the file and dropping the sidecar's split label does not help: the
    # arrays themselves are checked.
    renamed = tmp_path / "renamed.npz"
    renamed.write_bytes(heldout_path.read_bytes())
    renamed.with_suffix(".json").write_text(
        json.dumps({"file_sha256": hashlib.sha256(renamed.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    with pytest.raises(data.DataFormatError, match="held-out"):
        data._read_games(renamed)  # pyright: ignore[reportPrivateUsage]


def test_heldout_identity_residuals_returns_summaries_only(tmp_path: Path) -> None:
    processed = _store(tmp_path)
    results = data.heldout_identity_residuals(processed)
    assert results and all(isinstance(r, data.validation.IdentityResult) for r in results)
    # No Interpolating games here, so no offer reproduction; the C2 count-only checks
    # (protocol-validation A29, A30) carry indicator residuals only.
    assert {r.name for r in results} == {
        "pool_transition",
        "action_anomalies",
        "players_active_identity",
    }
    for r in results:
        if r.name != "pool_transition":
            assert r.max_abs_residual in (0.0, 1.0) and r.threshold == 0.0
    assert data.heldout_game_counts(processed) == {
        ("RL Agent (M1)", "Exp 1", 0): 1,
        ("RL Agent (M2)", "Exp 4", 2): 1,
    }


# --- verified download -----------------------------------------------------------------


def _fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    payload = b"a,b\n1,2\n"
    origin = tmp_path / "origin.csv"
    origin.write_bytes(payload)
    md5 = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode()
    monkeypatch.setattr(source, "SOURCE_SIZE_BYTES", len(payload))
    monkeypatch.setattr(source, "SOURCE_MD5_BASE64", md5)
    return origin


def test_download_verifies_and_never_downloads_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = _fake_source(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    got = source.download(raw, url=origin.as_uri())
    assert got.path == raw / source.SOURCE_FILENAME
    assert got.sha256 == hashlib.sha256(origin.read_bytes()).hexdigest()
    sidecar = json.loads(source.sidecar_path(got.path).read_text(encoding="utf-8"))
    assert sidecar["sha256"] == got.sha256 and sidecar["url"] == source.SOURCE_URL
    # A verified copy is kept: an unreachable URL is never contacted.
    again = source.download(raw, url=(tmp_path / "missing.csv").as_uri())
    assert again.sha256 == got.sha256
    assert sorted(p.name for p in raw.iterdir()) == [
        source.SOURCE_FILENAME,
        source.SOURCE_FILENAME + source.SIDECAR_SUFFIX,
    ]


def test_download_rejects_corrupt_files_without_touching_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = _fake_source(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    corrupt = raw / source.SOURCE_FILENAME
    corrupt.write_bytes(b"a,b\n1,3\n")
    with pytest.raises(source.SourceVerificationError, match="MD5"):
        source.download(raw, url=origin.as_uri())
    assert corrupt.read_bytes() == b"a,b\n1,3\n"
    other = tmp_path / "other"
    bad_origin = tmp_path / "bad.csv"
    bad_origin.write_bytes(b"short")
    with pytest.raises(source.SourceVerificationError, match="bytes"):
        source.download(other, url=bad_origin.as_uri())
    assert list(other.iterdir()) == []  # the partial download was removed
