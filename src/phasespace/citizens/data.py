"""Parse the Koster et al. (2025) CSV into per-game arrays, with held-out data discipline.

Commands::

    uv run python -m phasespace.citizens.data download   # data/raw/, verified, never twice
    uv run python -m phasespace.citizens.data parse      # data/processed/*.npz + sidecars

The CSV has one row per game round. A game is identified by (`mech_name_by_player`,
`launch_id`) (the notebook groups by `launch_id` inside one `mech_name_by_player`
subset); `mech_name_by_player` is "<mechanism> <cohort>" with cohort "Exp 1".."Exp 4"
(human groups) or "BC 1"/"BC 2" (the paper's simulated behavioural-clone games).

**Held-out discipline.** Human games (cohorts Exp 1-4) played under the Interpolating
baseline or the RL agents M1/M2 (including the Exp 4 "Game k" variants) are the
validation targets of the study. The parser writes them to a separate file
(`heldout.npz`) that is read only by

- `load_heldout`, which refuses unless the repository's validation protocol
  (`docs/studies/model-citizens/protocol-validation.md`, resolved from the repository
  root, never from a caller-supplied path) declares ``Status: frozen`` in its header,
  is tracked by git with no uncommitted changes, and the caller passes
  ``confirm_frozen=True``; it returns the protocol's SHA-256 with the games, so
  held-out-derived manifests can record which protocol text they ran under;
- `heldout_identity_residuals`, which returns only residual statistics and pass/fail
  counts of the arithmetic identity checks (pool transition, offer reproduction), and
- `heldout_game_counts`, which returns only game counts.

All three go through the private reader `_read_games`, which refuses a file holding
held-out human games unless it receives this module's private access token.

`load_training` returns every other game: human Equal/Mixed/Proportional games and all
BC games.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.citizens import validation
from phasespace.citizens.game import N_PLAYERS
from phasespace.citizens.records import (
    BC_COHORTS,
    HELDOUT_MECHANISMS,
    HUMAN_COHORTS,
    MECHANISMS,
    Games,
    field_names,
)
from phasespace.citizens.source import (
    SOURCE_FILENAME,
    SourceFile,
    download,
    file_digests,
    sidecar_path,
    verify_source,
)

PARSER_VERSION: Final = "citizens-parser-v1"
DEFAULT_RAW_DIR: Final = Path("data/raw")
DEFAULT_PROCESSED_DIR: Final = Path("data/processed")
TRAINING_FILE: Final = "training.npz"
HELDOUT_FILE: Final = "heldout.npz"
VALIDATION_PROTOCOL_PATH: Final = Path("docs/studies/model-citizens/protocol-validation.md")
MAX_ROUNDS_PADDED: Final = 64
"""Round axis length of the arrays (Exp 4 games run 25-49 rounds; Exps 1-3 have 40)."""
HELDOUT_POPULATION: Final = "heldout_human"

_EXP4_GAME: Final = re.compile(r"^(?P<base>.+) Game (?P<k>[1-3])$")
_HEADER_STATUS: Final = (
    re.compile(r"^\s*(?:\*\*)?Status(?::\*\*|\*\*:|:)\s*\**`?(?P<status>[A-Za-z_-]+)`?"),
    re.compile(r"^\|\s*Status\s*\|\s*\**`?(?P<status>[A-Za-z_-]+)`?"),
)
"""Header status lines: ``Status: frozen``, ``**Status:** frozen`` or a ``| Status |`` row."""
FROZEN: Final = "frozen"
_BOOL: Final = {"True": 1, "False": 0, "": -1}
_ZIP_EPOCH: Final = (1980, 1, 1, 0, 0, 0)


class ProtocolNotFrozenError(RuntimeError):
    """Held-out human games may be read only after the validation protocol is frozen."""


class DataFormatError(ValueError):
    """The CSV violates an assumption the parser relies on."""


class _HeldoutAccess:
    """Type of the private token that lets `_read_games` return held-out human games."""


_HELDOUT_ACCESS: Final = _HeldoutAccess()
"""Passed only by `load_heldout` (after the freeze checks) and the summary-only readers."""


# --- labels ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameLabel:
    """The parsed `mech_name_by_player` of a game."""

    mechanism: str  # base family, e.g. "Proportional Baseline" (no " Game k" suffix)
    cohort: str  # "Exp 1".."Exp 4", "BC 1", "BC 2"
    exp4_game: int  # 1..3 for the three consecutive Exp 4 games, 0 otherwise

    @property
    def is_human(self) -> bool:
        return self.cohort in HUMAN_COHORTS

    @property
    def is_heldout(self) -> bool:
        return self.is_human and self.mechanism in HELDOUT_MECHANISMS


def parse_label(mech_name_by_player: str) -> GameLabel:
    """Split "<mechanism>[ Game k] <cohort>" into its parts."""
    parts = mech_name_by_player.rsplit(" ", 2)
    if len(parts) != 3:
        raise DataFormatError(f"unexpected mech_name_by_player {mech_name_by_player!r}")
    name, cohort = parts[0], f"{parts[1]} {parts[2]}"
    if cohort not in HUMAN_COHORTS + BC_COHORTS:
        raise DataFormatError(f"unknown cohort in {mech_name_by_player!r}")
    exp4_game = 0
    match = _EXP4_GAME.match(name)
    if match:
        name, exp4_game = match.group("base"), int(match.group("k"))
        if cohort != "Exp 4":
            raise DataFormatError(f"'Game k' suffix outside Exp 4: {mech_name_by_player!r}")
    elif cohort == "Exp 4":
        raise DataFormatError(f"Exp 4 game without 'Game k' suffix: {mech_name_by_player!r}")
    if name not in MECHANISMS:
        raise DataFormatError(f"unknown mechanism in {mech_name_by_player!r}")
    return GameLabel(mechanism=name, cohort=cohort, exp4_game=exp4_game)


def is_heldout_label(mech_name_by_player: str) -> bool:
    return parse_label(mech_name_by_player).is_heldout


# --- CSV parsing ----------------------------------------------------------------------


def parse_array(text: str, n: int = N_PLAYERS) -> list[float]:
    """Parse "[1. 2. 3. 4.]", "[1.0, 2.0, 3.0, 4.0]" or "[[1. 2. 3. 4.]]"; empty -> NaNs."""
    stripped = text.replace("[", " ").replace("]", " ").replace(",", " ").split()
    if not stripped:
        return [math.nan] * n
    if len(stripped) != n:
        raise DataFormatError(f"expected {n} values in {text!r}")
    return [float(v) for v in stripped]


def _float(text: str) -> float:
    return float(text) if text != "" else math.nan


def _int(text: str) -> int:
    return int(float(text)) if text != "" else -1


@dataclass(slots=True)
class _Row:
    label: str
    launch_id: str
    round_id: int
    row_index: int
    values: dict[str, Any]


def _iter_rows(path: Path) -> Iterator[_Row]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            p = range(N_PLAYERS)
            values: dict[str, Any] = {
                "pool": _float(row["mechanism_observation.pool"]),
                "next_pool": _float(row["next_environment_state.pool"]),
                "offers": [_float(row[f"offer_{i}"]) for i in p],
                "actions": [_float(row[f"player_action_{i}"]) for i in p],
                "rewards": [_float(row[f"player_reward_{i}"]) for i in p],
                "prev_reciprocations": parse_array(
                    row["mechanism_observation.prev_reciprocations"]
                ),
                "prev_offers": parse_array(row["mechanism_observation.prev_offers"]),
                "next_reciprocations": parse_array(row["next_environment_state.reciprocations"]),
                "mechanism_reward": _float(row["mechanism_reward"]),
                "offer_gini": _float(row["offer_gini"]),
                "players_kept_active": _int(row["players_kept_active"]),
                "uniform_offer_ratio": _float(row["mechanism_parameters_uniform_offer_ratio"]),
                "distribute_pool_ratio": _float(row["mechanism_parameters_distribute_pool_ratio"]),
                "four_usable_players": _BOOL[row["four_usable_players"]],
                "completed": _BOOL[row["completed"]],
                "first_x_games": _BOOL[row["first_x_games"]],
                "four_players_present": _BOOL[row["four_players_present"]],
                "game_has_no_pool": _BOOL[row["game_has_no_pool"]],
                "player_present": [_BOOL[row[f"player_{i}_present"]] for i in p],
                "player_has_no_choice": [_BOOL[row[f"player_{i}_has_no_choice"]] for i in p],
            }
            yield _Row(
                label=row["mech_name_by_player"],
                launch_id=row["launch_id"],
                round_id=int(row["round_id"]),
                row_index=index,
                values=values,
            )


_PER_ROUND_FLOAT: Final = (
    "pool",
    "next_pool",
    "mechanism_reward",
    "offer_gini",
    "uniform_offer_ratio",
    "distribute_pool_ratio",
)
_PER_PLAYER_FLOAT: Final = (
    "offers",
    "actions",
    "rewards",
    "prev_reciprocations",
    "prev_offers",
    "next_reciprocations",
)
_PER_ROUND_FLAG: Final = (
    "four_usable_players",
    "completed",
    "first_x_games",
    "four_players_present",
    "game_has_no_pool",
)
_PER_PLAYER_FLAG: Final = ("player_present", "player_has_no_choice")


@dataclass(frozen=True, slots=True)
class ParseAudit:
    """Structural facts established while parsing (no behavioural quantities)."""

    total_rows: int
    games: int
    duplicate_game_keys: int  # (label, launch_id) seen again after another game started
    non_contiguous_games: int  # round ids not exactly 0..n-1 in file order
    launch_ids_in_several_labels: int  # same launch_id under more than one label


def parse_csv(path: Path) -> tuple[Games, Games, ParseAudit]:
    """Parse the CSV into (training, heldout) games and a structural audit.

    Rows of one game must be consecutive in the file with round ids 0..n-1; a violation
    is counted in the audit and raises DataFormatError (nothing is guessed).
    """
    games: list[tuple[str, str, int, list[_Row]]] = []
    seen: set[tuple[str, str]] = set()
    labels_by_launch: dict[str, set[str]] = {}
    duplicates = 0
    non_contiguous = 0
    total = 0
    current: list[_Row] = []
    for row in _iter_rows(path):
        total += 1
        key = (row.label, row.launch_id)
        if current and (current[0].label, current[0].launch_id) == key:
            current.append(row)
            continue
        if current:
            games.append((current[0].label, current[0].launch_id, current[0].row_index, current))
        if key in seen:
            duplicates += 1
        seen.add(key)
        labels_by_launch.setdefault(row.launch_id, set()).add(row.label)
        current = [row]
    if current:
        games.append((current[0].label, current[0].launch_id, current[0].row_index, current))
    for _, _, _, rows in games:
        if [r.round_id for r in rows] != list(range(len(rows))):
            non_contiguous += 1
    audit = ParseAudit(
        total_rows=total,
        games=len(games),
        duplicate_game_keys=duplicates,
        non_contiguous_games=non_contiguous,
        launch_ids_in_several_labels=sum(1 for v in labels_by_launch.values() if len(v) > 1),
    )
    if duplicates or non_contiguous:
        raise DataFormatError(f"game identity or round contiguity violated: {audit}")
    longest = max(len(rows) for *_, rows in games)
    if longest > MAX_ROUNDS_PADDED:
        raise DataFormatError(f"a game has {longest} rounds > {MAX_ROUNDS_PADDED}")
    parsed = [(parse_label(label), launch, first, rows) for label, launch, first, rows in games]
    training = [g for g in parsed if not g[0].is_heldout]
    heldout = [g for g in parsed if g[0].is_heldout]
    return _assemble(training), _assemble(heldout), audit


def _assemble(games: Sequence[tuple[GameLabel, str, int, list[_Row]]]) -> Games:
    g, t, p = len(games), MAX_ROUNDS_PADDED, N_PLAYERS
    per_round = {k: np.full((g, t), np.nan) for k in _PER_ROUND_FLOAT}
    per_player = {k: np.full((g, t, p), np.nan) for k in _PER_PLAYER_FLOAT}
    flags = {k: np.full((g, t), -1, dtype=np.int8) for k in _PER_ROUND_FLAG}
    player_flags = {k: np.full((g, t, p), -1, dtype=np.int8) for k in _PER_PLAYER_FLAG}
    active = np.full((g, t), -1, dtype=np.int64)
    for gi, (_, _, _, rows) in enumerate(games):
        for r in rows:
            ti = r.round_id
            for k in _PER_ROUND_FLOAT:
                per_round[k][gi, ti] = r.values[k]
            for k in _PER_PLAYER_FLOAT:
                per_player[k][gi, ti] = r.values[k]
            for k in _PER_ROUND_FLAG:
                flags[k][gi, ti] = r.values[k]
            for k in _PER_PLAYER_FLAG:
                player_flags[k][gi, ti] = r.values[k]
            active[gi, ti] = r.values["players_kept_active"]
    return Games(
        mechanism=np.array([lab.mechanism for lab, *_ in games], dtype=np.str_),
        cohort=np.array([lab.cohort for lab, *_ in games], dtype=np.str_),
        exp4_game=np.array([lab.exp4_game for lab, *_ in games], dtype=np.int64),
        launch_id=np.array([launch for _, launch, _, _ in games], dtype=np.str_),
        n_rounds=np.array([len(rows) for *_, rows in games], dtype=np.int64),
        first_row=np.array([first for _, _, first, _ in games], dtype=np.int64),
        pool=per_round["pool"],
        next_pool=per_round["next_pool"],
        mechanism_reward=per_round["mechanism_reward"],
        offer_gini=per_round["offer_gini"],
        uniform_offer_ratio=per_round["uniform_offer_ratio"],
        distribute_pool_ratio=per_round["distribute_pool_ratio"],
        players_kept_active=active,
        offers=per_player["offers"],
        actions=per_player["actions"],
        rewards=per_player["rewards"],
        prev_reciprocations=per_player["prev_reciprocations"],
        prev_offers=per_player["prev_offers"],
        next_reciprocations=per_player["next_reciprocations"],
        four_usable_players=flags["four_usable_players"],
        completed=flags["completed"],
        first_x_games=flags["first_x_games"],
        four_players_present=flags["four_players_present"],
        game_has_no_pool=flags["game_has_no_pool"],
        player_present=player_flags["player_present"],
        player_has_no_choice=player_flags["player_has_no_choice"],
    )


# --- storage --------------------------------------------------------------------------


def save_games(games: Games, path: Path, sidecar: dict[str, Any]) -> None:
    """Write `games` to a new .npz (never overwriting) plus a JSON sidecar."""
    if path.exists():
        raise FileExistsError(f"{path} exists; processed files are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh, zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as archive:
        # An .npz written entry by entry with a fixed timestamp, so that parsing the same
        # CSV with the same parser version gives byte-identical files (stable SHA-256).
        for name in field_names():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, getattr(games, name), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())
    payload = {**sidecar, "file_sha256": file_digests(path)[2], "games": games.n_games}
    with path.with_suffix(".json").open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")


def processed_path(processed_dir: Path, name: str) -> Path:
    """Processed files live in a parser-versioned subdirectory and are never rewritten."""
    return processed_dir / PARSER_VERSION / name


def processed_sidecar(processed_dir: Path, name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(
        processed_path(processed_dir, name).with_suffix(".json").read_text(encoding="utf-8")
    )
    return payload


def _contains_heldout(games: Games) -> bool:
    return bool(
        (np.isin(games.cohort, HUMAN_COHORTS) & np.isin(games.mechanism, HELDOUT_MECHANISMS)).any()
    )


def _read_games(path: Path, *, access: _HeldoutAccess | None = None) -> Games:
    """Read a processed file; held-out human games need the module's private token.

    Refuses before loading when the sidecar marks the file as the held-out split, and
    after loading when the arrays contain any held-out human game (defence in depth).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `just citizens-data` to download and parse the dataset"
        )
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    guarded = access is _HELDOUT_ACCESS
    refusal = (
        f"{path} contains held-out human games; they are read only through load_heldout "
        "(after the validation protocol freeze) or the identity-summary functions"
    )
    if sidecar.get("split") == "heldout" and not guarded:
        raise DataFormatError(refusal)
    if file_digests(path)[2] != sidecar["file_sha256"]:
        raise DataFormatError(f"{path}: SHA-256 differs from its sidecar; file was modified")
    with np.load(path, allow_pickle=False) as npz:
        games = Games(**{name: npz[name] for name in field_names()})
    if _contains_heldout(games) and not guarded:
        raise DataFormatError(refusal)
    return games


def load_training(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Games:
    """Human Equal/Mixed/Proportional games (Exp 1, 2, 4) and every BC game."""
    return _read_games(processed_path(processed_dir, TRAINING_FILE))


# --- held-out guard -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    """The validation protocol text that authorised a held-out read."""

    path: str  # repository-relative POSIX path
    sha256: str  # of the file's exact bytes at the time of the read
    status: str  # always "frozen"


@dataclass(frozen=True, slots=True)
class HeldoutGames:
    """Held-out human games and the frozen protocol they were read under."""

    games: Games
    protocol: FrozenProtocol


def _repo_root() -> Path:
    """The repository root: src/phasespace/citizens/data.py -> three levels up from src."""
    return Path(__file__).resolve().parents[3]


def header_status(text: str) -> str:
    """The status declared in a protocol's header (the lines before the first '## ').

    Exactly one status line is required there; a status further down does not count.
    """
    found: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            break
        for pattern in _HEADER_STATUS:
            match = pattern.match(line)
            if match:
                found.append(match.group("status"))
                break
    if len(found) != 1:
        raise ProtocolNotFrozenError(
            f"the protocol header (before the first '## ' section) must hold exactly one "
            f"Status line; found {len(found)}"
        )
    return found[0]


def _git_tracked_and_clean(root: Path, relative: Path) -> None:
    """Refuse unless `relative` is tracked by git in `root` and has no uncommitted changes."""
    target = relative.as_posix()
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", target],
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", target],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ProtocolNotFrozenError(f"git is not available to verify {target}") from exc
    if tracked.returncode != 0:
        raise ProtocolNotFrozenError(f"{target} is not tracked by git; commit the frozen text")
    if status.returncode != 0 or status.stdout.strip():
        raise ProtocolNotFrozenError(
            f"{target} has uncommitted changes (or git status failed); held-out games are "
            "read only under a committed protocol"
        )


def require_frozen_protocol(root: Path, relative: Path, confirm_frozen: bool) -> FrozenProtocol:
    """All freeze conditions for the protocol at `root / relative`; returns its SHA-256."""
    if not confirm_frozen:
        raise ProtocolNotFrozenError("pass confirm_frozen=True (--confirm-frozen) to proceed")
    path = root / relative
    if not path.is_file():
        raise ProtocolNotFrozenError(
            f"{relative.as_posix()} does not exist; held-out human games are read only after "
            "the validation protocol is frozen"
        )
    content = path.read_bytes()
    status = header_status(content.decode("utf-8"))
    if status != FROZEN:
        raise ProtocolNotFrozenError(
            f"protocol {relative.as_posix()} has header status {status!r}; held-out human "
            "games are read only after the freeze"
        )
    _git_tracked_and_clean(root, relative)
    return FrozenProtocol(
        path=relative.as_posix(), sha256=hashlib.sha256(content).hexdigest(), status=status
    )


def load_heldout(
    *, confirm_frozen: bool, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> HeldoutGames:
    """Held-out human games, only under the repository's frozen, committed protocol.

    The protocol path is fixed (`VALIDATION_PROTOCOL_PATH` under the repository root);
    record `result.protocol.sha256` in every manifest derived from these games.
    """
    protocol = require_frozen_protocol(_repo_root(), VALIDATION_PROTOCOL_PATH, confirm_frozen)
    games = _read_games(processed_path(processed_dir, HELDOUT_FILE), access=_HELDOUT_ACCESS)
    return HeldoutGames(games=games, protocol=protocol)


def heldout_identity_residuals(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> list[validation.IdentityResult]:
    """Arithmetic identity checks on the held-out human games, residual summaries only.

    The one sanctioned pre-freeze use of held-out rows (C1): the pool-transition identity
    over every row (all held-out games together and per mechanism and cohort) and the
    offer reproduction of the Interpolating rows. Only `IdentityResult` records (entries
    checked, residual maxima, pass/fail counts) leave this function; no behavioural
    quantity of a held-out game is computed or returned.
    """
    games = _read_games(processed_path(processed_dir, HELDOUT_FILE), access=_HELDOUT_ACCESS)
    results = [validation.pool_transition(games, HELDOUT_POPULATION)]
    for mechanism, cohort in sorted(
        set(zip(games.mechanism.tolist(), games.cohort.tolist(), strict=True))
    ):
        group = games.where(mechanism=mechanism, cohort=cohort)
        population = f"{HELDOUT_POPULATION}:{mechanism}:{cohort}"
        results.append(validation.pool_transition(group, population))
    results.extend(validation.offer_reproduction(games, HELDOUT_POPULATION, diagnostics=False))
    return results


def heldout_game_counts(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> dict[tuple[str, str, int], int]:
    """Number of held-out human games per (mechanism, cohort, Exp 4 game index)."""
    games = _read_games(processed_path(processed_dir, HELDOUT_FILE), access=_HELDOUT_ACCESS)
    counts: dict[tuple[str, str, int], int] = {}
    for key in zip(
        games.mechanism.tolist(), games.cohort.tolist(), games.exp4_game.tolist(), strict=True
    ):
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# --- command line ---------------------------------------------------------------------


def parse_command(raw_dir: Path, processed_dir: Path) -> ParseAudit:
    """Verify the raw CSV, parse it, and write training/heldout .npz files (once)."""
    source: SourceFile = verify_source(raw_dir / SOURCE_FILENAME)
    training, heldout, audit = parse_csv(source.path)
    sidecar: dict[str, Any] = {
        "parser_version": PARSER_VERSION,
        "source_file": source.path.name,
        "source_sha256": source.sha256,
        "source_md5_base64": source.md5_base64,
        "source_size_bytes": source.size_bytes,
        "source_sidecar": sidecar_path(source.path).name,
        "audit": {
            "total_rows": audit.total_rows,
            "games": audit.games,
            "duplicate_game_keys": audit.duplicate_game_keys,
            "non_contiguous_games": audit.non_contiguous_games,
            "launch_ids_in_several_labels": audit.launch_ids_in_several_labels,
        },
    }
    save_games(
        training, processed_path(processed_dir, TRAINING_FILE), {**sidecar, "split": "training"}
    )
    save_games(
        heldout, processed_path(processed_dir, HELDOUT_FILE), {**sidecar, "split": "heldout"}
    )
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.data")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download and verify the CSV (once)")
    sub.add_parser("parse", help="parse the verified CSV into data/processed (once)")
    args = parser.parse_args(argv)
    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    if args.command == "download":
        source = download(raw_dir)
        print(f"{source.path}  {source.size_bytes} bytes  sha256 {source.sha256}")
        return 0
    if processed_path(processed_dir, TRAINING_FILE).exists():
        print(f"{processed_path(processed_dir, '')} already holds the parsed dataset; kept")
        return 0
    audit = parse_command(raw_dir, processed_dir)
    print(f"parsed {audit.total_rows} rows into {audit.games} games -> {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
