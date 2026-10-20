"""C1 checks on the published Koster et al. (2025) CSV (skipped when data/raw is absent).

References are independent of the implementation: the published group counts and
z-values of the paper, values printed in the CSV, and the CSV's own recorded pools and
offers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from phasespace.citizens import data, export, game, validation
from phasespace.citizens.records import Games

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_CSV = RAW_DIR / "sustainable_behavior.csv"

pytestmark = pytest.mark.skipif(
    not RAW_CSV.exists(),
    reason="dataset not downloaded (run `just citizens-data` to enable the C1 data checks)",
)


def _repo_protocol_declares_frozen() -> bool:
    """Whether the project's own validation protocol header says `frozen` (test gating)."""
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    if not path.is_file():
        return False
    try:
        return data.header_status(path.read_text(encoding="utf-8")) == data.FROZEN
    except data.ProtocolNotFrozenError:
        return False


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("citizens-processed")
    data.parse_command(RAW_DIR, out)
    return out


@pytest.fixture(scope="module")
def training(processed: Path) -> Games:
    return data.load_training(processed)


def test_game_counts_match_the_published_design(processed: Path, training: Games) -> None:
    human = training.select(training.is_human)
    assert human.n_games == 280
    assert training.n_games - human.n_games == 3584
    for (mechanism, cohort, k), n in export.DESIGN_GAMES.items():
        if mechanism in ("Equal Baseline", "Mixed Baseline", "Proportional Baseline"):
            assert training.where(mechanism=mechanism, cohort=cohort, exp4_game=k).n_games == n
    heldout = data.heldout_game_counts(processed)
    expected = {
        key: n
        for key, n in export.DESIGN_GAMES.items()
        if key[0] not in ("Equal Baseline", "Mixed Baseline", "Proportional Baseline")
    }
    assert heldout == expected
    assert set(np.unique(training.n_rounds[training.cohort != "Exp 4"]).tolist()) == {40}


@pytest.mark.skipif(
    _repo_protocol_declares_frozen(),
    reason="the validation protocol is frozen; this checks the pre-freeze repository state",
)
def test_heldout_games_need_a_frozen_protocol(processed: Path) -> None:
    with pytest.raises(data.ProtocolNotFrozenError):
        data.load_heldout(confirm_frozen=True, processed_dir=processed)
    with pytest.raises(data.DataFormatError, match="held-out"):
        data._read_games(data.processed_path(processed, data.HELDOUT_FILE))  # pyright: ignore[reportPrivateUsage]


def test_notebook_filters_are_already_applied(training: Games) -> None:
    flags = validation.human_flags(training.select(training.is_human))
    assert flags["rows"] == 9832
    assert all(v == 0 for k, v in flags.items() if k != "rows")


def test_recorded_examples(training: Games) -> None:
    # CSV row 0: Proportional BC 1, launch 0, round 0 -> next row's pool 160.18524.
    first = training.select(training.first_row == 0)
    assert (first.mechanism[0], first.cohort[0]) == ("Proportional Baseline", "BC 1")
    np.testing.assert_allclose(first.offers[0, 0], [49.9997] * 4, atol=4e-6)
    predicted = game.next_pool(first.pool[0, 0], first.offers[0, 0], first.actions[0, 0])
    assert float(predicted) == pytest.approx(160.18524, abs=1e-4)
    assert first.pool[0, 1] == pytest.approx(160.18524, abs=1e-5)
    # Mixed Exp 1, launch 98522310, round 0: 200 - 199.9988 + 1.4 * 136 = 190.4012.
    mixed = training.select(training.launch_id == "98522310")
    assert float(mixed.actions[0, 0].sum()) == 136.0
    assert float(mixed.next_pool[0, 0]) == pytest.approx(190.4012, abs=1e-4)
    assert float(game.next_pool(200.0, mixed.offers[0, 0], mixed.actions[0, 0])) == (
        pytest.approx(190.4012, abs=1e-4)
    )


def test_pool_transition_every_training_row(training: Games) -> None:
    for population, games in (
        ("training_human", training.select(training.is_human)),
        ("bc", training.select(~training.is_human)),
    ):
        result = validation.pool_transition(games, population)
        assert result.passed, result
    bc = validation.pool_transition(training.select(~training.is_human), "bc")
    assert bc.details["games_legacy_rule"] == 0


def test_legacy_remainder_dropped_games(training: Games) -> None:
    # The frozen list holds 7 early Exp 1 Equal/Proportional training games; each needs
    # the legacy rule (the full rule fails in some row), and every row of each follows it
    # (checked by test_pool_transition_every_training_row, which assigns rules per game).
    assert len(validation.LEGACY_REMAINDER_DROPPED) == 7
    legacy = training.select(validation.legacy_games(training))
    assert legacy.n_games == 7
    assert set(legacy.cohort.tolist()) == {"Exp 1"}
    assert set(legacy.mechanism.tolist()) == {"Equal Baseline", "Proportional Baseline"}
    assert set(legacy.launch_id.tolist()) == validation.LEGACY_REMAINDER_DROPPED
    result = validation.pool_transition(training.select(training.is_human), "training_human")
    assert result.details["games_legacy_rule"] == 7
    assert result.details["legacy_games_full_rule_fails"] == 7


def test_offers_reproduced_for_every_baseline_row(training: Games) -> None:
    results = validation.offer_reproduction(training, "training")
    assert {r.name for r in results} == {
        "offers_equal",
        "offers_mixed",
        "offers_proportional",
        "offers_interpolating",
    }
    for r in results:
        assert r.passed, r
    equal = next(r for r in results if r.name == "offers_equal")
    assert equal.details["exact_entries"] == equal.entries


def test_withheld_amount_is_pinned_to_a_thousandth(training: Games) -> None:
    # As for k: only 0.001 coins withheld reproduces the recorded offers; nearby amounts
    # lose most exact entries and leave entries outside the tolerance.
    table = validation.withheld_alternatives(training)
    deployed = table["0.001"]
    assert deployed["outside_tolerance"] == 0
    assert deployed["exact"] > 0.95 * deployed["entries"]
    for withheld in ("0", "0.0005", "0.0015", "0.002"):
        alternative = table[withheld]
        assert alternative["entries"] == deployed["entries"]
        assert alternative["exact"] < 0.5 * deployed["exact"], withheld
        assert alternative["outside_tolerance"] > 0, withheld


def test_post_depletion_rows_in_training_games(training: Games) -> None:
    rows = validation.post_depletion(training.select(training.is_human))
    assert rows["depleted_games"] > 0 and rows["post_depletion_rows"] > 0
    assert rows["depleted_exps_1_2_games_not_40_rows"] == 0
    for key in (
        "rows_pool_not_floor",
        "rows_offer_not_0p0022",
        "rows_reciprocation_not_0",
        "rows_game_has_no_pool_not_true",
    ):
        assert rows[key] == 0, key


def test_interpolating_weight_is_k22_rounded_to_a_tenth(training: Games) -> None:
    laws = validation.interpolating_alternatives(training.select(~training.is_human))
    assert laws["k22_round1"] == 0
    assert min(v for k, v in laws.items() if k not in ("k22_round1", "entries")) > 0


def test_zero_previous_sum_only_at_the_floor(training: Games) -> None:
    rows = validation.zero_previous_sum(training.select(training.is_human))
    assert rows["rows"] > 0
    assert rows["rows_pool_not_at_floor"] == 0
    assert rows["rows_equal_split"] == rows["rows"]


def test_record_consistency(training: Games) -> None:
    for games in (training.select(training.is_human), training.select(~training.is_human)):
        for r in validation.record_consistency(games, "x"):
            assert r.passed, r


def test_published_exp1_z_values(training: Games) -> None:
    results = validation.published_z(training)
    assert [round(z.reproduced, 2) for z in results] == [4.58, 5.28, 5.89, 4.76]
    assert all(z.passed for z in results)


G1B_REASON = (
    "G1b (D33): 23 rows in 5 RL Agent (M1) Exp 1 games sit ~0.01 below the pool rule; "
    "root-caused only after the validation protocol freeze"
)
M1_EXP1 = f"{data.HELDOUT_POPULATION}:RL Agent (M1):Exp 1"


def test_heldout_offer_identity(processed: Path) -> None:
    results = data.heldout_identity_residuals(processed)
    offers = [r for r in results if r.name.startswith("offers_")]
    assert [r.name for r in offers] == ["offers_interpolating"]
    assert offers[0].passed
    assert set(offers[0].details) == {"exact_entries"}  # no behavioural diagnostics
    pool = [r for r in results if r.name == "pool_transition"]
    assert all(r.details["games_legacy_rule"] == 0 for r in pool)  # no legacy id is held out


@pytest.mark.xfail(strict=True, reason=G1B_REASON)
def test_heldout_pool_identity_strict(processed: Path) -> None:
    results = data.heldout_identity_residuals(processed)
    for r in results:
        if r.name == "pool_transition":
            assert r.passed, r.population


def test_heldout_pool_anomaly_stays_within_its_bound(processed: Path) -> None:
    # Regression bound of the deferred G1b anomaly; only failure counts and the residual
    # maximum are read.
    pool = [r for r in data.heldout_identity_residuals(processed) if r.name == "pool_transition"]
    failing = {r.population for r in pool if not r.passed}
    assert failing <= {data.HELDOUT_POPULATION, M1_EXP1}
    for r in pool:
        if r.population in (data.HELDOUT_POPULATION, M1_EXP1):
            assert r.n_fail <= 23
            assert r.details["games_fail"] <= 5
            assert r.max_abs_residual <= 0.0100 + validation.POOL_TOLERANCE


def test_export_run(processed: Path, tmp_path: Path) -> None:
    audit = export.compute_audit(processed)
    out = tmp_path / "runs" / "citizens-c1-test"
    manifest = export.write_run(
        out,
        audit,
        git_commit="0" * 40,
        git_dirty=True,
        created_at=datetime(2026, 10, 20, tzinfo=UTC),
    )
    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    checks = manifest["validation"]["checks"]
    assert all(c["passed"] for c in checks), [c["name"] for c in checks if not c["passed"]]
    assert not any(c["name"].startswith("pool_transition_heldout") for c in checks)
    assert sum(c["name"].startswith("z_") for c in checks) == 4
    gate = manifest["citizens"]["gate"]
    assert gate["name"] == "G1a"
    [anomaly] = gate["deferred"]
    assert anomaly["name"] == "heldout_pool_anomaly" and anomaly["within_bound"]
    assert manifest["status"] == "validated"
    assert manifest["schema_version"] == 1 and manifest["random_seed"] is None
    assert manifest["simulator"] is None and "solver" not in manifest
    assert manifest["model"]["units"] == "coin" and manifest["model"]["offer_rule"] == "deployed"
    identity = json.loads((out / "identity_checks.json").read_text(encoding="utf-8"))
    units = {c["name"]: c["unit"] for c in identity["columns"]}
    assert units["max_abs_residual"] == units["threshold"] == "coin"
    assert any("git_dirty" in line for line in manifest["limitations"])
    for artifact in manifest["artifacts"]:
        digest = hashlib.sha256((out / artifact["path"]).read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
    counts = json.loads((out / "game_counts.json").read_text(encoding="utf-8"))
    heldout_rows = [r for r in counts["rows"] if r[3] == "heldout"]
    assert len(heldout_rows) == 8 and all(r[5] is None for r in heldout_rows)
    lengths = json.loads((out / "round_lengths.json").read_text(encoding="utf-8"))
    for row in lengths["rows"]:
        assert not (row[1].startswith("Exp") and row[0] in data.HELDOUT_MECHANISMS)
    with pytest.raises(FileExistsError):
        export.write_run(
            out, audit, git_commit="0" * 40, git_dirty=True, created_at=datetime.now(UTC)
        )
