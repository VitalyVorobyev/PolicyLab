"""Post-freeze stages: refusal before the freeze, and their pure parts by hand.

No held-out human row is read here: every command is tested only for refusal; the G2
rule, the B4 type-weight choice and the single-game adapter are checked against
hand-derived references.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phasespace.citizens import config, data, heldout
from phasespace.citizens.evaluate import Estimate, ModelResult, OutcomeRow, RegimeResult
from phasespace.citizens.llm import runner
from phasespace.citizens.records import Games
from phasespace.citizens.simulate import EXACT_MECHANISMS


def _protocol_is_draft() -> bool:
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    return data.header_status(path.read_text(encoding="utf-8")) != data.FROZEN


COMMANDS = (
    ["g1b"],
    ["evaluate", "--fitted-run", "runs/none"],
    ["b4", "--variant", "plain", "--mechanism", "Interpolating", "--run-id", "x-v1"],
)


@pytest.mark.parametrize("argv", COMMANDS)
def test_every_stage_refuses_without_confirmation(argv: list[str]) -> None:
    with pytest.raises(data.ProtocolNotFrozenError, match="confirm"):
        heldout.main(argv)


@pytest.mark.skipif(not _protocol_is_draft(), reason="protocol frozen: refusal no longer applies")
@pytest.mark.parametrize("argv", COMMANDS)
def test_every_stage_refuses_a_draft_protocol(argv: list[str]) -> None:
    with pytest.raises(data.ProtocolNotFrozenError, match="draft"):
        heldout.main([argv[0], "--confirm-frozen", *argv[1:]])


def test_a_dirty_tree_is_refused_before_any_data_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = data.FrozenProtocol(path="p", sha256="0" * 64, status=data.FROZEN)

    def frozen_guard(*args: object) -> data.FrozenProtocol:
        return frozen

    monkeypatch.setattr(heldout.data, "require_frozen_protocol", frozen_guard)
    monkeypatch.setattr(heldout, "git_source", lambda: ("abc", True))

    def no_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("data were read")

    monkeypatch.setattr(heldout.data, "load_heldout", no_read)
    monkeypatch.setattr(heldout.data, "load_training", no_read)
    for argv in COMMANDS:
        with pytest.raises(heldout.DirtyTreeError):
            heldout.main([argv[0], "--confirm-frozen", *argv[1:]])


def test_published_heldout_z_values_and_tolerances() -> None:
    # Koster et al. 2025, Results (transcribed independently of the module).
    expected = {
        ("Exp 1", "surplus", "RL Agent (M1)", "Proportional Baseline", None): 3.25,
        ("Exp 1", "surplus", "RL Agent (M1)", "Mixed Baseline", None): 4.38,
        ("Exp 1", "surplus", "RL Agent (M1)", "Equal Baseline", None): 6.31,
        ("Exp 1", "gini", "Proportional Baseline", "RL Agent (M1)", None): 2.78,
        ("Exp 2", "surplus", "RL Agent (M1)", "Proportional Baseline", None): 2.9,
        ("Exp 2", "surplus", "Interpolating Baseline", "Proportional Baseline", None): 2.52,
        ("Exp 2", "gini", "Proportional Baseline", "Interpolating Baseline", None): 4.17,
        ("Exp 2", "gini", "RL Agent (M1)", "Interpolating Baseline", None): 3.67,
        ("Exp 3", "surplus", "RL Agent (M2)", "Interpolating Baseline", None): 2.35,
        ("Exp 3", "gini", "RL Agent (M2)", "Interpolating Baseline", None): 5.64,
        ("Exp 4", "surplus", "RL Agent (M2)", "Proportional Baseline", 1): 2.22,
        ("Exp 4", "surplus", "RL Agent (M2)", "Proportional Baseline", 2): 4.99,
        ("Exp 4", "surplus", "RL Agent (M2)", "Proportional Baseline", 3): 6.08,
    }
    got = {
        (z.cohort, z.endpoint, z.higher, z.lower, z.exp4_game): z.published
        for z in heldout.PUBLISHED_HELDOUT_Z
    }
    assert got == expected
    tol = {z.published: z.tolerance for z in heldout.PUBLISHED_HELDOUT_Z}
    assert tol[2.9] == pytest.approx(0.05)
    assert tol[3.25] == pytest.approx(0.005)


C = "validated (closed-loop)"
ONE = "validated (one-step)"
N = "not validated"


def _labels(**regimes: dict[str, str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for regime, models in regimes.items():
        out[regime.replace("_", "-")] = {"B0": N, "B1": N, "B2": N, "B3": N, "B23": N, **models}
    return out


ALL = {"R-I": True, "R-M1": True, "R-M2": True}
NO_B0 = {"R-I": False, "R-M1": False, "R-M2": False}


def test_g2_two_classes_on_one_common_regime_pass() -> None:
    d = heldout.g2_decision(_labels(R_I={"B1": C, "B2": C}, R_M1={}, R_M2={}), ALL, NO_B0)
    assert d["g2_passed"] is True
    assert d["regimes_meeting_o4"] == ["R-I"]


def test_g2_needs_a_common_regime_and_a_non_gru_class() -> None:
    split = _labels(R_I={"B1": C}, R_M1={"B2": C}, R_M2={})
    assert heldout.g2_decision(split, ALL, NO_B0)["g2_passed"] is False  # O4
    gru_only = _labels(R_I={"B1": C}, R_M1={}, R_M2={})
    assert heldout.g2_decision(gru_only, ALL, NO_B0)["g2_passed"] is False


def test_g2_fails_when_the_rational_control_passes_anywhere() -> None:
    # B0's label is "not validated" (it fails the likelihood part), yet it passes (a) on
    # R-M1: the negative control is decided by (a) alone (A23), so G2 fails.
    labels = _labels(R_I={"B1": C, "B2": C}, R_M1={}, R_M2={})
    d = heldout.g2_decision(labels, ALL, {**NO_B0, "R-M1": True})
    assert d["g2_passed"] is False
    assert d["b0_passes"]["R-M1"] is True
    assert d["regime_counts"]["R-M1"] is False


def test_g2_one_step_counts_only_with_a_closed_loop_validation_elsewhere() -> None:
    # B3 is closed-loop validated on R-I, so its one-step label on R-M1 counts; B2 has
    # no closed-loop validation anywhere, so its one-step label does not (O3).
    labels = _labels(R_I={"B3": C}, R_M1={"B2": ONE, "B3": ONE}, R_M2={})
    d = heldout.g2_decision(labels, ALL, NO_B0)
    assert d["counted_classes"]["R-M1"] == ["B3"]
    assert d["g2_passed"] is False
    labels["R-I"]["B2"] = C
    assert heldout.g2_decision(labels, ALL, NO_B0)["counted_classes"]["R-M1"] == ["B2", "B3"]


def test_g2_ignores_regimes_with_an_unverified_human_reference() -> None:
    labels = _labels(R_I={"B1": C, "B2": C}, R_M1={}, R_M2={})
    d = heldout.g2_decision(labels, {**ALL, "R-I": False}, NO_B0)
    assert d["g2_passed"] is False


def test_g2_does_not_count_a_label_that_fails_must_beat_marginal() -> None:
    labels = _labels(R_I={"B1": C, "B2": heldout.NOT_BEATING_M0_SIM}, R_M1={}, R_M2={})
    assert heldout.g2_decision(labels, ALL, NO_B0)["g2_passed"] is False


# --- criterion labels: B0 control, must beat marginal (A22, A23) -----------------------


def _row(metric: str, in_range: bool, w1n: float) -> OutcomeRow:
    return OutcomeRow(metric, Estimate(0.0, 0.0, 0.0), 0.1, 0.0, 0.0, in_range, w1n, 0.0, w1n)


def _result(name: str, in_range: int, w1n: float, delta: Estimate | None) -> ModelResult:
    metrics = ("surplus", "gini", "depletion_round", "players_active")
    rows = tuple(_row(m, k < in_range, w1n) for k, m in enumerate(metrics))
    return ModelResult(name, rows, -2.0, delta, 1.0, 1000)


GOOD = Estimate(0.1, 0.02, 0.2)
BAD = Estimate(-0.1, -0.2, -0.05)


def test_regime_label_must_beat_marginal() -> None:
    model = _result("B2", 4, 0.30, GOOD)  # (a) and (b) hold; mean normalised W1 0.30
    assert heldout.regime_label(model, True) == C
    assert heldout.regime_label(model, True, m0_sim_passes_a=True, m0_sim_w1=0.31) == C
    # Strictly lower is required: equal or higher fails where M0-sim passes (a).
    for m0 in (0.30, 0.29):
        label = heldout.regime_label(model, True, m0_sim_passes_a=True, m0_sim_w1=m0)
        assert label == heldout.NOT_BEATING_M0_SIM
    # Where M0-sim fails (a) the W1 comparison does not apply.
    assert heldout.regime_label(model, True, m0_sim_passes_a=False, m0_sim_w1=0.01) == C
    assert heldout.regime_label(_result("B2", 2, 0.1, GOOD), True) == N  # (a) fails
    assert heldout.regime_label(_result("B2", 4, 0.1, BAD), True) == N  # (b) fails
    # One-step tier: only (b) counts; the rule is about closed-loop validation.
    assert heldout.regime_label(_result("B2", 0, 9.0, GOOD), False, m0_sim_passes_a=True) == ONE
    assert heldout.regime_label(_result("M0-sim", 4, 0.1, None), True) == "control"


def test_b0_control_uses_the_deciding_criterion() -> None:
    b0 = _result("B0", 3, 0.5, BAD)  # passes (a), fails (b)
    assert b0.validated is False
    assert heldout.b0_control_passes(b0, closed_loop=True) is True
    assert heldout.b0_control_passes(b0, closed_loop=False) is False
    assert heldout.b0_control_passes(_result("B0", 2, 0.5, GOOD), closed_loop=False) is True


def test_w1_comparison_records_m0_sim() -> None:
    m0 = _result("M0-sim", 3, 0.40, None)
    m0 = ModelResult("M0-sim", m0.outcomes, None, None, 1.0, 1000)
    regime = RegimeResult(
        "R-I", "Interpolating", 10, 10, 100, {"Exp 2": 10}, {},
        (m0, _result("B2", 4, 0.30, GOOD), _result("B3", 4, 0.50, GOOD)),
        w1_scale={"surplus": 1.0},
    )  # fmt: skip
    passes_a, m0_w1, block = heldout.w1_comparison(regime, closed_loop=True)
    assert passes_a is True and m0_w1 == pytest.approx(0.40)
    assert block["rule_applies"] is True
    assert block["models"]["B2"]["beats_m0_sim"] is True
    assert block["models"]["B3"]["beats_m0_sim"] is False
    assert heldout.w1_comparison(regime, closed_loop=False)[2]["rule_applies"] is False


def test_simplex_grid_and_multinomial() -> None:
    grid = heldout.simplex_grid(0.1)
    assert len(grid) == 66
    assert all(math.isclose(sum(w), 1.0) for w in grid)
    comps = runner.compositions()
    w = (0.2, 0.5, 0.3)
    assert sum(heldout.composition_probability(c, w) for c in comps) == pytest.approx(1.0)
    # Multinomial(1, 2, 1; 4, w) = 12 * 0.2 * 0.25 * 0.3 = 0.18.
    assert heldout.composition_probability((1, 2, 1), w) == pytest.approx(0.18)


def test_type_weights_recover_a_known_mixture() -> None:
    # Composition means linear in the type counts: mean_m(comp) = sum_k comp_k v_mk / 4,
    # so the predicted mean under w is sum_k w_k v_mk (the multinomial expectation).
    v: dict[str, tuple[float, float, float]] = {
        "surplus": (8.0, 5.0, 2.0),
        "gini": (0.4, 0.2, 0.1),
        "depletion_round": (10.0, 25.0, 40.0),
        "players_active": (0.0, 2.0, 4.0),
    }
    means = {
        c: {m: sum(n * x for n, x in zip(c, v[m], strict=True)) / 4 for m in v}
        for c in runner.compositions()
    }
    target = (0.2, 0.5, 0.3)
    h = {m: sum(t * x for t, x in zip(target, v[m], strict=True)) for m in v}
    s = dict.fromkeys(v, 1.0)
    w, loss = heldout.choose_type_weights(means, h, s)
    assert w == pytest.approx(target)
    assert loss == pytest.approx(0.0, abs=1e-12)


def test_type_weight_ties_go_to_the_weight_nearest_uniform() -> None:
    names: tuple[str, ...] = ("surplus", "gini", "depletion_round", "players_active")
    means: dict[tuple[int, int, int], dict[str, float]] = {
        c: dict.fromkeys(names, 1.0) for c in runner.compositions()
    }
    h: dict[str, float] = dict.fromkeys(names, 1.0)
    w, _ = heldout.choose_type_weights(means, h, dict.fromkeys(h, 1.0))
    assert w == pytest.approx((0.3, 0.3, 0.4))  # a grid point nearest (1/3, 1/3, 1/3)


class _Half:
    async def complete(self, prompt: str) -> Any:
        import re

        from phasespace.citizens.llm.backend import CallResult

        n = int(re.findall(r"from 0 to (\d+)", prompt)[-1])
        return CallResult("{}", "", 0, 0.0, False, f'{{"return": {n // 2}}}', False, 0.0)

    def describe(self) -> dict[str, Any]:
        return {"kind": "fake"}


def test_single_game_adapter_matches_the_runner_baseline() -> None:
    def play(mechanism: runner.GameMechanism) -> runner.GameOutcome:
        spec = runner.GameSpec(0, 12, (None, None, None, None))
        return asyncio.run(
            runner.play_game(spec, mechanism, _Half(), asyncio.Semaphore(4), lambda r: None)
        )

    adapted = play(heldout.SingleGameMechanism(EXACT_MECHANISMS["Proportional"]))
    reference = play(runner.PROPORTIONAL)
    assert adapted.to_json()["rounds"] == reference.to_json()["rounds"]
    interp = play(heldout.SingleGameMechanism(EXACT_MECHANISMS["Interpolating"]))
    assert interp.status == "complete"
    # Round 1 is the equal split of R - 0.001 = 199.999, rounded to 4 decimals in float32.
    np.testing.assert_allclose(interp.rounds[0].offers, [199.999 / 4] * 4, atol=1e-4)


GAME_COLUMNS = ("status", "types", "surplus", "gini", "depletion_round", "players_active_end")


def _write_run(run_dir: Path, files: dict[str, bytes], citizens: dict[str, Any]) -> None:
    """A minimal run directory whose manifest lists each file with its SHA-256."""
    run_dir.mkdir(parents=True)
    for name, payload in files.items():
        (run_dir / name).write_bytes(payload)
    manifest = {
        "run_id": run_dir.name,
        "status": "computed_unvalidated",
        "source": {"git_dirty": False},
        "artifacts": [
            {"kind": "sample_table", "path": n, "sha256": hashlib.sha256(b).hexdigest()}
            for n, b in files.items()
        ],
        "citizens": citizens,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def _games_json(rows: list[list[Any]]) -> bytes:
    return json.dumps({"columns": [{"name": n} for n in GAME_COLUMNS], "rows": rows}).encode()


def _calibration_rows(per_composition: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for a, b, c in runner.compositions():
        types = ["T1"] * a + ["T2"] * b + ["T3"] * c
        rows += [["complete", json.dumps(types), 1.0, 0.1, 40.0, 4.0]] * per_composition
    return rows


def test_b4_weights_refuse_an_incomplete_calibration(tmp_path: Path) -> None:
    rows = [["complete", json.dumps(["T1"] * 4), 1.0, 0.1, 10.0, 0.0]]
    _write_run(tmp_path / "a", {"games.json": _games_json(rows)}, {})
    with pytest.raises(ValueError, match="15 type compositions"):
        heldout.calibration_composition_means(tmp_path / "a")


def test_calibration_needs_four_games_per_composition_and_a_matching_checksum(
    tmp_path: Path,
) -> None:
    rows = _calibration_rows(4)
    _write_run(tmp_path / "ok", {"games.json": _games_json(rows)}, {})
    means = heldout.calibration_composition_means(tmp_path / "ok")
    assert len(means) == 15 and means[(4, 0, 0)]["players_active"] == 4.0
    # A discarded game does not count; a composition with 3 complete games is refused.
    short = [*rows[:-1], ["discarded", rows[-1][1], None, None, None, None]]
    _write_run(tmp_path / "short", {"games.json": _games_json(short)}, {})
    with pytest.raises(ValueError, match="exactly 4 complete games"):
        heldout.calibration_composition_means(tmp_path / "short")
    _write_run(tmp_path / "extra", {"games.json": _games_json([*rows, rows[0]])}, {})
    with pytest.raises(ValueError, match="exactly 4 complete games"):
        heldout.calibration_composition_means(tmp_path / "extra")
    # games.json changed after the run was written: the SHA-256 no longer matches.
    (tmp_path / "ok" / "games.json").write_bytes(_games_json(_calibration_rows(4)) + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        heldout.calibration_composition_means(tmp_path / "ok")


def test_calibration_games_must_match_the_frozen_b4_setup(tmp_path: Path) -> None:
    from phasespace.citizens.llm.backend import ClaudeCLI

    backend = {**ClaudeCLI().describe(), "cli_version": "2.1.304 (Claude Code)"}
    config = runner.RunConfig(
        run_id="c", label="calibration", variant="calibrated", mechanism="m",
        target_games=60, n_rounds=40, workers=8, max_retries=2, discard_limit=6,
        seed=7001, backend=backend,
    ).to_json()  # fmt: skip
    (tmp_path / "manifest.json").write_text(json.dumps({"citizens": {"config": config}}))
    heldout.check_calibration_identity(tmp_path, backend)
    other = {**backend, "argv": [*backend["argv"][:-1], "2"]}
    with pytest.raises(SystemExit, match=r"backend\.argv"):
        heldout.check_calibration_identity(tmp_path, other)
    config["rules_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps({"citizens": {"config": config}}))
    with pytest.raises(SystemExit, match="rules_sha256"):
        heldout.check_calibration_identity(tmp_path, backend)
    config["rules_sha256"] = runner.RunConfig(
        run_id="c", label="c", variant="calibrated", mechanism="m", target_games=0,
        n_rounds=0, workers=0, max_retries=0, discard_limit=0, seed=0, backend={},
    ).rules_sha256  # fmt: skip
    config["backend"] = {**backend, "cli_version": "2.1.305 (Claude Code)"}
    (tmp_path / "manifest.json").write_text(json.dumps({"citizens": {"config": config}}))
    with pytest.raises(SystemExit, match=r"backend\.cli_version"):
        heldout.check_calibration_identity(tmp_path, backend)


def test_cli_version_is_pinned() -> None:
    assert config.B4_CLI_VERSION == "2.1.304"
    heldout.require_cli_version({"cli_version": "2.1.304 (Claude Code)"})
    for found in ("2.1.305 (Claude Code)", None, ""):
        with pytest.raises(SystemExit, match=r"2\.1\.304"):
            heldout.require_cli_version({"cli_version": found})


def test_b4_game_counts_are_the_protocol_values() -> None:
    assert heldout.allowed_b4_games("Interpolating") == (40, 30)
    assert heldout.allowed_b4_games("M1") == heldout.allowed_b4_games("M2") == (40, 30, 20)


FROZEN = data.FrozenProtocol(path="p", sha256="f" * 64, status=data.FROZEN)


def _post_freeze_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(confirm: bool) -> tuple[data.FrozenProtocol, str]:
        return FROZEN, "c0ffee"

    monkeypatch.setattr(heldout, "require_post_freeze", guard)


def test_b4_refuses_a_game_count_outside_the_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _post_freeze_ok(monkeypatch)
    argv = ["--runs-dir", str(tmp_path), "b4", "--confirm-frozen", "--variant", "plain",
            "--mechanism", "Interpolating", "--run-id", "x-v1"]  # fmt: skip
    for games in ("20", "25", "41"):
        with pytest.raises(SystemExit, match="--games"):
            heldout.main([*argv, "--games", games])


# --- evaluation inputs (A25) --------------------------------------------------------------


def _input_citizens(stage: str, **extra: Any) -> dict[str, Any]:
    return {
        "stage": stage,
        "validation_protocol": {"sha256": FROZEN.sha256, "status": data.FROZEN},
        "config_sha256": config.config_sha256(),
        **extra,
    }


def _surrogate_manifest(*, reference: bool = True, differs: float = 0.0) -> dict[str, Any]:
    checks = [
        {"name": f"surrogate_{k}_differs_from_reference", "observed": differs,
         "threshold": 0.0, "passed": differs == 0.0}
        for k in ("m1", "m2")
    ]  # fmt: skip
    return {
        "status": "validated" if differs == 0.0 else "computed_unvalidated",
        "validation": {"checks": checks if reference else []},
        "citizens": {
            "reference_parameters_sha256": {"M1": "a" * 64, "M2": "b" * 64} if reference else None
        },
    }


def test_surrogate_reproduction_problems() -> None:
    assert heldout.surrogate_reproduction_problems(_surrogate_manifest()) == []
    differs = heldout.surrogate_reproduction_problems(_surrogate_manifest(differs=1.0))
    assert any("M1 parameters differ" in p for p in differs)
    assert any("status" in p for p in differs)
    missing = heldout.surrogate_reproduction_problems(_surrogate_manifest(reference=False))
    assert any("--reproduce" in p for p in missing)
    assert any("no reproduction check for M2" in p for p in missing)


def _inputs(tmp_path: Path, surrogate: dict[str, Any], **fit_extra: Any) -> tuple[Path, Path, Path]:
    fitted, sur, g1b = tmp_path / "fit", tmp_path / "sur", tmp_path / "g1b"
    _write_run(fitted, {}, _input_citizens("c2_fit", fit_settings={"protocol_settings": True},
                                           **fit_extra))  # fmt: skip
    entry = {"bc_validation_fidelity": {"accepted": True}}
    _write_run(sur, {}, _input_citizens("c2_surrogate_fit", surrogates={"M1": entry, "M2": entry}))
    m = json.loads((sur / "manifest.json").read_text())
    m.update({k: v for k, v in surrogate.items() if k != "citizens"})
    m["citizens"].update(surrogate["citizens"])
    (sur / "manifest.json").write_text(json.dumps(m))
    z = json.dumps({"rows": [["n", "surplus", "Exp 2", None, M1_NAME, "P", 40, 40, 2.9, 2.9,
                              0.05, "true"]]}).encode()  # fmt: skip
    _write_run(g1b, {"z_values.json": z}, _input_citizens("c1_g1b"))
    return fitted, sur, g1b


M1_NAME = "RL Agent (M1)"


def test_load_inputs_refuses_a_surrogate_run_that_did_not_reproduce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phasespace.citizens import fit, surrogate

    def load_fitted(path: Path) -> str:
        return "fitted"

    def load_mechanism(path: Path, key: str) -> str:
        return f"mech-{key}"

    monkeypatch.setattr(fit, "load_fitted", load_fitted)
    monkeypatch.setattr(surrogate, "load_mechanism", load_mechanism)
    fitted, sur, g1b = _inputs(tmp_path / "bad", _surrogate_manifest(differs=1.0))
    with pytest.raises(ValueError, match="M1 parameters differ"):
        heldout.load_inputs(fitted, sur, g1b, FROZEN)
    with pytest.raises(ValueError, match="M1 parameters differ"):
        heldout.load_inputs(fitted, sur, g1b, FROZEN, override_surrogate_reproduction="  ")
    inputs = heldout.load_inputs(
        fitted, sur, g1b, FROZEN, override_surrogate_reproduction="owner: float noise only"
    )
    assert inputs.surrogate_reproduction["owner_override_reason"] == "owner: float noise only"
    assert inputs.surrogate_reproduction["problems"]
    good = heldout.load_inputs(*_inputs(tmp_path / "good", _surrogate_manifest()), FROZEN)
    assert good.surrogate_reproduction == {"problems": [], "owner_override_reason": None}
    assert good.mechanisms == {"M1": "mech-M1", "M2": "mech-M2"}
    assert good.g1b_failed_regimes == frozenset()


def test_load_inputs_refuses_runs_not_made_under_the_frozen_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted, sur, g1b = _inputs(tmp_path, _surrogate_manifest())
    m = json.loads((fitted / "manifest.json").read_text())
    m["citizens"]["stage"] = "c2_smoke"
    m["citizens"]["config_sha256"] = "0" * 64
    m["citizens"]["fit_settings"]["protocol_settings"] = False
    m["source"]["git_dirty"] = True
    (fitted / "manifest.json").write_text(json.dumps(m))
    other = data.FrozenProtocol(path="p", sha256="e" * 64, status=data.FROZEN)
    with pytest.raises(ValueError) as err:
        heldout.load_inputs(fitted, sur, g1b, other)
    text = str(err.value)
    for needle in ("stage 'c2_smoke'", "config_sha256", "dirty tree", "reduced fit settings",
                   "surrogate run: not made under the frozen protocol"):  # fmt: skip
        assert needle in text


# --- M2 Exp 4 layout and the B4 type targets ---------------------------------------------


def _toy_games(launch: list[str], game: list[int], n_rounds: list[int], t: int = 3) -> Games:
    g = len(launch)
    per_game: dict[str, Any] = {
        "mechanism": np.array([M1_NAME] * g),
        "cohort": np.array(["Exp 4"] * g),
        "exp4_game": np.array(game, dtype=np.int64),
        "launch_id": np.array(launch),
        "n_rounds": np.array(n_rounds, dtype=np.int64),
        "first_row": np.zeros(g, dtype=np.int64),
    }
    four = {"offers", "actions", "rewards", "prev_reciprocations", "prev_offers",
            "next_reciprocations", "player_present", "player_has_no_choice"}  # fmt: skip
    kw: dict[str, Any] = {
        f.name: per_game[f.name]
        if f.name in per_game
        else np.zeros((g, t, 4) if f.name in four else (g, t))
        for f in dataclasses.fields(Games)
    }
    return Games(**kw)


def test_m2_exp4_layout_repeats_each_human_set_in_game_order() -> None:
    human = _toy_games(["b", "a", "a", "b", "a", "b"], [2, 1, 3, 1, 2, 3], [7, 25, 9, 31, 12, 4])
    lengths, units = heldout._sim_layout_exp4(human, 2)  # pyright: ignore[reportPrivateUsage]
    # Set "a": games 1, 2, 3 have 25, 12, 9 rounds; set "b": 31, 7, 4.
    assert lengths.tolist() == [25, 12, 9, 25, 12, 9, 31, 7, 4, 31, 7, 4]
    assert units.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


@pytest.mark.skipif(
    not data.processed_path(PROCESSED, data.TRAINING_FILE).exists(), reason="dataset absent"
)
def test_human_type_targets_are_the_fit_split_proportional_exp1_exp2_games() -> None:
    from phasespace.citizens import metrics
    from phasespace.citizens.split import FIT, human_training_games, make_split

    training = data.load_training(PROCESSED)
    h, s, n = heldout.human_type_targets(training)
    human = human_training_games(training)
    mask = make_split(training).mask(human, FIT)
    mask &= (human.mechanism == "Proportional Baseline") & np.isin(human.cohort, ["Exp 1", "Exp 2"])
    g = human.select(mask)
    assert n == g.n_games == 64  # 32 fit units each of Exp 1 and Exp 2 Proportional
    last = g.offers[np.arange(g.n_games), g.n_rounds - 1]
    reference = {
        "surplus": metrics.surplus(g.rewards, g.n_rounds),
        "gini": metrics.reward_gini(g.rewards, g.n_rounds),
        "depletion_round": metrics.depletion_round(g.pool, g.n_rounds).astype(float),
        "players_active": (last >= 1.0).sum(axis=1).astype(float),
    }
    for k, v in reference.items():
        assert h[k] == pytest.approx(float(np.mean(v)), rel=1e-12)
        assert s[k] == pytest.approx(float(np.std(v, ddof=1)), rel=1e-12)


# --- B4 evaluation (A24) -------------------------------------------------------------------

REFERENCE = {
    "surplus": {"lo": 4.0, "hi": 6.0, "tau": 0.1, "w1_scale": 2.0},
    "gini": {"lo": 0.1, "hi": 0.2, "tau": 0.005, "w1_scale": 0.1},
    "depletion_round": {"lo": 39.0, "hi": 40.0, "tau": 0.5, "w1_scale": 0.5},
    "players_active": {"lo": 3.0, "hi": 3.5, "tau": 0.1, "w1_scale": 1.0},
}
HUMAN = {
    "surplus": [4.0, 6.0],
    "gini": [0.1, 0.3],
    "depletion_round": [40.0, 40.0],
    "players_active": [3.0, 4.0],
}


def _b4_values(n: int) -> dict[str, list[float]]:
    # surplus 5 (in range), gini 0.4 (out), depletion 40 (in), players active 3 (in).
    return {"surplus": [5.0] * n, "gini": [0.4] * n, "depletion_round": [40.0] * n,
            "players_active": [3.0] * n}  # fmt: skip


def test_b4_criterion_against_hand_computed_values() -> None:
    r = heldout.evaluate_b4("plain", "R-I", _b4_values(36), 4, HUMAN, REFERENCE,
                            m0_sim_passes_a=False, m0_sim_w1=None, closed_loop=True)  # fmt: skip
    assert [row.in_range for row in r.rows] == [True, False, True, True]
    assert r.pass_outcomes and r.label == C
    # Constant B4 values: the game bootstrap CI collapses to the value.
    assert (r.rows[0].b4.value, r.rows[0].b4.lo, r.rows[0].b4.hi) == (5.0, 5.0, 5.0)
    # W1 by hand: |5-4|/2 + |5-6|/2 = 1 -> /2; gini (0.3 + 0.1)/2 = 0.2 -> /0.1;
    # depletion 0; players active (0 + 1)/2 = 0.5 -> /1.  Mean (0.5 + 2 + 0 + 0.5)/4.
    assert [row.w1_normalised for row in r.rows] == pytest.approx([0.5, 2.0, 0.0, 0.5])
    assert r.mean_normalised_w1 == pytest.approx(0.75)
    assert r.discard_fraction == pytest.approx(0.1) and r.gating is True and r.counts
    beat = heldout.evaluate_b4("plain", "R-I", _b4_values(36), 4, HUMAN, REFERENCE,
                               m0_sim_passes_a=True, m0_sim_w1=0.76, closed_loop=True)  # fmt: skip
    assert beat.label == C
    lose = heldout.evaluate_b4("plain", "R-I", _b4_values(36), 4, HUMAN, REFERENCE,
                               m0_sim_passes_a=True, m0_sim_w1=0.75, closed_loop=True)  # fmt: skip
    assert lose.label == heldout.NOT_BEATING_M0_SIM and not lose.counts


def test_b4_gating_and_tiers() -> None:
    def run(n: int, discarded: int, closed_loop: bool = True) -> heldout.B4Result:
        return heldout.evaluate_b4("calibrated", "R-M1", _b4_values(n), discarded, HUMAN,
                                   REFERENCE, m0_sim_passes_a=False, m0_sim_w1=None,
                                   closed_loop=closed_loop)  # fmt: skip

    assert run(40, 5).discard_fraction == pytest.approx(5 / 45)
    assert run(40, 5).gating is False and run(40, 5).counts is False
    assert run(19, 0).gating is False  # below 20 complete games: exploratory (§7)
    assert run(20, 2).gating is True
    assert run(40, 0, closed_loop=False).label.startswith("diagnostic")
    with pytest.raises(ValueError, match="no complete game"):
        run(0, 3)


def test_b4_bootstrap_uses_game_resamples_with_seed_9001() -> None:
    values = _b4_values(10)
    values["surplus"] = [float(v) for v in range(10)]
    r = heldout.evaluate_b4("plain", "R-I", values, 0, HUMAN, REFERENCE,
                            m0_sim_passes_a=False, m0_sim_w1=None, closed_loop=True)  # fmt: skip
    draws = np.random.default_rng(9001).integers(0, 10, size=(10_000, 10))
    means = np.asarray(values["surplus"])[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    assert (r.rows[0].b4.lo, r.rows[0].b4.hi) == pytest.approx((lo, hi))


def test_b4_evaluate_command_on_synthetic_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _post_freeze_ok(monkeypatch)
    names = config.METRICS
    units = {"columns": [{"name": n} for n in ("regime", "cohort", "launch_id", *names)],
             "rows": [["R-I", "Exp 2", f"g{k}", *(HUMAN[m][k] for m in names)]
                      for k in range(2)]}  # fmt: skip
    comparison = {"R-I": {"m0_sim_passes_a": True, "m0_sim_mean_normalised_w1": 0.9}}
    _write_run(
        tmp_path / "eval",
        {heldout.HUMAN_UNITS_FILE: json.dumps(units).encode()},
        {
            "stage": "c2_heldout_eval",
            "validation_protocol": {"sha256": FROZEN.sha256},
            "config_sha256": config.config_sha256(),
            "human_reference": {"R-I": REFERENCE},
            "g2": {"w1_comparison": comparison},
            "closed_loop_tier": {"R-I": True},
        },
    )
    rows = [["complete", "[]", 5.0, 0.4, 40.0, 3.0]] * 36 + [["discarded", "[]", *[None] * 4]] * 4
    for variant in ("plain", "calibrated"):
        _write_run(
            tmp_path / f"b4-{variant}",
            {"games.json": _games_json(rows)},
            {
                "stage": "c2_b4_heldout",
                "frozen_protocol_sha256": FROZEN.sha256,
                "config_sha256": config.config_sha256(),
                "config": {"variant": variant},
                "mechanism_key": "Interpolating",
            },
        )
    argv = ["--runs-dir", str(tmp_path / "out"), "b4-evaluate", "--confirm-frozen",
            "--evaluation-run", str(tmp_path / "eval"), "--b4-run", str(tmp_path / "b4-plain"),
            "--b4-run", str(tmp_path / "b4-calibrated"), "--run-id", "b4-eval-v1"]  # fmt: skip
    assert heldout.main(argv) == 0
    manifest = json.loads((tmp_path / "out" / "b4-eval-v1" / "manifest.json").read_text())
    results = {r["variant"]: r for r in manifest["citizens"]["b4"]}
    assert results["plain"]["label"] == C  # 0.75 < 0.9
    assert results["plain"]["discard_fraction"] == pytest.approx(0.1)
    assert results["plain"]["b4_gating"] is True
    # A B4 run made under another protocol is refused.
    m = json.loads((tmp_path / "b4-plain" / "manifest.json").read_text())
    m["citizens"]["frozen_protocol_sha256"] = "0" * 64
    (tmp_path / "b4-plain" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="not made under the frozen protocol"):
        heldout.main([*argv[:-1], "b4-eval-v2"])


def test_surrogate_human_fidelity_runs_the_post_freeze_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phasespace.citizens import surrogate

    def frozen_guard(*args: object) -> data.FrozenProtocol:
        return FROZEN

    monkeypatch.setattr(heldout.data, "require_frozen_protocol", frozen_guard)
    monkeypatch.setattr(heldout, "git_source", lambda: ("abc", True))

    def no_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("held-out rows were read")

    monkeypatch.setattr(surrogate.data, "load_heldout", no_read)
    with pytest.raises(heldout.DirtyTreeError):
        surrogate.human_fidelity({}, confirm_frozen=True)


def test_count_only_identities_on_toy_rows() -> None:
    from phasespace.citizens import validation

    games = _toy_games(["a", "b"], [1, 2], [2, 1], t=2)
    offers = np.zeros((2, 2, 4))
    offers[0, 0] = [5.0, 2.5, 0.5, 5.0]
    offers[0, 1] = [5.0, 5.0, 5.0, 5.0]
    offers[1, 0] = [1.9996, 3.0, 3.0, 3.0]
    offers[1, 1] = [9.0, 9.0, 9.0, 9.0]  # padding (game b has 1 round): never counted
    actions = np.zeros((2, 2, 4))
    actions[0, 0] = [1.5, 3.0, 1.0, 1.0]  # non-integer; c > floor(e); c > 0 on e < 1; ok
    actions[0, 1] = [0.0, 5.0, -1.0, 2.0]  # ok; c = e = 5 ok; negative; ok
    actions[1, 0] = [2.0, 0.0, 0.0, 0.0]  # c = 2 > floor(1.9996) = 1
    actions[1, 1] = [99.0, 0.5, 0.0, 0.0]  # padding
    active = np.array([[3, 4], [4, 0]])  # row (0, 0): 3 players offered >= 1; (1, 0): 4
    active[0, 1] = 3  # mismatch: 4 offered >= 1
    games = dataclasses.replace(games, offers=offers, actions=actions, players_kept_active=active)
    r = validation.action_anomalies(games, "toy")
    assert r.entries == 12 and r.n_fail == 5
    assert dict(r.details) == {
        "non_integer_c": 1,
        "c_above_floor_e": 2,
        "positive_c_on_offer_below_1": 1,
        "negative_c": 1,
        "games_with_anomalies": 2,
    }
    assert r.max_abs_residual == 1.0  # indicator residuals: no behavioural value leaves
    p = validation.players_active_identity(games, "toy")
    assert (p.entries, p.n_fail, dict(p.details)) == (3, 1, {"games_with_mismatch": 1})
