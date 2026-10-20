"""C2 on the training data: split, the protocol 13 pre-freeze check, runs and evaluation.

Needs the parsed dataset (`just citizens-data`); skipped when it is absent. Only
TRAINING games are read (data.load_training); held-out games are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phasespace.citizens import config, data, evaluate, fit
from phasespace.citizens.models import b3_ewa as b3
from phasespace.citizens.models.b1_gru import GRUWeights, random_weights
from phasespace.citizens.models.base import streams_from_games, teacher_forced_log_prob
from phasespace.citizens.models.baselines import fit_m0
from phasespace.citizens.records import Games
from phasespace.citizens.simulate import EXACT_MECHANISMS
from phasespace.citizens.split import FIT, VALIDATION, human_training_games, make_split

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
pytestmark = pytest.mark.skipif(
    not data.processed_path(PROCESSED, data.TRAINING_FILE).exists(),
    reason="parsed dataset absent (run `just citizens-data`)",
)


@pytest.fixture(scope="module")
def training() -> Games:
    return data.load_training(PROCESSED)


def test_players_kept_active_equals_players_offered_at_least_one_coin(training: Games) -> None:
    """Protocol 13 pre-freeze check: the simulated definition of players active."""
    human = training.select(training.is_human)
    rows = human.round_mask()
    offered = (human.offers >= 1.0).sum(axis=-1)
    assert int(rows.sum()) == 9832
    assert np.array_equal(human.players_kept_active[rows], offered[rows])


def test_split_follows_protocol_6_1(training: Games) -> None:
    split = make_split(training)
    human = human_training_games(training)
    assert sum(u.split == FIT for u in split.units) == 160
    assert sum(u.split == VALIDATION for u in split.units) == 40
    for mechanism, cohort in config.SPLIT_STRATA:
        stratum = [u for u in split.units if (u.mechanism, u.cohort) == (mechanism, cohort)]
        assert sum(u.split == FIT for u in stratum) == 32
        assert sum(u.split == VALIDATION for u in stratum) == 8
    fit_mask, val_mask = split.mask(human, FIT), split.mask(human, VALIDATION)
    assert int(fit_mask.sum()) == 224 and int(val_mask.sum()) == 56
    assert not np.any(fit_mask & val_mask)
    # An Exp 4 group's three games are always on the same side.
    exp4 = human.cohort == "Exp 4"
    for key in set(human.launch_id[exp4].tolist()):
        sides = set(fit_mask[exp4 & (human.launch_id == key)].tolist())
        assert len(sides) == 1
    # Deterministic, and the first stratum's validation units are the last 8 of the first
    # permutation drawn by default_rng(20261020) over its sorted launch_ids.
    assert make_split(training) == split
    equal = sorted(set(human.where(mechanism="Equal Baseline").launch_id.tolist()))
    perm = np.random.default_rng(20261020).permutation(40)
    expected = {equal[k] for k in perm[32:].tolist()}
    got = {u.launch_id for u in split.units[:40] if u.split == VALIDATION}
    assert got == expected


def _toy_fitted(training: Games) -> fit.Fitted:
    split = make_split(training)
    fit_games, _ = fit.training_streams(training, split)
    fit_s = streams_from_games(fit_games)
    rng = np.random.default_rng(0)

    def float32_weights() -> GRUWeights:
        w = random_weights(rng, scale=0.2)
        return GRUWeights({k: v.astype(np.float32).astype(np.float64) for k, v in w.arrays.items()})

    def start(seed: int, key: str, values: list[float]) -> dict[str, object]:
        return {
            "seed": seed,
            "x0": values,
            key: values,
            "mean_nll": 3.0,
            "converged": True,
            "iterations": 1,
            "message": "toy",
        }

    return fit.Fitted(
        split=split,
        m0_bins=fit_m0(fit_s),
        p0_q=0.3,
        ewa_p0=b3.round1_levels(fit_s),
        b1=[
            fit.B1Member(seed, float32_weights(), {"updates": 10, "seconds": 1.0})
            for seed in (101, 102)
        ],
        b2_starts=[start(201, "theta", [0.1] * 12)],
        b3_starts=[start(301, "params", [0.5, 0.9, 0.2, 2.0, 1.0, 0.5])],
        b23_starts=[start(311, "params", [0.5, 0.9, 0.2, 2.0, 1.0, 0.5, 3.0])],
        settings=fit.FitSettings(b1_seeds=(101, 102), b1_max_updates=10),
        seconds={"B1": 1.0},
    )


def test_fit_run_round_trip_reproduces_the_likelihoods(training: Games, tmp_path: Path) -> None:
    fitted = _toy_fitted(training)
    sidecar = data.processed_sidecar(PROCESSED, data.TRAINING_FILE)
    manifest = fit.write_fit_run(tmp_path / "citizens-c2-fit-test", fitted, training, sidecar)
    assert manifest["status"] == "computed_unvalidated"
    protocol = manifest["citizens"]["validation_protocol"]
    assert protocol["status"] in ("draft", "frozen") and len(protocol["sha256"]) == 64
    assert manifest["citizens"]["config_sha256"] == config.config_sha256()
    loaded = fit.load_fitted(tmp_path / "citizens-c2-fit-test")
    _, val_games = fit.training_streams(training, loaded.split)
    val_s = streams_from_games(val_games.select(np.arange(val_games.n_games) < 6))
    for name, model in fitted.likelihood_models().items():
        before = teacher_forced_log_prob(model, val_s)
        after = teacher_forced_log_prob(loaded.likelihood_models()[name], val_s)
        assert np.array_equal(np.isnan(before), np.isnan(after))
        assert np.nansum(np.abs(before - after)) == 0.0, name
    table = json.loads((tmp_path / "citizens-c2-fit-test" / fit.LIKELIHOOD_FILE).read_text())
    assert {row[0] for row in table["rows"]} == set(evaluate.LIKELIHOOD_MODELS)
    with pytest.raises(FileExistsError):
        fit.write_fit_run(tmp_path / "citizens-c2-fit-test", fitted, training, sidecar)


def test_evaluate_regime_runs_on_training_validation_games(training: Games) -> None:
    fitted = _toy_fitted(training)
    human = human_training_games(training)
    val = human.select(fitted.split.mask(human, VALIDATION))
    games = evaluate.regime_games(val, "Equal Baseline", ("Exp 1",))
    assert games.n_games == 8
    result, simulated = evaluate.evaluate_regime(
        "Equal",
        "Equal",
        games,
        likelihood_models=fitted.likelihood_models(),
        sim_models=fitted.sim_models(),
        mechanism=EXACT_MECHANISMS["Equal"],
        n_sim=20,
        n_resamples=200,
        w1_splits=20,
    )
    assert result.n_units == 8 and set(simulated) == set(evaluate.SIM_MODELS)
    by_name = {m.model: m for m in result.models}
    assert set(by_name) == set(evaluate.LIKELIHOOD_MODELS) | {"M0-sim"}
    assert by_name["M0"].delta is None and by_name["M0-sim"].log_lik is None
    assert by_name["B0"].validated is False  # the rational control fails (b) by a wide margin
    # Exp 4 sets are one unit each with the mean of their three games.
    exp4 = human.where(cohort="Exp 4")
    units = evaluate.units_of(exp4)
    assert units.n_units == 40 and np.bincount(units.game_unit).tolist() == [3] * 40


def test_simulated_units_average_their_games(training: Games) -> None:
    # Grouping simulated games into equal-sized units leaves the mean unchanged and
    # averages the per-unit values, computed here by hand from the simulated games.
    fitted = _toy_fitted(training)
    human = human_training_games(training)
    games = evaluate.regime_games(human, "Mixed Baseline", ("Exp 1",))
    kwargs = {
        "likelihood_models": {"M0": fitted.likelihood_models()["M0"]},
        "sim_models": {"B0": fitted.sim_models()["B0"], "M0-sim": fitted.sim_models()["M0-sim"]},
        "mechanism": EXACT_MECHANISMS["Mixed"],
        "n_sim": 30,
        "n_resamples": 50,
        "w1_splits": 5,
    }
    plain, sims = evaluate.evaluate_regime("Mixed", "Mixed", games, **kwargs)  # pyright: ignore[reportArgumentType]
    units = np.repeat(np.arange(10), 3)
    grouped, _ = evaluate.evaluate_regime("Mixed", "Mixed", games, sim_unit=units, **kwargs)  # pyright: ignore[reportArgumentType]
    per_game = sims["M0-sim"].game_metrics()["surplus"]
    by_unit = per_game.reshape(10, 3).mean(axis=1)

    def row(result: evaluate.RegimeResult) -> evaluate.OutcomeRow:
        m = next(m for m in result.models if m.model == "M0-sim")
        assert m.outcomes is not None
        return next(o for o in m.outcomes if o.metric == "surplus")

    assert row(grouped).sim_mean == pytest.approx(row(plain).sim_mean)
    assert row(grouped).sim_sd == pytest.approx(float(by_unit.std(ddof=1)))
