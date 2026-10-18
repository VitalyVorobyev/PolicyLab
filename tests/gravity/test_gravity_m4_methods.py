"""M4 evaluator and methods: budget parity, caching, determinism, method mechanics.

Method logic is tested on the analytic stand-in `m4_toy` (no propagation); the physical
evaluation is tested on the validated M3 candidate c1 and against the M3 map code.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from m4_toy import surface, toy_evaluate
from scipy.stats import qmc

from phasespace.gravity.free_return import Outcome
from phasespace.gravity.free_return_map import SobolDesign
from phasespace.gravity.m4.catalog import build_catalog
from phasespace.gravity.m4.config import (
    ORACLE_SEEDS,
    BasinHoppingConfig,
    ContinuationConfig,
    CorridorEmitterConfig,
    DifferentialEvolutionConfig,
    MapElitesConfig,
    MethodConfig,
    MultiStartConfig,
    Unit3,
)
from phasespace.gravity.m4.evaluation import (
    BudgetExhaustedError,
    Evaluation,
    Evaluator,
    LogEntry,
    bin_index,
    cell_of,
    evaluate_unit,
    jacobi_scale,
    merit,
    residual,
    to_physical,
    to_unit,
)
from phasespace.gravity.m4.methods import (
    MethodRun,
    local_solve,
    predict,
    run_method,
    sobol_prefix,
)
from phasespace.gravity.m4.runs import evaluation_row

SEED = 4321
TOY_CONFIGS: dict[str, MethodConfig] = {
    "D": None,
    "MS": MultiStartConfig(20),
    "MBH": BasinHoppingConfig(0.02, 5, 30),
    "DE": DifferentialEvolutionConfig(30),
    "CC": ContinuationConfig(seed_batch=64, cap_per_target=30),
    "T": MapElitesConfig(0.005, 0.2, n_init=64, init_batch=32, batch=16),
    "T2": CorridorEmitterConfig(
        MapElitesConfig(0.005, 0.2, n_init=64, init_batch=32, batch=16), emitter_cap=10
    ),
}


class CountingToy:
    """Counts every call of the evaluation function (every 'propagation')."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, u: Unit3) -> Evaluation:
        self.calls += 1
        return toy_evaluate(u)


def _run(method: str, budget: int, seed: int = SEED) -> tuple[MethodRun, CountingToy]:
    fn = CountingToy()
    run = run_method(method, seed, TOY_CONFIGS[method], Evaluator(budget, fn))
    return run, fn


def _row(e: Evaluation) -> str:
    return json.dumps(evaluation_row(LogEntry(0, e, "", 0.0)))


# --- parameterization, descriptors, objective -------------------------------------------------


def test_to_physical_uses_qmc_scale_arithmetic() -> None:
    u = np.random.default_rng(0).random((50, 3))
    expected = qmc.scale(u, [0.0, 3.00, -10.0], [360.0, 3.25, 10.0])
    for row, x in zip(u, expected, strict=True):
        assert to_physical((float(row[0]), float(row[1]), float(row[2]))) == tuple(x)
    x = (229.14224099316039, 3.1567704135102614, 0.0)
    assert to_physical(to_unit(x)) == pytest.approx(x, rel=1e-15, abs=1e-12)


def test_sobol_stream_reproduces_the_m3_sobol_design() -> None:
    # The M4 stream (drawn lazily in doubling blocks) is the M3 S4 design's point order.
    stream = sobol_prefix(20261018, 3000)
    design = SobolDesign(seed=20261018, log2_samples=12).points()
    assert [to_physical(u) for u in stream] == design[:3000]


def test_bin_index_hand_values() -> None:
    d1 = (2.0, math.log10(30_000.0))
    assert bin_index(2.0, d1, 10) == 0
    assert bin_index(3.0, d1, 10) == 4  # 10 * 1 / 2.4771 = 4.04
    assert bin_index(math.log10(30_000.0), d1, 10) == 9  # upper edge clipped
    assert bin_index(4.9, (5.0, 10.0), 10) == 0  # t_ret < 5 d clipped into bin 0
    assert bin_index(7.5, (5.0, 10.0), 10) == 5
    assert bin_index(7.5, (5.0, 10.0), 20) == 10


def _toy_feasible() -> Evaluation:
    return toy_evaluate((0.5, surface(0.5, 0.4), 0.4))


def test_residual_merit_and_objective_hand_values() -> None:
    e = replace(_toy_feasible(), vacuum_perigee_altitude_km=48.0, perilune_altitude_km=50.0)
    r = residual(e)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx((2.0 - math.log10(50.0)) / 0.1)
    infeasible = replace(e, outcome=Outcome.PERILUNE_OUT_OF_BAND)
    assert merit(infeasible) == pytest.approx(100.0 + math.hypot(r[0], r[1]))
    undefined = replace(infeasible, vacuum_perigee_altitude_km=math.nan)
    assert list(residual(undefined)) == [1e3, 1e3]
    assert merit(undefined) == pytest.approx(1100.0)  # capped at 1000
    feasible = replace(_toy_feasible(), return_flight_path_angle_deg=-8.0)
    assert feasible.objective_deg == pytest.approx(1.5)
    assert merit(feasible) == pytest.approx(1.5)


def test_c1_evaluation_matches_m3_and_passes_the_scaled_gate() -> None:
    # c1 (run gravity-free-return-c1-v2): h_p = 1000 km, gamma_S = -6.4848 deg, t_ret 6.12 d.
    e = evaluate_unit(to_unit((229.14224099316039, 3.1567704135102614, 0.0)))
    assert e.outcome is Outcome.FEASIBLE and e.gate_passed
    assert e.perilune_altitude_km == pytest.approx(1000.0, abs=1e-3)
    assert e.objective_deg == pytest.approx(abs(-6.4848139563569 + 6.5), abs=1e-6)
    assert cell_of(e) == (4, 2)  # log10 1000 = 3 -> bin 4; 6.12 d -> bin 2
    assert e.perigee_time_days == e.return_time_days  # h_vp evaluated at the shell
    assert e.entry_sense == 1
    # S ~ 230 near departure: 2(1 - mu)/r1 ~ 115.8 and v^2 ~ 114 (protocol 5.2).
    assert 225.0 < e.jacobi_scale < 240.0
    assert e.jacobi_drift_rel < 1e-11


def test_jacobi_scale_is_the_maximum_of_the_terms() -> None:
    mu = 0.0121505856
    states = np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.1, 0.0]])
    r1 = [0.5 + mu, math.hypot(mu, 0.5)]
    r2 = [abs(0.5 - 1.0 + mu), math.hypot(1.0 - mu, 0.5)]
    terms = [
        0.25 + 2 * (1 - mu) / r1[0] + 2 * mu / r2[0],
        0.25 + 2 * (1 - mu) / r1[1] + 2 * mu / r2[1] + 0.01,
    ]
    assert jacobi_scale(states, mu) == pytest.approx(max(terms), rel=1e-14)


# --- evaluator -----------------------------------------------------------------------------


def test_evaluator_caches_counts_and_stops_at_the_budget() -> None:
    fn = CountingToy()
    ev = Evaluator(3, fn)
    a, b, c, d = (0.5, 0.5, 0.5), (0.6, 0.5, 0.5), (0.7, 0.5, 0.5), (0.8, 0.5, 0.5)
    ev.evaluate(a, "x")
    ev.evaluate(a, "x")
    assert (ev.used, ev.cache_hits, fn.calls) == (1, 1, 1)
    out = ev.evaluate_batch([b, b, c, d, a], "y")  # b twice, d beyond the budget
    assert [o is None for o in out] == [False, False, False, True, False]
    assert (ev.used, ev.cache_hits, fn.calls) == (3, 3, 3)  # a once, b and a again
    with pytest.raises(BudgetExhaustedError):
        ev.evaluate(d, "z")
    assert ev.evaluate(c, "z") is out[2]  # cached points are still served
    assert [e.index for e in ev.log] == [0, 1, 2]
    with pytest.raises(ValueError, match="must lie in"):
        evaluate_unit((1.5, 0.5, 0.5))


def test_parallel_batches_equal_serial_batches() -> None:
    points = sobol_prefix(SEED, 4)
    serial = Evaluator(4)
    serial.evaluate_batch(points, "p")
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = Evaluator(4, executor=pool, chunksize=1)
        parallel.evaluate_batch(points, "p")
    assert [_row(e.evaluation) for e in serial.log] == [_row(e.evaluation) for e in parallel.log]


def test_map_elites_log_is_independent_of_the_worker_count() -> None:
    cfg = TOY_CONFIGS["T"]
    serial = Evaluator(200, toy_evaluate)
    run_method("T", SEED, cfg, serial)
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = Evaluator(200, toy_evaluate, executor=pool)
        run_method("T", SEED, cfg, parallel)
    assert [e.evaluation.u for e in serial.log] == [e.evaluation.u for e in parallel.log]


# --- methods -------------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(TOY_CONFIGS))
def test_every_propagation_is_counted_and_the_budget_is_exact(method: str) -> None:
    run, fn = _run(method, 300)
    assert run.evaluator.used == 300 == fn.calls  # no propagation outside the counter
    assert run.status == "budget_used"
    assert len({e.evaluation.u for e in run.evaluator.log}) == 300  # no recount of repeats


@pytest.mark.parametrize("method", list(TOY_CONFIGS))
def test_methods_are_deterministic_given_the_seed(method: str) -> None:
    first, _ = _run(method, 150)
    again, _ = _run(method, 150)
    other, _ = _run(method, 150, seed=SEED + 1)
    us = [e.evaluation.u for e in first.evaluator.log]
    assert us == [e.evaluation.u for e in again.evaluator.log]
    assert us != [e.evaluation.u for e in other.evaluator.log]


def test_dense_map_and_shared_sobol_starts() -> None:
    dense, _ = _run("D", 64)
    sobol = sobol_prefix(SEED, 64)
    assert [e.evaluation.u for e in dense.evaluator.log] == sobol
    t, _ = _run("T", 200)
    assert [e.evaluation.u for e in t.evaluator.log[:64]] == sobol  # common random numbers
    ms, _ = _run("MS", 200)
    starts = [e.evaluation.u for e in ms.evaluator.log if e.tag == "start"]
    assert starts == sobol[: len(starts)]


def test_inner_optimizer_work_is_counted_within_the_per_start_cap() -> None:
    run, _ = _run("MS", 400)
    tags = [e.tag for e in run.evaluator.log]
    assert tags.count("local") > tags.count("start")  # FD columns and trials are counted
    longest = max(
        len(block) for block in "".join("s" if t == "start" else "l" for t in tags).split("s")
    )
    assert longest <= 20  # c_MS


def test_local_solve_reaches_the_analytic_root() -> None:
    # Toy r1 = 1000 (u1 - f(u0, u2)) with h_p in band: r = 0 on u1 = f(u0, u2).
    ev = Evaluator(500, toy_evaluate)
    solve = local_solve(ev, (0.5, 0.45, 0.4), residual, cap=200, tag="t")
    assert solve.status == "converged" and solve.final is not None
    u0, u1, u2 = solve.final.u
    assert abs(u1 - surface(u0, u2)) < 1e-8
    assert solve.propagations == ev.used


def test_de_population_is_rounded_to_a_power_of_two() -> None:
    run, _ = _run("DE", 300)
    assert run.counters["effective_population"] == 32  # NP = 30 with init="sobol"


def test_map_elites_archive_equals_the_catalog_of_its_log() -> None:
    run, _ = _run("T", 400)
    catalog = build_catalog([e.evaluation for e in run.evaluator.log])
    assert run.counters["archive_cells"] == len(catalog) > 0


def test_t2_emitter_is_capped_and_logged() -> None:
    run, _ = _run("T2", 400)
    tags = [e.tag for e in run.evaluator.log]
    assert tags.count("emitter") > 0
    runs = "".join("e" if t == "emitter" else "." for t in tags).split(".")
    assert max(len(r) for r in runs) <= 10


def test_continuation_solves_cells_and_counts_seeds() -> None:
    run, _ = _run("CC", 600)
    assert run.counters["branches"] >= 1
    assert run.counters["cells_solved_total"] >= 2
    assert Counter(e.tag for e in run.evaluator.log)["seed"] >= 64


def test_oracle_continuation_counts_its_seed_propagations() -> None:
    fn = CountingToy()
    run = run_method("CCstar", 0, ContinuationConfig(), Evaluator(50, fn))
    assert run.counters["oracle_seed_propagations"] == len(ORACLE_SEEDS)
    assert [e.tag for e in run.evaluator.log[:2]] == ["oracle_seed", "oracle_seed"]


def test_predictor_secant_and_nearest_neighbour() -> None:
    solved: dict[tuple[int, int], Unit3] = {(2, 3): (0.2, 0.3, 0.4), (3, 3): (0.25, 0.35, 0.4)}
    # Target (4, 3): in line with (3, 3) and (2, 3): 2 * u(3,3) - u(2,3).
    assert predict((4, 3), solved) == pytest.approx((0.3, 0.4, 0.4))
    # Target (3, 4): only the 4-neighbour (3, 3) is solved.
    assert predict((3, 4), solved) == (0.25, 0.35, 0.4)
    assert predict((7, 7), solved) is None
    # Extrapolation is clipped to the unit cube.
    edge = {(0, 0): (0.9, 0.5, 0.5), (0, 1): (0.95, 0.5, 0.5)}
    assert predict((0, 2), edge) == pytest.approx((1.0, 0.5, 0.5))


def test_run_method_rejects_a_mismatched_config() -> None:
    with pytest.raises(ValueError, match="does not take"):
        run_method("MS", SEED, MapElitesConfig(), Evaluator(10, toy_evaluate))
