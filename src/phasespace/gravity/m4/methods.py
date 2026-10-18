"""The compared methods of protocol section 6, all on one counting `Evaluator`.

Each method only *proposes* decision vectors in u in [0, 1]^3; every propagation goes
through the evaluator, so inner optimizer work (finite-difference columns, line-search
and trust-region trials, DE populations, emitter solves, CC predictors and correctors)
is counted against the same run budget N. A method ends when the evaluator refuses an
evaluation (`BudgetExhaustedError`); the interrupted solve or batch keeps what it already
evaluated. Every method's catalog is then built from its full log (`catalog`).

- D: the first N points of the seed's Sobol stream.
- MS: Sobol starts in order; a trf least-squares solve on r from every start where r is
  defined, capped at c_MS propagations per solve.
- MBH: monotonic basin hopping with the MS local solve (cap 200).
- DE: scipy differential_evolution on the merit J, with restarts (seed + 1000 k).
- CC: natural-parameter continuation in the descriptors, seeded in budget from Sobol
  batches; CC* is the same continuation seeded with the M3 candidates c1 and c2.
- T: MAP-Elites with iso+line-dd variation; T2 adds one r1 corridor solve per batch.

Sobol stream: `scipy.stats.qmc.Sobol(d=3, scramble=True, rng=default_rng(seed))`, shared
by D, MS, MBH, CC, T and T2 (common random numbers). Method-internal randomness uses
`default_rng((seed, stream))` (`config.RNG_STREAMS`).
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

from phasespace.gravity.dynamics import FloatArray
from phasespace.gravity.m4.config import (
    ORACLE_SEEDS,
    PROTOCOL,
    RNG_STREAMS,
    BasinHoppingConfig,
    ContinuationConfig,
    CorridorEmitterConfig,
    DifferentialEvolutionConfig,
    LocalSolve,
    MapElitesConfig,
    MethodConfig,
    MultiStartConfig,
    Unit3,
    to_jsonable,
)
from phasespace.gravity.m4.evaluation import (
    BudgetExhaustedError,
    Cell,
    Evaluation,
    Evaluator,
    cell_centre,
    cell_of,
    merit,
    residual,
    residual_defined,
    to_unit,
)

_UNDEF: Final = PROTOCOL.objective.undefined_residual
_BINS: Final = PROTOCOL.descriptors.bins


@dataclass(slots=True)
class MethodRun:
    """A finished method run: its log lives in the evaluator; counters are method-specific."""

    method: str
    seed: int
    budget: int
    config: dict[str, Any]
    evaluator: Evaluator
    counters: dict[str, int] = field(default_factory=dict[str, int])
    wall_seconds: float = 0.0
    status: str = "budget_used"


class SobolStream:
    """The seed's scrambled Sobol points in generator order, drawn lazily in blocks."""

    def __init__(self, seed: int, block_log2: int = 10) -> None:
        self.sampler = qmc.Sobol(d=3, scramble=True, rng=np.random.default_rng(seed))
        self.block_log2 = block_log2
        self.points: list[Unit3] = []
        self.position = 0

    def _extend(self) -> None:
        # random_base2 keeps the balance property; the total stays a power of two.
        m = self.block_log2 if not self.points else int(math.log2(len(self.points)))
        block = self.sampler.random_base2(m)
        self.points.extend((float(p[0]), float(p[1]), float(p[2])) for p in block)

    def take(self, n: int) -> list[Unit3]:
        while self.position + n > len(self.points):
            self._extend()
        out = self.points[self.position : self.position + n]
        self.position += n
        return out

    def __iter__(self) -> Iterator[Unit3]:
        while True:
            yield self.take(1)[0]


def sobol_prefix(seed: int, n: int) -> list[Unit3]:
    """The first n points of the seed's Sobol stream."""
    return SobolStream(seed).take(n)


def _clip(u: FloatArray) -> Unit3:
    c = np.clip(u, 0.0, 1.0)
    return float(c[0]), float(c[1]), float(c[2])


# --- shared local solve ------------------------------------------------------------------------


class _CapReachedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SolveResult:
    status: str
    """'converged' (least_squares returned), 'cap' or 'budget'."""
    touched: tuple[Evaluation, ...]
    """Every evaluation the solve requested, in order (cache hits included)."""
    final: Evaluation | None
    """least_squares' returned point, or the touched point with the lowest ||R|| if stopped."""
    propagations: int


def local_solve(
    ev: Evaluator,
    u0: Unit3,
    fun: Callable[[Evaluation], FloatArray],
    *,
    cap: int,
    tag: str,
    lo: Sequence[float] = (0.0, 0.0, 0.0),
    hi: Sequence[float] = (1.0, 1.0, 1.0),
    settings: LocalSolve = PROTOCOL.local_solve,
) -> SolveResult:
    """trf least squares of fun(evaluation(u)) from u0 in the box [lo, hi] (section 6).

    At most `cap` new propagations (cache hits are free, as in the global budget).
    """
    start = ev.used
    touched: list[Evaluation] = []

    def objective(x: FloatArray) -> FloatArray:
        u = _clip(x)
        if not ev.is_cached(u) and ev.used - start >= cap:
            raise _CapReachedError
        e = ev.evaluate(u, tag)
        touched.append(e)
        return fun(e)

    status = "converged"
    final: Evaluation | None = None
    try:
        sol = least_squares(
            objective,
            np.asarray(u0, dtype=np.float64),
            jac=settings.jac,
            bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
            method=settings.method,
            ftol=settings.tolerance,
            xtol=settings.tolerance,
            gtol=settings.tolerance,
            x_scale=settings.x_scale,
            max_nfev=100 * cap + 100,
        )
        # least_squares returns an evaluated iterate; through `objective` so that the cap
        # and the budget still hold if it ever were not.
        objective(np.asarray(sol.x, dtype=np.float64))
        final = touched[-1]
    except _CapReachedError:
        status = "cap"
    except BudgetExhaustedError:
        status = "budget"
    if final is None and touched:
        final = min(touched, key=lambda e: float(np.linalg.norm(fun(e))))
    return SolveResult(status, tuple(touched), final, ev.used - start)


def _best_merit(evals: Sequence[Evaluation]) -> Evaluation:
    """Lowest merit J; ties keep the earliest."""
    best = evals[0]
    for e in evals[1:]:
        if merit(e) < merit(best):
            best = e
    return best


def _count(counters: dict[str, int], key: str, by: int = 1) -> None:
    counters[key] = counters.get(key, 0) + by


# --- D ---------------------------------------------------------------------------------------


def run_dense(ev: Evaluator, seed: int) -> dict[str, int]:
    """D: the first N Sobol points, one propagation each."""
    ev.evaluate_batch(sobol_prefix(seed, ev.remaining), "sobol")
    return {}


# --- MS --------------------------------------------------------------------------------------


def run_multistart(ev: Evaluator, seed: int, cfg: MultiStartConfig) -> dict[str, int]:
    counters: dict[str, int] = {}
    try:
        for u in SobolStream(seed):
            start = ev.evaluate(u, "start")
            _count(counters, "starts")
            if not residual_defined(start):
                continue
            solve = local_solve(ev, u, residual, cap=cfg.max_propagations_per_start, tag="local")
            _count(counters, f"solves_{solve.status}")
            if solve.status == "budget":
                break
    except BudgetExhaustedError:
        pass
    return counters


# --- MBH -------------------------------------------------------------------------------------


def run_basin_hopping(ev: Evaluator, seed: int, cfg: BasinHoppingConfig) -> dict[str, int]:
    """Monotonic basin hopping (section 6). The result of a local solve, compared for
    acceptance, is the lowest-J evaluation it touched (for a fresh incumbent, its start
    included), not least_squares' returned x; ties keep the earliest.
    """
    counters: dict[str, int] = {}
    rng = np.random.default_rng((seed, RNG_STREAMS["MBH"]))
    stream = iter(SobolStream(seed))

    def fresh_incumbent() -> Evaluation | None:
        for u in stream:
            start = ev.evaluate(u, "start")
            _count(counters, "starts")
            if not residual_defined(start):
                continue
            solve = local_solve(ev, u, residual, cap=cfg.local_cap, tag="local")
            _count(counters, f"solves_{solve.status}")
            if solve.status == "budget":
                return None
            return _best_merit([start, *solve.touched])
        return None

    try:
        incumbent = fresh_incumbent()
        fails = 0
        while incumbent is not None:
            if fails >= cfg.n_fail:
                _count(counters, "restarts")
                incumbent = fresh_incumbent()
                fails = 0
                continue
            hop = _clip(np.asarray(incumbent.u) + rng.uniform(-cfg.delta, cfg.delta, 3))
            solve = local_solve(ev, hop, residual, cap=cfg.local_cap, tag="hop")
            _count(counters, f"solves_{solve.status}")
            if solve.status == "budget" or not solve.touched:
                break
            candidate = _best_merit(solve.touched)
            if merit(candidate) < merit(incumbent):
                incumbent = candidate
                fails = 0
                _count(counters, "accepted")
            else:
                fails += 1
    except BudgetExhaustedError:
        pass
    return counters


# --- DE --------------------------------------------------------------------------------------


class _PrefetchMap:
    """Map-like `workers` for DE: evaluates the population as one (parallel) batch first."""

    def __init__(self, ev: Evaluator) -> None:
        self.ev = ev

    def __call__[VT, RT](self, func: Callable[[VT], RT], iterable: Iterable[VT], /) -> list[RT]:
        items = list(iterable)
        self.ev.evaluate_batch([_clip(np.asarray(x, dtype=np.float64)) for x in items], "de")
        hits = self.ev.cache_hits
        out = [func(x) for x in items]
        self.ev.cache_hits = hits  # the same evaluations as the batch: not extra hits
        return out


def run_differential_evolution(
    ev: Evaluator, seed: int, cfg: DifferentialEvolutionConfig
) -> dict[str, int]:
    """scipy DE on J in the unit cube; restarts with seed + 1000 k until N is used."""
    counters: dict[str, int] = {}
    popsize = cfg.population // 3

    def energy(x: FloatArray) -> float:
        return merit(ev.evaluate(_clip(np.asarray(x, dtype=np.float64)), "de"))

    mapper = _PrefetchMap(ev)

    effective = cfg.effective_population  # init="sobol" rounds 3 * popsize up to 2^k
    counters["effective_population"] = effective
    k = 0
    while ev.remaining > 0:
        before = ev.used
        # Uses at most effective * (maxiter + 1) <= remaining propagations; a remainder
        # smaller than one population is spent by a restart that the budget interrupts.
        maxiter = max(0, ev.remaining // effective - 1)
        try:
            differential_evolution(
                energy,
                bounds=[(0.0, 1.0)] * 3,
                strategy=cfg.strategy,
                maxiter=maxiter,
                popsize=popsize,
                tol=0.0,
                mutation=cfg.mutation,
                recombination=cfg.recombination,
                rng=seed + cfg.restart_seed_stride * k,
                polish=False,
                init="sobol",
                updating=cfg.updating,
                workers=mapper,
            )
        except BudgetExhaustedError:
            break
        _count(counters, "runs")
        k += 1
        if ev.used == before:
            counters["stalled_stop"] = 1
            break
    return counters


# --- CC / CC* ----------------------------------------------------------------------------------


def _neighbours(cell: Cell, bins: int) -> list[Cell]:
    i, j = cell
    out = [(i + di, j + dj) for di, dj in ((-1, 0), (0, -1), (0, 1), (1, 0))]
    return sorted(c for c in out if 0 <= c[0] < bins and 0 <= c[1] < bins)


def predict(target: Cell, solved: dict[Cell, Unit3]) -> Unit3 | None:
    """Secant extrapolation from two in-line solved cells, else the nearest solved 4-neighbour.

    Candidate lines are tried in row-major order of their nearer cell; ties likewise.
    """
    i, j = target
    lines = sorted(
        [
            ((i - 1, j), (i - 2, j)),
            ((i, j - 1), (i, j - 2)),
            ((i, j + 1), (i, j + 2)),
            ((i + 1, j), (i + 2, j)),
        ]
    )
    for near, far in lines:
        if near in solved and far in solved:
            a, b = np.asarray(solved[near]), np.asarray(solved[far])
            return _clip(2.0 * a - b)
    for near in sorted(c for c in ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j)) if c in solved):
        return solved[near]
    return None


def cc_residual(
    target: Cell, cfg: ContinuationConfig, bins: int = _BINS
) -> Callable[[Evaluation], FloatArray]:
    """R = [(d1 - c1)/w1, (d2 - c2)/w2, 10 r1]; d2 = t_ret, or the time of the post-lunar
    Earth periapsis where h_vp is evaluated when the shell is not reached (section 6)."""
    (c1, c2), (w1, w2) = cell_centre(target, bins)

    def fun(e: Evaluation) -> FloatArray:
        h_p = e.perilune_altitude_km
        d2 = e.return_time_days if math.isfinite(e.return_time_days) else e.perigee_time_days
        if not (residual_defined(e) and math.isfinite(d2)):
            return np.array([_UNDEF, _UNDEF, _UNDEF])
        r1 = float(residual(e)[0])
        return np.array([(math.log10(h_p) - c1) / w1, (d2 - c2) / w2, cfg.r1_weight * r1])

    return fun


@dataclass(slots=True)
class _Continuation:
    ev: Evaluator
    cfg: ContinuationConfig
    counters: dict[str, int]
    solved: dict[Cell, Unit3] = field(default_factory=dict[Cell, Unit3])

    def branch(self, seed_cell: Cell, seed_u: Unit3) -> None:
        """Breadth-first continuation from one seed (raises BudgetExhaustedError)."""
        _count(self.counters, "branches")
        self.solved.setdefault(seed_cell, seed_u)
        attempted: set[Cell] = {seed_cell}
        queue: deque[Cell] = deque(_neighbours(seed_cell, _BINS))
        while queue:
            target = queue.popleft()
            if target in attempted or target in self.solved:
                continue
            attempted.add(target)
            guess = predict(target, self.solved)
            if guess is None:
                continue
            _count(self.counters, "targets")
            solve = local_solve(
                self.ev,
                guess,
                cc_residual(target, self.cfg),
                cap=self.cfg.cap_per_target,
                tag="cc",
            )
            if solve.status == "budget":
                raise BudgetExhaustedError
            landed = solve.final
            cell = cell_of(landed) if landed is not None else None
            if landed is None or cell is None:
                _count(self.counters, "corrector_failures")
                continue
            if cell != target:
                _count(self.counters, "landed_elsewhere")
            if cell in self.solved:
                continue
            self.solved[cell] = landed.u
            _count(self.counters, "cells_solved")
            queue.extend(c for c in _neighbours(cell, _BINS) if c not in attempted)


def run_continuation(ev: Evaluator, seed: int, cfg: ContinuationConfig) -> dict[str, int]:
    """CC: Sobol seeding batches; every feasible sample in a CC-empty cell starts a branch.

    Section 6: a corrector solves the cell its returned point bins into (feasible points
    only; it may differ from the target). Solved cells are shared by all branches: a
    target any branch solved is skipped and the predictor uses any branch's solutions.
    Seeds are taken in Sobol order, re-checking emptiness one at a time.
    """
    counters: dict[str, int] = {}
    cc = _Continuation(ev, cfg, counters)
    stream = SobolStream(seed)
    try:
        while ev.remaining > 0:
            batch = stream.take(cfg.seed_batch)
            results = ev.evaluate_batch(batch, "seed")
            for e in results:
                if e is None:
                    raise BudgetExhaustedError
                cell = cell_of(e)
                if cell is not None and cell not in cc.solved:
                    cc.branch(cell, e.u)
    except BudgetExhaustedError:
        pass
    counters["cells_solved_total"] = len(cc.solved)
    return counters


def run_continuation_oracle(ev: Evaluator, cfg: ContinuationConfig) -> dict[str, int]:
    """CC*: the continuation seeded with the M3 candidates c1, c2 (not budget-matched).

    The two seed propagations are logged (tag 'oracle_seed') and counted separately.
    """
    counters: dict[str, int] = {}
    cc = _Continuation(ev, cfg, counters)
    try:
        for _name, phase, dv, alpha in ORACLE_SEEDS:
            u = to_unit((phase, dv, alpha))
            e = ev.evaluate(u, "oracle_seed")
            _count(counters, "oracle_seed_propagations")
            cell = cell_of(e)
            if cell is None:
                _count(counters, "oracle_seed_infeasible")
                continue
            if cell not in cc.solved:
                cc.branch(cell, e.u)
    except BudgetExhaustedError:
        counters["safety_cap_reached"] = 1
    counters["cells_solved_total"] = len(cc.solved)
    return counters


# --- T / T2 ----------------------------------------------------------------------------------


@dataclass(slots=True)
class _Archive:
    elites: dict[Cell, Evaluation] = field(default_factory=dict[Cell, Evaluation])

    def add(self, e: Evaluation | None) -> bool:
        if e is None:
            return False
        cell = cell_of(e)
        if cell is None:
            return False
        current = self.elites.get(cell)
        if current is None or e.objective_deg < current.objective_deg:
            self.elites[cell] = e
            return True
        return False

    def ordered(self) -> list[Evaluation]:
        return [self.elites[c] for c in sorted(self.elites)]


def _initialise(ev: Evaluator, seed: int, cfg: MapElitesConfig, archive: _Archive) -> None:
    stream = SobolStream(seed)
    n = cfg.n_init
    while ev.remaining > 0:
        for e in ev.evaluate_batch(stream.take(n), "init"):
            archive.add(e)
        if archive.elites:
            return
        n = cfg.init_batch


def _offspring(
    rng: np.random.Generator, parents: Sequence[Evaluation], cfg: MapElitesConfig, count: int
) -> list[Unit3]:
    out: list[Unit3] = []
    for _ in range(count):
        i = int(rng.integers(len(parents)))
        j = int(rng.integers(len(parents)))
        ui, uj = np.asarray(parents[i].u), np.asarray(parents[j].u)
        iso = cfg.sigma_iso * rng.standard_normal(3)
        line = cfg.sigma_line * (uj - ui) * float(rng.standard_normal())
        out.append(_clip(ui + iso + line))
    return out


def _map_elites_loop(
    ev: Evaluator,
    seed: int,
    cfg: MapElitesConfig,
    stream_id: int,
    emitter: CorridorEmitterConfig | None,
) -> dict[str, int]:
    counters: dict[str, int] = {}
    rng = np.random.default_rng((seed, stream_id))
    archive = _Archive()
    _initialise(ev, seed, cfg, archive)
    stalled = 0
    while ev.remaining > 0 and archive.elites:
        before = ev.used
        n_offspring = cfg.batch
        if emitter is not None:
            elites = archive.ordered()
            target = elites[int(rng.integers(len(elites)))]
            centre = np.asarray(target.u)
            lo = np.clip(centre - emitter.emitter_half_width, 0.0, 1.0)
            hi = np.clip(centre + emitter.emitter_half_width, 0.0, 1.0)
            solve = local_solve(
                ev,
                target.u,
                lambda e: residual(e)[:1],
                cap=emitter.emitter_cap,
                tag="emitter",
                lo=tuple(float(v) for v in lo),
                hi=tuple(float(v) for v in hi),
            )
            _count(counters, f"emitter_{solve.status}")
            for e in solve.touched:
                archive.add(e)
            if solve.status == "budget":
                break
            n_offspring = cfg.batch - 1
        children = _offspring(rng, archive.ordered(), cfg, n_offspring)
        results = ev.evaluate_batch(children, "offspring")
        for e in results:
            if archive.add(e):
                _count(counters, "archive_insertions")
        _count(counters, "batches")
        stalled = stalled + 1 if ev.used == before else 0
        if stalled >= cfg.max_stalled_batches:
            counters["stalled_stop"] = 1
            break
    counters["archive_cells"] = len(archive.elites)
    return counters


def run_map_elites(ev: Evaluator, seed: int, cfg: MapElitesConfig) -> dict[str, int]:
    return _map_elites_loop(ev, seed, cfg, RNG_STREAMS["T"], None)


def run_map_elites_emitter(ev: Evaluator, seed: int, cfg: CorridorEmitterConfig) -> dict[str, int]:
    return _map_elites_loop(ev, seed, cfg.map_elites, RNG_STREAMS["T2"], cfg)


# --- dispatch --------------------------------------------------------------------------------


def run_method(method: str, seed: int, cfg: MethodConfig, ev: Evaluator) -> MethodRun:
    """Run one method on `ev` (whose budget is the run budget) and time it."""
    start = time.perf_counter()
    match method, cfg:
        case "D", None:
            counters = run_dense(ev, seed)
        case "MS", MultiStartConfig():
            counters = run_multistart(ev, seed, cfg)
        case "MBH", BasinHoppingConfig():
            counters = run_basin_hopping(ev, seed, cfg)
        case "DE", DifferentialEvolutionConfig():
            counters = run_differential_evolution(ev, seed, cfg)
        case "CC", ContinuationConfig():
            counters = run_continuation(ev, seed, cfg)
        case "CCstar", ContinuationConfig():
            counters = run_continuation_oracle(ev, cfg)
        case "T", MapElitesConfig():
            counters = run_map_elites(ev, seed, cfg)
        case "T2", CorridorEmitterConfig():
            counters = run_map_elites_emitter(ev, seed, cfg)
        case _:
            raise ValueError(f"method {method!r} does not take config {cfg!r}")
    wall = time.perf_counter() - start
    status = "budget_used" if ev.remaining == 0 else "stopped_early"
    return MethodRun(
        method=method,
        seed=seed,
        budget=ev.budget,
        config={} if cfg is None else to_jsonable(cfg),
        evaluator=ev,
        counters=counters,
        wall_seconds=wall,
        status=status,
    )
