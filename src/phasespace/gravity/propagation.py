"""Adaptive propagation of planar CR3BP states with explicit termination semantics.

`propagate` wraps `scipy.integrate.solve_ivp` and always reports why it stopped:

- ``horizon``: the requested duration was reached;
- ``earth_collision`` / ``moon_collision``: the distance to the body decreased through
  the configured collision radius. Detected two ways: a terminal event on
  g = distance - radius with direction -1 (decreasing; an outward crossing never
  fires), and, because a grazing pass can enter and leave the body within one solver
  step without g changing sign at the step ends, a non-terminal periapsis event per
  body (see below). If any periapsis lies inside a body, the first surface crossing
  before it is located on the dense output and the result is truncated there;
- ``start_inside_earth`` / ``start_inside_moon``: the initial state is strictly inside
  an enabled collision radius (g(t0) < 0). Nothing is propagated: the result holds only
  the initial state, no dense solution and nfev = 0;
- ``non_finite``: the equations of motion produced a non-finite derivative (e.g. a
  state exactly at a primary) and the solver could not continue;
- ``evaluation_limit``: more than `max_evaluations` right-hand-side evaluations were
  needed (e.g. a near-singular pass with collisions disabled); no trajectory is returned;
- ``solver_failure``: any other unsuccessful solver exit.

Periapsis events: for each enabled body, h = (r - r_body) . v (rotating-frame position
relative to the body dotted with the rotating-frame velocity; the bodies are fixed in
this frame) crossing zero from negative to positive. Residual blind spot: a periapsis
whose h changes sign twice within one solver step (an apsis pair inside a single step)
is not detected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import OdeSolution, solve_ivp
from scipy.optimize import brentq

from phasespace.gravity.dynamics import STATE_DIM, FloatArray, state_derivative
from phasespace.gravity.system import EarthMoonSystem

type MethodName = Literal["DOP853", "RK45", "Radau", "LSODA"]

INWARD: Final = -1.0
"""Event direction for a decreasing distance-minus-radius function (entering a body)."""
PERIAPSIS: Final = 1.0
"""Event direction for an increasing radial-velocity function (passing a periapsis)."""
DEFAULT_MAX_EVALUATIONS: Final = 1_000_000
"""Default budget of right-hand-side evaluations per propagation."""

_CROSSING_GRID: Final = 65
"""Points used to bracket the first surface crossing before an inside periapsis."""
_ROOT_TOL: Final = 4.0 * float(np.finfo(np.float64).eps)
"""brentq tolerances; the same values solve_ivp uses to locate events."""


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Declared integrator configuration recorded with every result."""

    method: MethodName = "DOP853"
    rtol: float = 1e-12
    atol: float = 1e-12

    def __post_init__(self) -> None:
        if not (self.rtol > 0.0 and self.atol > 0.0):
            raise ValueError("rtol and atol must be positive")


DEFAULT_SOLVER: Final = SolverSettings()


@dataclass(frozen=True, slots=True)
class CollisionRadii:
    """Collision radii in km; None disables the event for that body."""

    earth_km: float | None
    moon_km: float | None

    @staticmethod
    def of(system: EarthMoonSystem) -> CollisionRadii:
        """The system's physical body radii."""
        return CollisionRadii(earth_km=system.earth_radius_km, moon_km=system.moon_radius_km)


class Termination(StrEnum):
    HORIZON = "horizon"
    EARTH_COLLISION = "earth_collision"
    MOON_COLLISION = "moon_collision"
    START_INSIDE_EARTH = "start_inside_earth"
    START_INSIDE_MOON = "start_inside_moon"
    NON_FINITE = "non_finite"
    EVALUATION_LIMIT = "evaluation_limit"
    SOLVER_FAILURE = "solver_failure"


@dataclass(frozen=True, slots=True)
class EventRecord:
    """A detected event: name, time, state and the declared crossing direction."""

    name: str
    t: float
    state: tuple[float, float, float, float]
    direction: float


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Outcome of `propagate`.

    `t` and `states` hold the accepted solver steps (including the start and the final
    time). `dense` is the continuous solution on [t0, t_final] (None if integration never
    advanced). `events` holds collision events only (at most one, the terminating one);
    `periapses` holds the periapsis passages about enabled bodies up to `t_final`, named
    ``earth_periapsis`` / ``moon_periapsis``.
    """

    system: EarthMoonSystem
    solver: SolverSettings
    collision: CollisionRadii
    duration: float
    t: FloatArray
    states: FloatArray
    termination: Termination
    events: tuple[EventRecord, ...]
    periapses: tuple[EventRecord, ...]
    nfev: int
    message: str
    dense: OdeSolution | None

    @property
    def t_final(self) -> float:
        return float(self.t[-1])

    @property
    def final_state(self) -> FloatArray:
        return self.states[-1].copy()

    def sample(self, times: npt.ArrayLike) -> FloatArray:
        """Evaluate the dense solution at `times` (inside [t0, t_final]); shape (n, 4)."""
        if self.dense is None:
            raise ValueError("no dense solution available")
        ts = np.asarray(times, dtype=np.float64)
        if ts.ndim != 1:
            raise ValueError("times must be one-dimensional")
        lo, hi = float(self.t[0]), self.t_final
        if ts.size and (float(ts.min()) < lo or float(ts.max()) > hi):
            raise ValueError(f"sample times must lie in [{lo}, {hi}]")
        return np.asarray(self.dense(ts), dtype=np.float64).T


@dataclass(frozen=True, slots=True)
class _Body:
    """An enabled collision body in normalized units."""

    name: Literal["earth", "moon"]
    x: float
    radius: float
    collision: Termination
    start_inside: Termination

    def surface_gap(self, y: FloatArray) -> float:
        """distance - radius for one state."""
        return math.hypot(float(y[0]) - self.x, float(y[1])) - self.radius


class _DistanceEvent:
    """solve_ivp event g = distance(body) - radius; terminal, inward only."""

    def __init__(self, body: _Body) -> None:
        self.body = body
        self.terminal = True
        self.direction = INWARD

    def __call__(self, t: float, y: FloatArray) -> float:
        return self.body.surface_gap(y)


class _PeriapsisEvent:
    """solve_ivp event h = (r - r_body) . v; non-terminal, negative -> positive only."""

    def __init__(self, body: _Body) -> None:
        self.body = body
        self.terminal = False
        self.direction = PERIAPSIS

    def __call__(self, t: float, y: FloatArray) -> float:
        return (float(y[0]) - self.body.x) * float(y[2]) + float(y[1]) * float(y[3])


class _EvaluationLimitError(Exception):
    pass


class _Rhs:
    """Equations of motion that count evaluations and flag non-finite derivatives."""

    def __init__(self, mu: float, max_evaluations: int) -> None:
        self.mu = mu
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.saw_non_finite = False

    def __call__(self, t: float, y: FloatArray) -> FloatArray:
        self.evaluations += 1
        if self.evaluations > self.max_evaluations:
            raise _EvaluationLimitError
        dy = state_derivative(y, self.mu)
        if not np.all(np.isfinite(dy)):
            self.saw_non_finite = True
        return dy


def _record(name: str, t: float, y: npt.ArrayLike, direction: float) -> EventRecord:
    s = np.asarray(y, dtype=np.float64)
    return EventRecord(
        name=name,
        t=float(t),
        state=(float(s[0]), float(s[1]), float(s[2]), float(s[3])),
        direction=direction,
    )


def _first_surface_crossing(
    body: _Body, dense: OdeSolution, step_times: FloatArray, t_periapsis: float
) -> float:
    """First time before `t_periapsis` (where g < 0) at which g = distance - radius hits 0.

    The bracket starts at the last accepted step time before the periapsis with g > 0
    (g >= 0 at t0 because starts inside a body are rejected); the dense output is scanned
    on a uniform grid for the first sign change, which is then refined with brentq.
    """

    def gap(t: float) -> float:
        return body.surface_gap(np.asarray(dense(t), dtype=np.float64))

    j = int(np.searchsorted(step_times, t_periapsis, side="left")) - 1
    while j > 0 and gap(float(step_times[j])) <= 0.0:
        j -= 1
    lo = float(step_times[max(j, 0)])
    grid = np.linspace(lo, t_periapsis, _CROSSING_GRID)
    values = np.array([gap(float(t)) for t in grid])
    k = int(np.argmax(values <= 0.0))
    if k == 0:
        return lo
    root = brentq(gap, float(grid[k - 1]), float(grid[k]), xtol=_ROOT_TOL, rtol=_ROOT_TOL)
    return float(root)


def _truncate_dense(dense: OdeSolution, t_end: float) -> OdeSolution | None:
    """The dense solution restricted to [t0, t_end] (t_end inside its range)."""
    k = int(np.searchsorted(dense.ts, t_end, side="left"))
    if k == 0:
        return None
    ts = np.append(dense.ts[:k], t_end)
    return OdeSolution(ts, dense.interpolants[:k])


def propagate(
    state0: npt.ArrayLike,
    duration: float,
    system: EarthMoonSystem,
    *,
    solver: SolverSettings = DEFAULT_SOLVER,
    collision: CollisionRadii | None = None,
    t0: float = 0.0,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
) -> PropagationResult:
    """Integrate a rotating-frame state for at most `duration` normalized time units.

    Args:
        state0: Initial state (x, y, vx, vy), normalized; must be finite.
        duration: Positive, finite horizon in normalized time.
        system: Earth-Moon constants (mu and the km scale for collision radii).
        solver: Integrator settings; defaults to DOP853 with rtol = atol = 1e-12.
        collision: Collision radii; None means the system's body radii.
        t0: Initial normalized time.
        max_evaluations: Budget of right-hand-side evaluations before giving up.
    """
    y0 = np.asarray(state0, dtype=np.float64)
    if y0.shape != (STATE_DIM,):
        raise ValueError(f"state0 must have shape ({STATE_DIM},), got {y0.shape}")
    if not np.all(np.isfinite(y0)):
        raise ValueError("state0 must be finite")
    if not (math.isfinite(duration) and duration > 0.0):
        raise ValueError(f"duration must be finite and positive, got {duration}")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be >= 1")
    radii = CollisionRadii.of(system) if collision is None else collision

    bodies: list[_Body] = []
    if radii.earth_km is not None:
        bodies.append(
            _Body(
                "earth",
                -system.mu,
                radii.earth_km / system.length_scale_km,
                Termination.EARTH_COLLISION,
                Termination.START_INSIDE_EARTH,
            )
        )
    if radii.moon_km is not None:
        bodies.append(
            _Body(
                "moon",
                1.0 - system.mu,
                radii.moon_km / system.length_scale_km,
                Termination.MOON_COLLISION,
                Termination.START_INSIDE_MOON,
            )
        )

    rhs = _Rhs(system.mu, max_evaluations)

    def result(
        t: FloatArray,
        states: FloatArray,
        termination: Termination,
        records: tuple[EventRecord, ...],
        periapses: tuple[EventRecord, ...],
        nfev: int,
        message: str,
        dense: OdeSolution | None,
    ) -> PropagationResult:
        return PropagationResult(
            system=system,
            solver=solver,
            collision=radii,
            duration=duration,
            t=t,
            states=states,
            termination=termination,
            events=records,
            periapses=periapses,
            nfev=nfev,
            message=message,
            dense=dense,
        )

    def unpropagated(termination: Termination, nfev: int, message: str) -> PropagationResult:
        return result(
            np.array([t0]), y0[np.newaxis, :].copy(), termination, (), (), nfev, message, None
        )

    for body in bodies:
        if body.surface_gap(y0) < 0.0:
            return unpropagated(
                body.start_inside, 0, f"initial state is inside the {body.name} collision radius"
            )

    if not np.all(np.isfinite(rhs(t0, y0))):
        return unpropagated(Termination.NON_FINITE, 1, "non-finite derivative at the initial state")

    collision_events = [_DistanceEvent(b) for b in bodies]
    periapsis_events = [_PeriapsisEvent(b) for b in bodies]
    try:
        sol = solve_ivp(
            rhs,
            (t0, t0 + duration),
            y0,
            method=solver.method,
            dense_output=True,
            events=[*collision_events, *periapsis_events] or None,
            rtol=solver.rtol,
            atol=solver.atol,
        )
    except _EvaluationLimitError:
        return unpropagated(
            Termination.EVALUATION_LIMIT,
            rhs.evaluations,
            f"more than {max_evaluations} right-hand-side evaluations",
        )

    t_arr = np.asarray(sol.t, dtype=np.float64)
    states = np.asarray(sol.y, dtype=np.float64).T
    dense: OdeSolution | None = sol.sol
    n = len(bodies)
    t_events = list(sol.t_events) if bodies else []
    y_events = list(sol.y_events) if bodies else []

    hits: list[tuple[float, _Body]] = []
    periapses: list[EventRecord] = []
    for i, body in enumerate(bodies):
        hits.extend((float(t), body) for t in t_events[i])
        for t_p, y_p in zip(t_events[n + i], y_events[n + i], strict=True):
            periapses.append(_record(f"{body.name}_periapsis", float(t_p), y_p, PERIAPSIS))
            if dense is not None and body.surface_gap(np.asarray(y_p, dtype=np.float64)) < 0.0:
                hits.append((_first_surface_crossing(body, dense, t_arr, float(t_p)), body))
    periapses.sort(key=lambda r: r.t)

    if hits:
        t_hit, body = min(hits, key=lambda h: h[0])
        assert dense is not None  # a hit implies the integration advanced
        y_hit = np.asarray(dense(t_hit), dtype=np.float64)
        keep = t_arr < t_hit
        return result(
            np.append(t_arr[keep], t_hit),
            np.vstack([states[keep], y_hit]),
            body.collision,
            (_record(body.collision.value, t_hit, y_hit, INWARD),),
            tuple(p for p in periapses if p.t <= t_hit),
            int(sol.nfev),
            f"{body.name} collision radius crossed inward at t={t_hit!r}",
            _truncate_dense(dense, t_hit),
        )

    finite = bool(np.all(np.isfinite(states)))
    if sol.status == 0 and finite:
        termination = Termination.HORIZON
    elif rhs.saw_non_finite or not finite:
        termination = Termination.NON_FINITE
    else:
        termination = Termination.SOLVER_FAILURE
    return result(
        t_arr, states, termination, (), tuple(periapses), int(sol.nfev), str(sol.message), dense
    )
