"""Adaptive propagation of planar CR3BP states with explicit termination semantics.

`propagate` wraps `scipy.integrate.solve_ivp` and always reports why it stopped:

- ``horizon``: the requested duration was reached;
- ``earth_collision`` / ``moon_collision``: the distance to the body decreased through
  the configured collision radius (event direction -1, i.e. only inward crossings stop
  the integration; an outward crossing never fires);
- ``non_finite``: the equations of motion produced a non-finite derivative (e.g. a
  state exactly at a primary) and the solver could not continue;
- ``evaluation_limit``: more than `max_evaluations` right-hand-side evaluations were
  needed (e.g. a near-singular pass with collisions disabled); no trajectory is returned;
- ``solver_failure``: any other unsuccessful solver exit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import OdeSolution, solve_ivp

from phasespace.gravity.dynamics import STATE_DIM, FloatArray, state_derivative
from phasespace.gravity.system import EarthMoonSystem

type MethodName = Literal["DOP853", "RK45", "Radau", "LSODA"]

INWARD: Final = -1.0
"""Event direction for a decreasing distance-minus-radius function (entering a body)."""


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
    time). `dense` is the continuous solution (None if integration never advanced).
    """

    system: EarthMoonSystem
    solver: SolverSettings
    collision: CollisionRadii
    duration: float
    t: FloatArray
    states: FloatArray
    termination: Termination
    events: tuple[EventRecord, ...]
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


class _DistanceEvent:
    """solve_ivp event g = distance(body) - radius; terminal, inward only."""

    def __init__(self, name: str, body_x: float, radius: float) -> None:
        self.name = name
        self.body_x = body_x
        self.radius = radius
        self.terminal = True
        self.direction = INWARD

    def __call__(self, t: float, y: FloatArray) -> float:
        return math.hypot(float(y[0]) - self.body_x, float(y[1])) - self.radius


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


def propagate(
    state0: npt.ArrayLike,
    duration: float,
    system: EarthMoonSystem,
    *,
    solver: SolverSettings = DEFAULT_SOLVER,
    collision: CollisionRadii | None = None,
    t0: float = 0.0,
    max_evaluations: int = 1_000_000,
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

    events: list[_DistanceEvent] = []
    if radii.earth_km is not None:
        events.append(
            _DistanceEvent(
                Termination.EARTH_COLLISION, -system.mu, radii.earth_km / system.length_scale_km
            )
        )
    if radii.moon_km is not None:
        events.append(
            _DistanceEvent(
                Termination.MOON_COLLISION, 1.0 - system.mu, radii.moon_km / system.length_scale_km
            )
        )

    rhs = _Rhs(system.mu, max_evaluations)

    def result(
        t: FloatArray,
        states: FloatArray,
        termination: Termination,
        records: tuple[EventRecord, ...],
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
            nfev=nfev,
            message=message,
            dense=dense,
        )

    if not np.all(np.isfinite(rhs(t0, y0))):
        return result(
            np.array([t0]),
            y0[np.newaxis, :].copy(),
            Termination.NON_FINITE,
            (),
            1,
            "non-finite derivative at the initial state",
            None,
        )

    try:
        sol = solve_ivp(
            rhs,
            (t0, t0 + duration),
            y0,
            method=solver.method,
            dense_output=True,
            events=events or None,
            rtol=solver.rtol,
            atol=solver.atol,
        )
    except _EvaluationLimitError:
        return result(
            np.array([t0]),
            y0[np.newaxis, :].copy(),
            Termination.EVALUATION_LIMIT,
            (),
            rhs.evaluations,
            f"more than {max_evaluations} right-hand-side evaluations",
            None,
        )

    records: list[EventRecord] = []
    if events:
        for event, t_ev, y_ev in zip(events, sol.t_events, sol.y_events, strict=True):
            for k in range(len(t_ev)):
                yk = np.asarray(y_ev[k], dtype=np.float64)
                records.append(
                    EventRecord(
                        name=event.name,
                        t=float(t_ev[k]),
                        state=(float(yk[0]), float(yk[1]), float(yk[2]), float(yk[3])),
                        direction=event.direction,
                    )
                )
    records.sort(key=lambda r: r.t)

    t_arr = np.asarray(sol.t, dtype=np.float64)
    states = np.asarray(sol.y, dtype=np.float64).T
    finite = bool(np.all(np.isfinite(states)))
    if sol.status == 1 and records:
        termination = Termination(records[-1].name)
    elif sol.status == 0 and finite:
        termination = Termination.HORIZON
    elif rhs.saw_non_finite or not finite:
        termination = Termination.NON_FINITE
    else:
        termination = Termination.SOLVER_FAILURE
    return result(
        t_arr, states, termination, tuple(records), int(sol.nfev), str(sol.message), sol.sol
    )
