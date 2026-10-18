"""Free-return feasibility: one propagation per departure, classified by exact predicates.

Definitions (D20; docs/studies/gravity-atlas/m3-feasibility-design.md, section 3):

- Departure: single impulse from a 185 km circular parking orbit (`departure.tli_state`).
- Propagation: DOP853 rtol = atol = 1e-12 for at most T_MAX = 10 d, with the physical
  Earth and Moon collision radii and a terminal inward crossing of the Earth return shell
  R_S = R_earth + 121.92 km (Apollo entry interface, 400 000 ft, over a spherical Earth),
  with the grazing-safe periapsis detection of `propagation.propagate`.
- Feasible when the propagation ends on the return shell (an inbound crossing; the
  departure lies outside the shell and an outward crossing never fires), at least one
  Moon periapsis precedes that crossing, and the closest such Moon periapsis has an
  altitude in [100, 30 000] km. Moon collision, horizon (no return within T_MAX) and
  numerical failures are infeasible.

Every quantity is evaluated from solver events (the return-shell and periapsis events
located by root finding on the dense output), not from sampled states.

The osculating vacuum perigee is the perigee altitude of the two-body conic about the
Earth (GM = 1 - mu in normalized units, no Moon, no atmosphere) through the Earth-relative
inertial state. It is used as the second targeting residual because it remains defined
when a trajectory just misses the shell: it is evaluated at the first Earth approach
after the closest lunar approach -- the return-shell crossing if no Earth periapsis
precedes it, otherwise the first post-lunar Earth periapsis (`perigee_source`). The value
at the shell itself is recorded separately (`return_vacuum_perigee_altitude_km`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

import numpy as np

from phasespace.gravity.departure import PARKING_ALTITUDE_KM, tli_state
from phasespace.gravity.dynamics import FloatArray, jacobi_constant
from phasespace.gravity.propagation import (
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_SOLVER,
    CollisionRadii,
    EventRecord,
    PropagationResult,
    SolverSettings,
    Termination,
    propagate,
)
from phasespace.gravity.system import EARTH_MOON, EarthMoonSystem
from phasespace.gravity.units import (
    rotating_to_inertial,
    time_from_days,
    time_to_days,
    velocity_to_km_s,
)

ENTRY_INTERFACE_ALTITUDE_KM: Final = 121.92
"""Apollo entry interface, 400 000 ft (Apollo 11 Mission Report, Table 7-VII)."""
PERILUNE_BAND_KM: Final = (100.0, 30_000.0)
"""Perilune altitude band (Pavlak 2010, planar free-return family range)."""
T_MAX_DAYS: Final = 10.0
"""Flight-time window: the return must occur within 10 days of TLI."""
TARGET_PERILUNE_ALTITUDE_KM: Final = 1000.0
"""M3 targeting design point for the perilune altitude (declared, not Apollo's value)."""
TARGET_VACUUM_PERIGEE_ALTITUDE_KM: Final = 38.0
"""M3 targeting design point for the vacuum perigee: a round value between Apollo 11's
targeted 37.8 km and the 38.5 km our spherical model gives for its entry state."""
RESIDUAL_SCALES_KM: Final = (1000.0, 10.0)
"""Scales of the two targeting residuals (perilune, vacuum perigee)."""


class Outcome(StrEnum):
    """Feasibility verdict; every value except FEASIBLE is a failure reason."""

    FEASIBLE = "feasible"
    NUMERICAL_FAILURE = "numerical_failure"
    INVALID_START = "invalid_start"
    MOON_COLLISION = "moon_collision"
    EARTH_COLLISION = "earth_collision"
    NO_RETURN_WITHIN_T_MAX = "no_return_within_t_max"
    RETURN_BEFORE_LUNAR_ENCOUNTER = "return_before_lunar_encounter"
    PERILUNE_OUT_OF_BAND = "perilune_out_of_band"


type PerigeeSource = Literal["return_shell", "earth_periapsis", "none"]


@dataclass(frozen=True, slots=True)
class FreeReturnDefinition:
    """Declared predicates, departure model and propagation settings (D20 defaults)."""

    system: EarthMoonSystem = EARTH_MOON
    parking_altitude_km: float = PARKING_ALTITUDE_KM
    return_shell_altitude_km: float = ENTRY_INTERFACE_ALTITUDE_KM
    perilune_band_km: tuple[float, float] = PERILUNE_BAND_KM
    t_max_days: float = T_MAX_DAYS
    solver: SolverSettings = field(default=DEFAULT_SOLVER)
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS

    def __post_init__(self) -> None:
        lo, hi = self.perilune_band_km
        if not (0.0 <= lo < hi):
            raise ValueError(f"invalid perilune band {self.perilune_band_km}")
        if not (0.0 < self.return_shell_altitude_km < self.parking_altitude_km):
            # The departure must start outside the return shell, so the departure itself
            # can never be counted as a return.
            raise ValueError("the return shell must lie below the parking orbit")
        if not (math.isfinite(self.t_max_days) and self.t_max_days > 0.0):
            raise ValueError("t_max_days must be finite and positive")

    @property
    def return_shell_km(self) -> float:
        """Return-shell radius about the Earth centre, km."""
        return self.system.earth_radius_km + self.return_shell_altitude_km

    @property
    def duration(self) -> float:
        """Propagation horizon T_MAX in normalized time."""
        return time_from_days(self.t_max_days, self.system)


DEFAULT_DEFINITION: Final = FreeReturnDefinition()


@dataclass(frozen=True, slots=True)
class FreeReturnRecord:
    """Per-departure evaluation. NaN marks a quantity that is undefined for this outcome.

    Times are from TLI (t0 = 0). Return quantities refer to the return-shell crossing and
    are NaN unless the propagation ended on the shell after a lunar periapsis.
    """

    phase_deg: float
    delta_v_km_s: float
    impulse_angle_deg: float
    outcome: Outcome
    termination: Termination
    t_final_days: float
    perilune_altitude_km: float
    """Closest Moon periapsis before the return (or before t_final without a return)."""
    lunar_encounter_time_days: float
    lunar_passage_sense: int
    """Sign of the rotating-frame Moon-relative angular momentum at perilune (0: none)."""
    moon_periapses_in_band: int
    """Moon periapses in the perilune band before the return (or before t_final)."""
    earth_periapses_after_encounter: int
    """Earth periapses between the closest lunar approach and the return (or t_final)."""
    return_time_days: float
    return_speed_km_s: float
    """Earth-relative inertial speed at the return shell."""
    return_flight_path_angle_deg: float
    """Earth-relative inertial flight-path angle at the return shell (< 0: descending)."""
    return_vacuum_perigee_altitude_km: float
    """Osculating vacuum perigee altitude at the return-shell crossing (NaN: no return)."""
    vacuum_perigee_altitude_km: float
    """Targeting quantity: vacuum perigee at the first post-lunar Earth approach, i.e. the
    shell crossing if it comes before any post-lunar Earth periapsis, else the first
    post-lunar Earth periapsis (keeps the residual continuous across extra passes)."""
    perigee_source: PerigeeSource
    jacobi_constant: float
    jacobi_max_abs_drift_steps: float
    """max |C(s_i) - C(s_0)| over the accepted solver steps (the whole coast)."""
    nfev: int

    @property
    def feasible(self) -> bool:
        return self.outcome is Outcome.FEASIBLE


def osculating_perigee_radius(rel_inertial: FloatArray, gm: float) -> float:
    """Perigee radius of the two-body conic through an Earth-relative inertial state.

    Uses r_p = h^2 / (gm (1 + e)) with e^2 = 1 + 2 E h^2 / gm^2, E = v^2/2 - gm/r; valid
    for elliptic and hyperbolic conics (normalized units in, normalized units out).
    """
    x, y, vx, vy = (float(v) for v in rel_inertial)
    r = math.hypot(x, y)
    h = x * vy - y * vx
    energy = 0.5 * (vx * vx + vy * vy) - gm / r
    e = math.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (gm * gm)))
    return h * h / (gm * (1.0 + e))


def earth_relative_inertial(state: FloatArray, t: float, mu: float) -> FloatArray:
    """Earth-relative inertial state (x, y, vx, vy) of a rotating-frame state at time t."""
    return rotating_to_inertial(state, t, mu, origin="earth")


def earth_angular_momentum_sign(state: FloatArray, t: float, mu: float) -> int:
    """Sign of the Earth-relative inertial angular momentum (+1 posigrade, -1 retrograde)."""
    x, y, vx, vy = (float(v) for v in earth_relative_inertial(state, t, mu))
    return int(np.sign(x * vy - y * vx))


def flight_path_angle_deg(rel_inertial: FloatArray) -> float:
    """Angle of the velocity above the local horizontal, degrees (negative: descending)."""
    x, y, vx, vy = (float(v) for v in rel_inertial)
    sine = (x * vx + y * vy) / (math.hypot(x, y) * math.hypot(vx, vy))
    return math.degrees(math.asin(min(1.0, max(-1.0, sine))))


def _distance_km(state: tuple[float, float, float, float], center_x: float, scale: float) -> float:
    return math.hypot(state[0] - center_x, state[1]) * scale


def classify(
    result: PropagationResult,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    *,
    phase_deg: float = math.nan,
    delta_v_km_s: float = math.nan,
    impulse_angle_deg: float = math.nan,
) -> FreeReturnRecord:
    """Apply the feasibility predicates to one propagation (a pure function of `result`).

    `result` must come from a propagation with `definition`'s return shell enabled; the
    departure parameters are copied into the record only.
    """
    system = definition.system
    if result.system != system:
        raise ValueError("result was propagated with a different system")
    if result.return_shell_km != definition.return_shell_km:
        raise ValueError("result was propagated without the definition's return shell")
    if result.collision != CollisionRadii.of(system):
        raise ValueError("result was propagated without the physical collision radii")
    if result.duration != definition.duration:
        raise ValueError("result was propagated over a horizon other than T_MAX")
    mu = system.mu
    scale = system.length_scale_km
    moon_x = 1.0 - mu
    termination = result.termination
    t_final = result.t_final
    returned = termination is Termination.EARTH_RETURN_SHELL

    states = result.states
    if states.shape[0] > 0 and np.all(np.isfinite(states)):
        c = jacobi_constant(states, mu)
        jacobi0 = float(c[0])
        drift = float(np.max(np.abs(c - c[0])))
    else:
        jacobi0 = drift = math.nan

    lunar = [p for p in result.periapses if p.name == "moon_periapsis" and p.t < t_final]
    closest: EventRecord | None = (
        min(lunar, key=lambda p: _distance_km(p.state, moon_x, scale)) if lunar else None
    )
    lo, hi = definition.perilune_band_km
    h_p = t_moon = math.nan
    sense = 0
    in_band = 0
    after = 0
    if closest is not None:
        h_p = _distance_km(closest.state, moon_x, scale) - system.moon_radius_km
        t_moon = time_to_days(closest.t, system)
        x, y, vx, vy = closest.state
        sense = int(np.sign((x - moon_x) * vy - y * vx))
        in_band = sum(
            lo <= _distance_km(p.state, moon_x, scale) - system.moon_radius_km <= hi for p in lunar
        )
        after = sum(
            1 for p in result.periapses if p.name == "earth_periapsis" and closest.t < p.t < t_final
        )

    t_ret = speed = gamma = h_vp_ret = h_vp = math.nan
    source: PerigeeSource = "none"
    gm = 1.0 - mu

    def perigee_altitude(state: FloatArray, t: float) -> float:
        rel = earth_relative_inertial(state, t, mu)
        return osculating_perigee_radius(rel, gm) * scale - system.earth_radius_km

    if returned and closest is not None:
        s_ret = result.final_state
        rel = earth_relative_inertial(s_ret, t_final, mu)
        t_ret = time_to_days(t_final, system)
        speed = velocity_to_km_s(math.hypot(float(rel[2]), float(rel[3])), system)
        gamma = flight_path_angle_deg(rel)
        h_vp_ret = perigee_altitude(s_ret, t_final)
    if closest is not None:
        later = [
            p for p in result.periapses if p.name == "earth_periapsis" and closest.t < p.t < t_final
        ]
        if later:
            h_vp = perigee_altitude(np.array(later[0].state), later[0].t)
            source = "earth_periapsis"
        elif returned:
            h_vp = h_vp_ret
            source = "return_shell"

    outcome = _outcome(result, closest, h_p, (lo, hi))

    return FreeReturnRecord(
        phase_deg=phase_deg,
        delta_v_km_s=delta_v_km_s,
        impulse_angle_deg=impulse_angle_deg,
        outcome=outcome,
        termination=termination,
        t_final_days=time_to_days(t_final, system),
        perilune_altitude_km=h_p,
        lunar_encounter_time_days=t_moon,
        lunar_passage_sense=sense,
        moon_periapses_in_band=in_band,
        earth_periapses_after_encounter=after,
        return_time_days=t_ret,
        return_speed_km_s=speed,
        return_flight_path_angle_deg=gamma,
        return_vacuum_perigee_altitude_km=h_vp_ret,
        vacuum_perigee_altitude_km=h_vp,
        perigee_source=source,
        jacobi_constant=jacobi0,
        jacobi_max_abs_drift_steps=drift,
        nfev=result.nfev,
    )


def _outcome(
    result: PropagationResult,
    closest: EventRecord | None,
    h_p: float,
    band: tuple[float, float],
) -> Outcome:
    termination = result.termination
    if termination in (
        Termination.NON_FINITE,
        Termination.EVALUATION_LIMIT,
        Termination.SOLVER_FAILURE,
    ):
        return Outcome.NUMERICAL_FAILURE
    if termination in (
        Termination.START_INSIDE_EARTH,
        Termination.START_INSIDE_MOON,
        Termination.START_INSIDE_RETURN_SHELL,
    ):
        return Outcome.INVALID_START
    if termination is Termination.MOON_COLLISION:
        return Outcome.MOON_COLLISION
    if termination is Termination.EARTH_COLLISION:
        # Unreachable while the return shell lies outside the Earth radius; kept explicit.
        return Outcome.EARTH_COLLISION
    if termination is Termination.HORIZON:
        return Outcome.NO_RETURN_WITHIN_T_MAX
    assert termination is Termination.EARTH_RETURN_SHELL
    s = result.final_state
    if (float(s[0]) + result.system.mu) * float(s[2]) + float(s[1]) * float(s[3]) >= 0.0:
        # The shell event fires only on decreasing distance; an outbound state here would
        # mean an inconsistent event location.
        return Outcome.NUMERICAL_FAILURE
    if closest is None:
        return Outcome.RETURN_BEFORE_LUNAR_ENCOUNTER
    lo, hi = band
    if not (lo <= h_p <= hi):
        return Outcome.PERILUNE_OUT_OF_BAND
    return Outcome.FEASIBLE


def propagate_departure(
    phase_deg: float,
    delta_v_km_s: float,
    impulse_angle_deg: float = 0.0,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
    *,
    solver: SolverSettings | None = None,
) -> PropagationResult:
    """Propagate the TLI state for T_MAX with collisions and the return shell enabled."""
    system = definition.system
    state0 = tli_state(
        math.radians(phase_deg),
        delta_v_km_s,
        math.radians(impulse_angle_deg),
        system,
        parking_altitude_km=definition.parking_altitude_km,
    )
    return propagate(
        state0,
        definition.duration,
        system,
        solver=definition.solver if solver is None else solver,
        collision=CollisionRadii.of(system),
        return_shell_km=definition.return_shell_km,
        max_evaluations=definition.max_evaluations,
    )


def evaluate(
    phase_deg: float,
    delta_v_km_s: float,
    impulse_angle_deg: float = 0.0,
    definition: FreeReturnDefinition = DEFAULT_DEFINITION,
) -> FreeReturnRecord:
    """One propagation of one departure, classified."""
    result = propagate_departure(phase_deg, delta_v_km_s, impulse_angle_deg, definition)
    return classify(
        result,
        definition,
        phase_deg=phase_deg,
        delta_v_km_s=delta_v_km_s,
        impulse_angle_deg=impulse_angle_deg,
    )


def targeting_residual(record: FreeReturnRecord) -> tuple[float, float] | None:
    """F = [(h_p - 1000 km)/1000 km, (h_vp - 38 km)/10 km], or None if undefined.

    Defined when a lunar periapsis and a vacuum perigee (at the shell or at the first
    post-lunar Earth periapsis) exist, whatever the feasibility outcome.
    """
    h_p, h_vp = record.perilune_altitude_km, record.vacuum_perigee_altitude_km
    if not (math.isfinite(h_p) and math.isfinite(h_vp)):
        return None
    return (
        (h_p - TARGET_PERILUNE_ALTITUDE_KM) / RESIDUAL_SCALES_KM[0],
        (h_vp - TARGET_VACUUM_PERIGEE_ALTITUDE_KM) / RESIDUAL_SCALES_KM[1],
    )
