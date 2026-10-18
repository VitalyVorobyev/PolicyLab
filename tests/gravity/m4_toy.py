"""An analytic stand-in for the free-return evaluation, for fast M4 method tests.

The landscape mimics the structure of the real problem without any propagation:

- the vacuum perigee is h_vp = 38 km + 10^4 km * (u1 - f(u0, u2)), with
  f = 0.5 + 0.05 sin(2 pi u0) + 0.02 u2, so the feasible set is a thin slab around
  u1 = f (|u1 - f| within about 0.004-0.008 in u);
- r is undefined (no lunar encounter) for u0 < 0.1;
- h_p = 10^(2 + 2.5 u2) km, t_ret = 5 + 5.5 u0 days (> 10 d is a no-return);
- gamma_S = -6.5 deg - 0.08 deg/km * (h_vp - 38 km), so e = 0 on r1 = 0.

Feasible iff r is defined, 0 <= h_vp <= 80 km, h_p in [100, 30 000] km and t_ret <= 10 d.
"""

from __future__ import annotations

import math

from phasespace.gravity.free_return import Outcome
from phasespace.gravity.m4.config import Unit3
from phasespace.gravity.m4.evaluation import Evaluation, to_physical


def surface(u0: float, u2: float) -> float:
    return 0.5 + 0.05 * math.sin(2.0 * math.pi * u0) + 0.02 * u2


def toy_evaluate(u: Unit3) -> Evaluation:
    u0, u1, u2 = u
    phase, dv, alpha = to_physical(u)
    defined = u0 >= 0.1
    h_vp = 38.0 + 1.0e4 * (u1 - surface(u0, u2)) if defined else math.nan
    h_p = 10.0 ** (2.0 + 2.5 * u2) if defined else math.nan
    t_ret = 5.0 + 5.5 * u0
    if not defined:
        outcome = Outcome.RETURN_BEFORE_LUNAR_ENCOUNTER
    elif not (0.0 <= h_vp <= 80.0):
        outcome = Outcome.NO_RETURN_WITHIN_T_MAX
    elif not (100.0 <= h_p <= 30_000.0):
        outcome = Outcome.PERILUNE_OUT_OF_BAND
    elif t_ret > 10.0:
        outcome = Outcome.NO_RETURN_WITHIN_T_MAX
    else:
        outcome = Outcome.FEASIBLE
    feasible = outcome is Outcome.FEASIBLE
    return Evaluation(
        u=u,
        phase_deg=phase,
        delta_v_km_s=dv,
        impulse_angle_deg=alpha,
        outcome=outcome,
        m3_outcome=outcome,
        gate_passed=True,
        perilune_altitude_km=h_p,
        lunar_encounter_time_days=3.0 if defined else math.nan,
        lunar_passage_sense=-1 if defined else 0,
        return_time_days=t_ret if feasible else math.nan,
        return_speed_km_s=11.0 if feasible else math.nan,
        return_flight_path_angle_deg=-6.5 - 0.08 * (h_vp - 38.0) if feasible else math.nan,
        vacuum_perigee_altitude_km=h_vp,
        perigee_time_days=t_ret if defined else math.nan,
        entry_sense=1 if feasible else 0,
        jacobi_scale=230.0,
        jacobi_drift_rel=1e-13,
        nfev=1000,
    )
