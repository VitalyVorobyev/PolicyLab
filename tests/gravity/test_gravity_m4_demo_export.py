"""The M4 public demo export (`phasespace.gravity.m4.demo_export`).

Simplification is checked against analytic curves; paths against declared geometry (the
185 km parking orbit about a 6371 km Earth, the 121.92 km return shell, the Moon radius);
the fan's uncorrected feasibility against `solution_robustness`, the code that wrote the
stored robustness rows. The published files, once they exist, are checked against the
schema in docs/data-contract.md.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from phasespace.gravity.m4.config import GATES_PRIMARY, T_E
from phasespace.gravity.m4.demo_export import (
    FAN_SIMPLIFICATION,
    NOMINAL_SIMPLIFICATION,
    fan_path,
    nominal_path,
    segment_distances,
    simplify,
)
from phasespace.gravity.m4.evaluation import evaluate_unit, to_unit
from phasespace.gravity.m4.robustness import ErrorDraw, error_draws, solution_robustness

C1: Final = (229.14224099316039, 3.1567704135102614, 0.0)
"""The M3 candidate c1 (theta deg, delta-v km/s, alpha deg), a validated free return."""
L_KM: Final = 384_400.0
MU: Final = 0.012150584394709708
PARKING_RADIUS_KM: Final = 6371.0 + 185.0
RETURN_SHELL_KM: Final = 6371.0 + 121.92
MOON_RADIUS_KM: Final = 1737.4
PUBLISHED: Final = (
    Path(__file__).resolve().parents[2] / "web/public/research/gravity-atlas/m4-demo-atlas-v1"
)


def _polyline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Brute force: distance of each point to the nearest segment of the polyline."""
    best = np.full(points.shape[0], np.inf)
    for a, b in pairwise(poly):
        best = np.minimum(best, segment_distances(points, a, b))
    return best


def test_point_segment_distance_hand_values() -> None:
    a, b = np.array([0.0, 0.0]), np.array([4.0, 0.0])
    pts = np.array([[2.0, 3.0], [-3.0, 4.0], [7.0, -4.0], [1.0, 0.0]])
    assert segment_distances(pts, a, b).tolist() == [3.0, 5.0, 5.0, 0.0]


def test_collinear_points_reduce_to_their_endpoints() -> None:
    s = np.linspace(0.0, 1.0, 101)
    line = np.column_stack([3.0 * s - 1.0, -2.0 * s + 5.0])
    assert simplify(line, np.full(101, 1e-9)).tolist() == [0, 100]


@pytest.mark.parametrize("variable", [False, True])
def test_simplification_stays_within_tolerance_of_an_analytic_ellipse(variable: bool) -> None:
    a, b = 10_000.0, 3_000.0

    def ellipse(theta: np.ndarray) -> np.ndarray:
        return np.column_stack([a * np.cos(theta), b * np.sin(theta)])

    theta = np.linspace(0.0, 2.0 * math.pi, 4001)
    points = ellipse(theta)
    # Constant tolerance, or one that shrinks toward the high-curvature ends (as near a body).
    tol = 5.0 + 20.0 * float(variable) * np.abs(np.sin(theta))
    kept = simplify(points, tol, keep=[1000])
    assert 1000 in kept.tolist() and kept[0] == 0 and kept[-1] == 4000
    assert kept.size < 400
    poly = points[kept]
    assert np.all(_polyline_distance(points, poly) <= tol + 1e-9)
    # Between the input samples the curve stays within the tolerance plus the input chord
    # sagitta (step^2 / 8R, R >= b^2/a): checked on the true curve, 8x finer.
    fine = np.linspace(0.0, 2.0 * math.pi, 32001)
    step = a * (theta[1] - theta[0])
    sagitta = step**2 / (8.0 * b * b / a)
    tol_fine = 5.0 + 20.0 * float(variable) * np.abs(np.sin(fine))
    slack = sagitta + 20.0 * (theta[1] - theta[0])  # tolerance change across one input step
    assert np.all(_polyline_distance(ellipse(fine), poly) <= tol_fine + slack)


def _earth_distance(x_km: float, y_km: float) -> float:
    return math.hypot(x_km + MU * L_KM, y_km)


def test_nominal_path_of_c1_starts_on_the_parking_orbit_and_ends_on_the_return_shell() -> None:
    elite = evaluate_unit(to_unit(C1))
    assert elite.feasible
    check, path, perilune = nominal_path(elite, index=0)
    assert check.passed and check.perilune_diff_km == 0.0
    rounding = 0.5 * 10.0**-NOMINAL_SIMPLIFICATION.km_decimals * math.sqrt(2.0)
    assert 2 < path.points <= NOMINAL_SIMPLIFICATION.max_points
    assert abs(_earth_distance(path.x_km[0], path.y_km[0]) - PARKING_RADIUS_KM) <= rounding
    assert abs(_earth_distance(path.x_km[-1], path.y_km[-1]) - RETURN_SHELL_KM) <= rounding
    assert path.t_days[0] == 0.0
    assert path.t_days[-1] == pytest.approx(elite.return_time_days, abs=1e-5)
    assert all(t1 > t0 for t0, t1 in pairwise(path.t_days))
    # The perilune is a kept vertex at R_moon + h_p from the Moon centre.
    moon_d = math.hypot(perilune["x_km"] - (1.0 - MU) * L_KM, perilune["y_km"])
    assert abs(moon_d - MOON_RADIUS_KM - elite.perilune_altitude_km) <= rounding
    assert perilune["t_days"] in path.t_days


def test_nominal_path_refuses_a_record_it_does_not_reproduce() -> None:
    elite = evaluate_unit(to_unit(C1))
    shifted = replace(elite, perilune_altitude_km=elite.perilune_altitude_km + 0.02)
    with pytest.raises(ValueError, match="does not reproduce"):
        nominal_path(shifted, index=0)


def test_fan_uncorrected_feasibility_matches_solution_robustness() -> None:
    """The stored `uncorrected_feasible` rows come from `solution_robustness`; the fan's
    recomputation must agree draw by draw (non-protocol draws: seed 9002, T_E scales)."""
    elite = evaluate_unit(to_unit(C1))
    draws = [ErrorDraw(0.0, 0.0, 0.0), *error_draws(replace(T_E, draws=4, seed=9002))]
    rob = solution_robustness(
        elite, draws, executions=(GATES_PRIMARY,), normals=np.zeros((len(draws), 2))
    )
    fans = [fan_path(elite, d) for d in draws]
    assert [f.uncorrected_feasible for f in fans] == [d.uncorrected_feasible for d in rob.draws]
    assert fans[0].uncorrected_feasible and fans[0].outcome == "feasible"
    for fan in fans:
        p = fan.path
        assert 2 < p.points <= FAN_SIMPLIFICATION.max_points
        assert abs(_earth_distance(p.x_km[0], p.y_km[0]) - PARKING_RADIUS_KM) <= 1.0
        assert all(t1 > t0 for t0, t1 in pairwise(p.t_days))
        if fan.outcome == "feasible":
            assert abs(_earth_distance(p.x_km[-1], p.y_km[-1]) - RETURN_SHELL_KM) <= 1.0


# --- the published run ----------------------------------------------------------------------

ATLAS_KEYS: Final = {
    "schema_version", "run_id", "frame", "length_unit", "time_unit", "frame_definition", "mu",
    "length_scale_km", "time_scale_seconds", "earth", "moon", "parking_orbit_radius_km",
    "return_shell_radius_km", "perilune_band_km", "gamma_target_deg", "seed", "seed_choice",
    "main_seeds", "analysis_run_id", "budget_m_s", "errors", "draws", "gates",
    "survival_definition", "grid", "h1", "contrasts", "simplification", "fan_simplification",
    "arms",
}  # fmt: skip
ARM_KEYS: Final = {
    "arm", "config_index", "source_run_id", "robustness_run_id", "fan_path", "summary",
    "anytime_coverage", "seed_values", "seed_anytime_coverage", "elites",
}  # fmt: skip
ELITE_KEYS: Final = {
    "cell", "index", "theta_deg", "delta_v_km_s", "alpha_deg", "perilune_altitude_km",
    "lunar_encounter_time_days", "return_time_days", "entry_flight_path_angle_deg",
    "objective_deg", "survival", "uncorrected_feasible_draws", "perilune", "path",
}  # fmt: skip
FAN_KEYS: Final = {
    "schema_version", "run_id", "arm", "frame", "length_unit", "time_unit", "errors",
    "error_seed", "budget_m_s", "gates", "error_draws", "elites",
}  # fmt: skip
DRAW_KEYS: Final = {"outcome", "uncorrected_feasible", "survived", "t_days", "x_km", "y_km"}
OUTCOMES: Final = {
    "feasible", "numerical_failure", "invalid_start", "moon_collision", "earth_collision",
    "no_return_within_t_max", "return_before_lunar_encounter", "perilune_out_of_band",
}  # fmt: skip


def _check_path(path: dict[str, Any], cap: int) -> None:
    assert set(path) == {"t_days", "x_km", "y_km"}
    n = len(path["t_days"])
    assert 2 <= n <= cap and len(path["x_km"]) == n == len(path["y_km"])
    assert all(t1 > t0 for t0, t1 in pairwise(path["t_days"]))
    assert all(math.isfinite(v) for k in ("t_days", "x_km", "y_km") for v in path[k])


@pytest.mark.skipif(not PUBLISHED.is_dir(), reason="the demo run is not published yet")
def test_published_demo_run_matches_the_contract() -> None:
    manifest = json.loads((PUBLISHED / "manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["run_id"] == "gravity-m4-demo-atlas-v1"
    assert manifest["status"] == "computed_unvalidated"
    assert manifest["source"]["git_dirty"] is False
    assert manifest["m4"]["stage"] == "demo_export" and manifest["m4"]["methods"] == []
    for check in manifest["validation"]["checks"]:
        assert check["passed"] and check["observed"] <= check["threshold"]
    kinds = {a["path"]: a["kind"] for a in manifest["artifacts"]}
    for a in manifest["artifacts"]:
        data = (PUBLISHED / a["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == a["sha256"]
    assert kinds.pop("atlas.json") == "atlas" and set(kinds.values()) == {"error_fan"}

    atlas = json.loads((PUBLISHED / "atlas.json").read_text())
    assert set(atlas) == ATLAS_KEYS and atlas["run_id"] == manifest["run_id"]
    reachable = {tuple(c) for c in atlas["grid"]["reachable_cells"]}
    assert len(reachable) == 32
    assert [a["arm"] for a in atlas["arms"]] == ["D", "MS", "MBH", "DE", "CC", "T", "T2"]
    draws = atlas["draws"]
    for arm in atlas["arms"]:
        assert set(arm) == ARM_KEYS and kinds[arm["fan_path"]] == "error_fan"
        fan = json.loads((PUBLISHED / arm["fan_path"]).read_text())
        assert set(fan) == FAN_KEYS and fan["arm"] == arm["arm"]
        assert len(fan["error_draws"]) == draws
        assert [(e["cell"], e["index"]) for e in fan["elites"]] == [
            (e["cell"], e["index"]) for e in arm["elites"]
        ]
        assert len(arm["elites"]) == arm["seed_values"]["elites"]
        for elite, fan_elite in zip(arm["elites"], fan["elites"], strict=True):
            assert set(elite) == ELITE_KEYS and tuple(elite["cell"]) in reachable
            _check_path(elite["path"], NOMINAL_SIMPLIFICATION.max_points)
            assert len(fan_elite["draws"]) == draws
            for d in fan_elite["draws"]:
                assert set(d) == DRAW_KEYS and d["outcome"] in OUTCOMES
                _check_path({k: d[k] for k in ("t_days", "x_km", "y_km")}, 120)
                assert d["survived"] or not d["uncorrected_feasible"]
                assert d["uncorrected_feasible"] == (d["outcome"] == "feasible")
            free = sum(d["uncorrected_feasible"] for d in fan_elite["draws"])
            assert free == elite["uncorrected_feasible_draws"]
            survived = sum(d["survived"] for d in fan_elite["draws"])
            assert elite["survival"] == pytest.approx(survived / draws, abs=1e-12)
