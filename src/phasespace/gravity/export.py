"""Export the validated M1 reference run (Earth-Moon L1 Lyapunov orbit).

Usage::

    uv run python -m phasespace.gravity.export --out runs/<run_id>

Writes ``manifest.json`` and ``trajectory.json`` (schema v1, docs/data-contract.md) into
a new directory; an existing directory is never reused or overwritten. The orbit is
computed without catalog data except the x-axis crossing x0 that selects the family
member (linear guess, continuation, differential correction for vy0 and the period), and
is then compared with the JPL catalog entry; the run is ``validated`` only if every
recorded check passes, otherwise ``computed_unvalidated``.

Every check is satisfied when ``observed <= threshold`` (and observed is finite).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np

from phasespace.gravity.dynamics import (
    FloatArray,
    collinear_lagrange_points,
    jacobi_constant,
    primary_distances,
)
from phasespace.gravity.periodic import CORRECTOR_SOLVER, SymmetricOrbit, continue_lyapunov
from phasespace.gravity.propagation import (
    DEFAULT_SOLVER,
    PropagationResult,
    SolverSettings,
    Termination,
    propagate,
)
from phasespace.gravity.reference import (
    JPL_EARTH_MOON,
    JPL_L1_LYAPUNOV_REFERENCE,
    JPL_PERIODIC_ORBITS_URL,
    JPL_RETRIEVED_ON,
    CatalogOrbit,
)
from phasespace.gravity.system import EarthMoonSystem
from phasespace.gravity.units import length_to_km, time_to_days, velocity_to_km_s

SCHEMA_VERSION: Final = 1
STUDY_ID: Final = "gravity-atlas"
EXPERIMENT_ID: Final = "cr3bp-l1-lyapunov-reference-v1"
GENERATOR: Final = "phasespace.gravity.export"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"
FRAME: Final = "barycentric_synodic"
UNITS: Final = "normalized_cr3bp"
STATE_ORDER: Final = ("t", "x", "y", "vx", "vy")

SAMPLE_INTERVALS: Final = 1000
"""Uniform resampling: 1000 intervals over one period (1001 samples)."""
CONTINUATION_STEPS: Final = 10
JACOBI_DRIFT_PERIODS: Final = 3
COMPARISON_SOLVER: Final = SolverSettings(method="DOP853", rtol=1e-10, atol=1e-10)

# Thresholds (all normalized CR3BP units). Rationale in docs/studies/gravity-atlas/README.md.
CORRECTOR_TOLERANCE: Final = 1e-11
PERIODICITY_THRESHOLD: Final = 1e-9
JACOBI_DRIFT_THRESHOLD: Final = 1e-10
TOLERANCE_AGREEMENT_THRESHOLD: Final = 1e-7
CATALOG_STATE_THRESHOLD: Final = 1e-9
CATALOG_PERIOD_THRESHOLD: Final = 1e-9
CATALOG_JACOBI_THRESHOLD: Final = 1e-10

_RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class Check:
    """A validation check; passes when observed is finite and observed <= threshold."""

    name: str
    observed: float
    threshold: float

    @property
    def passed(self) -> bool:
        return math.isfinite(self.observed) and self.observed <= self.threshold

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed": self.observed,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def run_status(checks: Sequence[Check]) -> str:
    """'validated' only when there is at least one check and all pass."""
    if checks and all(c.passed for c in checks):
        return "validated"
    return "computed_unvalidated"


@dataclass(frozen=True, slots=True)
class ReferenceComputation:
    """Everything computed for the reference run."""

    system: EarthMoonSystem
    reference: CatalogOrbit
    orbit: SymmetricOrbit
    one_period: PropagationResult
    samples: FloatArray
    checks: tuple[Check, ...]
    metrics: dict[str, dict[str, Any]]


def _metric(value: float | int, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def compute_reference(sample_intervals: int = SAMPLE_INTERVALS) -> ReferenceComputation:
    """Compute, sample and check the reference L1 Lyapunov orbit."""
    system = JPL_EARTH_MOON
    reference = JPL_L1_LYAPUNOV_REFERENCE
    mu = system.mu
    l1_x = collinear_lagrange_points(mu)[0]
    orbit = continue_lyapunov(
        mu,
        1,
        reference.x0 - l1_x,
        steps=CONTINUATION_STEPS,
        tolerance=CORRECTOR_TOLERANCE,
        solver=CORRECTOR_SOLVER,
    )
    state0 = orbit.state0
    jacobi0 = float(jacobi_constant(state0, mu))

    one_period = propagate(state0, orbit.period, system, solver=DEFAULT_SOLVER)
    comparison = propagate(state0, orbit.period, system, solver=COMPARISON_SOLVER)
    long_run = propagate(state0, JACOBI_DRIFT_PERIODS * orbit.period, system, solver=DEFAULT_SOLVER)

    times = np.linspace(0.0, one_period.t_final, sample_intervals + 1)
    states = one_period.sample(times)
    samples = np.column_stack([times, states])

    long_times = np.linspace(0.0, long_run.t_final, JACOBI_DRIFT_PERIODS * sample_intervals + 1)
    long_jacobi = jacobi_constant(long_run.sample(long_times), mu)
    sample_jacobi = jacobi_constant(states, mu)
    r1, r2 = primary_distances(states, mu)

    collision_events = sum(len(r.events) for r in (one_period, comparison, long_run)) + sum(
        r.termination is not Termination.HORIZON for r in (one_period, comparison, long_run)
    )

    checks = (
        Check("corrector_half_period_abs_vx", orbit.half_period_abs_vx, CORRECTOR_TOLERANCE),
        Check(
            "periodicity_max_abs_state_error_one_period",
            float(np.max(np.abs(one_period.final_state - state0))),
            PERIODICITY_THRESHOLD,
        ),
        Check(
            "jacobi_max_abs_drift_exported_samples",
            float(np.max(np.abs(sample_jacobi - jacobi0))),
            JACOBI_DRIFT_THRESHOLD,
        ),
        Check(
            f"jacobi_max_abs_drift_{JACOBI_DRIFT_PERIODS}_periods",
            float(np.max(np.abs(long_jacobi - jacobi0))),
            JACOBI_DRIFT_THRESHOLD,
        ),
        Check(
            "tolerance_sets_max_abs_final_state_difference",
            float(np.max(np.abs(one_period.final_state - comparison.final_state))),
            TOLERANCE_AGREEMENT_THRESHOLD,
        ),
        Check("jpl_catalog_abs_diff_vy0", abs(orbit.vy0 - reference.vy0), CATALOG_STATE_THRESHOLD),
        Check(
            "jpl_catalog_abs_diff_period",
            abs(orbit.period - reference.period),
            CATALOG_PERIOD_THRESHOLD,
        ),
        Check(
            "jpl_catalog_abs_diff_jacobi", abs(jacobi0 - reference.jacobi), CATALOG_JACOBI_THRESHOLD
        ),
        Check("non_horizon_terminations_or_collision_events", float(collision_events), 0.0),
        Check("non_finite_sample_values", float(np.count_nonzero(~np.isfinite(samples))), 0.0),
        Check("non_increasing_sample_times", float(np.count_nonzero(np.diff(times) <= 0.0)), 0.0),
    )

    step = float(times[1] - times[0])
    metrics: dict[str, dict[str, Any]] = {
        "period_normalized": _metric(orbit.period, UNITS, "orbit period"),
        "period_days": _metric(
            time_to_days(orbit.period, system), "day", "orbit period (1 day = 86400 s)"
        ),
        "jacobi_constant": _metric(jacobi0, UNITS, "Jacobi constant of the initial state"),
        "initial_x_km": _metric(
            length_to_km(orbit.x0, system), "km", "x of the initial perpendicular x-axis crossing"
        ),
        "initial_vy_km_s": _metric(
            velocity_to_km_s(orbit.vy0, system), "km/s", "rotating-frame vy at the initial state"
        ),
        "x_extent_km": _metric(
            length_to_km(float(np.ptp(states[:, 0])), system), "km", "max(x) - min(x) of samples"
        ),
        "y_extent_km": _metric(
            length_to_km(float(np.ptp(states[:, 1])), system), "km", "max(y) - min(y) of samples"
        ),
        "min_earth_distance_km": _metric(
            length_to_km(float(np.min(r1)), system), "km", "min Earth-center distance of samples"
        ),
        "min_moon_distance_km": _metric(
            length_to_km(float(np.min(r2)), system), "km", "min Moon-center distance of samples"
        ),
        "sample_count": _metric(int(samples.shape[0]), "count", "rows in trajectory.json"),
        "sample_step_normalized": _metric(step, UNITS, "uniform time step between samples"),
        "sample_step_hours": _metric(
            time_to_days(step, system) * 24.0, "hour", "uniform time step between samples"
        ),
        "corrector_iterations_final_member": _metric(
            orbit.iterations, "count", "Newton iterations for the final continuation member"
        ),
        "jpl_stability_index": _metric(
            reference.stability_index, "dimensionless", "catalog stability index (>1 unstable)"
        ),
    }
    return ReferenceComputation(
        system=system,
        reference=reference,
        orbit=orbit,
        one_period=one_period,
        samples=samples,
        checks=checks,
        metrics=metrics,
    )


def _git(*args: str) -> str:
    cwd = Path(__file__).resolve().parent
    completed = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout


def git_source() -> tuple[str, bool]:
    """Current commit SHA and whether the working tree has uncommitted or untracked changes."""
    commit = _git("rev-parse", "HEAD").strip()
    dirty = bool(_git("status", "--porcelain").strip())
    return commit, dirty


def _dumps(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build_documents(
    computation: ReferenceComputation,
    run_id: str,
    *,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
) -> tuple[dict[str, Any], bytes]:
    """Return (manifest without artifacts, trajectory bytes)."""
    system = computation.system
    reference = computation.reference
    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "frame": FRAME,
        "time_unit": UNITS,
        "state_order": list(STATE_ORDER),
        "samples": computation.samples.tolist(),
    }
    timestamp = created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    limitations = [
        "Planar circular restricted three-body model: no lunar eccentricity, out-of-plane "
        "motion, solar or other perturbations, or ephemeris effects.",
        "Constants follow the JPL periodic-orbit catalog (length unit "
        f"{system.length_scale_km} km, not 384400 km) so the orbit is directly comparable "
        "with the catalog; km, day and km/s values depend on that convention.",
        "Samples are a uniform-time resampling of the DOP853 dense-output interpolant for "
        "playback; accepted solver steps are not exported.",
        f"The orbit is strongly unstable (catalog stability index {reference.stability_index}); "
        "small errors grow by roughly that factor per period, so long propagations diverge.",
        "'validated' means the recorded numerical checks passed; it is not a new scientific "
        "result.",
    ]
    if git_dirty:
        limitations.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "study_id": STUDY_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": run_status(computation.checks),
        "created_at": timestamp,
        "source": {
            "repository": REPOSITORY,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generator": GENERATOR,
            "generator_version": version("phasespace"),
        },
        "model": {
            "name": "planar-cr3bp",
            "coordinate_frame": FRAME,
            "units": UNITS,
            "mu": system.mu,
            "length_scale_km": system.length_scale_km,
            "time_scale_seconds": system.time_scale_s,
            "velocity_scale_km_s": system.velocity_scale_km_s,
            "system_name": system.name,
            "constants_source": system.source,
            "primaries": {
                "earth": {"x": -system.mu, "y": 0.0, "radius_km": system.earth_radius_km},
                "moon": {"x": 1.0 - system.mu, "y": 0.0, "radius_km": system.moon_radius_km},
            },
            "assumptions": [
                "circular_primaries",
                "massless_spacecraft",
                "planar",
                "unit_angular_rate_counterclockwise",
            ],
        },
        "solver": {
            "name": DEFAULT_SOLVER.method,
            "rtol": DEFAULT_SOLVER.rtol,
            "atol": DEFAULT_SOLVER.atol,
            "dense_output": True,
            "collision_events": {
                "earth_radius_km": system.earth_radius_km,
                "moon_radius_km": system.moon_radius_km,
                "direction": "inward (distance decreasing), terminal",
            },
            "comparison": {
                "name": COMPARISON_SOLVER.method,
                "rtol": COMPARISON_SOLVER.rtol,
                "atol": COMPARISON_SOLVER.atol,
            },
            "differential_correction": {
                "name": CORRECTOR_SOLVER.method,
                "rtol": CORRECTOR_SOLVER.rtol,
                "atol": CORRECTOR_SOLVER.atol,
                "half_period_abs_vx_tolerance": CORRECTOR_TOLERANCE,
                "continuation_steps": CONTINUATION_STEPS,
                "initial_guess": "linear L1 Lyapunov approximation, then continuation in x0",
            },
        },
        "random_seed": None,
        "artifacts": [],
        "metrics": computation.metrics,
        "limitations": limitations,
        "validation": {
            "checks": [c.to_json() for c in computation.checks],
            "validated_by": f"{GENERATOR} automated checks (pass: observed <= threshold)",
            "validated_at": timestamp,
            "reference": {
                "source": "NASA/JPL Three-Body Periodic Orbits API",
                "url": JPL_PERIODIC_ORBITS_URL,
                "retrieved_on": JPL_RETRIEVED_ON,
                "family": reference.family,
                "libration_point": reference.libration_point,
                "row_index": reference.row_index,
                "x0": reference.x0,
                "vy0": reference.vy0,
                "period": reference.period,
                "jacobi": reference.jacobi,
            },
        },
    }
    return manifest, _dumps(trajectory)


def write_run(
    out_dir: Path,
    computation: ReferenceComputation,
    *,
    git_commit: str,
    git_dirty: bool,
    created_at: datetime,
) -> dict[str, Any]:
    """Create `out_dir` (must not exist) with manifest.json and trajectory.json."""
    run_id = out_dir.name
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r} must match {_RUN_ID_PATTERN.pattern}")
    manifest, trajectory_bytes = build_documents(
        computation, run_id, git_commit=git_commit, git_dirty=git_dirty, created_at=created_at
    )
    manifest["artifacts"] = [
        {
            "kind": "trajectory",
            "path": "trajectory.json",
            "sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
        }
    ]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)  # never reuse or overwrite an existing run
    with (out_dir / "trajectory.json").open("xb") as fh:
        fh.write(trajectory_bytes)
    with (out_dir / "manifest.json").open("xb") as fh:
        fh.write(_dumps(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phasespace.gravity.export",
        description="Export the validated Earth-Moon L1 Lyapunov reference run.",
    )
    parser.add_argument("--out", type=Path, required=True, help="new run directory to create")
    args = parser.parse_args(argv)
    out_dir: Path = args.out
    if out_dir.exists():
        parser.error(f"{out_dir} already exists; runs are never overwritten")
    commit, dirty = git_source()
    computation = compute_reference()
    manifest = write_run(
        out_dir, computation, git_commit=commit, git_dirty=dirty, created_at=datetime.now(UTC)
    )
    failed = [c.name for c in computation.checks if not c.passed]
    print(f"wrote {out_dir} status={manifest['status']}")
    for check in computation.checks:
        mark = "ok  " if check.passed else "FAIL"
        print(f"  {mark} {check.name}: {check.observed:.3e} <= {check.threshold:.1e}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
