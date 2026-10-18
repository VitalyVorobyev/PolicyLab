"""Export the validated M1 reference run (Earth-Moon L1 Lyapunov orbit).

Usage::

    uv run python -m phasespace.gravity.export [--runs-dir runs] [--run-id <run_id>]

Writes ``manifest.json`` and ``trajectory.json`` (schema v1, docs/data-contract.md) into
the new directory ``<runs-dir>/<run-id>``. Files are written into a hidden temporary
sibling directory that is renamed into place only when complete; an existing run
directory is never reused or overwritten. The orbit is computed without catalog data
except the x-axis crossing x0 that selects the family member (linear guess,
continuation, differential correction for vy0 and the period), and is then compared
with the JPL catalog entry; the run is ``validated`` only if every recorded check
passes, otherwise ``computed_unvalidated``.

Every check is satisfied when ``observed <= threshold`` (and observed is finite).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import scipy

from phasespace.gravity.dynamics import (
    FloatArray,
    collinear_lagrange_points,
    jacobi_constant,
    primary_distances,
)
from phasespace.gravity.periodic import (
    CORRECTOR_SOLVER,
    SymmetricOrbit,
    continue_lyapunov,
    monodromy_matrix,
    stability_index,
)
from phasespace.gravity.propagation import (
    DEFAULT_MAX_EVALUATIONS,
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
EXPERIMENT_ID: Final = "cr3bp-l1-lyapunov-reference-v2"
DEFAULT_RUN_ID: Final = "gravity-l1-lyapunov-reference-v2"
"""Default run id. v1 was produced before the M1 numerics-review fixes; do not publish it."""
DEFAULT_RUNS_DIR: Final = Path("runs")
GENERATOR: Final = "phasespace.gravity.export"
REPOSITORY: Final = "VitalyVorobyev/PolicyLab"
FRAME: Final = "barycentric_synodic"
UNITS: Final = "normalized_cr3bp"
STATE_ORDER: Final = ("t", "x", "y", "vx", "vy")

SAMPLE_INTERVALS: Final = 1000
"""Uniform resampling: 1000 intervals over one period (1001 samples)."""
CONTINUATION_STEPS: Final = 10
JACOBI_DRIFT_PERIODS: Final = 3
TIGHT_SOLVER: Final = SolverSettings(method="DOP853", rtol=1e-13, atol=1e-13)
"""Same method as the exported run with 10x tighter tolerances."""
INDEPENDENT_SOLVER: Final = SolverSettings(method="LSODA", rtol=1e-13, atol=1e-13)
"""Independent integrator (ODEPACK LSODA, Adams/BDF multistep) for the sample error bound."""
MONODROMY_SOLVER: Final = CORRECTOR_SOLVER

# Thresholds (normalized CR3BP units unless stated). Rationale also in
# docs/studies/gravity-atlas/README.md ("M1 implementation notes").
CORRECTOR_TOLERANCE: Final = 1e-11
PERIODICITY_THRESHOLD: Final = 1e-9
JACOBI_DRIFT_THRESHOLD: Final = 1e-10
# Error bound of the exported samples. The run uses rtol = atol = 1e-12 and the orbit
# amplifies errors by lambda_max ~ 1.8e3 per period, so ~1e-12 * 1.8e3 ~ 2e-9 is the
# expected worst case; 1e-9 (0.39 m at the 389 703 km catalog length unit) is far below
# any plotted scale and ~50x above the observed 1.6e-11 (tight) and 2.2e-11 (LSODA).
SAMPLE_ERROR_THRESHOLD: Final = 1e-9
# Catalog agreement. The catalog lists vy0 and the period to 17 significant digits; the
# corrector stops at |vx| < 1e-11 at the half period, which leaves vy0 uncertain by
# |vx| / |d vx / d vy0| < 2e-12 here (|d vx / d vy0| ~ 6.8). 1e-10 leaves margin for the catalog's own
# convergence while still resolving 1e-7 m/s in vy0 and 0.04 ms in the period (observed:
# 4e-14 to 7e-14). tests/gravity/test_gravity_periodic.py uses the same values.
CATALOG_STATE_THRESHOLD: Final = 1e-10
CATALOG_PERIOD_THRESHOLD: Final = 1e-10
CATALOG_JACOBI_THRESHOLD: Final = 1e-10
# Relative. Monodromy entries are O(lambda_max) and integrated at 1e-13, and the catalog
# lists nu to 15 significant digits; observed 4e-13.
STABILITY_INDEX_REL_THRESHOLD: Final = 1e-9

_RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class Check:
    """A validation check; passes when observed is finite and observed <= threshold."""

    name: str
    observed: float
    threshold: float
    definition: str

    @property
    def passed(self) -> bool:
        return math.isfinite(self.observed) and self.observed <= self.threshold

    def to_json(self) -> dict[str, Any]:
        """JSON form; a non-finite observed value (not representable in JSON) becomes null."""
        return {
            "name": self.name,
            "observed": self.observed if math.isfinite(self.observed) else None,
            "threshold": self.threshold,
            "passed": self.passed,
            "definition": self.definition,
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


def _solver_json(solver: SolverSettings) -> dict[str, Any]:
    return {"name": solver.method, "rtol": solver.rtol, "atol": solver.atol}


def _max_abs_difference(a: PropagationResult, b: PropagationResult, times: FloatArray) -> float:
    return float(np.max(np.abs(a.sample(times) - b.sample(times))))


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
    tight = propagate(state0, orbit.period, system, solver=TIGHT_SOLVER)
    independent = propagate(state0, orbit.period, system, solver=INDEPENDENT_SOLVER)
    long_run = propagate(state0, JACOBI_DRIFT_PERIODS * orbit.period, system, solver=DEFAULT_SOLVER)
    runs = (one_period, tight, independent, long_run)
    lambda_max, nu = stability_index(
        monodromy_matrix(mu, state0, orbit.period, solver=MONODROMY_SOLVER)
    )

    times = np.linspace(0.0, one_period.t_final, sample_intervals + 1)
    states = one_period.sample(times)
    samples = np.column_stack([times, states])

    long_times = np.linspace(0.0, long_run.t_final, JACOBI_DRIFT_PERIODS * sample_intervals + 1)
    long_jacobi = jacobi_constant(long_run.sample(long_times), mu)
    sample_jacobi = jacobi_constant(states, mu)
    r1, r2 = primary_distances(states, mu)

    collision_events = sum(len(r.events) for r in runs) + sum(
        r.termination is not Termination.HORIZON for r in runs
    )
    jpl = "the JPL catalog entry in validation.reference"
    checks = (
        Check(
            "periodicity_max_abs_state_error_one_period",
            float(np.max(np.abs(one_period.final_state - state0))),
            PERIODICITY_THRESHOLD,
            "max |s(T) - s(0)| over (x, y, vx, vy) after one computed period T",
        ),
        Check(
            "jacobi_max_abs_drift_exported_samples",
            float(np.max(np.abs(sample_jacobi - jacobi0))),
            JACOBI_DRIFT_THRESHOLD,
            "max |C(s_i) - C(s_0)| over the exported samples",
        ),
        Check(
            f"jacobi_max_abs_drift_{JACOBI_DRIFT_PERIODS}_periods",
            float(np.max(np.abs(long_jacobi - jacobi0))),
            JACOBI_DRIFT_THRESHOLD,
            f"max |C(t) - C(0)| at {JACOBI_DRIFT_PERIODS * sample_intervals + 1} uniform "
            f"times over {JACOBI_DRIFT_PERIODS} periods; a conservation check only: it does "
            "not detect departure from the unstable orbit",
        ),
        Check(
            "tight_tolerance_max_abs_state_difference_samples",
            _max_abs_difference(one_period, tight, times),
            SAMPLE_ERROR_THRESHOLD,
            "max |s - s_ref| over (x, y, vx, vy) at the exported sample times; s_ref: "
            f"{TIGHT_SOLVER.method} rtol = atol = {TIGHT_SOLVER.rtol:g}",
        ),
        Check(
            "independent_integrator_max_abs_state_difference_samples",
            _max_abs_difference(one_period, independent, times),
            SAMPLE_ERROR_THRESHOLD,
            "max |s - s_ref| over (x, y, vx, vy) at the exported sample times; s_ref: "
            f"{INDEPENDENT_SOLVER.method} rtol = atol = {INDEPENDENT_SOLVER.rtol:g}",
        ),
        Check(
            "jpl_catalog_abs_diff_vy0",
            abs(orbit.vy0 - reference.vy0),
            CATALOG_STATE_THRESHOLD,
            f"|vy0 - vy0_catalog| against {jpl}; vy0 is computed without catalog velocities",
        ),
        Check(
            "jpl_catalog_abs_diff_period",
            abs(orbit.period - reference.period),
            CATALOG_PERIOD_THRESHOLD,
            f"|T - T_catalog| against {jpl}",
        ),
        Check(
            "jpl_catalog_abs_diff_jacobi",
            abs(jacobi0 - reference.jacobi),
            CATALOG_JACOBI_THRESHOLD,
            f"|C(s0) - C_catalog| against {jpl}. Not independent of jpl_catalog_abs_diff_vy0: "
            "x0 equals the catalog value and y0 = vx0 = 0, so the difference is "
            "|2 vy0 dvy0 + dvy0^2| up to the catalog's rounding of C; it checks only the "
            "Jacobi-constant convention",
        ),
        Check(
            "jpl_catalog_rel_diff_stability_index",
            abs(nu - reference.stability_index) / reference.stability_index,
            STABILITY_INDEX_REL_THRESHOLD,
            "|nu - nu_catalog| / nu_catalog with nu = (lambda_max + 1/lambda_max)/2 from the "
            f"computed monodromy matrix ({MONODROMY_SOLVER.method} rtol = atol = "
            f"{MONODROMY_SOLVER.rtol:g})",
        ),
        Check(
            "non_horizon_terminations_or_collision_events",
            float(collision_events),
            0.0,
            "collision events plus non-horizon terminations over the exported, tight, "
            "independent-integrator and multi-period propagations",
        ),
        Check(
            "non_finite_sample_values",
            float(np.count_nonzero(~np.isfinite(samples))),
            0.0,
            "count of non-finite values in trajectory samples",
        ),
        Check(
            "non_increasing_sample_times",
            float(np.count_nonzero(np.diff(times) <= 0.0)),
            0.0,
            "count of consecutive sample times with t[i+1] <= t[i]",
        ),
    )

    step = float(times[1] - times[0])
    convention = system.length_scale_convention
    km = f"km via length_scale_km = {system.length_scale_km} km ({convention})"
    km_s = f"km/s via velocity_scale_km_s = {system.velocity_scale_km_s} km/s ({convention})"
    days = f"time_scale_seconds = {system.time_scale_s} s, 1 day = 86400 s"
    metrics: dict[str, dict[str, Any]] = {
        "period_normalized": _metric(orbit.period, UNITS, "orbit period"),
        "period_days": _metric(time_to_days(orbit.period, system), "day", f"orbit period; {days}"),
        "jacobi_constant": _metric(jacobi0, UNITS, "Jacobi constant of the initial state"),
        "initial_x_km": _metric(
            length_to_km(orbit.x0, system),
            "km",
            f"barycentric x of the initial perpendicular x-axis crossing; {km}",
        ),
        "initial_vy_km_s": _metric(
            velocity_to_km_s(orbit.vy0, system),
            "km/s",
            f"rotating-frame vy at the initial state; {km_s}",
        ),
        "x_extent_km": _metric(
            length_to_km(float(np.ptp(states[:, 0])), system),
            "km",
            f"max(x) - min(x) of samples; {km}",
        ),
        "y_extent_km": _metric(
            length_to_km(float(np.ptp(states[:, 1])), system),
            "km",
            f"max(y) - min(y) of samples; {km}",
        ),
        "min_earth_distance_km": _metric(
            length_to_km(float(np.min(r1)), system),
            "km",
            f"min Earth-center distance of samples; {km}",
        ),
        "min_moon_distance_km": _metric(
            length_to_km(float(np.min(r2)), system),
            "km",
            f"min Moon-center distance of samples; {km}",
        ),
        "sample_count": _metric(int(samples.shape[0]), "count", "rows in trajectory.json"),
        "sample_step_normalized": _metric(step, UNITS, "uniform time step between samples"),
        "sample_step_hours": _metric(
            time_to_days(step, system) * 24.0,
            "hour",
            f"uniform time step between samples; {days}",
        ),
        "monodromy_lambda_max": _metric(
            lambda_max,
            "dimensionless",
            "largest eigenvalue modulus of the computed monodromy matrix: growth per period "
            "of a perturbation along the unstable direction",
        ),
        "monodromy_stability_index": _metric(
            nu,
            "dimensionless",
            "(lambda_max + 1/lambda_max)/2 from the computed monodromy matrix",
        ),
        "jpl_stability_index": _metric(
            reference.stability_index,
            "dimensionless",
            "catalog stability index nu = (lambda_max + 1/lambda_max)/2 (> 1: unstable)",
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


def environment_info() -> dict[str, str]:
    """Interpreter, numerical-library and platform versions of the generating process."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


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
    lambda_max = computation.metrics["monodromy_lambda_max"]["value"]
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
        f"{system.length_scale_km} km, not 384400 km; model.length_scale_convention) so the "
        "orbit is directly comparable with the catalog; km, day and km/s values depend on "
        "that convention.",
        "Samples are a uniform-time resampling of the DOP853 dense-output interpolant for "
        "playback; accepted solver steps are not exported.",
        f"The orbit is strongly unstable: the largest monodromy eigenvalue is "
        f"{lambda_max:.1f} (catalog stability index (lambda + 1/lambda)/2 = "
        f"{reference.stability_index}), so a small error along the unstable direction grows "
        "by that factor each period. The trajectory accuracy claim covers the exported "
        "single period only; Jacobi drift stays small even after a longer propagation has "
        "left the orbit, so it does not certify multi-period trajectories.",
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
        "environment": environment_info(),
        "model": {
            "name": "planar-cr3bp",
            "coordinate_frame": FRAME,
            "units": UNITS,
            "mu": system.mu,
            "length_scale_km": system.length_scale_km,
            "length_scale_convention": system.length_scale_convention,
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
            **_solver_json(DEFAULT_SOLVER),
            "dense_output": True,
            "max_evaluations": DEFAULT_MAX_EVALUATIONS,
            "collision_events": {
                "earth_radius_km": system.earth_radius_km,
                "moon_radius_km": system.moon_radius_km,
                "function": "distance_minus_radius",
                "direction": "decreasing",
                "terminal": True,
                "grazing_detection": "periapsis_below_radius",
                "start_inside": "not_propagated",
            },
            "comparisons": [
                {"role": "tight_tolerance", **_solver_json(TIGHT_SOLVER)},
                {"role": "independent_integrator", **_solver_json(INDEPENDENT_SOLVER)},
            ],
            "monodromy": _solver_json(MONODROMY_SOLVER),
            "differential_correction": {
                **_solver_json(CORRECTOR_SOLVER),
                "half_period_abs_vx_tolerance": CORRECTOR_TOLERANCE,
                "continuation_steps": CONTINUATION_STEPS,
                "initial_guess": "linear_l1_lyapunov_then_x0_continuation",
                "convergence": {
                    "iterations_final_member": computation.orbit.iterations,
                    "half_period_abs_vx": computation.orbit.half_period_abs_vx,
                },
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
                "stability_index": reference.stability_index,
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
    """Create `out_dir` (must not exist) with manifest.json and trajectory.json.

    Both files are written into a hidden temporary sibling directory that is renamed to
    `out_dir` only when complete, so a reader never sees a partial run. Raises
    FileExistsError if `out_dir` exists; the temporary directory is removed on failure.
    """
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
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.partial-", dir=out_dir.parent))
    try:
        staging.chmod(0o755)
        with (staging / "trajectory.json").open("xb") as fh:
            fh.write(trajectory_bytes)
        with (staging / "manifest.json").open("xb") as fh:
            fh.write(_dumps(manifest))
        # Re-check right before the rename: POSIX rename replaces an existing *empty*
        # directory, and a non-empty one makes it fail; neither may happen silently.
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError(f"{out_dir} already exists; runs are never overwritten")
        try:
            os.rename(staging, out_dir)
        except OSError as exc:
            if out_dir.exists():
                raise FileExistsError(
                    f"{out_dir} already exists; runs are never overwritten"
                ) from exc
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phasespace.gravity.export",
        description="Export the validated Earth-Moon L1 Lyapunov reference run.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"parent directory of run directories (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"name of the new run directory (default: {DEFAULT_RUN_ID})",
    )
    args = parser.parse_args(argv)
    runs_dir: Path = args.runs_dir
    run_id: str = args.run_id
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        parser.error(f"--run-id {run_id!r} must match {_RUN_ID_PATTERN.pattern}")
    out_dir = runs_dir / run_id
    if out_dir.exists():
        parser.error(f"{out_dir} already exists; runs are never overwritten")
    commit, dirty = git_source()
    computation = compute_reference()
    manifest = write_run(
        out_dir, computation, git_commit=commit, git_dirty=dirty, created_at=datetime.now(UTC)
    )
    failed = [c.name for c in computation.checks if not c.passed]
    print(f"wrote {out_dir} status={manifest['status']} git_dirty={dirty}")
    for check in computation.checks:
        mark = "ok  " if check.passed else "FAIL"
        print(f"  {mark} {check.name}: {check.observed:.3e} <= {check.threshold:.1e}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
