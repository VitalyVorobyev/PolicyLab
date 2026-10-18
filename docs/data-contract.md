# Data Contract — Immutable Static Research Runs (v1)

## Objectives

Allow **scientific Python tools** to export results and **static React** to load, validate and inspect them without a server. File layout can change behind a loader, but schema fields and units should be explicit and versioned.

## Directory model (illustrative)

```text
web/public/research/
  catalog.json
  gravity-atlas/
    benchmark-a/
      manifest.json
      trajectory.json
```

Only small, curated public samples belong in the deployable site. Full research files may live under ignored local `runs/`, a release asset, or an object store. Source code and experiment configurations should remain separately versioned.

A run directory is named by its `run_id` and holds `manifest.json` plus the files listed in `artifacts`. Exporters write all files into a hidden temporary sibling directory and rename it into place only when complete, so a run directory is never partially written. An existing run directory is never reused or overwritten.

## Schema v1 (precise)

Schema v1 has not been published yet. The definition below is what `phasespace.gravity.export` emits and what a strict validator should accept. Fields marked `?` are optional; every other field is required. A validator must reject unknown `schema_version` values, missing required fields, wrong types, values outside the listed enums, `null` where it is not allowed, and non-finite numbers. It may ignore unknown extra fields, but producers must not emit fields that are not listed here. All numbers are finite JSON numbers. JSON cannot encode NaN or infinity, so a non-finite check value is written as `null` (see `Check`).

```ts
type RunStatus = "illustrative" | "computed_unvalidated" | "validated";
type SolverName = "DOP853" | "RK45" | "Radau" | "LSODA";
type LengthScaleConvention =
  | "nssdca_lunar_semimajor_axis"   // 384 400 km (NSSDCA Moon Fact Sheet)
  | "jpl_periodic_orbits_catalog";  // 389 703.264829278 km (JPL three-body catalog lunit)
type MetricUnit =
  | "normalized_cr3bp" | "km" | "km/s" | "day" | "hour" | "count" | "dimensionless"
  | "deg";                     // angle in degrees (added for M3)
type Termination =             // why a propagation stopped (phasespace.gravity.propagation)
  | "horizon" | "earth_collision" | "moon_collision" | "earth_return_shell"
  | "start_inside_earth" | "start_inside_moon" | "start_inside_return_shell"
  | "non_finite" | "evaluation_limit" | "solver_failure";

interface Manifest {
  schema_version: 1;
  run_id: string;            // /^[a-z0-9][a-z0-9._-]{0,127}$/, equals the run directory name
  study_id: string;          // e.g. "gravity-atlas"
  experiment_id: string;     // e.g. "cr3bp-l1-lyapunov-reference-v2"
  status: RunStatus;
  created_at: string;        // UTC, exactly "YYYY-MM-DDTHH:MM:SSZ"
  source: Source;
  environment: Environment;
  model: Cr3bpModel;         // planar CR3BP runs; other models will get their own type
  solver: Solver;
  random_seed: number | null;        // integer; null = the run uses no randomness
  artifacts: Artifact[];             // at least one
  metrics: Record<string, Metric>;   // keys /^[a-z0-9_]+$/; may be {}
  limitations: string[];             // at least one
  validation?: Validation;           // required when status === "validated"
  termination?: Termination;         // termination of the propagation that produced the
                                     // trajectory artifact (M3 runs; distinguishes a
                                     // return from the horizon)
}

interface Source {
  repository: string;          // "owner/name"
  git_commit: string;          // 40 lowercase hex characters
  git_dirty: boolean;          // true: uncommitted or untracked changes at export time
  generator: string;           // Python module, e.g. "phasespace.gravity.export"
  generator_version: string;   // installed package version
}

interface Environment {        // versions of the generating process
  python: string;              // e.g. "3.12.12"
  python_implementation: string;  // e.g. "CPython"
  numpy: string;
  scipy: string;
  platform: string;            // platform.platform()
  machine: string;             // platform.machine(), e.g. "arm64"
}

interface Cr3bpModel {
  name: "planar-cr3bp";
  coordinate_frame: "barycentric_synodic";  // origin at the barycenter, x toward the Moon,
                                            // rotating counterclockwise at unit rate
  units: "normalized_cr3bp";   // length = primaries' separation, time = 1 / angular rate
  mu: number;                  // m_moon / (m_earth + m_moon), in (0, 0.5]
  length_scale_km: number;     // > 0; km per normalized length unit
  length_scale_convention: LengthScaleConvention;
  time_scale_seconds: number;  // > 0; seconds per normalized time unit
  velocity_scale_km_s: number; // = length_scale_km / time_scale_seconds
  system_name?: string;        // identifier of the constant set
  constants_source?: string;   // human-readable provenance of the constants
  primaries: {
    earth: Primary;            // x = -mu, y = 0
    moon: Primary;             // x = 1 - mu, y = 0
  };
  assumptions: string[];       // e.g. "circular_primaries", "massless_spacecraft", "planar",
                               // "unit_angular_rate_counterclockwise"
}

interface Primary {
  x: number;                   // normalized
  y: number;                   // normalized
  radius_km: number;           // > 0; radius used for collision detection
}

interface SolverSettings {
  name: SolverName;            // scipy.integrate.solve_ivp method
  rtol: number;                // > 0
  atol: number;                // > 0
}

interface Solver extends SolverSettings {  // settings of the propagation that produced the artifacts
  dense_output: boolean;
  max_evaluations: number;     // integer >= 1; right-hand-side budget per propagation
  collision_events: CollisionEvents;
  return_shell?: ReturnShell;  // M3: terminal Earth return shell, when enabled
  comparisons?: Comparison[];  // extra propagations used only by validation checks
  monodromy?: SolverSettings;  // integration of the variational equations over one period
  differential_correction?: DifferentialCorrection;
}

interface CollisionEvents {
  earth_radius_km: number | null;          // null = Earth collisions disabled
  moon_radius_km: number | null;           // null = Moon collisions disabled
  function: "distance_minus_radius";       // g = |r - r_body| - radius
  direction: "decreasing";                 // only inward crossings (g: + -> -) fire
  terminal: true;                          // the first collision stops the propagation
  grazing_detection: "periapsis_below_radius";
      // a periapsis inside the radius counts as a collision at the first surface
      // crossing before it, even if g has the same sign at both ends of a solver step
  start_inside: "not_propagated";          // g(t0) < 0: nothing is propagated
}

interface ReturnShell {        // terminal inward crossing of a sphere about the Earth centre
  radius_km: number;           // > earth_radius_km
  altitude_km: number;         // radius_km - earth radius (e.g. 121.92 km entry interface)
  function: "distance_minus_radius";
  direction: "decreasing";     // inbound crossings only; an outward crossing never fires
  terminal: true;              // ends the propagation (termination "earth_return_shell")
  grazing_detection: "periapsis_below_radius";
}

interface Comparison extends SolverSettings {
  role: "tight_tolerance" | "independent_integrator";
}

interface DifferentialCorrection extends SolverSettings {
  half_period_abs_vx_tolerance: number;    // stop when |vx| < this at the half-period crossing
  continuation_steps: number;              // integer >= 1
  initial_guess: "linear_l1_lyapunov_then_x0_continuation";
  convergence: {                           // a convergence record, not a validation check
    iterations_final_member: number;       // integer >= 1
    half_period_abs_vx: number;            // < half_period_abs_vx_tolerance by construction
  };
}

interface Artifact {
  kind: "trajectory" | "sample_table";   // "sample_table" added for M3 maps and searches
  path: string;                // relative to the run directory; no "/", "\\" or ".." segments
  sha256: string;              // 64 lowercase hex characters of the file's exact bytes
}

interface Metric {
  value: number;
  unit: MetricUnit;
  definition: string;          // what was computed; km, km/s and day values name the scale
                               // and length_scale_convention they were converted with
}

interface Validation {
  checks: Check[];             // at least one; status "validated" requires every passed === true
  validated_by: string;
  validated_at: string;        // same format as created_at
  reference?: CatalogReference;
}

interface Check {
  name: string;                // /^[a-z0-9_]+$/
  observed: number | null;     // null: the computation produced a non-finite value
  threshold: number;
  passed: boolean;             // === (observed !== null && observed <= threshold)
  definition: string;          // what observed measures, including any dependence on
                               // other checks
}

interface CatalogReference {   // published values the run is compared with
  source: string;
  url: string;                 // https URL
  retrieved_on: string;        // "YYYY-MM-DD"
  family: string;              // e.g. "lyapunov"
  libration_point: number;     // integer 1..5
  row_index: number;           // integer, row in the catalog response
  x0: number;                  // normalized initial state and period, as listed
  vy0: number;
  period: number;
  jacobi: number;
  stability_index: number;     // catalog nu = (lambda_max + 1/lambda_max)/2
}

interface Trajectory {         // artifact of kind "trajectory"
  schema_version: 1;
  run_id: string;              // equals manifest.run_id
  frame: "barycentric_synodic";
  time_unit: "normalized_cr3bp";
  state_order: ["t", "x", "y", "vx", "vy"];
  samples: number[][];         // at least 2 rows, each of length 5 in state_order;
                               // t strictly increasing; all values finite
  events?: TrajectoryEvent[];  // M3: marked points of the trajectory, in time order
}

interface TrajectoryEvent {
  name: string;                // /^[a-z0-9_]+$/, e.g. "departure", "perilune",
                               // "return_shell_crossing"
  t: number;                   // normalized time inside [samples[0][0], samples[-1][0]]
  state: [number, number, number, number];  // x, y, vx, vy at t (normalized, synodic)
}

interface SampleTable {        // artifact of kind "sample_table" (row-major table)
  schema_version: 1;
  run_id: string;              // equals manifest.run_id
  columns: {
    name: string;              // /^[a-z0-9_]+$/, unique
    unit: MetricUnit | null;   // null: categorical text column
    definition: string;
  }[];
  rows: (number | string | null)[][];  // one value per column; null = undefined for that
                                       // row (e.g. no return time without a return)
}
```

**M3 additions (all optional, additive).** `MetricUnit` `"deg"`, `Manifest.termination`, `Solver.return_shell`, `Trajectory.events`, `Artifact.kind` `"sample_table"` and `SampleTable` were added for the Gravity Atlas M3 free-return runs (`phasespace.gravity.free_return_export`). Runs without them, such as the published M1 reference run, remain valid. **The website validator (`web/src/lib/data/`) does not accept these fields and values yet; it must be updated before any M3 run is published to `web/public/research/`.** Full map tables (`samples.json`, tens of MB) stay local under `runs/`; only binned summaries are meant for publication.

Notes:

- Because `trajectory.json` contains `run_id`, its bytes and therefore its `sha256` depend on the run directory name. Re-exporting identical numbers under another `run_id` gives a different hash. This is intended: an artifact is tied to its run.
- `random_seed` is required so that "no randomness" is stated explicitly (`null`), not implied.
- `limitations` must say when `source.git_dirty` is true, because `git_commit` alone does not then identify the code.
- `velocity_scale_km_s`, `primaries.*.x` and `mu` are redundant on purpose. A validator should check that they are consistent, to relative precision 1e-12.

### Example `manifest.json` (shape only)

```json
{
  "schema_version": 1,
  "run_id": "gravity-benchmark-a",
  "study_id": "gravity-atlas",
  "experiment_id": "cr3bp-feasibility-v1",
  "status": "computed_unvalidated",
  "created_at": "2026-10-18T00:00:00Z",
  "source": {
    "repository": "VitalyVorobyev/PolicyLab",
    "git_commit": "<actual-commit-sha>",
    "git_dirty": false,
    "generator": "phasespace.gravity.export",
    "generator_version": "0.1.0"
  },
  "environment": {
    "python": "<version>", "python_implementation": "CPython", "numpy": "<version>",
    "scipy": "<version>", "platform": "<platform>", "machine": "<machine>"
  },
  "model": {
    "name": "planar-cr3bp",
    "coordinate_frame": "barycentric_synodic",
    "units": "normalized_cr3bp",
    "mu": 0.0121505856,
    "length_scale_km": 384400.0,
    "length_scale_convention": "nssdca_lunar_semimajor_axis",
    "time_scale_seconds": 375190.0,
    "velocity_scale_km_s": 1.0245,
    "primaries": {
      "earth": {"x": -0.0121505856, "y": 0.0, "radius_km": 6371.0},
      "moon": {"x": 0.9878494144, "y": 0.0, "radius_km": 1737.4}
    },
    "assumptions": ["circular_primaries", "massless_spacecraft", "planar"]
  },
  "solver": {
    "name": "DOP853", "rtol": 1e-12, "atol": 1e-12, "dense_output": true,
    "max_evaluations": 1000000,
    "collision_events": {
      "earth_radius_km": 6371.0, "moon_radius_km": 1737.4,
      "function": "distance_minus_radius", "direction": "decreasing", "terminal": true,
      "grazing_detection": "periapsis_below_radius", "start_inside": "not_propagated"
    }
  },
  "random_seed": null,
  "artifacts": [
    {"kind": "trajectory", "path": "trajectory.json", "sha256": "<actual-file-sha256>"}
  ],
  "metrics": {},
  "limitations": ["Example schema only; no physical validity is implied."]
}
```

**Important:** this is a *shape example*, not a valid existing record. Angle brackets are placeholders, `created_at` is illustrative, and the scales are approximate until calculated from declared constants. Do not commit this literal example as published scientific data. Exporters must fill real source SHA, timestamps, scales, checksums and solver settings.

### `trajectory.json` example

```json
{
  "schema_version": 1,
  "run_id": "gravity-benchmark-a",
  "frame": "barycentric_synodic",
  "time_unit": "normalized_cr3bp",
  "state_order": ["t", "x", "y", "vx", "vy"],
  "samples": [
    [0.0, 0.1, 0.2, 0.0, 0.1],
    [0.01, 0.101, 0.2009, 0.02, 0.08]
  ]
}
```

The rows above are **fabricated data-shape examples, not a CR3BP solution**. The UI must never present them as computed scientific results.

## Status and validation meaning

- `illustrative`: invented or drawn for demo; **not physical evidence**.
- `computed_unvalidated`: produced by a computational procedure but independent scientific validation remains incomplete.
- `validated`: physics/model-specific checks passed and their thresholds are recorded. Does **not** imply a new scientific discovery.

A manifest must include `validation` (see `Validation` above) when `validated`, and may include it otherwise. A run must not be labeled validated without such evidence. Convergence records of internal solvers, such as `solver.differential_correction.convergence`, are not validation checks: they hold by construction.

## Mandatory invariants

- Trajectory times strictly increase; all values finite; state vector order explicit; units are not guessed.
- For CR3BP, primaries, coordinate conventions, origin, and epoch/model scales are unambiguous.
- `sha256` matches artifact bytes; published files never overwritten in place.
- Dataset provenance includes precise code commit and the actual model/solver configuration.
- Frontend validates unknown versions, nulls, nonfinite values, length mismatches, wrong paths, empty runs and time ordering; fails clearly.
- No user-supplied path traversal or remote cross-origin scripts through manifest fields.
- Derived metrics carry physical units and definition, e.g. `departure_delta_v_km_s`, `return_shell_crossing_time_days`, `lunar_min_distance_km`; *only* include when actually computed.

## Comparison/protocol artifacts (M4)

Store independently versioned experiment definitions containing: predeclared candidate parameterization, feasibility thresholds, objective and descriptor bounds, budget measured in numerical propagations (and wall-clock as secondary), seed list, held-out perturbation distribution, numerical solver settings, and statistical analysis procedure.

Prefer schema validation on producer and consumer sides (Python + TypeScript). Avoid generating a complicated cross-language framework before first two consumers exist.
