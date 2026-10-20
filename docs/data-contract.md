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
  derived_from?: DerivedFrom;        // the run was derived from another run's artifact
                                     // without new propagations (e.g. the web outcome grid);
                                     // model, solver and random_seed are then the source's
}

interface DerivedFrom {        // provenance of a derived run (M3 web outcome grid)
  run_id: string;              // source run id (pattern as Manifest.run_id)
  experiment_id: string;       // source experiment
  artifact_path: string;       // source artifact the derivation read (plain file name)
  artifact_sha256: string;     // its sha256, verified by the generator before derivation
  git_commit: string;          // source run's source.git_commit
  git_dirty: boolean;          // source run's source.git_dirty
  derivation: string;          // what was derived and how (lossless / no recomputation)
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
  kind: "trajectory" | "sample_table" | "outcome_grid" | "atlas" | "error_fan";
      // "sample_table" added for M3 maps and searches; "outcome_grid" for the web map;
      // "atlas" and "error_fan" for the M4 demo run (see "M4 demo atlas" below)
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

type FreeReturnOutcome =       // phasespace.gravity.free_return.Outcome
  | "feasible" | "numerical_failure" | "invalid_start" | "moon_collision"
  | "earth_collision" | "no_return_within_t_max" | "return_before_lunar_encounter"
  | "perilune_out_of_band";

interface OutcomeGrid {        // artifact of kind "outcome_grid": a lossless, compact
                               // encoding of a 2-D grid map's per-cell outcomes
  schema_version: 1;
  run_id: string;              // equals manifest.run_id
  phase_axis: GridAxis;        // name "phase_deg", unit "deg"
  delta_v_axis: GridAxis;      // name "delta_v_km_s", unit "km/s"
  impulse_angle_deg: number;   // of every cell
  order: "phase_major";        // cell index = i_phase * delta_v_axis.count + i_delta_v
  outcomes: FreeReturnOutcome[];  // code table, unique: code k means outcomes[k]
  encoding: "base64_uint8";    // codes = base64 (RFC 4648, padded) of one uint8 per cell
  codes: string;               // decodes to exactly phase_axis.count * delta_v_axis.count
                               // bytes, each < outcomes.length
  feasible: {                  // one row per cell with outcome "feasible", in index order
    columns: SampleTable["columns"];  // first two: "phase_index", "delta_v_index" (count)
    rows: number[][];          // finite numbers, one per column
  };
}

interface GridAxis {
  name: string;
  unit: MetricUnit;
  min: number;                 // value at index 0
  step: number;                // > 0; value at index i = min + i * step
  count: number;               // integer >= 1
}
```

**M3 additions (all optional, additive).** `MetricUnit` `"deg"`, `Manifest.termination`, `Solver.return_shell`, `Trajectory.events`, `Artifact.kind` `"sample_table"` and `SampleTable` were added for the Gravity Atlas M3 free-return runs (`phasespace.gravity.free_return_export`). Runs without them, such as the published M1 reference run, remain valid. The website validator (`web/src/lib/data/`) checks all of them. Full map tables (`samples.json`, tens of MB) stay local under `runs/`.

**Web outcome grid (additive).** `Manifest.derived_from`, `Artifact.kind` `"outcome_grid"`, `FreeReturnOutcome` and `OutcomeGrid` were added so that the S1 map can be published without downsampling: `python -m phasespace.gravity.free_return_export web-map` (`just gravity-m3-web-map`) verifies the source map's `samples.json` sha256 and writes a new run holding every cell's outcome as base64 uint8 codes (~130 kB for 720 × 126 cells) plus the feasible cells' perilune altitude, encounter time, return time and passage sense, copied verbatim. Its status is `computed_unvalidated`, like its source.

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

**M4 runs (additive; `python -m phasespace.gravity.m4`, protocol `docs/studies/gravity-atlas/protocol.md`).** M4 manifests add one optional field, `m4`, and use the artifact kind `sample_table` (the demo run uses `atlas` and `error_fan`, below). Runs without it remain valid. M4 evidence runs stay local under `runs/`; only the curated demo run is published.

```ts
interface Manifest { /* ... */ m4?: M4Block; }

interface M4Block {
  protocol_path: string;       // "docs/studies/gravity-atlas/protocol.md"
  protocol_version: string;    // e.g. "v1"
  protocol_status: string;     // the protocol's Status row when the run was made ("draft" |
                               // "frozen"); only "frozen" runs can be comparative evidence
  config_sha256: string;       // sha256 of the canonical JSON (sorted keys, compact) of
                               // phasespace.gravity.m4.config.PROTOCOL: every declared constant
  stage: "smoke" | "tune" | "pilot" | "main" | "reference" | "ground_truth" | "robustness"
       | "calibration" | "selection" | "pilot_analysis" | "analysis" | "demo_export";
  methods: M4Method[];         // one per method in the run; [] for every other stage
  inputs?: { run_id: string; manifest_sha256: string; git_commit: string }[];
                               // analysis stages: every stored run read
  calibration?: object;        // stage-specific results of the analysis stages (below)
  selection?: object;
  pilot?: object;
  analysis?: object;
  demo?: object;               // demo_export: {seed, seed_choice, analysis_run_id,
                               // groundtruth_run_id, configs, errors, gates, budget_m_s,
                               // simplification: {nominal, fan}}
}

interface M4Method {
  name: "D" | "MS" | "MBH" | "DE" | "CC" | "CCstar" | "T" | "T2";
  config: Record<string, unknown>;  // the method's hyperparameters as plain JSON ({} for D)
  seed: number;                // the run seed (Sobol scrambling and method streams)
  budget: number;              // propagation budget N (the CC* safety cap for CCstar)
  propagations: number;        // propagations used; every evaluation counts, cache hits do not
  status: "budget_used" | "stopped_early";
  evaluations_path: string;    // sample_table: one row per propagation, in budget order
  catalog_path: string;        // sample_table: catalog per checkpoint (10 x 10) and the final
                               // 20 x 20 sensitivity catalog
}
```

- `random_seed` is the run seed (method runs), the GT map seed (`ground_truth`) or the error-draw seed (`robustness`).
- `evaluations*.json` columns: index, purpose tag, unit-cube coordinates, physical departure, M4 outcome (the M3 outcome, or `numerical_failure` when the scaled Jacobi gate of protocol 5.2 fails), M3 outcome, gate flag, the M3 record quantities, the time at which h_vp is evaluated, entry sense, Jacobi scale S and relative drift, nfev, e, the 10 × 10 cell, and per-propagation wall time (the only non-reproducible column). The `ground_truth` run stores feasible propagations only.
- Metrics carry budget counters (`propagations`, `cache_hits`, `rhs_evaluations`), numerical health (`numerical_failures`, `gate_failures`), cost (`wall_time_hours`, `propagation_time_hours`, `overhead_fraction`) and the method's own counters (`counter_*`), prefixed by the lower-case method name when a run holds several methods (smoke runs).
- A `robustness` run evaluates one injection set (`T_E`, `H_E_nominal` or `H_E_wide`) with every Gates execution-error set paired with it (protocol 9.1; `H_E_nominal` carries `gates_primary` and `gates_stress`). It holds `robustness.json`, one row per catalog elite and error draw: re-checked cell and index, singular-Jacobian flag, uncorrected feasibility, commanded |Δv| (km/s; 0 when no correction is commanded), one `executed_feasible_<gates set>` column per paired set (null when not propagated because the chord diverged), and the diagnostics chord convergence, divergence, feasibility with the exact commanded correction, iterations and nfev. Survival at budget B (decision D25) is `uncorrected_feasible or (commanded_delta_v_km_s ≤ B and executed_feasible_<set>)`. Its metrics record the re-check counts and the Gates-model parameters (`mcc_<gates set>_*`, `mcc_gates_normals_seed`, decision D24); its limitations name the injection and execution error sets and their seeds.
- A run made before the protocol is frozen states in `limitations` that it is not comparative evidence.
- **Analysis stages** (freeze checklist steps 4, 5, 6 and 8; `phasespace.gravity.m4.analysis`) read stored runs only and refuse any input whose manifest lacks the frozen `config_sha256`, `protocol_status = "frozen"` or `git_dirty = false`. Each records its inputs in `m4.inputs`.
  - `calibration` (`gravity-m4-calibration-v1`): `m4.calibration = {budget_m_s, medians}` and `calibration.json` (median pooled s_i of the D tuning elites per grid value of B).
  - `selection` (`gravity-m4-selection-v1`): `m4.selection = {budget_m_s, configs, configs_argument, satisfiable_tuning_sets}`; `tuning_scores.json` (P1, P2 and coverage against G_tune per tuning run), `selection.json` (per configuration: means, ranks, rank sum, selected) and `tuning_sets.json` (the 30 tuning sets and their satisfiability).
  - `pilot_analysis` (`gravity-m4-pilot-analysis-v1`): `m4.pilot = {seed_count, seed_count_unclamped, main_seeds, achieved_power, k3_triggered, pilot_median_survival, pe_per_draw_t_e, projection_total_pe, reductions, k4_triggered, ...}`; `pilot_scores.json`, `seed_count.json` (SD of T − X per baseline and endpoint) and `projection.json` (protocol 7.4 items per scenario).
  - `analysis` (`gravity-m4-analysis-v1`): `m4.analysis = {budget_m_s, configs, main_seeds, h1, criteria}`; `per_run.json` (every protocol section-10 metric per main run and for CC*), `anytime.json` (checkpoint snapshots), `results.json` (long format: summaries, anytime means with CIs, the ten primary contrasts with sign-flip p and Holm, H1, secondary estimates, CC*, H2, criteria K2/K5/K6/K7, ground-truth cell counts, LSODA sample), `lsoda.json` (the 5 % sample, full M3 S3 validation) and `heldout_sets.json` (held-out satisfiability).

**M4 demo atlas (additive; `python -m phasespace.gravity.m4 demo-export --confirm-frozen`, `just gravity-m4-demo-export`, module `phasespace.gravity.m4.demo_export`).** One curated, published run, `gravity-m4-demo-atlas-v1` (`web/public/research/gravity-atlas/m4-demo-atlas-v1/`, catalog kind `"atlas"`), for the public demo page. It reads stored evidence runs only (refusing any that is not evidence or not an input of `gravity-m4-analysis-v1` with the same manifest sha256) and re-propagates only what it draws: the seed-1001 elites of the seven budget-matched arms (main seed 1001 is the first main seed, fixed in advance; CC* is not included) and their 50 `H_E_nominal` draws without correction. The export fails, writing nothing, unless every re-propagated elite reproduces its stored record within the 5.6 re-check thresholds and every recomputed uncorrected feasibility equals its stored robustness row; these comparisons are recorded as `validation.checks`. Status stays `computed_unvalidated`: the checks verify reproduction of computed_unvalidated runs, not an independent validation (a `validation` block is allowed on such a run). `random_seed` is the `H_E_nominal` draw seed (7101); `m4.stage` is `"demo_export"`, `m4.inputs` lists every run read.

Numbers in records and summaries are copied from stored runs, never recomputed (except the checks above). Paths are display polylines: barycentric synodic frame in km (`x_km = x × length_scale_km`, Earth at `x_km = −μL`, Moon at `(1 − μ)L`), times in days after TLI, simplified by Douglas–Peucker with the per-point tolerance `min(abs_tolerance_km, rel_tolerance × distance to the nearer primary centre)` over the accepted solver steps refined with dense output (`simplification` blocks). Both files are compact JSON.

```ts
interface Polyline { t_days: number[]; x_km: number[]; y_km: number[] }
                     // equal lengths >= 2; t strictly increasing; nominal: 0.1 km and
                     // 1e-5 d resolution, <= 200 points; fan: 1 km and 1e-4 d, <= 120

interface Atlas {    // artifact of kind "atlas" (atlas.json)
  schema_version: 1; run_id: string;
  frame: "barycentric_synodic"; length_unit: "km"; time_unit: "day"; frame_definition: string;
  mu: number; length_scale_km: number; time_scale_seconds: number;
  earth: { x_km: number; y_km: number; radius_km: number };
  moon: { x_km: number; y_km: number; radius_km: number };
  parking_orbit_radius_km: number;   // 6556 (paths start here, about the Earth centre)
  return_shell_radius_km: number;    // 6492.92 (feasible paths end here)
  perilune_band_km: [number, number]; gamma_target_deg: number;  // -6.5 (objective e)
  seed: number; seed_choice: string; main_seeds: number[]; analysis_run_id: string;
  budget_m_s: number;                // calibrated B (1 m/s)
  errors: "H_E_nominal"; draws: number; gates: "gates_primary"; survival_definition: string;
  grid: {
    d1: GridDim;     // name "log10_perilune_altitude_km": min 2, max log10 30000, edges_km
    d2: GridDim;     // name "return_time_days": min 5, max 10, edges_days
    binning: string;
    reachable_cells: [number, number][];  // G (32 cells, [i_d1, i_d2]), 0-based
    reachable_definition: string;
  };
  h1: Record<string, string>;        // per baseline, from the analysis manifest
  contrasts: { endpoint: "P1" | "P2"; arm: "T"; baseline: string; mean_difference: number;
               ci_low: number; ci_high: number; p_value: number; p_holm: number;
               significant: number }[];          // the ten primary contrasts, verbatim
  simplification: object; fan_simplification: object;
  arms: AtlasArm[];                  // D, MS, MBH, DE, CC, T, T2 in this order
}
interface GridDim { name: string; definition: string; min: number; max: number; bins: number;
                    edges: number[];        // bins + 1, descriptor units
                    edges_km?: number[];    // d1 only: 10^edges (perilune altitude, km)
                    edges_days?: number[];  // d2 only: = edges (return time, days)
}
interface AtlasArm {
  arm: string; config_index: number; source_run_id: string; robustness_run_id: string;
  fan_path: string;                  // its error_fan artifact, e.g. "fan-t.json"
  summary: Record<"p1" | "p2" | "coverage" | "mean_s" | "elites",
                  { mean: number | null; median: number | null; sd: number | null;
                    min: number | null; max: number | null }>;   // over the 16 main seeds
  anytime_coverage: { checkpoint: number; mean: number; ci_low: number; ci_high: number }[];
  seed_values: Record<"p1" | "p2" | "coverage" | "mean_s" | "elites", number | null>;
  seed_anytime_coverage: { checkpoint: number; coverage: number }[];   // seed 1001
  elites: AtlasElite[];              // the seed-1001 catalog after 5.6, by cell; may be []
}
interface AtlasElite {
  cell: [number, number]; index: number;      // propagation index in the source run
  theta_deg: number; delta_v_km_s: number; alpha_deg: number;  // decision vector
  perilune_altitude_km: number; lunar_encounter_time_days: number;
  return_time_days: number; entry_flight_path_angle_deg: number;  // gamma_S
  objective_deg: number;                      // e = |gamma_S - gamma*|
  survival: number;                           // s_i at B, primary Gates set (stored rows)
  uncorrected_feasible_draws: number;         // of the 50 draws
  perilune: { t_days: number; x_km: number; y_km: number };  // a vertex of `path`
  path: Polyline;                             // TLI to the return-shell crossing
}

interface ErrorFan {  // artifact of kind "error_fan" (fan-<arm>.json, one per arm)
  schema_version: 1; run_id: string; arm: string;
  frame: "barycentric_synodic"; length_unit: "km"; time_unit: "day";
  errors: "H_E_nominal"; error_seed: number; budget_m_s: number; gates: "gates_primary";
  error_draws: { magnitude: number; direction_deg: number; timing_s: number }[];  // draw k
  elites: { cell: [number, number]; index: number;     // same order as AtlasArm.elites
            draws: (Polyline & {
              outcome: FreeReturnOutcome;   // M4 outcome without correction
              uncorrected_feasible: boolean; // recomputed; equals the stored row
              survived: boolean;            // stored: success at B (D25)
            })[] }[];                        // one per error draw, in draw order
}
```

A fan path is the uncorrected trajectory as the robustness runs judge it: the executed TLI propagated to t_c = 1 d and continued without a burn, ending at the return shell, a collision or T_MAX = 10 d. `survived` may be true for an infeasible path (the MCC rescued it). Its `catalog.json` entry uses kind `"atlas"`; it is added together with the website validator change that accepts the kinds `atlas` and `error_fan` (the validator rejects unknown kinds).

## Model Citizens addendum (C1 data audit)

**Scope (additive; `python -m phasespace.citizens.export`, `just citizens-audit`).** Model Citizens runs are not CR3BP propagations, so their manifests replace the CR3BP-specific fields. Everything else in schema v1 (run directory, `source`, `environment`, `artifacts`, `metrics`, `limitations`, `validation`, `Check`, `SampleTable`, immutability, checksums) applies unchanged. These runs stay local under `runs/` until a curated public sample is chosen; the website validator does not accept them yet.

```ts
type MetricUnit = /* ... */ | "coin";   // a coin of the Koster et al. (2025) common-pool game
                                        // (pool, offers, reciprocations, rewards, residuals)

interface CitizensManifest {   // Manifest with these differences:
  study_id: "model-citizens";
  model: CitizensModel;        // instead of Cr3bpModel
  simulator: null;             // instead of `solver` (which is absent): null = the run
                               // simulates no games (the C1 data audit). The first run that
                               // simulates games (C2) defines a CitizensSimulator type here,
                               // in an amendment, before it is emitted.
  random_seed: number | null;  // null for the data audit (no randomness)
  citizens: CitizensBlock;     // the one allowed extra block
  // termination, derived_from and m4 are not used.
}

interface CitizensModel {      // phasespace.citizens.game / .mechanisms constants
  name: "koster2025-common-pool";
  units: "coin";
  players: 4;
  max_pool: number;            // 200 (initial pool and cap)
  growth_factor: number;       // 1.4
  pool_floor: number;          // 0.01 (data-derived, D33)
  depletion_threshold: number; // 1
  exclusion_threshold: number; // 1
  rounds_exps_1_3: number;     // 40
  baseline_weights: { equal: number; mixed: number; proportional: number };  // 1, 0.5, 0
  interpolating_k: number;     // 22
  interpolating_weight_decimals: number;   // 1: w = round((R/200)^22, 1) (D33)
  recorded_offer_withheld: number;         // 0.001 coins withheld by the deployed software
  recorded_offer_decimals: number;         // 4
  offer_rule: "deployed" | "paper";        // the rule simulations use (D33: "deployed")
  legacy_remainder_dropped_games: number;  // size of the frozen legacy launch_id list
}

interface CitizensBlock {
  stage: "c1_data_audit";
  gate: {
    name: "G1a";               // the gate scope the run's validation.checks cover
    scope: string;             // plain-language list of what the checks cover
    status_semantics: string;  // restates the rule below
    deferred: DeferredItem[];  // known items outside the scope, each with a bound
    deferred_to_g1b: string[]; // work that waits for the validation protocol freeze
  };
  dataset: { url: string; size_bytes: number; md5_base64: string; sha256: string;
             license: string; citation: string; notebook: string };
  parser_version: string;
  processed: { training_sha256: string; heldout_sha256: string };  // processed .npz files
  tolerances: { pool: number; offer: number; offer_exact: number; record: number;
                z: number };   // coins, except z (dimensionless)
  validation_protocol: {
    path: string;              // "docs/studies/model-citizens/protocol-validation.md"
    held_out_access: string;   // what the run read of held-out games
    sha256?: string;           // required in any run that read held-out games through
                               // data.load_heldout: the frozen protocol's sha256
  };
  rules: { name: string; rule: string }[];      // game conventions established from data
  diagnostics: Record<string, unknown>;         // training-only counts behind the rules
  c2_heldout_counts?: {        // C2 count-only held-out checks (protocol-validation A29, A30);
                               // reported, never gates; indicator residuals (counts only)
    action_anomalies: { entries: number; n_flagged: number; non_integer_c: number;
                        c_above_floor_e: number; positive_c_on_offer_below_1: number;
                        negative_c: number; games_with_anomalies: number };
    players_active_identity: { entries: number; n_flagged: number; games_with_mismatch: number };
    flagged_by_mechanism_and_cohort: Record<string, number>;
    definition: string;
  };
}

interface DeferredItem {
  name: string;                // e.g. "heldout_pool_anomaly"
  gate: "G1b";
  definition: string;
  observed: Record<string, number | string[]>;  // counts and residual maxima only
  bound: Record<string, number | string>;       // documented before diagnosis
  within_bound: boolean;
}
```

**Status of a citizens run.** `validated` means: every check in `validation.checks` passed, and every `citizens.gate.deferred` item is `within_bound`. The checks cover only the scope named in `citizens.gate.name`. For the C1 audit that is G1a (D33): training-population identities, mechanism-offer reproduction including held-out Interpolating rows as an arithmetic identity, and the four training-only Exp 1 z-values. A deferred item inside its bound is a known, documented, undiagnosed exception. It is not validated, and the run makes no claim about it beyond the bound. G1 as a whole passes only when a G1b run, made after the validation protocol freeze, records the deferred work as passing checks. Otherwise the status is `computed_unvalidated`.

**Held-out data in citizens runs.** Before the validation protocol freeze, held-out human games (Interpolating, M1, M2 in Exps 1–4) may appear only as game counts and as residual summaries of the arithmetic identities: entries, maximum |residual| in coins, failing entries and failing games. Sample tables give no game lengths or per-game values for them. The only row information is the identity entry counts: row totals per mechanism and cohort.

## Model Citizens addendum (C2 models)

**Scope (additive; `python -m phasespace.citizens.fit`, `just citizens-c2-fit`, `just citizens-c2-smoke`).** C2 runs follow the C1 addendum with these differences. This amendment defines `CitizensSimulator`, as the C1 addendum requires before a run that simulates games is emitted.

```ts
interface CitizensC2Manifest {   // CitizensManifest with these differences:
  experiment_id: "citizens-c2-fit-v1" | "citizens-c2-smoke-v1";
  simulator: CitizensSimulator | null;   // null for a fitted-model run (no games simulated)
  random_seed: number;                   // the split seed 20261020; every other seed is in citizens.seeds
  citizens: CitizensC2Block;
}

interface CitizensSimulator {
  name: "phasespace.citizens.simulate";
  pool_rule: string;                  // full rule with the 0.01 floor (D33)
  offer_rule: "deployed" | "paper";   // D33: "deployed"
  rounds: number;                     // 40
  games_per_model_mechanism: number;  // S (protocol 7: 1000)
  seed_rule: string;                  // "50000 + 100 m + r (protocol 7)"
  feasibility_atol: number;           // protocol 5: sum(e) <= R + 1e-9
}

interface CitizensC2Block {
  stage: "c2_fit" | "c2_smoke";
  config_sha256: string;              // phasespace.citizens.config.config_sha256()
  config: Record<string, unknown>;    // every constant of the configuration module
  validation_protocol: {
    path: string;                     // docs/studies/model-citizens/protocol-validation.md
    version: string;                  // "v1"
    status: string;                   // header status when the run was made ("draft" before the freeze)
    sha256: string;                   // of the protocol text read at run time (draft or frozen)
    held_out_access: string;          // "none" for every pre-unlock C2 run
  };
  dataset: { source_sha256: string; parser_version: string; processed_training_sha256: string };
  seeds: Record<string, number | number[] | string>;
  split: Record<string, number | string>;     // unit, game and decision counts per split
  fit_settings: Record<string, unknown>;      // protocol_settings = false for reduced fits
  smoke?: Record<string, unknown>;            // c2_smoke only
}
```

**Artifacts.** `split.json` (sample table: unit, stratum, split), `models.json` (`kind: "fitted_parameters"`: every fitted parameter and every optimiser start), `b1-member-<seed>.npz` (`kind: "checkpoint"`: float32 GRU weights, deterministic bytes), `likelihoods.json` (fit/validation-split mean log-likelihood per model and Δℓ vs M0). Smoke runs add `criterion_*.json` and `outcomes_*.json` (protocol 8 tables on the training regimes). Later stages load fitted models only through the manifest's SHA-256 values (`fit.load_fitted`).

**Status of a C2 run.** A run made while `validation_protocol.status` is not `frozen` is always `computed_unvalidated`, whatever its checks say; its checks (split sizes, stop rule K1, feasibility of every simulated game) are recorded for inspection. A fit run made under the frozen protocol keeps `computed_unvalidated` too: it is an input, judged by the stages that load it.

## Model Citizens addendum (C2 surrogates, B4 and post-freeze stages)

**Scope (additive).** These runs follow the C1 and C2 addenda. They add stages, two simulator kinds and three artifact kinds. All of them stay local under `runs/`; raw B4 call records are never published (below).

```ts
type ArtifactKind = /* ... */
  | "model_parameters"   // equinox.tree_serialise_leaves bytes of a surrogate network
  | "llm_calls"          // JSONL, one record per CLI call: prompt, raw stdout/stderr,
                         // parsed value or error, latency, cost, reported model ids
  | "config";            // a B4 run's resumable configuration (config.json, verbatim)

type CitizensStage =
  | "c1_data_audit" | "c2_fit" | "c2_smoke"   // earlier addenda
  | "c2_surrogate_fit"   // phasespace.citizens.surrogate fit (just citizens-c2-surrogates)
  | "c2_b4_probes"       // memorisation and isolation probes (just citizens-b4-probe)
  | "c2_b4_pilot"        // B4-plain Proportional pilot (just citizens-b4-pilot)
  | "c2_b4_calibration"  // B4-calibrated calibration games (just citizens-b4-calibrate)
  | "c1_g1b"             // post-freeze: held-out z-values and pool anomaly rows (just citizens-g1b)
  | "c2_heldout_eval"    // post-freeze: the frozen held-out evaluation (just citizens-c2-evaluate)
  | "c2_b4_heldout"      // post-freeze: one B4 run per (variant, mechanism) (just citizens-b4-heldout)
  | "c2_b4_eval";        // post-freeze: B4 runs against the evaluation run (just citizens-b4-evaluate)

interface CitizensLlmSimulator {   // `simulator` of the B4 game runs
  kind: "llm_citizens";
  variant: "plain" | "calibrated";
  mechanism: string;               // e.g. "Proportional Baseline", "Interpolating", "RL Agent (M1) surrogate"
  offer_rule: "deployed";
  n_rounds: number;                // 40
  backend: { kind: "claude_cli"; argv: string[]; prompt_via: "stdin"; cwd: string;
             timeout_s: number; model_requested: string; effort_requested: string;
             cli_version: string | null };
  environment_passed: string[];    // names (never values) of the allowlisted variables passed
                                   // (A28; runs before A28 hold removed_environment instead)
  environment_rule: string;
  retry_policy: { invalid_answer_retries: number; infrastructure_retries: number;
                  infrastructure_backoff_s: number[]; discards: string };   // A27
  prompt_version: string;          // "b4-prompt-v2" (A32; "b4-prompt-v1" in superseded runs)
  system_prompt: string;
  rules_sha256: string;
}

interface CitizensProbeSimulator { // `simulator` of the probe run
  kind: "llm_probe";
  backend: CitizensLlmSimulator["backend"];
  environment_passed: string[];
}
```

- **`c2_surrogate_fit`.** `random_seed` is the BC split seed (20261021). `citizens` holds `pre_freeze` (true while the protocol is a draft), `training_data`, `split`, `train_config`, `acceptance`, `config_sha256`, `validation_protocol` (path, version, status, sha256), `reference_parameters_sha256` (the run whose parameter files must be reproduced, or null) and `surrogates.M1|M2` (parameter file, width, kept seed, every attempt, BC fit and validation fidelity). Artifacts: `surrogate-m1.eqx`, `surrogate-m2.eqx` (`model_parameters`), `training_curves.json`. Checks: the §6.4 BC thresholds and, with a reference, one `surrogate_<key>_differs_from_reference` check per mechanism (0 = bit-identical). Status: `validated` only if every check passes and the protocol was frozen; a pre-freeze run is `computed_unvalidated`.
- **`c2_b4_probes`, `c2_b4_pilot`, `c2_b4_calibration`, `c2_b4_heldout`.** `random_seed` is 7001. Artifacts: `config.json` (`config`), one `game-NNNN.calls.jsonl` per game and `abandoned.calls.jsonl` if any (`llm_calls`), `games.json` and `rounds.json` (`sample_table`: per-game metrics and type keys; per-round pool, offers, whole-coin returns, next pool); the probe run has `probes.jsonl` only. `citizens` holds `stage`, `label`, `pre_freeze`, `config`, `call_statistics`, `budget_projection` and `validation_protocol` (path, status, sha256 at run time, `held_out_access: "none"`). Held-out B4 runs add `frozen_protocol_sha256`, `config_sha256`, `git_commit_at_start`, `mechanism_key`, `surrogate_run` and `type_weights` (the chosen weights, objective, human targets and the calibration run's manifest sha256). Each call record carries `error_class` (`null`, `"invalid_answer"` or `"infrastructure"`, A27) and `infra_error`; `call_statistics` counts `invalid_attempts` and `infrastructure_attempts` separately, and `games.json` has an `infrastructure_calls` column (records written before A27 are classified by their error kind). Status is always `computed_unvalidated`: LLM sampling is not seeded.
- **`c1_g1b`.** `z_values.json` (13 published z-values involving held-out mechanisms, reproduced, tolerance, pass) and `pool_anomaly_rows.json` (every held-out pool row that misses the full rule: game, round, R, Σe, Σc, raw and full-rule next pool, recorded next pool, residual, offers, reciprocations). Checks: each z within half a unit of its last printed digit; the D33 anomaly bound. `validated` iff every check passes. This run, made after the freeze, contains held-out row values.
- **`c2_heldout_eval`.** `citizens.inputs` lists the fitted-model, surrogate and G1b runs with their manifest sha256; each must have been made under the same frozen protocol sha256 and `config_sha256` from a clean tree, or the stage refuses. Artifacts: `surrogate_fidelity_human.json`; `criterion_heldout.json` / `outcomes_heldout.json` (gating regimes), `criterion_diagnostic.json` / `outcomes_diagnostic.json` (per experiment, M2 Exp 4), `criterion_in_sample.json` / `outcomes_in_sample.json`. `human_units_heldout.json` (each gating regime's human unit metrics, the distribution the A22 W1 uses). Outcome tables add `w1_scale` and `w1_normalised`; criterion tables add `mean_normalised_w1`, `floored_decisions` and `excluded_decisions` (A29). `citizens.labels` holds every (regime, model) label (closed-loop labels include the must-beat-marginal rule, A22) and `citizens.g2` the G2 decision with `b0_passes` (A23), `m0_sim_passes_a` and `w1_comparison`; `citizens.human_reference` (per regime and metric: mean, CI, τ, W1 scale), `citizens.closed_loop_tier`, `citizens.surrogate_reproduction` (problems and the owner's override reason, A25) and `citizens.declared_handling` (A29 counts). Checks: feasibility of every simulated game, input checksums, regimes failing G1b. `validated` iff every check passes.
- **`c2_b4_eval`.** Reads the evaluation run and the B4 held-out runs only (their `human_units_heldout.json` and `games.json` verified by SHA-256; each made under the frozen protocol sha256 and current `config_sha256`). `citizens.b4` holds one record per (variant, regime): complete and discarded games, `discard_fraction`, `b4_gating`, metrics in range, `pass_outcomes`, `mean_normalised_w1`, `m0_sim_passes_a`, `beats_m0_sim`, `label`, `counts_as_validated`. Artifact `b4_outcomes.json` (B4 mean and game-bootstrap CI, B = 10,000, seed 9001; human CI; τ; in range; W1 and normalised W1). `random_seed` is 9001.
- **Post-freeze runs** (`c1_g1b`, `c2_heldout_eval`, `c2_b4_heldout`, `c2_b4_eval`) are made only under the frozen, committed protocol with `--confirm-frozen` and from a clean tree; `source.git_dirty` is always false and `citizens.validation_protocol.sha256` (or `frozen_protocol_sha256`) is the frozen text's sha256.

**Privacy of B4 call records.** `llm_calls` artifacts hold raw prompts and CLI output, which can contain the operator's account e-mail address (the CLI injects it into the model's environment context). B4 runs are never copied to `web/public/`; a public B4 sample is derived from `games.json` / `rounds.json` (parsed actions) only.

## Model Citizens addendum (C3 planner pipeline)

**Scope (additive; `python -m phasespace.citizens.rl.stage`, `just citizens-c3-baselines`, `just citizens-c3-train`, `just citizens-c3-g3`).** C3 runs follow the C1 and C2 addenda with these differences. They read only a C2 fitted-model run (verified through its manifest's SHA-256 values) and never any human data.

```ts
type MetricUnit = /* ... */ | "s";      // seconds (training wall time)
type ArtifactKind = /* ... */
  | "summary";           // a JSON document of a gate decision (g3.json)
type CitizensStage = /* ... */ | "c3_baselines" | "c3_train" | "c3_g3";

interface CitizensC3Simulator {   // `simulator` of every C3 run
  name: "phasespace.citizens.rl.citizens.rollout";
  pool_rule: string;              // full rule with the 0.01 floor (D33)
  offer_rule: string;             // deployed rule for the baselines; R * softmax shares for the planner
  rounds: number;                 // 40
  precision: string;              // float64 (jax_enable_x64); deployed offers float32
  citizen_draws: string;
  common_random_numbers: string;
}
```

- **Common fields.** `experiment_id` is `citizens-c3-<stage>-v1`; `citizens` holds `stage`, `pilot_version`, `c3_config_sha256` and `c3_config` (`phasespace.citizens.rl.config`), `c2_config_sha256`, `fit_run` (path, run id, manifest sha256, git commit, C2 config sha256; null for `c3_g3`) and `held_out_access` ("none"). Status is always `computed_unvalidated`: these are pre-protocol pilot runs, and their checks are recorded for inspection.
- **`c3_baselines`.** `random_seed` is the evaluation key seed. Artifacts (`sample_table`): `baselines.json` (per model and mechanism: k, ln k, games, mean and SD of total surplus and the four protocol-8 metrics), `baseline_games.json` (every evaluation game), `k_grid.json` (mean total surplus of every grid point ln k on the tuning games). `citizens.k_star` holds the re-tuned k* per model. Checks per model: pmf agreement with the NumPy reference (max absolute difference ≤ 1e-6, mass error ≤ 1e-9) and closed-loop agreement (max |z| ≤ 4.5 per mechanism).
- **`c3_train`.** `random_seed` is the training seed. Artifacts: `planner.eqx` (`model_parameters`), `learning_curve.json`, `eval_games.json` and `eval_shares.json` (`sample_table`). `citizens.training` holds the model, seed, iterations, environment steps, every PPO hyperparameter, explicit overrides and wall times. Check: training hours ≤ 24 (G3 budget).
- **`c3_g3`.** `random_seed` is the bootstrap seed (9001). `citizens.inputs` lists the baselines run, the training runs (run id, manifest sha256, seed, environment steps, hours), the fit run's manifest sha256 and the common training signature; `check_poolable` refuses runs from a dirty tree, repeated seeds, runs that differ in anything but the seed (model, PPO settings, overrides, iterations, commit, fit run, evaluation key), a model other than B1, or a fit run or evaluation key other than the baselines run's. `citizens.seeds` holds the evaluation key (31000), the sampled-policy noise key (31003) and the bootstrap seed. Artifact `g3.json` (`summary`):
  - `model`, `g3_pass` and `g3_rule` (the deterministic planner's paired 95 % CI against Interpolating k = 22 lies above 0, and every training run took ≤ 24 h);
  - `comparisons`, keyed `"<policy> - <baseline>"` for the policies `deterministic` (shares = softmax(μ)) and `sampled` (z drawn from the logistic-normal head with the noise key) and the baselines Interpolating k = 22 and k*. Each entry holds the planner mean and per-seed means, the baseline mean and k, the paired mean difference, its percentile CI (seeds and games resampled, 10,000 resamples) and `planner_above`;
  - `final_log_std` (per seed, the five log σ values) and `training_hours`;
  - `diagnostics` (reported, not gating): `kept_share_per_round` (40 values, mean over games of 1 − Σe/R, for the deterministic and sampled planner and for Interpolating k = 22, k* and Proportional), `frac_offer_histogram` (10-bin histogram of frac(e) over offers e ≥ 1 with a χ² test against uniform, for the deterministic planner and k = 22) and `b1_member_disagreement` (mean and 90th percentile of the B1 members' mean pairwise total-variation distance, teacher-forced, over non-forced decisions, per seed for the planner and per baseline).

  Checks: longest training run ≤ 24 h; the negated lower CI bound of deterministic planner − k = 22 ≤ 0; re-simulated planner and baseline totals reproduce the recorded per-game values within 1e-9 coins.

## Model Citizens addendum (C4 main comparison)

**Scope (additive; `python -m phasespace.citizens.rl.c4`, `just citizens-c4-pilot`, `just citizens-c4-train`, `just citizens-c4-crossplay`, `just citizens-c4-distill`, `just citizens-c4-b4`).** C4 runs follow the C3 addendum with these differences.

```ts
type CitizensStage = /* ... */ | "c4_pilot" | "c4_train" | "c4_crossplay" | "c4_distill";
```

- **Common fields.** `experiment_id` is `citizens-c4-<stage>-v1`; `citizens` also holds `c4_version`, `c4_config_sha256` and `c4_config` (`phasespace.citizens.rl.c4_config`). The crossplay, distill and B4 stages also record `c4_protocol` (path and SHA-256 of the frozen `docs/studies/model-citizens/protocol.md`). They refuse to run unless that protocol is frozen and committed and the tree is clean (`--confirm-frozen`).
- **`ppo.retention`** (every training run made after C3; absent means `true`). `false` marks the no-retention ablation planner: a 4-way softmax over the players with Σe = R − 0.001, as under the deployed rules. It is set with `--no-retention` and recorded in `overrides`. Runs with different values are never pooled.
- **`c4_pilot`.** These are PPO feasibility pilots, never C4 evidence (the C4 stages refuse them as inputs). The run id is `citizens-c4-pilot-<model>-s<seed>`, with the seeds 901 (B2), 902 (B3) and 903 (B23), which are never reused. `random_seed` is the training seed. `citizens.inputs` holds the C3 baselines run and the k* it supplies. Artifacts:
  - `planner.eqx` and `learning_curve.json` (as `c3_train`);
  - `pilot.json` (`summary`), scored only on the training model's C3 monitoring games (512 games, key 31002). It holds the final planner's deterministic and sampled mean total surplus (noise key 41010), Gini, depletion round, decisions per game (offers ≥ 1), the final log σ and the per-round kept share. It also holds the five baselines on the same games (Equal, Mixed, Proportional, Interpolating k = 22 and k*), the planner minus k = 22 and minus k*, and `plateau_monitor` / `plateau_sampled` (`c4.plateau_iteration`).

  Check: training hours ≤ 24.
- **`c4_train`.** Same as `c3_train` (`stage train --milestone c4`).
- **`c4_crossplay`.** `random_seed` is the C4 evaluation key (41000; sampled-policy noise key 41001; bootstrap seed 9101). `citizens.inputs` lists the baselines run and every training run (model, retention, seed, manifest sha256). Training runs are grouped per (model, retention) and each group must pass `check_poolable`. Pilot runs and reserved seeds are refused. Artifacts (`summary`):
  - `crossplay.json`, which holds:
    - `columns` (B1, B2, B3, B23);
    - `policies` (planner rows `Planner[<model>]` and `Planner-noret[<model>]` with their seeds, then Equal, Mixed, Proportional, Interpolating k=22 and `Interpolating k*[<model>]`);
    - `cells[policy][column][mode]` for the modes `deterministic` and `sampled` (sampled is null for rules). Each cell holds the mean and per-seed mean total surplus, the Gini, decisions per game, `kept_share_per_round` (40 values), `kept_share_last_rounds` (rounds 39 and 40), `b1_disagreement` per seed (mean and p90 TV, teacher-forced) and `ood` per seed. `ood` is set for the B2, B3 and B23 columns only. It holds the fraction of non-forced decisions outside the training range of any input the model reads, per input, and off the occupied cells of the (x1/4, x4) grid;
    - `paired_differences`: every planner row and mode minus every rule (deterministic) and every other planner row (same mode), per column, with the mean, per-seed differences and a percentile CI (seeds of both rows and games resampled);
    - `training_support` (ranges and grid occupancy from the fit-split human training rows of the C2 fit; held-out rows are never read).
  - `crossplay_series.json`: per-game total surplus and Gini of every (policy, column, mode, seed).
- **`c4_distill`.** `random_seed` is the distillation key (41002). Artifact `distill.json` (`summary`) holds, per planner and seed, on 1,000 games of its own training model:
  - the binned w(R) rule (w and kept share per R/200 bin, row counts, R²);
  - the grid-fitted Interpolating k (ln k, k, R²);
  - the regression tree (nodes, depth, minimum leaf, round-1 share, R²);
  - the mean total surplus of the planner and of each distilled rule on the same games.
- **Planner versus B4.** The run follows the C2 B4 run layout (`finalize_games_run`, label `c4-planner`), with frozen prompts and flags (`b4-prompt-v2`). The offers come from a planner's deterministic policy. `citizens` adds `c4_protocol`, `c4_config_sha256`, `planner` (run id, manifest sha256, policy, retention) and `type_weights` (calibrated variant).

Prefer schema validation on producer and consumer sides (Python + TypeScript). Avoid generating a complicated cross-language framework before first two consumers exist.
