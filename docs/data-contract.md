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

A fan path is the uncorrected trajectory as the robustness runs judge it: the executed TLI propagated to t_c = 1 d and continued without a burn, ending at the return shell, a collision or T_MAX = 10 d. `survived` may be true for an infeasible path (the MCC rescued it). The catalog entry uses kind `"atlas"`.

Prefer schema validation on producer and consumer sides (Python + TypeScript). Avoid generating a complicated cross-language framework before first two consumers exist.
