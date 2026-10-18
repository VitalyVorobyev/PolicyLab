/**
 * Validated, typed views of the static research data. The manifest and trajectory types follow
 * docs/data-contract.md (schema v1) field for field; optional contract fields are `null` (or an
 * empty array for `solver.comparisons`) when absent.
 */

export type StudyStatus = "active" | "planned" | "conditional";

export interface CatalogRun {
  run_id: string;
  /** Directory relative to /research/, validated as a safe relative path. */
  path: string;
  /** Kind of the run's single artifact; the loader checks it against the manifest. */
  kind: ArtifactKind;
  label: string;
}

export interface CatalogStudy {
  study_id: string;
  title: string;
  status: StudyStatus;
  runs: CatalogRun[];
}

export interface Catalog {
  schema_version: 1;
  studies: CatalogStudy[];
}

export const RUN_STATUSES = ["illustrative", "computed_unvalidated", "validated"] as const;
export type RunStatus = (typeof RUN_STATUSES)[number];

export const SOLVER_NAMES = ["DOP853", "RK45", "Radau", "LSODA"] as const;
export type SolverName = (typeof SOLVER_NAMES)[number];

export const LENGTH_SCALE_CONVENTIONS = ["nssdca_lunar_semimajor_axis", "jpl_periodic_orbits_catalog"] as const;
export type LengthScaleConvention = (typeof LENGTH_SCALE_CONVENTIONS)[number];

export const METRIC_UNITS = ["normalized_cr3bp", "km", "km/s", "day", "hour", "count", "dimensionless", "deg"] as const;
export type MetricUnit = (typeof METRIC_UNITS)[number];

/** Why a propagation stopped (phasespace.gravity.propagation.Termination). */
export const TERMINATIONS = [
  "horizon",
  "earth_collision",
  "moon_collision",
  "earth_return_shell",
  "start_inside_earth",
  "start_inside_moon",
  "start_inside_return_shell",
  "non_finite",
  "evaluation_limit",
  "solver_failure",
] as const;
export type Termination = (typeof TERMINATIONS)[number];

/** Free-return feasibility verdicts (phasespace.gravity.free_return.Outcome), in declaration order. */
export const FREE_RETURN_OUTCOMES = [
  "feasible",
  "numerical_failure",
  "invalid_start",
  "moon_collision",
  "earth_collision",
  "no_return_within_t_max",
  "return_before_lunar_encounter",
  "perilune_out_of_band",
] as const;
export type FreeReturnOutcome = (typeof FREE_RETURN_OUTCOMES)[number];

export const ARTIFACT_KINDS = ["trajectory", "sample_table", "outcome_grid"] as const;
export type ArtifactKind = (typeof ARTIFACT_KINDS)[number];

export const COMPARISON_ROLES = ["tight_tolerance", "independent_integrator"] as const;
export type ComparisonRole = (typeof COMPARISON_ROLES)[number];

export interface Source {
  repository: string;
  git_commit: string;
  git_dirty: boolean;
  generator: string;
  generator_version: string;
}

export interface Environment {
  python: string;
  python_implementation: string;
  numpy: string;
  scipy: string;
  platform: string;
  machine: string;
}

export interface Primary {
  x: number;
  y: number;
  /** Radius used for collision detection, km. */
  radius_km: number;
}

export interface Model {
  name: "planar-cr3bp";
  coordinate_frame: "barycentric_synodic";
  units: "normalized_cr3bp";
  mu: number;
  length_scale_km: number;
  length_scale_convention: LengthScaleConvention;
  time_scale_seconds: number;
  velocity_scale_km_s: number;
  system_name: string | null;
  constants_source: string | null;
  earth: Primary;
  moon: Primary;
  assumptions: string[];
}

export interface SolverSettings {
  name: SolverName;
  rtol: number;
  atol: number;
}

export interface CollisionEvents {
  /** null: Earth collisions disabled. */
  earth_radius_km: number | null;
  /** null: Moon collisions disabled. */
  moon_radius_km: number | null;
  function: "distance_minus_radius";
  direction: "decreasing";
  terminal: true;
  grazing_detection: "periapsis_below_radius";
  start_inside: "not_propagated";
}

/** Terminal inward crossing of a sphere about the Earth centre (M3 entry-interface shell). */
export interface ReturnShell {
  radius_km: number;
  altitude_km: number;
  function: "distance_minus_radius";
  direction: "decreasing";
  terminal: true;
  grazing_detection: "periapsis_below_radius";
}

export interface Comparison extends SolverSettings {
  role: ComparisonRole;
}

/** A record of the internal corrector's convergence. Not a validation check: it holds by construction. */
export interface ConvergenceRecord {
  iterations_final_member: number;
  half_period_abs_vx: number;
}

export interface DifferentialCorrection extends SolverSettings {
  half_period_abs_vx_tolerance: number;
  continuation_steps: number;
  initial_guess: "linear_l1_lyapunov_then_x0_continuation";
  convergence: ConvergenceRecord;
}

export interface Solver extends SolverSettings {
  dense_output: boolean;
  max_evaluations: number;
  collision_events: CollisionEvents;
  return_shell: ReturnShell | null;
  /** Extra propagations used only by validation checks; empty when the manifest lists none. */
  comparisons: Comparison[];
  monodromy: SolverSettings | null;
  differential_correction: DifferentialCorrection | null;
}

export interface Artifact {
  kind: ArtifactKind;
  path: string;
  sha256: string;
}

export interface Metric {
  value: number;
  unit: MetricUnit;
  definition: string;
}

export interface ValidationCheck {
  name: string;
  /** null: the computation produced a non-finite value (the check then cannot have passed). */
  observed: number | null;
  threshold: number;
  passed: boolean;
  definition: string;
}

export interface CatalogReference {
  source: string;
  url: string;
  retrieved_on: string;
  family: string;
  libration_point: number;
  row_index: number;
  x0: number;
  vy0: number;
  period: number;
  jacobi: number;
  stability_index: number;
}

export interface Validation {
  checks: ValidationCheck[];
  validated_by: string;
  validated_at: string;
  reference: CatalogReference | null;
}

/** Provenance of a run derived from another run's artifact without new propagations. */
export interface DerivedFrom {
  run_id: string;
  experiment_id: string;
  artifact_path: string;
  artifact_sha256: string;
  git_commit: string;
  git_dirty: boolean;
  derivation: string;
}

export interface Manifest {
  schema_version: 1;
  run_id: string;
  study_id: string;
  experiment_id: string;
  status: RunStatus;
  created_at: string;
  source: Source;
  environment: Environment;
  model: Model;
  solver: Solver;
  random_seed: number | null;
  artifacts: Artifact[];
  metrics: Record<string, Metric>;
  limitations: string[];
  validation: Validation | null;
  termination: Termination | null;
  derived_from: DerivedFrom | null;
}

export const STATE_ORDER = ["t", "x", "y", "vx", "vy"] as const;

/** A marked point of a trajectory (departure, perilune, return-shell crossing, ...). */
export interface TrajectoryEvent {
  name: string;
  t: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** Trajectory samples stored column-wise; every column has the same length (>= 2). */
export interface Trajectory {
  schema_version: 1;
  run_id: string;
  frame: "barycentric_synodic";
  time_unit: "normalized_cr3bp";
  state_order: readonly string[];
  t: Float64Array;
  x: Float64Array;
  y: Float64Array;
  vx: Float64Array;
  vy: Float64Array;
  length: number;
  /** In time order; empty when the artifact lists none. */
  events: TrajectoryEvent[];
}

export interface TableColumn {
  name: string;
  /** null: categorical text column. */
  unit: MetricUnit | null;
  definition: string;
}

export type TableValue = number | string | null;

/** Artifact of kind "sample_table": row-major values, one per column. */
export interface SampleTable {
  schema_version: 1;
  run_id: string;
  columns: TableColumn[];
  rows: TableValue[][];
}

export interface GridAxis {
  name: string;
  unit: MetricUnit;
  min: number;
  step: number;
  count: number;
}

/** Artifact of kind "outcome_grid", decoded: one outcome code per cell, phase-major. */
export interface OutcomeGrid {
  schema_version: 1;
  run_id: string;
  phaseAxis: GridAxis;
  deltaVAxis: GridAxis;
  impulse_angle_deg: number;
  outcomes: FreeReturnOutcome[];
  /** codes[iPhase * deltaVAxis.count + iDeltaV] indexes `outcomes`. */
  codes: Uint8Array;
  feasible: { columns: TableColumn[]; rows: number[][] };
}

export interface LoadedRun {
  manifest: Manifest;
  trajectory: Trajectory;
  manifestUrl: string;
  trajectoryUrl: string;
}

export interface LoadedTableRun {
  manifest: Manifest;
  table: SampleTable;
  manifestUrl: string;
  artifactUrl: string;
}

export interface LoadedGridRun {
  manifest: Manifest;
  grid: OutcomeGrid;
  manifestUrl: string;
  artifactUrl: string;
}
