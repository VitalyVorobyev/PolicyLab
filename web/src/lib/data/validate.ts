import { DataError } from "./errors";
import { isSafeRelativePath } from "./paths";
import {
  ARTIFACT_KINDS,
  COMPARISON_ROLES,
  FREE_RETURN_OUTCOMES,
  LENGTH_SCALE_CONVENTIONS,
  METRIC_UNITS,
  RUN_STATUSES,
  SOLVER_NAMES,
  STATE_ORDER,
  TERMINATIONS,
  ATLAS_ARMS,
  SUMMARY_KEYS,
  type Artifact,
  type ArtifactKind,
  type ArmId,
  type Atlas,
  type AtlasArm,
  type AtlasElite,
  type ErrorFan,
  type FanDraw,
  type Polyline,
  type SummaryStat,
  type DerivedFrom,
  type FreeReturnOutcome,
  type GridAxis,
  type OutcomeGrid,
  type ReturnShell,
  type SampleTable,
  type TableColumn,
  type TableValue,
  type TrajectoryEvent,
  type Catalog,
  type CatalogReference,
  type CatalogRun,
  type CatalogStudy,
  type CollisionEvents,
  type Comparison,
  type DifferentialCorrection,
  type Environment,
  type Manifest,
  type Metric,
  type MetricUnit,
  type Model,
  type Primary,
  type Solver,
  type SolverSettings,
  type Source,
  type StudyStatus,
  type Trajectory,
  type Validation,
  type ValidationCheck,
} from "./types";

export const SUPPORTED_SCHEMA_VERSION = 1;

/** Relative precision for the redundant fields the contract asks consumers to cross-check. */
const CONSISTENCY_RTOL = 1e-12;

/** Absolute tolerance (km) for return_shell.altitude_km = radius_km - Earth radius (float subtraction). */
const SHELL_ALTITUDE_ATOL_KM = 1e-9;

const RUN_ID = /^[a-z0-9][a-z0-9._-]{0,127}$/;
const SNAKE_KEY = /^[a-z0-9_]+$/;
const TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const GIT_COMMIT = /^[0-9a-f]{40}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const REPOSITORY = /^[^/\s]+\/[^/\s]+$/;

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A received value already described in words (e.g. "0 items"), printed as is. */
class Described {
  constructor(readonly text: string) {}
}

/** Short description of a received value for error messages. */
function describe(value: unknown): string {
  if (value instanceof Described) return value.text;
  if (value === undefined) return "nothing (the field is missing)";
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  if (typeof value === "object") return "an object";
  if (typeof value === "string") return JSON.stringify(value.length > 60 ? `${value.slice(0, 57)}...` : value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return typeof value;
}

/** Field-level checks that report a dotted path (e.g. `model.mu`) and the received value in every error. */
class Reader {
  constructor(
    private readonly source: string,
    private readonly prefix: string,
    private readonly value: Json,
  ) {}

  /** Throw a contract violation for `key` (or for this object itself when `key` is ""). */
  fail(key: string, expected: string, got: unknown = this.value[key]): never {
    const path = key === "" ? this.prefix.replace(/\.$/, "") : `${this.prefix}${key}`;
    throw new DataError("invalid", this.source, `${path}: expected ${expected}, got ${describe(got)}.`);
  }

  /** Present means "not undefined"; an explicit null is present (and then rejected by the typed getters). */
  has(key: string): boolean {
    return this.value[key] !== undefined;
  }

  keys(): string[] {
    return Object.keys(this.value);
  }

  string(key: string): string {
    const v = this.value[key];
    if (typeof v !== "string" || v.length === 0) this.fail(key, "a non-empty string");
    return v;
  }

  /** An optional (`?`) string: absent gives null; null itself is not allowed. */
  optionalString(key: string): string | null {
    return this.has(key) ? this.string(key) : null;
  }

  matching(key: string, pattern: RegExp, expected: string): string {
    const v = this.value[key];
    if (typeof v !== "string" || !pattern.test(v)) this.fail(key, expected);
    return v;
  }

  finite(key: string): number {
    const v = this.value[key];
    if (typeof v !== "number" || !Number.isFinite(v)) this.fail(key, "a finite number");
    return v;
  }

  positive(key: string): number {
    const v = this.finite(key);
    if (v <= 0) this.fail(key, "a positive number");
    return v;
  }

  nonNegative(key: string): number {
    const v = this.finite(key);
    if (v < 0) this.fail(key, "a non-negative number");
    return v;
  }

  integer(key: string, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY): number {
    const v = this.value[key];
    if (typeof v !== "number" || !Number.isSafeInteger(v) || v < min || v > max) {
      const range = Number.isFinite(max) ? ` from ${min} to ${max}` : Number.isFinite(min) ? ` >= ${min}` : "";
      this.fail(key, `an integer${range}`);
    }
    return v;
  }

  /** A required field whose value may be null (`number | null` in the contract). */
  nullable<T>(key: string, read: (key: string) => T): T | null {
    const v = this.value[key];
    if (v === undefined) this.fail(key, "a value or null");
    return v === null ? null : read(key);
  }

  boolean(key: string): boolean {
    const v = this.value[key];
    if (typeof v !== "boolean") this.fail(key, "true or false");
    return v;
  }

  literal<T extends string | number | boolean>(key: string, expected: T): T {
    const v = this.value[key];
    if (v !== expected) this.fail(key, JSON.stringify(expected));
    return expected;
  }

  oneOf<T extends string>(key: string, allowed: readonly T[]): T {
    const v = this.value[key];
    if (typeof v !== "string" || !(allowed as readonly string[]).includes(v)) {
      this.fail(key, `one of ${allowed.map((a) => `"${a}"`).join(", ")}`);
    }
    return v as T;
  }

  object(key: string): Reader {
    const v = this.value[key];
    if (!isObject(v)) this.fail(key, "an object");
    return new Reader(this.source, `${this.prefix}${key}.`, v);
  }

  array(key: string, minLength = 0): unknown[] {
    const v = this.value[key];
    if (!Array.isArray(v)) this.fail(key, "an array");
    if (v.length < minLength) this.fail(key, `an array with at least ${minLength} item${minLength === 1 ? "" : "s"}`, new Described(`${v.length} items`));
    return v;
  }

  objects(key: string, minLength = 0): Reader[] {
    return this.array(key, minLength).map((item, i) => {
      if (!isObject(item)) this.fail(`${key}[${i}]`, "an object", item);
      return new Reader(this.source, `${this.prefix}${key}[${i}].`, item);
    });
  }

  strings(key: string, minLength = 0): string[] {
    return this.array(key, minLength).map((item, i) => {
      if (typeof item !== "string" || item.length === 0) this.fail(`${key}[${i}]`, "a non-empty string", item);
      return item;
    });
  }

  /** A plain relative path; with `fileName`, a single segment (no "/"). */
  safePath(key: string, { fileName = false }: { fileName?: boolean } = {}): string {
    const v = this.string(key);
    if (!isSafeRelativePath(v) || (fileName && v.includes("/"))) {
      throw new DataError(
        "unsafe_path",
        this.source,
        `${this.prefix}${key}: "${v}" is not a plain ${fileName ? "file name inside the run directory" : "relative path inside the research directory"}.`,
      );
    }
    return v;
  }
}

function root(source: string, value: unknown, what: string): Reader {
  if (!isObject(value)) throw new DataError("invalid", source, `The ${what} is not a JSON object.`);
  const version = value.schema_version;
  if (version !== SUPPORTED_SCHEMA_VERSION) {
    throw new DataError(
      "schema_version",
      source,
      `The ${what} declares schema_version ${version === undefined ? "(missing)" : JSON.stringify(version)}; this site reads only version ${SUPPORTED_SCHEMA_VERSION}.`,
    );
  }
  return new Reader(source, "", value);
}

function close(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= CONSISTENCY_RTOL * Math.abs(expected);
}

const STUDY_STATUSES: readonly StudyStatus[] = ["active", "complete", "planned", "conditional"];

export function parseCatalog(value: unknown, source = "catalog.json"): Catalog {
  const r = root(source, value, "catalog");
  const studies: CatalogStudy[] = r.objects("studies").map((s) => ({
    study_id: s.string("study_id"),
    title: s.string("title"),
    status: s.oneOf("status", STUDY_STATUSES),
    runs: s.objects("runs").map(
      (run): CatalogRun => ({
        run_id: run.string("run_id"),
        path: run.safePath("path"),
        kind: run.oneOf("kind", ARTIFACT_KINDS),
        label: run.string("label"),
      }),
    ),
  }));
  return { schema_version: 1, studies };
}

function parseSource(r: Reader): Source {
  return {
    repository: r.matching("repository", REPOSITORY, 'a repository as "owner/name"'),
    git_commit: r.matching("git_commit", GIT_COMMIT, "a full 40-character lowercase commit SHA"),
    git_dirty: r.boolean("git_dirty"),
    generator: r.string("generator"),
    generator_version: r.string("generator_version"),
  };
}

function parseEnvironment(r: Reader): Environment {
  return {
    python: r.string("python"),
    python_implementation: r.string("python_implementation"),
    numpy: r.string("numpy"),
    scipy: r.string("scipy"),
    platform: r.string("platform"),
    machine: r.string("machine"),
  };
}

function parsePrimary(primaries: Reader, key: "earth" | "moon", expectedX: number): Primary {
  const p = primaries.object(key);
  const primary: Primary = { x: p.finite("x"), y: p.finite("y"), radius_km: p.positive("radius_km") };
  // The contract fixes the primaries at (-mu, 0) and (1 - mu, 0); refuse anything else.
  if (!close(primary.x, expectedX)) p.fail("x", `${expectedX} (${key === "earth" ? "-mu" : "1 - mu"})`);
  if (primary.y !== 0) p.fail("y", "0");
  return primary;
}

function parseModel(r: Reader): Model {
  const mu = r.positive("mu");
  if (mu > 0.5) r.fail("mu", "a mass parameter in (0, 0.5]");
  const length = r.positive("length_scale_km");
  const time = r.positive("time_scale_seconds");
  const velocity = r.positive("velocity_scale_km_s");
  if (!close(velocity, length / time)) {
    r.fail("velocity_scale_km_s", `length_scale_km / time_scale_seconds = ${length / time} (relative precision ${CONSISTENCY_RTOL})`);
  }
  const primaries = r.object("primaries");
  return {
    name: r.literal("name", "planar-cr3bp"),
    coordinate_frame: r.literal("coordinate_frame", "barycentric_synodic"),
    units: r.literal("units", "normalized_cr3bp"),
    mu,
    length_scale_km: length,
    length_scale_convention: r.oneOf("length_scale_convention", LENGTH_SCALE_CONVENTIONS),
    time_scale_seconds: time,
    velocity_scale_km_s: velocity,
    system_name: r.optionalString("system_name"),
    constants_source: r.optionalString("constants_source"),
    earth: parsePrimary(primaries, "earth", -mu),
    moon: parsePrimary(primaries, "moon", 1 - mu),
    assumptions: r.strings("assumptions"),
  };
}

function parseSettings(r: Reader): SolverSettings {
  return { name: r.oneOf("name", SOLVER_NAMES), rtol: r.positive("rtol"), atol: r.positive("atol") };
}

function parseCollisionEvents(r: Reader): CollisionEvents {
  return {
    earth_radius_km: r.nullable("earth_radius_km", (k) => r.positive(k)),
    moon_radius_km: r.nullable("moon_radius_km", (k) => r.positive(k)),
    function: r.literal("function", "distance_minus_radius"),
    direction: r.literal("direction", "decreasing"),
    terminal: r.literal("terminal", true),
    grazing_detection: r.literal("grazing_detection", "periapsis_below_radius"),
    start_inside: r.literal("start_inside", "not_propagated"),
  };
}

function parseDifferentialCorrection(r: Reader): DifferentialCorrection {
  const tolerance = r.positive("half_period_abs_vx_tolerance");
  const c = r.object("convergence");
  const convergence = {
    iterations_final_member: c.integer("iterations_final_member", 1),
    half_period_abs_vx: c.nonNegative("half_period_abs_vx"),
  };
  if (convergence.half_period_abs_vx >= tolerance) {
    c.fail("half_period_abs_vx", `a value below half_period_abs_vx_tolerance ${tolerance} (the corrector did not converge)`);
  }
  return {
    ...parseSettings(r),
    half_period_abs_vx_tolerance: tolerance,
    continuation_steps: r.integer("continuation_steps", 1),
    initial_guess: r.literal("initial_guess", "linear_l1_lyapunov_then_x0_continuation"),
    convergence,
  };
}

function parseReturnShell(r: Reader): ReturnShell {
  return {
    radius_km: r.positive("radius_km"),
    altitude_km: r.positive("altitude_km"),
    function: r.literal("function", "distance_minus_radius"),
    direction: r.literal("direction", "decreasing"),
    terminal: r.literal("terminal", true),
    grazing_detection: r.literal("grazing_detection", "periapsis_below_radius"),
  };
}

function parseSolver(r: Reader): Solver {
  return {
    ...parseSettings(r),
    dense_output: r.boolean("dense_output"),
    max_evaluations: r.integer("max_evaluations", 1),
    collision_events: parseCollisionEvents(r.object("collision_events")),
    return_shell: r.has("return_shell") ? parseReturnShell(r.object("return_shell")) : null,
    comparisons: r.has("comparisons")
      ? r.objects("comparisons").map((c): Comparison => ({ role: c.oneOf("role", COMPARISON_ROLES), ...parseSettings(c) }))
      : [],
    monodromy: r.has("monodromy") ? parseSettings(r.object("monodromy")) : null,
    differential_correction: r.has("differential_correction")
      ? parseDifferentialCorrection(r.object("differential_correction"))
      : null,
  };
}

function parseArtifact(a: Reader): Artifact {
  return {
    kind: a.oneOf("kind", ARTIFACT_KINDS),
    // Artifacts live directly in the run directory: a plain file name, no "/" segments.
    path: a.safePath("path", { fileName: true }),
    sha256: a.matching("sha256", SHA256, "64 lowercase hex digits"),
  };
}

function parseMetrics(r: Reader): Record<string, Metric> {
  const metrics: Record<string, Metric> = {};
  for (const key of r.keys()) {
    if (!SNAKE_KEY.test(key)) r.fail(key, "a metric key of lowercase letters, digits and underscores", key);
    const m = r.object(key);
    metrics[key] = { value: m.finite("value"), unit: m.oneOf("unit", METRIC_UNITS), definition: m.string("definition") };
  }
  return metrics;
}

function parseCheck(c: Reader): ValidationCheck {
  const check: ValidationCheck = {
    name: c.matching("name", SNAKE_KEY, "a check name of lowercase letters, digits and underscores"),
    observed: c.nullable("observed", (k) => c.finite(k)),
    threshold: c.finite("threshold"),
    passed: c.boolean("passed"),
    definition: c.string("definition"),
  };
  // passed === (observed !== null && observed <= threshold): a non-finite (null) value never passes.
  const expected = check.observed !== null && check.observed <= check.threshold;
  if (check.passed !== expected) {
    c.fail(
      "passed",
      `${String(expected)}, because observed ${check.observed === null ? "is non-finite (null)" : `${check.observed} ${expected ? "<=" : ">"} threshold ${check.threshold}`}`,
    );
  }
  return check;
}

function parseReference(r: Reader): CatalogReference {
  return {
    source: r.string("source"),
    url: r.matching("url", /^https:\/\/\S+$/, "an https URL"),
    retrieved_on: r.matching("retrieved_on", DATE, 'a date as "YYYY-MM-DD"'),
    family: r.string("family"),
    libration_point: r.integer("libration_point", 1, 5),
    row_index: r.integer("row_index", 0),
    x0: r.finite("x0"),
    vy0: r.finite("vy0"),
    period: r.finite("period"),
    jacobi: r.finite("jacobi"),
    stability_index: r.finite("stability_index"),
  };
}

function parseValidation(r: Reader): Validation {
  return {
    checks: r.objects("checks", 1).map(parseCheck),
    validated_by: r.string("validated_by"),
    validated_at: r.matching("validated_at", TIMESTAMP, 'a UTC time as "YYYY-MM-DDTHH:MM:SSZ"'),
    reference: r.has("reference") ? parseReference(r.object("reference")) : null,
  };
}

function parseDerivedFrom(r: Reader): DerivedFrom {
  return {
    run_id: r.matching("run_id", RUN_ID, "a run id of lowercase letters, digits, '.', '_' and '-' (at most 128)"),
    experiment_id: r.string("experiment_id"),
    artifact_path: r.safePath("artifact_path", { fileName: true }),
    artifact_sha256: r.matching("artifact_sha256", SHA256, "64 lowercase hex digits"),
    git_commit: r.matching("git_commit", GIT_COMMIT, "a full 40-character lowercase commit SHA"),
    git_dirty: r.boolean("git_dirty"),
    derivation: r.string("derivation"),
  };
}

export function parseManifest(value: unknown, source = "manifest.json"): Manifest {
  const r = root(source, value, "manifest");
  const manifest: Manifest = {
    schema_version: 1,
    run_id: r.matching("run_id", RUN_ID, "a run id of lowercase letters, digits, '.', '_' and '-' (at most 128)"),
    study_id: r.string("study_id"),
    experiment_id: r.string("experiment_id"),
    status: r.oneOf("status", RUN_STATUSES),
    created_at: r.matching("created_at", TIMESTAMP, 'a UTC time as "YYYY-MM-DDTHH:MM:SSZ"'),
    source: parseSource(r.object("source")),
    environment: parseEnvironment(r.object("environment")),
    model: parseModel(r.object("model")),
    solver: parseSolver(r.object("solver")),
    random_seed: r.nullable("random_seed", (k) => r.integer(k)),
    artifacts: r.objects("artifacts", 1).map(parseArtifact),
    metrics: parseMetrics(r.object("metrics")),
    limitations: r.strings("limitations", 1),
    validation: r.has("validation") ? parseValidation(r.object("validation")) : null,
    termination: r.has("termination") ? r.oneOf("termination", TERMINATIONS) : null,
    derived_from: r.has("derived_from") ? parseDerivedFrom(r.object("derived_from")) : null,
  };

  const shell = manifest.solver.return_shell;
  if (shell !== null) {
    const expected = shell.radius_km - manifest.model.earth.radius_km;
    if (!(shell.radius_km > manifest.model.earth.radius_km)) {
      r.object("solver").object("return_shell").fail("radius_km", `a radius above the Earth radius ${manifest.model.earth.radius_km} km`);
    }
    if (Math.abs(shell.altitude_km - expected) > SHELL_ALTITUDE_ATOL_KM) {
      r.object("solver").object("return_shell").fail("altitude_km", `radius_km - Earth radius = ${expected} km`);
    }
  }
  // The contract requires a dirty working tree to be disclosed, because git_commit alone then does not identify the code.
  if (manifest.source.git_dirty && !manifest.limitations.some((l) => l.includes("git_dirty"))) {
    throw new DataError("invalid", source, "source.git_dirty is true but no limitation discloses it (expected a limitation mentioning git_dirty).");
  }

  if (manifest.status === "validated") {
    if (manifest.validation === null) {
      throw new DataError("invalid", source, 'status is "validated" but the manifest records no validation checks.');
    }
    const failed = manifest.validation.checks.filter((c) => !c.passed).map((c) => c.name);
    if (failed.length > 0) {
      throw new DataError("invalid", source, `status is "validated" but these checks did not pass: ${failed.join(", ")}.`);
    }
  }
  // Single-artifact runs, or an atlas run: its atlas artifact first, then one error fan per arm.
  const [first, ...rest] = manifest.artifacts;
  const isAtlasRun = first?.kind === "atlas" && rest.length > 0 && rest.every((a) => a.kind === "error_fan");
  if (manifest.artifacts.length !== 1 && !isAtlasRun) {
    throw new DataError("invalid", source, `artifacts: expected exactly one artifact (this site shows single-artifact runs), got ${manifest.artifacts.length}.`);
  }
  return manifest;
}

/**
 * The run's artifact of the given kind; undefined otherwise. Single-artifact runs have one; an
 * atlas run has one "atlas" and several "error_fan" artifacts, told apart by `fileName`.
 */
export function artifactOf(manifest: Manifest, kind: ArtifactKind, fileName?: string): Artifact | undefined {
  if (manifest.artifacts.length === 1) {
    const only = manifest.artifacts[0];
    return only?.kind === kind && (fileName === undefined || only.path === fileName) ? only : undefined;
  }
  return manifest.artifacts.find((a) => a.kind === kind && (fileName === undefined || a.path === fileName));
}

/** The run's single trajectory artifact; undefined for other kinds. */
export function trajectoryArtifact(manifest: Manifest): Artifact | undefined {
  return artifactOf(manifest, "trajectory");
}

export function parseTrajectory(value: unknown, manifest: Manifest, source = "trajectory.json"): Trajectory {
  const r = root(source, value, "trajectory");
  const runId = r.string("run_id");
  if (runId !== manifest.run_id) {
    throw new DataError("invalid", source, `run_id "${runId}" does not match the manifest's run_id "${manifest.run_id}".`);
  }
  const frame = r.literal("frame", "barycentric_synodic");
  const timeUnit = r.literal("time_unit", "normalized_cr3bp");
  const order = r.strings("state_order");
  if (order.length !== STATE_ORDER.length || order.some((name, i) => name !== STATE_ORDER[i])) {
    r.fail("state_order", `exactly ${JSON.stringify(STATE_ORDER)}`, new Described(JSON.stringify(order)));
  }

  const rows = r.array("samples");
  if (rows.length === 0) throw new DataError("empty", source, "The trajectory contains no samples.");
  if (rows.length < 2) r.fail("samples", "at least 2 rows", new Described("1 row"));
  const n = rows.length;
  const columns = STATE_ORDER.map(() => new Float64Array(n));
  const [t, x, y, vx, vy] = columns as [Float64Array, Float64Array, Float64Array, Float64Array, Float64Array];
  for (let i = 0; i < n; i++) {
    const row: unknown = rows[i];
    if (!Array.isArray(row) || row.length !== STATE_ORDER.length) {
      throw new DataError("invalid", source, `samples[${i}]: expected ${STATE_ORDER.length} values to match state_order.`);
    }
    row.forEach((v: unknown, j) => {
      if (typeof v !== "number" || !Number.isFinite(v)) {
        throw new DataError("invalid", source, `samples[${i}][${j}] (${STATE_ORDER[j] ?? "?"}): expected a finite number, got ${describe(v)}.`);
      }
      const column = columns[j];
      if (column !== undefined) column[i] = v;
    });
    if (i > 0 && !((t[i] ?? 0) > (t[i - 1] ?? 0))) {
      throw new DataError("invalid", source, `samples[${i}]: time ${t[i]} does not strictly increase after ${t[i - 1]}.`);
    }
  }
  const t0 = t[0] ?? 0;
  const tN = t[n - 1] ?? 0;
  const events: TrajectoryEvent[] = r.has("events")
    ? r.objects("events").map((e): TrajectoryEvent => {
        const time = e.finite("t");
        if (time < t0 || time > tN) e.fail("t", `a time inside the samples' range [${t0}, ${tN}]`);
        const state = e.array("state");
        if (state.length !== 4 || !state.every((v) => typeof v === "number" && Number.isFinite(v))) {
          e.fail("state", "4 finite numbers (x, y, vx, vy)", new Described(JSON.stringify(state)));
        }
        const [x0, y0, vx0, vy0] = state as [number, number, number, number];
        return { name: e.matching("name", SNAKE_KEY, "an event name of lowercase letters, digits and underscores"), t: time, x: x0, y: y0, vx: vx0, vy: vy0 };
      })
    : [];
  events.forEach((event, i) => {
    const previous = events[i - 1];
    if (previous !== undefined && event.t < previous.t) {
      throw new DataError("invalid", source, `events[${i}] (${event.name}): time ${event.t} is before the previous event's ${previous.t}; events must be in time order.`);
    }
  });
  return { schema_version: 1, run_id: runId, frame, time_unit: timeUnit, state_order: order, t, x, y, vx, vy, length: n, events };
}

function checkRunId(r: Reader, manifest: Manifest, source: string): string {
  const runId = r.string("run_id");
  if (runId !== manifest.run_id) {
    throw new DataError("invalid", source, `run_id "${runId}" does not match the manifest's run_id "${manifest.run_id}".`);
  }
  return runId;
}

function parseColumns(r: Reader, key: string, minLength: number): TableColumn[] {
  const columns = r.objects(key, minLength).map(
    (c): TableColumn => ({
      name: c.matching("name", SNAKE_KEY, "a column name of lowercase letters, digits and underscores"),
      unit: c.nullable("unit", (k) => c.oneOf(k, METRIC_UNITS)),
      definition: c.string("definition"),
    }),
  );
  const seen = new Set<string>();
  columns.forEach((c, i) => {
    if (seen.has(c.name)) r.fail(`${key}[${i}]`, "a unique column name", c.name);
    seen.add(c.name);
  });
  return columns;
}

/** Artifact of kind "sample_table": numeric columns hold finite numbers or null, text columns strings or null. */
export function parseSampleTable(value: unknown, manifest: Manifest, source = "samples.json"): SampleTable {
  const r = root(source, value, "sample table");
  const runId = checkRunId(r, manifest, source);
  const columns = parseColumns(r, "columns", 1);
  const rows = r.array("rows");
  if (rows.length === 0) throw new DataError("empty", source, "The table contains no rows.");
  const out: TableValue[][] = rows.map((row: unknown, i) => {
    if (!Array.isArray(row) || row.length !== columns.length) {
      throw new DataError("invalid", source, `rows[${i}]: expected ${columns.length} values, one per column.`);
    }
    return row.map((v: unknown, j): TableValue => {
      const column = columns[j];
      if (v === null || column === undefined) return null;
      const ok = column.unit === null ? typeof v === "string" : typeof v === "number" && Number.isFinite(v);
      if (!ok) {
        throw new DataError(
          "invalid",
          source,
          `rows[${i}][${j}] (${column.name}): expected ${column.unit === null ? "a string" : "a finite number"} or null, got ${describe(v)}.`,
        );
      }
      return v as number | string;
    });
  });
  return { schema_version: 1, run_id: runId, columns, rows: out };
}

function parseAxis(r: Reader, name: string, unit: MetricUnit): GridAxis {
  return {
    name: r.literal("name", name),
    unit: r.literal("unit", unit),
    min: r.finite("min"),
    step: r.positive("step"),
    count: r.integer("count", 1),
  };
}

const BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

function decodeBase64(text: string): Uint8Array | null {
  if (!BASE64.test(text)) return null;
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/**
 * Artifact of kind "outcome_grid": decodes the base64 codes and checks the cell count, the code
 * range, the feasible rows against the feasible cells, and the counts against the manifest's
 * `outcome_<name>_count` metrics where present.
 */
export function parseOutcomeGrid(value: unknown, manifest: Manifest, source = "outcome_grid.json"): OutcomeGrid {
  const r = root(source, value, "outcome grid");
  const runId = checkRunId(r, manifest, source);
  const phaseAxis = parseAxis(r.object("phase_axis"), "phase_deg", "deg");
  const deltaVAxis = parseAxis(r.object("delta_v_axis"), "delta_v_km_s", "km/s");
  const impulse = r.finite("impulse_angle_deg");
  r.literal("order", "phase_major");
  r.literal("encoding", "base64_uint8");
  const outcomes = r.array("outcomes", 1).map((o, i): FreeReturnOutcome => {
    if (typeof o !== "string" || !(FREE_RETURN_OUTCOMES as readonly string[]).includes(o)) {
      r.fail(`outcomes[${i}]`, `one of ${FREE_RETURN_OUTCOMES.map((a) => `"${a}"`).join(", ")}`, o);
    }
    return o as FreeReturnOutcome;
  });
  if (new Set(outcomes).size !== outcomes.length) r.fail("outcomes", "unique outcome names", new Described(JSON.stringify(outcomes)));
  const cells = phaseAxis.count * deltaVAxis.count;
  const decoded = decodeBase64(r.string("codes"));
  if (decoded === null) return r.fail("codes", "base64 text (RFC 4648, padded)", new Described("other text"));
  const codes = decoded;
  if (codes.length !== cells) r.fail("codes", `${cells} codes (phase_axis.count × delta_v_axis.count)`, new Described(`${codes.length} codes`));
  const counts = new Array<number>(outcomes.length).fill(0);
  for (let k = 0; k < codes.length; k++) {
    const code = codes[k] ?? 0;
    if (code >= outcomes.length) r.fail("codes", `codes below ${outcomes.length} (the outcome table size)`, new Described(`code ${code} at cell ${k}`));
    counts[code] = (counts[code] ?? 0) + 1;
  }
  outcomes.forEach((outcome, code) => {
    const metric = manifest.metrics[`outcome_${outcome}_count`];
    if (metric !== undefined && metric.value !== counts[code]) {
      throw new DataError("invalid", source, `The grid has ${counts[code]} "${outcome}" cells, but the manifest metric outcome_${outcome}_count records ${metric.value}.`);
    }
  });

  const f = r.object("feasible");
  const columns = parseColumns(f, "columns", 2);
  if (columns[0]?.name !== "phase_index" || columns[1]?.name !== "delta_v_index") {
    f.fail("columns", 'first columns "phase_index" and "delta_v_index"', new Described(JSON.stringify(columns.map((c) => c.name))));
  }
  const feasibleCode = outcomes.indexOf("feasible");
  let previous = -1;
  const rows = f.array("rows").map((row: unknown, i): number[] => {
    if (!Array.isArray(row) || row.length !== columns.length || !row.every((v) => typeof v === "number" && Number.isFinite(v))) {
      throw new DataError("invalid", source, `feasible.rows[${i}]: expected ${columns.length} finite numbers, one per column.`);
    }
    const values = row as number[];
    const [iPhase = -1, iDv = -1] = values;
    if (!Number.isInteger(iPhase) || !Number.isInteger(iDv) || iPhase < 0 || iPhase >= phaseAxis.count || iDv < 0 || iDv >= deltaVAxis.count) {
      throw new DataError("invalid", source, `feasible.rows[${i}]: grid indices (${iPhase}, ${iDv}) are outside the grid.`);
    }
    const cell = iPhase * deltaVAxis.count + iDv;
    if (cell <= previous) throw new DataError("invalid", source, `feasible.rows[${i}]: rows must be in strictly increasing cell order.`);
    previous = cell;
    if (codes[cell] !== feasibleCode) {
      throw new DataError("invalid", source, `feasible.rows[${i}]: cell (${iPhase}, ${iDv}) is not a feasible cell.`);
    }
    return values;
  });
  const feasibleCount = feasibleCode < 0 ? 0 : (counts[feasibleCode] ?? 0);
  if (rows.length !== feasibleCount) {
    throw new DataError("invalid", source, `feasible.rows has ${rows.length} rows, but the grid has ${feasibleCount} feasible cells.`);
  }
  return { schema_version: 1, run_id: runId, phaseAxis, deltaVAxis, impulse_angle_deg: impulse, outcomes, codes, feasible: { columns, rows } };
}


// ---------------------------------------------------------------------------------------------
// M4 demo atlas: kinds "atlas" and "error_fan" (docs/data-contract.md, "M4 demo atlas").
// ---------------------------------------------------------------------------------------------

/** A display polyline: equal-length finite arrays, at least 2 points, time strictly increasing. */
function parsePolyline(r: Reader): Polyline {
  const number = (key: string): Float64Array => {
    const values = r.array(key, 2);
    const out = new Float64Array(values.length);
    values.forEach((v, i) => {
      if (typeof v !== "number" || !Number.isFinite(v)) r.fail(`${key}[${i}]`, "a finite number", v);
      out[i] = v;
    });
    return out;
  };
  const t = number("t_days");
  const x = number("x_km");
  const y = number("y_km");
  if (x.length !== t.length || y.length !== t.length) {
    r.fail("", "t_days, x_km and y_km of equal length", new Described(`${t.length}, ${x.length} and ${y.length} values`));
  }
  for (let i = 1; i < t.length; i++) {
    if (!((t[i] ?? 0) > (t[i - 1] ?? 0))) r.fail("t_days", "strictly increasing times", new Described(`${t[i]} after ${t[i - 1]} at index ${i}`));
  }
  return { t_days: t, x_km: x, y_km: y };
}

function parseCell(r: Reader, key: string, raw: unknown, bins: [number, number]): [number, number] {
  const [i, j] = Array.isArray(raw) ? (raw as unknown[]) : [];
  const ok = (v: unknown, n: number): v is number => typeof v === "number" && Number.isInteger(v) && v >= 0 && v < n;
  if (!Array.isArray(raw) || raw.length !== 2 || !ok(i, bins[0]) || !ok(j, bins[1])) {
    r.fail(key, `a [d1, d2] cell inside the ${bins[0]} x ${bins[1]} grid`, new Described(JSON.stringify(raw)));
  }
  return [i, j];
}

function parseEdges(r: Reader, key: string, bins: number): number[] {
  const edges = r.array(key).map((v, i) => {
    if (typeof v !== "number" || !Number.isFinite(v)) r.fail(`${key}[${i}]`, "a finite number", v);
    return v;
  });
  if (edges.length !== bins + 1) r.fail(key, `${bins + 1} edges (bins + 1)`, new Described(`${edges.length} edges`));
  if (edges.some((e, i) => i > 0 && !(e > (edges[i - 1] ?? 0)))) r.fail(key, "strictly increasing edges", new Described("edges that do not increase"));
  return edges;
}

function parseSummaryStat(r: Reader): SummaryStat {
  const stat = (key: string): number | null => r.nullable(key, (k) => r.finite(k));
  return { mean: stat("mean"), median: stat("median"), sd: stat("sd"), min: stat("min"), max: stat("max") };
}

function parseAtlasElite(e: Reader, bins: [number, number], draws: number): AtlasElite {
  const survival = e.finite("survival");
  if (survival < 0 || survival > 1) e.fail("survival", "a fraction between 0 and 1");
  const perilune = e.object("perilune");
  return {
    cell: parseCell(e, "cell", e.array("cell"), bins),
    index: e.integer("index", 0),
    theta_deg: e.finite("theta_deg"),
    delta_v_km_s: e.finite("delta_v_km_s"),
    alpha_deg: e.finite("alpha_deg"),
    perilune_altitude_km: e.positive("perilune_altitude_km"),
    return_time_days: e.positive("return_time_days"),
    entry_flight_path_angle_deg: e.finite("entry_flight_path_angle_deg"),
    survival,
    uncorrected_feasible_draws: e.integer("uncorrected_feasible_draws", 0, draws),
    perilune: { t_days: perilune.finite("t_days"), x_km: perilune.finite("x_km"), y_km: perilune.finite("y_km") },
    path: parsePolyline(e.object("path")),
  };
}

function parseBody(r: Reader): { x_km: number; y_km: number; radius_km: number } {
  return { x_km: r.finite("x_km"), y_km: r.finite("y_km"), radius_km: r.positive("radius_km") };
}

/** Artifact of kind "atlas": the seven arms' seed-1001 catalogs, 16-seed summaries and the grid. */
export function parseAtlas(value: unknown, manifest: Manifest, source = "atlas.json"): Atlas {
  const r = root(source, value, "atlas");
  const runId = checkRunId(r, manifest, source);
  r.literal("frame", "barycentric_synodic");
  r.literal("length_unit", "km");
  r.literal("time_unit", "day");
  const draws = r.integer("draws", 1);

  const g = r.object("grid");
  const d1 = g.object("d1");
  const d2 = g.object("d2");
  const bins1 = d1.integer("bins", 1);
  const bins2 = d2.integer("bins", 1);
  const bins: [number, number] = [bins1, bins2];
  const grid = {
    d1: { bins: bins1, edges: parseEdges(d1, "edges", bins1), edges_km: parseEdges(d1, "edges_km", bins1) },
    d2: { bins: bins2, edges: parseEdges(d2, "edges", bins2), edges_days: parseEdges(d2, "edges_days", bins2) },
    reachable_cells: g.array("reachable_cells", 1).map((raw, i) => parseCell(g, `reachable_cells[${i}]`, raw, bins)),
  };
  const reachable = new Set(grid.reachable_cells.map(([i, j]) => `${i},${j}`));
  if (reachable.size !== grid.reachable_cells.length) g.fail("reachable_cells", "distinct cells");

  const arms: AtlasArm[] = r.objects("arms").map((a, n): AtlasArm => {
    const arm = a.oneOf("arm", ATLAS_ARMS);
    if (arm !== ATLAS_ARMS[n]) a.fail("arm", `"${ATLAS_ARMS[n] ?? "?"}" (arms come in the order ${ATLAS_ARMS.join(", ")})`);
    const summary = a.object("summary");
    const seen = new Set<string>();
    const elites = a.objects("elites").map((e) => {
      const elite = parseAtlasElite(e, bins, draws);
      const key = `${elite.cell[0]},${elite.cell[1]}`;
      if (!reachable.has(key) || seen.has(key)) e.fail("cell", "a distinct reachable cell", new Described(key));
      seen.add(key);
      return elite;
    });
    return {
      arm,
      fan_path: a.safePath("fan_path", { fileName: true }),
      summary: Object.fromEntries(SUMMARY_KEYS.map((k) => [k, parseSummaryStat(summary.object(k))])) as AtlasArm["summary"],
      elites,
    };
  });
  if (arms.length !== ATLAS_ARMS.length) r.fail("arms", `${ATLAS_ARMS.length} arms`, new Described(`${arms.length} arms`));

  return {
    schema_version: 1,
    run_id: runId,
    earth: parseBody(r.object("earth")),
    moon: parseBody(r.object("moon")),
    parking_orbit_radius_km: r.positive("parking_orbit_radius_km"),
    seed: r.integer("seed"),
    draws,
    budget_m_s: r.positive("budget_m_s"),
    grid,
    contrasts: r.objects("contrasts").map((c) => ({ endpoint: c.oneOf("endpoint", ["P1", "P2"] as const), baseline: c.string("baseline"), p_holm: c.finite("p_holm") })),
    arms,
  };
}

/** Artifact of kind "error_fan": every elite's error-draw paths, without correction. */
export function parseErrorFan(value: unknown, manifest: Manifest, source = "fan.json"): ErrorFan {
  const r = root(source, value, "error fan");
  const runId = checkRunId(r, manifest, source);
  const arm: ArmId = r.oneOf("arm", ATLAS_ARMS);
  r.literal("frame", "barycentric_synodic");
  r.literal("length_unit", "km");
  r.literal("time_unit", "day");
  // The contract types `error_draws` as the draw list; the published file stores the draw count.
  const raw = (value as Json).error_draws;
  const draws = Array.isArray(raw) ? raw.length : r.integer("error_draws", 1);
  const elites = r.objects("elites").map((e) => {
    const cell = e.array("cell");
    if (cell.length !== 2 || !cell.every((v) => typeof v === "number" && Number.isInteger(v) && v >= 0)) {
      e.fail("cell", "a [d1, d2] cell", new Described(JSON.stringify(cell)));
    }
    const list = e.objects("draws").map(
      (d): FanDraw => ({
        outcome: d.oneOf("outcome", FREE_RETURN_OUTCOMES),
        uncorrected_feasible: d.boolean("uncorrected_feasible"),
        survived: d.boolean("survived"),
        ...parsePolyline(d),
      }),
    );
    if (list.length !== draws) e.fail("draws", `${draws} draws`, new Described(`${list.length} draws`));
    return { cell: cell as [number, number], index: e.integer("index", 0), draws: list };
  });
  return { schema_version: 1, run_id: runId, arm, draws, elites };
}

/** The fan must describe exactly the arm's catalog, in the same order; anything else would mislabel paths. */
export function checkFanMatchesArm(fan: ErrorFan, arm: AtlasArm, source: string): void {
  if (fan.arm !== arm.arm) throw new DataError("invalid", source, `The fan is for arm "${fan.arm}", but the atlas lists it under "${arm.arm}".`);
  const same =
    fan.elites.length === arm.elites.length &&
    fan.elites.every((f, k) => {
      const e = arm.elites[k];
      return e !== undefined && f.index === e.index && f.cell[0] === e.cell[0] && f.cell[1] === e.cell[1];
    });
  if (!same) throw new DataError("invalid", source, `The fan's trajectories (cells and indices) differ from arm "${arm.arm}" in the atlas.`);
}
