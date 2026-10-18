// @vitest-environment node
import { describe, expect, it } from "vitest";
import { clone, publicJson, RUN_PATH } from "../../test/fixtures";
import { DataError, type DataErrorKind } from "./errors";
import { isSafeRelativePath } from "./paths";
import { parseCatalog, parseManifest, parseTrajectory } from "./validate";

type JsonObject = Record<string, unknown>;

/** Checked index access for fixtures (fails the test instead of using a non-null assertion). */
function at<T>(items: T[], index: number): T {
  const item = items[index];
  if (item === undefined) throw new Error(`fixture has no item ${index}`);
  return item;
}

const goodCatalog = publicJson("/research/catalog.json") as JsonObject;
const goodManifest = publicJson(`/research/${RUN_PATH}/manifest.json`) as JsonObject;
const goodTrajectory = publicJson(`/research/${RUN_PATH}/trajectory.json`) as JsonObject;

function expectDataError(fn: () => unknown, kind: DataErrorKind, message?: RegExp): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(DataError);
  expect((caught as DataError).kind).toBe(kind);
  if (message !== undefined) expect((caught as DataError).message).toMatch(message);
}

function manifestWith(mutate: (m: JsonObject) => void): JsonObject {
  const m = clone(goodManifest);
  mutate(m);
  return m;
}

function trajectoryWith(mutate: (t: JsonObject & { samples: unknown[][] }) => void): JsonObject {
  const t = clone(goodTrajectory) as JsonObject & { samples: unknown[][] };
  mutate(t);
  return t;
}

const manifest = parseManifest(goodManifest);

describe("good fixtures (the published reference run)", () => {
  it("parses the catalog", () => {
    const catalog = parseCatalog(goodCatalog);
    expect(catalog.studies.map((s) => s.study_id)).toEqual(["gravity-atlas", "last-library", "algorithm-forge", "constitution-machine"]);
    expect(catalog.studies[0]?.runs.find((r) => r.run_id === "gravity-l1-lyapunov-reference-v2")).toEqual({
      run_id: "gravity-l1-lyapunov-reference-v2",
      path: RUN_PATH,
      kind: "trajectory",
      label: "Earth–Moon L1 Lyapunov orbit (JPL catalog reference)",
    });
  });

  it("parses the manifest with its declared scales, primaries and checks", () => {
    expect(manifest.run_id).toBe("gravity-l1-lyapunov-reference-v2");
    expect(manifest.status).toBe("validated");
    expect(manifest.model.mu).toBe(0.01215058560962404);
    expect(manifest.model.length_scale_km).toBe(389703.264829278);
    expect(manifest.model.length_scale_convention).toBe("jpl_periodic_orbits_catalog");
    expect(manifest.model.earth).toEqual({ x: -0.01215058560962404, y: 0, radius_km: 6371 });
    expect(manifest.model.moon.radius_km).toBe(1737.1);
    expect(manifest.source.git_dirty).toBe(false);
    expect(manifest.source.git_commit).toBe("733e4d6edaba013a88b7c10f8708696d5807caf4");
    expect(manifest.validation?.checks).toHaveLength(12);
    expect(manifest.validation?.checks.every((c) => c.definition.length > 0)).toBe(true);
    expect(manifest.validation?.reference?.stability_index).toBe(904.465406637648);
    expect(manifest.metrics.period_days).toEqual({
      value: 12.688293770809597,
      unit: "day",
      definition: "orbit period; time_scale_seconds = 382981.289129055 s, 1 day = 86400 s",
    });
  });

  it("parses the environment and the solver blocks", () => {
    expect(manifest.environment).toEqual({
      python: "3.12.12",
      python_implementation: "CPython",
      numpy: "2.5.4",
      scipy: "1.18.1",
      platform: "macOS-27.0.1-arm64-arm-64bit",
      machine: "arm64",
    });
    const { solver } = manifest;
    expect(solver.name).toBe("DOP853");
    expect(solver.max_evaluations).toBe(1000000);
    expect(solver.collision_events).toEqual({
      earth_radius_km: 6371,
      moon_radius_km: 1737.1,
      function: "distance_minus_radius",
      direction: "decreasing",
      terminal: true,
      grazing_detection: "periapsis_below_radius",
      start_inside: "not_propagated",
    });
    expect(solver.comparisons.map((c) => [c.role, c.name])).toEqual([
      ["tight_tolerance", "DOP853"],
      ["independent_integrator", "LSODA"],
    ]);
    expect(solver.monodromy).toEqual({ name: "DOP853", rtol: 1e-13, atol: 1e-13 });
    expect(solver.differential_correction?.convergence).toEqual({
      iterations_final_member: 4,
      half_period_abs_vx: 5.637797086815999e-15,
    });
  });

  it("does not treat the corrector's convergence record as a validation check", () => {
    expect(manifest.validation?.checks.map((c) => c.name)).not.toContain("corrector_half_period_abs_vx");
  });

  it("parses the trajectory column-wise", () => {
    const trajectory = parseTrajectory(goodTrajectory, manifest);
    expect(trajectory.length).toBe(1001);
    expect(trajectory.t[0]).toBe(0);
    expect(trajectory.x[0]).toBe(0.8150724480121228);
    expect(trajectory.vy[0]).toBe(0.21717137836685385);
    expect(trajectory.t[1000]).toBeCloseTo(2.8624598979521805, 12);
  });

  it("accepts optional blocks when they are absent", () => {
    const m = parseManifest(
      manifestWith((x) => {
        const solver = x.solver as JsonObject;
        delete solver.comparisons;
        delete solver.monodromy;
        delete solver.differential_correction;
        const model = x.model as JsonObject;
        delete model.system_name;
        delete model.constants_source;
        delete (x.validation as JsonObject).reference;
      }),
    );
    expect(m.solver.comparisons).toEqual([]);
    expect(m.solver.monodromy).toBeNull();
    expect(m.solver.differential_correction).toBeNull();
    expect(m.model.system_name).toBeNull();
    expect(m.model.constants_source).toBeNull();
    expect(m.validation?.reference).toBeNull();
  });

  it("accepts disabled collision bodies (null radius)", () => {
    const m = parseManifest(
      manifestWith((x) => {
        ((x.solver as JsonObject).collision_events as JsonObject).moon_radius_km = null;
      }),
    );
    expect(m.solver.collision_events.moon_radius_km).toBeNull();
  });

  it("accepts mu = 0.5 (equal masses)", () => {
    const m = parseManifest(
      manifestWith((x) => {
        const model = x.model as JsonObject;
        model.mu = 0.5;
        model.primaries = { earth: { x: -0.5, y: 0, radius_km: 6371 }, moon: { x: 0.5, y: 0, radius_km: 1737.1 } };
      }),
    );
    expect(m.model.mu).toBe(0.5);
  });
});

/** Mutations of nested manifest blocks, for the corruption tables below. */
const model = (m: JsonObject): JsonObject => m.model as JsonObject;
const solver = (m: JsonObject): JsonObject => m.solver as JsonObject;
const collision = (m: JsonObject): JsonObject => solver(m).collision_events as JsonObject;
const correction = (m: JsonObject): JsonObject => solver(m).differential_correction as JsonObject;
const checks = (m: JsonObject): JsonObject[] => (m.validation as JsonObject).checks as JsonObject[];
const reference = (m: JsonObject): JsonObject => (m.validation as JsonObject).reference as JsonObject;

describe("schema versions", () => {
  it.each([2, 0, "1", null, undefined])("rejects manifest schema_version %s", (version) => {
    expectDataError(() => parseManifest(manifestWith((m) => (m.schema_version = version))), "schema_version");
  });

  it("rejects an unknown trajectory schema version", () => {
    expectDataError(() => parseTrajectory(trajectoryWith((t) => (t.schema_version = 2)), manifest), "schema_version");
  });

  it("rejects an unknown catalog schema version", () => {
    expectDataError(() => parseCatalog({ ...goodCatalog, schema_version: 3 }), "schema_version");
  });

  it("rejects a non-object document", () => {
    expectDataError(() => parseManifest([1, 2]), "invalid");
    expectDataError(() => parseTrajectory("text", manifest), "invalid");
  });
});

describe("manifest corruption", () => {
  it.each([
    ["missing run_id", (m: JsonObject) => delete m.run_id, /run_id/],
    ["unknown status", (m: JsonObject) => (m.status = "approved"), /status/],
    ["non-finite mu", (m: JsonObject) => ((m.model as JsonObject).mu = null), /model\.mu/],
    ["mu out of range", (m: JsonObject) => ((m.model as JsonObject).mu = 0.7), /mu/],
    ["negative length scale", (m: JsonObject) => ((m.model as JsonObject).length_scale_km = -1), /length_scale_km/],
    ["unknown frame", (m: JsonObject) => ((m.model as JsonObject).coordinate_frame = "inertial"), /coordinate_frame/],
    ["mu above 0.5", (m: JsonObject) => ((m.model as JsonObject).mu = 0.5000001), /mu: expected a mass parameter in \(0, 0\.5\]/],
    ["short commit", (m: JsonObject) => ((m.source as JsonObject).git_commit = "4a7b722"), /git_commit/],
    ["bad checksum format", (m: JsonObject) => (at(m.artifacts as JsonObject[], 0).sha256 = "abc"), /sha256/],
    ["no artifacts", (m: JsonObject) => (m.artifacts = []), /artifacts: expected an array with at least 1 item, got 0 items/],
    [
      "two trajectory artifacts",
      (m: JsonObject) => (m.artifacts = [at(m.artifacts as JsonObject[], 0), at(m.artifacts as JsonObject[], 0)]),
      /expected exactly one artifact .*got 2/,
    ],
    ["metric without unit", (m: JsonObject) => delete ((m.metrics as JsonObject).period_days as JsonObject).unit, /period_days\.unit/],
    ["fractional seed", (m: JsonObject) => (m.random_seed = 1.5), /random_seed/],
    [
      "Earth not at -mu",
      (m: JsonObject) => (((m.model as JsonObject).primaries as JsonObject).earth = { x: 0, y: 0, radius_km: 6371 }),
      /primaries\.earth/,
    ],
    ["run_id with upper case", (m: JsonObject) => (m.run_id = "Gravity-Run"), /run_id/],
    ["created_at with offset", (m: JsonObject) => (m.created_at = "2026-10-18T11:52:59+00:00"), /created_at/],
    ["missing environment", (m: JsonObject) => delete m.environment, /environment: expected an object, got nothing/],
    ["null scipy version", (m: JsonObject) => ((m.environment as JsonObject).scipy = null), /environment\.scipy: expected a non-empty string, got null/],
    ["missing git_dirty", (m: JsonObject) => delete (m.source as JsonObject).git_dirty, /source\.git_dirty/],
    ["repository without owner", (m: JsonObject) => ((m.source as JsonObject).repository = "PolicyLab"), /source\.repository/],
    ["unknown model name", (m: JsonObject) => (model(m).name = "spatial-cr3bp"), /model\.name: expected "planar-cr3bp"/],
    [
      "unknown length scale convention",
      (m: JsonObject) => (model(m).length_scale_convention = "mean_distance"),
      /model\.length_scale_convention: expected one of "nssdca_lunar_semimajor_axis", "jpl_periodic_orbits_catalog", got "mean_distance"/,
    ],
    ["missing length scale convention", (m: JsonObject) => delete model(m).length_scale_convention, /length_scale_convention/],
    ["inconsistent velocity scale", (m: JsonObject) => (model(m).velocity_scale_km_s = 1.0175517), /velocity_scale_km_s/],
    ["missing velocity scale", (m: JsonObject) => delete model(m).velocity_scale_km_s, /velocity_scale_km_s/],
    ["missing primaries", (m: JsonObject) => delete model(m).primaries, /model\.primaries/],
    ["missing Moon radius", (m: JsonObject) => delete ((model(m).primaries as JsonObject).moon as JsonObject).radius_km, /primaries\.moon\.radius_km/],
    ["null system_name", (m: JsonObject) => (model(m).system_name = null), /system_name: expected a non-empty string, got null/],
    ["unknown solver", (m: JsonObject) => (solver(m).name = "Euler"), /solver\.name: expected one of "DOP853"/],
    ["missing max_evaluations", (m: JsonObject) => delete solver(m).max_evaluations, /solver\.max_evaluations/],
    ["fractional max_evaluations", (m: JsonObject) => (solver(m).max_evaluations = 10.5), /solver\.max_evaluations: expected an integer >= 1/],
    ["zero rtol", (m: JsonObject) => (solver(m).rtol = 0), /solver\.rtol/],
    ["missing collision events", (m: JsonObject) => delete solver(m).collision_events, /solver\.collision_events/],
    ["unknown event direction", (m: JsonObject) => (collision(m).direction = "either"), /collision_events\.direction: expected "decreasing"/],
    ["free-text event direction", (m: JsonObject) => (collision(m).direction = "inward (distance decreasing), terminal"), /direction/],
    ["non-terminal events", (m: JsonObject) => (collision(m).terminal = false), /collision_events\.terminal: expected true/],
    ["missing grazing detection", (m: JsonObject) => delete collision(m).grazing_detection, /grazing_detection/],
    ["missing collision radius key", (m: JsonObject) => delete collision(m).earth_radius_km, /earth_radius_km: expected a value or null/],
    ["unknown comparison role", (m: JsonObject) => (at(solver(m).comparisons as JsonObject[], 1).role = "other_integrator"), /comparisons\[1\]\.role/],
    ["unknown comparison integrator", (m: JsonObject) => (at(solver(m).comparisons as JsonObject[], 1).name = "BDF"), /comparisons\[1\]\.name/],
    ["monodromy without atol", (m: JsonObject) => delete (solver(m).monodromy as JsonObject).atol, /monodromy\.atol/],
    ["unknown initial guess", (m: JsonObject) => (correction(m).initial_guess = "linear L1 Lyapunov approximation"), /initial_guess/],
    ["zero continuation steps", (m: JsonObject) => (correction(m).continuation_steps = 0), /continuation_steps/],
    ["missing convergence record", (m: JsonObject) => delete correction(m).convergence, /differential_correction\.convergence/],
    [
      "unconverged corrector",
      (m: JsonObject) => ((correction(m).convergence as JsonObject).half_period_abs_vx = 1e-9),
      /convergence\.half_period_abs_vx: expected a value below half_period_abs_vx_tolerance/,
    ],
    ["unknown artifact kind", (m: JsonObject) => (at(m.artifacts as JsonObject[], 0).kind = "samples"), /artifacts\[0\]\.kind/],
    ["unknown metric unit", (m: JsonObject) => (((m.metrics as JsonObject).period_days as JsonObject).unit = "days"), /period_days\.unit: expected one of/],
    ["metric without definition", (m: JsonObject) => delete ((m.metrics as JsonObject).period_days as JsonObject).definition, /period_days\.definition/],
    ["bare-number metric", (m: JsonObject) => ((m.metrics as JsonObject).period_days = 12.7), /metrics\.period_days: expected an object/],
    ["metric key with a dash", (m: JsonObject) => ((m.metrics as JsonObject)["period-days"] = { value: 1, unit: "day", definition: "x" }), /period-days/],
    ["missing random_seed", (m: JsonObject) => delete m.random_seed, /random_seed: expected a value or null/],
    ["no limitations", (m: JsonObject) => (m.limitations = []), /limitations: expected an array with at least 1 item/],
    ["no checks", (m: JsonObject) => (checks(m).length = 0), /checks: expected an array with at least 1 item/],
    ["check without definition", (m: JsonObject) => delete at(checks(m), 0).definition, /checks\[0\]\.definition/],
    ["check with NaN encoded as a string", (m: JsonObject) => (at(checks(m), 0).observed = "NaN"), /checks\[0\]\.observed: expected a finite number/],
    ["check with missing observed", (m: JsonObject) => delete at(checks(m), 0).observed, /checks\[0\]\.observed: expected a value or null/],
    ["check with null threshold", (m: JsonObject) => (at(checks(m), 0).threshold = null), /checks\[0\]\.threshold/],
    [
      "non-finite check marked as passed",
      (m: JsonObject) => (at(checks(m), 0).observed = null),
      /checks\[0\]\.passed: expected false, because observed is non-finite \(null\)/,
    ],
    ["passed flag that contradicts the threshold", (m: JsonObject) => (at(checks(m), 0).observed = 2e-9), /checks\[0\]\.passed: expected false/],
    ["failed flag that contradicts the threshold", (m: JsonObject) => (at(checks(m), 0).passed = false), /checks\[0\]\.passed: expected true/],
    ["validated_at as a date", (m: JsonObject) => ((m.validation as JsonObject).validated_at = "2026-10-18"), /validated_at/],
    ["non-https reference URL", (m: JsonObject) => (reference(m).url = "http://ssd-api.jpl.nasa.gov/x"), /reference\.url: expected an https URL/],
    ["libration point 6", (m: JsonObject) => (reference(m).libration_point = 6), /libration_point: expected an integer from 1 to 5/],
    ["reference without stability index", (m: JsonObject) => delete reference(m).stability_index, /reference\.stability_index/],
  ])("rejects %s", (_name, mutate, message) => {
    expectDataError(() => parseManifest(manifestWith(mutate)), "invalid", message);
  });

  it("refuses a validated status without recorded checks", () => {
    expectDataError(() => parseManifest(manifestWith((m) => delete m.validation)), "invalid", /no validation checks/);
  });

  it("refuses a validated status when a check failed", () => {
    const m = manifestWith((x) => {
      const check = at(checks(x), 0);
      check.observed = 2e-9;
      check.passed = false;
    });
    expectDataError(() => parseManifest(m), "invalid", /did not pass: periodicity_max_abs_state_error_one_period/);
  });

  it("refuses a validated status when a check value was non-finite", () => {
    const m = manifestWith((x) => {
      const check = at(checks(x), 3);
      check.observed = null;
      check.passed = false;
    });
    expectDataError(() => parseManifest(m), "invalid", /did not pass: tight_tolerance_max_abs_state_difference_samples/);
  });

  it("keeps a non-finite check as null and failed in an unvalidated run", () => {
    const m = parseManifest(
      manifestWith((x) => {
        x.status = "computed_unvalidated";
        const check = at(checks(x), 4);
        check.observed = null;
        check.passed = false;
      }),
    );
    expect(m.validation?.checks[4]).toMatchObject({
      name: "independent_integrator_max_abs_state_difference_samples",
      observed: null,
      passed: false,
    });
  });

  it("allows computed_unvalidated runs without validation evidence", () => {
    const m = parseManifest(
      manifestWith((x) => {
        x.status = "computed_unvalidated";
        delete x.validation;
      }),
    );
    expect(m.validation).toBeNull();
  });
});

describe("artifact path safety", () => {
  it.each([
    "../secrets.json",
    "/etc/passwd",
    "https://evil.example/trajectory.json",
    "//evil.example/x.json",
    "a/../../b.json",
    "dir\\file.json",
    "%2e%2e/x.json",
    "trajectory.json?x=1",
    "./trajectory.json",
    "data/trajectory.json",
    "",
  ])("rejects %j in a manifest", (path) => {
    expectDataError(
      () => parseManifest(manifestWith((m) => (at(m.artifacts as JsonObject[], 0).path = path))),
      path === "" ? "invalid" : "unsafe_path",
    );
  });

  it("rejects an unsafe run path in the catalog", () => {
    const catalog = clone(goodCatalog);
    at(at(catalog.studies as JsonObject[], 0).runs as JsonObject[], 0).path = "../../outside";
    expectDataError(() => parseCatalog(catalog), "unsafe_path");
  });

  it("accepts plain nested relative paths", () => {
    expect(isSafeRelativePath("gravity-atlas/l1-lyapunov-reference-v2")).toBe(true);
    expect(isSafeRelativePath("trajectory.json")).toBe(true);
  });
});

describe("trajectory corruption", () => {
  it("rejects a run_id that differs from the manifest", () => {
    expectDataError(() => parseTrajectory(trajectoryWith((t) => (t.run_id = "other-run")), manifest), "invalid", /does not match/);
  });

  it("reports an empty run", () => {
    expectDataError(() => parseTrajectory(trajectoryWith((t) => (t.samples = [])), manifest), "empty");
  });

  it.each([
    ["null value", (t: { samples: unknown[][] }) => (at(t.samples, 3)[2] = null), /samples\[3\]\[2\]/],
    ["string value", (t: { samples: unknown[][] }) => (at(t.samples, 3)[1] = "0.8"), /samples\[3\]\[1\]/],
    ["overflowed value", (t: { samples: unknown[][] }) => (at(t.samples, 3)[1] = Number.POSITIVE_INFINITY), /finite/],
    ["short row", (t: { samples: unknown[][] }) => at(t.samples, 10).pop(), /samples\[10\]: expected 5 values/],
    ["long row", (t: { samples: unknown[][] }) => at(t.samples, 10).push(0), /samples\[10\]: expected 5 values/],
    ["row that is not an array", (t: { samples: unknown[] }) => (t.samples[4] = { t: 0 }), /samples\[4\]/],
    ["repeated time", (t: { samples: unknown[][] }) => (at(t.samples, 6)[0] = at(t.samples, 5)[0]), /strictly increase/],
    ["decreasing time", (t: { samples: unknown[][] }) => (at(t.samples, 6)[0] = 0), /strictly increase/],
  ])("rejects a %s", (_name, mutate, message) => {
    expectDataError(() => parseTrajectory(trajectoryWith(mutate), manifest), "invalid", message);
  });

  it("rejects columns in any order other than t, x, y, vx, vy", () => {
    const reordered = trajectoryWith((t) => {
      t.state_order = ["x", "t", "y", "vx", "vy"];
      t.samples = t.samples.map((row) => [row[1], row[0], row[2], row[3], row[4]]);
    });
    expectDataError(() => parseTrajectory(reordered, manifest), "invalid", /state_order: expected exactly \["t","x","y","vx","vy"\]/);
  });

  it("rejects a single-sample trajectory", () => {
    expectDataError(() => parseTrajectory(trajectoryWith((t) => (t.samples = t.samples.slice(0, 1))), manifest), "invalid", /at least 2 rows/);
  });

  it("rejects a missing state column", () => {
    const t = trajectoryWith((x) => {
      x.state_order = ["t", "x", "y", "vx", "speed"];
    });
    expectDataError(() => parseTrajectory(t, manifest), "invalid", /state_order/);
  });

  it("rejects duplicate state columns", () => {
    const t = trajectoryWith((x) => {
      x.state_order = ["t", "x", "x", "vx", "vy"];
    });
    expectDataError(() => parseTrajectory(t, manifest), "invalid", /state_order/);
  });

  it("rejects a frame other than the barycentric synodic frame", () => {
    expectDataError(() => parseTrajectory(trajectoryWith((t) => (t.frame = "inertial")), manifest), "invalid", /frame/);
  });
});
