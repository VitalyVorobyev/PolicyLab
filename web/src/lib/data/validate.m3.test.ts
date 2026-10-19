// @vitest-environment node
import { describe, expect, it } from "vitest";
import { clone, publicJson, publishedRunPath } from "../../test/fixtures";
import { DataError, type DataErrorKind } from "./errors";
import { parseManifest, parseOutcomeGrid, parseSampleTable, parseTrajectory } from "./validate";

type JsonObject = Record<string, unknown>;

const C1 = publishedRunPath((r) => r.run_id === "gravity-free-return-c1-v2");
const GRID = publishedRunPath((r) => r.kind === "outcome_grid");
const PROBE = publishedRunPath((r) => r.kind === "sample_table");

const c1Manifest = publicJson(`/research/${C1}/manifest.json`) as JsonObject;
const c1Trajectory = publicJson(`/research/${C1}/trajectory.json`) as JsonObject;
const gridManifest = publicJson(`/research/${GRID}/manifest.json`) as JsonObject;
const gridJson = publicJson(`/research/${GRID}/outcome_grid.json`) as JsonObject;
const probeManifest = publicJson(`/research/${PROBE}/manifest.json`) as JsonObject;
const probeJson = publicJson(`/research/${PROBE}/samples.json`) as JsonObject;

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

function mutated(value: JsonObject, mutate: (v: JsonObject) => void): JsonObject {
  const copy = clone(value);
  mutate(copy);
  return copy;
}

const obj = (v: unknown): JsonObject => v as JsonObject;
const list = (v: unknown): JsonObject[] => v as JsonObject[];
const first = <T>(items: T[]): T => {
  const item = items[0];
  if (item === undefined) throw new Error("fixture is empty");
  return item;
};

const c1 = parseManifest(c1Manifest);
const gridRunManifest = parseManifest(gridManifest);
const probeRunManifest = parseManifest(probeManifest);

/** base64 of raw bytes (the outcome-grid encoding). */
function base64(bytes: number[]): string {
  return btoa(String.fromCharCode(...bytes));
}

/** A 3 x 2 synthetic grid with known codes, checked against a matching synthetic manifest. */
function syntheticGrid(codes: number[], feasibleRows: number[][]): JsonObject {
  return {
    schema_version: 1,
    run_id: gridRunManifest.run_id,
    phase_axis: { name: "phase_deg", unit: "deg", min: 0, step: 0.5, count: 3 },
    delta_v_axis: { name: "delta_v_km_s", unit: "km/s", min: 3.0, step: 0.002, count: 2 },
    impulse_angle_deg: 0,
    order: "phase_major",
    outcomes: ["feasible", "moon_collision", "no_return_within_t_max"],
    encoding: "base64_uint8",
    codes: base64(codes),
    feasible: {
      columns: [
        { name: "phase_index", unit: "count", definition: "i" },
        { name: "delta_v_index", unit: "count", definition: "j" },
        { name: "perilune_altitude_km", unit: "km", definition: "h" },
      ],
      rows: feasibleRows,
    },
  };
}

const syntheticManifest = { ...gridRunManifest, metrics: {} };

describe("M3 fixtures (published free-return runs)", () => {
  it("parses a validated free-return manifest with the M3 fields", () => {
    expect(c1.status).toBe("validated");
    expect(c1.termination).toBe("earth_return_shell");
    expect(c1.solver.return_shell).toEqual({
      radius_km: 6492.92,
      altitude_km: 121.92,
      function: "distance_minus_radius",
      direction: "decreasing",
      terminal: true,
      grazing_detection: "periapsis_below_radius",
    });
    expect(c1.metrics.departure_phase_deg?.unit).toBe("deg");
    expect(c1.validation?.checks).toHaveLength(list(obj(c1Manifest.validation).checks).length);
    expect(c1.validation?.checks.every((c) => c.passed)).toBe(true);
    expect(c1.derived_from).toBeNull();
  });

  it("parses the trajectory events in time order, inside the sampled range", () => {
    const trajectory = parseTrajectory(c1Trajectory, c1);
    expect(trajectory.events.map((e) => e.name)).toEqual(["departure", "perilune", "return_shell_crossing"]);
    const [departure, , ret] = trajectory.events;
    expect(departure?.t).toBe(trajectory.t[0]);
    expect(ret?.t).toBe(trajectory.t[trajectory.length - 1]);
  });

  it("parses the outcome grid and its derived-from provenance; counts agree with the manifest", () => {
    expect(gridRunManifest.derived_from?.artifact_sha256).toBe("ed13294de9b804213e4f0c5b5a7c0dec13ae2d602f45a0252a1964050f761824");
    expect(gridRunManifest.derived_from?.run_id).toBe("gravity-free-return-map-2d-grid-v1");
    const grid = parseOutcomeGrid(gridJson, gridRunManifest);
    expect(grid.codes).toHaveLength(720 * 126);
    const counts = new Map<string, number>();
    for (const code of grid.codes) counts.set(grid.outcomes[code] ?? "?", (counts.get(grid.outcomes[code] ?? "?") ?? 0) + 1);
    expect(counts.get("feasible")).toBe(161);
    expect(counts.get("no_return_within_t_max")).toBe(64681);
    expect(counts.get("perilune_out_of_band")).toBe(22281);
    expect(counts.get("return_before_lunar_encounter")).toBe(3314);
    expect(counts.get("moon_collision")).toBe(283);
    const sense = grid.feasible.columns.findIndex((c) => c.name === "lunar_passage_sense");
    expect(grid.feasible.rows.filter((row) => (row[sense] ?? 0) > 0)).toHaveLength(5);
  });

  it("decodes a synthetic grid exactly", () => {
    const grid = parseOutcomeGrid(syntheticGrid([2, 0, 1, 2, 0, 2], [[0, 1, 500], [2, 0, 900]]), syntheticManifest);
    expect([...grid.codes]).toEqual([2, 0, 1, 2, 0, 2]);
    expect(grid.feasible.rows).toEqual([[0, 1, 500], [2, 0, 900]]);
  });

  it("parses the probe table", () => {
    const table = parseSampleTable(probeJson, probeRunManifest);
    expect(table.rows).toHaveLength(7002);
  });
});

describe("M3 manifest corruption", () => {
  const shell = (m: JsonObject): JsonObject => obj(obj(m.solver).return_shell);
  it.each([
    ["unknown termination", (m: JsonObject) => (m.termination = "returned"), /termination: expected one of/],
    ["unknown metric unit", (m: JsonObject) => (obj(obj(m.metrics).departure_phase_deg).unit = "rad"), /departure_phase_deg\.unit/],
    ["inconsistent shell altitude", (m: JsonObject) => (shell(m).altitude_km = 121.0), /return_shell\.altitude_km: expected radius_km - Earth radius/],
    ["shell inside the Earth", (m: JsonObject) => (shell(m).radius_km = 6000), /return_shell\.radius_km: expected a radius above the Earth radius/],
    ["non-terminal shell", (m: JsonObject) => (shell(m).terminal = false), /return_shell\.terminal/],
    ["outward shell crossings", (m: JsonObject) => (shell(m).direction = "increasing"), /return_shell\.direction/],
    ["unknown artifact kind", (m: JsonObject) => (first(list(m.artifacts)).kind = "image"), /artifacts\[0\]\.kind/],
    ["undisclosed dirty tree", (m: JsonObject) => (obj(m.source).git_dirty = true), /git_dirty is true but no limitation discloses it/],
  ])("rejects %s", (_name, mutate, message) => {
    expectDataError(() => parseManifest(mutated(c1Manifest, mutate)), "invalid", message);
  });

  it.each([
    ["short source sha256", (m: JsonObject) => (obj(m.derived_from).artifact_sha256 = "ed13"), /derived_from\.artifact_sha256/],
    ["missing derivation text", (m: JsonObject) => delete obj(m.derived_from).derivation, /derived_from\.derivation/],
    ["source run id with capitals", (m: JsonObject) => (obj(m.derived_from).run_id = "Map"), /derived_from\.run_id/],
  ])("rejects derived_from with %s", (_name, mutate, message) => {
    expectDataError(() => parseManifest(mutated(gridManifest, mutate)), "invalid", message);
  });

  it("rejects a derived_from path that leaves the run directory", () => {
    const m = mutated(gridManifest, (x) => (obj(x.derived_from).artifact_path = "../samples.json"));
    expectDataError(() => parseManifest(m), "unsafe_path", /derived_from\.artifact_path/);
  });
});

describe("trajectory event corruption", () => {
  const events = (t: JsonObject): JsonObject[] => list(t.events);
  it.each([
    ["an event after the last sample", (t: JsonObject) => (first(events(t)).t = 99), /events\[0\]\.t: expected a time inside/],
    ["an event state with 3 values", (t: JsonObject) => (first(events(t)).state = [0, 0, 0]), /events\[0\]\.state: expected 4 finite numbers/],
    ["a non-finite event state", (t: JsonObject) => (first(events(t)).state = [0, 0, "NaN", 0]), /events\[0\]\.state/],
    ["events out of time order", (t: JsonObject) => t.events = [...events(t)].reverse(), /must be in time order/],
    ["an event name with capitals", (t: JsonObject) => (first(events(t)).name = "Perilune"), /events\[0\]\.name/],
    ["events that are not an array", (t: JsonObject) => (t.events = {}), /events: expected an array/],
  ])("rejects %s", (_name, mutate, message) => {
    expectDataError(() => parseTrajectory(mutated(c1Trajectory, mutate), c1), "invalid", message);
  });
});

describe("sample table corruption", () => {
  const rows = (t: JsonObject): unknown[][] => t.rows as unknown[][];
  const columns = (t: JsonObject): JsonObject[] => list(t.columns);
  it.each([
    ["a run_id from another run", (t: JsonObject) => (t.run_id = "other"), /does not match the manifest/],
    ["a short row", (t: JsonObject) => first(rows(t)).pop(), /rows\[0\]: expected 20 values/],
    ["text in a numeric column", (t: JsonObject) => (first(rows(t))[0] = "229"), /rows\[0\]\[0\] \(phase_deg\): expected a finite number or null/],
    ["a number in a text column", (t: JsonObject) => (first(rows(t))[3] = 1), /rows\[0\]\[3\] \(outcome\): expected a string or null/],
    ["a duplicate column name", (t: JsonObject) => (at1(columns(t)).name = "phase_deg"), /columns\[1\]: expected a unique column name/],
    ["an unknown column unit", (t: JsonObject) => (first(columns(t)).unit = "rad"), /columns\[0\]\.unit/],
    ["a column without definition", (t: JsonObject) => delete first(columns(t)).definition, /columns\[0\]\.definition/],
  ])("rejects %s", (_name, mutate, message) => {
    expectDataError(() => parseSampleTable(mutated(probeJson, mutate), probeRunManifest), "invalid", message);
  });

  it("reports a table without rows as empty", () => {
    expectDataError(() => parseSampleTable(mutated(probeJson, (t) => (t.rows = [])), probeRunManifest), "empty");
  });
});

function at1<T>(items: T[]): T {
  const item = items[1];
  if (item === undefined) throw new Error("fixture has no second item");
  return item;
}

describe("outcome grid corruption", () => {
  const good = (): JsonObject => syntheticGrid([2, 0, 1, 2, 0, 2], [[0, 1, 500], [2, 0, 900]]);
  it.each([
    ["too few codes", (g: JsonObject) => (g.codes = base64([2, 0, 1, 2, 0])), /codes: expected 6 codes/],
    ["a code outside the outcome table", (g: JsonObject) => (g.codes = base64([2, 0, 1, 2, 0, 3])), /codes below 3/],
    ["codes that are not base64", (g: JsonObject) => (g.codes = "AAAA*AAA"), /codes: expected base64 text/],
    ["an unknown outcome", (g: JsonObject) => (g.outcomes = ["feasible", "crashed", "no_return_within_t_max"]), /outcomes\[1\]/],
    ["duplicate outcomes", (g: JsonObject) => (g.outcomes = ["feasible", "feasible", "no_return_within_t_max"]), /unique outcome names/],
    ["a feasible row on a non-feasible cell", (g: JsonObject) => (obj(g.feasible).rows = [[0, 0, 500], [2, 0, 900]]), /is not a feasible cell/],
    ["feasible rows out of order", (g: JsonObject) => (obj(g.feasible).rows = [[2, 0, 900], [0, 1, 500]]), /strictly increasing cell order/],
    ["a missing feasible row", (g: JsonObject) => (obj(g.feasible).rows = [[0, 1, 500]]), /has 1 rows, but the grid has 2 feasible cells/],
    ["a feasible row outside the grid", (g: JsonObject) => (obj(g.feasible).rows = [[0, 1, 500], [3, 0, 900]]), /outside the grid/],
    ["non-index first columns", (g: JsonObject) => (first(list(obj(g.feasible).columns)).name = "phase"), /first columns "phase_index"/],
    ["a zero axis step", (g: JsonObject) => (obj(g.phase_axis).step = 0), /phase_axis\.step: expected a positive number/],
    ["another cell order", (g: JsonObject) => (g.order = "delta_v_major"), /order/],
    ["another encoding", (g: JsonObject) => (g.encoding = "rle"), /encoding/],
    ["a wrong axis unit", (g: JsonObject) => (obj(g.delta_v_axis).unit = "m/s"), /delta_v_axis\.unit/],
  ])("rejects %s", (_name, mutate, message) => {
    expectDataError(() => parseOutcomeGrid(mutated(good(), mutate), syntheticManifest), "invalid", message);
  });

  it("rejects counts that contradict the manifest's outcome metrics", () => {
    const m = mutated(gridManifest, (x) => (obj(obj(x.metrics).outcome_feasible_count).value = 160));
    expectDataError(() => parseOutcomeGrid(gridJson, parseManifest(m)), "invalid", /outcome_feasible_count records 160/);
  });
});
