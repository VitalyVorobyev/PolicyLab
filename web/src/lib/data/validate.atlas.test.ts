// @vitest-environment node
import { describe, expect, it } from "vitest";
import { clone, publicBytes, publicJson, staticFetcher } from "../../test/fixtures";
import { DataError } from "./errors";
import { loadAtlasRun, loadErrorFan } from "./load";
import { sha256Hex } from "./sha256";
import { checkFanMatchesArm, parseAtlas, parseErrorFan, parseManifest } from "./validate";

type J = Record<string, unknown>;
const DIR = "gravity-atlas/m4-demo-atlas-v1";
function elite0(a: J): J {
  const elite = ((a.arms as J[])[5]?.elites as J[] | undefined)?.[0];
  if (elite === undefined) throw new Error("no elite");
  return elite;
}
const manifest = parseManifest(publicJson(`/research/${DIR}/manifest.json`));
const atlasJson = publicJson(`/research/${DIR}/atlas.json`) as J;
const fanJson = publicJson(`/research/${DIR}/fan-mbh.json`) as J;

function kind(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    if (e instanceof DataError) return `${e.kind}: ${e.message}`;
    throw e;
  }
  return "no error";
}

describe("atlas and error_fan kinds", () => {
  it("accepts an atlas run with one atlas and several error fans", () => {
    expect(manifest.artifacts.map((a) => a.kind)).toEqual(["atlas", ...Array<string>(7).fill("error_fan")]);
  });

  it("rejects a run whose first artifact is not the atlas but that has several", () => {
    const m = clone(publicJson(`/research/${DIR}/manifest.json`)) as J;
    (m.artifacts as J[]).reverse();
    expect(kind(() => parseManifest(m))).toMatch(/expected exactly one artifact/);
  });

  it("parses the published atlas: seven arms in order, 32 reachable cells, 10 x 10 grid", () => {
    const atlas = parseAtlas(atlasJson, manifest);
    expect(atlas.arms.map((a) => a.arm)).toEqual(["D", "MS", "MBH", "DE", "CC", "T", "T2"]);
    expect(atlas.grid.reachable_cells).toHaveLength(32);
    expect(atlas.grid.d1.edges_km).toHaveLength(11);
    expect(atlas.draws).toBe(50);
    expect(atlas.arms[5]?.elites).toHaveLength(32);
    expect(atlas.arms[2]?.elites).toHaveLength(0);
  });

  it.each([
    ["wrong run id", (a: J) => { a.run_id = "other"; }, /run_id/],
    ["wrong frame", (a: J) => { a.frame = "inertial"; }, /frame/],
    ["arms out of order", (a: J) => { (a.arms as J[]).reverse(); }, /arm/],
    ["unreachable elite cell", (a: J) => { elite0(a).cell = [0, 9]; }, /reachable cell/],
    ["survival above one", (a: J) => { elite0(a).survival = 1.2; }, /survival/],
    ["decreasing path time", (a: J) => { (elite0(a).path as J).t_days = [2, 1]; }, /t_days/],
    ["non-finite coordinate", (a: J) => { (elite0(a).path as J).x_km = [null, 1]; }, /x_km/],
    ["unknown schema version", (a: J) => { a.schema_version = 2; }, /schema_version/],
  ])("rejects an atlas with %s", (_name, mutate, message) => {
    const bad = clone(atlasJson);
    mutate(bad);
    expect(kind(() => parseAtlas(bad, manifest))).toMatch(message);
  });

  it("parses a fan and checks it against its arm", () => {
    const atlas = parseAtlas(atlasJson, manifest);
    const fan = parseErrorFan(fanJson, manifest);
    expect(fan.arm).toBe("MBH");
    const [, ms, mbh] = atlas.arms;
    if (ms === undefined || mbh === undefined) throw new Error("arms missing");
    expect(() => { checkFanMatchesArm(fan, mbh, "fan-mbh.json"); }).not.toThrow();
    expect(kind(() => { checkFanMatchesArm(fan, ms, "fan-mbh.json"); })).toMatch(/arm "MBH"/);
  });

  it("rejects a fan whose draw has an unknown outcome", () => {
    const bytes = publicJson(`/research/${DIR}/fan-ms.json`) as J;
    const bad = clone(bytes);
    ((((bad.elites as J[])[0] as J).draws as J[])[0] as J).outcome = "exploded";
    expect(kind(() => parseErrorFan(bad, manifest))).toMatch(/outcome/);
  });

  it("loads the atlas and a fan through the checksum-verifying loaders", async () => {
    const run = await loadAtlasRun(DIR, staticFetcher(), "gravity-m4-demo-atlas-v1");
    expect(run.atlas.arms).toHaveLength(7);
    const fan = await loadErrorFan(DIR, "fan-ms.json", staticFetcher(), "gravity-m4-demo-atlas-v1");
    expect(fan.elites).toHaveLength(1);
    expect(fan.elites[0]?.draws).toHaveLength(50);
  });

  it("refuses a fan whose bytes differ from the manifest checksum", async () => {
    const url = `/research/${DIR}/fan-ms.json`;
    const tampered = new TextDecoder().decode(publicBytes(url)).replace("0", "1");
    expect(await sha256Hex(new TextEncoder().encode(tampered).buffer)).not.toBe(manifest.artifacts.find((a) => a.path === "fan-ms.json")?.sha256);
    await expect(loadErrorFan(DIR, "fan-ms.json", staticFetcher({ [url]: new Response(tampered) }))).rejects.toMatchObject({ kind: "checksum" });
  });
});
