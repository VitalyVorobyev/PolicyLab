// @vitest-environment node
import { describe, expect, it } from "vitest";
import { publicBytes, publishedRunPath, RUN_PATH, staticFetcher } from "../../test/fixtures";
import { DataError } from "./errors";
import { loadCatalog, loadGridRun, loadRun, loadTableRun } from "./load";
import { sha256Hex } from "./sha256";

const TRAJECTORY_URL = `/research/${RUN_PATH}/trajectory.json`;
const MANIFEST_URL = `/research/${RUN_PATH}/manifest.json`;

async function rejection(promise: Promise<unknown>): Promise<DataError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(DataError);
    return error as DataError;
  }
  throw new Error("expected the load to fail");
}

describe("loadRun", () => {
  it("loads, verifies and parses the published reference run", async () => {
    const run = await loadRun(RUN_PATH, staticFetcher(), "gravity-l1-lyapunov-reference-v2");
    expect(run.manifest.run_id).toBe("gravity-l1-lyapunov-reference-v2");
    expect(run.trajectory.length).toBe(1001);
    expect(run.trajectoryUrl).toBe(TRAJECTORY_URL);
    expect(run.manifestUrl).toBe(MANIFEST_URL);
  });

  it("computes the same SHA-256 as the manifest records", async () => {
    const bytes = publicBytes(TRAJECTORY_URL);
    const hex = await sha256Hex(bytes.buffer);
    expect(hex).toBe("4f4655c91643eecc25bd6319a5835114a0d6a797ecb8b978b730405daea2fa53");
  });

  it("refuses trajectory bytes that differ from the manifest checksum, even if they are valid JSON", async () => {
    const original = new TextDecoder().decode(publicBytes(TRAJECTORY_URL));
    const tampered = original.replace("0.8150724480121228", "0.8150724480121229");
    expect(tampered).not.toBe(original);
    const error = await rejection(loadRun(RUN_PATH, staticFetcher({ [TRAJECTORY_URL]: new Response(tampered) })));
    expect(error.kind).toBe("checksum");
    expect(error.message).toMatch(/4f4655c9/);
  });

  it("reports a missing file with its HTTP status", async () => {
    const error = await rejection(loadRun(RUN_PATH, staticFetcher({ [MANIFEST_URL]: new Response("", { status: 404 }) })));
    expect(error.kind).toBe("http");
    expect(error.message).toMatch(/404/);
  });

  it("reports an HTML fallback page served instead of JSON as a parse error", async () => {
    const html = new Response("<!doctype html><html></html>", { headers: { "content-type": "text/html" } });
    const error = await rejection(loadRun(RUN_PATH, staticFetcher({ [MANIFEST_URL]: html })));
    expect(error.kind).toBe("parse");
  });

  it("reports an empty file", async () => {
    const error = await rejection(loadRun(RUN_PATH, staticFetcher({ [MANIFEST_URL]: new Response("  ") })));
    expect(error.kind).toBe("empty");
  });

  it("reports network failures", async () => {
    const error = await rejection(loadRun(RUN_PATH, staticFetcher({ [MANIFEST_URL]: new TypeError("Failed to fetch") })));
    expect(error.kind).toBe("network");
  });

  it("refuses a manifest whose run_id differs from the catalog entry", async () => {
    const error = await rejection(loadRun(RUN_PATH, staticFetcher(), "some-other-run"));
    expect(error.kind).toBe("invalid");
    expect(error.message).toMatch(/some-other-run/);
  });

  it("refuses a run whose artifact kind differs from the requested kind", async () => {
    const error = await rejection(loadTableRun(RUN_PATH, staticFetcher()));
    expect(error.kind).toBe("invalid");
    expect(error.message).toMatch(/kind "sample_table".*"trajectory"/);
  });

  it("verifies the outcome grid's checksum before decoding it", async () => {
    const path = publishedRunPath((r) => r.kind === "outcome_grid");
    const url = `/research/${path}/outcome_grid.json`;
    const tampered = new TextDecoder().decode(publicBytes(url)).replace('"codes":"', '"codes":"A');
    const error = await rejection(loadGridRun(path, staticFetcher({ [url]: new Response(tampered) })));
    expect(error.kind).toBe("checksum");
  });

  it("refuses unsafe run paths before fetching anything", async () => {
    const requested: string[] = [];
    const fetcher = (url: string): Promise<Response> => {
      requested.push(url);
      return Promise.resolve(new Response("{}"));
    };
    const error = await rejection(loadRun("../outside", fetcher));
    expect(error.kind).toBe("unsafe_path");
    expect(requested).toEqual([]);
  });
});

describe("loadCatalog", () => {
  it("loads the published catalog, and every listed run loads and verifies", async () => {
    const catalog = await loadCatalog(staticFetcher());
    const runs = catalog.studies.flatMap((s) => s.runs);
    expect(runs.length).toBeGreaterThan(0);
    for (const run of runs) {
      const load = { trajectory: loadRun, sample_table: loadTableRun, outcome_grid: loadGridRun }[run.kind];
      const loaded = await load(run.path, staticFetcher(), run.run_id);
      expect(loaded.manifest.study_id).toBe(catalog.studies.find((s) => s.runs.includes(run))?.study_id);
    }
    expect(new Set(runs.map((r) => r.kind))).toEqual(new Set(["trajectory", "sample_table", "outcome_grid"]));
  });
});
