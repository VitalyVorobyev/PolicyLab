// @vitest-environment node
import { describe, expect, it } from "vitest";
import { clone, publicJson, publishedRunPath, staticFetcher } from "../../test/fixtures";
import { parseCitizensDemo, parseCitizensManifest, type CitizensManifest } from "./citizens";
import { DataError } from "./errors";
import { loadCitizensRun } from "./load";

const PATH = publishedRunPath((r) => r.kind === "citizens_demo");
const rawManifest = publicJson(`/research/${PATH}/manifest.json`) as Record<string, unknown>;
const rawDemo = publicJson(`/research/${PATH}/citizens_demo.json`) as Record<string, unknown> & { traces: Record<string, unknown>[] };
const manifest: CitizensManifest = parseCitizensManifest(rawManifest);

function kindOf(fn: () => unknown): string {
  try {
    fn();
  } catch (error) {
    return error instanceof DataError ? error.kind : "other";
  }
  return "none";
}

describe("Model Citizens run", () => {
  it("loads, verifies and parses the published run", async () => {
    const run = await loadCitizensRun(PATH, staticFetcher());
    expect(run.manifest.study_id).toBe("model-citizens");
    expect(run.demo.game).toMatchObject({ players: 4, rounds: 40, max_pool: 200, growth_factor: 1.4 });
    // 8 governors x 4 citizen models x 3 games, plus LLM games.
    expect(run.demo.traces.filter((t) => ["B1", "B2", "B3", "B23"].includes(t.citizens))).toHaveLength(96);
    expect(run.demo.crossplay.columns).toEqual(["B1", "B2", "B3", "B23"]);
  });

  it("follows the game rule in every published trace (independent check on the exported numbers)", () => {
    const demo = parseCitizensDemo(rawDemo, manifest);
    for (const g of [...demo.traces, ...demo.human_games.filter((h) => h.split === "training")]) {
      for (let t = 0; t < g.rounds; t++) {
        const e = g.offers[t] ?? [];
        const c = g.returns[t] ?? [];
        const sum = (a: number[]): number => a.reduce((s, v) => s + v, 0);
        const next = Math.max(0.01, Math.min(200, (g.pool[t] ?? 0) - sum(e) + 1.4 * sum(c)));
        expect(Math.abs((g.pool[t + 1] ?? 0) - next)).toBeLessThanOrEqual(0.06);
        expect(sum(e)).toBeLessThanOrEqual((g.pool[t] ?? 0) + 0.02);
      }
    }
  });

  it("rejects a trace whose kept coins are not offer minus returned", () => {
    const bad = clone(rawDemo);
    const trace = bad.traces[0] as { kept: number[][] };
    const row = trace.kept[3];
    if (row !== undefined) row[1] = (row[1] ?? 0) + 5;
    expect(kindOf(() => parseCitizensDemo(bad, manifest))).toBe("invalid");
  });

  it("rejects a run id mismatch, a short pool series and a manifest with another artifact kind", () => {
    expect(kindOf(() => parseCitizensDemo({ ...clone(rawDemo), run_id: "other" }, manifest))).toBe("invalid");
    const short = clone(rawDemo);
    (short.traces[0] as { pool: number[] }).pool.pop();
    expect(kindOf(() => parseCitizensDemo(short, manifest))).toBe("invalid");
    const m = clone(rawManifest) as { artifacts: { kind: string }[] };
    const first = m.artifacts[0];
    if (first !== undefined) first.kind = "trajectory";
    expect(kindOf(() => parseCitizensManifest(m))).toBe("invalid");
  });

  it("refuses an artifact whose bytes differ from the manifest checksum", async () => {
    const url = `/research/${PATH}/citizens_demo.json`;
    const tampered = JSON.stringify({ ...rawDemo, data_attribution: "changed" });
    await expect(loadCitizensRun(PATH, staticFetcher({ [url]: new Response(tampered) }))).rejects.toMatchObject({ kind: "checksum" });
  });
});
