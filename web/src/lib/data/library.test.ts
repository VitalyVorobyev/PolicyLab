// @vitest-environment node
import { describe, expect, it } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { DataError } from "./errors";
import { loadLibraryRun, type Fetcher } from "./load";

function publicFetcher(): Fetcher {
  return (url) => Promise.resolve(new Response(publicBytes(url)));
}

describe("published The Last Library confirmatory run", () => {
  it("verifies SHA-256 checksums for results.json and samples.json and parses all families", async () => {
    const loaded = await loadLibraryRun("last-library/main-v1", publicFetcher(), "library-main-v1");
    expect(loaded.manifest.run_id).toBe("library-main-v1");
    expect(loaded.manifest.status).toBe("validated");
    expect(loaded.manifest.source.git_dirty).toBe(false);

    expect(Object.keys(loaded.results.hypotheses)).toEqual(["H1", "H2a", "H2b", "H3a", "H3b", "H4"]);
    expect(loaded.results.hypotheses.H1?.n_passed).toBe(1);
    expect(loaded.results.hypotheses.H2b?.all_passed).toBe(true);
    expect(loaded.results.hypotheses.H3a?.n_passed).toBe(0);
    expect(loaded.results.hypotheses.H3b?.n_passed).toBe(3);

    expect(loaded.samples.stationary).toHaveLength(3);
    expect(loaded.samples.rule_change).toHaveLength(3);
    expect(loaded.samples.stationary[0]?.layouts_by_key_gen[0]?.heldout_preview.terrains).toHaveLength(9);
  });

  it("rejects results.json when SHA-256 digest does not match", async () => {
    const fetcher: Fetcher = (url) => {
      const bytes = publicBytes(url);
      if (url.endsWith("results.json")) {
        const tampered = new Uint8Array(bytes);
        tampered[10] = (tampered[10] ?? 0) ^ 0xff;
        return Promise.resolve(new Response(tampered));
      }
      return Promise.resolve(new Response(bytes));
    };
    await expect(loadLibraryRun("last-library/main-v1", fetcher, "library-main-v1")).rejects.toMatchObject({
      name: "DataError",
      kind: "checksum" satisfies DataError["kind"],
    });
  });
});
