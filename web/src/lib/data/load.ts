import { DataError } from "./errors";
import { isSafeRelativePath, researchUrl } from "./paths";
import { sha256Hex } from "./sha256";
import type { ArtifactKind, Catalog, LoadedGridRun, LoadedRun, LoadedTableRun, Manifest } from "./types";
import { artifactOf, parseCatalog, parseManifest, parseOutcomeGrid, parseSampleTable, parseTrajectory } from "./validate";

export type Fetcher = (url: string) => Promise<Response>;

const defaultFetch: Fetcher = (url) => fetch(url, { credentials: "same-origin" });

async function fetchBytes(url: string, fetcher: Fetcher): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetcher(url);
  } catch (error) {
    throw new DataError("network", url, `The request for ${url} did not complete (${String(error)}).`);
  }
  if (!response.ok) {
    throw new DataError("http", url, `The server answered ${response.status} for ${url}.`);
  }
  return response.arrayBuffer();
}

function parseJson(bytes: ArrayBuffer, url: string): unknown {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new DataError("parse", url, `${url} is not UTF-8 text.`);
  }
  if (text.trim().length === 0) throw new DataError("empty", url, `${url} is empty.`);
  try {
    return JSON.parse(text);
  } catch {
    throw new DataError("parse", url, `${url} is not valid JSON.`);
  }
}

export async function loadCatalog(fetcher: Fetcher = defaultFetch): Promise<Catalog> {
  const url = researchUrl("catalog.json");
  return parseCatalog(parseJson(await fetchBytes(url, fetcher), url), url);
}

interface VerifiedArtifact {
  manifest: Manifest;
  json: unknown;
  manifestUrl: string;
  artifactUrl: string;
}

/**
 * Load a run directory (relative to /research/): validate the manifest against the data contract,
 * check that its single artifact has the expected kind, and verify the artifact bytes against the
 * manifest's SHA-256 before parsing them as JSON.
 */
async function loadVerified(runPath: string, kind: ArtifactKind, fetcher: Fetcher, expectedRunId?: string): Promise<VerifiedArtifact> {
  if (!isSafeRelativePath(runPath)) {
    throw new DataError("unsafe_path", runPath, `"${runPath}" is not a plain relative path inside the research directory.`);
  }
  const manifestUrl = researchUrl(runPath, "manifest.json");
  const manifest = parseManifest(parseJson(await fetchBytes(manifestUrl, fetcher), manifestUrl), manifestUrl);
  if (expectedRunId !== undefined && manifest.run_id !== expectedRunId) {
    throw new DataError(
      "invalid",
      manifestUrl,
      `The catalog lists run "${expectedRunId}" here, but the manifest's run_id is "${manifest.run_id}".`,
    );
  }
  const artifact = artifactOf(manifest, kind);
  if (artifact === undefined) {
    throw new DataError(
      "invalid",
      manifestUrl,
      `Expected an artifact of kind "${kind}", but the manifest lists ${manifest.artifacts.map((a) => `"${a.kind}"`).join(", ")}.`,
    );
  }
  const artifactUrl = researchUrl(runPath, artifact.path);
  const bytes = await fetchBytes(artifactUrl, fetcher);
  const actual = await sha256Hex(bytes);
  if (actual !== artifact.sha256) {
    throw new DataError(
      "checksum",
      artifactUrl,
      `SHA-256 of ${artifact.path} is ${actual}, but the manifest records ${artifact.sha256}.`,
    );
  }
  return { manifest, json: parseJson(bytes, artifactUrl), manifestUrl, artifactUrl };
}

/** A run whose artifact is a trajectory (kind "trajectory"). */
export async function loadRun(runPath: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<LoadedRun> {
  const { manifest, json, manifestUrl, artifactUrl } = await loadVerified(runPath, "trajectory", fetcher, expectedRunId);
  return { manifest, trajectory: parseTrajectory(json, manifest, artifactUrl), manifestUrl, trajectoryUrl: artifactUrl };
}

/** A run whose artifact is a sample table (kind "sample_table"). */
export async function loadTableRun(runPath: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<LoadedTableRun> {
  const { manifest, json, manifestUrl, artifactUrl } = await loadVerified(runPath, "sample_table", fetcher, expectedRunId);
  return { manifest, table: parseSampleTable(json, manifest, artifactUrl), manifestUrl, artifactUrl };
}

/** A run whose artifact is an outcome grid (kind "outcome_grid"). */
export async function loadGridRun(runPath: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<LoadedGridRun> {
  const { manifest, json, manifestUrl, artifactUrl } = await loadVerified(runPath, "outcome_grid", fetcher, expectedRunId);
  return { manifest, grid: parseOutcomeGrid(json, manifest, artifactUrl), manifestUrl, artifactUrl };
}
