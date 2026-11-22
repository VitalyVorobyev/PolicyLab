import { parseCitizensDemo, parseCitizensManifest, type CitizensManifest, type LoadedCitizensRun } from "./citizens";
import { DataError } from "./errors";
import { parseLibraryManifest, parseLibraryResults, parseLibrarySamples, type LibraryManifest, type LoadedLibraryRun } from "./library";
import { isSafeRelativePath, researchUrl } from "./paths";
import { sha256Hex } from "./sha256";
import type { Artifact, ArtifactKind, Catalog, ErrorFan, LoadedAtlasRun, LoadedGridRun, LoadedRun, LoadedTableRun, Manifest } from "./types";
import { artifactOf, parseAtlas, parseCatalog, parseErrorFan, parseManifest, parseOutcomeGrid, parseSampleTable, parseTrajectory } from "./validate";

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
async function loadVerified(
  runPath: string,
  kind: ArtifactKind,
  fetcher: Fetcher,
  expectedRunId?: string,
  fileName?: string,
): Promise<VerifiedArtifact> {
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
  const artifact = artifactOf(manifest, kind, fileName);
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

/** A run whose artifacts are an atlas (kind "atlas") and its error fans; only the atlas is loaded here. */
export async function loadAtlasRun(runPath: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<LoadedAtlasRun> {
  const { manifest, json, manifestUrl, artifactUrl } = await loadVerified(runPath, "atlas", fetcher, expectedRunId);
  return { manifest, atlas: parseAtlas(json, manifest, artifactUrl), manifestUrl, artifactUrl };
}

async function fetchCheckedArtifact(
  runPath: string,
  artifact: Artifact,
  fetcher: Fetcher,
): Promise<{ artifactUrl: string; json: unknown }> {
  const artifactUrl = researchUrl(runPath, artifact.path);
  const bytes = await fetchBytes(artifactUrl, fetcher);
  const actual = await sha256Hex(bytes);
  if (actual !== artifact.sha256) {
    throw new DataError("checksum", artifactUrl, `SHA-256 of ${artifact.path} is ${actual}, but the manifest records ${artifact.sha256}.`);
  }
  return { artifactUrl, json: parseJson(bytes, artifactUrl) };
}

/**
 * A Model Citizens web demo run: its own manifest layout (no CR3BP model or solver), one
 * `citizens_demo` artifact verified against the manifest's SHA-256 before it is parsed.
 */
export async function loadCitizensRun(runPath: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<LoadedCitizensRun> {
  if (!isSafeRelativePath(runPath)) {
    throw new DataError("unsafe_path", runPath, `"${runPath}" is not a plain relative path inside the research directory.`);
  }
  const manifestUrl = researchUrl(runPath, "manifest.json");
  const manifest = parseCitizensManifest(parseJson(await fetchBytes(manifestUrl, fetcher), manifestUrl), manifestUrl);
  if (expectedRunId !== undefined && manifest.run_id !== expectedRunId) {
    throw new DataError("invalid", manifestUrl, `The catalog lists run "${expectedRunId}" here, but the manifest's run_id is "${manifest.run_id}".`);
  }
  const artifact = manifest.artifacts[0];
  if (artifact === undefined) throw new DataError("invalid", manifestUrl, "The manifest lists no artifact.");
  const { artifactUrl, json } = await fetchCheckedArtifact(runPath, artifact, fetcher);
  return { manifest, demo: parseCitizensDemo(json, manifest, artifactUrl), manifestUrl, artifactUrl };
}

/**
 * A The Last Library confirmatory run: verifies SHA-256 digests of both `results.json` and `samples.json`
 * before parsing either artifact.
 */
export async function loadLibraryRun(runPath: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<LoadedLibraryRun> {
  if (!isSafeRelativePath(runPath)) {
    throw new DataError("unsafe_path", runPath, `"${runPath}" is not a plain relative path inside the research directory.`);
  }
  const manifestUrl = researchUrl(runPath, "manifest.json");
  const manifest = parseLibraryManifest(parseJson(await fetchBytes(manifestUrl, fetcher), manifestUrl), manifestUrl);
  if (expectedRunId !== undefined && manifest.run_id !== expectedRunId) {
    throw new DataError("invalid", manifestUrl, `The catalog lists run "${expectedRunId}" here, but the manifest's run_id is "${manifest.run_id}".`);
  }
  const resultsArt = manifest.artifacts.find((a) => a.kind === "confirmatory_results");
  const samplesArt = manifest.artifacts.find((a) => a.kind === "sample_lineages");
  if (resultsArt === undefined || samplesArt === undefined) {
    throw new DataError("invalid", manifestUrl, 'Expected both "confirmatory_results" and "sample_lineages" artifacts.');
  }
  const [resLoaded, sampLoaded] = await Promise.all([
    fetchCheckedArtifact(runPath, resultsArt, fetcher),
    fetchCheckedArtifact(runPath, samplesArt, fetcher),
  ]);
  return {
    manifest,
    results: parseLibraryResults(resLoaded.json, manifest, resLoaded.artifactUrl),
    samples: parseLibrarySamples(sampLoaded.json, manifest, sampLoaded.artifactUrl),
    manifestUrl,
    resultsUrl: resLoaded.artifactUrl,
    samplesUrl: sampLoaded.artifactUrl,
  };
}

/** One arm's error fan from an atlas run, verified against the manifest's checksum for that file. */
export async function loadErrorFan(runPath: string, fileName: string, fetcher: Fetcher = defaultFetch, expectedRunId?: string): Promise<ErrorFan> {
  const { manifest, json, artifactUrl } = await loadVerified(runPath, "error_fan", fetcher, expectedRunId, fileName);
  return parseErrorFan(json, manifest, artifactUrl);
}

export type AnyManifest = Manifest | CitizensManifest | LibraryManifest;

/**
 * Load and validate a run's `manifest.json` without downloading large data artifacts unless inspected separately.
 */
export async function loadAnyManifest(
  runPath: string,
  kind: ArtifactKind,
  fetcher: Fetcher = defaultFetch,
  expectedRunId?: string,
): Promise<{ manifest: AnyManifest; manifestUrl: string }> {
  if (!isSafeRelativePath(runPath)) {
    throw new DataError("unsafe_path", runPath, `"${runPath}" is not a plain relative path inside the research directory.`);
  }
  const manifestUrl = researchUrl(runPath, "manifest.json");
  const json = parseJson(await fetchBytes(manifestUrl, fetcher), manifestUrl);
  const manifest: AnyManifest =
    kind === "citizens_demo"
      ? parseCitizensManifest(json, manifestUrl)
      : kind === "confirmatory_results" || kind === "sample_lineages"
        ? parseLibraryManifest(json, manifestUrl)
        : parseManifest(json, manifestUrl);
  if (expectedRunId !== undefined && manifest.run_id !== expectedRunId) {
    throw new DataError(
      "invalid",
      manifestUrl,
      `The catalog lists run "${expectedRunId}" here, but the manifest's run_id is "${manifest.run_id}".`,
    );
  }
  return { manifest, manifestUrl };
}
