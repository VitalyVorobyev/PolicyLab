#!/usr/bin/env node
// Copy one exported research run into web/public/research/ so it can be served statically.
//
// Usage: node web/scripts/publish-run.mjs <source-run-dir> <destination>
//   <source-run-dir>  a directory containing manifest.json (e.g. runs/gravity-l1-lyapunov-reference-v2)
//   <destination>     a path relative to web/public/research/ (e.g. gravity-atlas/l1-lyapunov-reference-v2)
//
// The script verifies every artifact's SHA-256 against the manifest before and after copying,
// refuses runs whose provenance records uncommitted changes, and never overwrites an existing
// published directory. Files are copied byte for byte. It does not edit catalog.json; it prints
// the entry to add.

import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SAFE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const RESEARCH_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "public", "research");

/** @param {string} message */
function fail(message) {
  console.error(`publish-run: ${message}`);
  process.exit(1);
}

/** @param {string} path */
function isSafeRelativePath(path) {
  const segments = path.split("/");
  return segments.length > 0 && segments.every((s) => SAFE_SEGMENT.test(s) && s !== "." && s !== "..");
}

/** @param {string} file */
function sha256(file) {
  return createHash("sha256").update(readFileSync(file)).digest("hex");
}

const [source, destination] = process.argv.slice(2);
if (source === undefined || destination === undefined) {
  fail("usage: publish-run.mjs <source-run-dir> <destination relative to web/public/research>");
}
if (!isSafeRelativePath(destination)) {
  fail(`destination must be a plain relative path (letters, digits, . _ -), got "${destination}"`);
}

const sourceDir = resolve(source);
const manifestPath = join(sourceDir, "manifest.json");
if (!existsSync(manifestPath)) fail(`no manifest.json in ${sourceDir}`);

/** @type {any} */
let manifest;
try {
  manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
} catch (error) {
  fail(`manifest.json is not valid JSON: ${String(error)}`);
}
if (manifest.schema_version !== 1) fail(`unsupported schema_version ${String(manifest.schema_version)}`);
if (typeof manifest.run_id !== "string" || manifest.run_id === "") fail("manifest has no run_id");
if (manifest.source?.git_dirty !== false) {
  fail("manifest provenance is not from a clean commit (source.git_dirty must be false); re-export after committing");
}
if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length === 0) fail("manifest lists no artifacts");

/** @type {{path: string, sha256: string}[]} */
const artifacts = manifest.artifacts.map((/** @type {any} */ a) => {
  if (typeof a?.path !== "string" || !isSafeRelativePath(a.path)) fail(`unsafe artifact path ${JSON.stringify(a?.path)}`);
  if (typeof a.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(a.sha256)) fail(`artifact ${a.path} has no valid sha256`);
  return { path: a.path, sha256: a.sha256 };
});

for (const artifact of artifacts) {
  const file = join(sourceDir, artifact.path);
  if (!existsSync(file)) fail(`missing artifact ${artifact.path}`);
  const actual = sha256(file);
  if (actual !== artifact.sha256) fail(`checksum mismatch for ${artifact.path}: manifest ${artifact.sha256}, file ${actual}`);
}

const destDir = join(RESEARCH_ROOT, ...destination.split("/"));
if (existsSync(destDir)) fail(`${destDir} already exists; published runs are never overwritten`);

const staging = `${destDir}.staging-${process.pid}`;
mkdirSync(staging, { recursive: true });
try {
  for (const name of ["manifest.json", ...artifacts.map((a) => a.path)]) {
    const target = join(staging, name);
    mkdirSync(dirname(target), { recursive: true });
    cpSync(join(sourceDir, name), target, { errorOnExist: true, force: false });
  }
  if (sha256(join(staging, "manifest.json")) !== sha256(manifestPath)) throw new Error("manifest copy differs");
  for (const artifact of artifacts) {
    if (sha256(join(staging, artifact.path)) !== artifact.sha256) throw new Error(`copy of ${artifact.path} differs`);
  }
  renameSync(staging, destDir);
} catch (error) {
  rmSync(staging, { recursive: true, force: true });
  fail(String(error));
}

console.log(`Published ${manifest.run_id} (${manifest.status}) to ${destDir}`);
console.log("Add this run to web/public/research/catalog.json under its study:");
console.log(JSON.stringify({ run_id: manifest.run_id, path: destination, label: "<short label>" }, null, 2));
