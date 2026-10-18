import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Fetcher } from "../lib/data/load";

/** Directory served as the site root (web/public). */
export const PUBLIC_DIR = join(import.meta.dirname, "..", "..", "public");
export const RUN_PATH = "gravity-atlas/l1-lyapunov-reference-v2";

export function publicBytes(url: string): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(readFileSync(join(PUBLIC_DIR, ...url.split("/").filter(Boolean))));
}

export function publicJson(url: string): unknown {
  return JSON.parse(new TextDecoder().decode(publicBytes(url)));
}

/** Deep copy so each test can corrupt its own instance. */
export function clone<T>(value: T): T {
  return structuredClone(value);
}

/** A fetcher that serves files from web/public, with optional per-URL overrides. */
export function staticFetcher(overrides: Record<string, Response | Error> = {}): Fetcher {
  return (url) => {
    const override = overrides[url];
    if (override instanceof Error) return Promise.reject(override);
    if (override !== undefined) return Promise.resolve(override);
    try {
      return Promise.resolve(new Response(publicBytes(url)));
    } catch {
      return Promise.resolve(new Response("Not found", { status: 404 }));
    }
  };
}
