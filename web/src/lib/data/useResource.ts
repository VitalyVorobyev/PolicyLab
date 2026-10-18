import { useCallback, useEffect, useState } from "react";
import { loadCatalog, loadGridRun, loadRun, loadTableRun } from "./load";
import type { Catalog, LoadedGridRun, LoadedRun, LoadedTableRun } from "./types";

export type Resource<T> =
  | { status: "loading" }
  | { status: "ready"; value: T }
  | { status: "error"; error: unknown; retry: () => void };

const cache = new Map<string, Promise<unknown>>();

function cached<T>(key: string, load: () => Promise<T>): Promise<T> {
  let promise = cache.get(key) as Promise<T> | undefined;
  if (promise === undefined) {
    promise = load();
    cache.set(key, promise);
    // Failed loads are not cached, so "Try again" re-requests the files.
    promise.catch(() => cache.delete(key));
  }
  return promise;
}

type Settled<T> = { key: string; attempt: number } & ({ ok: true; value: T } | { ok: false; error: unknown });

/**
 * Load a keyed resource once per page view (shared between components) and expose
 * loading / ready / error states. `load` must be a stable, module-level function.
 */
function useResource<T>(key: string, load: (key: string) => Promise<T>): Resource<T> {
  const [attempt, setAttempt] = useState(0);
  const [settled, setSettled] = useState<Settled<T> | null>(null);

  useEffect(() => {
    let active = true;
    cached(key, () => load(key)).then(
      (value) => {
        if (active) setSettled({ key, attempt, ok: true, value });
      },
      (error: unknown) => {
        if (active) setSettled({ key, attempt, ok: false, error });
      },
    );
    return () => {
      active = false;
    };
  }, [key, attempt, load]);

  const retry = useCallback(() => {
    cache.delete(key);
    setAttempt((a) => a + 1);
  }, [key]);

  if (settled?.key !== key || settled.attempt !== attempt) return { status: "loading" };
  return settled.ok ? { status: "ready", value: settled.value } : { status: "error", error: settled.error, retry };
}

const catalogLoader = (): Promise<Catalog> => loadCatalog();

export function useCatalog(): Resource<Catalog> {
  return useResource("catalog", catalogLoader);
}

/** Key format: "<kind>:<run_id>@<path>" so the catalog's run_id is checked against the manifest. */
function keyed<T>(load: (path: string, fetcher: undefined, runId: string) => Promise<T>): (key: string) => Promise<T> {
  return (key) => {
    const colon = key.indexOf(":");
    const at = key.indexOf("@");
    return load(key.slice(at + 1), undefined, key.slice(colon + 1, at));
  };
}

const runLoader = keyed(loadRun);
const tableLoader = keyed(loadTableRun);
const gridLoader = keyed(loadGridRun);

export function useRun(runId: string, path: string): Resource<LoadedRun> {
  return useResource(`trajectory:${runId}@${path}`, runLoader);
}

export function useTableRun(runId: string, path: string): Resource<LoadedTableRun> {
  return useResource(`sample_table:${runId}@${path}`, tableLoader);
}

export function useGridRun(runId: string, path: string): Resource<LoadedGridRun> {
  return useResource(`outcome_grid:${runId}@${path}`, gridLoader);
}

/** Key format: "trajectories:" + "<run_id>@<path>" entries joined by "|". */
const runListLoader = (key: string): Promise<LoadedRun[]> => {
  const entries = key.slice("trajectories:".length).split("|").filter((e) => e !== "");
  return Promise.all(entries.map((entry) => cached(`trajectory:${entry}`, () => runLoader(`trajectory:${entry}`))));
};

/** Several trajectory runs at once (shared with `useRun`'s cache); ready when all are loaded. */
export function useRunList(runs: readonly { run_id: string; path: string }[]): Resource<LoadedRun[]> {
  return useResource(`trajectories:${runs.map((r) => `${r.run_id}@${r.path}`).join("|")}`, runListLoader);
}
