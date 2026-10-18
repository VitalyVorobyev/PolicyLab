import { useMemo, type ReactNode } from "react";
import { ErrorState, LoadingState } from "../../components/DataState";
import { AlertIcon, DownloadIcon } from "../../components/icons";
import { StatusBadge } from "../../components/StatusBadge";
import { fixed, shortSha, utcDateTime } from "../../lib/format";
import type { CatalogRun, LoadedGridRun, LoadedTableRun, Manifest } from "../../lib/data/types";
import { useGridRun, useTableRun, type Resource } from "../../lib/data/useResource";
import { commitUrl } from "../../lib/links";
import { FeasibilityMap, type MapCandidate } from "./FeasibilityMap";
import { countOutcomes, feasibleCells } from "./grid";
import { bandWidthMs, ProbeStrip, probeRows } from "./ProbeStrip";

function Provenance({ manifest, manifestUrl, artifactUrl, artifactNote }: { manifest: Manifest; manifestUrl: string; artifactUrl: string; artifactNote: string }) {
  const artifact = manifest.artifacts[0];
  const derived = manifest.derived_from;
  return (
    <div className="min-w-0 border-t border-rule pt-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <p className="font-medium break-all">{manifest.run_id}</p>
        <StatusBadge status={manifest.status} />
      </div>
      {manifest.source.git_dirty && (
        <p className="mt-2 flex items-start gap-2 rounded-sm border border-caution/40 bg-caution/5 px-3 py-2 text-sm text-caution">
          <AlertIcon className="mt-0.5 size-4 shrink-0" />
          <span>
            Development build: this run was generated from a working tree with uncommitted changes, so commit{" "}
            {shortSha(manifest.source.git_commit)} alone does not reproduce it. It must be regenerated from a clean commit before release.
          </span>
        </p>
      )}
      <dl className="mt-3 grid grid-cols-1 gap-x-5 gap-y-1 text-[0.92rem] sm:grid-cols-[9rem_minmax(0,1fr)]">
        <dt className="text-graphite">Experiment</dt>
        <dd className="break-words">{manifest.experiment_id}</dd>
        <dt className="text-graphite">Created</dt>
        <dd>{utcDateTime(manifest.created_at)}</dd>
        <dt className="text-graphite">Code</dt>
        <dd>
          <a className="prose-link numeric" href={commitUrl(manifest.source.git_commit)} rel="noreferrer" title={manifest.source.git_commit}>
            {shortSha(manifest.source.git_commit)}
          </a>
          , {manifest.source.git_dirty ? "uncommitted changes present" : "clean working tree"}
        </dd>
        {derived !== null && (
          <>
            <dt className="text-graphite">Derived from</dt>
            <dd className="min-w-0">
              <span className="break-all">{derived.run_id}</span>, {derived.artifact_path} (SHA-256{" "}
              <span className="numeric break-all" title={derived.artifact_sha256}>
                {derived.artifact_sha256.slice(0, 12)}…
              </span>
              , verified), code{" "}
              <a className="prose-link numeric" href={commitUrl(derived.git_commit)} rel="noreferrer" title={derived.git_commit}>
                {shortSha(derived.git_commit)}
              </a>
              {derived.git_dirty ? " (uncommitted changes)" : ""}.
              <span className="mt-1 block text-sm text-graphite">{derived.derivation}</span>
            </dd>
          </>
        )}
      </dl>
      <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-[0.92rem]">
        <li>
          <a href={manifestUrl} download className="prose-link inline-flex items-center gap-1.5">
            <DownloadIcon /> manifest.json
          </a>
        </li>
        <li>
          <a href={artifactUrl} download className="prose-link inline-flex items-center gap-1.5" title={artifact?.sha256}>
            <DownloadIcon /> {artifact?.path}
          </a>{" "}
          <span className="text-sm text-graphite">{artifactNote}</span>
        </li>
      </ul>
      <details className="mt-3 text-[0.92rem]">
        <summary className="cursor-pointer text-graphite marker:text-quiet hover:text-ink">Limitations recorded with this run</summary>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 marker:text-quiet">
          {manifest.limitations.map((l) => (
            <li key={l} className="[overflow-wrap:anywhere]">{l}</li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function ProbeFindings({ probe, grid }: { probe: LoadedTableRun; grid: LoadedGridRun | null }) {
  const rows = useMemo(() => probeRows(probe.table), [probe.table]);
  const step = probe.manifest.metrics.grid_delta_v_step_km_s?.value;
  const plus = useMemo(() => {
    if (grid === null) return null;
    const cells = [...feasibleCells(grid.grid).values()].filter((c) => (c.values.lunar_passage_sense ?? 0) > 0);
    if (cells.length === 0) return null;
    const thetas = cells.map((c) => grid.grid.phaseAxis.min + c.i * grid.grid.phaseAxis.step);
    return { count: cells.length, lo: Math.min(...thetas), hi: Math.max(...thetas) };
  }, [grid]);
  const gridStepMs = grid === null ? null : grid.grid.deltaVAxis.step * 1000;
  return (
    <div className="grid gap-8 lg:grid-cols-12 lg:gap-10">
      <figure className="on-canvas min-w-0 self-start rounded-sm bg-canvas p-3 text-canvas-strong sm:p-5 lg:col-span-8">
        <figcaption className="mb-3 text-sm text-canvas-text">
          {rows.map((r) => r.outcomes.length.toLocaleString("en-US")).join(" + ")} tangential departures,{" "}
          {step !== undefined ? `${fixed(step * 1e6, 0)} mm/s apart` : "evenly spaced"}, at two fixed phases. Same outcome colors as the map; a
          pixel column holding any feasible sample is drawn feasible. Brackets mark each contiguous feasible band.
        </figcaption>
        <ProbeStrip run={probe} />
      </figure>
      <div className="self-start lg:col-span-4">
        <ul className="space-y-3">
          {rows.map((r) =>
            r.feasibleRuns.map((f) => {
              const widths = step === undefined ? null : bandWidthMs(f.count, step);
              return (
                <li key={`${r.phaseDeg}-${f.first}`} className="border-t border-rule pt-2">
                  <p className="text-sm text-graphite">θ = {fixed(r.phaseDeg, 0)}°</p>
                  <p className="numeric text-[1.15rem] leading-snug">
                    {widths === null ? `${f.count} feasible samples` : `${fixed(widths[0], 2)}–${fixed(widths[1], 2)} m/s wide`}
                  </p>
                  <p className="numeric text-sm text-graphite">
                    Δv {fixed(f.first, 5)}–{fixed(f.last, 5)} km/s, {f.count} feasible samples
                    {r.feasibleRuns.length === 1 ? ", one contiguous band" : ""}
                  </p>
                </li>
              );
            }),
          )}
        </ul>
        <p className="mt-5 text-[0.95rem]">
          <span className="font-medium">At least two families at this resolution.</span> Feasible departures with lunar passage sense −1
          fill bands several grid cells wide.
          {plus !== null && gridStepMs !== null && (
            <>
              {" "}
              The {plus.count} grid cells with sense +1 (θ {fixed(plus.lo, 1)}–{fixed(plus.hi, 1)}°) are hits on windows narrower than the{" "}
              {fixed(gridStepMs, 0)} m/s grid step, so the grid cannot measure that family’s extent, and it may miss thinner regions altogether.
            </>
          )}{" "}
          The comparison’s coverage ground truth will need adaptive refinement at band edges.
        </p>
      </div>
    </div>
  );
}

interface Props {
  gridRun: CatalogRun;
  probeRun: CatalogRun | undefined;
  candidates: MapCandidate[];
  selectedRunId: string;
  onSelect: (runId: string) => void;
}

/** Loads the optional probe run; hooks cannot be called conditionally, so this is a component. */
function WithProbe({ run, children }: { run: CatalogRun; children: (probe: Resource<LoadedTableRun>) => ReactNode }) {
  return children(useTableRun(run.run_id, run.path));
}

export function FeasibilitySection(props: Props) {
  return props.probeRun === undefined ? (
    <Section {...props} probe={null} />
  ) : (
    <WithProbe run={props.probeRun}>{(probe) => <Section {...props} probe={probe} />}</WithProbe>
  );
}

function Section({ gridRun, probeRun, candidates, selectedRunId, onSelect, probe }: Props & { probe: Resource<LoadedTableRun> | null }) {
  const grid = useGridRun(gridRun.run_id, gridRun.path);
  const loadedGrid = grid.status === "ready" ? grid.value : null;
  const summary = useMemo(() => {
    if (loadedGrid === null) return null;
    const counts = countOutcomes(loadedGrid.grid);
    return { feasible: counts.get("feasible") ?? 0, total: loadedGrid.grid.codes.length };
  }, [loadedGrid]);
  const metric = (key: string): number | undefined => loadedGrid?.manifest.metrics[key]?.value;
  const parking = metric("parking_altitude_km");
  const tMax = metric("t_max_days");

  return (
    <section aria-labelledby="map-title" className="mt-20 border-t border-rule pt-10">
      <div className="grid gap-x-12 gap-y-4 lg:grid-cols-12">
        <div className="lg:col-span-4">
          <h2 id="map-title" className="text-3xl sm:text-4xl">
            Where free returns exist
          </h2>
        </div>
        <div className="max-w-[64ch] lg:col-span-8">
          <p className="font-serif text-[1.15rem] leading-relaxed">
            Every cell below is one departure: a single tangential impulse{parking !== undefined ? ` from the ${fixed(parking, 0)} km parking orbit` : ""}{" "}
            at phase θ with magnitude Δv, propagated once{tMax !== undefined ? ` for up to ${fixed(tMax, 0)} days` : ""} and classified.
            {summary !== null &&
              ` ${summary.feasible.toLocaleString("en-US")} of ${summary.total.toLocaleString("en-US")} departures (${fixed((100 * summary.feasible) / summary.total, 2)} %) are free returns.`}{" "}
            They lie in a thin band next to the Moon-collision cells.
          </p>
          <p className="mt-3 text-graphite">
            Each cell comes from one propagation and is not validated on its own; only the two marked candidates carry validation checks.
          </p>
        </div>
      </div>

      <div className="mt-8">
        {grid.status === "loading" && <LoadingState label="Loading and verifying the outcome grid…" />}
        {grid.status === "error" && <ErrorState error={grid.error} onRetry={grid.retry} />}
        {grid.status === "ready" && <FeasibilityMap run={grid.value} candidates={candidates} selectedRunId={selectedRunId} onSelect={onSelect} />}
      </div>

      {probeRun !== undefined && probe !== null && (
        <section aria-labelledby="probe-title-heading" className="mt-14">
          <h3 id="probe-title-heading" className="text-2xl">
            How thin is the band?
          </h3>
          <p className="mt-2 mb-6 max-w-[64ch] text-graphite">
            A finer probe along Δv at one phase inside each feasible region measures band widths the grid cannot resolve.
          </p>
          {probe.status === "loading" && <LoadingState label="Loading and verifying the Δv probe…" />}
          {probe.status === "error" && <ErrorState error={probe.error} onRetry={probe.retry} />}
          {probe.status === "ready" && <ProbeFindings probe={probe.value} grid={loadedGrid} />}
        </section>
      )}

      <section aria-labelledby="map-provenance-title" className="mt-14">
        <h3 id="map-provenance-title" className="text-2xl">
          Map data and provenance
        </h3>
        <div className="mt-4 grid gap-8 lg:grid-cols-2 lg:gap-12">
          {grid.status === "ready" && (
            <Provenance
              manifest={grid.value.manifest}
              manifestUrl={grid.value.manifestUrl}
              artifactUrl={grid.value.artifactUrl}
              artifactNote="every cell's outcome, lossless"
            />
          )}
          {probe?.status === "ready" && (
            <Provenance
              manifest={probe.value.manifest}
              manifestUrl={probe.value.manifestUrl}
              artifactUrl={probe.value.artifactUrl}
              artifactNote={`${probe.value.table.rows.length.toLocaleString("en-US")} rows, ${probe.value.table.columns.length} columns`}
            />
          )}
        </div>
      </section>
    </section>
  );
}
