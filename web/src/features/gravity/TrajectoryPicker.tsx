import { StatusBadge } from "../../components/StatusBadge";
import { fixed } from "../../lib/format";
import type { CatalogRun, Manifest } from "../../lib/data/types";
import { isFreeReturn, trajectoryStyle } from "./trajectoryStyle";

/** One line about what a trajectory is, from its manifest only. */
export function trajectorySummary(manifest: Manifest | undefined): string {
  if (manifest === undefined) return "Loading and verifying…";
  const m = manifest.metrics;
  if (isFreeReturn(manifest)) {
    const parts = [
      m.perilune_altitude_km !== undefined ? `perilune ${fixed(m.perilune_altitude_km.value, 0)} km` : null,
      m.return_shell_crossing_time_days !== undefined ? `entry interface on day ${fixed(m.return_shell_crossing_time_days.value, 2)}` : null,
    ].filter((p): p is string => p !== null);
    return parts.length > 0 ? `Free return: ${parts.join(", ")}` : "Free return";
  }
  return "Engine check: a periodic orbit matched against the JPL catalog";
}

interface Props {
  runs: CatalogRun[];
  manifests: Map<string, Manifest>;
  selected: string;
  onSelect: (runId: string) => void;
}

/** Trajectory selection as a radio group of cards: line pattern, name, one-line summary and status. */
export function TrajectoryPicker({ runs, manifests, selected, onSelect }: Props) {
  return (
    <fieldset className="mb-6">
      <legend className="mb-3 font-sans text-sm font-medium text-graphite">Trajectory</legend>
      <div className="grid gap-2 sm:gap-3 md:grid-cols-3">
        {runs.map((run, index) => {
          const manifest = manifests.get(run.run_id);
          const style = trajectoryStyle(index);
          const checked = run.run_id === selected;
          return (
            <label
              key={run.run_id}
              className={`relative block cursor-pointer rounded-sm border px-3.5 py-3 transition-colors max-sm:py-2 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-ink ${
                checked ? "border-ink bg-sheet shadow-[inset_0_-3px_0_var(--color-accent)]" : "border-rule-strong hover:border-graphite hover:bg-sheet/60"
              }`}
            >
              <input
                type="radio"
                name="trajectory"
                value={run.run_id}
                checked={checked}
                onChange={() => { onSelect(run.run_id); }}
                className="sr-only"
              />
              <span className="flex items-start gap-3">
                <svg viewBox="0 0 40 14" className="mt-1 h-3.5 w-10 shrink-0 rounded-[2px] bg-canvas" aria-hidden="true">
                  <line x1="5" x2="35" y1="7" y2="7" stroke={style.stroke} strokeWidth="1.8" strokeDasharray={style.dash} strokeLinecap="round" />
                </svg>
                <span className="min-w-0 flex-1">
                  <span className="block font-medium leading-snug">{run.label}</span>
                  <span className="mt-0.5 block text-sm leading-snug text-graphite max-sm:hidden">{trajectorySummary(manifest)}</span>
                </span>
                {manifest !== undefined && <StatusBadge status={manifest.status} compact />}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
