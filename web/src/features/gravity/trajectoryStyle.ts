import { humanize } from "../../lib/format";
import type { Manifest } from "../../lib/data/types";

/**
 * Line styles for separate trajectories: each differs in color *and* dash pattern, so the picker
 * swatches and the drawings stay distinguishable without color. Assigned by position among the
 * study's published trajectories.
 */
export interface TrajectoryStyle {
  stroke: string;
  /** SVG stroke-dasharray; undefined for a solid line. */
  dash: string | undefined;
  /** Plain-language name of the pattern, for legends. */
  pattern: string;
}

const STYLES: readonly TrajectoryStyle[] = [
  { stroke: "#e4e9f0", dash: undefined, pattern: "solid line" },
  { stroke: "#8fc3e3", dash: "9 5", pattern: "dashed line" },
  { stroke: "#d8c79a", dash: "1.5 4", pattern: "dotted line" },
];

export function trajectoryStyle(index: number): TrajectoryStyle {
  return STYLES[index % STYLES.length] ?? STYLES[0] ?? { stroke: "#e4e9f0", dash: undefined, pattern: "solid line" };
}

/** A free-return run: propagated with the terminal Earth return shell (M3 candidates). */
export function isFreeReturn(manifest: Manifest): boolean {
  return manifest.solver.return_shell !== null;
}

/** Short candidate tag from a run id, e.g. "gravity-free-return-c1-v2" -> "c1"; null if none. */
export function candidateTag(runId: string): string | null {
  const match = /-(c\d+)-/.exec(runId);
  return match?.[1] ?? null;
}

const EVENT_LABELS: Record<string, string> = {
  departure: "TLI",
  perilune: "Perilune",
  return_shell_crossing: "Entry interface",
};

/** Label for a trajectory event name; unknown names are shown as words. */
export function eventLabel(name: string): string {
  return EVENT_LABELS[name] ?? humanize(name);
}

const EVENT_DESCRIPTIONS: Record<string, string> = {
  departure: "Translunar injection: the single impulse from the circular parking orbit",
  perilune: "Closest approach to the Moon before the return",
  return_shell_crossing: "Inbound crossing of the entry-interface shell; the propagation ends here",
};

export function eventDescription(name: string): string | undefined {
  return EVENT_DESCRIPTIONS[name];
}
