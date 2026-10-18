import { fixed } from "../../lib/format";
import type { FreeReturnOutcome, Manifest } from "../../lib/data/types";

/**
 * Drawing styles for free-return outcomes on the navy canvas. Fills step up in lightness from the
 * common "no return" to the rare feasible cells, and the minority classes also carry a pattern
 * (hatch, dot, hollow) in the zoomed raster and the legend, so the map does not rely on hue alone.
 */
export type OutcomePattern = "none" | "hatch" | "dot" | "hollow";

export interface OutcomeStyle {
  fill: string;
  /** Color of the pattern drawn over the fill. */
  mark: string;
  pattern: OutcomePattern;
}

export const OUTCOME_STYLE: Record<FreeReturnOutcome, OutcomeStyle> = {
  no_return_within_t_max: { fill: "#172740", mark: "#172740", pattern: "none" },
  perilune_out_of_band: { fill: "#34507a", mark: "#34507a", pattern: "none" },
  return_before_lunar_encounter: { fill: "#6d84a6", mark: "#22324d", pattern: "dot" },
  moon_collision: { fill: "#c3c9d2", mark: "#5d6878", pattern: "hatch" },
  earth_collision: { fill: "#5b84b1", mark: "#0c1829", pattern: "hatch" },
  numerical_failure: { fill: "#d0605e", mark: "#0c1829", pattern: "dot" },
  invalid_start: { fill: "#a07a52", mark: "#0c1829", pattern: "dot" },
  feasible: { fill: "#52d1a4", mark: "#52d1a4", pattern: "none" },
};

/** Feasible cells whose lunar passage sense is +1 are drawn hollow (a second family at this resolution). */
export const FEASIBLE_PLUS_STYLE: OutcomeStyle = { fill: "#52d1a4", mark: "#0c1829", pattern: "hollow" };

/** Order of the legend: what a reader looks for first. */
export const OUTCOME_ORDER: readonly FreeReturnOutcome[] = [
  "feasible",
  "moon_collision",
  "return_before_lunar_encounter",
  "perilune_out_of_band",
  "no_return_within_t_max",
  "earth_collision",
  "numerical_failure",
  "invalid_start",
];

/** Plain-language outcome names; bounds come from the manifest's definition metrics when present. */
export function outcomeLabel(outcome: FreeReturnOutcome, manifest: Manifest | null): string {
  const metric = (key: string): number | undefined => manifest?.metrics[key]?.value;
  const tMax = metric("t_max_days");
  const lo = metric("perilune_band_min_km");
  const hi = metric("perilune_band_max_km");
  switch (outcome) {
    case "feasible":
      return "Feasible free return";
    case "no_return_within_t_max":
      return tMax === undefined ? "No return within the time window" : `No return within ${fixed(tMax, 0)} days`;
    case "perilune_out_of_band":
      return lo === undefined || hi === undefined
        ? "Returns, but perilune outside the band"
        : `Returns, but perilune outside ${fixed(lo, 0)}–${fixed(hi, 0)} km`;
    case "return_before_lunar_encounter":
      return "Returns before any lunar encounter";
    case "moon_collision":
      return "Moon collision";
    case "earth_collision":
      return "Earth collision";
    case "numerical_failure":
      return "Numerical failure";
    case "invalid_start":
      return "Invalid start";
  }
}

/** Fill one cell of a raster with an outcome's fill and pattern (patterns only when the cell is large enough). */
export function paintCell(ctx: CanvasRenderingContext2D, style: OutcomeStyle, x: number, y: number, w: number, h: number): void {
  ctx.fillStyle = style.fill;
  ctx.fillRect(x, y, w, h);
  if (style.pattern === "none" || w < 5 || h < 5) return;
  ctx.fillStyle = style.mark;
  ctx.strokeStyle = style.mark;
  if (style.pattern === "hatch") {
    ctx.lineWidth = Math.max(1, Math.min(w, h) * 0.12);
    ctx.beginPath();
    ctx.moveTo(x + 1, y + h - 1);
    ctx.lineTo(x + w - 1, y + 1);
    ctx.stroke();
  } else if (style.pattern === "dot") {
    ctx.beginPath();
    ctx.arc(x + w / 2, y + h / 2, Math.max(1, Math.min(w, h) * 0.16), 0, 2 * Math.PI);
    ctx.fill();
  } else {
    const inset = Math.max(1.5, Math.min(w, h) * 0.28);
    ctx.fillRect(x + inset, y + inset, w - 2 * inset, h - 2 * inset);
  }
}
