import { fixed } from "../../lib/format";
import type { ArmId, Atlas, AtlasArm, AtlasElite } from "../../lib/data/types";

export type Cell = readonly [number, number];

export const cellKey = (cell: Cell): string => `${cell[0]},${cell[1]}`;

export function sameCell(a: Cell | null, b: Cell | null): boolean {
  return a !== null && b !== null && a[0] === b[0] && a[1] === b[1];
}

export function eliteAt(arm: AtlasArm, cell: Cell | null): AtlasElite | undefined {
  return cell === null ? undefined : arm.elites.find((e) => sameCell(e.cell, cell));
}

/** The elite closest to a preferred cell (grid distance), so that switching method keeps the viewer near where they were. */
export function nearestElite(arm: AtlasArm, preferred: Cell | null): AtlasElite | undefined {
  if (preferred === null) return arm.elites[0];
  let best: AtlasElite | undefined;
  let bestDistance = Infinity;
  for (const e of arm.elites) {
    const d = (e.cell[0] - preferred[0]) ** 2 + (e.cell[1] - preferred[1]) ** 2;
    if (d < bestDistance) {
      best = e;
      bestDistance = d;
    }
  }
  return best;
}

export type Direction = "left" | "right" | "up" | "down";

/**
 * The next filled cell in an arrow direction. Cells are [d1, d2] with d1 growing to the right and
 * d2 (trip time) growing upwards. Among cells strictly beyond the current one along the direction,
 * the nearest wins, with sideways offset counted double so that "right" prefers the same row.
 */
export function neighbor(cells: readonly Cell[], from: Cell, direction: Direction): Cell | undefined {
  const horizontal = direction === "left" || direction === "right";
  const sign = direction === "right" || direction === "up" ? 1 : -1;
  let best: Cell | undefined;
  let bestScore = Infinity;
  for (const c of cells) {
    const forward = (horizontal ? c[0] - from[0] : c[1] - from[1]) * sign;
    if (forward <= 0) continue;
    const score = forward + 2 * Math.abs(horizontal ? c[1] - from[1] : c[0] - from[0]);
    if (score < bestScore) {
      best = c;
      bestScore = score;
    }
  }
  return best;
}

/** Sequential survival scale for the navy canvas: dark violet to light yellow (0 to 1), as fixed stops. */
const STOPS: readonly (readonly [number, number, number])[] = [
  [91, 58, 154],
  [47, 111, 176],
  [31, 166, 160],
  [127, 208, 106],
  [242, 230, 90],
];

export function survivalColor(s: number): string {
  const x = Math.min(1, Math.max(0, s)) * (STOPS.length - 1);
  const k = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - k;
  const a = STOPS[k] ?? STOPS[0] ?? [0, 0, 0];
  const b = STOPS[k + 1] ?? a;
  return `rgb(${[0, 1, 2].map((i) => Math.round((a[i] ?? 0) + ((b[i] ?? 0) - (a[i] ?? 0)) * f)).join(" ")})`;
}

/** Dark text on the light end of the scale, light text on the dark end (both above 4.5:1). */
export function survivalTextIsDark(s: number): boolean {
  return s >= 0.42;
}

export const SURVIVAL_GRADIENT = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map(survivalColor).join(", ")})`;

export const percent = (fraction: number): string => `${Math.round(fraction * 100)}\u00a0%`;

export function altitudeLabel(km: number): string {
  return `${fixed(km < 1000 ? Math.round(km) : Math.round(km / 10) * 10, 0)} km`;
}

export const daysLabel = (days: number): string => `${fixed(days, 1)} days`;

/** Edge labels for the closest-approach axis: two significant figures, as the edges are not round numbers. */
export const axisKm = (km: number): string => fixed(Number(km.toPrecision(2)), 0);

/** Display facts of one elite, shared by the readout and the accessible cell names. */
export function eliteSummary(e: AtlasElite): string {
  return `Closest approach ${altitudeLabel(e.perilune_altitude_km)}, trip ${daysLabel(e.return_time_days)}, survives ${percent(e.survival)} of launch errors`;
}

export interface ArmInfo {
  id: ArmId;
  name: string;
  /** Short label for the method switch. */
  short: string;
  line: string;
}

export const ARM_INFO: readonly ArmInfo[] = [
  { id: "D", name: "Dense sweep", short: "Dense sweep", line: "Fire 16,384 launches spread evenly over all settings and keep the best in each square. No search at all: the baseline to beat." },
  { id: "MS", name: "Multi-start", short: "Multi-start", line: "Start a local solver from many points; each run homes in on one nearby free return." },
  { id: "MBH", name: "Basin hopping", short: "Basin hopping", line: "Polish the best answer, kick it randomly, polish again, keep it if it improved." },
  { id: "DE", name: "Differential evolution", short: "Diff. evolution", line: "A population of launches that breeds new ones from the differences between its members." },
  { id: "CC", name: "Continuation", short: "Continuation", line: "Find one free return, then walk along its family square by square, re-solving at each step: the classical astrodynamics approach." },
  { id: "T", name: "MAP-Elites", short: "MAP-Elites", line: "Keep the best trajectory found in every square, and mutate those champions to improve them and to reach new squares." },
  { id: "T2", name: "MAP-Elites + polish", short: "MAP-Elites + polish", line: "MAP-Elites with a local solver that polishes one champion per batch." },
];

export function armOf(atlas: Atlas, id: ArmId): AtlasArm {
  const arm = atlas.arms.find((a) => a.arm === id);
  if (arm === undefined) throw new RangeError(`The atlas has no arm ${id}.`);
  return arm;
}

/** A 16-run mean; a missing mean (never present for the published arms) reads as zero. */
export const mean = (arm: AtlasArm, key: keyof AtlasArm["summary"]): number => arm.summary[key].mean ?? 0;
