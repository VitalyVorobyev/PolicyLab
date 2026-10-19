import type { Polyline } from "../../lib/data/types";

/** Point on a polyline at time t (linear between stored vertices; held at the ends). */
export function pointAt(line: Polyline, t: number): { x: number; y: number } {
  const ts = line.t_days;
  const n = ts.length;
  if (t <= (ts[0] ?? 0)) return { x: line.x_km[0] ?? 0, y: line.y_km[0] ?? 0 };
  if (t >= (ts[n - 1] ?? 0)) return { x: line.x_km[n - 1] ?? 0, y: line.y_km[n - 1] ?? 0 };
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if ((ts[mid] ?? 0) <= t) lo = mid;
    else hi = mid;
  }
  const f = (t - (ts[lo] ?? 0)) / ((ts[hi] ?? 1) - (ts[lo] ?? 0));
  return {
    x: (line.x_km[lo] ?? 0) + f * ((line.x_km[hi] ?? 0) - (line.x_km[lo] ?? 0)),
    y: (line.y_km[lo] ?? 0) + f * ((line.y_km[hi] ?? 0) - (line.y_km[lo] ?? 0)),
  };
}

/** Number of vertices at or before time t (for drawing the travelled part of a path). */
export function countUpTo(line: Polyline, t: number): number {
  const ts = line.t_days;
  let lo = 0;
  let hi = ts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((ts[mid] ?? 0) <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** A pixel rectangle showing a window of the data plane at equal scale in x and y. */
export interface View {
  /** Pixel centre of the view. */
  ox: number;
  oy: number;
  /** Data point (km) shown at the centre. */
  cx: number;
  cy: number;
  /** Pixels per km. */
  scale: number;
}

export const toX = (v: View, x: number): number => v.ox + (x - v.cx) * v.scale;
export const toY = (v: View, y: number): number => v.oy - (y - v.cy) * v.scale;

export interface Box {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/** Fit a data box into a pixel rectangle with equal scale, centred. */
export function fitView(box: Box, rect: { x: number; y: number; w: number; h: number }): View {
  const spanX = Math.max(box.xMax - box.xMin, 1);
  const spanY = Math.max(box.yMax - box.yMin, 1);
  return {
    ox: rect.x + rect.w / 2,
    oy: rect.y + rect.h / 2,
    cx: (box.xMin + box.xMax) / 2,
    cy: (box.yMin + box.yMax) / 2,
    scale: Math.min(rect.w / spanX, rect.h / spanY),
  };
}

/** Bounds of a path together with both bodies (padded by their radii). */
export function sceneBox(line: Polyline, bodies: readonly { x: number; y: number; r: number }[]): Box {
  const box = { xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity };
  const add = (x: number, y: number): void => {
    box.xMin = Math.min(box.xMin, x);
    box.xMax = Math.max(box.xMax, x);
    box.yMin = Math.min(box.yMin, y);
    box.yMax = Math.max(box.yMax, y);
  };
  for (let i = 0; i < line.x_km.length; i++) add(line.x_km[i] ?? 0, line.y_km[i] ?? 0);
  for (const b of bodies) {
    add(b.x - b.r, b.y - b.r);
    add(b.x + b.r, b.y + b.r);
  }
  return box;
}

/**
 * Half-width (km) of the Moon inset: about four times the closest-approach distance, kept between
 * a window that shows the Moon clearly and one that is still a real zoom of the main view.
 */
export function insetHalfWidthKm(perilouneAltitudeKm: number, moonRadiusKm: number): number {
  return Math.min(80000, Math.max(8000, 4 * (perilouneAltitudeKm + moonRadiusKm)));
}

export type Corner = "tl" | "tr" | "bl" | "br";

export function cornerOrigin(c: Corner, width: number, height: number, size: number, margin: number): { x: number; y: number } {
  return {
    x: c === "tl" || c === "bl" ? margin : width - margin - size,
    y: c === "tl" || c === "tr" ? margin : height - margin - size,
  };
}

/** The stage corner whose inset square covers the fewest points of the path (ties: top right first). */
export function quietestCorner(
  screenPoints: readonly { x: number; y: number }[],
  width: number,
  height: number,
  size: number,
  margin: number,
): Corner {
  let best: Corner = "tr";
  let bestCount = Infinity;
  for (const c of ["tr", "br", "tl", "bl"] as const) {
    const r = cornerOrigin(c, width, height, size, margin);
    let count = 0;
    for (const p of screenPoints) if (p.x >= r.x - 8 && p.x <= r.x + size + 8 && p.y >= r.y - 8 && p.y <= r.y + size + 8) count++;
    if (count < bestCount) {
      best = c;
      bestCount = count;
    }
  }
  return best;
}
