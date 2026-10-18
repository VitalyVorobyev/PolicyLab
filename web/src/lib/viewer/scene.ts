import type { Model, Trajectory } from "../data/types";
import { lagrangeL1 } from "./lagrange";
import { boundsOf, type Bounds, type Transform } from "./transform";

export type Camera = "system" | "region" | "earth" | "moon";

/** Fill colors for the primaries; always paired with text labels in the viewer. */
export const EARTH_FILL = "#5b84b1";
export const MOON_FILL = "#b9c0c9";

export const CAMERA_LABEL: Record<Camera, string> = {
  region: "Orbit close-up",
  system: "Earth–Moon system",
  earth: "Near Earth",
  moon: "Near Moon",
};

/** Half-widths (km) of the near-body views: departure and entry arcs, and the lunar flyby. */
export const NEAR_EARTH_HALF_WIDTH_KM = 15000;
export const NEAR_MOON_HALF_WIDTH_KM = 9000;

function around(x: number, y: number, halfKm: number, lengthScaleKm: number): Bounds {
  const h = halfKm / lengthScaleKm;
  return { xMin: x - h, xMax: x + h, yMin: y - h, yMax: y + h };
}

/**
 * Data bounds for each camera preset. "system" frames both primaries and the trajectory;
 * "region" frames the trajectory together with L1 and the Moon; "earth" and "moon" are fixed
 * windows around one primary (in km, converted with the manifest's length scale).
 */
export function cameraBounds(camera: Camera, model: Model, trajectory: Trajectory): Bounds {
  if (camera === "earth") return around(model.earth.x, model.earth.y, NEAR_EARTH_HALF_WIDTH_KM, model.length_scale_km);
  if (camera === "moon") return around(model.moon.x, model.moon.y, NEAR_MOON_HALF_WIDTH_KM, model.length_scale_km);
  const orbit = boundsOf(trajectory.x, trajectory.y);
  const l1 = lagrangeL1(model.mu);
  if (camera === "system") {
    const earthR = model.earth.radius_km / model.length_scale_km;
    const xs = [orbit.xMin, orbit.xMax, model.earth.x - earthR, model.moon.x];
    return boundsOf(xs, [orbit.yMin, orbit.yMax, 0, 0], 0.06);
  }
  return boundsOf([orbit.xMin, orbit.xMax, l1, model.moon.x], [orbit.yMin, orbit.yMax, 0, 0], 0.08);
}

/** SVG path through the samples in screen space, rounded to 0.01 px. */
export function samplePath(transform: Transform, xs: ArrayLike<number>, ys: ArrayLike<number>, from = 0, to = xs.length): string {
  const parts: string[] = [];
  for (let i = Math.max(0, from); i < Math.min(to, xs.length); i++) {
    const sx = transform.toScreenX(xs[i] ?? 0).toFixed(2);
    const sy = transform.toScreenY(ys[i] ?? 0).toFixed(2);
    parts.push(`${parts.length === 0 ? "M" : "L"}${sx} ${sy}`);
  }
  return parts.join("");
}

/** Smallest on-screen radius at which a body is drawn; below it, the body is enlarged and labeled so. */
export const MIN_BODY_RADIUS_PX = 2.5;

export interface BodyGlyph {
  cx: number;
  cy: number;
  r: number;
  /** True when r is the physical radius at this zoom; false when enlarged for visibility or unknown. */
  toScale: boolean;
}

export function bodyGlyph(transform: Transform, x: number, y: number, radiusKm: number | null, lengthScaleKm: number): BodyGlyph {
  const cx = transform.toScreenX(x);
  const cy = transform.toScreenY(y);
  if (radiusKm === null) return { cx, cy, r: MIN_BODY_RADIUS_PX * 1.6, toScale: false };
  const r = (radiusKm / lengthScaleKm) * transform.scale;
  return r >= MIN_BODY_RADIUS_PX ? { cx, cy, r, toScale: true } : { cx, cy, r: MIN_BODY_RADIUS_PX, toScale: false };
}
