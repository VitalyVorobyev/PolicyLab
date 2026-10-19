import type { Atlas, AtlasElite, FanDraw, Polyline } from "../../lib/data/types";
import { EARTH_FILL, MOON_FILL } from "../../lib/viewer/scene";
import { fixed } from "../../lib/format";
import {
  cornerOrigin,
  countUpTo,
  fitView,
  insetHalfWidthKm,
  pointAt,
  quietestCorner,
  sceneBox,
  toX,
  toY,
  type Corner,
  type View,
} from "./orbitGeometry";

export interface Colors {
  text: string;
  strong: string;
  accent: string;
  ok: string;
  fail: string;
  bg: string;
  raised: string;
  grid: string;
  axis: string;
}

const FALLBACK: Colors = {
  text: "#a8b5c7",
  strong: "#e4e9f0",
  accent: "#e3a72f",
  ok: "#4fd1b0",
  fail: "#f2785c",
  bg: "#0c1829",
  raised: "#13223a",
  grid: "#1d2e48",
  axis: "#3a4e6b",
};

/** Canvas colors come from the site's CSS tokens, so the stage follows the theme. */
export function readColors(): Colors {
  if (typeof document === "undefined") return FALLBACK;
  const style = getComputedStyle(document.documentElement);
  const get = (name: string, fallback: string): string => style.getPropertyValue(name).trim() || fallback;
  return {
    text: get("--color-canvas-text", FALLBACK.text),
    strong: get("--color-canvas-strong", FALLBACK.strong),
    accent: get("--color-accent", FALLBACK.accent),
    ok: get("--color-fan-ok", FALLBACK.ok),
    fail: get("--color-fan-fail", FALLBACK.fail),
    bg: get("--color-canvas", FALLBACK.bg),
    raised: get("--color-canvas-raised", FALLBACK.raised),
    grid: get("--color-canvas-grid", FALLBACK.grid),
    axis: get("--color-canvas-axis", FALLBACK.axis),
  };
}

interface ViewPaths {
  view: View;
  ok: Path2D;
  fail: Path2D;
  nominal: Path2D;
}

export interface Inset {
  x: number;
  y: number;
  size: number;
  corner: Corner;
  halfWidthKm: number;
}

export interface Scene {
  width: number;
  height: number;
  elite: AtlasElite | null;
  draws: FanDraw[] | null;
  duration: number;
  earth: Atlas["earth"];
  moon: Atlas["moon"];
  main: ViewPaths;
  inset: (ViewPaths & Inset) | null;
}

function pathThrough(line: Polyline, view: View, into: Path2D): void {
  for (let i = 0; i < line.t_days.length; i++) {
    const x = toX(view, line.x_km[i] ?? 0);
    const y = toY(view, line.y_km[i] ?? 0);
    if (i === 0) into.moveTo(x, y);
    else into.lineTo(x, y);
  }
}

function buildPaths(view: View, nominal: Polyline | null, draws: FanDraw[] | null): ViewPaths {
  const ok = new Path2D();
  const fail = new Path2D();
  const nom = new Path2D();
  if (nominal !== null) pathThrough(nominal, view, nom);
  for (const d of draws ?? []) pathThrough(d, view, d.uncorrected_feasible ? ok : fail);
  return { view, ok, fail, nominal: nom };
}

/** Everything that depends on the selection and the stage size; rebuilt only when one of those changes. */
export function buildScene(atlas: Atlas, elite: AtlasElite | null, draws: FanDraw[] | null, width: number, height: number): Scene {
  const margin = 14;
  const rect = { x: margin, y: margin + 6, w: Math.max(40, width - 2 * margin), h: Math.max(40, height - 2 * margin - 6) };
  const bodies = [
    { x: atlas.earth.x_km, y: atlas.earth.y_km, r: atlas.earth.radius_km },
    { x: atlas.moon.x_km, y: atlas.moon.y_km, r: atlas.moon.radius_km },
  ];
  // Without a selection, frame the two bodies and the typical band of paths between them.
  const nominal = elite?.path ?? null;
  const base = nominal !== null ? sceneBox(nominal, bodies) : { xMin: atlas.earth.x_km - 20000, xMax: atlas.moon.x_km + 20000, yMin: -90000, yMax: 90000 };
  const main = buildPaths(fitView(base, rect), nominal, draws);

  let inset: Scene["inset"] = null;
  if (elite !== null) {
    const size = Math.round(Math.min(190, Math.max(112, Math.min(width, height) * 0.36)));
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < elite.path.x_km.length; i++) pts.push({ x: toX(main.view, elite.path.x_km[i] ?? 0), y: toY(main.view, elite.path.y_km[i] ?? 0) });
    const corner = quietestCorner(pts, width, height, size, margin);
    const origin = cornerOrigin(corner, width, height, size, margin);
    const half = insetHalfWidthKm(elite.perilune_altitude_km, atlas.moon.radius_km);
    const view = fitView(
      { xMin: atlas.moon.x_km - half, xMax: atlas.moon.x_km + half, yMin: atlas.moon.y_km - half, yMax: atlas.moon.y_km + half },
      { x: origin.x, y: origin.y, w: size, h: size },
    );
    inset = { ...buildPaths(view, nominal, draws), ...origin, size, corner, halfWidthKm: half };
  }
  return { width, height, elite, draws, duration: elite?.return_time_days ?? 1, earth: atlas.earth, moon: atlas.moon, main, inset };
}

const TAU = Math.PI * 2;

function drawBody(ctx: CanvasRenderingContext2D, view: View, body: { x_km: number; y_km: number; radius_km: number }, fill: string, label: string | null, c: Colors): void {
  const cx = toX(view, body.x_km);
  const cy = toY(view, body.y_km);
  const r = Math.max(2.4, body.radius_km * view.scale);
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, TAU);
  ctx.fillStyle = fill;
  ctx.fill();
  if (label !== null) {
    ctx.font = '500 12px "IBM Plex Sans", system-ui, sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    halo(ctx, label, cx, cy + r + 5, c);
  }
}

/** Text with a thin background-coloured outline, so labels stay legible over the fan of paths. */
function halo(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, c: Colors, fill = c.strong): void {
  ctx.lineJoin = "round";
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = c.bg;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = fill;
  ctx.fillText(text, x, y);
}

function drawLayer(ctx: CanvasRenderingContext2D, s: Scene, vp: ViewPaths, u: number, c: Colors, labels: boolean, markerScale: number): void {
  const t = u * s.duration;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  // Error variants: thin paths, the ones that come home in one colour, the ones that fail in another.
  ctx.lineWidth = 0.9;
  ctx.globalAlpha = 0.5;
  ctx.strokeStyle = c.fail;
  ctx.stroke(vp.fail);
  ctx.strokeStyle = c.ok;
  ctx.stroke(vp.ok);
  ctx.globalAlpha = 1;

  if (s.elite !== null) {
    // The chosen trajectory: faint in full, bright where it has been flown.
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = c.strong;
    ctx.globalAlpha = 0.4;
    ctx.stroke(vp.nominal);
    ctx.globalAlpha = 1;
    const n = countUpTo(s.elite.path, t);
    if (n > 1) {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = toX(vp.view, s.elite.path.x_km[i] ?? 0);
        const y = toY(vp.view, s.elite.path.y_km[i] ?? 0);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      const p = pointAt(s.elite.path, t);
      ctx.lineTo(toX(vp.view, p.x), toY(vp.view, p.y));
      ctx.lineWidth = 2.4;
      ctx.stroke();
    }
  }

  drawBody(ctx, vp.view, s.earth, EARTH_FILL, labels ? "Earth" : null, c);
  drawBody(ctx, vp.view, s.moon, MOON_FILL, labels ? "Moon" : null, c);

  // Variants: filled dot = came home, ring = did not. The chosen trajectory is the larger amber dot.
  for (const d of s.draws ?? []) {
    const p = pointAt(d, t);
    const x = toX(vp.view, p.x);
    const y = toY(vp.view, p.y);
    ctx.beginPath();
    ctx.arc(x, y, 2.3 * markerScale, 0, TAU);
    if (d.uncorrected_feasible) {
      ctx.fillStyle = c.ok;
      ctx.fill();
    } else {
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = c.fail;
      ctx.stroke();
    }
  }
  if (s.elite !== null) {
    const p = pointAt(s.elite.path, t);
    ctx.beginPath();
    ctx.arc(toX(vp.view, p.x), toY(vp.view, p.y), 4.6 * markerScale, 0, TAU);
    ctx.fillStyle = c.accent;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = c.strong;
    ctx.stroke();
  }
}

/** Draw one frame at progress u in [0, 1] along the chosen trajectory. */
export function renderScene(ctx: CanvasRenderingContext2D, s: Scene, u: number, c: Colors): void {
  ctx.clearRect(0, 0, s.width, s.height);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, s.width, s.height);
  ctx.clip();
  drawLayer(ctx, s, s.main, u, c, true, 1);
  ctx.restore();

  const inset = s.inset;
  if (inset === null) return;
  const mx = toX(s.main.view, s.moon.x_km);
  const my = toY(s.main.view, s.moon.y_km);
  // Where the inset looks, drawn on the main view.
  ctx.save();
  ctx.setLineDash([2, 3]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = c.text;
  ctx.globalAlpha = 0.8;
  ctx.beginPath();
  ctx.arc(mx, my, Math.max(5, inset.halfWidthKm * s.main.view.scale), 0, TAU);
  ctx.stroke();
  ctx.restore();

  const r = inset.size / 2;
  const cx = inset.x + r;
  const cy = inset.y + r;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, TAU);
  ctx.fillStyle = c.bg;
  ctx.fill();
  ctx.clip();
  drawLayer(ctx, s, inset, u, c, false, 1);
  ctx.restore();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, TAU);
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = c.axis;
  ctx.stroke();

  // Scale bar and caption for the inset.
  const barKm = niceBar(inset.halfWidthKm);
  const barPx = barKm * inset.view.scale;
  ctx.font = '11px "IBM Plex Sans", system-ui, sans-serif';
  ctx.textAlign = "center";
  const by = cy + r - 26;
  ctx.fillStyle = c.text;
  ctx.fillRect(cx - barPx / 2, by, barPx, 1.5);
  ctx.textBaseline = "top";
  halo(ctx, `${fixed(barKm, 0)} km`, cx, by + 4, c, c.text);
  halo(ctx, "Moon, close up", cx, inset.y + 10, c, c.text);
}

/** A 1/2/5 × 10^k km length close to a quarter of the inset's width. */
export function niceBar(halfWidthKm: number): number {
  const raw = (halfWidthKm * 2) / 4;
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * power;
}
