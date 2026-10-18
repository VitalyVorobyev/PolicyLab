/**
 * The single data -> screen transform used by the orbit viewer.
 *
 * Data coordinates are normalized CR3BP units in the barycentric synodic frame (x right, y up).
 * Screen coordinates are SVG user units with y pointing down. The transform uses one scale
 * factor for both axes, so the aspect ratio of the physical geometry is preserved exactly.
 */

export interface Bounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface Transform {
  /** Screen units per normalized length unit (identical for x and y). */
  readonly scale: number;
  readonly width: number;
  readonly height: number;
  readonly insets: Insets;
  /** The data region actually visible inside the insets (a superset of the requested bounds). */
  readonly visible: Bounds;
  toScreenX(x: number): number;
  toScreenY(y: number): number;
  toDataX(sx: number): number;
  toDataY(sy: number): number;
}

/**
 * Fit `bounds` into a `width` x `height` viewport minus `insets`, centered, with equal x/y scale.
 * Throws on degenerate input rather than producing NaN coordinates.
 */
export function fitTransform(bounds: Bounds, width: number, height: number, insets: Insets): Transform {
  const innerW = width - insets.left - insets.right;
  const innerH = height - insets.top - insets.bottom;
  const spanX = bounds.xMax - bounds.xMin;
  const spanY = bounds.yMax - bounds.yMin;
  if (!(innerW > 0 && innerH > 0)) throw new RangeError("Viewport is smaller than its insets.");
  if (!(spanX > 0 && spanY > 0) || !Number.isFinite(spanX) || !Number.isFinite(spanY)) {
    throw new RangeError("Bounds must have a positive, finite extent on both axes.");
  }
  const scale = Math.min(innerW / spanX, innerH / spanY);
  const cx = (bounds.xMin + bounds.xMax) / 2;
  const cy = (bounds.yMin + bounds.yMax) / 2;
  const screenCx = insets.left + innerW / 2;
  const screenCy = insets.top + innerH / 2;
  const halfW = innerW / scale / 2;
  const halfH = innerH / scale / 2;

  return {
    scale,
    width,
    height,
    insets,
    visible: { xMin: cx - halfW, xMax: cx + halfW, yMin: cy - halfH, yMax: cy + halfH },
    toScreenX: (x) => screenCx + (x - cx) * scale,
    toScreenY: (y) => screenCy - (y - cy) * scale,
    toDataX: (sx) => cx + (sx - screenCx) / scale,
    toDataY: (sy) => cy - (sy - screenCy) / scale,
  };
}

/** A "nice" step (1, 2 or 5 times a power of ten) giving roughly `target` intervals over `span`. */
export function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target);
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const nice = fraction < 1.5 ? 1 : fraction < 3.5 ? 2 : fraction < 7.5 ? 5 : 10;
  return nice * power;
}

/** Multiples of `step` within [min, max], rounded to the step's precision to avoid 0.30000000000000004. */
export function ticks(min: number, max: number, step: number): number[] {
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  const out: number[] = [];
  for (let k = Math.ceil(min / step); k * step <= max + step * 1e-9; k++) {
    out.push(Number((k * step).toFixed(decimals)));
  }
  return out;
}

/** Bounding box of point sets, expanded by `margin` (a fraction of the larger span) on every side. */
export function boundsOf(xs: ArrayLike<number>, ys: ArrayLike<number>, margin = 0): Bounds {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] ?? Number.NaN;
    const y = ys[i] ?? Number.NaN;
    xMin = Math.min(xMin, x);
    xMax = Math.max(xMax, x);
    yMin = Math.min(yMin, y);
    yMax = Math.max(yMax, y);
  }
  const pad = Math.max(xMax - xMin, yMax - yMin) * margin;
  return { xMin: xMin - pad, xMax: xMax + pad, yMin: yMin - pad, yMax: yMax + pad };
}
