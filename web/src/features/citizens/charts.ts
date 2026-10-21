/** Small shared helpers for the page's SVG charts. */

export const MODEL_DASH: Record<string, string | undefined> = {
  B1: undefined,
  B2: "7 3",
  B3: "2 3",
  B23: "9 3 2 3",
};

/** Spread label positions (sorted by y) so that neighbours are at least `gap` apart, inside [lo, hi]. */
export function spreadLabels<T extends { y: number }>(labels: T[], gap: number, lo: number, hi: number): (T & { ly: number })[] {
  const sorted = [...labels].sort((a, b) => a.y - b.y).map((l) => ({ ...l, ly: l.y }));
  for (let pass = 0; pass < 20; pass++) {
    let moved = false;
    for (let i = 1; i < sorted.length; i++) {
      const a = sorted[i - 1];
      const b = sorted[i];
      if (a === undefined || b === undefined) continue;
      const overlap = gap - (b.ly - a.ly);
      if (overlap > 0) {
        a.ly -= overlap / 2;
        b.ly += overlap / 2;
        moved = true;
      }
    }
    for (const l of sorted) l.ly = Math.max(lo, Math.min(hi, l.ly));
    if (!moved) break;
  }
  return sorted;
}

/** Round a range outwards to multiples of `step`. */
export function niceRange(min: number, max: number, step: number): [number, number] {
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}
