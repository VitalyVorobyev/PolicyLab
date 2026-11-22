export interface LabelPos {
  id: string;
  rawY: number;
  text: string;
  color: string;
}

export interface PlacedLabel extends LabelPos {
  y: number;
}

/** Space vertically adjacent right-edge SVG labels by at least `minGap` px without exceeding `[minY, maxY]`. */
export function placeRightLabels(items: readonly LabelPos[], minY: number, maxY: number, minGap = 13): PlacedLabel[] {
  const sorted = [...items].sort((a, b) => a.rawY - b.rawY);
  const ys: number[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const item = sorted[i];
    if (item === undefined) continue;
    const floorY = i === 0 ? minY : (ys[i - 1] ?? minY) + minGap;
    ys.push(Math.max(item.rawY, floorY));
  }
  for (let i = ys.length - 1; i >= 0; i--) {
    const ceilingY = i === ys.length - 1 ? maxY : (ys[i + 1] ?? maxY) - minGap;
    ys[i] = Math.min(ys[i] ?? maxY, ceilingY);
  }
  return sorted.map((item, i) => ({ ...item, y: ys[i] ?? item.rawY }));
}
