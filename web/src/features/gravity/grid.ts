import type { FreeReturnOutcome, GridAxis, OutcomeGrid } from "../../lib/data/types";

/** A rectangle of grid cells, inclusive index ranges. */
export interface CellRegion {
  i0: number;
  i1: number;
  j0: number;
  j1: number;
}

/** Axis value at index i (the cell centre), rounded to remove float noise from min + i * step. */
export function axisValue(axis: GridAxis, i: number): number {
  return Number((axis.min + i * axis.step).toPrecision(12));
}

/** Inclusive index range of the cells whose centres lie in [lo, hi], clamped to the axis. */
export function indexRange(axis: GridAxis, lo: number, hi: number): [number, number] {
  const a = Math.max(0, Math.ceil((lo - axis.min) / axis.step - 1e-9));
  const b = Math.min(axis.count - 1, Math.floor((hi - axis.min) / axis.step + 1e-9));
  return [a, Math.max(a, b)];
}

/** Fractional index of a value on an axis (cell centres are integers). */
export function fractionalIndex(axis: GridAxis, value: number): number {
  return (value - axis.min) / axis.step;
}

export function regionFor(grid: OutcomeGrid, phase: [number, number], deltaV: [number, number]): CellRegion {
  const [i0, i1] = indexRange(grid.phaseAxis, phase[0], phase[1]);
  const [j0, j1] = indexRange(grid.deltaVAxis, deltaV[0], deltaV[1]);
  return { i0, i1, j0, j1 };
}

export function fullRegion(grid: OutcomeGrid): CellRegion {
  return { i0: 0, i1: grid.phaseAxis.count - 1, j0: 0, j1: grid.deltaVAxis.count - 1 };
}

export function codeAt(grid: OutcomeGrid, i: number, j: number): number {
  return grid.codes[i * grid.deltaVAxis.count + j] ?? 0;
}

export function outcomeAt(grid: OutcomeGrid, i: number, j: number): FreeReturnOutcome {
  return grid.outcomes[codeAt(grid, i, j)] ?? "no_return_within_t_max";
}

/** Number of cells per outcome (every outcome of the code table listed, zeros included). */
export function countOutcomes(grid: OutcomeGrid): Map<FreeReturnOutcome, number> {
  const counts = new Map<FreeReturnOutcome, number>(grid.outcomes.map((o) => [o, 0]));
  for (const code of grid.codes) {
    const outcome = grid.outcomes[code];
    if (outcome !== undefined) counts.set(outcome, (counts.get(outcome) ?? 0) + 1);
  }
  return counts;
}

/** Feasible-cell columns keyed by cell index, with named access. */
export interface FeasibleCell {
  i: number;
  j: number;
  values: Record<string, number>;
}

export function feasibleCells(grid: OutcomeGrid): Map<number, FeasibleCell> {
  const names = grid.feasible.columns.map((c) => c.name);
  const out = new Map<number, FeasibleCell>();
  for (const row of grid.feasible.rows) {
    const values: Record<string, number> = {};
    names.forEach((name, k) => {
      values[name] = row[k] ?? Number.NaN;
    });
    const i = row[0] ?? 0;
    const j = row[1] ?? 0;
    out.set(i * grid.deltaVAxis.count + j, { i, j, values });
  }
  return out;
}

/**
 * Downsample a region to a pw x ph pixel raster of outcome codes. Where a pixel covers several
 * cells it takes the outcome that comes first in `priority` (so a single feasible cell is never
 * hidden by its neighbours); where cells are larger than pixels each pixel takes its cell. Rows
 * run top to bottom with delta-v decreasing.
 */
export function rasterize(grid: OutcomeGrid, region: CellRegion, pw: number, ph: number, priority: readonly FreeReturnOutcome[]): Uint8Array {
  const rank = grid.outcomes.map((o) => {
    const r = priority.indexOf(o);
    return r < 0 ? priority.length : r;
  });
  const ni = region.i1 - region.i0 + 1;
  const nj = region.j1 - region.j0 + 1;
  const out = new Uint8Array(pw * ph);
  for (let py = 0; py < ph; py++) {
    // Pixel row py covers delta-v cells from the top (high j) down.
    const jTop = region.j1 - Math.floor((py * nj) / ph);
    const jBottom = region.j1 - Math.max(Math.floor((py * nj) / ph), Math.ceil(((py + 1) * nj) / ph) - 1);
    for (let px = 0; px < pw; px++) {
      const iLeft = region.i0 + Math.floor((px * ni) / pw);
      const iRight = region.i0 + Math.max(Math.floor((px * ni) / pw), Math.ceil(((px + 1) * ni) / pw) - 1);
      let best = codeAt(grid, iLeft, jTop);
      for (let i = iLeft; i <= iRight; i++) {
        for (let j = jBottom; j <= jTop; j++) {
          const code = codeAt(grid, i, j);
          if ((rank[code] ?? 0) < (rank[best] ?? 0)) best = code;
        }
      }
      out[py * pw + px] = best;
    }
  }
  return out;
}
