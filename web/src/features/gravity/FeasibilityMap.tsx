import { memo, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { fixed } from "../../lib/format";
import type { FreeReturnOutcome, LoadedGridRun, OutcomeGrid } from "../../lib/data/types";
import { useElementWidth } from "../../lib/useElementWidth";
import { niceStep, ticks } from "../../lib/viewer/transform";
import {
  axisValue,
  countOutcomes,
  feasibleCells,
  fractionalIndex,
  fullRegion,
  outcomeAt,
  rasterize,
  regionFor,
  type CellRegion,
  type FeasibleCell,
} from "./grid";
import { FEASIBLE_PLUS_STYLE, OUTCOME_ORDER, OUTCOME_STYLE, outcomeLabel, paintCell, type OutcomeStyle } from "./outcomes";

/** The zoomed region: the feasible band and its neighbourhood (both families seen on the grid). */
export const DETAIL_PHASE_DEG: [number, number] = [210, 256];
export const DETAIL_DELTA_V_KM_S: [number, number] = [3.12, 3.19];

export interface MapCandidate {
  runId: string;
  tag: string;
  label: string;
  phaseDeg: number;
  deltaVKmS: number;
}

function styleOf(outcome: FreeReturnOutcome, cell: FeasibleCell | undefined): OutcomeStyle {
  if (outcome === "feasible" && (cell?.values.lunar_passage_sense ?? -1) > 0) return FEASIBLE_PLUS_STYLE;
  return OUTCOME_STYLE[outcome];
}

/** Size a canvas for the device pixel ratio and return its 2-D context scaled to CSS pixels. */
function prepare(canvas: HTMLCanvasElement | null, width: number, height: number): CanvasRenderingContext2D | null {
  if (canvas === null) return null;
  const dpr = typeof window === "undefined" ? 1 : Math.min(3, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  if (ctx === null) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

interface Frame {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function plotW(f: Frame): number {
  return f.width - f.left - f.right;
}

function plotH(f: Frame): number {
  return f.height - f.top - f.bottom;
}

/** Screen position of fractional cell indices inside a region. */
function placer(frame: Frame, region: CellRegion) {
  const ni = region.i1 - region.i0 + 1;
  const nj = region.j1 - region.j0 + 1;
  const cw = plotW(frame) / ni;
  const ch = plotH(frame) / nj;
  return {
    cw,
    ch,
    x: (i: number) => frame.left + (i - region.i0 + 0.5) * cw,
    y: (j: number) => frame.top + (region.j1 - j + 0.5) * ch,
    cell: (px: number, py: number): { i: number; j: number } | null => {
      const i = region.i0 + Math.floor((px - frame.left) / cw);
      const j = region.j1 - Math.floor((py - frame.top) / ch);
      return i < region.i0 || i > region.i1 || j < region.j0 || j > region.j1 ? null : { i, j };
    },
  };
}

function Axes({ grid, frame, region, small = false }: { grid: OutcomeGrid; frame: Frame; region: CellRegion; small?: boolean }) {
  const p = placer(frame, region);
  const thetaLo = axisValue(grid.phaseAxis, region.i0);
  const thetaHi = axisValue(grid.phaseAxis, region.i1);
  const dvLo = axisValue(grid.deltaVAxis, region.j0);
  const dvHi = axisValue(grid.deltaVAxis, region.j1);
  const thetaStep = niceStep(thetaHi - thetaLo, Math.max(3, Math.round(plotW(frame) / (small ? 70 : 90))));
  const dvStep = niceStep(dvHi - dvLo, small ? 1 : Math.max(3, Math.round(plotH(frame) / 55)));
  const dvDecimals = Math.max(2, -Math.floor(Math.log10(dvStep)));
  const xOf = (theta: number): number => p.x(fractionalIndex(grid.phaseAxis, theta));
  const yOf = (dv: number): number => p.y(fractionalIndex(grid.deltaVAxis, dv));
  const bottom = frame.height - frame.bottom;
  return (
    <g aria-hidden="true" className="text-[11px]">
      <rect x={frame.left} y={frame.top} width={plotW(frame)} height={plotH(frame)} fill="none" stroke="var(--color-canvas-axis)" />
      {ticks(thetaLo, thetaHi, thetaStep).map((theta) => (
        <g key={`t${theta}`}>
          <line x1={xOf(theta)} x2={xOf(theta)} y1={bottom} y2={bottom + 4} stroke="var(--color-canvas-axis)" />
          <text x={xOf(theta)} y={bottom + 16} textAnchor="middle" className="numeric fill-canvas-text">
            {fixed(theta, 0)}°
          </text>
        </g>
      ))}
      {(small ? [dvLo, dvHi] : ticks(dvLo, dvHi, dvStep)).map((dv) => (
        <g key={`v${dv}`}>
          <line x1={frame.left - 4} x2={frame.left} y1={yOf(dv)} y2={yOf(dv)} stroke="var(--color-canvas-axis)" />
          <text x={frame.left - 7} y={yOf(dv) + 4} textAnchor="end" className="numeric fill-canvas-text">
            {fixed(dv, dvDecimals)}
          </text>
        </g>
      ))}
      {!small && (
        <>
          <text x={frame.left + plotW(frame)} y={bottom + 34} textAnchor="end" className="fill-canvas-text">
            Departure phase θ (deg)
          </text>
          <text x={0} y={frame.top - 6} textAnchor="start" className="fill-canvas-text">
            Δv (km/s)
          </text>
        </>
      )}
    </g>
  );
}

/** Full-grid overview: a pixel covering several cells shows the outcome listed first in the legend. */
const Overview = memo(function Overview({ grid, width, detail }: { grid: OutcomeGrid; width: number; detail: CellRegion }) {
  const frame = useMemo<Frame>(() => ({ width, height: width < 640 ? 112 : 132, left: 52, right: 12, top: 10, bottom: 24 }), [width]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const region = useMemo(() => fullRegion(grid), [grid]);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = prepare(canvas, frame.width, frame.height);
    if (ctx === null || canvas === null) return;
    const dpr = canvas.width / frame.width;
    const pw = Math.max(1, Math.round(plotW(frame) * dpr));
    const ph = Math.max(1, Math.round(plotH(frame) * dpr));
    const codes = rasterize(grid, region, pw, ph, OUTCOME_ORDER);
    const image = ctx.createImageData(pw, ph);
    const colors = grid.outcomes.map((o) => hexToRgb(OUTCOME_STYLE[o].fill));
    codes.forEach((code, k) => {
      const [r, g, b] = colors[code] ?? [0, 0, 0];
      image.data[4 * k] = r;
      image.data[4 * k + 1] = g;
      image.data[4 * k + 2] = b;
      image.data[4 * k + 3] = 255;
    });
    ctx.putImageData(image, Math.round(frame.left * dpr), Math.round(frame.top * dpr));
  }, [grid, region, frame]);
  const p = placer(frame, region);
  return (
    <div className="relative">
      <canvas ref={canvasRef} style={{ width: frame.width, height: frame.height }} className="block" aria-hidden="true" />
      <svg className="absolute inset-0" width={frame.width} height={frame.height} aria-hidden="true">
        <Axes grid={grid} frame={frame} region={region} small />
        <rect
          x={p.x(detail.i0 - 0.5)}
          y={p.y(detail.j1 + 0.5)}
          width={p.x(detail.i1 + 0.5) - p.x(detail.i0 - 0.5)}
          height={p.y(detail.j0 - 0.5) - p.y(detail.j1 + 0.5)}
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth={1.5}
        />
      </svg>
    </div>
  );
});

/** The zoomed raster: one rectangle per grid cell, with patterns for the minority outcomes. */
const DetailCanvas = memo(function DetailCanvas({ grid, frame, region, cells }: { grid: OutcomeGrid; frame: Frame; region: CellRegion; cells: Map<number, FeasibleCell> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = prepare(canvasRef.current, frame.width, frame.height);
    if (ctx === null) return;
    const dpr = (canvasRef.current?.width ?? frame.width) / frame.width;
    const snap = (v: number): number => Math.round(v * dpr) / dpr;
    const p = placer(frame, region);
    for (let i = region.i0; i <= region.i1; i++) {
      const x0 = snap(p.x(i - 0.5));
      const x1 = snap(p.x(i + 0.5));
      for (let j = region.j0; j <= region.j1; j++) {
        const y0 = snap(p.y(j + 0.5));
        const y1 = snap(p.y(j - 0.5));
        const outcome = outcomeAt(grid, i, j);
        paintCell(ctx, styleOf(outcome, cells.get(i * grid.deltaVAxis.count + j)), x0, y0, x1 - x0, y1 - y0);
      }
    }
  }, [grid, frame, region, cells]);
  return <canvas ref={canvasRef} style={{ width: frame.width, height: frame.height }} className="block" aria-hidden="true" />;
});

function describeCell(grid: OutcomeGrid, run: LoadedGridRun, cells: Map<number, FeasibleCell>, i: number, j: number): string {
  const outcome = outcomeAt(grid, i, j);
  const where = `θ = ${fixed(axisValue(grid.phaseAxis, i), 1)}°, Δv = ${fixed(axisValue(grid.deltaVAxis, j), 3)} km/s`;
  const cell = cells.get(i * grid.deltaVAxis.count + j);
  if (outcome === "feasible" && cell !== undefined) {
    const v = cell.values;
    return (
      `${where}: feasible. Perilune ${fixed(v.perilune_altitude_km ?? Number.NaN, 0)} km on day ${fixed(v.lunar_encounter_time_days ?? Number.NaN, 2)}, ` +
      `return on day ${fixed(v.return_time_days ?? Number.NaN, 2)}, lunar passage sense ${(v.lunar_passage_sense ?? 0) > 0 ? "+1" : "−1"}.`
    );
  }
  return `${where}: ${outcomeLabel(outcome, run.manifest).toLowerCase()}.`;
}

interface DetailProps {
  run: LoadedGridRun;
  width: number;
  region: CellRegion;
  cells: Map<number, FeasibleCell>;
  candidates: MapCandidate[];
  selectedRunId: string;
  onSelect: (runId: string) => void;
}

function Detail({ run, width, region, cells, candidates, selectedRunId, onSelect }: DetailProps) {
  const { grid } = run;
  const small = width < 640;
  const frame = useMemo<Frame>(
    () => ({ width, height: Math.round(Math.min(440, Math.max(260, width * (small ? 0.8 : 0.42)))), left: 52, right: 12, top: 18, bottom: 42 }),
    [width, small],
  );
  const p = placer(frame, region);
  const [cursor, setCursor] = useState<{ i: number; j: number } | null>(null);

  const start = (): { i: number; j: number } => {
    const c = candidates[0];
    if (c !== undefined) {
      return { i: Math.round(fractionalIndex(grid.phaseAxis, c.phaseDeg)), j: Math.round(fractionalIndex(grid.deltaVAxis, c.deltaVKmS)) };
    }
    return { i: Math.round((region.i0 + region.i1) / 2), j: Math.round((region.j0 + region.j1) / 2) };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>): void => {
    const rect = event.currentTarget.getBoundingClientRect();
    const cell = p.cell(event.clientX - rect.left, event.clientY - rect.top);
    if (cell !== null && (cell.i !== cursor?.i || cell.j !== cursor.j)) setCursor(cell);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const moves: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    const move = moves[event.key];
    if (move === undefined) return;
    event.preventDefault();
    const stride = event.shiftKey ? 5 : 1;
    const from = cursor ?? start();
    setCursor({
      i: Math.min(region.i1, Math.max(region.i0, from.i + move[0] * stride)),
      j: Math.min(region.j1, Math.max(region.j0, from.j + move[1] * stride)),
    });
  };

  return (
    <div>
      <div
        className="relative"
        style={{ width: frame.width, height: frame.height }}
        tabIndex={0}
        role="group"
        aria-roledescription="outcome map"
        aria-label="Zoomed outcome map. Arrow keys move between grid cells (Shift moves five); the cell is described below."
        aria-describedby="map-cell-readout"
        onPointerMove={onPointerMove}
        onFocus={() => { if (cursor === null) setCursor(start()); }}
        onKeyDown={onKeyDown}
      >
        <DetailCanvas grid={grid} frame={frame} region={region} cells={cells} />
        <svg className="pointer-events-none absolute inset-0" width={frame.width} height={frame.height} aria-hidden="true">
          <Axes grid={grid} frame={frame} region={region} />
          {cursor !== null && (
            <rect
              x={p.x(cursor.i - 0.5) - 1}
              y={p.y(cursor.j + 0.5) - 1}
              width={p.cw + 2}
              height={p.ch + 2}
              fill="none"
              stroke="var(--color-accent-strong)"
              strokeWidth={2}
            />
          )}
        </svg>
        {candidates.map((c) => {
          const x = p.x(fractionalIndex(grid.phaseAxis, c.phaseDeg));
          const y = p.y(fractionalIndex(grid.deltaVAxis, c.deltaVKmS));
          const selected = c.runId === selectedRunId;
          return (
            <button
              key={c.runId}
              type="button"
              onClick={() => { onSelect(c.runId); }}
              aria-pressed={selected}
              aria-label={`Show ${c.label} in the viewer (θ ${fixed(c.phaseDeg, 3)}°, Δv ${fixed(c.deltaVKmS, 5)} km/s)`}
              title={`${c.label}: θ ${fixed(c.phaseDeg, 3)}°, Δv ${fixed(c.deltaVKmS, 5)} km/s`}
              className="group absolute flex items-center gap-1.5"
              style={{ left: x - 11, top: y - 11 }}
            >
              <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
                <circle cx="11" cy="11" r="8" fill="none" stroke="var(--color-canvas)" strokeWidth="4" />
                <circle
                  cx="11"
                  cy="11"
                  r="8"
                  fill="none"
                  stroke={selected ? "var(--color-accent-strong)" : "var(--color-canvas-strong)"}
                  strokeWidth="2"
                  className="group-hover:stroke-[var(--color-accent-strong)]"
                />
                <circle cx="11" cy="11" r="2.2" fill={selected ? "var(--color-accent-strong)" : "var(--color-canvas-strong)"} />
              </svg>
              <span
                className={`numeric rounded-sm px-1.5 text-[12px] font-medium ${
                  selected ? "bg-accent text-ink" : "bg-canvas text-canvas-strong group-hover:text-accent-strong"
                }`}
              >
                {c.tag}
              </span>
            </button>
          );
        })}
      </div>
      <p id="map-cell-readout" aria-live="polite" className="numeric min-h-[3.25rem] px-1 pt-3 text-[0.92rem] text-canvas-strong">
        {cursor === null ? (
          <span className="text-canvas-text">Point at a cell, or focus the map and use the arrow keys, to read its outcome.</span>
        ) : (
          describeCell(grid, run, cells, cursor.i, cursor.j)
        )}
      </p>
    </div>
  );
}

function Swatch({ style }: { style: OutcomeStyle }) {
  return (
    <svg viewBox="0 0 16 16" className="size-4 shrink-0" aria-hidden="true">
      <rect width="16" height="16" fill={style.fill} stroke="var(--color-canvas-axis)" strokeWidth="0.75" />
      {style.pattern === "hatch" && <line x1="2" y1="14" x2="14" y2="2" stroke={style.mark} strokeWidth="2" />}
      {style.pattern === "dot" && <circle cx="8" cy="8" r="2.6" fill={style.mark} />}
      {style.pattern === "hollow" && <rect x="4.5" y="4.5" width="7" height="7" fill={style.mark} />}
    </svg>
  );
}

export function OutcomeLegend({ run, cells }: { run: LoadedGridRun; cells: Map<number, FeasibleCell> }) {
  const { grid } = run;
  const counts = useMemo(() => countOutcomes(grid), [grid]);
  const total = grid.codes.length;
  const plus = [...cells.values()].filter((c) => (c.values.lunar_passage_sense ?? 0) > 0).length;
  const feasible = counts.get("feasible") ?? 0;
  const share = (n: number): string => (n === 0 ? "" : `${fixed((100 * n) / total, n / total < 0.01 ? 2 : 1)} %`);
  const rows: { key: string; style: OutcomeStyle; label: string; count: number }[] = [];
  for (const outcome of OUTCOME_ORDER) {
    if (!grid.outcomes.includes(outcome)) continue;
    const n = counts.get(outcome) ?? 0;
    if (outcome === "feasible") {
      rows.push({ key: "feasible-minus", style: OUTCOME_STYLE.feasible, label: "Feasible, lunar passage sense −1", count: feasible - plus });
      rows.push({ key: "feasible-plus", style: FEASIBLE_PLUS_STYLE, label: "Feasible, lunar passage sense +1", count: plus });
    } else {
      rows.push({ key: outcome, style: OUTCOME_STYLE[outcome], label: outcomeLabel(outcome, run.manifest), count: n });
    }
  }
  return (
    <table className="w-full border-collapse text-[0.92rem]">
      <caption className="sr-only">Grid cells per outcome, of {total.toLocaleString("en-US")} departures</caption>
      <thead className="sr-only">
        <tr>
          <th scope="col">Outcome</th>
          <th scope="col">Cells</th>
          <th scope="col">Share</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.key} className={`border-t border-canvas-grid ${row.count === 0 ? "text-canvas-text" : ""}`}>
            <th scope="row" className="py-1.5 pr-3 text-left font-normal">
              <span className="flex items-center gap-2.5">
                <Swatch style={row.style} />
                {row.label}
              </span>
            </th>
            <td className="numeric py-1.5 pr-3 text-right whitespace-nowrap">{row.count.toLocaleString("en-US")}</td>
            <td className="numeric py-1.5 text-right whitespace-nowrap text-canvas-text">{share(row.count)}</td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr className="border-t border-canvas-axis">
          <th scope="row" className="py-1.5 pr-3 text-left font-medium">
            All departures
          </th>
          <td className="numeric py-1.5 pr-3 text-right font-medium">{total.toLocaleString("en-US")}</td>
          <td />
        </tr>
      </tfoot>
    </table>
  );
}

interface Props {
  run: LoadedGridRun;
  candidates: MapCandidate[];
  selectedRunId: string;
  onSelect: (runId: string) => void;
}

/** The S1 outcome grid: full-range overview, zoomed raster with the candidates marked, and a legend with counts. */
export function FeasibilityMap({ run, candidates, selectedRunId, onSelect }: Props) {
  const { grid } = run;
  const [ref, width] = useElementWidth<HTMLDivElement>(820);
  const detail = useMemo(() => regionFor(grid, DETAIL_PHASE_DEG, DETAIL_DELTA_V_KM_S), [grid]);
  const cells = useMemo(() => feasibleCells(grid), [grid]);
  const thetaLo = axisValue(grid.phaseAxis, detail.i0);
  const thetaHi = axisValue(grid.phaseAxis, detail.i1);
  const dvLo = axisValue(grid.deltaVAxis, detail.j0);
  const dvHi = axisValue(grid.deltaVAxis, detail.j1);
  const fullTheta = [axisValue(grid.phaseAxis, 0), axisValue(grid.phaseAxis, grid.phaseAxis.count - 1)];
  const fullDv = [axisValue(grid.deltaVAxis, 0), axisValue(grid.deltaVAxis, grid.deltaVAxis.count - 1)];

  return (
    <div className="grid gap-8 lg:grid-cols-12 lg:gap-10">
      <figure className="on-canvas min-w-0 rounded-sm bg-canvas p-3 text-canvas-strong sm:p-5 lg:col-span-8">
        <div ref={ref} className="min-w-0">
          <figcaption className="mb-2 text-sm text-canvas-text">
            Whole grid: θ {fixed(fullTheta[0] ?? 0, 0)}–{fixed(fullTheta[1] ?? 0, 1)}°, Δv {fixed(fullDv[0] ?? 0, 2)}–{fixed(fullDv[1] ?? 0, 2)} km/s,{" "}
            {grid.phaseAxis.count} × {grid.deltaVAxis.count} cells. The outlined box is enlarged below.
          </figcaption>
          <Overview grid={grid} width={width} detail={detail} />
          <p className="mt-5 mb-2 text-sm text-canvas-text">
            Enlarged: θ {fixed(thetaLo, 0)}–{fixed(thetaHi, 0)}°, Δv {fixed(dvLo, 2)}–{fixed(dvHi, 2)} km/s, one rectangle per departure (
            {fixed(grid.phaseAxis.step, 1)}° × {fixed(grid.deltaVAxis.step * 1000, 0)} m/s). Rings mark the validated candidates; select one to
            show it in the viewer.
          </p>
          <Detail run={run} width={width} region={detail} cells={cells} candidates={candidates} selectedRunId={selectedRunId} onSelect={onSelect} />
        </div>
      </figure>
      <div className="on-canvas self-start rounded-sm bg-canvas p-4 text-canvas-strong sm:p-5 lg:col-span-4">
        <h4 className="mb-3 font-sans text-sm font-medium text-canvas-text">Outcome of each departure</h4>
        <OutcomeLegend run={run} cells={cells} />
        <p className="mt-4 text-sm text-canvas-text">
          Where one overview pixel covers several cells it shows the outcome listed first here, so isolated feasible or collision
          cells stay visible. Cell patterns, not only colors, separate the rare outcomes in the enlarged map.
        </p>
      </div>
    </div>
  );
}
