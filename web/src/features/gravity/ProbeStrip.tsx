import { useEffect, useMemo, useRef } from "react";
import { fixed } from "../../lib/format";
import { FREE_RETURN_OUTCOMES, type FreeReturnOutcome, type LoadedTableRun, type SampleTable } from "../../lib/data/types";
import { useElementWidth } from "../../lib/useElementWidth";
import { niceStep, ticks } from "../../lib/viewer/transform";
import { OUTCOME_ORDER, OUTCOME_STYLE } from "./outcomes";

export interface ProbeRow {
  phaseDeg: number;
  deltaV: Float64Array;
  outcomes: FreeReturnOutcome[];
  /** Contiguous feasible runs: first and last delta-v (km/s) and sample count. */
  feasibleRuns: { first: number; last: number; count: number }[];
}

/** Group a probe table (phase_deg, delta_v_km_s, outcome columns) by phase, in delta-v order. */
export function probeRows(table: SampleTable): ProbeRow[] {
  const col = (name: string): number => {
    const k = table.columns.findIndex((c) => c.name === name);
    if (k < 0) throw new Error(`The probe table has no ${name} column.`);
    return k;
  };
  const cPhase = col("phase_deg");
  const cDv = col("delta_v_km_s");
  const cOutcome = col("outcome");
  const groups = new Map<number, { dv: number; outcome: FreeReturnOutcome }[]>();
  for (const row of table.rows) {
    const phase = row[cPhase];
    const dv = row[cDv];
    const outcome = row[cOutcome];
    if (typeof phase !== "number" || typeof dv !== "number" || typeof outcome !== "string") throw new Error("The probe table has an incomplete row.");
    if (!(FREE_RETURN_OUTCOMES as readonly string[]).includes(outcome)) throw new Error(`The probe table has an unknown outcome "${outcome}".`);
    const list = groups.get(phase) ?? [];
    list.push({ dv, outcome: outcome as FreeReturnOutcome });
    groups.set(phase, list);
  }
  return [...groups.entries()].map(([phaseDeg, list]) => {
    list.sort((a, b) => a.dv - b.dv);
    const feasibleRuns: ProbeRow["feasibleRuns"] = [];
    let current: { first: number; last: number; count: number } | null = null;
    for (const s of list) {
      if (s.outcome === "feasible") {
        if (current === null) {
          current = { first: s.dv, last: s.dv, count: 1 };
        } else {
          current.last = s.dv;
          current.count += 1;
        }
      } else if (current !== null) {
        feasibleRuns.push(current);
        current = null;
      }
    }
    if (current !== null) feasibleRuns.push(current);
    return { phaseDeg, deltaV: Float64Array.from(list.map((s) => s.dv)), outcomes: list.map((s) => s.outcome), feasibleRuns };
  });
}

/** Band width bounds in m/s for n contiguous feasible samples at step d: [(n - 1) d, (n + 1) d]. */
export function bandWidthMs(count: number, stepKmS: number): [number, number] {
  return [(count - 1) * stepKmS * 1000, (count + 1) * stepKmS * 1000];
}

const ROW_H = 26;
const ROW_GAP = 46;

export function ProbeStrip({ run }: { run: LoadedTableRun }) {
  const rows = useMemo(() => probeRows(run.table), [run.table]);
  const step = run.manifest.metrics.grid_delta_v_step_km_s?.value ?? null;
  const [ref, width] = useElementWidth<HTMLDivElement>(820);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const left = 64;
  const right = 14;
  const top = 34;
  const height = top + rows.length * ROW_H + (rows.length - 1) * ROW_GAP + 40;
  const dvMin = Math.min(...rows.map((r) => r.deltaV[0] ?? 0));
  const dvMax = Math.max(...rows.map((r) => r.deltaV[r.deltaV.length - 1] ?? 0));
  const plot = width - left - right;
  const xOf = (dv: number): number => left + ((dv - dvMin) / (dvMax - dvMin)) * plot;
  const rowTop = (k: number): number => top + k * (ROW_H + ROW_GAP);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (ctx === null) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const pw = Math.round(plot * dpr);
    const priority = (o: FreeReturnOutcome): number => OUTCOME_ORDER.indexOf(o);
    rows.forEach((row, k) => {
      const n = row.outcomes.length;
      const y0 = Math.round(rowTop(k) * dpr);
      const h = Math.round(ROW_H * dpr);
      for (let px = 0; px < pw; px++) {
        // Samples whose delta-v falls in this pixel column; a feasible sample wins (thin windows stay visible).
        const a = Math.floor((px * n) / pw);
        const b = Math.max(a, Math.ceil(((px + 1) * n) / pw) - 1);
        let best = row.outcomes[a] ?? "no_return_within_t_max";
        for (let s = a; s <= b; s++) {
          const o = row.outcomes[s];
          if (o !== undefined && priority(o) < priority(best)) best = o;
        }
        ctx.fillStyle = OUTCOME_STYLE[best].fill;
        ctx.fillRect(Math.round(left * dpr) + px, y0, 1, h);
      }
    });
  }, [rows, width, height, plot]);

  const tickStep = niceStep(dvMax - dvMin, Math.max(3, Math.round(plot / 95)));
  return (
    <div ref={ref} className="min-w-0">
      <div className="relative" style={{ width, height }}>
        <canvas ref={canvasRef} style={{ width, height }} className="block" aria-hidden="true" />
        <svg className="absolute inset-0" width={width} height={height} role="img" aria-labelledby="probe-title probe-desc">
          <title id="probe-title">Fine Δv probe</title>
          <desc id="probe-desc">
            {rows
              .map(
                (r) =>
                  `At θ = ${fixed(r.phaseDeg, 0)}°, ${r.outcomes.length} departures from ${fixed(r.deltaV[0] ?? 0, 3)} to ${fixed(r.deltaV[r.deltaV.length - 1] ?? 0, 3)} km/s; ` +
                  (r.feasibleRuns.length === 0
                    ? "none is feasible."
                    : r.feasibleRuns.map((f) => `feasible from ${fixed(f.first, 5)} to ${fixed(f.last, 5)} km/s (${f.count} samples)`).join("; ") + "."),
              )
              .join(" ")}
          </desc>
          {rows.map((row, k) => {
            const y = rowTop(k);
            return (
              <g key={row.phaseDeg} className="text-[12px]">
                <text x={left - 8} y={y + ROW_H / 2 + 4} textAnchor="end" className="numeric fill-canvas-strong">
                  θ = {fixed(row.phaseDeg, 0)}°
                </text>
                <rect x={left} y={y} width={plot} height={ROW_H} fill="none" stroke="var(--color-canvas-axis)" />
                {row.feasibleRuns.map((f) => {
                  const x0 = xOf(f.first);
                  const x1 = Math.max(x0 + 2, xOf(f.last));
                  const mid = (x0 + x1) / 2;
                  const widths = step === null ? null : bandWidthMs(f.count, step);
                  const anchor = mid > left + plot * 0.72 ? "end" : mid < left + plot * 0.28 ? "start" : "middle";
                  const tx = anchor === "end" ? x1 : anchor === "start" ? x0 : mid;
                  return (
                    <g key={f.first}>
                      <path d={`M${x0} ${y - 4}v-5h${x1 - x0}v5`} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={1.2} />
                      <text x={tx} y={y - 14} textAnchor={anchor} className="numeric fill-canvas-strong">
                        {widths === null
                          ? `${f.count} feasible samples`
                          : `${fixed(widths[0], 2)}–${fixed(widths[1], 2)} m/s wide (${f.count} samples)`}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
          <g className="text-[11px]">
            {ticks(dvMin, dvMax, tickStep).map((dv) => (
              <g key={dv}>
                <line x1={xOf(dv)} x2={xOf(dv)} y1={height - 36} y2={height - 32} stroke="var(--color-canvas-axis)" />
                <text x={xOf(dv)} y={height - 20} textAnchor="middle" className="numeric fill-canvas-text">
                  {fixed(dv, 2)}
                </text>
              </g>
            ))}
            <text x={left + plot} y={height - 4} textAnchor="end" className="fill-canvas-text">
              Δv (km/s)
            </text>
          </g>
        </svg>
      </div>
    </div>
  );
}
