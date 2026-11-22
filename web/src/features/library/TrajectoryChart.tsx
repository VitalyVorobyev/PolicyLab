import { useState } from "react";
import type { LibraryArmSummary, LibraryResults } from "../../lib/data/library";
import { fixed } from "../../lib/format";
import { armInfo } from "./armMeta";
import { placeRightLabels, type LabelPos } from "./chartUtils";

type MetricKey = "heldout" | "entropy" | "accuracy";

const TRAJECTORY_ARMS: readonly { arm: string; budgetOverride?: number }[] = [
  { arm: "none", budgetOverride: 0 },
  { arm: "unlimited", budgetOverride: 0 },
  { arm: "example_pedagogical" },
  { arm: "rule" },
  { arm: "rule_confidence" },
  { arm: "rule_hedged" },
];

const METRICS: Record<MetricKey, { label: string; yMax: number; yLabel: string; scale: number }> = {
  heldout: { label: "Held-out regret", yMax: 34, yLabel: "Mean held-out regret", scale: 1 },
  entropy: { label: "Posterior entropy (bits)", yMax: 16, yLabel: "Belief entropy (bits)", scale: 1 },
  accuracy: { label: "MAP accuracy (%)", yMax: 100, yLabel: "Correct MAP terrains (%)", scale: 100 },
};

function metricSeries(summary: LibraryArmSummary, metric: MetricKey): number[] {
  if (metric === "entropy") return summary.entropy_bits_mean_by_gen;
  if (metric === "accuracy") return summary.map_accuracy_mean_by_gen.map((v) => v * 100);
  return summary.heldout_regret_mean_by_gen;
}

/** Generational learning and shock-recovery trajectories (gen 0..9) at the chosen bit budget. */
export function TrajectoryChart({ results, budgetBits }: { results: LibraryResults; budgetBits: number }) {
  const [scenario, setScenario] = useState<"stationary" | "rule_change">("rule_change");
  const [metric, setMetric] = useState<MetricKey>("heldout");

  const summaries = scenario === "stationary" ? results.stationary : results.rule_change;
  const nGen = results.config.n_generations;
  const gens = Array.from({ length: nGen }, (_, g) => g);
  const spec = METRICS[metric];

  let dynamicYMax = spec.yMax;
  if (metric === "heldout") {
    const peak = Math.max(
      10,
      ...TRAJECTORY_ARMS.flatMap(({ arm, budgetOverride }) => {
        const s = summaries[`${arm}_b${budgetOverride ?? budgetBits}`];
        return s !== undefined ? s.heldout_regret_mean_by_gen : [0];
      }),
    );
    dynamicYMax = Math.ceil(peak / 5) * 5;
  }

  const W = 740;
  const H = 330;
  const padL = 56;
  const padR = 136;
  const padT = 26;
  const padB = 46;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xPos = (g: number): number => padL + (g / Math.max(1, nGen - 1)) * plotW;
  const yPos = (v: number): number => padT + plotH - (Math.min(dynamicYMax, Math.max(0, v)) / dynamicYMax) * plotH;
  const yTicks = Array.from({ length: 5 }, (_, i) => (dynamicYMax / 4) * i);
  const rawLabels: LabelPos[] = [];
  for (const { arm, budgetOverride } of TRAJECTORY_ARMS) {
    const key = `${arm}_b${budgetOverride ?? budgetBits}`;
    const s = summaries[key];
    if (s === undefined) continue;
    const series = metricSeries(s, metric);
    const last = series[series.length - 1] ?? 0;
    const meta = armInfo(arm);
    rawLabels.push({
      id: key,
      rawY: yPos(last),
      text: `${meta.short} (${fixed(last, metric === "accuracy" ? 0 : 1)}${metric === "accuracy" ? "%" : ""})`,
      color: meta.color,
    });
  }
  const placedLabels = placeRightLabels(rawLabels, padT + 8, padT + plotH - 4, 13);

  return (
    <div className="rounded-sm bg-canvas p-5 text-canvas-strong sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-canvas-grid pb-4">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Trajectory scenario">
          <button
            type="button"
            aria-pressed={scenario === "stationary"}
            onClick={() => { setScenario("stationary"); }}
            className={`rounded-sm border px-3 py-1.5 text-xs font-medium ${
              scenario === "stationary"
                ? "border-accent bg-accent text-ink"
                : "border-canvas-axis bg-canvas-raised text-canvas-text hover:text-canvas-strong"
            }`}
          >
            Stationary worlds
          </button>
          <button
            type="button"
            aria-pressed={scenario === "rule_change"}
            onClick={() => { setScenario("rule_change"); }}
            className={`rounded-sm border px-3 py-1.5 text-xs font-medium ${
              scenario === "rule_change"
                ? "border-accent bg-accent text-ink"
                : "border-canvas-axis bg-canvas-raised text-canvas-text hover:text-canvas-strong"
            }`}
          >
            Rule change at gen {results.config.change_generation}
          </button>
        </div>

        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Trajectory metric">
          {(Object.keys(METRICS) as MetricKey[]).map((k) => {
            const active = metric === k;
            return (
              <button
                key={k}
                type="button"
                aria-pressed={active}
                onClick={() => { setMetric(k); }}
                className={`rounded-sm border px-2.5 py-1 text-xs ${
                  active
                    ? "border-canvas-strong bg-canvas-raised font-medium text-canvas-strong"
                    : "border-canvas-grid text-canvas-text hover:text-canvas-strong"
                }`}
              >
                {METRICS[k].label}
              </button>
            );
          })}
        </div>
      </div>

      <p className="mt-3 text-xs text-canvas-text">
        Comparing generational trajectories at bit budget <span className="font-mono text-accent-strong">B = {budgetBits} bits</span> (None &amp; Unlimited shown at B=0).
      </p>

      <div className="mt-4 overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[36rem]" role="img" aria-label={`Generational trajectory chart for B = ${budgetBits} bits`}>
          {yTicks.map((t) => {
            const py = yPos(t);
            return (
              <g key={t}>
                <line x1={padL} x2={W - padR} y1={py} y2={py} stroke="var(--color-canvas-grid)" />
                <text x={padL - 8} y={py + 4} textAnchor="end" fill="var(--color-canvas-text)" fontSize="11">
                  {Math.round(t * 10) / 10}
                </text>
              </g>
            );
          })}

          {gens.map((g) => (
            <g key={g}>
              <line x1={xPos(g)} x2={xPos(g)} y1={padT} y2={padT + plotH} stroke="var(--color-canvas-grid)" />
              <text x={xPos(g)} y={H - padB + 18} textAnchor="middle" fill="var(--color-canvas-text)" fontSize="11">
                {g}
              </text>
            </g>
          ))}

          {scenario === "rule_change" && (
            <g>
              <line
                x1={xPos(results.config.change_generation)}
                x2={xPos(results.config.change_generation)}
                y1={padT}
                y2={padT + plotH}
                stroke="var(--color-fan-fail)"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
              <text x={xPos(results.config.change_generation) + 5} y={padT - 8} fill="var(--color-fan-fail)" fontSize="10.5">
                rule shift
              </text>
            </g>
          )}

          {TRAJECTORY_ARMS.map(({ arm, budgetOverride }) => {
            const key = `${arm}_b${budgetOverride ?? budgetBits}`;
            const s = summaries[key];
            if (s === undefined) return null;
            const series = metricSeries(s, metric);
            const meta = armInfo(arm);
            const d = series.map((v, g) => `${g === 0 ? "M" : "L"}${xPos(g).toFixed(1)},${yPos(v).toFixed(1)}`).join(" ");
            return (
              <g key={key}>
                <path d={d} fill="none" stroke={meta.color} strokeWidth="2.1" strokeDasharray={meta.dash || undefined} />
                {series.map((v, g) => (
                  <circle key={g} cx={xPos(g)} cy={yPos(v)} r="3" fill={meta.color} />
                ))}
              </g>
            );
          })}

          {placedLabels.map((lbl) => (
            <g key={lbl.id}>
              {Math.abs(lbl.y - lbl.rawY) > 2 && (
                <line x1={W - padR} x2={W - padR + 5} y1={lbl.rawY} y2={lbl.y} stroke={lbl.color} strokeOpacity="0.55" strokeWidth="1" />
              )}
              <text x={W - padR + 7} y={lbl.y + 3.5} fill={lbl.color} fontSize="10.5" fontWeight="500">
                {lbl.text}
              </text>
            </g>
          ))}

          <text x={padL + plotW / 2} y={H - 8} textAnchor="middle" fill="var(--color-canvas-text)" fontSize="11.5">
            Generation (0..{nGen - 1})
          </text>
          <text x={14} y={padT + plotH / 2} textAnchor="middle" fill="var(--color-canvas-text)" fontSize="11.5" transform={`rotate(-90 14 ${padT + plotH / 2})`}>
            {spec.yLabel}
          </text>
        </svg>
      </div>
    </div>
  );
}
