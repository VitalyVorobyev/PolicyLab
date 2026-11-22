import { useState } from "react";
import type { LibraryResults } from "../../lib/data/library";
import { fixed } from "../../lib/format";
import { armInfo } from "./armMeta";
import { placeRightLabels, type LabelPos } from "./chartUtils";

export type CurveView = "stationary_final" | "change_at_split" | "change_final";

const CORE_SWEEP_ARMS = ["example_random", "example_pedagogical", "rule", "rule_confidence", "rule_hedged"] as const;
const PLACEBO_ARMS = ["placebo_random", "placebo_shuffled"] as const;

const VIEW_LABELS: Record<CurveView, { label: string; subtitle: string }> = {
  stationary_final: {
    label: "Stationary gen 9",
    subtitle: "Mean held-out regret after 10 cumulative stationary generations (N = 200).",
  },
  change_at_split: {
    label: "Rule-change shock (gen 5)",
    subtitle: "Immediate post-change held-out regret in the generation when latent terrain costs invert.",
  },
  change_final: {
    label: "Post-change recovery (gen 9)",
    subtitle: "Held-out regret 4 generations after the unannounced latent-rule shift.",
  },
};

function pointValues(
  view: CurveView,
  point: {
    stationary_final_mean: number;
    stationary_final_se: number;
    change_at_split_mean: number;
    change_at_split_se: number;
    change_final_mean: number;
    change_final_se: number;
  },
): { mean: number; se: number } {
  if (view === "stationary_final") return { mean: point.stationary_final_mean, se: point.stationary_final_se };
  if (view === "change_at_split") return { mean: point.change_at_split_mean, se: point.change_at_split_se };
  return { mean: point.change_final_mean, se: point.change_final_se };
}

function markerShape(marker: string, cx: number, cy: number, color: string) {
  if (marker === "square") return <rect x={cx - 3.5} y={cy - 3.5} width="7" height="7" fill={color} />;
  if (marker === "diamond") return <polygon points={`${cx},${cy - 4.5} ${cx + 4.5},${cy} ${cx},${cy + 4.5} ${cx - 4.5},${cy}`} fill={color} />;
  if (marker === "triangle") return <polygon points={`${cx},${cy - 4.5} ${cx + 4.2},${cy + 3.5} ${cx - 4.2},${cy + 3.5}`} fill={color} />;
  if (marker === "cross") {
    return (
      <path
        d={`M${cx - 3.2} ${cy - 3.2}L${cx + 3.2} ${cy + 3.2}M${cx + 3.2} ${cy - 3.2}L${cx - 3.2} ${cy + 3.2}`}
        stroke={color}
        strokeWidth="1.6"
      />
    );
  }
  return <circle cx={cx} cy={cy} r="3.6" fill={color} />;
}

/**
 * Interactive SVG budget-sweep chart showing held-out regret across 0..60 inherited bits,
 * paired with clickable budget selector buttons synchronized with `TrajectoryChart`.
 */
export function BudgetCurveChart({
  results,
  selectedBudget,
  onSelectBudget,
}: {
  results: LibraryResults;
  selectedBudget: number;
  onSelectBudget: (bits: number) => void;
}) {
  const [view, setView] = useState<CurveView>("stationary_final");
  const [showPlacebos, setShowPlacebos] = useState(false);

  const budgets = results.budget_curves.budgets;
  const displayedArms = showPlacebos ? [...CORE_SWEEP_ARMS, ...PLACEBO_ARMS] : [...CORE_SWEEP_ARMS];
  const baselineNone = results.budget_curves.baselines.none;
  const baselineUnlim = results.budget_curves.baselines.unlimited;
  const noneYVal = baselineNone !== undefined ? pointValues(view, baselineNone).mean : null;
  const unlimYVal = baselineUnlim !== undefined ? pointValues(view, baselineUnlim).mean : null;

  const yCandidates: number[] = [noneYVal ?? 0, unlimYVal ?? 0];
  for (const arm of displayedArms) {
    for (const p of results.budget_curves.arms[arm] ?? []) {
      const v = pointValues(view, p);
      yCandidates.push(v.mean + v.se);
    }
  }
  const maxVal = Math.max(...yCandidates, 10);
  const yMax = Math.ceil(maxVal / 4) * 4;
  const yTicks = Array.from({ length: 5 }, (_, i) => (yMax / 4) * i);

  const W = 740;
  const H = 360;
  const padL = 56;
  const padR = 136;
  const padT = 26;
  const padB = 46;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xPos = (bits: number): number => padL + (bits / 60) * plotW;
  const yPos = (val: number): number => padT + plotH - (Math.min(yMax, Math.max(0, val)) / yMax) * plotH;

  const rawLabels: LabelPos[] = [];
  if (noneYVal !== null && noneYVal <= yMax) {
    rawLabels.push({ id: "none", rawY: yPos(noneYVal), text: `None (${fixed(noneYVal, 1)})`, color: "#94a3b8" });
  }
  if (unlimYVal !== null && unlimYVal <= yMax) {
    rawLabels.push({ id: "unlimited", rawY: yPos(unlimYVal), text: `Unlimited (${fixed(unlimYVal, 1)})`, color: "#38bdf8" });
  }
  for (const arm of displayedArms) {
    const pts = results.budget_curves.arms[arm] ?? [];
    const last = pts[pts.length - 1];
    if (last !== undefined) {
      const val = pointValues(view, last).mean;
      const meta = armInfo(arm);
      rawLabels.push({ id: arm, rawY: yPos(val), text: `${meta.short} (${fixed(val, 1)})`, color: meta.color });
    }
  }
  const placedLabels = placeRightLabels(rawLabels, padT + 8, padT + plotH - 4, 13);

  const hoverPoint = (arm: string): { mean: number; se: number } | undefined => {
    const p = (results.budget_curves.arms[arm] ?? []).find((x) => x.budget_bits === selectedBudget);
    return p === undefined ? undefined : pointValues(view, p);
  };

  return (
    <div className="rounded-sm bg-canvas p-5 text-canvas-strong sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-canvas-grid pb-4">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Budget curve scenario">
          {(Object.keys(VIEW_LABELS) as CurveView[]).map((k) => {
            const active = view === k;
            return (
              <button
                key={k}
                type="button"
                aria-pressed={active}
                onClick={() => { setView(k); }}
                className={`rounded-sm border px-3 py-1.5 text-xs font-medium transition-colors ${
                  active
                    ? "border-accent bg-accent text-ink"
                    : "border-canvas-axis bg-canvas-raised text-canvas-text hover:text-canvas-strong"
                }`}
              >
                {VIEW_LABELS[k].label}
              </button>
            );
          })}
        </div>
        <label className="inline-flex cursor-pointer items-center gap-2 text-xs text-canvas-text">
          <input
            type="checkbox"
            checked={showPlacebos}
            onChange={(e) => { setShowPlacebos(e.target.checked); }}
            className="accent-accent"
          />
          <span>Show placebo controls (random / shuffled)</span>
        </label>
      </div>

      <p className="mt-3 text-xs text-canvas-text">{VIEW_LABELS[view].subtitle}</p>

      <div className="mt-4 overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[36rem]" role="img" aria-label={`Held-out regret across bit budgets for ${VIEW_LABELS[view].label}`}>
          {yTicks.map((t) => {
            const py = yPos(t);
            return (
              <g key={t}>
                <line x1={padL} x2={W - padR} y1={py} y2={py} stroke="var(--color-canvas-grid)" strokeWidth="1" />
                <text x={padL - 8} y={py + 4} textAnchor="end" fill="var(--color-canvas-text)" fontSize="11">
                  {t}
                </text>
              </g>
            );
          })}

          {budgets.map((b) => {
            const px = xPos(b);
            const active = b === selectedBudget;
            return (
              <g key={b}>
                <line
                  x1={px}
                  x2={px}
                  y1={padT}
                  y2={padT + plotH}
                  stroke={active ? "var(--color-accent)" : "var(--color-canvas-grid)"}
                  strokeWidth={active ? 1.5 : 1}
                  strokeDasharray={active ? "3 3" : undefined}
                />
                <text
                  x={px}
                  y={H - padB + 18}
                  textAnchor="middle"
                  fill={active ? "var(--color-accent-strong)" : "var(--color-canvas-text)"}
                  fontSize="11"
                  fontWeight={active ? 600 : 400}
                >
                  {b}
                </text>
              </g>
            );
          })}

          {/* Reference markers: H=16 bits latent entropy and 19 bits dense capacity */}
          <line x1={xPos(16)} x2={xPos(16)} y1={padT} y2={padT + plotH} stroke="var(--color-canvas-axis)" strokeDasharray="2 3" />
          <text x={xPos(16)} y={padT - 8} textAnchor="middle" fill="var(--color-canvas-text)" fontSize="10">
            H=16
          </text>
          <line x1={xPos(19)} x2={xPos(19)} y1={padT} y2={padT + plotH} stroke="var(--color-accent)" strokeOpacity="0.65" strokeDasharray="4 2" />
          <text x={xPos(19) + 4} y={padT - 8} textAnchor="start" fill="var(--color-accent-strong)" fontSize="10">
            19b dense
          </text>

          {/* Horizontal baseline lines */}
          {noneYVal !== null && noneYVal <= yMax && (
            <line x1={padL} x2={W - padR} y1={yPos(noneYVal)} y2={yPos(noneYVal)} stroke="#94a3b8" strokeDasharray="5 4" strokeWidth="1.2" />
          )}
          {unlimYVal !== null && unlimYVal <= yMax && (
            <line x1={padL} x2={W - padR} y1={yPos(unlimYVal)} y2={yPos(unlimYVal)} stroke="#38bdf8" strokeDasharray="7 3" strokeWidth="1.2" />
          )}

          {/* Swept arm lines + SE error bars + markers */}
          {displayedArms.map((arm) => {
            const pts = results.budget_curves.arms[arm] ?? [];
            const meta = armInfo(arm);
            const pathD = pts
              .map((p, idx) => {
                const { mean } = pointValues(view, p);
                return `${idx === 0 ? "M" : "L"}${xPos(p.budget_bits).toFixed(1)},${yPos(mean).toFixed(1)}`;
              })
              .join(" ");
            return (
              <g key={arm}>
                <path d={pathD} fill="none" stroke={meta.color} strokeWidth="2.1" strokeDasharray={meta.dash || undefined} />
                {pts.map((p) => {
                  const { mean, se } = pointValues(view, p);
                  const cx = xPos(p.budget_bits);
                  const cy = yPos(mean);
                  const top = yPos(mean + se);
                  const bot = yPos(mean - se);
                  return (
                    <g key={p.budget_bits}>
                      <line x1={cx} x2={cx} y1={top} y2={bot} stroke={meta.color} strokeOpacity="0.45" strokeWidth="1.2" />
                      {markerShape(meta.marker, cx, cy, meta.color)}
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* Deconflicted endpoint labels */}
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
            Inherited bit budget B (bits / generation)
          </text>
          <text x={14} y={padT + plotH / 2} textAnchor="middle" fill="var(--color-canvas-text)" fontSize="11.5" transform={`rotate(-90 14 ${padT + plotH / 2})`}>
            Held-out regret (±1 SE)
          </text>
        </svg>
      </div>

      {/* Interactive budget selector */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4 border-t border-canvas-grid pt-4">
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Select bit budget B">
          <span className="mr-1 text-xs text-canvas-text">Inspect budget B:</span>
          {budgets.map((b) => {
            const active = b === selectedBudget;
            return (
              <button
                key={b}
                type="button"
                aria-pressed={active}
                onClick={() => { onSelectBudget(b); }}
                className={`numeric rounded-sm border px-2.5 py-1 font-mono text-xs ${
                  active
                    ? "border-accent bg-accent font-semibold text-ink"
                    : "border-canvas-axis bg-canvas-raised text-canvas-text hover:text-canvas-strong"
                }`}
              >
                {b}b
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
          {CORE_SWEEP_ARMS.map((arm) => {
            const val = hoverPoint(arm);
            if (val === undefined) return null;
            const meta = armInfo(arm);
            return (
              <span key={arm} className="numeric" style={{ color: meta.color }}>
                {meta.short}: {fixed(val.mean, 2)}±{fixed(val.se, 2)}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
