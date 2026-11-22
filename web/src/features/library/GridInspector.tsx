import { useState } from "react";
import type { LibrarySamples } from "../../lib/data/library";
import { fixed } from "../../lib/format";
import { armInfo } from "./armMeta";

const SAMPLE_COMPARE_ARMS = [
  "rule",
  "rule_hedged",
  "rule_confidence",
  "example_pedagogical",
  "example_random",
  "unlimited",
  "none",
] as const;

/** Cell fills on the dark navy canvas graded gently by true latent cost level (0..3 -> 1..16). */
const LEVEL_CELL_FILL = ["#14332d", "#182d49", "#3c2f18", "#4b1f27"] as const;
const LEVEL_TEXT_COLOR = ["#6ee7b7", "#93c5fd", "#fcd34d", "#fca5a5"] as const;

function polylinePoints(path: readonly [number, number][], cellPx: number): string {
  return path.map(([r, c]) => `${(c + 0.5) * cellPx},${(r + 0.5) * cellPx}`).join(" ");
}

/**
 * Interactive 9×9 Voronoi terrain inspector comparing inherited record contents and
 * shortest planned paths (`samples.json`) for sample lineages 0, 1, and 2.
 */
export function GridInspector({ samples }: { samples: LibrarySamples }) {
  const [scenario, setScenario] = useState<"stationary" | "rule_change">("rule_change");
  const [lineageIdx, setLineageIdx] = useState<number>(samples.sample_indices[0] ?? 0);
  const [budgetBits, setBudgetBits] = useState<number>(samples.sample_budgets.includes(20) ? 20 : (samples.sample_budgets[0] ?? 20));
  const [gen, setGen] = useState<number>(samples.config.change_generation);
  const [arm, setArm] = useState<string>("rule");

  const lineages = scenario === "stationary" ? samples.stationary : samples.rule_change;
  const lineage = lineages.find((l) => l.lineage_index === lineageIdx) ?? lineages[0];
  if (lineage === undefined) return null;

  const layoutEntry =
    lineage.layouts_by_key_gen.find((l) => l.generation === gen) ?? lineage.layouts_by_key_gen[0];
  if (layoutEntry === undefined) return null;

  const keyGen = layoutEntry.generation;
  const specKey = arm === "none" || arm === "unlimited" ? `${arm}_b0` : `${arm}_b${budgetBits}`;
  const selectedTrace = lineage.arms[specKey];
  const oracleTrace = lineage.arms.oracle_b0;

  const isPostShift = scenario === "rule_change" && keyGen >= samples.config.change_generation;
  const activeLevels = isPostShift ? lineage.rule_after : lineage.rule_initial;
  const levelCosts = samples.config.level_costs;

  const grid = layoutEntry.heldout_preview.terrains;
  const N = grid.length;
  const cellPx = 38;
  const sizePx = N * cellPx;

  const oraclePath = oracleTrace?.planned_paths_by_key_gen[String(keyGen)] ?? [];
  const selectedPath = selectedTrace?.planned_paths_by_key_gen[String(keyGen)] ?? [];
  const genMetrics = selectedTrace?.generations.find((g) => g.generation === keyGen);
  const armMeta = armInfo(arm);

  return (
    <div className="rounded-sm bg-canvas p-5 text-canvas-strong sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-canvas-grid pb-4">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Grid inspector scenario">
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
            Stationary lineage
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
            Rule-change lineage (shift at gen {samples.config.change_generation})
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Sample lineage seed index">
          <span className="text-xs text-canvas-text">Lineage:</span>
          {lineages.map((l) => {
            const active = l.lineage_index === lineage.lineage_index;
            return (
              <button
                key={l.lineage_index}
                type="button"
                aria-pressed={active}
                onClick={() => { setLineageIdx(l.lineage_index); }}
                className={`numeric rounded-sm border px-2.5 py-1 font-mono text-xs ${
                  active
                    ? "border-accent bg-accent font-medium text-ink"
                    : "border-canvas-axis bg-canvas-raised text-canvas-text hover:text-canvas-strong"
                }`}
              >
                #{l.lineage_index} (seed {l.seed})
              </button>
            );
          })}
        </div>
      </div>

      {/* Generation, Budget, and Arm selectors */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Key generation">
          <span className="mr-1 text-xs text-canvas-text">Generation:</span>
          {samples.key_generations.map((g) => {
            const active = g === keyGen;
            return (
              <button
                key={g}
                type="button"
                aria-pressed={active}
                onClick={() => { setGen(g); }}
                className={`numeric rounded-sm border px-2.5 py-1 font-mono text-xs ${
                  active
                    ? "border-canvas-strong bg-canvas-raised font-medium text-canvas-strong"
                    : "border-canvas-grid text-canvas-text hover:text-canvas-strong"
                }`}
              >
                gen {g}
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Sample bit budget">
          <span className="mr-1 text-xs text-canvas-text">Budget:</span>
          {samples.sample_budgets.map((b) => {
            const active = b === budgetBits;
            return (
              <button
                key={b}
                type="button"
                aria-pressed={active}
                disabled={arm === "none" || arm === "unlimited"}
                onClick={() => { setBudgetBits(b); }}
                className={`numeric rounded-sm border px-2.5 py-1 font-mono text-xs disabled:opacity-40 ${
                  active
                    ? "border-canvas-strong bg-canvas-raised font-medium text-canvas-strong"
                    : "border-canvas-grid text-canvas-text hover:text-canvas-strong"
                }`}
              >
                {b}b
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5" role="group" aria-label="Compared arm">
        {SAMPLE_COMPARE_ARMS.map((a) => {
          const active = a === arm;
          const info = armInfo(a);
          return (
            <button
              key={a}
              type="button"
              aria-pressed={active}
              onClick={() => { setArm(a); }}
              className={`rounded-sm border px-2.5 py-1 text-xs font-medium ${
                active
                  ? "border-accent bg-accent text-ink"
                  : "border-canvas-axis bg-canvas-raised text-canvas-text hover:text-canvas-strong"
              }`}
            >
              {info.short}
            </button>
          );
        })}
      </div>

      {/* Main grid + side inspection */}
      <div className="mt-5 grid gap-6 lg:grid-cols-12">
        <div className="lg:col-span-6">
          <svg
            viewBox={`0 0 ${sizePx} ${sizePx}`}
            className="mx-auto block w-full max-w-[22rem] rounded-sm border border-canvas-axis bg-canvas"
            role="img"
            aria-label={`9 by 9 Voronoi navigation grid for lineage ${lineage.lineage_index} at generation ${keyGen}`}
          >
            {grid.map((row, r) =>
              row.map((t, c) => {
                const isStart = r === layoutEntry.heldout_preview.start[0] && c === layoutEntry.heldout_preview.start[1];
                const isGoal = r === layoutEntry.heldout_preview.goal[0] && c === layoutEntry.heldout_preview.goal[1];
                const lvl = t >= 0 ? (activeLevels[t] ?? 0) : 0;
                const fill = t < 0 ? "#1e293b" : (LEVEL_CELL_FILL[lvl] ?? "#1e293b");
                const txtColor = t < 0 ? "#e2e8f0" : (LEVEL_TEXT_COLOR[lvl] ?? "#cbd5e1");
                const cellLabel = isStart ? "S" : isGoal ? "G" : `T${t}`;
                return (
                  <g key={`${r}-${c}`}>
                    <rect
                      x={c * cellPx}
                      y={r * cellPx}
                      width={cellPx}
                      height={cellPx}
                      fill={fill}
                      stroke="var(--color-canvas-grid)"
                      strokeWidth="1"
                    />
                    <text
                      x={(c + 0.5) * cellPx}
                      y={(r + 0.5) * cellPx + 4}
                      textAnchor="middle"
                      fill={txtColor}
                      fontSize={isStart || isGoal ? "12" : "9.5"}
                      fontWeight={isStart || isGoal ? 700 : 500}
                      fontFamily="monospace"
                    >
                      {cellLabel}
                    </text>
                  </g>
                );
              }),
            )}

            {/* Dashed oracle reference path */}
            {oraclePath.length > 1 && (
              <polyline
                points={polylinePoints(oraclePath, cellPx)}
                fill="none"
                stroke="#94a3b8"
                strokeWidth="3"
                strokeDasharray="6 4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* Solid selected-arm planned path */}
            {selectedPath.length > 1 && (
              <polyline
                points={polylinePoints(selectedPath, cellPx)}
                fill="none"
                stroke={armMeta.color}
                strokeWidth="3.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}
          </svg>

          <div className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-canvas-text">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 border-t-2 border-dashed border-[#94a3b8]" />
              Oracle path
            </span>
            <span className="inline-flex items-center gap-1.5" style={{ color: armMeta.color }}>
              <span className="inline-block h-0.5 w-4" style={{ background: armMeta.color }} />
              {armMeta.short} path
            </span>
          </div>
        </div>

        <div className="space-y-4 lg:col-span-6">
          <div className="rounded-sm border border-canvas-axis bg-canvas-raised p-4">
            <p className="text-xs font-medium text-canvas-text uppercase">Selected state: {armMeta.label}</p>
            <dl className="mt-3 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
              <div>
                <dt className="text-canvas-text">Held-out regret</dt>
                <dd className="numeric mt-0.5 text-sm font-medium text-canvas-strong">
                  {genMetrics !== undefined ? fixed(genMetrics.heldout_regret, 2) : "–"}
                </dd>
              </div>
              <div>
                <dt className="text-canvas-text">Bits received</dt>
                <dd className="numeric mt-0.5 text-sm font-medium text-canvas-strong">
                  {genMetrics !== undefined ? `${genMetrics.bits_used}/${arm === "none" || arm === "unlimited" ? "—" : budgetBits}` : "–"}
                </dd>
              </div>
              <div>
                <dt className="text-canvas-text">Posterior H</dt>
                <dd className="numeric mt-0.5 text-sm font-medium text-canvas-strong">
                  {genMetrics !== undefined ? `${fixed(genMetrics.posterior_entropy_bits, 2)}b` : "–"}
                </dd>
              </div>
              <div>
                <dt className="text-canvas-text">MAP accuracy</dt>
                <dd className="numeric mt-0.5 text-sm font-medium text-canvas-strong">
                  {genMetrics !== undefined ? `${fixed(genMetrics.map_accuracy * 100, 0)}%` : "–"}
                </dd>
              </div>
            </dl>
          </div>

          <div className="rounded-sm border border-canvas-axis bg-canvas-raised p-4">
            <p className="text-xs font-medium text-canvas-text uppercase">
              Incoming record decoded by generation {keyGen}
            </p>
            <div className="mt-2 font-mono text-xs leading-relaxed text-accent-strong">
              {genMetrics === undefined || genMetrics.record_summary.length === 0 ? (
                <span className="text-canvas-text">
                  {keyGen === 0 ? "(generation 0 starts from uniform prior; no predecessor)" : "(empty record / none)"}
                </span>
              ) : (
                genMetrics.record_summary.join("  ")
              )}
            </div>
          </div>

          <div className="rounded-sm border border-canvas-axis bg-canvas-raised p-4">
            <p className="text-xs font-medium text-canvas-text uppercase">
              True terrain cost table {isPostShift ? "(shifted after gen 5)" : "(initial rule)"}
            </p>
            <div className="mt-2.5 grid grid-cols-4 gap-2 font-mono text-xs">
              {activeLevels.map((lvl, t) => {
                const shifted = isPostShift && lineage.rule_initial[t] !== lvl;
                return (
                  <div
                    key={t}
                    className="rounded-sm border border-canvas-grid px-2 py-1.5"
                    style={{ background: LEVEL_CELL_FILL[lvl] ?? "#1e293b", color: LEVEL_TEXT_COLOR[lvl] ?? "#cbd5e1" }}
                  >
                    <div className="flex items-center justify-between">
                      <span>T{t}</span>
                      {shifted && <span className="text-[10px] text-accent-strong">was L{lineage.rule_initial[t]}</span>}
                    </div>
                    <div className="mt-0.5 font-semibold">
                      L{lvl} (cost {levelCosts[lvl] ?? "?"})
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
