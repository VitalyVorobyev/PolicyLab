import type { LibraryResults } from "../../lib/data/library";
import { useCatalog, useLibraryRun } from "../../lib/data/useResource";
import { fixed } from "../../lib/format";

function MiniCurves({ results }: { results: LibraryResults }) {
  const arms = [
    { key: "rule", label: "Rule", color: "#e3a72f", dash: "" },
    { key: "rule_hedged", label: "Rule+hedge", color: "#4fd1b0", dash: "" },
    { key: "example_pedagogical", label: "Ex. ped.", color: "#eb6834", dash: "5 2" },
  ] as const;

  const budgets = results.budget_curves.budgets;
  const W = 520;
  const H = 220;
  const padL = 44;
  const padR = 92;
  const padT = 22;
  const padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xPos = (b: number): number => padL + (b / 60) * plotW;
  const yMax = 10;
  const yPos = (v: number): number => padT + plotH - (Math.min(yMax, Math.max(0, v)) / yMax) * plotH;

  return (
    <div className="rounded-sm bg-canvas p-4 sm:p-5" aria-hidden="true">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {[0, 2, 4, 6, 8, 10].map((v) => (
          <g key={v}>
            <line x1={padL} x2={W - padR} y1={yPos(v)} y2={yPos(v)} stroke="var(--color-canvas-grid)" />
            <text x={padL - 6} y={yPos(v) + 3.5} textAnchor="end" fill="var(--color-canvas-text)" fontSize="10">
              {v}
            </text>
          </g>
        ))}

        <line x1={xPos(19)} x2={xPos(19)} y1={padT} y2={padT + plotH} stroke="var(--color-accent)" strokeOpacity="0.65" strokeDasharray="3 2" />
        <text x={xPos(19) + 3} y={padT + 10} fill="var(--color-accent-strong)" fontSize="9.5">
          19b cliff
        </text>

        {budgets.map((b) => (
          <text key={b} x={xPos(b)} y={H - 12} textAnchor="middle" fill="var(--color-canvas-text)" fontSize="10">
            {b}b
          </text>
        ))}

        {arms.map(({ key, label, color, dash }, idx) => {
          const pts = results.budget_curves.arms[key] ?? [];
          const d = pts
            .map((p, i) => `${i === 0 ? "M" : "L"}${xPos(p.budget_bits).toFixed(1)},${yPos(p.stationary_final_mean).toFixed(1)}`)
            .join(" ");
          const last = pts[pts.length - 1];
          const labelY = Math.max(padT + 10, Math.min(padT + plotH, yPos(last?.stationary_final_mean ?? 0) + (idx === 0 ? -8 : idx === 1 ? 8 : 0)));
          return (
            <g key={key}>
              <path d={d} fill="none" stroke={color} strokeWidth="2.2" strokeDasharray={dash || undefined} />
              {pts.map((p) => (
                <circle key={p.budget_bits} cx={xPos(p.budget_bits)} cy={yPos(p.stationary_final_mean)} r="2.8" fill={color} />
              ))}
              {last !== undefined && (
                <text x={W - padR + 6} y={labelY} fill={color} fontSize="10" fontWeight="500">
                  {label} ({fixed(last.stationary_final_mean, 1)})
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function LoadedThumb({ runId, path }: { runId: string; path: string }) {
  const run = useLibraryRun(runId, path);
  return run.status === "ready" ? <MiniCurves results={run.value.results} /> : <div className="min-h-[14rem] rounded-sm bg-canvas" aria-hidden="true" />;
}

/** Gallery preview of The Last Library read from the published confirmatory artifact. */
export function LibraryThumbnail() {
  const catalog = useCatalog();
  const entry =
    catalog.status === "ready"
      ? catalog.value.studies.find((s) => s.study_id === "last-library")?.runs.find((r) => r.kind === "confirmatory_results")
      : undefined;
  return entry === undefined ? <div className="min-h-[14rem] rounded-sm bg-canvas" aria-hidden="true" /> : <LoadedThumb runId={entry.run_id} path={entry.path} />;
}
