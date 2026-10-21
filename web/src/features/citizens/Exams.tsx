import { useState } from "react";
import { CheckCircleIcon } from "../../components/icons";
import { Term } from "../../components/Tooltip";
import type { ExamRow, Exams as ExamsData } from "../../lib/data/citizens";
import { fixed } from "../../lib/format";
import { citizenInfo, MODEL_KEYS, modelColor } from "./model";

const REGIME_LABEL: Record<string, string> = {
  "R-I": "Simple rule (k = 22)",
  "R-M1": "DeepMind AI (M1)",
  "R-M2": "DeepMind AI (M2)",
};

function regimeLabel(regime: string): string {
  return REGIME_LABEL[regime] ?? regime;
}

function Verdict({ pass, children }: { pass: boolean; children?: React.ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1 text-sm font-medium ${pass ? "text-valid" : "text-danger"}`}>
      {pass ? (
        <CheckCircleIcon />
      ) : (
        <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6">
          <circle cx="8" cy="8" r="6.5" />
          <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" strokeLinecap="round" />
        </svg>
      )}
      {children ?? (pass ? "passes" : "fails")}
    </span>
  );
}

function ScoreBar({ row, max }: { row: ExamRow; max: number }) {
  const pct = (v: number): string => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-5 flex-1" aria-hidden="true">
        <div className="absolute inset-y-[9px] left-0 right-0 bg-rule" />
        <div className="absolute inset-y-1 left-0 rounded-r-[3px]" style={{ width: pct(row.delta_ell), background: modelColor(row.model) }} />
        <div className="absolute top-[3px] h-[14px] border-x border-ink" style={{ left: pct(row.delta_ci[0]), width: `calc(${pct(row.delta_ci[1])} - ${pct(row.delta_ci[0])})` }}>
          <div className="absolute inset-x-0 top-[6px] h-px bg-ink" />
        </div>
      </div>
      <span className="numeric w-10 text-right text-sm">{fixed(row.delta_ell, 2)}</span>
    </div>
  );
}

function MetricDots({ inRange }: { inRange: number }) {
  return (
    <span className="inline-flex gap-1" aria-hidden="true">
      {[0, 1, 2, 3].map((i) => (
        <span key={i} className={`inline-block size-3 rounded-[2px] border ${i < inRange ? "border-ink bg-ink" : "border-rule-strong bg-transparent"}`} />
      ))}
    </span>
  );
}

/** The two held-out exams per citizen model, ordered by the next-move score so the inversion shows. */
export function Exams({ exams }: { exams: ExamsData }) {
  const regimes = exams.regimes.filter((r) => exams.rows.some((row) => row.regime === r));
  const [regime, setRegime] = useState(regimes.includes("R-I") ? "R-I" : (regimes[0] ?? ""));
  const rows = exams.rows
    .filter((r) => r.regime === regime && (MODEL_KEYS as readonly string[]).includes(r.model))
    .sort((a, b) => b.delta_ell - a.delta_ell);
  const max = Math.max(0.05, ...exams.rows.map((r) => r.delta_ci[1])) * 1.05;
  const wholeGameTestable = exams.closed_loop_tier[regime] ?? true;
  const units = rows[0]?.units;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm">
        <span className="text-graphite">
          <Term tip="Both exams use human games under a rule none of the models saw during fitting (held out until the protocol was frozen).">Exam on games under</Term>
        </span>
        <div role="group" aria-label="Unseen rule" className="flex flex-wrap gap-1">
          {regimes.map((r) => (
            <button
              key={r}
              type="button"
              aria-pressed={r === regime}
              onClick={() => { setRegime(r); }}
              className={`rounded-sm border px-2.5 py-1 text-[0.8125rem] ${r === regime ? "border-ink bg-sheet font-medium" : "border-rule text-graphite hover:border-rule-strong"}`}
            >
              {regimeLabel(r)}
            </button>
          ))}
        </div>
        {units !== undefined && <span className="text-quiet">{units} held-out games</span>}
      </div>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[20rem] border-collapse text-left">
          <thead>
            <tr className="border-b border-rule-strong text-xs text-graphite">
              <th scope="col" className="w-[38%] py-2 pr-4 font-medium">
                Model
              </th>
              <th scope="col" className="w-[34%] py-2 pr-6 font-medium">
                <Term tip="How much better than a baseline that ignores the history the model predicts each player's next return on held-out players: mean log-likelihood gain per decision (Δℓ, nats), with its 95 % interval. Passing needs the whole interval above zero.">
                  Exam 1 · next move
                </Term>
              </th>
              <th scope="col" className="py-2 font-medium">
                <Term tip="Simulate whole games under the rule and compare four outcomes (surplus, inequality, when the pool runs dry, players still active) with the held-out human games. Passing needs at least 3 of the 4 inside the human range.">
                  Exam 2 · whole game
                </Term>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const info = citizenInfo(row.model);
              return (
                <tr key={row.model} className="border-b border-rule align-middle">
                  <th scope="row" className="py-3 pr-4 font-normal">
                    <span className="flex items-center gap-2">
                      <span className="inline-block size-2.5 shrink-0 rounded-full" style={{ background: modelColor(row.model) }} aria-hidden="true" />
                      <span className="font-medium">{row.model}</span>
                      <span className="hidden text-graphite sm:inline">{info?.name}</span>
                    </span>
                  </th>
                  <td className="py-3 pr-6">
                    <ScoreBar row={row} max={max} />
                    <span className="sr-only">
                      Δℓ {fixed(row.delta_ell, 3)}, 95 % interval {fixed(row.delta_ci[0], 3)} to {fixed(row.delta_ci[1], 3)}; {row.pass_likelihood ? "passes" : "fails"}
                    </span>
                  </td>
                  <td className="py-3">
                    {wholeGameTestable ? (
                      <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <MetricDots inRange={row.metrics_in_range} />
                        <span className="numeric hidden text-sm text-graphite sm:inline">{row.metrics_in_range} of 4</span>
                        <Verdict pass={row.pass_outcomes} />
                      </span>
                    ) : (
                      <Term tip="The simulated stand-in for this AI mechanism failed its own check against human games, so whole games under it cannot be simulated reliably.">
                        <span className="text-sm text-quiet">not testable</span>
                      </Term>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** One sentence on the inversion, computed from the exam rows of the closed-loop regime. */
export function examsHeadline(exams: ExamsData): string | null {
  const rows = exams.rows.filter((r) => r.regime === "R-I" && (MODEL_KEYS as readonly string[]).includes(r.model)).sort((a, b) => b.delta_ell - a.delta_ell);
  if (rows.length < 2) return null;
  const passing = rows.filter((r) => r.pass_outcomes).map((r) => r.model);
  if (passing.length === 0 || !rows.every((r) => r.pass_likelihood)) return null;
  const topTwo = rows.slice(0, 2);
  const none = topTwo.every((r) => r.metrics_in_range === 0);
  return `All ${rows.length === 4 ? "four" : rows.length} models pass the next-move exam, but only ${passing.join(", ")} passes the whole-game exam${
    none ? `; the two best next-move predictors (${topTwo.map((r) => r.model).join(" and ")}) have none of the four game outcomes inside the human range` : ""
  }.`;
}
