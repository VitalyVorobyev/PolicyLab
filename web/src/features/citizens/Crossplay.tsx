import { Term, Tooltip } from "../../components/Tooltip";
import type { CitizensDemo, CrossplayCell, Hypotheses } from "../../lib/data/citizens";
import { fixed } from "../../lib/format";
import { wholeGameModels } from "./findings";
import { cellOf, citizenInfo, coins, governorInfo, kLabel, MODEL_KEYS, modelColor, signedCoins, signedPercent, SIMPLE_RULE } from "./model";

export const RULE_ROWS = ["Equal", "Proportional", SIMPLE_RULE];
export const AI_ROWS = [...MODEL_KEYS.map((m) => `Planner[${m}]`), "Planner[Mix]"];

/** Diverging bins of the relative difference against the column's tuned simple rule. */
const BINS = [0.03, 0.1, 0.25, 0.5];

function step(relative: number): number {
  const a = Math.abs(relative);
  let s = 0;
  for (const edge of BINS) if (a >= edge) s++;
  return s;
}

export function fill(relative: number): { background: string; color: string } {
  const s = step(relative);
  if (s === 0) return { background: "var(--color-even)", color: "var(--color-ink)" };
  const tone = relative > 0 ? "better" : "worse";
  return { background: `var(--color-${tone}-${s})`, color: s >= 3 ? "#ffffff" : "var(--color-ink)" };
}

/** Which pre-registered claim (if any) confirmed this cell against the column's k*. */
function confirmation(h: Hypotheses | null, policy: string, column: string): "loss" | "gain" | null {
  if (h === null) return null;
  const name = `${policy}@${column} vs Interpolating k*[${column}]`;
  const has = (families: string[]): boolean => families.some((f) => (h.confirmed[f] ?? []).includes(name));
  if (has(["H1-reversal", "H2a-loss", "own-model-limited"])) return "loss";
  if (has(["H1-transfer", "own-model-gain"])) return "gain";
  return null;
}

function cellTip(cell: CrossplayCell, kStar: number, games: number): string {
  const g = governorInfo(cell.policy)?.name ?? cell.policy;
  const c = citizenInfo(cell.column)?.name ?? cell.column;
  const d = cell.vs_k_star;
  const seeds = cell.seeds.filter((s) => s !== null).length;
  let text = `${g} with ${cell.column} (${c.toLowerCase()}): ${coins(cell.total_surplus_mean)} coins per game on average over ${games.toLocaleString("en-US")} games${seeds > 1 ? ` and ${seeds} training seeds` : ""}. `;
  text += `That is ${signedCoins(d.difference)} (${signedPercent(d.relative)}) against the simple rule tuned for these citizens (k = ${kLabel(kStar)}).`;
  if (d.ci !== null) text += ` 95 % interval ${signedCoins(d.ci[0])} to ${signedCoins(d.ci[1])} coins.`;
  if (d.seeds_positive !== null && d.seeds_negative !== null && seeds > 1) text += ` ${d.seeds_positive} of ${seeds} seeds above, ${d.seeds_negative} below.`;
  const extra = cell.extra_metrics ?? {};
  const depleted = extra.depleted_fraction_mean;
  const depletion = extra.depletion_round_mean;
  const active = extra.players_active_mean;
  if (depleted !== undefined) text += ` The pool ran dry in ${Math.round(depleted * 100)} % of games`;
  if (depletion !== undefined) text += `${depleted === undefined ? " Depletion round" : "; depletion round"} ${fixed(depletion, 1)} on average (40 if never)`;
  if (depleted !== undefined || depletion !== undefined) text += ".";
  if (active !== undefined) text += ` Players offered at least one coin in the last round: ${fixed(active, 1)} of 4.`;
  return text;
}

/** The cross-play matrix: every governor against every citizen model, coloured against the tuned simple rule. */
export function CrossplayHeatmap({ demo, onPick }: { demo: CitizensDemo; onPick: (policy: string, column: string) => void }) {
  const cp = demo.crossplay;
  const columns = cp.columns.filter((c) => (MODEL_KEYS as readonly string[]).includes(c));
  const passed = wholeGameModels(demo.exams);
  const groups = [RULE_ROWS, AI_ROWS].map((rows) => rows.filter((r) => cp.policies.includes(r)));
  const traced = (policy: string, column: string): boolean => demo.traces.some((t) => t.governor === policy && t.citizens === column);

  return (
    <figure>
      <div role="table" aria-label="Mean surplus per game of each governor with each citizen model, against the tuned simple rule" className="grid grid-cols-[minmax(5.5rem,0.9fr)_repeat(4,minmax(0,1fr))] gap-[3px] text-sm">
        <div role="row" className="contents">
          <div role="columnheader" className="flex items-end pb-1 text-xs text-graphite">
            Governor ↓ · citizens →
          </div>
          {columns.map((c) => (
            <div key={c} role="columnheader" className="pb-1.5 text-center">
              <span className="flex items-center justify-center gap-1.5 font-medium">
                <span className="inline-block size-2 rounded-full" style={{ background: modelColor(c) }} aria-hidden="true" />
                {c}
              </span>
              <span className="hidden text-xs text-graphite sm:block">{citizenInfo(c)?.name}</span>
              {passed.includes(c) && (
                <Term tip="The only citizen model that passed the whole-game exam on human games it had never seen." className="text-[0.6875rem] font-medium text-valid">
                  passed both exams
                </Term>
              )}
            </div>
          ))}
        </div>
        {groups.map((rows, gi) =>
          rows.map((policy, ri) => {
            const info = governorInfo(policy);
            return (
              <div key={policy} role="row" className={`contents ${gi === 1 && ri === 0 ? "[&>*]:mt-2" : ""}`}>
                <div role="rowheader" className="flex items-center pr-1 text-[0.8125rem] leading-tight">
                  <span className="hidden sm:inline">{info?.name ?? policy}</span>
                  <span className="sm:hidden">{info?.short ?? policy}</span>
                </div>
                {columns.map((c) => {
                  const cell = cellOf(demo, policy, c);
                  if (cell === undefined) return <div key={c} role="cell" />;
                  const style = fill(cell.vs_k_star.relative);
                  const home = info?.trainedOn === c;
                  const confirmed = confirmation(demo.hypotheses, policy, c);
                  const clickable = traced(policy, c);
                  const content = (
                    <>
                      <span className="numeric block text-[0.9375rem] leading-tight font-medium">{signedPercent(cell.vs_k_star.relative)}</span>
                      <span className="numeric block text-[0.6875rem] leading-tight opacity-80">{coins(cell.total_surplus_mean)}</span>
                      {confirmed !== null && (
                        <span className="absolute top-0.5 right-1 text-[0.625rem]" aria-hidden="true">
                          ◆
                        </span>
                      )}
                    </>
                  );
                  const className = `relative flex h-12 w-full flex-col items-center justify-center rounded-[2px] ${home ? "outline-2 -outline-offset-2 outline-ink" : ""}`;
                  return (
                    <div key={c} role="cell" className="min-w-0">
                      <Tooltip content={cellTip(cell, cp.k_star[c] ?? Number.NaN, cp.games) + (confirmed === null ? "" : ` Pre-registered test: confirmed ${confirmed}.`) + (home ? " Home: the citizens this governor was trained against." : "")}>
                        {(props) =>
                          clickable ? (
                            <button
                              type="button"
                              {...props}
                              onClick={(e) => {
                                props.onClick?.(e);
                                onPick(policy, c);
                              }}
                              aria-label={`${info?.name ?? policy} with ${c}: ${coins(cell.total_surplus_mean)} coins, ${signedPercent(cell.vs_k_star.relative)} against the tuned simple rule. Play in the demonstration.`}
                              className={`${className} cursor-pointer transition-[filter] hover:brightness-95`}
                              style={style}
                            >
                              {content}
                            </button>
                          ) : (
                            <span {...props} tabIndex={0} className={className} style={style}>
                              {content}
                            </span>
                          )
                        }
                      </Tooltip>
                    </div>
                  );
                })}
              </div>
            );
          }),
        )}
      </div>
      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-graphite">
        <span>
          Mean surplus per game against the{" "}
          <Term tip="The simple rule with its one setting k re-tuned for each citizen model (k*), so every column is compared with the best simple rule for those citizens; see “Tune the rule yourself”.">
            simple rule tuned for each column
          </Term>
        </span>
        <span className="flex items-center gap-[2px]" aria-hidden="true">
          {[-0.6, -0.3, -0.15, -0.05, 0, 0.05, 0.15, 0.3, 0.6].map((v) => (
            <span key={v} className="inline-block h-3 w-5 first:rounded-l-[2px] last:rounded-r-[2px]" style={{ background: fill(v).background }} />
          ))}
        </span>
        <span>
          <span className="text-danger">less</span> · <span style={{ color: "var(--color-better-4)" }}>more</span>; framed: home model
          {demo.hypotheses !== null && <> · ◆ confirmed by a pre-registered test</>}
        </span>
      </figcaption>
    </figure>
  );
}



const H1_TEXT: Record<string, string> = {
  refuted: "Refuted: at least one AI governor loses to the simple rule on another model by more than the minimum effect, confirmed.",
  supported: "Supported: every AI governor beats the simple rule on every other model, confirmed.",
  "not refuted": "Not refuted, not shown: no confirmed loss, but transfer is not confirmed everywhere.",
};

const H2_TEXT: Record<string, string> = {
  supported: "Supported: a governor designed on a model that only passed the next-move exam loses to the simple rule on B1, confirmed. Next-move validation was not enough.",
  refuted: "Refuted: all three such governors are confirmed not worse than the simple rule on B1.",
  inconclusive: "Inconclusive: neither a confirmed loss nor confirmed non-inferiority on B1.",
};

export function Verdicts({ h }: { h: Hypotheses }) {
  return (
    <dl className="grid gap-x-10 gap-y-4 border-l-2 border-ink pl-5 sm:grid-cols-2">
      <div>
        <dt className="text-sm font-medium">
          <Term tip={`Pre-registered hypothesis H1: on every other citizen model, each AI governor beats the simple rule tuned for that model by more than ${fixed(h.delta, 0)} coins per game.`}>
            Do AI governors transfer?
          </Term>
        </dt>
        <dd className="mt-1 text-graphite">{H1_TEXT[h.verdicts.H1] ?? h.verdicts.H1}</dd>
      </div>
      <div>
        <dt className="text-sm font-medium">
          <Term tip={`Pre-registered hypothesis H2: on B1, governors trained on B2, B3 or B23 lose to the simple rule tuned for B1 by more than ${fixed(h.delta, 0)} coins per game.`}>
            Is next-move validation enough?
          </Term>
        </dt>
        <dd className="mt-1 text-graphite">{H2_TEXT[h.verdicts.H2] ?? h.verdicts.H2}</dd>
      </div>
    </dl>
  );
}
