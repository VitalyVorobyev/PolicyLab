import { Term } from "../../components/Tooltip";
import type { CitizensDemo } from "../../lib/data/citizens";
import { coins, governorInfo } from "./model";

const REGIME: Record<string, string> = { "R-I": "Simple rule (k = 22)", "R-M1": "DeepMind AI (M1)", "R-M2": "DeepMind AI (M2)" };
const VARIANT: Record<string, string> = { calibrated: "Calibrated", plain: "Uncalibrated" };

function Mark({ pass, testable }: { pass: boolean; testable: boolean }) {
  if (!testable) return <span className="text-quiet">diagnostic only</span>;
  return <span className={pass ? "font-medium text-valid" : "text-danger"}>{pass ? "passes" : "fails"}</span>;
}

function passedText(records: { variant: string; regime: string; closed_loop: boolean; counts_as_validated: boolean }[]): string {
  if (records.length === 0) return ".";
  const parts = [...new Set(records.map((r) => r.variant))].sort((a, b) => (a === "calibrated" ? -1 : b === "calibrated" ? 1 : a.localeCompare(b))).map((v) => {
    const passed = records.filter((r) => r.variant === v && r.closed_loop && r.counts_as_validated).map((r) => REGIME[r.regime] ?? r.regime);
    const name = v === "calibrated" ? "the calibrated version (player types matched to human play)" : `the ${(VARIANT[v] ?? v).toLowerCase()} version`;
    return `${name} ${passed.length === 0 ? "passed on none of the unseen rules" : `passed on games under ${passed.join(" and ")}`}`;
  });
  return `: ${parts.join("; ")}.`;
}

/** LLM players (B4): the whole-game exam per variant and rule, and the games against the AI governors when present. */
export function LlmCitizens({ demo }: { demo: CitizensDemo }) {
  const records = demo.b4_c2?.records ?? [];
  const variants = [...new Set(records.map((r) => r.variant))].sort((a, b) => (a === "calibrated" ? -1 : b === "calibrated" ? 1 : a.localeCompare(b)));
  const regimes = [...new Set(records.map((r) => r.regime))];
  const c4 = (demo.b4_rows ?? []).filter((r) => r.source === "c4");
  const c4Governors = [...new Set(c4.map((r) => r.governor))];
  const cli = [...new Set(c4.map((r) => r.cli_version).filter((v): v is string => v !== null))];
  const h2b4 = demo.hypotheses?.h2_b4 ?? null;

  return (
    <div className="grid grid-cols-1 gap-x-12 gap-y-8 lg:grid-cols-12">
      <div className="lg:col-span-5">
        <p className="text-graphite">
          A fifth population: Claude Haiku plays every citizen, reading the same rules people read. It took the same whole-game exam as the four
          models{passedText(records)}
        </p>
        {c4.length === 0 && <p className="mt-3 text-sm text-quiet">Games between the LLM players and the AI governors are running; their results will appear here.</p>}
      </div>
      <div className="min-w-0 space-y-8 lg:col-span-7">
        {records.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[20rem] text-left text-sm">
              <caption className="mb-2 text-left text-xs text-graphite">
                <Term tip="Whole-game exam: at least 3 of 4 game outcomes inside the range of held-out human games, 40 LLM games per cell. Under M1 the simulated stand-in for the rule failed its own check, so the result there is a diagnostic.">
                  Whole-game exam
                </Term>
                , outcomes inside the human range (of 4)
              </caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-graphite">
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    LLM players
                  </th>
                  {regimes.map((g) => (
                    <th key={g} scope="col" className="py-1.5 pr-3 font-medium">
                      {REGIME[g] ?? g}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {variants.map((v) => (
                  <tr key={v} className="border-b border-rule">
                    <th scope="row" className="py-2 pr-3 font-normal">
                      {VARIANT[v] ?? v}
                    </th>
                    {regimes.map((g) => {
                      const r = records.find((x) => x.variant === v && x.regime === g);
                      return (
                        <td key={g} className="py-2 pr-3">
                          {r === undefined ? (
                            "–"
                          ) : (
                            <span className="inline-flex flex-wrap gap-x-2">
                              <span className="numeric">{r.metrics_in_range}</span>
                              <Mark pass={r.counts_as_validated} testable={r.closed_loop} />
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {c4.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[20rem] text-left text-sm">
              <caption className="mb-2 text-left text-xs text-graphite">
                Mean surplus per game with LLM players{" "}
                <Term tip={`These games ran on Claude Code CLI ${cli.join(", ") || "(see the report)"} rather than the version pinned for the earlier exams: the tool updated itself, and the change was recorded and disclosed before any of these results (decision D37). Prompts, model and settings are unchanged.`}>
                  (newer CLI version)
                </Term>
              </caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-graphite">
                  <th scope="col" className="py-1.5 font-medium">
                    Governor
                  </th>
                  {variants.map((v) => (
                    <th key={v} scope="col" className="py-1.5 text-right font-medium">
                      {VARIANT[v] ?? v}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {c4Governors.map((g) => (
                  <tr key={g} className="border-b border-rule">
                    <th scope="row" className="py-1.5 font-normal">
                      {governorInfo(g)?.name ?? g}
                    </th>
                    {variants.map((v) => {
                      const r = c4.find((x) => x.governor === g && x.variant === v);
                      return (
                        <td key={v} className="numeric py-1.5 text-right">
                          {r?.total_surplus_mean == null ? "–" : coins(r.total_surplus_mean)}
                          {r !== undefined && <span className="text-xs text-quiet"> ({r.games_complete} games)</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            {h2b4 !== null && (
              <ul className="mt-3 space-y-1 text-sm text-graphite">
                {Object.entries(h2b4).map(([variant, v]) => (
                  <li key={variant}>
                    {VARIANT[variant] ?? variant}:{" "}
                    {v.verdict !== null ? `pre-registered test of next-move-designed governors on LLM players: ${v.verdict}` : (v.label ?? "descriptive only")}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
