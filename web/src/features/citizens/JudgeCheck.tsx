import { Term } from "../../components/Tooltip";
import type { E1 } from "../../lib/data/citizens";
import { citizenInfo, modelColor } from "./model";

const SCORES = "e1_scores.json";

/** Rows shown, in order; B0 (returns nothing) is a control mentioned in the tooltip. */
const SHOWN = ["B1", "B2", "B3", "B23", "M0-sim", "B4-calibrated", "B4-plain"] as const;
const LEARNERS = ["B3", "B23"];

const NAMES: Record<string, string> = { "M0-sim": "Control: random returns at human rates" };

export interface JudgeScore {
  model: string;
  scorable: number;
  agreements: number;
  excluded: number;
  notScorable: number;
}

/** Per-model agreement counts of the E1 run (`e1_scores.json`), read by column name. */
export function judgeScores(e1: E1): JudgeScore[] {
  const table = e1.tables[SCORES];
  if (table === undefined) return [];
  const col = (name: string): number => table.columns.findIndex((c) => c.name === name);
  const [m, s, a, x, n] = ["model", "scorable", "agreements", "excluded", "not_scorable"].map(col);
  if ([m, s, a, x, n].some((i) => i === undefined || i < 0)) return [];
  const num = (v: unknown): number => (typeof v === "number" ? v : Number.NaN);
  return table.rows.map((row) => ({
    model: String(row[m ?? 0]),
    scorable: num(row[s ?? 0]),
    agreements: num(row[a ?? 0]),
    excluded: num(row[x ?? 0]),
    notScorable: num(row[n ?? 0]),
  }));
}

function list(items: string[]): string {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1] ?? ""}`;
}

/** The one-line reading of the scores, only as strong as the data allows. */
export function judgeTakeaway(scores: JudgeScore[]): string | null {
  const perfect = (k: string): boolean => scores.some((s) => s.model === k && s.scorable > 0 && s.agreements === s.scorable);
  if (!perfect("B1")) return null;
  const learners = LEARNERS.filter(perfect);
  if (learners.length === LEARNERS.length) {
    return `B1 orders all of them right — but so do the learner models (${list(learners)}), so this check says B1 is a sound judge, not the uniquely right one.`;
  }
  const others = ["B2", ...LEARNERS].filter(perfect);
  return others.length === 0 ? "B1 is the only model of people that orders all of them right." : `B1 orders all of them right, and so does ${list(others)}.`;
}

/** "Is the judge credible?": how many published human comparisons each model orders the way people did (exploratory E1). */
export function JudgeCheck({ e1 }: { e1: E1 }) {
  const scores = judgeScores(e1);
  const rows = SHOWN.map((k) => scores.find((s) => s.model === k)).filter((s): s is JudgeScore => s !== undefined && s.scorable > 0);
  const main = scores.find((s) => s.model === "B1");
  if (rows.length === 0 || main === undefined) return null;
  const b0 = scores.find((s) => s.model === "B0");
  const takeaway = judgeTakeaway(scores);
  const llm = rows.filter((r) => r.model.startsWith("B4"));
  const tip =
    `Exploratory: designed after the held-out human outcomes had been seen, and it cannot change a pre-registered verdict. ` +
    `The ${main.excluded} comparisons involving DeepMind's first AI manager are left out (our stand-in for it failed the human check), ` +
    `and ${main.notScorable} more cannot be scored with the stored simulations.` +
    (llm.length > 0 ? ` The LLM players were run only under the held-out rules, so ${llm[0]?.scorable ?? 0} comparisons apply to them.` : "") +
    (b0 === undefined ? "" : ` A control that returns nothing (B0) gets ${b0.agreements} of ${b0.scorable}.`);

  return (
    <div className="max-w-[44rem]">
      <h3 className="font-serif text-xl">Is the judge credible?</h3>
      <p className="mt-2 text-[0.95rem] text-graphite">
        DeepMind compared its human-tested rules in pairs. Of the{" "}
        <Term tip={tip}>{main.scorable} comparisons we can score</Term>, how many does each model order the same way people did?
      </p>
      <table className="mt-4 w-full text-sm">
        <caption className="sr-only">Published comparisons each model orders the same way as people</caption>
        <tbody>
          {rows.map((r) => {
            const info = citizenInfo(r.model);
            const color = r.model.startsWith("B") && !r.model.startsWith("B4") ? modelColor(r.model) : "var(--color-graphite)";
            return (
              <tr key={r.model} className="border-b border-rule">
                <th scope="row" className="py-1.5 pr-3 text-left font-normal whitespace-nowrap">
                  {r.model.startsWith("M0") ? "" : <span className="mr-1.5 font-medium">{r.model.replace(/^B4-.*/, "B4")}</span>}
                  <span className="text-graphite">{NAMES[r.model] ?? info?.name ?? r.model}</span>
                </th>
                <td className="w-full py-1.5" aria-hidden="true">
                  <span className="flex gap-[3px]">
                    {Array.from({ length: r.scorable }, (_, i) => (
                      <span
                        key={i}
                        className="inline-block size-2.5 rounded-[2px]"
                        style={i < r.agreements ? { background: color } : { boxShadow: `inset 0 0 0 1.5px ${color}` }}
                      />
                    ))}
                  </span>
                </td>
                <td className="numeric py-1.5 pl-3 text-right whitespace-nowrap">
                  {r.agreements} / {r.scorable}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {takeaway !== null && <p className="mt-3 text-[0.95rem] font-medium text-ink">{takeaway}</p>}
    </div>
  );
}
