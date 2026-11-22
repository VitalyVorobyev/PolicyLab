import { AlertIcon, CheckCircleIcon } from "../../components/icons";
import type { LibraryHypothesisFamily } from "../../lib/data/library";
import { fixed } from "../../lib/format";

const FAMILY_ORDER = ["H1", "H2a", "H2b", "H3a", "H3b", "H4"] as const;

function formatP(p: number): string {
  if (p < 1e-3) return p.toExponential(2);
  return fixed(p, 4);
}

function signedFixed(x: number, decimals: number): string {
  const s = fixed(x, decimals);
  return x > 0 ? `+${s}` : s;
}

/** Pre-registered Holm-adjusted hypothesis test table shared across `/library` and `/library/report`. */
export function VerdictTable({ hypotheses }: { hypotheses: Record<string, LibraryHypothesisFamily> }) {
  const families = FAMILY_ORDER.map((id) => hypotheses[id]).filter((f): f is LibraryHypothesisFamily => f !== undefined);

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {families.map((fam) => {
          const badgeTone = fam.all_passed ? "text-valid border-rule-strong" : fam.n_passed > 0 ? "text-caution border-rule-strong" : "text-danger border-rule-strong";
          return (
            <div key={fam.family_id} className={`rounded-sm border bg-sheet p-3.5 ${badgeTone}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-xs font-semibold text-ink">{fam.family_id}</span>
                {fam.n_passed > 0 ? <CheckCircleIcon className="size-4 shrink-0" /> : <AlertIcon className="size-4 shrink-0" />}
              </div>
              <p className="numeric mt-1 font-mono text-sm font-medium text-ink">
                {fam.n_passed}/{fam.n_contrasts} supported
              </p>
              <p className="mt-1 text-xs leading-snug text-graphite">{fam.title}</p>
            </div>
          );
        })}
      </div>

      <div className="overflow-x-auto rounded-sm border border-rule bg-sheet">
        <table className="w-full min-w-[44rem] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-rule-strong bg-paper text-xs text-quiet uppercase">
              <th scope="col" className="px-3.5 py-2.5 font-medium">
                Contrast
              </th>
              <th scope="col" className="px-3.5 py-2.5 text-right font-medium">
                Mean diff (95% CI)
              </th>
              <th scope="col" className="px-3.5 py-2.5 text-right font-medium">
                t-stat
              </th>
              <th scope="col" className="px-3.5 py-2.5 text-right font-medium">
                Holm p
              </th>
              <th scope="col" className="px-3.5 py-2.5 text-right font-medium">
                d_z
              </th>
              <th scope="col" className="px-3.5 py-2.5 text-right font-medium">
                Verdict
              </th>
            </tr>
          </thead>
          <tbody>
            {families.flatMap((fam) =>
              fam.contrasts.map((c) => (
                <tr key={c.contrast_id} className="border-b border-rule last:border-b-0">
                  <th scope="row" className="px-3.5 py-2.5 font-normal text-ink">
                    <span className="mr-2 font-mono text-xs font-medium text-quiet">{c.family}</span>
                    <span>{c.label}</span>
                  </th>
                  <td className="numeric px-3.5 py-2.5 text-right font-mono text-xs whitespace-nowrap text-ink">
                    {signedFixed(c.mean_diff, 3)} [{signedFixed(c.ci95_low, 3)}, {signedFixed(c.ci95_high, 3)}]
                  </td>
                  <td className="numeric px-3.5 py-2.5 text-right font-mono text-xs text-graphite">{fixed(c.t_stat, 2)}</td>
                  <td className="numeric px-3.5 py-2.5 text-right font-mono text-xs text-graphite">{formatP(c.p_holm)}</td>
                  <td className="numeric px-3.5 py-2.5 text-right font-mono text-xs text-graphite">{fixed(c.cohens_dz, 3)}</td>
                  <td className="px-3.5 py-2.5 text-right whitespace-nowrap">
                    <span className={`inline-flex items-center gap-1 text-xs font-medium ${c.passed ? "text-valid" : "text-danger"}`}>
                      {c.passed ? <CheckCircleIcon className="size-3.5" /> : <AlertIcon className="size-3.5" />}
                      {c.passed ? "Supported" : "Not supported"}
                    </span>
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
