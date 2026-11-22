import { useState, type ReactNode } from "react";
import { Link } from "../../lib/router";
import { ProvenanceBlock, StatusBadge } from "../../components/Article";
import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import type { LoadedLibraryRun } from "../../lib/data/library";
import { useCatalog, useLibraryRun } from "../../lib/data/useResource";
import { fixed } from "../../lib/format";
import { docUrl } from "../../lib/links";
import { usePageTitle } from "../../lib/usePageTitle";
import { BudgetCurveChart } from "./BudgetCurveChart";
import { GridInspector } from "./GridInspector";
import { TrajectoryChart } from "./TrajectoryChart";
import { VerdictTable } from "./VerdictTable";

const STUDY_ID = "last-library";

const PROBLEM =
  "Each generation of Bayesian agents navigates unfamiliar 9×9 Voronoi terrain grids for two short episodes, then hands a tiny record—measured in exact bits around the 16-bit latent rule entropy—to an otherwise fresh successor. " +
  "We asked when compressed abstract rules beat bit-matched pedagogical examples under stationary worlds, whether those rules lock successors into costly mistakes after an unannounced terrain cost shift, and how receiver hazard hedging affects the trade-off.";

function Section({ title, id, children, lead }: { title: string; id: string; children: ReactNode; lead?: ReactNode }) {
  return (
    <section aria-labelledby={id} className="mt-16 sm:mt-24">
      <h2 id={id} className="text-2xl sm:text-3xl">
        {title}
      </h2>
      {lead !== undefined && <div className="mt-3 max-w-[68ch] text-graphite">{lead}</div>}
      <div className="mt-7">{children}</div>
    </section>
  );
}

function LibraryContent({ run }: { run: LoadedLibraryRun }) {
  const { manifest, manifestUrl, results, samples } = run;
  const [selectedBudget, setSelectedBudget] = useState<number>(20);

  const stRule20 = results.stationary.rule_b20?.heldout_regret_mean_by_gen[results.config.n_generations - 1];
  const stPed20 = results.stationary.example_pedagogical_b20?.heldout_regret_mean_by_gen[results.config.n_generations - 1];
  const cGen = results.config.change_generation;
  const rcRule20 = results.rule_change.rule_b20?.heldout_regret_mean_by_gen[cGen];
  const rcHedged20 = results.rule_change.rule_hedged_b20?.heldout_regret_mean_by_gen[cGen];
  const rcNone = results.rule_change.none_b0?.heldout_regret_mean_by_gen[cGen];

  return (
    <>
      <header className="mt-8 rounded-sm border border-rule bg-sheet p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <StatusBadge status={manifest.status} />
          <p className="numeric font-mono text-xs text-quiet">
            N = {results.n_lineages} paired lineages per scenario · budgets 0..{results.config.bit_budgets.at(-1) ?? 60} bits
          </p>
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          <div className="border-l-2 border-rule-strong pl-3.5">
            <p className="text-xs text-quiet uppercase">19-bit capacity jump (B = 20)</p>
            <p className="numeric mt-1 font-serif text-xl text-ink">
              Rule {stRule20 !== undefined ? fixed(stRule20, 2) : "–"} vs. Examples {stPed20 !== undefined ? fixed(stPed20, 2) : "–"}
            </p>
            <p className="mt-1 text-xs text-graphite">Once dense 19-bit capacity opens, abstract rules slash stationary gen-9 regret.</p>
          </div>

          <div className="border-l-2 border-rule-strong pl-3.5">
            <p className="text-xs text-quiet uppercase">Rule-change lock-in (gen {cGen})</p>
            <p className="numeric mt-1 font-serif text-xl text-ink">
              Rule {rcRule20 !== undefined ? fixed(rcRule20, 2) : "–"} vs. None {rcNone !== undefined ? fixed(rcNone, 2) : "–"}
            </p>
            <p className="mt-1 text-xs text-graphite">Unhedged rules (and even more so Unlimited posterior passing) overshoot when terrain costs invert.</p>
          </div>

          <div className="border-l-2 border-rule-strong pl-3.5">
            <p className="text-xs text-quiet uppercase">Receiver hedging vs. confidence</p>
            <p className="numeric mt-1 font-serif text-xl text-ink">
              Hedged {rcHedged20 !== undefined ? fixed(rcHedged20, 2) : "–"} (cuts overshoot)
            </p>
            <p className="mt-1 text-xs text-graphite">Sender confidence calibration failed (H3a), while 30% receiver hedging mitigated gen-{cGen} shock (H3b).</p>
          </div>
        </div>
      </header>

      <Section
        id="budget-curves"
        title="1. The bit-capacity cliff across 0 to 60 bits"
        lead={
          <p>
            Sweep the inherited bit budget across all 11 arms under stationary worlds and after an unannounced rule shift at generation {cGen}. Click
            any budget pill (<code className="font-mono text-xs">10b</code>, <code className="font-mono text-xs">20b</code>,{" "}
            <code className="font-mono text-xs">40b</code>…) to drive the generational trajectories below.
          </p>
        }
      >
        <BudgetCurveChart results={results} selectedBudget={selectedBudget} onSelectBudget={setSelectedBudget} />
      </Section>

      <Section
        id="trajectories"
        title="2. Generational trajectories & the cumulative ratchet"
        lead={
          <p>
            Track how held-out regret, belief entropy, and MAP accuracy evolve from generation 0 to {results.config.n_generations - 1}. Notice that at{" "}
            <code className="font-mono text-xs">B = 20</code>, <code className="font-mono text-xs">rule</code> climbs steadily while{" "}
            <code className="font-mono text-xs">example_pedagogical</code> stalls—yet under a rule shift at gen {cGen}, lossless{" "}
            <code className="font-mono text-xs">unlimited</code> suffers severe Bayesian lock-in.
          </p>
        }
      >
        <TrajectoryChart results={results} budgetBits={selectedBudget} />
      </Section>

      <Section
        id="inspector"
        title="3. Voronoi gridworld & decoded record inspector"
        lead={
          <p>
            Inspect fixed sample lineages 0, 1, and 2 directly from <code className="font-mono text-xs">samples.json</code>: compare the true underlying
            Voronoi terrain cost map, exact inherited tokens decoded by each generation, and the shortest paths actually chosen against the oracle.
          </p>
        }
      >
        <GridInspector samples={samples} />
      </Section>

      <Section
        id="verdicts"
        title="4. Pre-registered Holm-adjusted hypothesis verdicts"
        lead={
          <p>
            All 16 confirmatory contrasts evaluated across 200 stationary (<code className="font-mono text-xs">300000..300199</code>) and 200
            rule-change (<code className="font-mono text-xs">400000..400199</code>) lineages at minimum margin <code className="font-mono text-xs">δ = 0.50</code>{" "}
            and Holm-corrected <code className="font-mono text-xs">α = 0.05</code>.
          </p>
        }
      >
        <VerdictTable hypotheses={results.hypotheses} />
      </Section>

      <Section id="provenance" title="5. Run provenance & automated sanity checks">
        <ProvenanceBlock manifest={manifest} manifestUrl={manifestUrl} />
      </Section>
    </>
  );
}

function LoadedLibrary({ runId, path }: { runId: string; path: string }) {
  const run = useLibraryRun(runId, path);
  if (run.status === "loading") return <LoadingState label="Verifying SHA-256 digests and loading confirmatory data…" />;
  if (run.status === "error") return <ErrorState error={run.error} onRetry={run.retry} />;
  return <LibraryContent run={run.value} />;
}

export function LibraryPage() {
  usePageTitle("The Last Library – Phase Space");
  const catalog = useCatalog();
  const entry =
    catalog.status === "ready"
      ? catalog.value.studies.find((s) => s.study_id === STUDY_ID)?.runs.find((r) => r.kind === "confirmatory_results")
      : undefined;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-8 sm:px-8">
      <div className="max-w-3xl pt-10 sm:pt-14">
        <nav aria-label="Breadcrumb" className="text-sm text-graphite">
          <Link to="/" className="prose-link">
            Studies
          </Link>
          <span aria-hidden="true" className="mx-2 text-quiet">
            /
          </span>
          <span aria-current="page">The Last Library</span>
        </nav>

        <h1 className="mt-3 text-4xl sm:text-5xl">The Last Library</h1>
        <p className="mt-5 text-lg leading-relaxed text-graphite">{PROBLEM}</p>

        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          <Link to="/library/report" className="rounded-sm bg-ink px-4 py-2 font-medium text-sheet no-underline hover:bg-canvas-raised">
            Read full report &amp; references
          </Link>
          <a className="prose-link inline-flex items-center gap-1.5 text-graphite" href={docUrl("docs/studies/last-library/protocol.md")} rel="noreferrer">
            Frozen protocol (D39)
            <ExternalIcon className="size-3.5" />
          </a>
          <a className="prose-link inline-flex items-center gap-1.5 text-graphite" href={docUrl("docs/studies/last-library/README.md")} rel="noreferrer">
            Study specification
            <ExternalIcon className="size-3.5" />
          </a>
        </div>
      </div>

      {catalog.status === "loading" && <div className="mt-10"><LoadingState label="Reading catalog…" /></div>}
      {catalog.status === "error" && <div className="mt-10"><ErrorState error={catalog.error} onRetry={catalog.retry} /></div>}
      {catalog.status === "ready" && (entry === undefined ? (
        <div className="mt-10">
          <ErrorState error={new Error("No Last Library confirmatory run found in catalog.json.")} />
        </div>
      ) : (
        <LoadedLibrary runId={entry.run_id} path={entry.path} />
      ))}
    </div>
  );
}
