import { ArticleBody, ArticleHeader, ArticleSection, Callout, CitationList, ProvenanceBlock, type TocItem } from "../../components/Article";
import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import type { LoadedLibraryRun } from "../../lib/data/library";
import { useCatalog, useLibraryRun } from "../../lib/data/useResource";
import { fixed } from "../../lib/format";
import { docUrl } from "../../lib/links";
import { LAST_LIBRARY_LITERATURE } from "../../lib/literature";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { VerdictTable } from "./VerdictTable";

const STUDY_ID = "last-library";
const REPORT_DOC = "docs/studies/last-library/report.md";
const PROTOCOL_DOC = "docs/studies/last-library/protocol.md";

const TOC: readonly TocItem[] = [
  { id: "summary", label: "Summary & headline verdicts" },
  { id: "model", label: "World model & bit channel" },
  { id: "threshold", label: "19-bit capacity threshold (H1)" },
  { id: "ratchet", label: "Cumulative ratchet (Q3 / H4)" },
  { id: "change", label: "Rule-change lock-in & hedging (H2–H3)" },
  { id: "verdicts", label: "Pre-registered hypothesis table" },
  { id: "limits", label: "Limits, reproducibility & provenance" },
  { id: "references", label: "Verified literature" },
];

function findContrast(run: LoadedLibraryRun, family: string, id: string) {
  return run.results.hypotheses[family]?.contrasts.find((c) => c.contrast_id === id);
}

function LibraryReportBody({ run }: { run: LoadedLibraryRun }) {
  const { manifest, manifestUrl, results } = run;
  const gFinal = results.config.n_generations - 1;
  const gChange = results.config.change_generation;

  const h1b20 = findContrast(run, "H1", "H1_b20");
  const h2ab20 = findContrast(run, "H2a", "H2a_b20");
  const h2bb20 = findContrast(run, "H2b", "H2b_b20");
  const h3a = findContrast(run, "H3a", "H3a_b40");
  const h3bc20 = findContrast(run, "H3b", "H3b_change_b20");
  const h3bs20 = findContrast(run, "H3b", "H3b_stationary_b20");
  const h4Rule = findContrast(run, "H4", "H4_rule_b20");
  const h4Ped = findContrast(run, "H4", "H4_example_pedagogical_b20");

  const stNone = results.stationary.none_b0?.heldout_regret_mean_by_gen[gFinal];
  const stUnlim = results.stationary.unlimited_b0?.heldout_regret_mean_by_gen[gFinal];
  const stRule20 = results.stationary.rule_b20?.heldout_regret_mean_by_gen[gFinal];
  const stConf40 = results.stationary.rule_confidence_b40?.heldout_regret_mean_by_gen[gFinal];
  const stHedg20 = results.stationary.rule_hedged_b20?.heldout_regret_mean_by_gen[gFinal];
  const stPed20 = results.stationary.example_pedagogical_b20?.heldout_regret_mean_by_gen[gFinal];
  const stPed40 = results.stationary.example_pedagogical_b40?.heldout_regret_mean_by_gen[gFinal];

  const rcNone = results.rule_change.none_b0?.heldout_regret_mean_by_gen[gChange];
  const rcUnlim5 = results.rule_change.unlimited_b0?.heldout_regret_mean_by_gen[gChange];
  const rcUnlim9 = results.rule_change.unlimited_b0?.heldout_regret_mean_by_gen[gFinal];
  const rcRule5 = results.rule_change.rule_b20?.heldout_regret_mean_by_gen[gChange];
  const rcConf5 = results.rule_change.rule_confidence_b40?.heldout_regret_mean_by_gen[gChange];
  const rcHedg5 = results.rule_change.rule_hedged_b20?.heldout_regret_mean_by_gen[gChange];
  const rcPed20 = results.rule_change.example_pedagogical_b20?.heldout_regret_mean_by_gen[gChange];
  const rcPed40 = results.rule_change.example_pedagogical_b40?.heldout_regret_mean_by_gen[gChange];

  return (
    <ArticleBody toc={TOC}>
      <ArticleSection id="summary" number="1" title="Summary & Headline Verdicts">
        <p>
          <strong>The Last Library</strong> measures when an inherited bit bottleneck supports cumulative learning across disconnected generations of
          otherwise fresh Bayesian agents, and when compressed cultural knowledge turns harmful after an environmental rule change. Across{" "}
          {results.n_lineages} stationary and {results.n_lineages} rule-change paired lineages swept from 0 to{" "}
          {results.config.bit_budgets.at(-1) ?? 60} bits, we observe both a sharp bit-capacity cliff and an asymmetry between sender persuasion and
          receiver skepticism:
        </p>

        <div className="grid gap-4 pt-1 sm:grid-cols-2">
          <Callout title="Supported: dense rule cliff & cumulative ratchet (B = 20)" tone="ink">
            At the 19-bit dense rule threshold (B = 20), abstract rules slash stationary gen-{gFinal} regret to{" "}
            {stRule20 !== undefined ? fixed(stRule20, 2) : "–"} vs. {stPed20 !== undefined ? fixed(stPed20, 2) : "–"} for pedagogical examples (
            <strong>H1_b20</strong>: +{h1b20 !== undefined ? fixed(h1b20.mean_diff, 3) : "–"}, p_holm ={" "}
            {h1b20 !== undefined ? h1b20.p_holm.toExponential(2) : "–"}) and drive cumulative gen 1→{gFinal} learning (<strong>H4</strong>: +
            {h4Rule !== undefined ? fixed(h4Rule.mean_diff, 3) : "–"}), while examples plateau ({h4Ped !== undefined ? fixed(h4Ped.mean_diff, 3) : "–"}).
          </Callout>

          <Callout title="Mixed / refutations: confidence backfires, hedging protects" tone="caution">
            Sender confidence weights at B = 40 slightly worsen gen-{gChange} overshoot (<strong>H3a</strong>:{" "}
            {h3a !== undefined ? fixed(h3a.mean_diff, 3) : "–"}, refuting sender calibration). Receiver hazard hedging (h ={" "}
            {fixed(results.config.hedge_discount, 2)}) cuts shock by +{h3bc20 !== undefined ? fixed(h3bc20.mean_diff, 3) : "–"} (
            <strong>H3b</strong>) while preserving B = 20 stationary gains (+{h3bs20 !== undefined ? fixed(h3bs20.mean_diff, 3) : "–"}).
          </Callout>
        </div>
      </ArticleSection>

      <ArticleSection id="model" number="2" title="World Model, Channel & Baselines">
        <p>
          Each generation <code className="font-mono text-xs">g ∈ 0..{gFinal}</code> navigates {results.config.episodes_per_generation} Thompson-sampled{" "}
          {results.config.grid_size}×{results.config.grid_size} Voronoi terrain grids (from <code className="font-mono text-xs">4</code> active terrains per
          map drawn out of <code className="font-mono text-xs">{results.config.n_terrains}</code> terrain types, true costs{" "}
          <code className="font-mono text-xs">({results.config.level_costs.join(", ")})</code>, softmax observation noise{" "}
          <code className="font-mono text-xs">β = {fixed(results.config.noise_beta, 1)}</code>), and is evaluated on{" "}
          {results.config.heldout_layouts_per_gen} held-out maps. Uniform ignorance across all {results.config.n_terrains}×{results.config.n_levels} terrain
          levels has exact entropy <code className="font-mono text-xs">H = 16 bits</code>.
        </p>
        <p>
          Seven budgeted arms sweep exact bit counts{" "}
          <code className="font-mono text-xs">{results.config.bit_budgets.join(", ")}</code>: <code className="font-mono text-xs">example_random</code>{" "}
          and <code className="font-mono text-xs">example_pedagogical</code> encode 5-bit <code className="font-mono text-xs">(terrain, observed level)</code>{" "}
          experiences (<code className="font-mono text-xs">⌊B/5⌋</code> pairs); <code className="font-mono text-xs">rule</code> and{" "}
          <code className="font-mono text-xs">rule_hedged</code> pack all 8 MAP states densely at <code className="font-mono text-xs">19 bits</code> (
          <code className="font-mono text-xs">5^8 &lt; 2^19</code>) or sparse subsets at 5 bits each;{" "}
          <code className="font-mono text-xs">rule_confidence</code> requires <code className="font-mono text-xs">32 bits</code> dense; and{" "}
          <code className="font-mono text-xs">placebo_random</code> / <code className="font-mono text-xs">placebo_shuffled</code> verify no information leaks
          beyond actual inherited signal.
        </p>
      </ArticleSection>

      <ArticleSection id="threshold" number="3" title="The 19-Bit Capacity Threshold (H1)">
        <p>
          Contrary to naive claims that rules always beat examples at any budget, <strong>H1</strong> confirms superiority only once the bit budget crosses the
          19-bit dense ternary packing threshold. Below 19 bits (<code className="font-mono text-xs">B = 10, 15</code>), a rule sender can transmit only 2 or 3
          lowest-entropy terrains, omitting active terrains needed on unseen Voronoi layouts; neither <code className="font-mono text-xs">H1_b10</code> nor{" "}
          <code className="font-mono text-xs">H1_b15</code> passes the <code className="font-mono text-xs">δ = 0.50</code> margin.
        </p>
        <p>
          At <code className="font-mono text-xs">B = 20</code>, <code className="font-mono text-xs">rule</code> crosses dense threshold (stationary gen-9{" "}
          <code className="font-mono text-xs">{stRule20 !== undefined ? fixed(stRule20, 2) : "–"}</code> vs.{" "}
          <code className="font-mono text-xs">example_pedagogical@20</code>{" "}
          <code className="font-mono text-xs">{stPed20 !== undefined ? fixed(stPed20, 2) : "–"}</code>). Pushing the channel wider to{" "}
          <code className="font-mono text-xs">B = 40</code> gives <code className="font-mono text-xs">example_pedagogical</code> 8 well-chosen experiences,
          narrowing stationary regret to <code className="font-mono text-xs">{stPed40 !== undefined ? fixed(stPed40, 2) : "–"}</code> (vs.{" "}
          <code className="font-mono text-xs">{stConf40 !== undefined ? fixed(stConf40, 2) : "–"}</code> for dense{" "}
          <code className="font-mono text-xs">rule_confidence@40</code> and{" "}
          <code className="font-mono text-xs">{stHedg20 !== undefined ? fixed(stHedg20, 2) : "–"}</code> for{" "}
          <code className="font-mono text-xs">rule_hedged</code>).
        </p>
      </ArticleSection>

      <ArticleSection id="ratchet" number="4" title="Cumulative Cultural Ratchet Beyond Generation 1 (Q3 / H4)">
        <p>
          Under stationary terrain costs, uncompressed posterior inheritance (<code className="font-mono text-xs">unlimited</code>) reaches{" "}
          <code className="font-mono text-xs">{stUnlim !== undefined ? fixed(stUnlim, 2) : "–"}</code> vs.{" "}
          <code className="font-mono text-xs">{stNone !== undefined ? fixed(stNone, 2) : "–"}</code> for fresh learners (<code className="font-mono text-xs">none</code>)
          and matches <code className="font-mono text-xs">single_lifetime</code> to floating-point precision. Crucially, <strong>H4</strong> demonstrates a
          qualitative ratchet divergence at <code className="font-mono text-xs">B = 20</code>: <code className="font-mono text-xs">rule@20</code> drops from gen
          1 to gen {gFinal} by <code className="font-mono text-xs">+{h4Rule !== undefined ? fixed(h4Rule.mean_diff, 2) : "–"}</code> regret units, whereas
          bit-bottlenecked <code className="font-mono text-xs">example_pedagogical@20</code> cannot accumulate observations beyond 4 pairs per generation
          and fails to improve after generation 1 (<code className="font-mono text-xs">{h4Ped !== undefined ? fixed(h4Ped.mean_diff, 2) : "–"}</code>).
        </p>
      </ArticleSection>

      <ArticleSection id="change" number="5" title="Unannounced Rule Shift & the Sender–Receiver Asymmetry (H2–H3)">
        <p>
          At generation {gChange} in the rule-change condition, every terrain strictly mutates to a different cost level via a derangement. The result exposes
          a severe fragility: lossless <code className="font-mono text-xs">unlimited</code> Bayesian transfer experiences catastrophic lock-in, spiking to{" "}
          <code className="font-mono text-xs">{rcUnlim5 !== undefined ? fixed(rcUnlim5, 2) : "–"}</code> at gen {gChange} and remaining stuck at{" "}
          <code className="font-mono text-xs">{rcUnlim9 !== undefined ? fixed(rcUnlim9, 2) : "–"}</code> at gen {gFinal}—worse than learning from scratch (
          <code className="font-mono text-xs">none = {rcNone !== undefined ? fixed(rcNone, 2) : "–"}</code>)!
        </p>
        <p>
          Compressed <code className="font-mono text-xs">rule@20</code> spikes to{" "}
          <code className="font-mono text-xs">{rcRule5 !== undefined ? fixed(rcRule5, 2) : "–"}</code> at gen {gChange}, higher than{" "}
          <code className="font-mono text-xs">example_pedagogical@20</code> (
          <code className="font-mono text-xs">{rcPed20 !== undefined ? fixed(rcPed20, 2) : "–"}</code>, diff +
          {h2ab20 !== undefined ? fixed(h2ab20.mean_diff, 2) : "–"}) and{" "}
          <code className="font-mono text-xs">none</code> (diff +{h2bb20 !== undefined ? fixed(h2bb20.mean_diff, 2) : "–"}). At B = 40, pedagogical
          examples absorb enough evidence to stiffen too (<code className="font-mono text-xs">{rcPed40 !== undefined ? fixed(rcPed40, 2) : "–"}</code>).
          Encoding confidence (<strong>H3a</strong>) amplifies overshoot (<code className="font-mono text-xs">{rcConf5 !== undefined ? fixed(rcConf5, 2) : "–"}</code>),
          whereas receiver hazard hedging (<strong>H3b</strong>) dampens gen-{gChange} shock to{" "}
          <code className="font-mono text-xs">{rcHedg5 !== undefined ? fixed(rcHedg5, 2) : "–"}</code> while keeping B = 20 stationary superiority.
        </p>
      </ArticleSection>

      <ArticleSection id="verdicts" number="6" title="Pre-Registered Hypothesis Table">
        <VerdictTable hypotheses={results.hypotheses} />
      </ArticleSection>

      <ArticleSection id="limits" number="7" title="Limitations, Reproducibility & Artifact Provenance">
        <ul>
          {manifest.limitations.map((lim) => (
            <li key={lim}>{lim}</li>
          ))}
          <li>
            Protocol v1 (<code className="font-mono text-xs">config_sha256: {results.config_sha256.slice(0, 12)}…</code>) froze seeds{" "}
            <code className="font-mono text-xs">{results.seeds.stationary_start}..{results.seeds.stationary_end_inclusive}</code> and{" "}
            <code className="font-mono text-xs">{results.seeds.rule_change_start}..{results.seeds.rule_change_end_inclusive}</code> before evaluation (D39).
          </li>
        </ul>
        <ProvenanceBlock manifest={manifest} manifestUrl={manifestUrl} />
      </ArticleSection>

      <ArticleSection id="references" number="8" title="Verified Literature">
        <CitationList citations={LAST_LIBRARY_LITERATURE} />
      </ArticleSection>
    </ArticleBody>
  );
}

function LoadedLibraryReport({ runId, path }: { runId: string; path: string }) {
  const run = useLibraryRun(runId, path);
  if (run.status === "loading") return <LoadingState label="Verifying SHA-256 digests and loading report data…" />;
  if (run.status === "error") return <ErrorState error={run.error} onRetry={run.retry} />;
  return <LibraryReportBody run={run.value} />;
}

export function LibraryReportPage() {
  usePageTitle("The Last Library Report – Phase Space");
  const catalog = useCatalog();
  const entry =
    catalog.status === "ready"
      ? catalog.value.studies.find((s) => s.study_id === STUDY_ID)?.runs.find((r) => r.kind === "confirmatory_results")
      : undefined;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-8 sm:px-8">
      <ArticleHeader
        crumbs={[{ label: "Studies", to: "/" }, { label: "The Last Library", to: "/library" }, { label: "Report" }]}
        title="The Last Library: rules, bits & environmental lock-in"
        subtitle="When compressed rules beat pedagogical examples across generations, why sender certainty backfires under shift, and how receiver hedging recovers."
        actions={
          <>
            <Link to="/library" className="rounded-sm bg-ink px-4 py-2 text-sm font-medium text-sheet no-underline hover:bg-canvas-raised">
              Open interactive viewer
            </Link>
            <a className="prose-link inline-flex items-center gap-1.5 text-sm text-graphite" href={docUrl(REPORT_DOC)} rel="noreferrer">
              Markdown report in GitHub
              <ExternalIcon className="size-3.5" />
            </a>
            <a className="prose-link inline-flex items-center gap-1.5 text-sm text-graphite" href={docUrl(PROTOCOL_DOC)} rel="noreferrer">
              Frozen protocol (D39)
              <ExternalIcon className="size-3.5" />
            </a>
          </>
        }
      />

      {catalog.status === "loading" && <div className="mt-8"><LoadingState label="Reading catalog…" /></div>}
      {catalog.status === "error" && <div className="mt-8"><ErrorState error={catalog.error} onRetry={catalog.retry} /></div>}
      {catalog.status === "ready" && (entry === undefined ? (
        <div className="mt-8">
          <ErrorState error={new Error("No Last Library confirmatory run found in catalog.json.")} />
        </div>
      ) : (
        <LoadedLibraryReport runId={entry.run_id} path={entry.path} />
      ))}
    </div>
  );
}
