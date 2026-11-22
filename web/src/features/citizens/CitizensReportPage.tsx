import { ArticleBody, ArticleHeader, ArticleSection, Callout, CitationList, ProvenanceBlock, type TocItem } from "../../components/Article";
import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import type { CitizensDemo, LoadedCitizensRun } from "../../lib/data/citizens";
import { useCatalog, useCitizensRun } from "../../lib/data/useResource";
import { fixed } from "../../lib/format";
import { CITIZENS_LITERATURE } from "../../lib/literature";
import { docUrl } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { Verdicts } from "./Crossplay";
import { Exams, examsHeadline } from "./Exams";
import { JudgeCheck } from "./JudgeCheck";
import { cellOf, coins, governorInfo, kLabel, MODEL_KEYS, signedCoins, signedPercent, SIMPLE_RULE } from "./model";

const STUDY_ID = "model-citizens";
const REPORT_DOC = "docs/studies/model-citizens/report.md";
const PROTOCOL_DOC = "docs/studies/model-citizens/protocol.md";

const TOC: readonly TocItem[] = [
  { id: "summary", label: "Summary & verdicts" },
  { id: "game", label: "Common-pool game & C1 audit" },
  { id: "exams", label: "Four models, two exams" },
  { id: "crossplay", label: "Cross-play & ablations" },
  { id: "distillation", label: "Mechanism distillation" },
  { id: "llm-judge", label: "LLM citizens & E1 judge check" },
  { id: "limits", label: "Scope, limits & provenance" },
  { id: "references", label: "Verified literature" },
];

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, x) => sum + x, 0) / values.length;
}

function rangeText(values: readonly number[], decimals: number): string {
  if (values.length === 0) return "–";
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const fLo = fixed(lo, decimals);
  const fHi = fixed(hi, decimals);
  return fLo === fHi ? fLo : `${fLo}–${fHi}`;
}

function CellMark({ demo, policy, column }: { demo: CitizensDemo; policy: string; column: string }) {
  const h = demo.hypotheses;
  if (h === null) return null;
  const label = `${policy}@${column} vs Interpolating k*[${column}]`;
  const has = (family: string): boolean => (h.confirmed[family] ?? []).includes(label);
  if (has("H1-reversal") || has("H2a-loss") || has("own-model-limited")) {
    return (
      <span className="ml-1 text-[0.7rem] text-danger" title="Confirmed below the tuned simple rule by more than δ">
        ◆
      </span>
    );
  }
  if (has("H1-transfer") || has("own-model-gain")) {
    return (
      <span className="ml-1 text-[0.7rem] text-valid" title="Confirmed above the tuned simple rule by more than δ">
        ◆
      </span>
    );
  }
  return null;
}

function ReportBody({ run }: { run: LoadedCitizensRun }) {
  const { demo, manifest, manifestUrl } = run;
  const columns = demo.crossplay.columns.filter((c) => (MODEL_KEYS as readonly string[]).includes(c));
  const examSummary = demo.exams === null ? null : examsHeadline(demo.exams);
  const b4Records = demo.b4_c2?.records ?? [];
  const oneStepLossPct = ["B2", "B3", "B23"]
    .map((m) => cellOf(demo, `Planner[${m}]`, "B1")?.vs_k_star.relative)
    .filter((v): v is number => v !== undefined)
    .map((r) => Math.round(Math.abs(r) * 100));

  return (
    <ArticleBody toc={TOC}>
      <ArticleSection id="summary" number="1" title="Summary & Headline Verdicts">
        <p>
          Model Citizens tests whether reinforcement-learning (RL) policy designers trained against different behavioural models of people—all fitted
          to the exact same human baseline games—discover the same mechanism and transfer across models. It is an empirical test of the{" "}
          <strong>Lucas critique</strong> applied to AI mechanism design: when a new rule changes the environment, does a behavioural simulator that
          only predicted next moves under old rules still hold when a policy optimises against it?
        </p>
        <p>
          Across four citizen models built from the Koster et al. (2025) dataset, all four pass a pre-registered held-out one-step likelihood test, yet
          only the GRU clone <strong>B1</strong> passes the closed-loop whole-game outcome test on the unseen Interpolating rule. When PPO planners
          trained on the one-step-only models (<strong>B2</strong>, <strong>B3</strong>, <strong>B23</strong>) are judged against B1, all three lose{" "}
          {oneStepLossPct.length > 0 ? `${Math.min(...oneStepLossPct)}–${Math.max(...oneStepLossPct)} %` : "roughly a quarter"} of surplus to B1&apos;s
          re-tuned simple rule, every seed of each.
        </p>
        {demo.hypotheses !== null && (
          <div className="my-5 rounded-sm border border-rule bg-sheet p-5">
            <p className="mb-3 text-xs font-medium tracking-wider text-quiet uppercase">
              Pre-registered C4 confirmatory verdicts (deterministic mode, δ = {fixed(demo.hypotheses.delta, 0)} coins)
            </p>
            <Verdicts h={demo.hypotheses} />
          </div>
        )}
        <Callout title="Headline limitation: judged by simulators, not new human players" tone="caution">
          <p>
            Every AI governor here is evaluated against simulated citizen populations, and B1 itself passed the closed-loop exam narrowly (3 of 4
            metrics on the Interpolating regime, with Gini inside range only via the frozen tolerance τ = 0.005). No C4 planner has been tested with
            real human players.
          </p>
        </Callout>
      </ArticleSection>

      <ArticleSection id="game" number="2" title="The Common-Pool Game & Replay Audit (C1 / G1)">
        <p>
          {demo.game.players} players share a pool starting at <span className="numeric">{demo.game.max_pool}</span> coins for{" "}
          <span className="numeric">{demo.game.rounds}</span> rounds. In each round, a governor divides the current pool into four player offers{" "}
          <span className="font-mono text-[0.92rem]">eᵢ</span>; each player decides how many coins <span className="font-mono text-[0.92rem]">cᵢ</span>{" "}
          to return, keeping <span className="font-mono text-[0.92rem]">eᵢ − cᵢ</span>. Returned coins grow by{" "}
          <span className="numeric">×{demo.game.growth_factor}</span> according to the pool update rule{" "}
          <code className="rounded-[2px] bg-sheet px-1.5 py-0.5 font-mono text-[0.875rem] text-ink">{demo.game.pool_rule}</code>.
        </p>
        <p>
          <strong>Data audit and game reconstruction (C1 / G1b).</strong> From {demo.data_attribution}, we audited 4,264 recorded games (167,782 rows):
          280 human training games (Equal, Mixed, Proportional), 400 held-out human games (Interpolating k = 22, M1, M2), and 3,584 simulated
          behavioural-clone games. Replay checks uncovered four exact deployment details (D33): the <span className="numeric">{demo.game.pool_floor}</span>{" "}
          coin floor, 0.001 coins withheld per offer, float32 rounding to 4 decimals, and quantised weights{" "}
          <code className="rounded-[2px] bg-sheet px-1.5 py-0.5 font-mono text-[0.875rem] text-ink">round((R/200)^22, 1)</code>. All 17 published
          Wilcoxon rank-sum z-values match within tolerance, and 23 rows in 5 held-out M1 Exp 1 games sit 0.0100 coins below the full rule without
          altering any depletion round (D33).
        </p>
      </ArticleSection>

      <ArticleSection id="exams" number="3" title="Four Models of People, Two Exams (C2)">
        <p>
          All four trainable citizen models were fitted strictly to the training split of Equal, Mixed, and Proportional games before any held-out
          human game was accessed (frozen under C2 validation protocol D34):
        </p>
        <ul className="list-disc space-y-1.5 pl-6 text-[0.975rem] text-graphite">
          <li>
            <strong className="text-ink">B1 (Neural clone):</strong> 4-member ensemble of DeepMind&apos;s GRU imitation clones conditioned on offers
            and past reciprocations.
          </li>
          <li>
            <strong className="text-ink">B2 (Conditional cooperators):</strong> 12-parameter structural model combining a zero hurdle, a full-return
            spike, and an interior beta-binomial distribution responsive to offer generosity and others&apos; returns.
          </li>
          <li>
            <strong className="text-ink">B3 (Learners):</strong> 6-parameter Experience-Weighted Attraction (EWA) model over 11 discrete return
            fractions (Camerer &amp; Ho 1999).
          </li>
          <li>
            <strong className="text-ink">B23 (Hybrid learners):</strong> B3 augmented with a 1-parameter conditional-cooperation pull toward others&apos;
            prior return shares (7 parameters).
          </li>
        </ul>
        {examSummary !== null && <p className="font-medium text-ink">{examSummary}</p>}
        {demo.exams !== null && (
          <div className="my-4 rounded-sm border border-rule bg-sheet p-5">
            <Exams exams={demo.exams} />
          </div>
        )}
        <p>
          <strong>Why the exams invert.</strong> B3 and B23 achieve the highest per-decision log-likelihood gain (Δℓ = 0.180 and 0.174 nat on R-I, above
          B1&apos;s 0.104), yet when rolled out for 40 rounds under an unseen mechanism they collapse the pool far too early (mean depletion round 23.4
          and 18.5 vs. human 35.2). B2 errs in the opposite direction, sustaining the pool too long with unnaturally equal payouts (Gini 0.12 vs. human
          0.18 on R-I). Following G2, only B1 earned a closed-loop validation label on R-I; the M1 surrogate failed human-row acceptance, leaving R-M1 as
          a one-step-only diagnostic (D35).
        </p>
      </ArticleSection>

      <ArticleSection id="crossplay" number="4" title="Cross-Play, Transfer & Ablations (C4)">
        <p>
          Under frozen C4 protocol v1 (D36), PPO planners (set-encoder + GRU policy, 1,000 iterations = 40.96M decisions per seed, 5 seeds per main
          model, 3 seeds each for the no-retention ablation S1 and mixture planner S2) were trained and cross-played across{" "}
          <span className="numeric">{demo.crossplay.games.toLocaleString("en-US")}</span> paired evaluation games per column. Cells below show deterministic
          seed means and percent difference against each column&apos;s re-tuned simple rule <span className="font-mono text-[0.9rem]">k*</span> (
          {columns.map((c) => `${c}: ${kLabel(demo.crossplay.k_star[c] ?? 0)}`).join(", ")}).
        </p>
        <div className="my-4 overflow-x-auto rounded-sm border border-rule bg-sheet p-4">
          <table className="w-full min-w-[34rem] border-collapse text-left text-sm">
            <caption className="mb-3 text-left text-xs text-graphite">
              Cross-play mean total surplus per game (coins) and % relative to column k* · ◆ confirmed against k* under Holm α = 0.05
            </caption>
            <thead>
              <tr className="border-b border-rule-strong text-xs text-graphite">
                <th scope="col" className="py-2 pr-4 font-medium">
                  Governor
                </th>
                {columns.map((col) => (
                  <th key={col} scope="col" className="py-2 pr-3 text-right font-medium">
                    {col} <span className="font-normal text-quiet">(k*={kLabel(demo.crossplay.k_star[col] ?? 0)})</span>
                  </th>
                ))}
                <th scope="col" className="py-2 text-right font-medium">
                  Worst-col share
                </th>
              </tr>
            </thead>
            <tbody>
              {demo.crossplay.policies.map((policy) => {
                const info = governorInfo(policy);
                const rob = demo.crossplay.robustness[policy];
                return (
                  <tr key={policy} className="border-b border-rule last:border-b-0">
                    <th scope="row" className="py-2 pr-4 font-medium text-ink">
                      {info?.name ?? policy}
                    </th>
                    {columns.map((col) => {
                      const cell = cellOf(demo, policy, col);
                      if (cell === undefined) {
                        return (
                          <td key={col} className="py-2 pr-3 text-right text-quiet">
                            –
                          </td>
                        );
                      }
                      const isHome = info?.trainedOn === col;
                      return (
                        <td key={col} className={`numeric py-2 pr-3 text-right whitespace-nowrap ${isHome ? "font-semibold" : ""}`}>
                          <span>{coins(cell.total_surplus_mean)}</span>{" "}
                          <span className="text-xs text-graphite">({signedPercent(cell.vs_k_star.relative)})</span>
                          <CellMark demo={demo} policy={policy} column={col} />
                        </td>
                      );
                    })}
                    <td className="numeric py-2 text-right whitespace-nowrap text-graphite">
                      {rob === undefined ? "–" : `${Math.round(rob.worst_ratio * 100)} % (${rob.worst_column})`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p>
          <strong>
            Pre-registered findings (run <code className="font-mono text-[0.875rem]">citizens-c4-hypotheses-v1</code>).
          </strong>
        </p>
        <ul className="list-disc space-y-2 pl-6 text-[0.975rem] text-graphite">
          <li>
            <strong className="text-ink">H1 refuted (no universal transfer):</strong> 7 of the 12 off-diagonal cells are confirmed reversals more than
            δ = 16 coins below <span className="font-mono text-[0.9rem]">k*</span>; only the two sister learning models transfer to one another (
            <span className="font-mono text-[0.875rem]">Planner[B3]@B23</span> and <span className="font-mono text-[0.875rem]">Planner[B23]@B3</span>).
          </li>
          <li>
            <strong className="text-ink">H2 supported (one-step validation is insufficient on B1):</strong> all three H2a-loss cells vs{" "}
            <span className="font-mono text-[0.9rem]">k*[B1]</span> (−315 to −364 coins) and all three H2b-loss cells vs{" "}
            <span className="font-mono text-[0.9rem]">Planner[B1]</span> (−591 to −641 coins) are confirmed losses across all five seeds. Sampled
            evaluation agrees on every confirmatory verdict (zero disagreement).
          </li>
          <li>
            <strong className="text-ink">S1 (no-retention ablation):</strong> training a 4-way policy that cannot retain coins removes most of
            Planner[B2]&apos;s deficit on B1 (−315 to −42 coins) but leaves H1 <strong>refuted</strong> with 4 confirmed reversals on B3/B23;{" "}
            <code className="rounded-[2px] bg-sheet px-1 py-0.5 font-mono text-[0.85rem]">attributable_to_retention = false</code>.
          </li>
          <li>
            <strong className="text-ink">S2 (mixture planner):</strong> training uniformly across all four citizen models achieves 0 confirmed reversals
            (<strong className="text-ink">not refuted</strong>), 2 confirmed transfers (B3, B23), rough parity with{" "}
            <span className="font-mono text-[0.9rem]">k*[B1]</span> on B1 (−13 coins, 95 % CI −114 to +62), and a worst-column ratio of 82 % vs. 45 %
            for {SIMPLE_RULE}.
          </li>
        </ul>
      </ArticleSection>

      <ArticleSection id="distillation" number="5" title="Mechanism Distillation">
        <p>
          Each planner was played on 1,000 distillation games against its training model and approximated both by a binned equal-vs-proportional weight
          curve <span className="font-mono text-[0.9rem]">w(R)</span> (10 bins of <span className="font-mono text-[0.9rem]">R/200</span>) and by a
          1-parameter power law <span className="font-mono text-[0.9rem]">w = (R/200)^k</span>.
        </p>
        <div className="my-4 overflow-x-auto rounded-sm border border-rule bg-sheet p-4">
          <table className="w-full min-w-[30rem] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-rule-strong text-xs text-graphite">
                <th scope="col" className="py-2 pr-4 font-medium">
                  AI Governor
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  w-rule R² (seed range)
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  Fitted k range
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-medium">
                  Planner gain vs k*
                </th>
                <th scope="col" className="py-2 text-right font-medium">
                  w(R) curve gain vs k*
                </th>
              </tr>
            </thead>
            <tbody>
              {demo.distilled.planners.map((p) => {
                const r2s = p.r2_per_seed.filter((v): v is number => v !== null);
                const plannerDiff = mean(p.planner_minus_k_star_per_seed);
                const wDiff = mean(p.w_rule_minus_k_star_per_seed);
                return (
                  <tr key={p.policy} className="border-b border-rule last:border-b-0">
                    <th scope="row" className="py-2 pr-4 font-medium text-ink">
                      {governorInfo(p.policy)?.name ?? p.policy}
                    </th>
                    <td className="numeric py-2 pr-3 text-right text-graphite">{rangeText(r2s, 2)}</td>
                    <td className="numeric py-2 pr-3 text-right text-graphite">{rangeText(p.fitted_k_per_seed, 1)}</td>
                    <td className="numeric py-2 pr-3 text-right">{signedCoins(plannerDiff)} coins</td>
                    <td className="numeric py-2 text-right">{signedCoins(wDiff)} coins</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p>
          All four binned <span className="font-mono text-[0.9rem]">w(R)</span> rules meet the §9.6 R² ≥ 0.50 fidelity bar, yet they diverge across models
          (pre-registered §7 verdict: <strong>different mechanisms</strong>; mean between-model <span className="font-mono text-[0.9rem]">w(R)</span>{" "}
          L1 = 0.162 main / 0.174 no-retention). More importantly, none of the one-weight curves captures the RL planners&apos; surplus gains (B1&apos;s
          distilled curve loses {coins(Math.abs(mean(demo.distilled.planners[0]?.w_rule_minus_k_star_per_seed ?? [])))} coins to{" "}
          <span className="font-mono text-[0.9rem]">k*[B1]</span> while the planner gains{" "}
          {signedCoins(mean(demo.distilled.planners[0]?.planner_minus_k_star_per_seed ?? []))} coins): the neural policies encode dynamic history
          and horizon effects outside the 1-weight interpolating family.
        </p>
      </ArticleSection>

      <ArticleSection id="llm-judge" number="6" title="LLM Citizens & Judge Credibility (B4 & E1)">
        <p>
          As an out-of-family fifth population (evaluation-only, never trained against), Claude Haiku played both <strong>plain</strong> and{" "}
          <strong>type-calibrated</strong> (weights 0.7 / 0.0 / 0.3) roles in 40-game closed-loop exams under the frozen C2 protocol:
        </p>
        {b4Records.length > 0 && (
          <div className="my-4 overflow-x-auto rounded-sm border border-rule bg-sheet p-4">
            <table className="w-full min-w-[26rem] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-rule-strong text-xs text-graphite">
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    Variant
                  </th>
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    Regime
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Games
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Metrics in range
                  </th>
                  <th scope="col" className="py-1.5 font-medium">
                    C2 Label
                  </th>
                </tr>
              </thead>
              <tbody>
                {b4Records.map((r) => (
                  <tr key={`${r.variant}-${r.regime}`} className="border-b border-rule last:border-b-0">
                    <th scope="row" className="py-1.5 pr-3 font-normal text-ink">
                      {r.variant === "calibrated" ? "Calibrated" : "Uncalibrated (plain)"}
                    </th>
                    <td className="py-1.5 pr-3 font-mono text-xs text-graphite">{r.regime}</td>
                    <td className="numeric py-1.5 pr-3 text-right text-graphite">{r.games_complete}</td>
                    <td className="numeric py-1.5 pr-3 text-right text-ink">{r.metrics_in_range} of 4</td>
                    <td className={`py-1.5 text-xs ${r.counts_as_validated ? "font-medium text-valid" : "text-graphite"}`}>{r.label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p>
          Under frozen C4 protocol §9.4, <strong>B4-calibrated</strong> qualifies as a validated evaluation population for exploratory/confirmatory{" "}
          <span className="font-mono text-[0.9rem]">H2-B4</span> (only 1 of 10 C4 B4 planner-vs-LLM runs—<code className="font-mono text-[0.85rem]">citizens-c4-b4-calibrated-b1-v1</code> on CLI 2.1.306 under disclosure D37—is finished so far, so <span className="font-mono text-[0.9rem]">H2-B4</span> remains pending).
        </p>
        {demo.e1 !== null && (
          <div className="my-5 rounded-sm border border-rule bg-sheet p-5">
            <JudgeCheck e1={demo.e1} />
          </div>
        )}
      </ArticleSection>

      <ArticleSection id="limits" number="7" title="Scope Disclosure, Limitations & Provenance">
        <p>
          <strong>D35 scope disclosure:</strong> the initial study plan anticipated multiple closed-loop-validated models at G2. When only B1 passed
          closed-loop evaluation while all four models passed one-step likelihood, the C4 hypothesis framing was explicitly updated and frozen in{" "}
          <a className="prose-link" href={docUrl(PROTOCOL_DOC)} rel="noreferrer">
            protocol.md
          </a>{" "}
          before any RL planner was trained.
        </p>
        <p>
          <strong>Open gap toward Model Citizens II:</strong> because exploratory E1 shows B1, B3, and B23 all score 8/8 on scorable historical
          Wilcoxon pairs, historical pairwise ordering alone cannot validate new planners; testing whether policy rankings transfer to humans requires
          new multi-mechanism human data.
        </p>
        <ProvenanceBlock manifest={manifest} manifestUrl={manifestUrl} />
      </ArticleSection>

      <ArticleSection id="references" number="8" title="Verified Literature">
        <CitationList citations={CITIZENS_LITERATURE} />
      </ArticleSection>
    </ArticleBody>
  );
}

function LoadedReport({ runId, path }: { runId: string; path: string }) {
  const resource = useCitizensRun(runId, path);
  if (resource.status === "loading") return <LoadingState label="Loading and verifying Model Citizens data…" />;
  if (resource.status === "error") return <ErrorState error={resource.error} onRetry={resource.retry} />;
  return <ReportBody run={resource.value} />;
}

export function CitizensReportPage() {
  usePageTitle("Model Citizens Report – Phase Space");
  const catalog = useCatalog();
  const entry =
    catalog.status === "ready"
      ? catalog.value.studies.find((s) => s.study_id === STUDY_ID)?.runs.find((r) => r.kind === "citizens_demo")
      : undefined;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-8 sm:px-8">
      <ArticleHeader
        crumbs={[{ label: "Studies", to: "/" }, { label: "Model Citizens", to: "/citizens" }, { label: "Report" }]}
        title="Model Citizens v1 — Study Report & Literature"
        subtitle="Do RL policy designers trained against different validated models of people discover the same mechanism? An empirical test of the Lucas critique in a common-pool resource game."
        actions={
          <>
            <Link
              to="/citizens"
              className="inline-flex items-center rounded-sm bg-ink px-4 py-2 text-sm font-medium text-sheet no-underline hover:bg-canvas-raised"
            >
              Open interactive game demonstration
            </Link>
            <a className="prose-link inline-flex items-center gap-1.5 text-sm text-graphite" href={docUrl(REPORT_DOC)} rel="noreferrer">
              Full Markdown technical report
              <ExternalIcon className="size-3.5" />
            </a>
            <a className="prose-link inline-flex items-center gap-1.5 text-sm text-graphite" href={docUrl(PROTOCOL_DOC)} rel="noreferrer">
              Frozen protocol
              <ExternalIcon className="size-3.5" />
            </a>
          </>
        }
      />

      <div className="mt-8">
        {catalog.status === "loading" && <LoadingState label="Loading the catalog…" />}
        {catalog.status === "error" && <ErrorState error={catalog.error} onRetry={catalog.retry} />}
        {catalog.status === "ready" &&
          (entry === undefined ? (
            <p className="text-graphite">The Model Citizens demonstration artifact has not been published yet.</p>
          ) : (
            <LoadedReport runId={entry.run_id} path={entry.path} />
          ))}
      </div>
    </div>
  );
}
