import { useState, type ReactNode } from "react";
import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import { Term } from "../../components/Tooltip";
import type { CitizensDemo, LoadedCitizensRun } from "../../lib/data/citizens";
import { useCatalog, useCitizensRun } from "../../lib/data/useResource";
import { docUrl, REPOSITORY_URL } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { CrossplayHeatmap, Verdicts } from "./Crossplay";
import { crossplayFindings, distillCaveat, narrowPass, wholeGameModels } from "./findings";
import { Exams, examsHeadline } from "./Exams";
import { DEFAULT_SELECTION, GameDemo, validSelection, type Selection } from "./GameDemo";
import { GameExplainer } from "./GameExplainer";
import { JudgeCheck } from "./JudgeCheck";
import { prefersReducedMotion } from "./hooks";
import { LlmCitizens } from "./LlmCitizens";
import { MechanismChart } from "./MechanismChart";
import { cellOf, CITIZENS, MODEL_KEYS, signedPercent } from "./model";
import { TuneRule } from "./TuneRule";

const STUDY_ID = "model-citizens";
const DOI_URL = "https://doi.org/10.1038/s41467-025-58043-7";

const PROBLEM =
  "Four players share a pool of coins for 40 rounds. A governor decides how much of the pool to offer each player; players keep part and return the rest, which grows and refills the pool. " +
  "DeepMind trained an AI governor against simulated players and distilled a simple rule that worked with real people. " +
  "We asked what happens when the simulated players change: if you train AI governors against different models of people, each built from the same human games, do they find the same rule, and does it still work on the others?";

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

function Placeholder({ children }: { children: ReactNode }) {
  return <div className="grid min-h-[30rem] place-items-center rounded-sm bg-canvas p-6 text-canvas-text">{children}</div>;
}

function caveats(demo: CitizensDemo): { title: string; body: ReactNode }[] {
  const judges = wholeGameModels(demo.exams);
  const judge = judges.length === 1 ? judges[0] : undefined;
  const judgeOther = demo.exams?.rows.filter((r) => r.model === judge && r.regime !== "R-I" && (demo.exams?.closed_loop_tier[r.regime] ?? false)) ?? [];
  const learning = (["B3", "B23"] as const).map((m) => ({ m, home: cellOf(demo, `Planner[${m}]`, m) }));
  const narrow = judge === undefined ? null : narrowPass(demo.exams, judge);
  return [
    {
      title: "Judged by simulators, not people.",
      body: "Every AI-governor result here is measured on simulated citizens. None of these governors has played with real people.",
    },
    {
      title: judge === undefined ? "No model passed every exam." : `${judge} is the best available judge, and it passed narrowly.`,
      body:
        judge === undefined ? (
          "Read each column as what would happen if people behaved like that model."
        ) : (
          <>
            It is the only model that{" "}
            {narrow === null ? (
              "passed the whole-game exam"
            ) : (
              <Term tip={`Under the simple rule: ${narrow}`}>passed the whole-game exam</Term>
            )}
            {judgeOther.length > 0 && judgeOther.every((r) => !r.pass_outcomes) ? ", and it failed that exam under the second unseen rule" : ""}. The other
            columns show what would happen if people behaved like those models.
          </>
        ),
    },
    {
      title: "Home gains on the learning models look like exploitation.",
      body: (
        <>
          The governors trained on B3 and B23 gain the most at home
          {learning.every((l) => l.home !== undefined) && <> ({learning.map((l) => `${signedPercent(l.home?.vs_k_star.relative ?? 0)} on ${l.m}`).join(", ")})</>} yet
          lose on {judge ?? "B1"}: they may have learned quirks of the learning models rather than of people.
        </>
      ),
    },
    {
      title: "Next: do the rankings hold on people?",
      body: "The open question is whether the order in which each model ranks governors matches the order on new human data. That test needs a new dataset.",
    },
  ];
}

function Content({ run }: { run: LoadedCitizensRun }) {
  const { demo } = run;
  const [selection, setSelection] = useState<Selection>(() => validSelection(demo, DEFAULT_SELECTION));
  const [nonce, setNonce] = useState(0);
  const pick = (policy: string, column: string): void => {
    setSelection(validSelection(demo, { governor: policy, citizens: column, game: 0 }));
    setNonce((n) => n + 1);
    document.getElementById("demo")?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  };
  const findings = crossplayFindings(demo);
  const distill = distillCaveat(demo.distilled);
  const examLine = demo.exams === null ? null : examsHeadline(demo.exams);
  const hasModels = MODEL_KEYS.every((m) => CITIZENS.some((c) => c.key === m));

  return (
    <>
      <div id="demo" className="scroll-mt-4">
        <GameDemo key={nonce} demo={demo} selection={selection} onSelect={setSelection} />
      </div>

      <section aria-labelledby="problem" className="mt-12 lg:hidden">
        <h2 id="problem" className="mb-3 text-2xl">
          The problem
        </h2>
        <p className="max-w-[62ch] text-graphite">{PROBLEM}</p>
      </section>

      <Section id="game" title="The game in 30 seconds">
        <GameExplainer game={demo.game} />
      </Section>

      {demo.exams !== null && hasModels && (
        <Section
          id="exams"
          title="Four models of people, two exams"
          lead={
            <>
              <p>
                Each model was fitted to the same human games under simple rules, then examined on games under rules it had never seen. Exam 1 asks
                whether it predicts each player&apos;s next move; exam 2 whether whole simulated games look like real ones.
              </p>
              {examLine !== null && <p className="mt-3 font-medium text-ink">{examLine}</p>}
            </>
          }
        >
          <Exams exams={demo.exams} />
        </Section>
      )}

      <Section
        id="crossplay"
        title="Every governor against every model"
        lead={
          <>
            <p>
              An AI governor was trained against each model, then every governor played {demo.crossplay.games.toLocaleString("en-US")} games against
              every model. Click a cell to watch it above.
            </p>
            {demo.hypotheses === null && (
              <p className="mt-2 text-sm text-quiet">
                <Term tip="These are averages over simulated games. The pre-registered statistical tests (with their minimum effect and multiple-comparison corrections) decide the hypotheses; their verdicts will be added here.">
                  Descriptive results
                </Term>
                ; the pre-registered verdicts are being computed.
              </p>
            )}
          </>
        }
      >
        {demo.hypotheses !== null && (
          <div className="mb-8">
            <Verdicts h={demo.hypotheses} />
          </div>
        )}
        <div className="grid grid-cols-1 gap-x-12 gap-y-12 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <CrossplayHeatmap demo={demo} onPick={pick} />
          </div>
          <div className="lg:col-span-5">
            <h3 className="font-serif text-xl">Did they learn the same rule?</h3>
            <p className="mt-2 mb-4 text-[0.95rem] text-graphite">
              Described with the simple rule&apos;s own ingredients, the four AI governors split the pool in visibly different ways.
            </p>
            <MechanismChart distilled={demo.distilled} maxPool={demo.game.max_pool} />
            {distill !== null && (
              <p className="mt-3 border-l-2 border-caution pl-3 text-[0.9rem] text-graphite">
                <Term tip={distill.tip}>But these curves explain the AI governors poorly</Term>: they recover little or none of the governors&apos; gain over the simple
                rule, so the governors do something the curves don&apos;t capture.
              </p>
            )}
          </div>
        </div>
        {findings.length > 0 && (
          <ul className="mt-12 grid gap-x-10 gap-y-6 text-[0.95rem] text-graphite sm:grid-cols-2">
            {findings.map((f) => (
              <li key={f.lead} className="border-t border-rule-strong pt-3">
                <p className="font-medium text-ink">{f.tip === undefined ? f.lead : <Term tip={f.tip}>{f.lead}</Term>}</p>
                <p className="mt-1">{f.text}</p>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section
        id="tune"
        title="Tune the rule yourself"
        lead={
          <p>
            The simple rule has one setting, k: the fuller the pool relative to its cap, the more of it is split equally, and k sets how full is full.
            Move k and see how each model of people responds.
          </p>
        }
      >
        <TuneRule tuning={demo.k_tuning} />
      </Section>

      {(demo.b4_c2 !== null || demo.b4_rows !== null) && (
        <Section id="llm" title="LLM players">
          <LlmCitizens demo={demo} />
        </Section>
      )}

      <Section id="limits" title="What this does and doesn’t show">
        {demo.e1 !== null && (
          <div className="mb-10">
            <JudgeCheck e1={demo.e1} />
          </div>
        )}
        <ul className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
          {caveats(demo).map((c) => (
            <li key={c.title} className="border-t border-rule-strong pt-3">
              <p className="font-medium">{c.title}</p>
              <p className="mt-1 text-[0.95rem] text-graphite">{c.body}</p>
            </li>
          ))}
        </ul>
      </Section>

      <footer className="mt-16 border-t border-rule pt-6 text-sm text-graphite">
        <p className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <Link className="prose-link font-medium text-ink" to="/citizens/report">
            Research report &amp; references
          </Link>
          <a className="prose-link inline-flex items-center gap-1 text-graphite" href={docUrl("docs/studies/model-citizens/report.md")} rel="noreferrer">
            Technical report (Markdown)
            <ExternalIcon className="size-3.5" />
          </a>
          <span>
            Human data: Koster et al. 2025,{" "}
            <a className="prose-link" href={DOI_URL} rel="noreferrer">
              Nature Communications 16:2824
            </a>
            , CC BY 4.0.
          </span>
        </p>
        <p className="mt-2 text-xs text-quiet">
          <Term tip={`Status: computed, not independently validated. The page shows ${run.manifest.experiment_id} as published, verified against its checksums in your browser.`}>
            Reproducible
          </Term>
          : every number comes from recorded runs with their code commit, configuration, seeds and checksums,{" "}
          <a className="prose-link" href={REPOSITORY_URL} rel="noreferrer">
            in the repository
          </a>
          .
        </p>
      </footer>
    </>
  );
}

function Loaded({ runId, path }: { runId: string; path: string }) {
  const resource = useCitizensRun(runId, path);
  if (resource.status === "loading") {
    return (
      <Placeholder>
        <LoadingState label="Loading and checking the games…" />
      </Placeholder>
    );
  }
  if (resource.status === "error") {
    return (
      <div className="rounded-sm bg-canvas p-6 sm:p-10">
        <ErrorState error={resource.error} onRetry={resource.retry} tone="canvas" />
      </div>
    );
  }
  return <Content run={resource.value} />;
}

function Body() {
  const catalog = useCatalog();
  if (catalog.status !== "ready") {
    return <Placeholder>{catalog.status === "loading" ? <LoadingState label="Loading the catalog…" /> : <ErrorState error={catalog.error} onRetry={catalog.retry} tone="canvas" />}</Placeholder>;
  }
  const entry = catalog.value.studies.find((s) => s.study_id === STUDY_ID)?.runs.find((r) => r.kind === "citizens_demo");
  if (entry === undefined) return <Placeholder>The games have not been published yet.</Placeholder>;
  return <Loaded runId={entry.run_id} path={entry.path} />;
}

export function CitizensPage() {
  usePageTitle("Model Citizens – Phase Space");
  return (
    <div className="mx-auto max-w-7xl px-4 pb-8 sm:px-8">
      <header className="grid gap-x-12 gap-y-3 pt-6 pb-5 sm:pt-8 sm:pb-6 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <nav aria-label="Breadcrumb" className="text-sm text-graphite">
            <Link to="/" className="prose-link">
              Studies
            </Link>
            <span aria-hidden="true" className="mx-2 text-quiet">
              /
            </span>
            <span aria-current="page">Model Citizens</span>
          </nav>
          <h1 className="mt-1 text-[2.5rem] sm:text-5xl">Model Citizens</h1>
          <p className="mt-2 max-w-[36ch] font-serif text-[1.3rem] leading-snug sm:text-[1.5rem]">
            Do AI governors trained on different models of people learn the same rule?
          </p>
          <p className="mt-3 text-sm text-graphite">
            <Link to="/citizens/report" className="prose-link font-medium text-ink">
              Read full report &amp; references
            </Link>
          </p>
        </div>
        <p className="max-w-[78ch] text-[0.925rem] leading-relaxed text-graphite max-lg:hidden lg:col-span-7 lg:pt-6">{PROBLEM}</p>
      </header>

      <Body />
    </div>
  );
}
