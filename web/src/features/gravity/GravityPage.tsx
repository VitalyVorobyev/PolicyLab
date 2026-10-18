import { useState } from "react";
import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import { StatusBadge } from "../../components/StatusBadge";
import type { CatalogRun } from "../../lib/data/types";
import { useCatalog, useRun } from "../../lib/data/useResource";
import { docUrl } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { OrbitViewer } from "./OrbitViewer";
import { RunDetails } from "./RunDetails";

const FULL_QUESTION =
  "At matched compute, does a MAP-Elites archive of planar Earth–Moon free-return trajectories contain more solutions " +
  "that stay feasible under unseen injection errors and changed constraints than multi-start search, basin hopping, " +
  "differential evolution, a continuation-family catalog and a dense quasi-random map?";

const SHORT_QUESTION =
  "At matched compute, do diverse archives of Earth–Moon free-returns survive unseen errors better than strong optimizers?";

const REPORT_STATUS = "Research report forthcoming, after the comparison is run.";

function SpecLink() {
  return (
    <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/studies/gravity-atlas/README.md")} rel="noreferrer">
      Full study specification
      <ExternalIcon className="size-3.5" />
    </a>
  );
}

const STAGES: { name: string; state: "done" | "current" | "next"; text: string }[] = [
  { name: "Literature scan", state: "done", text: "Nearest prior work identified; question reframed." },
  { name: "Physics core", state: "done", text: "Planar CR3BP integrator validated against the JPL catalog." },
  { name: "Public viewer", state: "current", text: "This page: validated data rendered from static files." },
  { name: "Free-return feasibility", state: "next", text: "Find and validate a free-return candidate, or report that none was found." },
  { name: "Comparative study", state: "next", text: "Frozen protocol, matched budgets, several seeds, held-out perturbations." },
];

function StudyStages() {
  return (
    <ol className="space-y-3">
      {STAGES.map((stage, i) => (
        <li key={stage.name} className="grid grid-cols-[1.75rem_1fr] gap-x-2">
          <span
            className={`numeric mt-0.5 grid size-6 place-items-center rounded-full border text-xs ${
              stage.state === "done"
                ? "border-ink bg-ink text-sheet"
                : stage.state === "current"
                  ? "border-accent-ink bg-accent text-ink"
                  : "border-rule-strong text-graphite"
            }`}
            aria-hidden="true"
          >
            {i + 1}
          </span>
          <div>
            <p className={`font-medium ${stage.state === "next" ? "text-graphite" : ""}`}>
              {stage.name}
              <span className="sr-only">
                {stage.state === "done" ? " (done)" : stage.state === "current" ? " (current)" : " (not started)"}
              </span>
            </p>
            <p className="text-sm text-graphite">{stage.text}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

function RunView({ run }: { run: CatalogRun }) {
  const resource = useRun(run.run_id, run.path);
  if (resource.status === "loading") {
    return (
      <div className="grid min-h-[22rem] place-items-center rounded-sm bg-canvas text-canvas-text">
        <LoadingState label="Loading and verifying the trajectory…" />
      </div>
    );
  }
  if (resource.status === "error") {
    return (
      <div className="rounded-sm bg-canvas p-6 sm:p-10">
        <ErrorState error={resource.error} onRetry={resource.retry} tone="canvas" />
      </div>
    );
  }
  const loaded = resource.value;
  return (
    <>
      <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 sm:mb-5">
        <h2 className="text-[1.35rem] sm:text-3xl">{run.label}</h2>
        <StatusBadge status={loaded.manifest.status} />
      </div>
      {/* Keyed by run so that switching trajectories resets playback predictably. */}
      <OrbitViewer key={loaded.manifest.run_id} run={loaded} />

      <section aria-labelledby="scope-title" className="mt-16 grid gap-10 border-t border-rule pt-8 lg:grid-cols-12 lg:gap-12">
        <h2 id="scope-title" className="sr-only">
          What this is, what it is not, and where the study stands
        </h2>
        <div className="lg:col-span-4">
          <h3 className="text-2xl">What this is</h3>
          <p className="mt-3 max-w-prose">
            A reference periodic orbit around the Earth–Moon L1 point, computed from first principles by this project’s
            integrator and matched against the NASA/JPL periodic-orbit catalog. It shows that the physics engine is
            trustworthy before the study relies on it.
          </p>
        </div>
        <div className="lg:col-span-4">
          <h3 className="text-2xl">What this is not</h3>
          <p className="mt-3 max-w-prose">
            It is not a free-return trajectory, not an optimization result, and not evidence for or against the research
            question. Free-return candidates and method comparisons will appear here once they are computed and validated.
          </p>
        </div>
        <div className="lg:col-span-4">
          <h3 className="text-2xl">Where the study stands</h3>
          <div className="mt-4">
            <StudyStages />
          </div>
        </div>
      </section>

      <section aria-labelledby="evidence-title" className="mt-16">
        <h2 id="evidence-title" className="text-3xl">
          Evidence and provenance
        </h2>
        <p className="mt-3 mb-8 max-w-prose text-graphite">Everything below is read from the run’s manifest.</p>
        <RunDetails run={loaded} />
      </section>
    </>
  );
}

export function GravityPage() {
  usePageTitle("Gravity Atlas – Phase Space");
  const catalog = useCatalog();
  const [selected, setSelected] = useState(0);

  const study = catalog.status === "ready" ? catalog.value.studies.find((s) => s.study_id === "gravity-atlas") : undefined;
  const runs = study?.runs ?? [];
  const run = runs[Math.min(selected, runs.length - 1)];

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-8">
      <header className="grid gap-x-12 gap-y-4 pt-7 pb-7 sm:gap-y-5 sm:pt-14 sm:pb-10 lg:grid-cols-12">
        <div className="lg:col-span-4">
          <nav aria-label="Breadcrumb" className="text-graphite">
            <Link to="/" className="prose-link">
              Studies
            </Link>
            <span aria-hidden="true" className="mx-2 text-quiet">
              /
            </span>
            <span aria-current="page">Gravity Atlas</span>
          </nav>
          <h1 className="mt-1 text-[2.5rem] sm:mt-2 sm:text-6xl">Gravity Atlas</h1>
        </div>
        <div className="lg:col-span-8 lg:pt-9">
          {/* Small screens: a one-sentence lead with the full question one tap away, so the viewer is reachable sooner. */}
          <div className="md:hidden">
            <p className="font-serif text-[1.075rem] leading-relaxed">{SHORT_QUESTION}</p>
            <details className="mt-2 text-[0.95rem] text-graphite">
              <summary className="cursor-pointer marker:text-quiet hover:text-ink">Full research question and status</summary>
              <p className="mt-2 font-serif text-base leading-relaxed text-ink">{FULL_QUESTION}</p>
              <p className="mt-2">{REPORT_STATUS}</p>
            </details>
            <p className="mt-2 text-[0.95rem] text-graphite">
              <SpecLink />
            </p>
          </div>
          <div className="max-md:hidden">
            <p className="max-w-[62ch] font-serif text-[1.3rem] leading-relaxed">{FULL_QUESTION}</p>
            <p className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-graphite">
              <SpecLink />
              <span>{REPORT_STATUS}</span>
            </p>
          </div>
        </div>
      </header>

      <section aria-label="Trajectory viewer">
        {catalog.status === "loading" && <LoadingState label="Loading the research catalog…" />}
        {catalog.status === "error" && <ErrorState error={catalog.error} onRetry={catalog.retry} />}
        {catalog.status === "ready" && run === undefined && (
          <p className="max-w-prose border-l-2 border-rule-strong py-2 pl-4">
            No trajectories have been published for this study yet. Only computed runs appear here, so this space stays
            empty until one exists.
          </p>
        )}
        {runs.length > 1 && (
          <label className="mb-6 flex flex-wrap items-center gap-3">
            <span className="font-medium">Trajectory</span>
            <select
              value={selected}
              onChange={(event) => { setSelected(Number(event.currentTarget.value)); }}
              className="rounded-sm border border-rule-strong bg-sheet px-2 py-1.5"
            >
              {runs.map((r, i) => (
                <option key={r.run_id} value={i}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
        )}
        {run !== undefined && <RunView key={run.run_id} run={run} />}
      </section>
    </div>
  );
}
