import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import { Term } from "../../components/Tooltip";
import { useAtlasRun, useCatalog } from "../../lib/data/useResource";
import { docUrl, REPOSITORY_URL } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { ARM_INFO } from "./atlasModel";
import { Demo } from "./Demo";
import { Findings } from "./Findings";

const PROBLEM =
  "A free return is a path that swings around the Moon and falls back to Earth with no engine burn after launch: the safety net Apollo 13 came home on. " +
  "There is not one free return but a family: passing the Moon anywhere from 100 to 30,000 km up and coming home after 5 to 10 days. " +
  "A mission designer wants a catalog with a good trajectory of every kind, and ones that still work when the launch burn is slightly off. " +
  "We asked which search method builds the best catalog from the same budget: 16,384 simulated launches each.";

function Section({ title, children, id }: { title: string; children: React.ReactNode; id: string }) {
  return (
    <section aria-labelledby={id} className="mt-16 sm:mt-20">
      <h2 id={id} className="mb-6 text-2xl sm:text-3xl">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Demonstration() {
  const catalog = useCatalog();
  const entry = catalog.status === "ready" ? catalog.value.studies.find((s) => s.study_id === "gravity-atlas")?.runs.find((r) => r.kind === "atlas") : undefined;
  return catalog.status === "ready" && entry !== undefined ? <LoadedDemo runId={entry.run_id} path={entry.path} /> : <DemoPlaceholder catalog={catalog} />;
}

function DemoPlaceholder({ catalog }: { catalog: ReturnType<typeof useCatalog> }) {
  return (
    <div className="grid min-h-[30rem] place-items-center rounded-sm bg-canvas p-6 text-canvas-text">
      {catalog.status === "loading" && <LoadingState label="Loading the atlas…" />}
      {catalog.status === "error" && <ErrorState error={catalog.error} onRetry={catalog.retry} tone="canvas" />}
      {catalog.status === "ready" && <p>The atlas has not been published yet.</p>}
    </div>
  );
}

function LoadedDemo({ runId, path }: { runId: string; path: string }) {
  const resource = useAtlasRun(runId, path);
  if (resource.status === "loading") {
    return (
      <div className="grid min-h-[30rem] place-items-center rounded-sm bg-canvas text-canvas-text">
        <LoadingState label="Loading and checking the atlas…" />
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
  return (
    <>
      <Demo run={resource.value} runPath={path} />
      <Section id="found" title="What we found">
        <Findings atlas={resource.value.atlas} />
      </Section>
    </>
  );
}

export function GravityPage() {
  usePageTitle("Gravity Atlas – Phase Space");
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
            <span aria-current="page">Gravity Atlas</span>
          </nav>
          <h1 className="mt-1 text-[2.5rem] sm:text-5xl">Gravity Atlas</h1>
          <p className="mt-2 max-w-[36ch] font-serif text-[1.3rem] leading-snug sm:text-[1.5rem]">
            Which search method finds the free returns that survive a bad launch?
          </p>
        </div>
        {/* On wide screens the problem sits beside the title; on phones it moves below the demonstration. */}
        <p className="max-w-[78ch] text-[0.925rem] leading-relaxed text-graphite max-lg:hidden lg:col-span-7 lg:pt-6">{PROBLEM}</p>
      </header>

      <Demonstration />

      <section aria-labelledby="problem" className="mt-16 sm:mt-20 lg:hidden">
        <h2 id="problem" className="mb-4 text-2xl sm:text-3xl">
          The problem
        </h2>
        <p className="max-w-[62ch] text-graphite">{PROBLEM}</p>
      </section>

      <Section id="did" title="What we did">
        <div className="grid gap-x-12 gap-y-10 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <p className="text-graphite">
              Seven methods searched the same launch settings (where on a 185 km parking orbit to fire, how hard, at what angle) in a planar Earth–Moon{" "}
              <Term tip="A spacecraft too light to pull on anything, attracted by the Earth and the Moon, which circle their common centre; everything stays in one plane. The demonstration draws it in the frame that turns with the Moon.">
                three-body model
              </Term>
              . Each kept its{" "}
              <Term tip="Best means the entry angle back into the atmosphere closest to −6.5°.">best trajectory</Term> in every square of the atlas. Each kept trajectory was hit with 50 small launch errors in burn size, direction and timing, with one correction of up to 1 m/s allowed, and
              checked against 62 mission requirements none of the methods had seen. Every method ran 16 times; scores are averages. The protocol was{" "}
              <Term tip="Budgets, seeds, scores and statistical tests were written down and frozen before any comparison run, so nothing could be tuned to the outcome.">
                frozen before any of these runs
              </Term>
              .
            </p>
          </div>
          <div className="lg:col-span-7">
            <h3 className="sr-only">The seven methods</h3>
            <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
              {ARM_INFO.map((m) => (
                <div key={m.id}>
                  <dt className="font-medium">{m.name}</dt>
                  <dd className="text-[0.95rem] text-graphite">{m.line}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </Section>

      <p className="mt-16 text-sm text-graphite">
        <a className="prose-link inline-flex items-center gap-1" href={REPOSITORY_URL} rel="noreferrer">
          Code
          <ExternalIcon className="size-3.5" />
        </a>
        <span aria-hidden="true" className="mx-2 text-quiet">
          ·
        </span>
        <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/studies/gravity-atlas/protocol.md")} rel="noreferrer">
          Full protocol
          <ExternalIcon className="size-3.5" />
        </a>
        <span aria-hidden="true" className="mx-2 text-quiet">
          ·
        </span>
        <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/decisions.md")} rel="noreferrer">
          Decision log
          <ExternalIcon className="size-3.5" />
        </a>
      </p>
    </div>
  );
}
