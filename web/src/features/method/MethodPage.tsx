import { useState } from "react";
import { ArticleBody, ArticleHeader, ArticleSection, Callout, CitationList, ProvenanceBlock, StatusBadge, type TocItem } from "../../components/Article";
import { ErrorState, LoadingState } from "../../components/DataState";
import { CheckCircleIcon, ExternalIcon } from "../../components/icons";
import { loadAnyManifest, type AnyManifest } from "../../lib/data/load";
import type { CatalogRun } from "../../lib/data/types";
import { useCatalog } from "../../lib/data/useResource";
import { LAST_LIBRARY_LITERATURE } from "../../lib/literature";
import { docUrl } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";

const TOC: readonly TocItem[] = [
  { id: "principles", label: "Research standards" },
  { id: "architecture", label: "Static-first architecture" },
  { id: "ledger", label: "Live artifact & checksum inspector" },
  { id: "reproduce", label: "Local reproduction" },
  { id: "latest-study", label: "Latest published study: The Last Library" },
];

type InspectEntry =
  | { status: "loading" }
  | { status: "ready"; manifest: AnyManifest; manifestUrl: string }
  | { status: "error"; error: unknown };

function ArtifactLedger() {
  const catalog = useCatalog();
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [entries, setEntries] = useState<Record<string, InspectEntry>>({});

  const fetchManifest = (run: CatalogRun): void => {
    setEntries((prev) => ({ ...prev, [run.run_id]: { status: "loading" } }));
    void loadAnyManifest(run.path, run.kind, undefined, run.run_id).then(
      ({ manifest, manifestUrl }) => {
        setEntries((prev) => ({ ...prev, [run.run_id]: { status: "ready", manifest, manifestUrl } }));
      },
      (error: unknown) => {
        setEntries((prev) => ({ ...prev, [run.run_id]: { status: "error", error } }));
      },
    );
  };

  const toggle = (run: CatalogRun): void => {
    const isOpen = open[run.run_id] === true;
    if (isOpen) {
      setOpen((prev) => ({ ...prev, [run.run_id]: false }));
      return;
    }
    setOpen((prev) => ({ ...prev, [run.run_id]: true }));
    if (entries[run.run_id]?.status !== "ready") {
      fetchManifest(run);
    }
  };

  if (catalog.status === "loading") return <LoadingState label="Loading the published catalog…" />;
  if (catalog.status === "error") return <ErrorState error={catalog.error} onRetry={catalog.retry} />;

  return (
    <div className="space-y-6">
      {catalog.value.studies.map((study) => (
        <div key={study.study_id} className="rounded-sm border border-rule bg-sheet p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-rule pb-3">
            <h3 className="font-serif text-xl text-ink">{study.title}</h3>
            <span className="text-xs font-medium tracking-wide text-quiet uppercase">
              {study.status} · {study.runs.length} published {study.runs.length === 1 ? "run" : "runs"}
            </span>
          </div>

          {study.runs.length === 0 ? (
            <p className="mt-3 text-sm text-graphite">No computed runs published yet.</p>
          ) : (
            <ul className="mt-3 divide-y divide-rule">
              {study.runs.map((run) => {
                const isOpen = open[run.run_id] === true;
                const state = entries[run.run_id];
                return (
                  <li key={run.run_id} className="py-3 first:pt-1 last:pb-1">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="font-mono text-sm font-medium break-all text-ink">{run.run_id}</p>
                        <p className="mt-0.5 text-sm text-graphite">{run.label}</p>
                        <p className="mt-1 font-mono text-xs text-quiet">
                          kind: {run.kind} · path: /research/{run.path}
                        </p>
                      </div>
                      <button
                        type="button"
                        aria-expanded={isOpen}
                        onClick={() => {
                          toggle(run);
                        }}
                        className="shrink-0 rounded-sm border border-rule-strong bg-paper px-3 py-1.5 text-xs font-medium text-ink hover:border-ink"
                      >
                        {isOpen ? "Hide manifest" : "Inspect manifest & checks"}
                      </button>
                    </div>
                    {isOpen && state !== undefined && (
                      <div className="mt-4">
                        {state.status === "loading" && <LoadingState label={`Loading manifest for ${run.run_id}…`} />}
                        {state.status === "error" && (
                          <ErrorState
                            error={state.error}
                            onRetry={() => {
                              fetchManifest(run);
                            }}
                          />
                        )}
                        {state.status === "ready" && <ProvenanceBlock manifest={state.manifest} manifestUrl={state.manifestUrl} />}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
}

export function MethodPage() {
  usePageTitle("Method & Documentation – Phase Space");

  return (
    <div className="mx-auto max-w-7xl px-4 pb-8 sm:px-8">
      <ArticleHeader
        crumbs={[{ label: "Studies", to: "/" }, { label: "Method & Documentation" }]}
        title="How Phase Space runs & publishes research"
        subtitle="Falsifiable questions, pre-registered frozen protocols, immutable artifacts verified in your browser, and the studies coming next."
        actions={
          <>
            <Link to="/citizens/report" className="prose-link text-sm font-medium text-ink">
              Model Citizens report
            </Link>
            <Link to="/library/report" className="prose-link text-sm font-medium text-ink">
              The Last Library report
            </Link>
            <Link to="/gravity/report" className="prose-link text-sm font-medium text-ink">
              Gravity Atlas report
            </Link>
          </>
        }
      />

      <div className="mt-8">
        <ArticleBody toc={TOC}>
          <ArticleSection id="principles" number="1" title="Research Standards & Verification Tiers">
            <p>
              Phase Space is designed around small, self-contained computational studies. Every comparative experiment defines a falsifiable question,
              searches nearest prior literature with verified DOIs or arXiv records before claiming novelty, freezes hyperparameters, seeds, budgets,
              and statistical corrections in a committed protocol before evaluating held-out data or running comparisons, and publishes null or negative
              outcomes as plainly as positive ones.
            </p>
            <div className="grid gap-4 pt-2 sm:grid-cols-3">
              <div className="rounded-sm border border-rule bg-sheet p-4">
                <StatusBadge status="validated" />
                <p className="mt-2 text-sm text-graphite">
                  All automated numerical or held-out checks recorded in <code className="font-mono text-xs">validation.checks</code> passed against
                  pre-set thresholds (and independent references where applicable).
                </p>
              </div>
              <div className="rounded-sm border border-rule bg-sheet p-4">
                <StatusBadge status="computed_unvalidated" />
                <p className="mt-2 text-sm text-graphite">
                  Deterministic artifact computed by recorded code and seeds, verified by reproduction or internal checks, without an independent
                  external ground truth.
                </p>
              </div>
              <div className="rounded-sm border border-rule bg-sheet p-4">
                <StatusBadge status="illustrative" />
                <p className="mt-2 text-sm text-graphite">
                  Qualitative schematic or placeholder only; never mixed into comparative tables or scientific verdicts.
                </p>
              </div>
            </div>
          </ArticleSection>

          <ArticleSection id="architecture" number="2" title="Static-First Architecture & Data Contract">
            <p>
              The browser renders and verifies precomputed science; it never runs authoritative optimization or simulation on the fly. All experiments
              run locally in deterministic Python, writing immutable directories under <code className="font-mono text-xs">runs/</code> containing a
              schema-v1 <code className="font-mono text-xs">manifest.json</code> (recording commit SHA, dirty flag, configuration hash, solver/model
              settings, seeds, limitations, and artifact SHA-256 digests).
            </p>
            <p>
              Curated public subsets are copied into <code className="font-mono text-xs">web/public/research/</code> by{" "}
              <code className="font-mono text-xs">web/scripts/publish-run.mjs</code>, which refuses uncommitted source trees unless explicitly flagged
              for temporary development. On every page load, the frontend computes each file&apos;s SHA-256 digest in Web Crypto and validates schema
              invariants before drawing a pixel.
            </p>
          </ArticleSection>

          <ArticleSection id="ledger" number="3" title="Live Artifact & Checksum Inspector">
            <p>
              Below is the live <code className="font-mono text-xs">catalog.json</code> served with this deployment. Inspect any published run to
              fetch and validate its <code className="font-mono text-xs">manifest.json</code>, git commit provenance, SHA-256 artifact checksums, and
              automated validation checks.
            </p>
            <ArtifactLedger />
          </ArticleSection>

          <ArticleSection id="reproduce" number="4" title="Local Reproduction & Primary Documents">
            <p>
              The full codebase and every frozen protocol live in the public repository. Standard project checks and study pipelines run via{" "}
              <code className="font-mono text-xs">just</code>:
            </p>
            <pre className="overflow-x-auto rounded-sm bg-canvas p-4 font-mono text-xs leading-relaxed text-canvas-strong">
              <code>{`# Full verification suite (Python lint, format, typecheck, pytest + Web lint, tsc, Vitest, build)
just verify

# Gravity Atlas M1 reference orbit & M4 public atlas export
just gravity-reference
just gravity-m4-demo-export --confirm-frozen --workers 10

# Model Citizens hypothesis verdicts & interactive web demo export
just citizens-c4-hypotheses citizens-c4-hypotheses-v1 --confirm-frozen ...
just citizens-web-export citizens-web-demo-dev4 --confirm-frozen ...

# The Last Library feasibility gate & frozen confirmatory sweep
just library-feasibility library-feasibility-dev2
just library-run library-main-v1 --confirm-frozen --workers 8

# Start local interactive site preview
just web-dev`}</code>
            </pre>
            <ul className="flex flex-wrap gap-x-6 gap-y-2 pt-2 text-sm text-graphite">
              <li>
                <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/research-methodology.md")} rel="noreferrer">
                  Research methodology
                  <ExternalIcon className="size-3.5" />
                </a>
              </li>
              <li>
                <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/data-contract.md")} rel="noreferrer">
                  Data contract (schema v1)
                  <ExternalIcon className="size-3.5" />
                </a>
              </li>
              <li>
                <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/system-design.md")} rel="noreferrer">
                  System design
                  <ExternalIcon className="size-3.5" />
                </a>
              </li>
              <li>
                <a className="prose-link inline-flex items-center gap-1" href={docUrl("docs/decisions.md")} rel="noreferrer">
                  Decision log
                  <ExternalIcon className="size-3.5" />
                </a>
              </li>
            </ul>
          </ArticleSection>

          <ArticleSection id="latest-study" number="5" title="Latest Published Study: The Last Library (Complete)">
            <Callout title="Published &amp; Validated — confirmatory verdicts across 0–60 bits" tone="ink">
              <p className="flex items-center gap-2">
                <CheckCircleIcon className="shrink-0 text-valid" />
                <span>
                  The Last Library confirmatory sweep (<code className="font-mono text-xs">library-main-v1</code>, 200 stationary &amp; 200 rule-change
                  lineages) is validated and published. It documents a sharp 19-bit dense rule capacity cliff (H1 at B=20), sender confidence
                  backfiring under rule change (H3a refuted), and 30% receiver hazard hedging cutting post-change overshoot (H3b).{" "}
                  <strong>Algorithm Forge</strong> remains the next planned study.
                </span>
              </p>
            </Callout>
            <p>
              <strong>Sharpened research question (Q1).</strong> The original v0 concept asked whether an inherited record up to 256 bytes enables
              cumulative learning. The literature scan showed that Prystawski, Arumugam &amp; Goodman (2023, 2025) already established that abstract
              rules outperform concrete examples on fixed stationary chains and characterized the nonlinear channel threshold in bits—while noting that
              256 bytes (2,048 bits) would fail to bind in a small gridworld.
            </p>
            <p>
              The published experiment sweeps the inherited record capacity <em>in exact bits</em> around the 16-bit latent entropy (0 to 60 bits),
              evaluates every fresh generation on <strong>held-out grid layouts</strong> governed by the latent rule family, compares rule records
              against pedagogically selected example records and shuffled/random placebos, and tests both sender confidence weights and receiver hazard
              hedging when latent rules invert unannounced at generation 5.
            </p>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
              <Link to="/library" className="rounded-sm bg-ink px-4 py-2 font-medium text-sheet no-underline hover:bg-canvas-raised">
                Open interactive workbench
              </Link>
              <Link to="/library/report" className="prose-link font-medium text-ink">
                Read full study report
              </Link>
              <a className="prose-link inline-flex items-center gap-1 text-graphite" href={docUrl("docs/studies/last-library/protocol.md")} rel="noreferrer">
                Frozen protocol (D39)
                <ExternalIcon className="size-3.5" />
              </a>
              <a className="prose-link inline-flex items-center gap-1 text-graphite" href={docUrl("docs/studies/last-library/report.md")} rel="noreferrer">
                Markdown report
                <ExternalIcon className="size-3.5" />
              </a>
            </div>
            <div className="pt-4">
              <h3 className="mb-4 font-serif text-xl text-ink">Verified literature</h3>
              <CitationList citations={LAST_LIBRARY_LITERATURE} />
            </div>
          </ArticleSection>
        </ArticleBody>
      </div>
    </div>
  );
}
