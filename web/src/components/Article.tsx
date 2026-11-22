import type { ReactNode } from "react";
import type { AnyManifest } from "../lib/data/load";
import type { RunStatus } from "../lib/data/types";
import type { Citation } from "../lib/literature";
import { commitUrl } from "../lib/links";
import { Link } from "../lib/router";
import { AlertIcon, CheckCircleIcon, DashedCircleIcon, ExternalIcon } from "./icons";

export interface Crumb {
  label: string;
  to?: string;
}

export interface TocItem {
  id: string;
  label: string;
}

const STATUS_META: Record<RunStatus, { label: string; className: string; validated: boolean }> = {
  validated: { label: "Validated", className: "text-valid", validated: true },
  computed_unvalidated: { label: "Computed (unvalidated)", className: "text-caution", validated: false },
  illustrative: { label: "Illustrative", className: "text-graphite", validated: false },
};

/** Non-colour-only artifact status badge matching `docs/frontend-design.md`. */
export function StatusBadge({ status }: { status: RunStatus }) {
  const meta = STATUS_META[status];
  const Icon = meta.validated ? CheckCircleIcon : DashedCircleIcon;
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${meta.className}`}>
      <Icon className="shrink-0" />
      <span>{meta.label}</span>
    </span>
  );
}

/** Editorial header for reports and methodology pages. */
export function ArticleHeader({
  crumbs,
  title,
  subtitle,
  actions,
}: {
  crumbs: readonly Crumb[];
  title: string;
  subtitle: string;
  actions?: ReactNode;
}) {
  return (
    <header className="border-b border-rule pt-8 pb-8 sm:pt-12 sm:pb-10">
      <nav aria-label="Breadcrumb" className="text-sm text-graphite">
        {crumbs.map((c, i) => (
          <span key={c.label}>
            {i > 0 && (
              <span aria-hidden="true" className="mx-2 text-quiet">
                /
              </span>
            )}
            {c.to !== undefined ? (
              <Link to={c.to} className="prose-link">
                {c.label}
              </Link>
            ) : (
              <span aria-current="page">{c.label}</span>
            )}
          </span>
        ))}
      </nav>
      <h1 className="mt-3 max-w-4xl text-[2.4rem] sm:text-5xl">{title}</h1>
      <p className="mt-4 max-w-[66ch] font-serif text-xl leading-snug text-graphite sm:text-[1.45rem]">{subtitle}</p>
      {actions !== undefined && <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">{actions}</div>}
    </header>
  );
}

/** 12-column responsive layout: sticky contents rail on desktop, jump list on mobile. */
export function ArticleBody({ toc, children }: { toc: readonly TocItem[]; children: ReactNode }) {
  return (
    <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-10 lg:grid-cols-12">
      <aside className="lg:col-span-3">
        <nav aria-label="Contents" className="lg:sticky lg:top-8">
          <p className="text-xs font-medium tracking-wider text-quiet uppercase">Contents</p>
          <ol className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm lg:flex-col lg:gap-y-2">
            {toc.map((item) => (
              <li key={item.id}>
                <a href={`#${item.id}`} className="prose-link text-graphite hover:text-ink">
                  {item.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>
      </aside>
      <div className="min-w-0 space-y-12 lg:col-span-9">{children}</div>
    </div>
  );
}

/** Numbered narrative section separated by a thin rule. */
export function ArticleSection({
  id,
  number,
  title,
  children,
}: {
  id: string;
  number: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section aria-labelledby={id} className="border-t border-rule pt-8 first:border-t-0 first:pt-0">
      <h2 id={id} className="scroll-mt-8 text-2xl sm:text-3xl">
        <span className="numeric mr-2 text-quiet">{number}.</span>
        {title}
      </h2>
      <div className="mt-5 space-y-4 text-[1.0125rem] leading-relaxed text-ink">{children}</div>
    </section>
  );
}

/** Left-bordered summary or caveat highlight. */
export function Callout({
  title,
  tone = "ink",
  children,
}: {
  title: string;
  tone?: "ink" | "caution";
  children: ReactNode;
}) {
  const borderClass = tone === "caution" ? "border-caution" : "border-ink";
  return (
    <div className={`rounded-sm border border-rule border-l-2 ${borderClass} bg-sheet py-4 pr-5 pl-5`}>
      <p className="font-medium text-ink">{title}</p>
      <div className="mt-1.5 text-[0.95rem] leading-relaxed text-graphite">{children}</div>
    </div>
  );
}

/** Ordered list of verified DOIs and preprints tied to the literature scan. */
export function CitationList({ citations }: { citations: readonly Citation[] }) {
  return (
    <ol className="space-y-4 text-[0.95rem]">
      {citations.map((c, i) => (
        <li key={c.id} className="border-b border-rule pb-4 last:border-b-0 last:pb-0">
          <p className="text-ink">
            <span className="numeric mr-2 text-xs text-quiet">[{i + 1}]</span>
            <span className="font-medium">{c.authors}</span> ({c.year}). “{c.title}.”{" "}
            <span className="italic">{c.venue}</span>.{" "}
            <a className="prose-link inline-flex items-center gap-1 whitespace-nowrap" href={c.url} rel="noreferrer">
              {c.identifier}
              <ExternalIcon className="size-3.5" />
            </a>
          </p>
          <p className="mt-1 pl-6 text-sm text-graphite">{c.relevance}</p>
        </li>
      ))}
    </ol>
  );
}

function formatCheckValue(value: number | null): string {
  if (value === null) return "non-finite (null)";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e5) return value.toExponential(2);
  if (Number.isInteger(value)) return value.toLocaleString("en-US");
  return Number(value.toPrecision(4)).toString();
}

/** Manifest provenance, SHA-256 checksums, automated check residuals, and stated limitations. */
export function ProvenanceBlock({ manifest, manifestUrl }: { manifest: AnyManifest; manifestUrl: string }) {
  const checks = manifest.validation?.checks ?? [];
  return (
    <div className="rounded-sm border border-rule bg-sheet p-5 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-rule pb-3">
        <div>
          <p className="font-mono text-xs text-quiet">Run ID</p>
          <p className="font-mono text-[0.9rem] font-medium text-ink">{manifest.run_id}</p>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <StatusBadge status={manifest.status} />
          <a className="prose-link inline-flex items-center gap-1 text-xs font-medium text-ink" href={manifestUrl}>
            manifest.json
          </a>
        </div>
      </div>

      <dl className="mt-4 grid gap-x-8 gap-y-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs text-quiet">Created (UTC)</dt>
          <dd className="numeric mt-0.5 font-mono text-xs text-ink">{manifest.created_at}</dd>
        </div>
        <div>
          <dt className="text-xs text-quiet">Code commit</dt>
          <dd className="mt-0.5 font-mono text-xs text-ink">
            <a className="prose-link inline-flex items-center gap-1" href={commitUrl(manifest.source.git_commit)} rel="noreferrer">
              {manifest.source.git_commit.slice(0, 7)}
              <ExternalIcon className="size-3" />
            </a>
            {manifest.source.git_dirty ? " (git_dirty)" : " (clean)"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-quiet">Generator</dt>
          <dd className="mt-0.5 font-mono text-xs break-all text-ink">
            {manifest.source.generator} v{manifest.source.generator_version}
          </dd>
        </div>
      </dl>

      <div className="mt-5 border-t border-rule pt-4">
        <p className="text-xs font-medium text-graphite">Artifacts (SHA-256 verified on load)</p>
        <ul className="mt-2 space-y-1.5">
          {manifest.artifacts.map((a) => (
            <li key={a.path} className="flex flex-wrap items-baseline justify-between gap-2 font-mono text-xs">
              <span className="text-ink">
                {a.path} <span className="text-quiet">({a.kind})</span>
              </span>
              <span title={a.sha256} className="text-graphite">
                sha256:{a.sha256.slice(0, 12)}…
              </span>
            </li>
          ))}
        </ul>
      </div>

      {checks.length > 0 && (
        <div className="mt-5 border-t border-rule pt-4">
          <p className="text-xs font-medium text-graphite">Recorded automated checks ({checks.length})</p>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[24rem] border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-rule-strong text-quiet">
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    Check
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Observed
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Threshold
                  </th>
                  <th scope="col" className="py-1.5 text-right font-medium">
                    Result
                  </th>
                </tr>
              </thead>
              <tbody>
                {checks.map((c) => (
                  <tr key={c.name} className="border-b border-rule last:border-b-0">
                    <th scope="row" className="py-1.5 pr-3 font-mono font-normal text-ink" title={c.definition}>
                      {c.name}
                    </th>
                    <td className="numeric py-1.5 pr-3 text-right font-mono text-graphite">{formatCheckValue(c.observed)}</td>
                    <td className="numeric py-1.5 pr-3 text-right font-mono text-graphite">{formatCheckValue(c.threshold)}</td>
                    <td className="py-1.5 text-right whitespace-nowrap">
                      <span className={`inline-flex items-center gap-1 font-medium ${c.passed ? "text-valid" : "text-danger"}`}>
                        {c.passed ? <CheckCircleIcon className="size-3.5" /> : <AlertIcon className="size-3.5" />}
                        {c.passed ? "pass" : "fail"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {manifest.limitations.length > 0 && (
        <div className="mt-5 border-t border-rule pt-4">
          <p className="text-xs font-medium text-graphite">Stated limitations</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-relaxed text-graphite">
            {manifest.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
