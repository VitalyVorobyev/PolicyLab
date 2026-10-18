import { isDataError, type DataErrorKind } from "../lib/data/errors";
import { AlertIcon } from "./icons";

const TITLES: Record<DataErrorKind, { title: string; hint: string }> = {
  network: {
    title: "The research data could not be downloaded",
    hint: "Check your connection and try again. The site needs only static files, so a reload usually fixes this.",
  },
  http: {
    title: "A research file is missing from this deployment",
    hint: "The published file was not found. If you maintain this site, check that the run directory under /research/ was deployed.",
  },
  parse: {
    title: "A research file is not readable JSON",
    hint: "The file is corrupt or was replaced by something else (for example an HTML error page). Re-publish the run with the publish script.",
  },
  schema_version: {
    title: "This data uses a schema version the site does not understand",
    hint: "The data is newer (or older) than this viewer. Update the site, or re-export the run with a supported schema.",
  },
  invalid: {
    title: "The run does not satisfy the data contract",
    hint: "Nothing from this run is shown, because displaying partially valid data could mislead. Re-export the run and check docs/data-contract.md.",
  },
  empty: {
    title: "The run contains no data to show",
    hint: "The file is well-formed but empty. Re-export the run; empty runs are never published intentionally.",
  },
  checksum: {
    title: "A data file does not match its recorded checksum",
    hint: "The data differs from what the manifest certifies, so it is not shown. Re-publish the run from its original export.",
  },
  unsafe_path: {
    title: "A data path points outside the research directory",
    hint: "The catalog or manifest names a path that is not allowed. Fix the path in the published files.",
  },
};

export function LoadingState({ label }: { label: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 py-6 text-graphite">
      <span className="inline-block size-2 animate-pulse rounded-full bg-accent motion-reduce:animate-none" aria-hidden="true" />
      {label}
    </div>
  );
}

/** Accessible error panel with what went wrong, the technical detail, and what to do. */
export function ErrorState({ error, onRetry, tone = "paper" }: { error: unknown; onRetry?: () => void; tone?: "paper" | "canvas" }) {
  const known = isDataError(error) ? TITLES[error.kind] : null;
  const title = known?.title ?? "Something went wrong while showing this page";
  const hint = known?.hint ?? "Reload the page. If the problem persists, please open an issue on GitHub.";
  const detail = error instanceof Error ? error.message : String(error);
  const source = isDataError(error) ? error.source : null;
  const canvas = tone === "canvas";
  return (
    <div
      role="alert"
      className={`max-w-prose rounded-sm border-l-2 border-danger py-4 pl-5 pr-4 ${canvas ? "bg-canvas-raised text-canvas-strong" : "bg-sheet text-ink"}`}
    >
      <p className="flex items-center gap-2 font-medium">
        <AlertIcon className={canvas ? "text-accent" : "text-danger"} />
        {title}
      </p>
      <p className={`mt-2 ${canvas ? "text-canvas-text" : "text-graphite"}`}>{hint}</p>
      <p className={`mt-3 text-sm break-words ${canvas ? "text-canvas-text" : "text-quiet"}`}>
        Detail: {detail}
        {source !== null && !detail.includes(source) ? ` (${source})` : null}
      </p>
      {onRetry !== undefined && (
        <button
          type="button"
          onClick={onRetry}
          className={`mt-4 rounded-sm border px-3 py-1.5 text-sm font-medium ${canvas ? "border-canvas-axis hover:border-accent" : "border-rule-strong hover:border-ink"}`}
        >
          Try again
        </button>
      )}
    </div>
  );
}
