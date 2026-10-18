import type { RunStatus } from "../lib/data/types";
import { CheckCircleIcon, DashedCircleIcon, HalfCircleIcon } from "./icons";

const STATUS: Record<RunStatus, { label: string; description: string; className: string; Icon: typeof CheckCircleIcon }> = {
  validated: {
    label: "Validated",
    description: "Recorded numerical checks passed against declared thresholds. Not a claim of a new scientific result.",
    className: "text-valid border-valid/40 bg-valid/5",
    Icon: CheckCircleIcon,
  },
  computed_unvalidated: {
    label: "Computed (not validated)",
    description: "Produced by a computation; independent validation is incomplete.",
    className: "text-caution border-caution/40 bg-caution/5",
    Icon: HalfCircleIcon,
  },
  illustrative: {
    label: "Illustrative",
    description: "Drawn or invented for explanation. Not physical evidence.",
    className: "text-graphite border-rule-strong bg-sheet",
    Icon: DashedCircleIcon,
  },
};

/** Run status as icon + text (never color alone), with the meaning available to assistive tech. */
export function StatusBadge({ status, compact = false }: { status: RunStatus; compact?: boolean }) {
  const { label, description, className, Icon } = STATUS[status];
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-sm border text-sm font-medium ${compact ? "px-1.5 py-0.5" : "px-2 py-0.5"} ${className}`}
      title={description}
    >
      <Icon />
      <span className={compact ? "max-sm:sr-only" : undefined}>
        {label}
        <span className="sr-only">: {description}</span>
      </span>
    </span>
  );
}

export function statusDescription(status: RunStatus): string {
  return STATUS[status].description;
}
