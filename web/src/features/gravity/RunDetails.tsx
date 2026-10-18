import type { ReactNode } from "react";
import { CheckCircleIcon, CrossCircleIcon, DownloadIcon, ExternalIcon } from "../../components/icons";
import { StatusBadge, statusDescription } from "../../components/StatusBadge";
import { fixed, humanize, km, scientific, shortSha, utcDateTime } from "../../lib/format";
import type {
  ReturnShell,
  Termination,
  CatalogReference,
  CollisionEvents,
  DifferentialCorrection,
  LoadedRun,
  Metric,
  Solver,
  SolverSettings,
  ValidationCheck,
} from "../../lib/data/types";
import { trajectoryArtifact } from "../../lib/data/validate";
import { commitUrl } from "../../lib/links";
import { speedDifference } from "./FreeReturnMetrics";
import { CHECK_GROUPS, CHECK_LABELS, COMPARISON_ROLE_LABEL, CONVENTION_LABEL, METRIC_LABELS, metricRole, type MetricRole } from "./labels";

const URL_PATTERN = /(https?:\/\/[^\s,)]+)/g;

/** Render text with bare http(s) URLs turned into links. */
function Linkified({ text }: { text: string }) {
  const parts = text.split(URL_PATTERN);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <a key={i} className="prose-link break-all" href={part} rel="noreferrer">
            {part}
          </a>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function formatMetric(metric: Metric): string {
  switch (metric.unit) {
    case "km":
      // Small distances (altitudes, margins) keep two decimals; orbit-scale ones are whole kilometres.
      return km(metric.value, Math.abs(metric.value) < 1000 ? 2 : 0);
    case "day":
      return `${fixed(metric.value, 4)} d`;
    case "hour":
      return `${fixed(metric.value, 3)} h`;
    case "count":
      return metric.value.toLocaleString("en-US");
    case "km/s":
      // Differences below 0.1 km/s (e.g. delta-v margins) read better in m/s or mm/s.
      return Math.abs(metric.value) < 0.1 ? speedDifference(metric.value) : `${fixed(metric.value, 5)} km/s`;
    case "normalized_cr3bp":
      return `${metric.value.toPrecision(12)} (normalized)`;
    case "dimensionless":
      return Number.isInteger(metric.value) ? signedInteger(metric.value) : metric.value.toPrecision(12);
    case "deg":
      return `${fixed(metric.value, 4)}°`;
  }
}

function signedInteger(value: number): string {
  return value < 0 ? `−${Math.abs(value)}` : String(value);
}

/** A check's observed value; null means the computation produced a non-finite number. */
function formatObserved(observed: number | null): string {
  return observed === null ? "non-finite" : scientific(observed);
}

function tolerances(s: SolverSettings): string {
  return s.rtol === s.atol
    ? `${s.name}, rtol = atol = ${scientific(s.rtol, 1)}`
    : `${s.name}, rtol ${scientific(s.rtol, 1)}, atol ${scientific(s.atol, 1)}`;
}

const METRIC_GROUPS: { role: MetricRole; title: string | null; note: string | null }[] = [
  { role: "result", title: null, note: null },
  { role: "margin", title: "Margins (non-gating)", note: "Recorded with the run; they do not decide the status." },
  {
    role: "diagnostic",
    title: "Backward-integration diagnostics",
    note: "Not validation checks: the decomposition behind the backward-integration checks, from the exported and the tighter-tolerance run.",
  },
  {
    role: "superseded",
    title: "Superseded criteria",
    note: "Kept for the record and not used for the status; the replacement checks are in the table above.",
  },
];

function Section({ id, title, children, wide = false }: { id: string; title: string; children: ReactNode; wide?: boolean }) {
  return (
    <section aria-labelledby={id} className={`min-w-0 border-t border-rule pt-6 ${wide ? "lg:col-span-2" : ""}`}>
      <h3 id={id} className="text-2xl">
        {title}
      </h3>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Fact({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div className="contents">
      <dt className="text-graphite">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}

const DL = "grid grid-cols-1 gap-x-6 gap-y-1.5 sm:grid-cols-[11.5rem_minmax(0,1fr)]";

function groupChecks(checks: ValidationCheck[]): { title: string; checks: ValidationCheck[] }[] {
  const groups = CHECK_GROUPS.map((g) => ({ ...g, checks: [] as ValidationCheck[] }));
  const other: ValidationCheck[] = [];
  for (const check of checks) {
    const group = groups.find((g) => g.prefixes.some((p) => check.name.startsWith(p)));
    (group?.checks ?? other).push(check);
  }
  return [...groups, { title: "Other checks", checks: other }].filter((g) => g.checks.length > 0);
}

function ChecksTable({ checks }: { checks: ValidationCheck[] }) {
  return (
    <div className="mt-5 overflow-x-auto">
      <table className="w-full border-collapse text-[0.95rem] max-sm:text-sm">
        <caption className="sr-only">Numerical validation checks recorded in the run manifest</caption>
        <thead>
          <tr className="border-b border-rule-strong text-left text-sm text-graphite">
            <th scope="col" className="py-2 pr-4 font-medium">
              Check and what it measures
            </th>
            <th scope="col" className="py-2 pr-4 text-right font-medium max-sm:pr-2">
              Observed
            </th>
            <th scope="col" className="py-2 pr-4 text-right font-medium max-sm:pr-2">
              Threshold
            </th>
            <th scope="col" className="py-2 font-medium">
              <span className="max-sm:sr-only">Result</span>
            </th>
          </tr>
        </thead>
        {groupChecks(checks).map((group) => (
          <tbody key={group.title}>
            <tr>
              <th scope="colgroup" colSpan={4} className="pt-5 pb-1.5 text-left font-sans text-sm font-medium text-graphite">
                {group.title}
              </th>
            </tr>
            {group.checks.map((check) => (
              <tr key={check.name} className="border-t border-rule align-top">
                <th scope="row" className="py-2.5 pr-4 text-left font-normal max-sm:pr-2">
                  {CHECK_LABELS[check.name] ?? humanize(check.name)}
                  <span className="mt-0.5 block max-w-[68ch] text-sm leading-snug text-graphite">{check.definition}</span>
                  <span className="numeric mt-0.5 block text-xs break-all text-quiet max-sm:hidden">{check.name}</span>
                </th>
                <td className={`numeric py-2.5 pr-4 text-right whitespace-nowrap max-sm:pr-2 ${check.observed === null ? "text-danger" : ""}`}>
                  {formatObserved(check.observed)}
                </td>
                <td className="numeric py-2.5 pr-4 text-right whitespace-nowrap max-sm:pr-2">{scientific(check.threshold)}</td>
                <td className="py-2.5 whitespace-nowrap">
                  {check.passed ? (
                    <span className="inline-flex items-center gap-1.5 text-valid">
                      <CheckCircleIcon /> <span className="max-sm:sr-only">Passed</span>
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 text-danger">
                      <CrossCircleIcon /> <span className="max-sm:sr-only">Failed</span>
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        ))}
      </table>
    </div>
  );
}

function ReferenceEntry({ reference }: { reference: CatalogReference }) {
  return (
    <details className="mt-4 max-w-3xl border-t border-rule pt-3">
      <summary className="cursor-pointer text-graphite marker:text-quiet hover:text-ink">
        Catalog entry the run is compared with
      </summary>
      <dl className={`${DL} mt-3 text-[0.95rem]`}>
        <Fact label="Source">
          <a className="prose-link" href={reference.url} rel="noreferrer">
            {reference.source}
          </a>
          , retrieved {reference.retrieved_on}
        </Fact>
        <Fact label="Entry">
          {humanize(reference.family)} family, L{reference.libration_point}, row {reference.row_index}
        </Fact>
        <Fact label="x0, vy0">
          <span className="numeric">
            {reference.x0}, {reference.vy0}
          </span>
        </Fact>
        <Fact label="Period">
          <span className="numeric">{reference.period}</span>
        </Fact>
        <Fact label="Jacobi constant">
          <span className="numeric">{reference.jacobi}</span>
        </Fact>
        <Fact label="Stability index">
          <span className="numeric">{reference.stability_index}</span>
        </Fact>
      </dl>
    </details>
  );
}

function collisionText(events: CollisionEvents): string {
  const bodies = [
    events.earth_radius_km !== null ? `Earth (radius ${km(events.earth_radius_km, 1)})` : null,
    events.moon_radius_km !== null ? `Moon (radius ${km(events.moon_radius_km, 1)})` : null,
  ].filter((b): b is string => b !== null);
  return bodies.length === 0 ? "Disabled for both bodies." : `Surfaces of ${bodies.join(" and ")}.`;
}

function ConvergenceRecord({ correction }: { correction: DifferentialCorrection }) {
  const { convergence } = correction;
  return (
    <figure className="mt-4 rounded-sm border border-rule bg-sheet px-4 py-3">
      <figcaption className="text-sm font-medium">Convergence record</figcaption>
      <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1.5 text-[0.95rem] sm:grid-cols-[auto_minmax(0,1fr)]">
        <Fact label="Iterations, final family member">
          <span className="numeric">{convergence.iterations_final_member}</span>
        </Fact>
        <Fact label="|vx| at the half-period crossing">
          <span className="numeric">
            {scientific(convergence.half_period_abs_vx)} (stop below {scientific(correction.half_period_abs_vx_tolerance)})
          </span>
        </Fact>
      </dl>
      <p className="mt-2 text-sm text-graphite">
        A record of how the corrector stopped, not a validation check: it is below the tolerance by construction.
      </p>
    </figure>
  );
}

const TERMINATION_TEXT: Record<Termination, string> = {
  horizon: "Reached the end of the time window",
  earth_collision: "Earth collision",
  moon_collision: "Moon collision",
  earth_return_shell: "Inbound crossing of the return shell",
  start_inside_earth: "Started inside the Earth (not propagated)",
  start_inside_moon: "Started inside the Moon (not propagated)",
  start_inside_return_shell: "Started inside the return shell (not propagated)",
  non_finite: "Non-finite state",
  evaluation_limit: "Evaluation budget exhausted",
  solver_failure: "Solver failure",
};

function ReturnShellDetails({ shell, termination }: { shell: ReturnShell; termination: Termination | null }) {
  return (
    <>
      <h4 className="mt-6 font-sans text-sm font-medium text-graphite">Return shell (entry interface)</h4>
      <dl className={`${DL} mt-2`}>
        <Fact label="Sphere">
          <span className="numeric">
            Radius {km(shell.radius_km, 2)} about the Earth centre, {km(shell.altitude_km, 2)} above the mean radius
          </span>
        </Fact>
        <Fact label="Event">
          Inbound crossings only ({shell.direction}); terminal ({shell.function}); grazing passes caught ({shell.grazing_detection})
        </Fact>
        {termination !== null && (
          <Fact label="This propagation ended">
            {TERMINATION_TEXT[termination]} <span className="numeric text-graphite">({termination})</span>
          </Fact>
        )}
      </dl>
    </>
  );
}

function SolverDetails({ solver, termination }: { solver: Solver; termination: Termination | null }) {
  const events = solver.collision_events;
  const correction = solver.differential_correction;
  return (
    <>
      <dl className={DL}>
        <Fact label="Propagation">
          <span className="numeric">{tolerances(solver)}</span>
        </Fact>
        <Fact label="Dense output">{solver.dense_output ? "Yes" : "No"}</Fact>
        <Fact label="Evaluation budget">
          <span className="numeric">{solver.max_evaluations.toLocaleString("en-US")}</span> right-hand-side evaluations per
          propagation
        </Fact>
      </dl>

      <h4 className="mt-6 font-sans text-sm font-medium text-graphite">Collision events</h4>
      <dl className={`${DL} mt-2`}>
        <Fact label="Bodies">{collisionText(events)}</Fact>
        <Fact label="Event function">
          <span className="numeric">g = |r − r_body| − radius</span> ({events.function})
        </Fact>
        <Fact label="Direction">Inward crossings only, g decreasing through zero ({events.direction})</Fact>
        <Fact label="Terminal">Yes: the first collision stops the propagation</Fact>
        <Fact label="Grazing">
          A periapsis inside the radius counts as a collision, even within one solver step ({events.grazing_detection})
        </Fact>
        <Fact label="Starting inside">Nothing is propagated ({events.start_inside})</Fact>
      </dl>
      {solver.return_shell !== null && <ReturnShellDetails shell={solver.return_shell} termination={termination} />}

      {(solver.comparisons.length > 0 || solver.monodromy !== null) && (
        <>
          <h4 className="mt-6 font-sans text-sm font-medium text-graphite">Propagations used only by validation checks</h4>
          <dl className={`${DL} mt-2`}>
            {solver.comparisons.map((c) => (
              <Fact key={c.role} label={COMPARISON_ROLE_LABEL[c.role]}>
                <span className="numeric">{tolerances(c)}</span>
              </Fact>
            ))}
            {solver.monodromy !== null && (
              <Fact label="Monodromy matrix">
                <span className="numeric">{tolerances(solver.monodromy)}</span>, variational equations over one period
              </Fact>
            )}
          </dl>
        </>
      )}

      {correction !== null && (
        <>
          <h4 className="mt-6 font-sans text-sm font-medium text-graphite">Differential correction</h4>
          <dl className={`${DL} mt-2`}>
            <Fact label="Integrator">
              <span className="numeric">{tolerances(correction)}</span>
            </Fact>
            <Fact label="Stop criterion">
              |vx| below <span className="numeric">{scientific(correction.half_period_abs_vx_tolerance)}</span> at the
              half-period crossing
            </Fact>
            <Fact label="Initial guess">
              Linear L1 Lyapunov approximation, then <span className="numeric">{correction.continuation_steps}</span>{" "}
              continuation steps in x0
            </Fact>
          </dl>
          <ConvergenceRecord correction={correction} />
        </>
      )}
    </>
  );
}

export function RunDetails({ run }: { run: LoadedRun }) {
  const { manifest, trajectoryUrl, manifestUrl } = run;
  const { model, source, environment, validation } = manifest;
  const trajectory = trajectoryArtifact(manifest);

  return (
    <div className="grid grid-cols-1 gap-x-12 gap-y-12 lg:grid-cols-2">
      <Section id="checks-title" title="Validation checks" wide>
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={manifest.status} />
          <p className="text-graphite">{statusDescription(manifest.status)}</p>
        </div>
        {validation === null ? (
          <p className="mt-4">This run records no validation checks.</p>
        ) : (
          <>
            <ChecksTable checks={validation.checks} />
            <p className="mt-4 max-w-prose text-sm text-graphite">
              Checked by {validation.validated_by}, {utcDateTime(validation.validated_at)}. A check passes when the observed
              value is at most the threshold; a non-finite value never passes. Units are those named in each definition (normalized
              units where none is named); counts are plain numbers.
            </p>
            {validation.reference !== null && <ReferenceEntry reference={validation.reference} />}
          </>
        )}
      </Section>

      <Section id="quantities-title" title="Recorded quantities" wide>
        {METRIC_GROUPS.map(({ role, title, note }) => {
          const entries = Object.entries(manifest.metrics).filter(([key, metric]) => metricRole(key, metric.definition) === role);
          if (entries.length === 0) return null;
          return (
            <div key={role} className="mt-6 first:mt-0">
              {title !== null && (
                <h4 className="font-sans text-sm font-medium text-graphite">
                  {title}
                  {note !== null && <span className="block font-normal text-quiet">{note}</span>}
                </h4>
              )}
              <dl className="mt-2 grid grid-cols-1 gap-x-12 lg:grid-cols-2">
                {entries.map(([key, metric]) => (
                  <div key={key} className="flex items-baseline justify-between gap-x-6 border-t border-rule py-2">
                    <dt className="min-w-0">
                      {role === "superseded" && (
                        <span className="mr-2 inline-block rounded-sm border border-rule-strong px-1.5 text-xs font-medium text-graphite">Superseded</span>
                      )}
                      {METRIC_LABELS[key] ?? humanize(key)}
                      <span className="mt-0.5 block text-sm leading-snug break-words text-graphite">{metric.definition}</span>
                    </dt>
                    <dd className={`numeric text-right whitespace-nowrap ${role === "superseded" ? "text-graphite line-through decoration-rule-strong" : ""}`}>
                      {formatMetric(metric)}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          );
        })}
      </Section>

      <Section id="model-title" title="Model and constants">
        <dl className={DL}>
          <Fact label="Model">Planar circular restricted three-body problem ({model.name})</Fact>
          <Fact label="Frame">Barycentric, rotating with the primaries (synodic)</Fact>
          <Fact label="Mass parameter μ">
            <span className="numeric">{model.mu}</span>
          </Fact>
          <Fact label="Length unit">
            <span className="numeric">{km(model.length_scale_km, 3)}</span>: {CONVENTION_LABEL[model.length_scale_convention]}{" "}
            <span className="numeric text-graphite">({model.length_scale_convention})</span>
          </Fact>
          <Fact label="Earth">
            <span className="numeric">
              x = {fixed(model.earth.x, 8)} LU, radius {km(model.earth.radius_km, 1)}
            </span>
          </Fact>
          <Fact label="Moon">
            <span className="numeric">
              x = {fixed(model.moon.x, 8)} LU, radius {km(model.moon.radius_km, 1)}
            </span>
          </Fact>
          <Fact label="Assumptions">{model.assumptions.map(humanize).join("; ")}</Fact>
          {model.constants_source !== null && (
            <Fact label="Source of constants">
              <Linkified text={model.constants_source} />
            </Fact>
          )}
        </dl>
      </Section>

      <Section id="solver-title" title="Solver">
        <SolverDetails solver={manifest.solver} termination={manifest.termination} />
      </Section>

      <Section id="provenance-title" title="Provenance">
        <dl className={DL}>
          <Fact label="Run">
            <span className="break-all">{manifest.run_id}</span>
          </Fact>
          <Fact label="Experiment">{manifest.experiment_id}</Fact>
          <Fact label="Created">{utcDateTime(manifest.created_at)}</Fact>
          <Fact label="Generator">
            {source.generator} {source.generator_version}
          </Fact>
          <Fact label="Source commit">
            <a className="prose-link numeric" href={commitUrl(source.git_commit)} rel="noreferrer" title={source.git_commit}>
              {shortSha(source.git_commit)}
            </a>{" "}
            in {source.repository}
          </Fact>
          <Fact label="Working tree">
            {source.git_dirty
              ? "Uncommitted changes were present, so the commit alone does not reproduce this run"
              : "Clean: generated from the committed code"}
          </Fact>
          <Fact label="Random seed">{manifest.random_seed === null ? "None (the computation uses no randomness)" : manifest.random_seed}</Fact>
        </dl>
        <h4 className="mt-6 font-sans text-sm font-medium text-graphite">Environment</h4>
        <dl className={`${DL} mt-2`}>
          <Fact label="Python">
            <span className="numeric">{environment.python}</span> ({environment.python_implementation})
          </Fact>
          <Fact label="NumPy">
            <span className="numeric">{environment.numpy}</span>
          </Fact>
          <Fact label="SciPy">
            <span className="numeric">{environment.scipy}</span>
          </Fact>
          <Fact label="Platform">
            <span className="break-all">{environment.platform}</span> ({environment.machine})
          </Fact>
        </dl>
      </Section>

      <Section id="limitations-title" title="Limitations">
        <ul className="list-disc space-y-2 pl-5 marker:text-quiet">
          {manifest.limitations.map((limitation) => (
            <li key={limitation} className="[overflow-wrap:anywhere]">{limitation}</li>
          ))}
        </ul>
      </Section>

      <Section id="downloads-title" title="Data files" wide>
        <p className="max-w-prose text-graphite">
          The exact files this page rendered. The trajectory was verified against its SHA-256 in the manifest before display.
        </p>
        <ul className="mt-4 grid gap-3 sm:grid-cols-2">
          <li>
            <a href={manifestUrl} download className="prose-link inline-flex items-center gap-2 font-medium">
              <DownloadIcon /> manifest.json
            </a>
            <span className="block text-sm text-graphite">Provenance, environment, model, solver, checks and limitations</span>
          </li>
          <li>
            <a href={trajectoryUrl} download className="prose-link inline-flex items-center gap-2 font-medium">
              <DownloadIcon /> {trajectory?.path ?? "trajectory.json"}
            </a>
            <span className="block text-sm break-all text-graphite">
              {run.trajectory.length.toLocaleString("en-US")} samples of ({run.trajectory.state_order.join(", ")})
              {run.trajectory.events.length > 0 ? ` and ${run.trajectory.events.length} marked events` : ""}. SHA-256{" "}
              {trajectory?.sha256}
            </span>
          </li>
        </ul>
        <p className="mt-4 text-sm text-graphite">
          <a className="prose-link inline-flex items-center gap-1" href={commitUrl(source.git_commit)} rel="noreferrer">
            Browse the code that produced it
            <ExternalIcon className="size-3.5" />
          </a>
        </p>
      </Section>
    </div>
  );
}
