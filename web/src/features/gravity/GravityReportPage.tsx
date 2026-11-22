import { ArticleBody, ArticleHeader, ArticleSection, Callout, CitationList, ProvenanceBlock, StatusBadge, type TocItem } from "../../components/Article";
import { ErrorState, LoadingState } from "../../components/DataState";
import { ExternalIcon } from "../../components/icons";
import type { CatalogRun, LoadedAtlasRun, LoadedRun, Manifest } from "../../lib/data/types";
import { useAtlasRun, useCatalog, useRunList, type Resource } from "../../lib/data/useResource";
import { fixed } from "../../lib/format";
import { GRAVITY_LITERATURE } from "../../lib/literature";
import { commitUrl, docUrl } from "../../lib/links";
import { Link } from "../../lib/router";
import { usePageTitle } from "../../lib/usePageTitle";
import { ARM_INFO, armOf, mean, percent } from "./atlasModel";

const STUDY_ID = "gravity-atlas";
const PROTOCOL_DOC = "docs/studies/gravity-atlas/protocol.md";
const DECISIONS_DOC = "docs/decisions.md";

const TOC: readonly TocItem[] = [
  { id: "summary", label: "Summary & verdict" },
  { id: "physics", label: "Validated CR3BP physics" },
  { id: "geometry", label: "Thin feasible band" },
  { id: "comparison", label: "Seven methods (M4)" },
  { id: "tradeoff", label: "Knife-edge trade-off" },
  { id: "limits", label: "Limitations & provenance" },
  { id: "references", label: "Verified literature" },
];

function checkResidual(manifest: Manifest, pattern: string): string {
  const match = manifest.validation?.checks.find((c) => c.name.includes(pattern));
  if (match === undefined || match.observed === null) return "–";
  const v = match.observed;
  if (v === 0) return "0";
  return Math.abs(v) < 1e-2 ? v.toExponential(2) : fixed(v, 4);
}

function formatPValue(p: number): string {
  if (p < 0.001) return p.toExponential(2);
  return fixed(p, 4);
}

function armLabel(id: string): string {
  return ARM_INFO.find((a) => a.id === id)?.name ?? id;
}

function ReportBody({
  atlasRun,
  trajectories,
}: {
  atlasRun: LoadedAtlasRun;
  trajectories: Resource<LoadedRun[]>;
}) {
  const { atlas, manifest, manifestUrl } = atlasRun;
  const squares = atlas.grid.reachable_cells.length;
  const worstP = Math.max(...atlas.contrasts.map((c) => c.p_holm));
  const armT = armOf(atlas, "T");
  const armD = armOf(atlas, "D");
  const armCC = armOf(atlas, "CC");
  const armDE = armOf(atlas, "DE");

  return (
    <ArticleBody toc={TOC}>
      <ArticleSection id="summary" number="1" title="Summary & Headline Verdict">
        <p>
          A free-return trajectory slingshots around the Moon and falls back to Earth&apos;s atmospheric entry interface without a post-launch burn.
          Classical continuation traces one continuous family at a time, while global optimizers often hunt for a single nominal winner. Gravity Atlas
          asks a catalog question under a matched budget of <span className="numeric">16,384</span> propagated launches per method across 16 independent
          random seeds: does quality-diversity search (MAP-Elites) build a catalog that keeps more trajectories viable when the launch burn is slightly
          off, or when mission requirements change?
        </p>
        <Callout title="Headline M4 verdict (pre-registered protocol, H1 supported)" tone="ink">
          <p>
            <strong>H1 is supported against all five baselines through coverage.</strong> Across 16 runs at equal propagation budget, MAP-Elites holds an
            average of <span className="numeric">{fixed(mean(armT, "p1"), 1)}</span> error-surviving trajectories (vs.{" "}
            <span className="numeric">{fixed(mean(armDE, "p1"), 1)}</span> for differential evolution,{" "}
            <span className="numeric">{fixed(mean(armCC, "p1"), 1)}</span> for continuation, and{" "}
            <span className="numeric">{fixed(mean(armD, "p1"), 1)}</span> for the dense sweep) and meets{" "}
            <span className="numeric">{percent(mean(armT, "p2"))}</span> of 62 held-out mission requirements (all Holm-adjusted p ≤{" "}
            <span className="numeric">{formatPValue(worstP)}</span>). Yet individual MAP-Elites solutions survive launch errors less often than the
            dense sweep&apos;s (<span className="numeric">{percent(mean(armT, "mean_s"))}</span> vs.{" "}
            <span className="numeric">{percent(mean(armD, "mean_s"))}</span>). The study is closed under D29.
          </p>
        </Callout>
      </ArticleSection>

      <ArticleSection id="physics" number="2" title="Validated Three-Body Physics (M1)">
        <p>
          All trajectories live in the planar circular restricted three-body problem (CR3BP) in a barycentric synodic frame rotating with the Moon
          (mass parameter <span className="numeric font-mono text-[0.9rem]">μ = {manifest.model.mu.toFixed(10)}</span>, Earth–Moon scale{" "}
          <span className="numeric">{fixed(manifest.model.length_scale_km, 1)}</span> km, velocity unit{" "}
          <span className="numeric">{fixed(manifest.model.velocity_scale_km_s, 6)}</span> km/s). Spacecraft launch from a{" "}
          <span className="numeric">{fixed(atlas.parking_orbit_radius_km - atlas.earth.radius_km, 0)}</span> km circular Earth parking orbit with a
          single impulse and coast under <span className="font-mono text-[0.9rem]">{manifest.solver.name}</span> (
          <span className="numeric font-mono text-[0.875rem]">
            rtol = atol = {manifest.solver.rtol.toExponential(0)}
          </span>
          ) with periapsis-based grazing-collision detection and an inbound <span className="numeric">121.92</span> km entry-interface shell.
        </p>

        {trajectories.status === "loading" && <LoadingState label="Loading validated reference trajectories…" />}
        {trajectories.status === "error" && <ErrorState error={trajectories.error} onRetry={trajectories.retry} />}
        {trajectories.status === "ready" && trajectories.value.length > 0 && (
          <div className="my-4 overflow-x-auto rounded-sm border border-rule bg-sheet p-4">
            <table className="w-full min-w-[34rem] border-collapse text-left text-sm">
              <caption className="mb-2 text-left text-xs text-graphite">
                Published trajectory runs in Gravity Atlas, each SHA-256 verified against its manifest
              </caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-graphite">
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    Run ID
                  </th>
                  <th scope="col" className="py-1.5 pr-3 font-medium">
                    Status
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Checks passed
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    Jacobi drift
                  </th>
                  <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                    vs LSODA (10⁻¹³)
                  </th>
                  <th scope="col" className="py-1.5 text-right font-medium">
                    Commit
                  </th>
                </tr>
              </thead>
              <tbody>
                {trajectories.value.map((r) => {
                  const checks = r.manifest.validation?.checks ?? [];
                  const passed = checks.filter((c) => c.passed).length;
                  return (
                    <tr key={r.manifest.run_id} className="border-b border-rule last:border-b-0">
                      <th scope="row" className="py-2 pr-3 font-mono text-xs font-normal text-ink">
                        {r.manifest.run_id}
                      </th>
                      <td className="py-2 pr-3">
                        <StatusBadge status={r.manifest.status} />
                      </td>
                      <td className="numeric py-2 pr-3 text-right text-graphite">
                        {passed} / {checks.length}
                      </td>
                      <td className="numeric py-2 pr-3 text-right font-mono text-xs text-graphite">{checkResidual(r.manifest, "jacobi")}</td>
                      <td className="numeric py-2 pr-3 text-right font-mono text-xs text-graphite">
                        {checkResidual(r.manifest, "independent_integrator")}
                      </td>
                      <td className="py-2 text-right font-mono text-xs">
                        <a className="prose-link inline-flex items-center gap-1" href={commitUrl(r.manifest.source.git_commit)} rel="noreferrer">
                          {r.manifest.source.git_commit.slice(0, 7)}
                          <ExternalIcon className="size-3" />
                        </a>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </ArticleSection>

      <ArticleSection id="geometry" number="3" title="The Thin Feasible Ribbon (M3)">
        <p>
          Before running optimizer comparisons, M3 mapped the three-dimensional launch space (departure phase θ ∈ [0°, 360°), tangential impulse Δv ∈
          [3.00, 3.25] km/s, flight-path offset α ∈ [−10°, +10°]) using a dense 720 × 126 planar grid and a 2¹⁷-sample (131,072) scrambled Sobol
          sweep. Free returns do not fill broad regions: only <span className="numeric">243</span> of <span className="numeric">131,072</span> launches
          (<span className="numeric">0.185 %</span>) qualify, occupying <span className="numeric">27</span> of <span className="numeric">100</span>{" "}
          perilune-altitude × flight-time descriptor cells and splitting cleanly into posigrade (<span className="font-mono text-[0.875rem]">c1</span>)
          and retrograde (<span className="font-mono text-[0.875rem]">c2</span>) entry branches.
        </p>
        <p>
          Fine <span className="numeric">0.02</span> m/s probes confirm why local searches struggle: at some departure phases the feasible Δv window is
          less than <span className="numeric">0.4</span> m/s wide. Most random restarts collide with the Moon, miss the entry shell within 10 days, or
          return before reaching lunar vicinity.
        </p>
      </ArticleSection>

      <ArticleSection id="comparison" number="4" title="Seven Methods at Matched Budget (M4)">
        <p>
          Each method searched the 3-D launch box under the exact same <span className="numeric">16,384</span>-propagation budget (16 seeds each) and
          stored its best trajectory (entry flight-path angle closest to −6.5°) per reachable descriptor square (<span className="numeric">{squares}</span>{" "}
          squares in all). Kept trajectories were re-checked against <span className="numeric">{atlas.draws}</span> held-out launch-burn perturbations
          with up to <span className="numeric">{fixed(atlas.budget_m_s, 1)}</span> m/s correction allowed and tested against 62 unseen mission envelopes.
        </p>

        <div className="my-4 overflow-x-auto rounded-sm border border-rule bg-sheet p-4">
          <table className="w-full min-w-[34rem] border-collapse text-left text-sm">
            <caption className="mb-2 text-left text-xs text-graphite">16-seed means at matched 16,384-propagation budget</caption>
            <thead>
              <tr className="border-b border-rule-strong text-xs text-graphite">
                <th scope="col" className="py-1.5 pr-4 font-medium">
                  Method
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  Elites filled (of {squares})
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  P1: Expected survivors
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  P2: Requirements met
                </th>
                <th scope="col" className="py-1.5 text-right font-medium">
                  Mean per-elite survival
                </th>
              </tr>
            </thead>
            <tbody>
              {ARM_INFO.map((info) => {
                const arm = armOf(atlas, info.id);
                return (
                  <tr key={info.id} className="border-b border-rule last:border-b-0">
                    <th scope="row" className="py-2 pr-4 font-medium text-ink">
                      {info.name}
                    </th>
                    <td className="numeric py-2 pr-3 text-right text-graphite">{fixed(mean(arm, "elites"), 1)}</td>
                    <td className="numeric py-2 pr-3 text-right font-medium text-ink">{fixed(mean(arm, "p1"), 1)}</td>
                    <td className="numeric py-2 pr-3 text-right text-ink">{percent(mean(arm, "p2"))}</td>
                    <td className="numeric py-2 text-right text-graphite">{percent(mean(arm, "mean_s"))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="my-4 overflow-x-auto rounded-sm border border-rule bg-sheet p-4">
          <table className="w-full min-w-[28rem] border-collapse text-left text-sm">
            <caption className="mb-2 text-left text-xs text-graphite">
              Pre-registered paired sign-flip contrasts: MAP-Elites (T) vs. five baselines (16 seeds, Holm-adjusted)
            </caption>
            <thead>
              <tr className="border-b border-rule-strong text-xs text-graphite">
                <th scope="col" className="py-1.5 pr-3 font-medium">
                  Endpoint
                </th>
                <th scope="col" className="py-1.5 pr-3 font-medium">
                  Comparison
                </th>
                <th scope="col" className="py-1.5 pr-3 text-right font-medium">
                  Holm-adjusted p
                </th>
                <th scope="col" className="py-1.5 text-right font-medium">
                  Pre-registered decision (α = 0.05)
                </th>
              </tr>
            </thead>
            <tbody>
              {atlas.contrasts.map((c) => {
                const sig = c.p_holm <= 0.05;
                return (
                  <tr key={`${c.endpoint}-${c.baseline}`} className="border-b border-rule last:border-b-0">
                    <td className="py-1.5 pr-3 font-mono text-xs text-graphite">{c.endpoint}</td>
                    <th scope="row" className="py-1.5 pr-3 font-normal text-ink">
                      MAP-Elites (T) vs. {armLabel(c.baseline)} ({c.baseline})
                    </th>
                    <td className="numeric py-1.5 pr-3 text-right font-mono text-xs text-graphite">{formatPValue(c.p_holm)}</td>
                    <td className={`py-1.5 text-right text-xs font-medium ${sig ? "text-valid" : "text-graphite"}`}>
                      {sig ? "significant" : "not significant"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ArticleSection>

      <ArticleSection id="tradeoff" number="5" title="The Knife-Edge Trade-off">
        <p>
          Why does MAP-Elites achieve the highest expected survivor count (<span className="numeric">{fixed(mean(armT, "p1"), 1)}</span>) despite
          having lower per-trajectory survival (<span className="numeric">{percent(mean(armT, "mean_s"))}</span>) than the dense sweep (
          <span className="numeric">{percent(mean(armD, "mean_s"))}</span>)?
        </p>
        <p>
          The dense sweep fills only <span className="numeric">{fixed(mean(armD, "elites"), 1)}</span> easy, wide-corridor squares near the middle of
          the atlas, whereas MAP-Elites reaches <span className="numeric">{fixed(mean(armT, "elites"), 1)}</span> of {squares} squares—including thin,
          high-sensitivity edges near low perilune or short flight times. Furthermore, within each square the archive stores whichever launch hits
          closest to the nominal <span className="numeric">−6.5°</span> entry angle without penalizing local sensitivity. As predicted by Flageat
          &amp; Cully (2023) for uncertain quality-diversity domains, selecting purely on nominal objective pushes elites toward knife-edge basins.
        </p>
      </ArticleSection>

      <ArticleSection id="limits" number="6" title="Limitations & Provenance">
        <ul className="list-disc space-y-2 pl-6 text-[0.975rem] text-graphite">
          <li>
            <strong className="text-ink">Physical idealization:</strong> planar circular three-body dynamics only; excludes lunar eccentricity,
            out-of-plane inclinations, solar gravity, finite burns, and real ephemerides. A qualifying “return” is an inbound crossing of a 121.92 km
            sphere, not an atmospheric re-entry guidance simulation.
          </li>
          <li>
            <strong className="text-ink">Correction model:</strong> survival evaluates single impulsive corrections up to{" "}
            <span className="numeric">{fixed(atlas.budget_m_s, 1)}</span> m/s under the pre-registered perturbation distribution; real navigation
            schedules and multi-burn dispersions were not modelled.
          </li>
          <li>
            <strong className="text-ink">Study status (D29):</strong> Gravity Atlas is formally closed after M4; individual nominal elites on the site
            reflect seed 1001, while the 16-run statistics above carry the inferential evidence.
          </li>
        </ul>
        <div className="mt-6">
          <ProvenanceBlock manifest={manifest} manifestUrl={manifestUrl} />
        </div>
      </ArticleSection>

      <ArticleSection id="references" number="7" title="Verified Literature">
        <CitationList citations={GRAVITY_LITERATURE} />
      </ArticleSection>
    </ArticleBody>
  );
}

function LoadedGravityReport({
  atlasEntry,
  trajectoryEntries,
}: {
  atlasEntry: CatalogRun;
  trajectoryEntries: readonly CatalogRun[];
}) {
  const atlasResource = useAtlasRun(atlasEntry.run_id, atlasEntry.path);
  const trajectories = useRunList(trajectoryEntries);

  if (atlasResource.status === "loading") return <LoadingState label="Loading and verifying Gravity Atlas data…" />;
  if (atlasResource.status === "error") return <ErrorState error={atlasResource.error} onRetry={atlasResource.retry} />;
  return <ReportBody atlasRun={atlasResource.value} trajectories={trajectories} />;
}

export function GravityReportPage() {
  usePageTitle("Gravity Atlas Report – Phase Space");
  const catalog = useCatalog();
  const study = catalog.status === "ready" ? catalog.value.studies.find((s) => s.study_id === STUDY_ID) : undefined;
  const atlasEntry = study?.runs.find((r) => r.kind === "atlas");
  const trajectoryEntries = study?.runs.filter((r) => r.kind === "trajectory") ?? [];

  return (
    <div className="mx-auto max-w-7xl px-4 pb-8 sm:px-8">
      <ArticleHeader
        crumbs={[{ label: "Studies", to: "/" }, { label: "Gravity Atlas", to: "/gravity" }, { label: "Report" }]}
        title="Gravity Atlas — Study Report & Literature"
        subtitle="At matched propagation budgets, does a MAP-Elites archive of Earth–Moon free returns contain more trajectories that survive launch errors and meet unseen mission requirements than five baselines?"
        actions={
          <>
            <Link
              to="/gravity"
              className="inline-flex items-center rounded-sm bg-ink px-4 py-2 text-sm font-medium text-sheet no-underline hover:bg-canvas-raised"
            >
              Open interactive orbital atlas
            </Link>
            <a className="prose-link inline-flex items-center gap-1.5 text-sm text-graphite" href={docUrl(PROTOCOL_DOC)} rel="noreferrer">
              Frozen M4 protocol
              <ExternalIcon className="size-3.5" />
            </a>
            <a className="prose-link inline-flex items-center gap-1.5 text-sm text-graphite" href={docUrl(DECISIONS_DOC)} rel="noreferrer">
              Decision log (D28–D29)
              <ExternalIcon className="size-3.5" />
            </a>
          </>
        }
      />

      <div className="mt-8">
        {catalog.status === "loading" && <LoadingState label="Loading the catalog…" />}
        {catalog.status === "error" && <ErrorState error={catalog.error} onRetry={catalog.retry} />}
        {catalog.status === "ready" &&
          (atlasEntry === undefined ? (
            <p className="text-graphite">The Gravity Atlas run has not been published yet.</p>
          ) : (
            <LoadedGravityReport atlasEntry={atlasEntry} trajectoryEntries={trajectoryEntries} />
          ))}
      </div>
    </div>
  );
}
