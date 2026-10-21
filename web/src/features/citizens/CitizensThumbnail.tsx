import { useCatalog, useCitizensRun } from "../../lib/data/useResource";
import type { CitizensDemo } from "../../lib/data/citizens";
import { AI_ROWS, fill, RULE_ROWS } from "./Crossplay";
import { cellOf, governorInfo, MODEL_KEYS, signedPercent } from "./model";

function Grid({ demo }: { demo: CitizensDemo }) {
  const rows = [...RULE_ROWS, ...AI_ROWS].filter((r) => demo.crossplay.policies.includes(r));
  return (
    <div className="rounded-sm bg-canvas p-4 sm:p-6" aria-hidden="true">
      <div className="grid grid-cols-[minmax(4.5rem,0.8fr)_repeat(4,minmax(0,1fr))] gap-[3px] text-[0.75rem]">
        <span />
        {MODEL_KEYS.map((m) => (
          <span key={m} className="pb-1 text-center text-canvas-text">
            {m}
          </span>
        ))}
        {rows.map((policy) => (
          <div key={policy} className="contents">
            <span className="flex items-center text-canvas-text">{governorInfo(policy)?.short ?? policy}</span>
            {MODEL_KEYS.map((m) => {
              const cell = cellOf(demo, policy, m);
              const style = cell === undefined ? {} : fill(cell.vs_k_star.relative);
              return (
                <span key={m} className="numeric flex h-7 items-center justify-center rounded-[2px] sm:h-8" style={style}>
                  {cell === undefined ? "" : signedPercent(cell.vs_k_star.relative)}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function Loaded({ runId, path }: { runId: string; path: string }) {
  const run = useCitizensRun(runId, path);
  return run.status === "ready" ? <Grid demo={run.value.demo} /> : <div className="min-h-[16rem] rounded-sm bg-canvas" aria-hidden="true" />;
}

/** The gallery's picture of Model Citizens: the cross-play matrix, read from the published run. */
export function CitizensThumbnail() {
  const catalog = useCatalog();
  const entry = catalog.status === "ready" ? catalog.value.studies.find((s) => s.study_id === "model-citizens")?.runs.find((r) => r.kind === "citizens_demo") : undefined;
  return entry === undefined ? <div className="min-h-[16rem] rounded-sm bg-canvas" aria-hidden="true" /> : <Loaded runId={entry.run_id} path={entry.path} />;
}
