import { useState } from "react";
import { ErrorState } from "../../components/DataState";
import { Term } from "../../components/Tooltip";
import type { ArmId, LoadedAtlasRun } from "../../lib/data/types";
import { useErrorFan } from "../../lib/data/useResource";
import { fixed } from "../../lib/format";
import { AtlasGrid } from "./AtlasGrid";
import { ARM_INFO, armOf, eliteAt, mean, nearestElite, percent, type Cell } from "./atlasModel";
import { MethodSwitch } from "./MethodSwitch";
import { OrbitStage } from "./OrbitStage";

/** The default view: MAP-Elites, with a low-perilune trajectory of about six days (Apollo-like). */
const DEFAULT_ARM: ArmId = "T";
const DEFAULT_CELL: Cell = [1, 1];

function Stat({ label, tip, value }: { label: string; tip: string; value: string }) {
  return (
    <div>
      <dt className="text-xs leading-tight text-canvas-text">
        <Term tip={tip}>{label}</Term>
      </dt>
      <dd className="numeric mt-1 font-serif text-2xl leading-none text-canvas-strong">{value}</dd>
    </div>
  );
}

export function Demo({ run, runPath }: { run: LoadedAtlasRun; runPath: string }) {
  const { atlas } = run;
  const [armId, setArmId] = useState<ArmId>(DEFAULT_ARM);
  // The square the viewer chose (kept across methods, so switching lands on the nearest filled square).
  const [pinned, setPinned] = useState<Cell | null>(DEFAULT_CELL);
  const [hovered, setHovered] = useState<Cell | null>(null);

  const arm = armOf(atlas, armId);
  const info = ARM_INFO.find((a) => a.id === armId);
  const elite = (hovered !== null ? eliteAt(arm, hovered) : undefined) ?? nearestElite(arm, pinned) ?? null;
  const active: Cell | null = elite?.cell ?? null;

  // The arm's fan is fetched when the arm is first shown, never before.
  const fan = useErrorFan(run.manifest.run_id, runPath, arm.fan_path);
  const draws = fan.status === "ready" && elite !== null ? (fan.value.elites.find((f) => f.index === elite.index)?.draws ?? null) : null;

  const squares = atlas.grid.reachable_cells.length;

  return (
    <section aria-label="Interactive demonstration" className="on-canvas overflow-hidden rounded-sm bg-canvas text-canvas-text">
      <div className="flex flex-col lg:grid lg:grid-cols-[minmax(0,1fr)_23.5rem]">
        <div className="order-2 border-b border-canvas-grid p-3 sm:p-4 lg:order-none lg:col-span-2 lg:row-start-1">
          <MethodSwitch
            atlas={atlas}
            value={armId}
            onChange={setArmId}
          />
          <div className="mt-3 grid items-end gap-x-10 gap-y-4 lg:grid-cols-[minmax(0,1fr)_auto]">
            <p className="max-w-[62ch] text-[0.8125rem] leading-snug sm:text-sm">
              <span className="font-medium text-canvas-strong">{info?.name}.</span> {info?.line}
            </p>
            <div>
              <p className="mb-1.5 text-xs text-canvas-text lg:text-right">
                <Term tip="Each method ran 16 times with different random seeds and the same budget; these are the averages.">Average of 16 runs</Term>
              </p>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
            <Stat
              label="Expected survivors"
              tip="How many of the catalog's trajectories we expect to still come home after a launch error: each counts by its survival rate, and the rates are added up."
              value={fixed(mean(arm, "p1"), 1)}
            />
            <Stat
              label="Unseen requirements met"
              tip="The share of 62 mission requirements, hidden from every method, that the catalog can meet with at least one of its trajectories."
              value={percent(mean(arm, "p2"))}
            />
            <Stat
              label="Squares filled"
              tip={`How many of the ${squares} reachable squares hold a trajectory.`}
              value={`${fixed(mean(arm, "elites"), 1)} of ${squares}`}
            />
            <Stat
              label="Survival per trajectory"
              tip="The average survival of the trajectories the method kept."
              value={percent(mean(arm, "mean_s"))}
            />
              </dl>
            </div>
          </div>
        </div>

        <div className="order-1 flex min-h-0 flex-col lg:order-none lg:col-start-1 lg:row-start-2">
          <OrbitStage
            atlas={atlas}
            elite={elite}
            draws={draws}
            emptyNote="In this run the method found no free return in any square."
          />
          {fan.status === "error" && (
            <div className="px-4 pb-4 sm:px-5">
              <ErrorState error={fan.error} onRetry={fan.retry} tone="canvas" />
            </div>
          )}
        </div>

        <div className="order-3 border-t border-canvas-grid p-4 sm:p-5 lg:order-none lg:col-start-2 lg:row-start-2 lg:border-t-0 lg:border-l">
          <p className="mb-3 text-sm text-canvas-text">
            <span className="font-medium text-canvas-strong">Atlas of free returns</span>
            {" · "}
            <Term tip="The atlas shows one run of each method: seed 1001, chosen in advance. On this single run differential evolution scores higher than MAP-Elites, so the 16-run averages below are the evidence, not one run.">
              one example run
            </Term>
          </p>
          <AtlasGrid
            atlas={atlas}
            arm={arm}
            active={active}
            onHover={setHovered}
            onSelect={(cell) => {
              setPinned(cell);
              setHovered(null);
            }}
          />
        </div>
      </div>
    </section>
  );
}
