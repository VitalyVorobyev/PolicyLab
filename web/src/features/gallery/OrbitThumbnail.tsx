import { useMemo } from "react";
import type { LoadedRun } from "../../lib/data/types";
import { useCatalog, useRun } from "../../lib/data/useResource";
import { bodyGlyph, cameraBounds, EARTH_FILL, MOON_FILL, samplePath } from "../../lib/viewer/scene";
import { fitTransform } from "../../lib/viewer/transform";

const W = 640;
const H = 300;

function Plot({ run }: { run: LoadedRun }) {
  const { manifest, trajectory } = run;
  const { model } = manifest;
  const scene = useMemo(() => {
    const t = fitTransform(cameraBounds("system", model, trajectory), W, H, { top: 36, right: 28, bottom: 36, left: 28 });
    return {
      path: samplePath(t, trajectory.x, trajectory.y),
      earth: bodyGlyph(t, model.earth.x, model.earth.y, model.earth.radius_km, model.length_scale_km),
      moon: bodyGlyph(t, model.moon.x, model.moon.y, model.moon.radius_km, model.length_scale_km),
      axisY: t.toScreenY(0),
    };
  }, [model, trajectory]);
  return (
    <>
      <line x1={0} x2={W} y1={scene.axisY} y2={scene.axisY} stroke="var(--color-canvas-grid)" />
      <path d={scene.path} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={1.4} />
      <circle cx={scene.earth.cx} cy={scene.earth.cy} r={scene.earth.r} fill={EARTH_FILL} />
      <circle cx={scene.moon.cx} cy={scene.moon.cy} r={scene.moon.r} fill={MOON_FILL} />
      <text x={scene.earth.cx} y={scene.earth.cy + scene.earth.r + 22} textAnchor="middle" className="fill-canvas-text text-[17px]">
        Earth
      </text>
      <text x={scene.moon.cx} y={scene.moon.cy + 24} textAnchor="middle" className="fill-canvas-text text-[17px]">
        Moon
      </text>
    </>
  );
}

function LoadedThumbnail({ runId, path }: { runId: string; path: string }) {
  const run = useRun(runId, path);
  return run.status === "ready" ? <Plot run={run.value} /> : null;
}

/**
 * The actual published reference orbit, drawn to scale in the rotating frame. If the data is
 * unavailable the canvas stays empty rather than showing a decorative stand-in.
 */
export function OrbitThumbnail() {
  const catalog = useCatalog();
  const run = catalog.status === "ready" ? catalog.value.studies.find((s) => s.study_id === "gravity-atlas")?.runs[0] : undefined;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block h-auto w-full"
      role="img"
      aria-label="The computed Earth–Moon L1 Lyapunov orbit, drawn to scale in the rotating frame, between Earth on the left and the Moon on the right."
    >
      <rect width={W} height={H} fill="var(--color-canvas)" />
      {run !== undefined && <LoadedThumbnail runId={run.run_id} path={run.path} />}
    </svg>
  );
}
