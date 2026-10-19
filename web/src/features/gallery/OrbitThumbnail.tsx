import { useMemo } from "react";
import { fixed } from "../../lib/format";
import type { CatalogRun, LoadedRun } from "../../lib/data/types";
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
      events: trajectory.events.map((e) => ({ name: e.name, x: t.toScreenX(e.x), y: t.toScreenY(e.y) })),
    };
  }, [model, trajectory]);
  return (
    <>
      <line x1={0} x2={W} y1={scene.axisY} y2={scene.axisY} stroke="var(--color-canvas-grid)" />
      <path d={scene.path} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={1.4} />
      <circle cx={scene.earth.cx} cy={scene.earth.cy} r={scene.earth.r} fill={EARTH_FILL} />
      <circle cx={scene.moon.cx} cy={scene.moon.cy} r={scene.moon.r} fill={MOON_FILL} />
      {scene.events
        .filter((e) => e.name === "perilune")
        .map((e) => (
          <path key={e.name} d={`M${e.x} ${e.y - 6}l6 6l-6 6l-6 -6z`} fill="var(--color-canvas)" stroke="var(--color-canvas-strong)" strokeWidth={1.5} />
        ))}
      <text x={scene.earth.cx} y={scene.earth.cy + scene.earth.r + 22} textAnchor="middle" className="fill-canvas-text text-[17px]">
        Earth
      </text>
      <text x={scene.moon.cx - 16} y={scene.moon.cy + 6} textAnchor="end" className="fill-canvas-text text-[17px]">
        Moon
      </text>
    </>
  );
}

/** The study's featured trajectory: the first published trajectory run in the catalog. */
export function useFeaturedRun(): CatalogRun | undefined {
  const catalog = useCatalog();
  return catalog.status === "ready"
    ? catalog.value.studies.find((s) => s.study_id === "gravity-atlas")?.runs.find((r) => r.kind === "trajectory")
    : undefined;
}

/** A caption read from the featured run's manifest; nothing is stated that the manifest does not record. */
export function FeaturedCaption({ run }: { run: CatalogRun }) {
  const loaded = useRun(run.run_id, run.path);
  if (loaded.status !== "ready") return <>{run.label}, in the rotating Earth–Moon frame.</>;
  const m = loaded.value.manifest.metrics;
  const facts = [
    m.departure_delta_v_km_s !== undefined ? `one ${fixed(m.departure_delta_v_km_s.value, 3)} km/s impulse` : null,
    m.perilune_altitude_km !== undefined ? `a ${fixed(m.perilune_altitude_km.value, 0)} km perilune` : null,
    m.return_shell_crossing_time_days !== undefined ? `back at the entry interface on day ${fixed(m.return_shell_crossing_time_days.value, 2)}` : null,
  ].filter((f): f is string => f !== null);
  return (
    <>
      {run.label}{facts.length > 0 ? `: ${facts.join(", ")}` : ""}. Rotating Earth–Moon frame, drawn to scale.
    </>
  );
}

function LoadedThumbnail({ runId, path }: { runId: string; path: string }) {
  const run = useRun(runId, path);
  return run.status === "ready" ? <Plot run={run.value} /> : null;
}

/**
 * The actual published featured trajectory, drawn to scale in the rotating frame. If the data is
 * unavailable the canvas stays empty rather than showing a decorative stand-in.
 */
export function OrbitThumbnail() {
  const run = useFeaturedRun();
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block h-auto w-full"
      role="img"
      aria-label={`${run?.label ?? "The featured computed trajectory"}, drawn to scale in the rotating Earth–Moon frame, with Earth on the left and the Moon on the right.`}
    >
      <rect width={W} height={H} fill="var(--color-canvas)" />
      {run !== undefined && <LoadedThumbnail runId={run.run_id} path={run.path} />}
    </svg>
  );
}
