import { memo, useMemo } from "react";
import { fixed } from "../../lib/format";
import type { LoadedRun, Trajectory } from "../../lib/data/types";
import { lagrangeL1 } from "../../lib/viewer/lagrange";
import { bodyGlyph, cameraBounds, EARTH_FILL, MOON_FILL, samplePath, type BodyGlyph } from "../../lib/viewer/scene";
import { fitTransform, niceStep, ticks, type Transform } from "../../lib/viewer/transform";
import { usePlaybackState, type PlaybackStore } from "./playback";

/** Samples drawn in the highlighted recent path behind the current sample. */
const TRAIL_SAMPLES = 30;

function Body({ glyph, fill, label, below }: { glyph: BodyGlyph; fill: string; label: string; below: number }) {
  return (
    <g>
      <circle cx={glyph.cx} cy={glyph.cy} r={glyph.r} fill={fill} />
      {!glyph.toScale && <circle cx={glyph.cx} cy={glyph.cy} r={glyph.r + 2.5} fill="none" stroke={fill} strokeWidth={0.8} strokeDasharray="1.5 1.5" />}
      <text x={glyph.cx} y={glyph.cy + Math.max(glyph.r, 4) + below} textAnchor="middle" className="fill-canvas-strong text-[13px] font-medium">
        {label}
      </text>
    </g>
  );
}

function Axes({ transform, lengthScaleKm }: { transform: Transform; lengthScaleKm: number }) {
  const { visible, insets, width, height } = transform;
  const plotRight = width - insets.right;
  const plotBottom = height - insets.bottom;
  const target = Math.max(3, Math.round((width - insets.left - insets.right) / 110));
  const step = niceStep(visible.xMax - visible.xMin, target);
  const xs = ticks(visible.xMin, visible.xMax, step);
  const ys = ticks(visible.yMin, visible.yMax, step);
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));

  // Scale bar: a 1/2/5 × 10^k km length close to a fifth of the plot width.
  const visibleKm = (visible.xMax - visible.xMin) * lengthScaleKm;
  const barKm = niceStep(visibleKm / 5, 1);
  const barPx = (barKm / lengthScaleKm) * transform.scale;
  const barX = plotRight - barPx - 8;
  const barY = plotBottom - 14;

  return (
    <g className="text-[11px]" aria-hidden="true">
      {xs.map((x) => (
        <g key={`x${x}`}>
          <line x1={transform.toScreenX(x)} x2={transform.toScreenX(x)} y1={insets.top} y2={plotBottom} stroke="var(--color-canvas-grid)" />
          <text x={transform.toScreenX(x)} y={plotBottom + 16} textAnchor="middle" className="numeric fill-canvas-text">
            {fixed(x, decimals)}
          </text>
        </g>
      ))}
      {ys.map((y) => (
        <g key={`y${y}`}>
          <line x1={insets.left} x2={plotRight} y1={transform.toScreenY(y)} y2={transform.toScreenY(y)} stroke="var(--color-canvas-grid)" />
          <text x={insets.left - 6} y={transform.toScreenY(y) + 4} textAnchor="end" className="numeric fill-canvas-text">
            {fixed(y, decimals)}
          </text>
        </g>
      ))}
      <rect x={insets.left} y={insets.top} width={plotRight - insets.left} height={plotBottom - insets.top} fill="none" stroke="var(--color-canvas-axis)" />
      <text x={plotRight} y={plotBottom + 32} textAnchor="end" className="fill-canvas-text">
        x (LU)
      </text>
      <text x={insets.left - 6} y={insets.top - 8} textAnchor="end" className="fill-canvas-text">
        y (LU)
      </text>
      <g>
        <rect x={barX - 8} y={barY - 18} width={barPx + 16} height={28} fill="var(--color-canvas)" opacity={0.85} />
        <path d={`M${barX} ${barY - 4}v4h${barPx}v-4`} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={1.2} />
        <text x={barX + barPx / 2} y={barY - 7} textAnchor="middle" className="numeric fill-canvas-strong">
          {barKm.toLocaleString("en-US")} km
        </text>
      </g>
    </g>
  );
}

/** Everything that does not change during playback. Memoized so frames never re-render it. */
const StaticScene = memo(function StaticScene({ run, transform }: { run: LoadedRun; transform: Transform }) {
  const { model } = run.manifest;
  const { trajectory } = run;
  const scene = useMemo(() => {
    const l1 = lagrangeL1(model.mu);
    return {
      path: samplePath(transform, trajectory.x, trajectory.y),
      earth: bodyGlyph(transform, model.earth.x, model.earth.y, model.earth.radius_km, model.length_scale_km),
      moon: bodyGlyph(transform, model.moon.x, model.moon.y, model.moon.radius_km, model.length_scale_km),
      l1: { x: transform.toScreenX(l1), y: transform.toScreenY(0) },
    };
  }, [model, trajectory, transform]);
  const inView = (x: number, y: number): boolean =>
    x >= transform.insets.left && x <= transform.width - transform.insets.right && y >= transform.insets.top && y <= transform.height - transform.insets.bottom;

  return (
    <>
      <Axes transform={transform} lengthScaleKm={model.length_scale_km} />
      <path d={scene.path} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={1.25} strokeOpacity={0.85} strokeLinejoin="round" />
      {inView(scene.l1.x, scene.l1.y) && (
        <g aria-hidden="true">
          <path d={`M${scene.l1.x - 4} ${scene.l1.y - 4}l8 8M${scene.l1.x + 4} ${scene.l1.y - 4}l-8 8`} stroke="var(--color-canvas-strong)" strokeWidth={1.3} />
          <text x={scene.l1.x - 8} y={scene.l1.y - 7} textAnchor="end" className="fill-canvas-strong text-[12px] font-medium">
            L1
          </text>
        </g>
      )}
      {inView(scene.earth.cx, scene.earth.cy) && <Body glyph={scene.earth} fill={EARTH_FILL} label="Earth" below={16} />}
      {inView(scene.moon.cx, scene.moon.cy) && <Body glyph={scene.moon} fill={MOON_FILL} label="Moon" below={16} />}
    </>
  );
});

/**
 * Locator inset for the close-up: the whole Earth–Moon system at small scale, with the region
 * shown in the main view outlined. It has its own fixed-aspect transform; the main view's
 * transform is unaffected.
 */
const Locator = memo(function Locator({ run, main }: { run: LoadedRun; main: Transform }) {
  const { model } = run.manifest;
  const { trajectory } = run;
  const plotW = main.width - main.insets.left - main.insets.right;
  const w = Math.round(Math.min(220, Math.max(112, plotW * 0.3)));
  const h = Math.round(w * 0.36);
  const t = useMemo(() => fitTransform(cameraBounds("system", model, trajectory), w, h, { top: 6, right: 8, bottom: 6, left: 8 }), [model, trajectory, w, h]);
  const earth = bodyGlyph(t, model.earth.x, model.earth.y, model.earth.radius_km, model.length_scale_km);
  const moonX = t.toScreenX(model.moon.x);
  const v = main.visible;
  const rx = t.toScreenX(v.xMin);
  const ry = t.toScreenY(v.yMax);
  return (
    <g transform={`translate(${main.width - main.insets.right - w - 10} ${main.insets.top + 10})`} aria-hidden="true">
      <rect width={w} height={h + 16} fill="var(--color-canvas)" stroke="var(--color-canvas-axis)" />
      <line x1={0} x2={w} y1={t.toScreenY(0)} y2={t.toScreenY(0)} stroke="var(--color-canvas-grid)" />
      <path d={samplePath(t, trajectory.x, trajectory.y)} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={0.8} />
      <circle cx={earth.cx} cy={earth.cy} r={Math.max(earth.r, 1.5)} fill={EARTH_FILL} />
      <circle cx={moonX} cy={t.toScreenY(0)} r={1.5} fill={MOON_FILL} />
      <rect
        x={rx}
        y={ry}
        width={(v.xMax - v.xMin) * t.scale}
        height={(v.yMax - v.yMin) * t.scale}
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth={1}
      />
      <text x={6} y={h + 11} className="fill-canvas-text text-[10px]">
        Whole system
      </text>
    </g>
  );
});

/** The only part of the drawing that changes per frame. */
function CurrentSample({ store, trajectory, transform }: { store: PlaybackStore; trajectory: Trajectory; transform: Transform }) {
  const { index } = usePlaybackState(store);
  const trail = samplePath(transform, trajectory.x, trajectory.y, index - TRAIL_SAMPLES, index + 1);
  const cx = transform.toScreenX(trajectory.x[index] ?? 0);
  const cy = transform.toScreenY(trajectory.y[index] ?? 0);
  return (
    <g aria-hidden="true">
      {index > 0 && <path d={trail} fill="none" stroke="var(--color-accent)" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />}
      <circle cx={cx} cy={cy} r={7.5} fill="none" stroke="var(--color-accent)" strokeWidth={1.5} />
      <circle cx={cx} cy={cy} r={3.5} fill="var(--color-accent-strong)" />
    </g>
  );
}

interface Props {
  run: LoadedRun;
  transform: Transform;
  store: PlaybackStore;
  title: string;
  description: string;
  /** Draw the whole-system locator inset (used by the close-up view). */
  locator: boolean;
}

export function OrbitCanvas({ run, transform, store, title, description, locator }: Props) {
  return (
    <svg
      viewBox={`0 0 ${transform.width} ${transform.height}`}
      width={transform.width}
      height={transform.height}
      className="block max-w-full select-none"
      role="img"
      aria-labelledby="orbit-title orbit-desc"
    >
      <title id="orbit-title">{title}</title>
      <desc id="orbit-desc">{description}</desc>
      <StaticScene run={run} transform={transform} />
      {locator && <Locator run={run} main={transform} />}
      <CurrentSample store={store} trajectory={run.trajectory} transform={transform} />
    </svg>
  );
}
