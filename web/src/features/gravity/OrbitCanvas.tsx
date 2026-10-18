import { memo, useMemo } from "react";
import { fixed } from "../../lib/format";
import type { LoadedRun, Trajectory, TrajectoryEvent } from "../../lib/data/types";
import { lagrangeL1 } from "../../lib/viewer/lagrange";
import { bodyGlyph, cameraBounds, EARTH_FILL, MOON_FILL, samplePath, type BodyGlyph, type Camera } from "../../lib/viewer/scene";
import { fitTransform, niceStep, ticks, type Transform } from "../../lib/viewer/transform";
import { usePlaybackState, type PlaybackStore } from "./playback";
import { eventLabel, type TrajectoryStyle } from "./trajectoryStyle";
import { SECONDS_PER_DAY } from "./units";

/** Samples drawn in the highlighted recent path behind the current sample. */
const TRAIL_SAMPLES = 30;

/** Event marker shapes: shape, not only color, tells the events apart. */
export function EventGlyph({ name, cx, cy, size = 5 }: { name: string; cx: number; cy: number; size?: number }) {
  const common = { fill: "var(--color-canvas)", stroke: "var(--color-canvas-strong)", strokeWidth: 1.5 };
  if (name === "perilune") {
    return <path d={`M${cx} ${cy - size}L${cx + size} ${cy}L${cx} ${cy + size}L${cx - size} ${cy}Z`} {...common} />;
  }
  if (name === "return_shell_crossing") {
    const h = size * 1.1;
    return <path d={`M${cx - h} ${cy - h * 0.75}L${cx + h} ${cy - h * 0.75}L${cx} ${cy + h}Z`} {...common} />;
  }
  if (name === "departure") return <circle cx={cx} cy={cy} r={size * 0.85} {...common} />;
  return <rect x={cx - size * 0.75} y={cy - size * 0.75} width={size * 1.5} height={size * 1.5} {...common} />;
}

function Body({ glyph, fill, label, below }: { glyph: BodyGlyph; fill: string; label: string | null; below: number }) {
  return (
    <g>
      <circle cx={glyph.cx} cy={glyph.cy} r={glyph.r} fill={fill} />
      {!glyph.toScale && <circle cx={glyph.cx} cy={glyph.cy} r={glyph.r + 2.5} fill="none" stroke={fill} strokeWidth={0.8} strokeDasharray="1.5 1.5" />}
      {label !== null && (
        <text x={glyph.cx} y={glyph.cy + Math.max(glyph.r, 4) + below} textAnchor="middle" className="fill-canvas-strong text-[13px] font-medium">
          {label}
        </text>
      )}
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

/** Screen radius (px) of the return shell and its gap above the Earth's surface at this zoom. */
export function shellPixels(run: LoadedRun, transform: Transform): { radius: number; gap: number } | null {
  const { model, solver } = run.manifest;
  if (solver.return_shell === null) return null;
  const perKm = transform.scale / model.length_scale_km;
  return { radius: solver.return_shell.radius_km * perKm, gap: solver.return_shell.altitude_km * perKm };
}

/** The shell is drawn only when it is visibly separate from the Earth's surface. */
export const MIN_SHELL_GAP_PX = 0.75;

interface EventMark {
  event: TrajectoryEvent;
  x: number;
  y: number;
  /** Unit vector from the nearer primary towards the event, for label placement. */
  ux: number;
  uy: number;
  label: string;
}

function eventMarks(run: LoadedRun, transform: Transform): EventMark[] {
  const { model } = run.manifest;
  const days = model.time_scale_seconds / SECONDS_PER_DAY;
  return run.trajectory.events.map((event) => {
    const x = transform.toScreenX(event.x);
    const y = transform.toScreenY(event.y);
    const dEarth = Math.hypot(event.x - model.earth.x, event.y - model.earth.y);
    const dMoon = Math.hypot(event.x - model.moon.x, event.y - model.moon.y);
    const body = dEarth < dMoon ? model.earth : model.moon;
    const bx = transform.toScreenX(body.x);
    const by = transform.toScreenY(body.y);
    const norm = Math.hypot(x - bx, y - by) || 1;
    return { event, x, y, ux: (x - bx) / norm, uy: (y - by) / norm, label: `${eventLabel(event.name)}, day ${fixed(event.t * days, 2)}` };
  });
}

/** Everything that does not change during playback. Memoized so frames never re-render it. */
const StaticScene = memo(function StaticScene({
  run,
  transform,
  style,
  camera,
}: {
  run: LoadedRun;
  transform: Transform;
  style: TrajectoryStyle;
  camera: Camera;
}) {
  const { model } = run.manifest;
  const { trajectory } = run;
  const scene = useMemo(() => {
    const l1 = lagrangeL1(model.mu);
    return {
      path: samplePath(transform, trajectory.x, trajectory.y),
      earth: bodyGlyph(transform, model.earth.x, model.earth.y, model.earth.radius_km, model.length_scale_km),
      moon: bodyGlyph(transform, model.moon.x, model.moon.y, model.moon.radius_km, model.length_scale_km),
      l1: { x: transform.toScreenX(l1), y: transform.toScreenY(0) },
      events: eventMarks(run, transform),
      shell: shellPixels(run, transform),
    };
  }, [model, trajectory, transform, run]);
  const { insets } = transform;
  const inView = (x: number, y: number, pad = 0): boolean =>
    x >= insets.left - pad && x <= transform.width - insets.right + pad && y >= insets.top - pad && y <= transform.height - insets.bottom + pad;
  const showShell = scene.shell !== null && scene.shell.gap >= MIN_SHELL_GAP_PX;
  const freeReturn = scene.events.length > 0;
  // Near Earth, the Earth label sits at the centre of the (large) disk; elsewhere below it.
  const earthLabelInside = camera === "earth";
  const clipId = `plot-clip-${run.manifest.run_id}`;

  return (
    <>
      <defs>
        <clipPath id={clipId}>
          <rect x={insets.left} y={insets.top} width={transform.width - insets.left - insets.right} height={transform.height - insets.top - insets.bottom} />
        </clipPath>
      </defs>
      <Axes transform={transform} lengthScaleKm={model.length_scale_km} />
      <g clipPath={`url(#${clipId})`}>
        {inView(scene.earth.cx, scene.earth.cy, scene.earth.r) && (
          <Body glyph={scene.earth} fill={EARTH_FILL} label={earthLabelInside ? null : "Earth"} below={freeReturn ? 18 : 16} />
        )}
        {earthLabelInside && (
          <text x={scene.earth.cx} y={scene.earth.cy + 5} textAnchor="middle" className="fill-canvas text-[14px] font-medium">
            Earth
          </text>
        )}
        {showShell && scene.shell !== null && (
          <circle
            cx={scene.earth.cx}
            cy={scene.earth.cy}
            r={scene.shell.radius}
            fill="none"
            stroke="#f2a28f"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
        )}
        {showShell && scene.shell !== null && camera === "earth" && run.manifest.solver.return_shell !== null && (
          <g aria-hidden="true">
            <line
              x1={scene.earth.cx}
              x2={scene.earth.cx}
              y1={scene.earth.cy + scene.shell.radius + 2}
              y2={scene.earth.cy + scene.shell.radius + 14}
              stroke="#f2a28f"
              strokeWidth={1}
            />
            <text x={scene.earth.cx} y={scene.earth.cy + scene.shell.radius + 28} textAnchor="middle" className="fill-[#f2a28f] text-[12px]" stroke="var(--color-canvas)" strokeWidth={3} paintOrder="stroke">
              Entry-interface shell, {fixed(run.manifest.solver.return_shell.altitude_km, 2)} km up: {fixed(scene.shell.gap, 1)} px at this scale
            </text>
          </g>
        )}
        {inView(scene.moon.cx, scene.moon.cy, scene.moon.r) && (
          <Body glyph={scene.moon} fill={MOON_FILL} label={camera === "moon" ? null : "Moon"} below={16} />
        )}
        {camera === "moon" && (
          <text x={scene.moon.cx} y={scene.moon.cy + 5} textAnchor="middle" className="fill-canvas text-[14px] font-medium">
            Moon
          </text>
        )}
        <path
          d={scene.path}
          fill="none"
          stroke={style.stroke}
          strokeWidth={1.4}
          strokeOpacity={0.9}
          strokeDasharray={style.dash}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {inView(scene.l1.x, scene.l1.y) && (
          <g aria-hidden="true">
            <path d={`M${scene.l1.x - 4} ${scene.l1.y - 4}l8 8M${scene.l1.x + 4} ${scene.l1.y - 4}l-8 8`} stroke="var(--color-canvas-strong)" strokeWidth={1.3} />
            <text x={scene.l1.x - 8} y={scene.l1.y - 7} textAnchor="end" className="fill-canvas-strong text-[12px] font-medium">
              L1
            </text>
          </g>
        )}
        {scene.events.map((mark) =>
          inView(mark.x, mark.y) ? (
            <g key={mark.event.name} aria-hidden="true">
              <EventGlyph name={mark.event.name} cx={mark.x} cy={mark.y} />
            </g>
          ) : null,
        )}
      </g>
      {scene.events.map((mark) => {
        if (!inView(mark.x, mark.y)) return null;
        // At system scale the departure and entry sit on the Earth's disk; they are labeled in the near-Earth view.
        if (camera === "system" && mark.event.name !== "perilune") return null;
        const offset = 16;
        const estimated = mark.label.length * 6.6;
        let anchor: "start" | "middle" | "end" = mark.ux < -0.3 ? "end" : mark.ux > 0.3 ? "start" : "middle";
        let lx = mark.x + mark.ux * offset;
        let ly = mark.y + mark.uy * offset + 4;
        const plotLeft = insets.left + 4;
        const plotRight = transform.width - insets.right - 4;
        if (anchor === "start" && lx + estimated > plotRight) {
          anchor = "end";
          lx = Math.min(plotRight, mark.x - 8);
          ly = mark.y - 24;
        } else if (anchor === "end" && lx - estimated < plotLeft) {
          anchor = "start";
          lx = Math.max(plotLeft, mark.x + 10);
          ly = mark.y - 12;
        } else if (anchor === "middle") {
          lx = Math.min(plotRight - estimated / 2, Math.max(plotLeft + estimated / 2, lx));
        }
        return (
          <text
            key={`label-${mark.event.name}`}
            x={lx}
            y={ly}
            textAnchor={anchor}
            className="fill-canvas-strong text-[12px]"
            stroke="var(--color-canvas)"
            strokeWidth={3}
            paintOrder="stroke"
            aria-hidden="true"
          >
            {mark.label}
          </text>
        );
      })}
    </>
  );
});

/**
 * Locator inset for the close-up views: the whole Earth–Moon system at small scale, with the
 * region shown in the main view outlined. It has its own fixed-aspect transform; the main view's
 * transform is unaffected.
 */
const Locator = memo(function Locator({ run, main }: { run: LoadedRun; main: Transform }) {
  const { model } = run.manifest;
  const { trajectory } = run;
  const plotW = main.width - main.insets.left - main.insets.right;
  const w = Math.round(Math.min(220, Math.max(112, plotW * 0.3)));
  const bounds = cameraBounds("system", model, trajectory);
  const h = Math.round(Math.min(w * 0.7, Math.max(w * 0.3, (w * (bounds.yMax - bounds.yMin)) / (bounds.xMax - bounds.xMin))));
  const t = useMemo(() => fitTransform(bounds, w, h, { top: 6, right: 8, bottom: 6, left: 8 }), [bounds, w, h]);
  const earth = bodyGlyph(t, model.earth.x, model.earth.y, model.earth.radius_km, model.length_scale_km);
  const moonX = t.toScreenX(model.moon.x);
  const v = main.visible;
  const rx = t.toScreenX(v.xMin);
  const ry = t.toScreenY(v.yMax);
  const rw = Math.max(3, (v.xMax - v.xMin) * t.scale);
  const rh = Math.max(3, (v.yMax - v.yMin) * t.scale);
  return (
    <g transform={`translate(${main.width - main.insets.right - w - 10} ${main.insets.top + 10})`} aria-hidden="true">
      <rect width={w} height={h + 16} fill="var(--color-canvas)" stroke="var(--color-canvas-axis)" />
      <line x1={0} x2={w} y1={t.toScreenY(0)} y2={t.toScreenY(0)} stroke="var(--color-canvas-grid)" />
      <path d={samplePath(t, trajectory.x, trajectory.y)} fill="none" stroke="var(--color-canvas-strong)" strokeWidth={0.8} />
      <circle cx={earth.cx} cy={earth.cy} r={Math.max(earth.r, 1.5)} fill={EARTH_FILL} />
      <circle cx={moonX} cy={t.toScreenY(0)} r={1.5} fill={MOON_FILL} />
      <rect x={rx - (rw - (v.xMax - v.xMin) * t.scale) / 2} y={ry - (rh - (v.yMax - v.yMin) * t.scale) / 2} width={rw} height={rh} fill="none" stroke="var(--color-accent)" strokeWidth={1} />
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
  style: TrajectoryStyle;
  camera: Camera;
  title: string;
  description: string;
}

export function OrbitCanvas({ run, transform, store, style, camera, title, description }: Props) {
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
      <StaticScene run={run} transform={transform} style={style} camera={camera} />
      {camera !== "system" && <Locator run={run} main={transform} />}
      <CurrentSample store={store} trajectory={run.trajectory} transform={transform} />
    </svg>
  );
}
