import { useMemo, useState, type KeyboardEvent, type ReactNode } from "react";
import { fixed, km } from "../../lib/format";
import type { LoadedRun, TrajectoryEvent } from "../../lib/data/types";
import { useElementWidth } from "../../lib/useElementWidth";
import { bodyGlyph, CAMERA_LABEL, cameraBounds, EARTH_FILL, MOON_FILL, type Camera } from "../../lib/viewer/scene";
import { fitTransform, type Insets } from "../../lib/viewer/transform";
import { EventGlyph, MIN_SHELL_GAP_PX, OrbitCanvas, shellPixels } from "./OrbitCanvas";
import { handlePlaybackKey, PlaybackControls, type Speed } from "./PlaybackControls";
import { usePlaybackClock, usePlaybackState, usePlaybackStore, type PlaybackStore } from "./playback";
import { CONVENTION_LABEL } from "./labels";
import { StateReadout } from "./StateReadout";
import { eventDescription, eventLabel, isFreeReturn, type TrajectoryStyle } from "./trajectoryStyle";
import { SECONDS_PER_DAY, unitsFor, type Units } from "./units";

function canvasHeight(width: number): number {
  return Math.round(Math.min(620, Math.max(320, width * (width < 640 ? 0.95 : 0.62))));
}

function insetsFor(width: number): Insets {
  return width < 640 ? { top: 30, right: 12, bottom: 40, left: 46 } : { top: 34, right: 18, bottom: 44, left: 54 };
}

const SWATCH = "h-3 w-7 shrink-0";

function LegendItem({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <li className="flex items-start gap-3">
      <svg className={`${SWATCH} mt-1.5`} viewBox="0 0 28 12" aria-hidden="true">
        {icon}
      </svg>
      <span>{children}</span>
    </li>
  );
}

interface LegendProps {
  run: LoadedRun;
  style: TrajectoryStyle;
  earthToScale: boolean;
  moonToScale: boolean;
  shell: { gap: number } | null;
}

function Legend({ run, style, earthToScale, moonToScale, shell }: LegendProps) {
  const { trajectory, manifest } = run;
  const returnShell = manifest.solver.return_shell;
  // The legend sits on paper, so the trajectory swatch uses ink with the trajectory's dash pattern.
  return (
    <ul className="space-y-2 text-[0.92rem]">
      <LegendItem icon={<line x1="1" x2="27" y1="6" y2="6" stroke="var(--color-ink)" strokeWidth="1.5" strokeDasharray={style.dash} strokeLinecap="round" />}>
        Trajectory ({style.pattern}), {trajectory.length.toLocaleString("en-US")} computed samples
      </LegendItem>
      <LegendItem
        icon={
          <>
            <line x1="2" x2="20" y1="6" y2="6" stroke="var(--color-accent)" strokeWidth="3" strokeLinecap="round" />
            <circle cx="22" cy="6" r="3" fill="var(--color-accent)" stroke="var(--color-accent-ink)" strokeWidth="1" />
          </>
        }
      >
        Current sample and recent path
      </LegendItem>
      {trajectory.events.length > 0 && (
        <li className="flex items-start gap-3">
          <svg className={`${SWATCH} mt-1.5`} viewBox="0 0 28 12" aria-hidden="true">
            <rect width="28" height="12" fill="var(--color-canvas)" rx="1" />
            {trajectory.events.slice(0, 3).map((event, i) => (
              <EventGlyph key={event.name} name={event.name} cx={5 + i * 9} cy={6} size={3.2} />
            ))}
          </svg>
          <span>Events recorded in the trajectory file: {trajectory.events.map((e) => eventLabel(e.name)).join(", ")}</span>
        </li>
      )}
      <LegendItem
        icon={
          <>
            <circle cx="8" cy="6" r="5" fill={EARTH_FILL} />
            <circle cx="21" cy="6" r="3" fill={MOON_FILL} stroke="var(--color-graphite)" strokeWidth="0.6" />
          </>
        }
      >
        Earth and Moon at their model positions,{" "}
        {earthToScale && moonToScale ? "drawn at true radius" : "a dashed ring marks a body enlarged to stay visible"}
      </LegendItem>
      {returnShell !== null && (
        <LegendItem icon={<line x1="1" x2="27" y1="6" y2="6" stroke="#c4553f" strokeWidth="1.5" strokeDasharray="4 3" />}>
          Entry-interface shell, {fixed(returnShell.altitude_km, 2)} km above the surface.{" "}
          {shell !== null && shell.gap >= MIN_SHELL_GAP_PX
            ? `Drawn to scale: ${fixed(shell.gap, 1)} px above the Earth at this zoom.`
            : `Not drawn here: at this zoom it would sit ${shell === null ? "under a pixel" : `${fixed(shell.gap, 2)} px`} above the Earth's surface.`}
        </LegendItem>
      )}
      <LegendItem icon={<path d="M10 2l8 8M18 2l-8 8" stroke="var(--color-ink)" strokeWidth="1.4" />}>L1 Lagrange point, computed from μ</LegendItem>
    </ul>
  );
}

/** Index of the recorded sample closest in time to t. */
function nearestSample(times: Float64Array, t: number): number {
  let best = 0;
  for (let i = 1; i < times.length; i++) {
    if (Math.abs((times[i] ?? 0) - t) < Math.abs((times[best] ?? 0) - t)) best = i;
  }
  return best;
}

function EventList({ run, store, units }: { run: LoadedRun; store: PlaybackStore; units: Units }) {
  const { trajectory } = run;
  const { index } = usePlaybackState(store);
  const targets = useMemo(() => trajectory.events.map((e) => nearestSample(trajectory.t, e.t)), [trajectory]);
  return (
    <div className="border-t border-canvas-grid px-2 py-2">
      <p id="events-title" className="px-2 pb-1.5 text-sm text-canvas-text">
        Events: go to the nearest recorded sample
      </p>
      <ol aria-labelledby="events-title" className="grid gap-1 sm:grid-cols-3">
        {trajectory.events.map((event: TrajectoryEvent, i) => {
          const target = targets[i] ?? 0;
          const here = index === target;
          return (
            <li key={event.name}>
              <button
                type="button"
                onClick={() => { store.seek(target); }}
                aria-current={here ? "step" : undefined}
                title={eventDescription(event.name)}
                className={`grid w-full grid-cols-[1.25rem_1fr] items-center gap-x-2 rounded-sm border px-2.5 py-1.5 text-left hover:bg-canvas-grid ${
                  here ? "border-accent" : "border-canvas-grid"
                }`}
              >
                <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true">
                  <EventGlyph name={event.name} cx={8} cy={8} size={4.6} />
                </svg>
                <span className="flex items-baseline justify-between gap-2">
                  <span className="font-medium text-canvas-strong">{eventLabel(event.name)}</span>
                  <span className="numeric text-sm whitespace-nowrap text-canvas-text">day {fixed(units.days(event.t), 4)}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

interface Props {
  run: LoadedRun;
  style: TrajectoryStyle;
}

export function OrbitViewer({ run, style }: Props) {
  const { manifest, trajectory } = run;
  const { model } = manifest;
  const freeReturn = isFreeReturn(manifest);
  const cameras: Camera[] = freeReturn ? ["system", "earth", "moon"] : ["region", "system"];
  const units = useMemo(() => unitsFor(model), [model]);
  const store = usePlaybackStore(trajectory.length);
  const [speed, setSpeed] = useState<Speed>(1);
  const [camera, setCamera] = useState<Camera>(cameras[0] ?? "system");
  usePlaybackClock(store, trajectory.t, (speed * SECONDS_PER_DAY) / model.time_scale_seconds);

  const [frameRef, width] = useElementWidth<HTMLDivElement>(880);
  const transform = useMemo(
    () => fitTransform(cameraBounds(camera, model, trajectory), width, canvasHeight(width), insetsFor(width)),
    [camera, model, trajectory, width],
  );
  const earth = bodyGlyph(transform, model.earth.x, model.earth.y, model.earth.radius_km, model.length_scale_km);
  const moon = bodyGlyph(transform, model.moon.x, model.moon.y, model.moon.radius_km, model.length_scale_km);
  const shell = shellPixels(run, transform);

  const totalDays = units.days((trajectory.t[trajectory.length - 1] ?? 0) - (trajectory.t[0] ?? 0));
  const describeSample = (index: number): string =>
    `Day ${fixed(units.days(trajectory.t[index] ?? 0), 2)} of ${fixed(totalDays, 2)}, sample ${index + 1} of ${trajectory.length}`;

  const xExtent = manifest.metrics.x_extent_km;
  const yExtent = manifest.metrics.y_extent_km;
  const eventText = trajectory.events.map((e) => `${eventLabel(e.name)} at day ${fixed(units.days(e.t), 2)}`).join(", ");
  const description =
    `Trajectory ${manifest.run_id} in the rotating Earth–Moon frame, origin at the barycenter, x axis toward the Moon. ` +
    `Earth is at x = ${fixed(model.earth.x, 5)} and the Moon at x = ${fixed(model.moon.x, 5)} normalized length units. ` +
    `${trajectory.length} samples spanning ${fixed(totalDays, 2)} days` +
    (xExtent && yExtent ? `; the orbit spans ${km(xExtent.value)} in x and ${km(yExtent.value)} in y.` : ".") +
    (eventText !== "" ? ` Marked events: ${eventText}.` : "");

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    handlePlaybackKey(store, event);
  };

  return (
    <div className="grid gap-8 lg:grid-cols-12 lg:gap-10">
      <div className="on-canvas min-w-0 self-start overflow-hidden rounded-sm bg-canvas text-canvas-strong lg:col-span-8">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-canvas-grid px-4 py-2.5">
          <p className="text-sm text-canvas-text">Rotating Earth–Moon frame, origin at the barycenter</p>
          <div role="group" aria-label="View" className="flex rounded-sm border border-canvas-axis p-0.5 text-sm">
            {cameras.map((c) => (
              <button
                key={c}
                type="button"
                aria-pressed={camera === c}
                onClick={() => { setCamera(c); }}
                className={`rounded-[1px] px-2.5 py-1 ${camera === c ? "bg-canvas-strong font-medium text-canvas" : "text-canvas-text hover:text-canvas-strong"}`}
              >
                {CAMERA_LABEL[c]}
              </button>
            ))}
          </div>
        </div>
        <div
          ref={frameRef}
          tabIndex={0}
          role="group"
          aria-roledescription="orbit viewer"
          aria-label="Orbit viewer. Space plays or pauses, left and right arrows step one sample, Shift with an arrow steps ten, Home returns to the start."
          onKeyDown={onKeyDown}
          className="relative"
        >
          <OrbitCanvas
            run={run}
            transform={transform}
            store={store}
            style={style}
            camera={camera}
            title={`${CAMERA_LABEL[camera]} view of the computed trajectory`}
            description={description}
          />
        </div>
        <div className="border-t border-canvas-grid">
          <PlaybackControls store={store} describeSample={describeSample} speed={speed} onSpeedChange={setSpeed} />
        </div>
        {trajectory.events.length > 0 && <EventList run={run} store={store} units={units} />}
      </div>

      <aside aria-label="Current state, events, legend and units" className="space-y-8 lg:col-span-4">
        <section aria-labelledby="readout-title">
          <h3 id="readout-title" className="sr-only">
            Current state
          </h3>
          <StateReadout store={store} trajectory={trajectory} units={units} model={freeReturn ? model : null} />
          <p className="mt-3 text-sm text-graphite">
            Rotating-frame state of the recorded sample; playback steps through recorded samples only and interpolates nothing.
          </p>
        </section>
        <section aria-labelledby="legend-title">
          <h3 id="legend-title" className="mb-3 font-sans text-sm font-medium text-graphite">
            Legend
          </h3>
          <Legend run={run} style={style} earthToScale={earth.toScale} moonToScale={moon.toScale} shell={shell} />
        </section>
        <details className="group">
          <summary className="cursor-pointer font-sans text-sm font-medium text-graphite marker:text-quiet hover:text-ink">
            Units and scales
          </summary>
          <div className="mt-2">
          <dl className="numeric grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[0.92rem]">
            <dt>1 LU</dt>
            <dd>{km(model.length_scale_km, 3)}</dd>
            <dt>1 TU</dt>
            <dd>
              {fixed(model.time_scale_seconds, 3)} s = {fixed(units.timeUnitDays, 4)} d
            </dd>
            <dt>1 VU</dt>
            <dd>{fixed(model.velocity_scale_km_s, 6)} km/s</dd>
          </dl>
          <p className="mt-3 text-sm text-graphite">
            Scales are read from the run manifest
            {model.system_name !== null ? ` (constant set “${model.system_name}”)` : ""}. Length unit convention:{" "}
            {CONVENTION_LABEL[model.length_scale_convention]}.
            {Math.abs(model.length_scale_km - 384400) > 0.5
              ? ` That is ${km(model.length_scale_km, 0)}, not the 384,400 km mean Earth–Moon distance, so every kilometre value here depends on that convention.`
              : ""}
          </p>
          </div>
        </details>
      </aside>
    </div>
  );
}
