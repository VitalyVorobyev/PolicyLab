import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Term } from "../../components/Tooltip";
import { fixed } from "../../lib/format";
import type { Atlas, AtlasElite, FanDraw } from "../../lib/data/types";
import { altitudeLabel, daysLabel, percent } from "./atlasModel";
import { buildScene, readColors, renderScene, type Colors } from "./orbitScene";

/** Seconds for one flight, and how long the end of the flight is held before it repeats. */
const FLIGHT_SECONDS = 14;
const HOLD_SECONDS = 1.6;

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function useSize(ref: React.RefObject<HTMLElement | null>): { w: number; h: number } {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (el === null || typeof ResizeObserver === "undefined") return;
    const update = (): void => {
      const r = el.getBoundingClientRect();
      setSize((s) => (Math.abs(s.w - r.width) < 0.5 && Math.abs(s.h - r.height) < 0.5 ? s : { w: r.width, h: r.height }));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => { observer.disconnect(); };
  }, [ref]);
  return size;
}

function PlayIcon({ playing }: { playing: boolean }) {
  return (
    <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true" fill="currentColor">
      {playing ? <path d="M3.5 2h3v12h-3zM9.5 2h3v12h-3z" /> : <path d="M4 2l10 6-10 6z" />}
    </svg>
  );
}

function Swatch({ kind }: { kind: "chosen" | "ok" | "fail" }) {
  return (
    <svg viewBox="0 0 22 10" className="h-2.5 w-[1.375rem] shrink-0" aria-hidden="true">
      <line x1="0" x2="22" y1="5" y2="5" strokeWidth={kind === "chosen" ? 2.4 : 1.2} stroke={kind === "chosen" ? "var(--color-canvas-strong)" : kind === "ok" ? "var(--color-fan-ok)" : "var(--color-fan-fail)"} />
      {kind === "chosen" && <circle cx="17" cy="5" r="3.4" fill="var(--color-accent)" stroke="var(--color-canvas-strong)" strokeWidth="1" />}
      {kind === "ok" && <circle cx="17" cy="5" r="2.3" fill="var(--color-fan-ok)" />}
      {kind === "fail" && <circle cx="17" cy="5" r="2.1" fill="none" stroke="var(--color-fan-fail)" strokeWidth="1.2" />}
    </svg>
  );
}

interface StageProps {
  atlas: Atlas;
  elite: AtlasElite | null;
  /** The 50 launch-error variants of the elite (null while they load). */
  draws: FanDraw[] | null;
  /** Shown when the selected method found no free return. */
  emptyNote?: string;
}

/**
 * The hero: the selected free return in the rotating Earth–Moon frame, with its launch-error
 * variants flown alongside as a fan. All paths are stored polylines; the browser only draws them
 * and interpolates positions between stored vertices.
 */
export function OrbitStage({ atlas, elite, draws, emptyNote }: StageProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sliderRef = useRef<HTMLInputElement>(null);
  const readoutRef = useRef<HTMLSpanElement>(null);
  const size = useSize(frameRef);
  const reduced = useMemo(prefersReducedMotion, []);
  const [playing, setPlaying] = useState(!reduced);
  const colors = useRef<Colors | null>(null);

  // Progress along the flight in [0, 1]; kept out of React state so animation never re-renders the page.
  const progress = useRef(0);
  const startedFor = useRef<AtlasElite | null>(null);
  if (startedFor.current === null && elite !== null) {
    // First frame: a moment after closest approach, when the variants have visibly spread. Reduced motion: the end.
    const peri = elite.perilune.t_days / elite.return_time_days;
    progress.current = reduced ? 1 : Math.min(0.95, peri + 0.06);
    startedFor.current = elite;
  }

  const scene = useMemo(
    () => (size.w > 40 && size.h > 40 && typeof Path2D !== "undefined" ? buildScene(atlas, elite, draws, size.w, size.h) : null),
    [atlas, elite, draws, size.w, size.h],
  );
  const sceneRef = useRef(scene);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const s = sceneRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas === null || s === null || !ctx) return;
    colors.current ??= readColors();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(s.width * dpr) || canvas.height !== Math.round(s.height * dpr)) {
      canvas.width = Math.round(s.width * dpr);
      canvas.height = Math.round(s.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderScene(ctx, s, progress.current, colors.current);
    const days = progress.current * s.duration;
    if (readoutRef.current) readoutRef.current.textContent = `Day ${fixed(days, 1)} of ${fixed(s.duration, 1)}`;
    if (sliderRef.current) sliderRef.current.value = String(Math.round(progress.current * 1000));
  }, []);

  useEffect(() => {
    sceneRef.current = scene;
    paint();
  }, [scene, paint]);

  useEffect(() => {
    if (!playing || elite === null) return;
    let last: number | null = null;
    let phase = progress.current * FLIGHT_SECONDS;
    let frame = requestAnimationFrame(function tick(now) {
      if (last !== null) {
        phase += Math.min(now - last, 100) / 1000;
        if (phase >= FLIGHT_SECONDS + HOLD_SECONDS) phase -= FLIGHT_SECONDS + HOLD_SECONDS;
        progress.current = Math.min(1, phase / FLIGHT_SECONDS);
        paint();
      }
      last = now;
      frame = requestAnimationFrame(tick);
    });
    return () => { cancelAnimationFrame(frame); };
  }, [playing, elite, paint]);

  const facts = elite;
  const ok = facts?.uncorrected_feasible_draws ?? 0;
  const total = atlas.draws;
  const label =
    facts === null
      ? "Rotating Earth–Moon frame. This method found no free return, so there is no trajectory to show."
      : `Rotating Earth–Moon frame: a free return from low Earth orbit passes ${altitudeLabel(facts.perilune_altitude_km)} above the Moon and is back after ${daysLabel(facts.return_time_days)}. ${total} launch-error variants are drawn around it, ${ok} of which come home without a correction.`;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 px-4 pt-3.5 sm:px-5">
        <p className="font-serif text-lg leading-snug text-canvas-strong sm:text-xl">
          {facts === null ? (
            "No free return found"
          ) : (
            <>
              {altitudeLabel(facts.perilune_altitude_km)} above the Moon, home in {daysLabel(facts.return_time_days)}
            </>
          )}
        </p>
        {facts !== null && (
          <p className="text-sm text-canvas-text">
            <Term
              tip={`Each of ${total} small launch errors (burn size, direction and timing) is tried, with one correction of up to ${fixed(atlas.budget_m_s, 0)} m/s allowed. This is the share that still comes home.`}
            >
              Survives
            </Term>{" "}
            <span className="numeric text-base font-medium text-canvas-strong">{percent(facts.survival)}</span> of launch errors
          </p>
        )}
      </div>

      <div ref={frameRef} className="relative aspect-[5/4] lg:aspect-auto lg:min-h-[19rem] lg:flex-1">
        <canvas ref={canvasRef} role="img" aria-label={label} className="absolute inset-0 size-full" style={{ width: "100%", height: "100%" }} />
        {facts === null && (
          <p className="absolute inset-x-0 top-1/2 mx-auto max-w-[24ch] -translate-y-1/2 text-center text-sm text-canvas-text">{emptyNote}</p>
        )}
      </div>

      <div className="flex items-center gap-3 px-4 sm:px-5">
        <button
          type="button"
          disabled={facts === null}
          onClick={() => { setPlaying((p) => !p); }}
          aria-label={playing ? "Pause" : "Play"}
          className="grid size-9 shrink-0 place-items-center rounded-full border border-canvas-axis text-canvas-strong hover:border-canvas-text disabled:opacity-40"
        >
          <PlayIcon playing={playing} />
        </button>
        <input
          ref={sliderRef}
          type="range"
          min={0}
          max={1000}
          step={1}
          defaultValue={Math.round(progress.current * 1000)}
          disabled={facts === null}
          aria-label="Time since launch"
          onInput={(e) => {
            setPlaying(false);
            progress.current = Number(e.currentTarget.value) / 1000;
            paint();
          }}
          className="time-slider min-w-0 flex-1"
        />
        <span ref={readoutRef} className="numeric w-28 shrink-0 text-right text-sm text-canvas-text" aria-hidden="true" />
      </div>

      <ul className="flex flex-wrap gap-x-5 gap-y-1 px-4 pb-3.5 text-xs text-canvas-text sm:px-5" aria-label="Legend">
        <li className="flex items-center gap-2">
          <Swatch kind="chosen" />
          Chosen free return
        </li>
        <li className="flex items-center gap-2">
          <Swatch kind="ok" />
          <Term tip="The same launch with a small error in burn size, direction and timing, flown with no correction burn.  A few paths that fail here would be rescued by the correction, which the survival figure counts.">
            Variant that comes home
          </Term>
          <span className="numeric">{ok}</span>
        </li>
        <li className="flex items-center gap-2">
          <Swatch kind="fail" />
          Variant that fails
          <span className="numeric">{total - ok}</span>
        </li>
      </ul>
    </div>
  );
}
