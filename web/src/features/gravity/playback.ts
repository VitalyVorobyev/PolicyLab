import { useEffect, useRef, useState, useSyncExternalStore } from "react";

/**
 * Playback state for one trajectory, kept outside React so that per-frame updates re-render
 * only the components that subscribe (marker, slider, readout), never the page.
 *
 * Playback shows recorded samples only: the current index always points at an exported row,
 * and no positions are interpolated between samples.
 */
export interface PlaybackState {
  index: number;
  playing: boolean;
}

export class PlaybackStore {
  private state: PlaybackState;
  private readonly listeners = new Set<() => void>();
  readonly length: number;

  constructor(length: number, playing = false) {
    if (!(Number.isInteger(length) && length > 0)) throw new RangeError("length must be a positive integer");
    this.length = length;
    this.state = { index: 0, playing: playing && length > 1 };
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getState = (): PlaybackState => this.state;

  private set(next: Partial<PlaybackState>): void {
    const merged = { ...this.state, ...next };
    if (merged.index === this.state.index && merged.playing === this.state.playing) return;
    this.state = merged;
    for (const listener of this.listeners) listener();
  }

  /** Move to a sample index (clamped). `fromPlayback` keeps playing; user seeks pause. */
  seek(index: number, fromPlayback = false): void {
    const clamped = Math.min(this.length - 1, Math.max(0, Math.round(index)));
    this.set(fromPlayback ? { index: clamped } : { index: clamped, playing: false });
  }

  step(delta: number): void {
    this.seek(this.state.index + delta);
  }

  restart(): void {
    this.set({ index: 0 });
  }

  play(): void {
    this.set({ playing: this.length > 1 });
  }

  pause(): void {
    this.set({ playing: false });
  }

  toggle(): void {
    if (this.state.playing) this.pause();
    else this.play();
  }
}

export function usePlaybackState(store: PlaybackStore): PlaybackState {
  return useSyncExternalStore(store.subscribe, store.getState, store.getState);
}

/** Index of the last sample whose time is <= t (times strictly increase). */
export function sampleAt(times: Float64Array, t: number): number {
  let lo = 0;
  let hi = times.length - 1;
  if (t <= (times[0] ?? 0)) return 0;
  if (t >= (times[hi] ?? 0)) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if ((times[mid] ?? 0) <= t) lo = mid;
    else hi = mid;
  }
  return lo;
}

/**
 * Drive the store with requestAnimationFrame while it is playing. Simulation time advances at
 * `rate` normalized time units per wall-clock second and wraps to the start at the end.
 */
export function usePlaybackClock(store: PlaybackStore, times: Float64Array, rate: number): void {
  const { playing } = usePlaybackState(store);
  const rateRef = useRef(rate);
  useEffect(() => {
    rateRef.current = rate;
  }, [rate]);

  useEffect(() => {
    if (!playing) return;
    const t0 = times[0] ?? 0;
    const span = (times[times.length - 1] ?? 0) - t0;
    let simTime = times[store.getState().index] ?? t0;
    let last: number | null = null;
    let frame = requestAnimationFrame(function tick(now) {
      if (last !== null) {
        // Cap the step so a background tab does not jump far ahead on return.
        const dt = Math.min(now - last, 100) / 1000;
        const current = store.getState().index;
        // A seek while playing moves the clock to the chosen sample.
        if (sampleAt(times, simTime) !== current) simTime = times[current] ?? t0;
        simTime += dt * rateRef.current;
        if (simTime > t0 + span) simTime = t0 + ((simTime - t0) % span);
        store.seek(sampleAt(times, simTime), true);
      }
      last = now;
      frame = requestAnimationFrame(tick);
    });
    return () => { cancelAnimationFrame(frame); };
  }, [playing, store, times]);
}

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

/** Whether the visitor asked for reduced motion (false where matchMedia is unavailable). */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(REDUCED_MOTION).matches;
}

/** One store per trajectory; autoplay only when reduced motion is not requested. */
export function usePlaybackStore(length: number): PlaybackStore {
  const [store] = useState(() => new PlaybackStore(length, !prefersReducedMotion()));
  return store;
}
