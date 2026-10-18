import type { KeyboardEvent, ReactNode } from "react";
import { PauseIcon, PlayIcon, RestartIcon, StepBackIcon, StepForwardIcon } from "../../components/icons";
import { usePlaybackState, type PlaybackStore } from "./playback";

export const SPEEDS_DAYS_PER_SECOND = [0.5, 1, 2, 4] as const;
export type Speed = (typeof SPEEDS_DAYS_PER_SECOND)[number];

/**
 * Keyboard shortcuts for the viewer: Space toggles playback, arrows step one sample
 * (ten with Shift), Home restarts, End jumps to the last sample. Returns true if handled.
 * Native controls (buttons, slider, select) keep their own keyboard behaviour.
 */
export function handlePlaybackKey(store: PlaybackStore, event: KeyboardEvent<HTMLElement>): boolean {
  const target = event.target as HTMLElement;
  if (target.closest("button, input, select, a") !== null) return false;
  if (event.metaKey || event.ctrlKey || event.altKey) return false;
  const stride = event.shiftKey ? 10 : 1;
  switch (event.key) {
    case " ":
    case "k":
      store.toggle();
      break;
    case "ArrowLeft":
      store.step(-stride);
      break;
    case "ArrowRight":
      store.step(stride);
      break;
    case "Home":
      store.seek(0);
      break;
    case "End":
      store.seek(store.length - 1);
      break;
    default:
      return false;
  }
  event.preventDefault();
  return true;
}

function ControlButton({ label, onClick, children, pressed }: { label: string; onClick: () => void; children: ReactNode; pressed?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      onClick={onClick}
      className="grid size-10 shrink-0 place-items-center rounded-sm text-canvas-strong hover:bg-canvas-grid active:bg-canvas-axis"
    >
      {children}
    </button>
  );
}

interface Props {
  store: PlaybackStore;
  /** Formats a sample index for the slider's spoken value, e.g. "day 5.29 of 12.69". */
  describeSample: (index: number) => string;
  speed: Speed;
  onSpeedChange: (speed: Speed) => void;
}

export function PlaybackControls({ store, describeSample, speed, onSpeedChange }: Props) {
  const { index, playing } = usePlaybackState(store);
  const last = store.length - 1;
  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-1 px-2 py-1.5 sm:flex-nowrap sm:gap-x-2">
      <ControlButton label={playing ? "Pause" : "Play"} onClick={() => { store.toggle(); }}>
        {playing ? <PauseIcon /> : <PlayIcon />}
      </ControlButton>
      <ControlButton label="Restart from the first sample" onClick={() => { store.restart(); }}>
        <RestartIcon />
      </ControlButton>
      <ControlButton label="Step back one sample" onClick={() => { store.step(-1); }}>
        <StepBackIcon />
      </ControlButton>
      <ControlButton label="Step forward one sample" onClick={() => { store.step(1); }}>
        <StepForwardIcon />
      </ControlButton>
      <label className="order-last flex w-full min-w-0 items-center gap-3 px-2 sm:order-none sm:w-auto sm:flex-1">
        <span className="sr-only">Time</span>
        <input
          type="range"
          className="time-slider"
          min={0}
          max={last}
          step={1}
          value={index}
          aria-valuetext={describeSample(index)}
          onChange={(event) => { store.seek(Number(event.currentTarget.value)); }}
        />
      </label>
      <label className="ml-auto flex shrink-0 items-center gap-2 pr-1 text-sm text-canvas-text sm:ml-0">
        Speed
        <select
          value={speed}
          onChange={(event) => { onSpeedChange(Number(event.currentTarget.value) as Speed); }}
          className="rounded-sm border border-canvas-axis bg-canvas px-1.5 py-1 text-canvas-strong"
        >
          {SPEEDS_DAYS_PER_SECOND.map((s) => (
            <option key={s} value={s}>
              {s === 1 ? "1 day/s" : `${s} days/s`}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
