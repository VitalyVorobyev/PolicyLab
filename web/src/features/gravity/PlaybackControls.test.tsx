import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { handlePlaybackKey, PlaybackControls, type Speed } from "./PlaybackControls";
import { PlaybackStore, sampleAt } from "./playback";

function Harness({ store }: { store: PlaybackStore }) {
  const [speed, setSpeed] = useState<Speed>(1);
  return (
    <div
      role="group"
      aria-label="viewer"
      tabIndex={0}
      onKeyDown={(event) => {
        handlePlaybackKey(store, event);
      }}
    >
      <PlaybackControls store={store} speed={speed} onSpeedChange={setSpeed} describeSample={(i) => `sample ${i + 1}`} />
    </div>
  );
}

describe("PlaybackControls", () => {
  it("toggles play and pause and reflects the state in the button label", async () => {
    const user = userEvent.setup();
    const store = new PlaybackStore(100, false);
    render(<Harness store={store} />);

    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(store.getState().playing).toBe(true);
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(store.getState().playing).toBe(false);
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
  });

  it("steps one sample, clamps at the ends and pauses playback", async () => {
    const user = userEvent.setup();
    const store = new PlaybackStore(3, true);
    render(<Harness store={store} />);

    await user.click(screen.getByRole("button", { name: "Step forward one sample" }));
    expect(store.getState()).toEqual({ index: 1, playing: false });
    await user.click(screen.getByRole("button", { name: "Step forward one sample" }));
    await user.click(screen.getByRole("button", { name: "Step forward one sample" }));
    expect(store.getState().index).toBe(2);
    await user.click(screen.getByRole("button", { name: "Step back one sample" }));
    expect(store.getState().index).toBe(1);
  });

  it("restarts from the first sample without changing play state", async () => {
    const user = userEvent.setup();
    const store = new PlaybackStore(50, false);
    store.seek(30);
    store.play();
    render(<Harness store={store} />);
    await user.click(screen.getByRole("button", { name: "Restart from the first sample" }));
    expect(store.getState()).toEqual({ index: 0, playing: true });
  });

  it("scrubs with the time slider and exposes a readable value", () => {
    const store = new PlaybackStore(1001, true);
    render(<Harness store={store} />);
    const slider = screen.getByRole("slider", { name: "Time" });
    expect(slider).toHaveAttribute("max", "1000");
    fireEvent.change(slider, { target: { value: "500" } });
    expect(store.getState()).toEqual({ index: 500, playing: false });
    expect(slider).toHaveAttribute("aria-valuetext", "sample 501");
  });

  it("updates when the store changes outside the component (as the animation clock does)", () => {
    const store = new PlaybackStore(10, false);
    render(<Harness store={store} />);
    act(() => {
      store.seek(7, true);
    });
    expect(screen.getByRole("slider", { name: "Time" })).toHaveValue("7");
  });

  it("supports keyboard control on the viewer: Space, arrows, Shift+arrows, Home, End", async () => {
    const user = userEvent.setup();
    const store = new PlaybackStore(100, false);
    render(<Harness store={store} />);
    screen.getByRole("group", { name: "viewer" }).focus();

    await user.keyboard(" ");
    expect(store.getState().playing).toBe(true);
    await user.keyboard(" ");
    expect(store.getState().playing).toBe(false);
    await user.keyboard("{ArrowRight}{ArrowRight}");
    expect(store.getState().index).toBe(2);
    await user.keyboard("{Shift>}{ArrowRight}{/Shift}");
    expect(store.getState().index).toBe(12);
    await user.keyboard("{ArrowLeft}");
    expect(store.getState().index).toBe(11);
    await user.keyboard("{End}");
    expect(store.getState().index).toBe(99);
    await user.keyboard("{Home}");
    expect(store.getState().index).toBe(0);
  });

  it("does not double-handle Space on a focused button", async () => {
    const user = userEvent.setup();
    const store = new PlaybackStore(100, false);
    render(<Harness store={store} />);
    screen.getByRole("button", { name: "Play" }).focus();
    await user.keyboard(" ");
    expect(store.getState().playing).toBe(true);
  });

  it("changes speed through the select", async () => {
    const user = userEvent.setup();
    render(<Harness store={new PlaybackStore(10)} />);
    const select = screen.getByRole("combobox", { name: "Speed" });
    await user.selectOptions(select, "4");
    expect(select).toHaveValue("4");
  });
});

describe("PlaybackStore and sampleAt", () => {
  it("never plays a single-sample trajectory", () => {
    const store = new PlaybackStore(1, true);
    expect(store.getState().playing).toBe(false);
    store.play();
    expect(store.getState().playing).toBe(false);
  });

  it("finds the last recorded sample at or before a time, without interpolation", () => {
    const times = new Float64Array([0, 0.5, 1.5, 3]);
    expect(sampleAt(times, -1)).toBe(0);
    expect(sampleAt(times, 0.49)).toBe(0);
    expect(sampleAt(times, 0.5)).toBe(1);
    expect(sampleAt(times, 2.9)).toBe(2);
    expect(sampleAt(times, 10)).toBe(3);
  });
});
