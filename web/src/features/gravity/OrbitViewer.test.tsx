import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { publicJson, publishedRunPath, RUN_PATH } from "../../test/fixtures";
import type { LoadedRun } from "../../lib/data/types";
import { parseManifest, parseTrajectory } from "../../lib/data/validate";
import { OrbitViewer } from "./OrbitViewer";
import { trajectoryStyle } from "./trajectoryStyle";

function load(path: string): LoadedRun {
  const manifest = parseManifest(publicJson(`/research/${path}/manifest.json`));
  return {
    manifest,
    trajectory: parseTrajectory(publicJson(`/research/${path}/trajectory.json`), manifest),
    manifestUrl: `/research/${path}/manifest.json`,
    trajectoryUrl: `/research/${path}/trajectory.json`,
  };
}

const c1 = load(publishedRunPath((r) => r.run_id === "gravity-free-return-c1-v2"));

describe("OrbitViewer", () => {
  it("offers system and near-body views for a free return, starting with the whole system", () => {
    render(<OrbitViewer run={c1} style={trajectoryStyle(0)} />);
    expect(screen.getByRole("button", { name: "Earth–Moon system" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Near Earth" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Near Moon" })).toBeInTheDocument();
  });

  it("keeps the close-up and system views for the reference orbit", () => {
    render(<OrbitViewer run={load(RUN_PATH)} style={trajectoryStyle(2)} />);
    expect(screen.getByRole("button", { name: "Orbit close-up" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: "Near Earth" })).toBeNull();
  });

  it("jumps to the recorded sample nearest an event", async () => {
    render(<OrbitViewer run={c1} style={trajectoryStyle(0)} />);
    await userEvent.setup().click(screen.getByRole("button", { name: /Perilune/ }));
    expect(screen.getByRole("button", { name: /Perilune/ })).toHaveAttribute("aria-current", "step");
    // The perilune altitude read off the sample matches the 1,000 km target to 0.1 km.
    expect(screen.getByText("1,000.0")).toBeInTheDocument();
  });

  it("states the entry-interface shell honestly when it is below a pixel at this zoom", () => {
    render(<OrbitViewer run={c1} style={trajectoryStyle(0)} />);
    expect(screen.getByText(/121\.92 km above the surface\. Not drawn here/)).toBeInTheDocument();
  });
});
