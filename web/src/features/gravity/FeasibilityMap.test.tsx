import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { publicJson, publishedRunPath } from "../../test/fixtures";
import type { LoadedGridRun } from "../../lib/data/types";
import { parseManifest, parseOutcomeGrid } from "../../lib/data/validate";
import { FeasibilityMap, type MapCandidate } from "./FeasibilityMap";
import { FreeReturnMetrics } from "./FreeReturnMetrics";
import { rasterize } from "./grid";
import { OUTCOME_ORDER } from "./outcomes";

const GRID = publishedRunPath((r) => r.kind === "outcome_grid");
const manifest = parseManifest(publicJson(`/research/${GRID}/manifest.json`));
const run: LoadedGridRun = {
  manifest,
  grid: parseOutcomeGrid(publicJson(`/research/${GRID}/outcome_grid.json`), manifest),
  manifestUrl: `/research/${GRID}/manifest.json`,
  artifactUrl: `/research/${GRID}/outcome_grid.json`,
};
const CANDIDATES: MapCandidate[] = [
  { runId: "gravity-free-return-c1-v2", tag: "c1", label: "Free return c1", phaseDeg: 229.142, deltaVKmS: 3.15677 },
  { runId: "gravity-free-return-c2-v2", tag: "c2", label: "Free return c2", phaseDeg: 225.816, deltaVKmS: 3.17801 },
];

describe("FeasibilityMap", () => {
  it("lists every outcome with its count, splitting feasible cells by lunar passage sense", () => {
    render(<FeasibilityMap run={run} candidates={CANDIDATES} selectedRunId="gravity-free-return-c1-v2" onSelect={() => undefined} />);
    const legend = screen.getByRole("table");
    const row = (label: RegExp): HTMLElement => {
      const cell = within(legend).getByText(label).closest("tr");
      if (cell === null) throw new Error(`no row ${String(label)}`);
      return cell;
    };
    expect(within(row(/sense −1/)).getByText("156")).toBeInTheDocument();
    expect(within(row(/sense \+1/)).getByText("5")).toBeInTheDocument();
    expect(within(row(/^Moon collision$/)).getByText("283")).toBeInTheDocument();
    expect(within(row(/No return within 10 days/)).getByText("64,681")).toBeInTheDocument();
    expect(within(row(/^Numerical failure$/)).getByText("0")).toBeInTheDocument();
    expect(within(legend).getByText("90,720")).toBeInTheDocument();
  });

  it("selects a candidate from its marker, and marks the selected one as pressed", async () => {
    const onSelect = vi.fn();
    render(<FeasibilityMap run={run} candidates={CANDIDATES} selectedRunId="gravity-free-return-c1-v2" onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: /Show Free return c1/ })).toHaveAttribute("aria-pressed", "true");
    await userEvent.setup().click(screen.getByRole("button", { name: /Show Free return c2/ }));
    expect(onSelect).toHaveBeenCalledWith("gravity-free-return-c2-v2");
  });

  it("explains that cells show grid-point outcomes and candidates lie between grid points", () => {
    render(<FeasibilityMap run={run} candidates={CANDIDATES} selectedRunId="gravity-free-return-c1-v2" onSelect={() => undefined} />);
    expect(screen.getByText(/Each rectangle shows the outcome at its grid point only\. The candidates were refined between grid points/)).toBeInTheDocument();
    const marker = screen.getByRole("button", { name: /Show Free return c2/ });
    expect(marker).toHaveAccessibleName(/Refined between grid points; the cell beneath shows the outcome at its grid point only\./);
    expect(marker).toHaveAttribute("title", expect.stringMatching(/Refined between grid points/) as string);
  });

  it("describes the focused cell and moves with the arrow keys", () => {
    render(<FeasibilityMap run={run} candidates={CANDIDATES} selectedRunId="gravity-free-return-c1-v2" onSelect={() => undefined} />);
    const map = screen.getByRole("group", { name: /Zoomed outcome map/ });
    fireEvent.focus(map);
    const readout = document.getElementById("map-cell-readout");
    // Focus starts on the first candidate's cell: theta 229.0 deg, delta-v 3.156 km/s.
    expect(readout?.textContent).toMatch(/θ = 229\.0°, Δv = 3\.156 km\/s/);
    fireEvent.keyDown(map, { key: "ArrowRight" });
    expect(readout?.textContent).toMatch(/θ = 229\.5°/);
    fireEvent.keyDown(map, { key: "ArrowUp", shiftKey: true });
    expect(readout?.textContent).toMatch(/Δv = 3\.166 km\/s/);
  });
});

describe("rasterize", () => {
  it("never hides a lone feasible cell when pixels cover several cells", () => {
    const { grid } = run;
    const pixels = rasterize(grid, { i0: 0, i1: grid.phaseAxis.count - 1, j0: 0, j1: grid.deltaVAxis.count - 1 }, 90, 20, OUTCOME_ORDER);
    const feasible = grid.outcomes.indexOf("feasible");
    const ni = grid.phaseAxis.count;
    const nj = grid.deltaVAxis.count;
    let checked = 0;
    for (let i = 0; i < ni; i++) {
      for (let j = 0; j < nj; j++) {
        if (grid.codes[i * nj + j] !== feasible) continue;
        const px = Math.floor((i * 90) / ni);
        const py = Math.floor(((nj - 1 - j) * 20) / nj);
        expect(pixels[py * 90 + px]).toBe(feasible);
        checked++;
      }
    }
    expect(checked).toBe(161);
    const cellsPerPixel = (ni / 90) * (nj / 20);
    expect(cellsPerPixel).toBeGreaterThan(1);
  });
});

describe("FreeReturnMetrics", () => {
  const C1 = publishedRunPath((r) => r.run_id === "gravity-free-return-c1-v2");
  const c1 = parseManifest(publicJson(`/research/${C1}/manifest.json`));

  it("reads every value from the manifest, with margins in readable units and no invented comparison", () => {
    render(<FreeReturnMetrics manifest={c1} />);
    expect(screen.getByText("3.15677 km/s")).toBeInTheDocument();
    expect(screen.getByText("+1, posigrade")).toBeInTheDocument();
    expect(screen.getByText("9.25 mm/s")).toBeInTheDocument();
    expect(screen.getByText("11.3 m/s")).toBeInTheDocument();
    expect(screen.getAllByText("Beyond it: no return within the time window")).toHaveLength(2);
    expect(screen.queryByText(/Plausibility/)).toBeNull();
  });

  it("omits a quantity the manifest does not record", () => {
    const metrics = Object.fromEntries(Object.entries(c1.metrics).filter(([key]) => key !== "entry_sense"));
    render(<FreeReturnMetrics manifest={{ ...c1, metrics }} />);
    expect(screen.queryByText("Entry sense")).toBeNull();
    expect(screen.getByText("Entry speed")).toBeInTheDocument();
  });
});
