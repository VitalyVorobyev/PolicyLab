import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { clone, publicJson, RUN_PATH } from "../../test/fixtures";
import type { LoadedRun } from "../../lib/data/types";
import { parseManifest, parseTrajectory } from "../../lib/data/validate";
import { RunDetails } from "./RunDetails";

type JsonObject = Record<string, unknown>;

const manifestJson = publicJson(`/research/${RUN_PATH}/manifest.json`) as JsonObject;
const trajectoryJson = publicJson(`/research/${RUN_PATH}/trajectory.json`);

function loaded(mutate: (m: JsonObject) => void = () => undefined): LoadedRun {
  const raw = clone(manifestJson);
  mutate(raw);
  const manifest = parseManifest(raw);
  return {
    manifest,
    trajectory: parseTrajectory(trajectoryJson, manifest),
    manifestUrl: `/research/${RUN_PATH}/manifest.json`,
    trajectoryUrl: `/research/${RUN_PATH}/trajectory.json`,
  };
}

function checkRow(name: string): HTMLElement {
  const row = screen.getByText(name).closest("tr");
  if (row === null) throw new Error(`no table row for ${name}`);
  return row;
}

describe("RunDetails", () => {
  it("shows every check with its definition, including the independent-integrator and stability-index checks", () => {
    render(<RunDetails run={loaded()} />);
    const table = screen.getByRole("table");
    expect(within(table).getAllByText("Passed")).toHaveLength(12);
    const independent = checkRow("independent_integrator_max_abs_state_difference_samples");
    expect(within(independent).getByText("Difference from an independent integrator")).toBeInTheDocument();
    expect(within(independent).getByText(/s_ref: LSODA rtol = atol = 1e-13/)).toBeInTheDocument();
    const stability = checkRow("jpl_catalog_rel_diff_stability_index");
    expect(within(stability).getByText(/from the computed monodromy matrix/)).toBeInTheDocument();
    expect(within(table).getByText("Agreement with other propagations, including entry conditions")).toBeInTheDocument();
  });

  it("shows the corrector's convergence record outside the checks table", () => {
    render(<RunDetails run={loaded()} />);
    const record = screen.getByText("Convergence record").closest("figure");
    if (record === null) throw new Error("no convergence record");
    expect(within(record).getByText("4")).toBeInTheDocument();
    expect(within(record).getByText(/not a validation check/)).toBeInTheDocument();
    expect(within(screen.getByRole("table")).queryByText(/convergence/i)).toBeNull();
  });

  it("shows the generating environment in the provenance", () => {
    render(<RunDetails run={loaded()} />);
    const provenance = screen.getByRole("region", { name: "Provenance" });
    expect(within(provenance).getByText("3.12.12")).toBeInTheDocument();
    expect(within(provenance).getByText("2.5.4")).toBeInTheDocument();
    expect(within(provenance).getByText("1.18.1")).toBeInTheDocument();
    expect(within(provenance).getByText("macOS-27.0.1-arm64-arm-64bit")).toBeInTheDocument();
  });

  it("shows the length-scale convention and the limitations exactly as the manifest states them", () => {
    render(<RunDetails run={loaded()} />);
    const model = screen.getByRole("region", { name: "Model and constants" });
    expect(within(model).getByText("(jpl_periodic_orbits_catalog)")).toBeInTheDocument();
    expect(within(model).getByText(/JPL three-body periodic-orbit catalog length unit/)).toBeInTheDocument();
    const limitations = screen.getByRole("region", { name: "Limitations" });
    expect(within(limitations).getAllByRole("listitem")).toHaveLength(5);
    expect(within(limitations).getByText(/largest monodromy eigenvalue is 1808\.9/)).toBeInTheDocument();
  });

  it("titles the Jacobi checks as conservation only when the run has no periodicity check", () => {
    const { unmount } = render(<RunDetails run={loaded()} />);
    expect(within(screen.getByRole("table")).getByText("Periodicity and conservation")).toBeInTheDocument();
    unmount();
    const run = loaded((m) => {
      const validation = m.validation as JsonObject;
      validation.checks = (validation.checks as JsonObject[]).filter((c) => !String(c.name).startsWith("periodicity_"));
    });
    render(<RunDetails run={run} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("Conservation")).toBeInTheDocument();
    expect(within(table).queryByText("Periodicity and conservation")).toBeNull();
  });

  it("renders a non-finite observed value as non-finite and failed", () => {
    const run = loaded((m) => {
      m.status = "computed_unvalidated";
      const check = ((m.validation as JsonObject).checks as JsonObject[])[0];
      if (check === undefined) throw new Error("fixture has no checks");
      check.observed = null;
      check.passed = false;
    });
    render(<RunDetails run={run} />);
    const row = checkRow("periodicity_max_abs_state_error_one_period");
    expect(within(row).getByText("non-finite")).toBeInTheDocument();
    expect(within(row).getByText("Failed")).toBeInTheDocument();
  });
});
