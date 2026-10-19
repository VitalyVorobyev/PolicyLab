import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { GravityPage } from "./GravityPage";

let requested: string[] = [];

beforeEach(() => {
  requested = [];
  vi.stubGlobal("fetch", (url: string) => {
    requested.push(url);
    try {
      return Promise.resolve(new Response(publicBytes(url)));
    } catch {
      return Promise.resolve(new Response("missing", { status: 404 }));
    }
  });
});
afterEach(() => { vi.unstubAllGlobals(); });

const fans = (): string[] => requested.filter((u) => u.includes("/fan-"));
const cellLabel = (text: RegExp): HTMLElement => screen.getByRole("button", { name: text });

describe("Gravity demonstration", () => {
  it("opens on MAP-Elites with the 187 km, 5.7 day trajectory selected, and loads only that arm's fan", async () => {
    render(<GravityPage />);
    const selected = await screen.findByRole("button", { pressed: true });
    expect(selected).toHaveAccessibleName(/Closest approach 187 km, trip 5\.7 days, survives 66\s%/);
    expect(screen.getByRole("radio", { name: /^MAP-Elites\s*expected/ })).toBeChecked();
    await waitFor(() => { expect(fans()).toEqual(["/research/gravity-atlas/m4-demo-atlas-v1/fan-t.json"]); });
  });

  it("refills the atlas when the method changes, and fetches that arm's fan lazily", async () => {
    const user = userEvent.setup();
    render(<GravityPage />);
    await screen.findByRole("button", { pressed: true });
    expect(screen.getAllByRole("button", { name: /survives/ })).toHaveLength(32);
    await user.click(screen.getByRole("radio", { name: /^Dense sweep/ }));
    await waitFor(() => { expect(screen.getAllByRole("button", { name: /survives/ })).toHaveLength(12); });
    expect(fans()).toContain("/research/gravity-atlas/m4-demo-atlas-v1/fan-d.json");
    expect(fans()).not.toContain("/research/gravity-atlas/m4-demo-atlas-v1/fan-de.json");
    // The 16-run averages come from the file's summaries.
    expect(screen.getByText("Expected survivors").closest("div")).toHaveTextContent("12.2");
    expect(screen.getByText("Survival per trajectory").closest("div")).toHaveTextContent(/88\s%/);
  });

  it("shows an empty atlas when a method found nothing in the example run", async () => {
    const user = userEvent.setup();
    render(<GravityPage />);
    await screen.findByRole("button", { pressed: true });
    await user.click(screen.getByRole("radio", { name: /^Basin hopping/ }));
    await waitFor(() => { expect(screen.queryAllByRole("button", { name: /survives/ })).toHaveLength(0); });
    expect(screen.getByText("No free return found")).toBeInTheDocument();
  });

  it("selects a square's trajectory on click and moves between filled squares with arrow keys", async () => {
    const user = userEvent.setup();
    render(<GravityPage />);
    await screen.findByRole("button", { pressed: true });
    await user.click(cellLabel(/Closest approach 5,?530 km/));
    expect(screen.getByRole("button", { pressed: true })).toHaveAccessibleName(/survives 100\s%/);
    expect(screen.getByText(/5,530 km above the Moon/)).toBeInTheDocument();
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("button", { pressed: true })).not.toHaveAccessibleName(/5,530 km/);
  });
});
