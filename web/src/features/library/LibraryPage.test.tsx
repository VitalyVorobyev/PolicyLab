import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { LibraryPage } from "./LibraryPage";

beforeEach(() => {
  vi.stubGlobal("fetch", (url: string) => {
    try {
      return Promise.resolve(new Response(publicBytes(url)));
    } catch {
      return Promise.resolve(new Response("missing", { status: 404 }));
    }
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("The Last Library interactive page", () => {
  it("loads verified confirmatory data, synchronizes budget controls, and inspects sample records", async () => {
    const user = userEvent.setup();
    render(<LibraryPage />);

    await screen.findByText(/1\. The bit-capacity cliff across 0 to 60 bits/);
    expect(screen.getByText(/Comparing generational trajectories at bit budget/)).toHaveTextContent("B = 20 bits");
    expect(screen.getByText("Stationary efficiency: rule beats pedagogical examples at gen 9")).toBeInTheDocument();
    expect(screen.getByText(/sha256:db7805447c31/)).toBeInTheDocument();

    const budgetGroup = screen.getByRole("group", { name: "Select bit budget B" });
    await user.click(within(budgetGroup).getByRole("button", { name: "40b" }));
    expect(screen.getByText(/Comparing generational trajectories at bit budget/)).toHaveTextContent("B = 40 bits");

    const togglePlacebos = screen.getByLabelText(/Show placebo controls/);
    await user.click(togglePlacebos);
    expect(screen.getByText(/Placebo rand/)).toBeInTheDocument();

    const armGroup = screen.getByRole("group", { name: "Compared arm" });
    await user.click(within(armGroup).getByRole("button", { name: "Rule+hedge" }));
    expect(screen.getByText(/Selected state: Hedged rule/)).toBeInTheDocument();
  });
});
