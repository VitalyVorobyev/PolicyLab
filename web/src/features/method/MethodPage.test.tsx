import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { MethodPage } from "./MethodPage";

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

describe("Method & Documentation page", () => {
  it("lists catalog runs, inspects a run manifest on click, and links The Last Library study", async () => {
    const user = userEvent.setup();
    render(<MethodPage />);
    await screen.findByText("citizens-web-demo-dev4");
    expect(screen.getByText("gravity-l1-lyapunov-reference-v2")).toBeInTheDocument();
    expect(screen.getAllByText("library-main-v1").length).toBeGreaterThanOrEqual(1);

    const [firstButton] = screen.getAllByRole("button", { name: "Inspect manifest & checks" });
    if (firstButton === undefined) throw new Error("expected an inspect button");
    await user.click(firstButton);

    await screen.findByText(/trace_totals_reproduce_crossplay/);
    expect(screen.getByText(/sha256:ebfb572cd19b/)).toBeInTheDocument();
    expect(screen.getByText(/Published & Validated — confirmatory verdicts across 0–60 bits/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /doi:10\.31234\/osf\.io\/q4tz8/ })).toHaveAttribute("href", "https://doi.org/10.31234/osf.io/q4tz8");
  });
});
