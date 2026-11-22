import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { GravityReportPage } from "./GravityReportPage";

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

describe("Gravity Atlas report page", () => {
  it("renders the M4 7-arm table, Holm contrasts, validated M1 trajectories, and verified citations", async () => {
    render(<GravityReportPage />);
    await screen.findByRole("heading", { name: /Seven Methods at Matched Budget/ });
    expect(screen.getByText(/H1 is supported against all five baselines through coverage/)).toBeInTheDocument();
    expect(screen.getByText("16-seed means at matched 16,384-propagation budget")).toBeInTheDocument();
    expect(screen.getAllByText(/MAP-Elites \(T\) vs\. Differential evolution \(DE\)/)).toHaveLength(2);
    await waitFor(() => {
      expect(screen.getByText("gravity-l1-lyapunov-reference-v2")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Validated").length).toBeGreaterThanOrEqual(3);
    expect(screen.getByRole("link", { name: /arXiv:2302\.00463/ })).toHaveAttribute("href", "https://arxiv.org/abs/2302.00463");
    expect(screen.getByRole("link", { name: /arXiv:2412\.20023/ })).toHaveAttribute("href", "https://arxiv.org/abs/2412.20023");
  });
});
