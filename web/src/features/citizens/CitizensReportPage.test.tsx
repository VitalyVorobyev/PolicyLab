import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { CitizensReportPage } from "./CitizensReportPage";

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

describe("Model Citizens report page", () => {
  it("renders pre-registered verdicts, live cross-play & exam sections, provenance, and verified literature", async () => {
    render(<CitizensReportPage />);
    await screen.findByText(/Refuted: at least one AI governor loses to the simple rule on another model/);
    expect(screen.getByText(/Supported: a governor designed on a model that only passed the next-move exam loses to the simple rule on B1/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Four Models of People, Two Exams/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Cross-Play, Transfer & Ablations/ })).toBeInTheDocument();
    expect(screen.getByText("citizens-web-demo-dev4")).toBeInTheDocument();
    const doi = screen.getByRole("link", { name: /10\.1038\/s41467-025-58043-7/ });
    expect(doi).toHaveAttribute("href", "https://doi.org/10.1038/s41467-025-58043-7");
  });
});
