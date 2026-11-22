import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { LibraryReportPage } from "./LibraryReportPage";

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

describe("The Last Library report page", () => {
  it("renders report findings, Holm-adjusted verdict table, provenance, and verified citations", async () => {
    render(<LibraryReportPage />);

    await screen.findByRole("heading", { name: /Summary & Headline Verdicts/ });
    expect(screen.getByRole("heading", { name: /The 19-Bit Capacity Threshold/ })).toBeInTheDocument();
    expect(screen.getByText(/Sender confidence calibration at B=40/)).toBeInTheDocument();
    expect(screen.getByText(/sha256:db7805447c31/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /doi:10\.31234\/osf\.io\/q4tz8/ })).toHaveAttribute(
      "href",
      "https://doi.org/10.31234/osf.io/q4tz8",
    );
  });
});
