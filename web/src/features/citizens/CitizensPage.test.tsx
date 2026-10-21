import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publicBytes } from "../../test/fixtures";
import { CitizensPage } from "./CitizensPage";

beforeEach(() => {
  vi.stubGlobal("fetch", (url: string) => {
    try {
      return Promise.resolve(new Response(publicBytes(url)));
    } catch {
      return Promise.resolve(new Response("missing", { status: 404 }));
    }
  });
  // jsdom does not lay out or scroll.
  Element.prototype.scrollIntoView = (): void => undefined;
  // No animation in tests: reduced motion shows the finished game.
  vi.stubGlobal("matchMedia", (query: string) => ({ matches: query.includes("reduce"), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

const demo = (): HTMLElement => screen.getByRole("region", { name: "Interactive demonstration" });

describe("Model Citizens page", () => {
  it("opens on the B3-trained AI governor with neural-clone citizens, compared with the simple rule", async () => {
    render(<CitizensPage />);
    await screen.findByRole("region", { name: "Interactive demonstration" });
    expect(screen.getByRole("radio", { name: "AI · B3" })).toBeChecked();
    expect(screen.getByRole("radio", { name: /B1 · Neural clone/ })).toBeChecked();
    // The cross-play mean of this cell, copied from the run (938 coins), and the simple rule's (1,284).
    expect(within(demo()).getByText(/Simulated game: 1,763 coins in all/)).toBeInTheDocument();
    expect(within(demo()).getByText("938")).toBeInTheDocument();
    expect(within(demo()).getByText("Round 40")).toBeInTheDocument();
  });

  it("switches to the rule governors with real people, and offers DeepMind's governor only there", async () => {
    const user = userEvent.setup();
    render(<CitizensPage />);
    await screen.findByRole("region", { name: "Interactive demonstration" });
    expect(screen.queryByRole("radio", { name: "DeepMind AI" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "Real people" }));
    expect(screen.getByRole("radio", { name: "Simple rule" })).toBeChecked();
    expect(screen.queryByRole("radio", { name: "AI · B3" })).not.toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "DeepMind AI" })).toBeInTheDocument();
    expect(within(demo()).getByText(/Recorded game \d+ \(Exp \d\), held-out data/)).toBeInTheDocument();
    // Two recorded games per mechanism.
    expect(within(demo()).getAllByRole("button", { name: /^[0-9]$/ })).toHaveLength(2);
  });

  it("loads a cross-play cell into the demonstration when it is clicked", async () => {
    const user = userEvent.setup();
    render(<CitizensPage />);
    await screen.findByRole("region", { name: "Interactive demonstration" });
    await user.click(screen.getByRole("button", { name: /^Equal split with B3: 385 coins/ }));
    await waitFor(() => {
      expect(screen.getByRole("radio", { name: "Equal" })).toBeChecked();
    });
    expect(screen.getByRole("radio", { name: /B3 · Learners/ })).toBeChecked();
  });

  it("states the descriptive findings with numbers read from the run", async () => {
    render(<CitizensPage />);
    await screen.findByRole("region", { name: "Interactive demonstration" });
    expect(screen.getByText("Each AI governor is the best at home.")).toBeInTheDocument();
    expect(screen.getByText(/produced a governor that loses 24–28 % of the surplus to the simple rule on B1/)).toBeInTheDocument();
    expect(screen.getByText(/82 % of what that world's own AI governor earns, against 45 % for the simple rule/)).toBeInTheDocument();
    expect(screen.getByText(/the pre-registered verdicts are being computed/)).toBeInTheDocument();
    expect(screen.getByText(/All four models pass the next-move exam, but only B1 passes the whole-game exam/)).toBeInTheDocument();
  });

  it("shows how often each model orders the published human comparisons right, read from the E1 run", async () => {
    render(<CitizensPage />);
    await screen.findByRole("region", { name: "Interactive demonstration" });
    const panel = screen.getByRole("heading", { name: "Is the judge credible?" }).parentElement as HTMLElement;
    const row = (name: RegExp): HTMLElement => within(panel).getByRole("row", { name });
    expect(within(row(/Neural clone/)).getByText("8 / 8")).toBeInTheDocument();
    expect(within(row(/Conditional cooperators/)).getByText("7 / 8")).toBeInTheDocument();
    expect(within(row(/random returns/)).getByText("5 / 8")).toBeInTheDocument();
    expect(within(row(/LLM players/)).getByText("2 / 2")).toBeInTheDocument();
    expect(within(row(/LLM, uncalibrated/)).getByText("1 / 2")).toBeInTheDocument();
    expect(within(panel).getByText(/but so do the learner models \(B3 and B23\), so this check says B1 is a sound judge/)).toBeInTheDocument();
  });

  it("warns that the one-weight curves do not explain the AI governors", async () => {
    render(<CitizensPage />);
    await screen.findByRole("region", { name: "Interactive demonstration" });
    expect(screen.getByText("But these curves explain the AI governors poorly")).toBeInTheDocument();
  });
});
