import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Term } from "./Tooltip";

describe("Tooltip", () => {
  it("describes its trigger, opens on keyboard focus and closes on Escape", async () => {
    const user = userEvent.setup();
    render(<Term tip="A plain explanation">term</Term>);
    const trigger = screen.getByText("term");
    const tip = screen.getByRole("tooltip", { hidden: true });
    expect(trigger).toHaveAttribute("aria-describedby", tip.id);
    expect(tip).toBeInTheDocument();
    expect(tip).not.toBeVisible();
    await user.tab();
    expect(trigger).toHaveFocus();
    expect(screen.getByRole("tooltip")).toHaveTextContent("A plain explanation");
    await user.keyboard("{Escape}");
    expect(screen.getByRole("tooltip", { hidden: true })).not.toBeVisible();
  });

  it("opens on mouse hover and closes on leave", async () => {
    const user = userEvent.setup();
    render(<Term tip="Hover text">term</Term>);
    await user.hover(screen.getByText("term"));
    expect(screen.getByRole("tooltip")).toBeVisible();
    await user.unhover(screen.getByText("term"));
    expect(screen.getByRole("tooltip", { hidden: true })).not.toBeVisible();
  });

  it("toggles on a touch tap", async () => {
    render(<Term tip="Tap text">term</Term>);
    const trigger = screen.getByText("term");
    const { fireEvent } = await import("@testing-library/react");
    fireEvent.pointerDown(trigger, { pointerType: "touch" });
    fireEvent.click(trigger);
    expect(screen.getByRole("tooltip")).toBeVisible();
    fireEvent.pointerDown(trigger, { pointerType: "touch" });
    fireEvent.click(trigger);
    expect(screen.getByRole("tooltip", { hidden: true })).not.toBeVisible();
  });
});
