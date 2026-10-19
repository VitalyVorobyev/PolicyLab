import { describe, expect, it } from "vitest";
import { axisKm, neighbor, nearestElite, survivalColor, type Cell } from "./atlasModel";
import type { AtlasArm, AtlasElite } from "../../lib/data/types";

const cells: Cell[] = [[1, 1], [2, 1], [2, 2], [5, 5]];

describe("neighbor", () => {
  it("moves along the row first, then to the nearest cell beyond", () => {
    expect(neighbor(cells, [1, 1], "right")).toEqual([2, 1]);
    expect(neighbor(cells, [2, 1], "up")).toEqual([2, 2]);
    expect(neighbor(cells, [2, 2], "right")).toEqual([5, 5]);
  });
  it("returns undefined at the edge", () => {
    expect(neighbor(cells, [1, 1], "left")).toBeUndefined();
    expect(neighbor(cells, [5, 5], "up")).toBeUndefined();
  });
});

describe("nearestElite", () => {
  const arm = { elites: [{ cell: [1, 1] }, { cell: [6, 6] }] } as unknown as AtlasArm;
  it("keeps the viewer near the preferred square", () => {
    expect((nearestElite(arm, [5, 5]) as AtlasElite).cell).toEqual([6, 6]);
    expect((nearestElite(arm, [0, 2]) as AtlasElite).cell).toEqual([1, 1]);
    expect(nearestElite({ elites: [] } as unknown as AtlasArm, [1, 1])).toBeUndefined();
  });
});

describe("formatting", () => {
  it("labels axis edges with two significant figures", () => {
    expect([100, 312.9, 979.1, 9587.3, 30000.00000001].map(axisKm)).toEqual(["100", "310", "980", "9,600", "30,000"]);
  });
  it("runs the survival scale from the first to the last stop", () => {
    expect(survivalColor(0)).toBe("rgb(91 58 154)");
    expect(survivalColor(1)).toBe("rgb(242 230 90)");
  });
});
