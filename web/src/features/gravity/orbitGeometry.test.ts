import { describe, expect, it } from "vitest";
import { countUpTo, fitView, insetHalfWidthKm, pointAt, toX, toY } from "./orbitGeometry";

const line = { t_days: Float64Array.of(0, 1, 3), x_km: Float64Array.of(0, 10, 30), y_km: Float64Array.of(0, 0, 20) };

describe("orbitGeometry", () => {
  it("interpolates linearly between vertices and holds at the ends", () => {
    expect(pointAt(line, 0.5)).toEqual({ x: 5, y: 0 });
    expect(pointAt(line, 2)).toEqual({ x: 20, y: 10 });
    expect(pointAt(line, -1)).toEqual({ x: 0, y: 0 });
    expect(pointAt(line, 9)).toEqual({ x: 30, y: 20 });
  });
  it("counts the vertices already flown", () => {
    expect([0, 0.5, 1, 2.9, 3].map((t) => countUpTo(line, t))).toEqual([1, 1, 2, 2, 3]);
  });
  it("fits with equal scale and a centred, y-up mapping", () => {
    const v = fitView({ xMin: 0, xMax: 100, yMin: 0, yMax: 50 }, { x: 0, y: 0, w: 200, h: 200 });
    expect(v.scale).toBe(2);
    expect(toX(v, 50)).toBe(100);
    expect(toY(v, 50)).toBeLessThan(toY(v, 0));
  });
  it("keeps the Moon inset between a close-up and a real zoom", () => {
    expect(insetHalfWidthKm(100, 1737.4)).toBe(8000);
    expect(insetHalfWidthKm(29367, 1737.4)).toBe(80000);
  });
});
