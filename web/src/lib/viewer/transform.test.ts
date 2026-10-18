// @vitest-environment node
import { describe, expect, it } from "vitest";
import { publicJson, RUN_PATH } from "../../test/fixtures";
import { parseManifest, parseTrajectory } from "../data/validate";
import { lagrangeL1 } from "./lagrange";
import { bodyGlyph, cameraBounds, MIN_BODY_RADIUS_PX } from "./scene";
import { boundsOf, fitTransform, niceStep, ticks } from "./transform";

const INSETS = { top: 10, right: 20, bottom: 30, left: 40 };

describe("fitTransform", () => {
  it("uses one scale for both axes, limited by the tighter dimension", () => {
    // 2 x 1 data region into a 400 x 400 viewport with insets: inner 340 x 360.
    const t = fitTransform({ xMin: 0, xMax: 2, yMin: 0, yMax: 1 }, 400, 400, INSETS);
    expect(t.scale).toBeCloseTo(170, 12);
    const dx = t.toScreenX(1) - t.toScreenX(0);
    const dy = t.toScreenY(0) - t.toScreenY(1);
    expect(dx).toBeCloseTo(dy, 12);
  });

  it("flips y so that +y points up on screen", () => {
    const t = fitTransform({ xMin: -1, xMax: 1, yMin: -1, yMax: 1 }, 200, 200, { top: 0, right: 0, bottom: 0, left: 0 });
    expect(t.toScreenY(1)).toBeLessThan(t.toScreenY(-1));
    expect(t.toScreenX(1)).toBeGreaterThan(t.toScreenX(-1));
  });

  it("centers the bounds inside the insets and keeps them fully visible", () => {
    const bounds = { xMin: 0.8, xMax: 1.0, yMin: -0.1, yMax: 0.1 };
    const t = fitTransform(bounds, 900, 500, INSETS);
    const midX = (INSETS.left + 900 - INSETS.right) / 2;
    const midY = (INSETS.top + 500 - INSETS.bottom) / 2;
    expect(t.toScreenX(0.9)).toBeCloseTo(midX, 9);
    expect(t.toScreenY(0)).toBeCloseTo(midY, 9);
    for (const [x, y] of [
      [bounds.xMin, bounds.yMin],
      [bounds.xMax, bounds.yMax],
    ] as const) {
      expect(t.toScreenX(x)).toBeGreaterThanOrEqual(INSETS.left - 1e-9);
      expect(t.toScreenX(x)).toBeLessThanOrEqual(900 - INSETS.right + 1e-9);
      expect(t.toScreenY(y)).toBeGreaterThanOrEqual(INSETS.top - 1e-9);
      expect(t.toScreenY(y)).toBeLessThanOrEqual(500 - INSETS.bottom + 1e-9);
    }
    expect(t.visible.xMin).toBeLessThanOrEqual(bounds.xMin);
    expect(t.visible.yMax).toBeGreaterThanOrEqual(bounds.yMax);
  });

  it("round-trips data -> screen -> data", () => {
    const t = fitTransform({ xMin: -0.2, xMax: 1.1, yMin: -0.3, yMax: 0.3 }, 731, 457, INSETS);
    for (const [x, y] of [
      [0.8150724480121228, 0],
      [-0.01215058560962404, 0],
      [0.987849414390376, 0.0001],
    ] as const) {
      expect(t.toDataX(t.toScreenX(x))).toBeCloseTo(x, 12);
      expect(t.toDataY(t.toScreenY(y))).toBeCloseTo(y, 12);
    }
  });

  it("is deterministic", () => {
    const a = fitTransform({ xMin: 0, xMax: 1, yMin: 0, yMax: 1 }, 640, 480, INSETS);
    const b = fitTransform({ xMin: 0, xMax: 1, yMin: 0, yMax: 1 }, 640, 480, INSETS);
    expect(a.toScreenX(0.123)).toBe(b.toScreenX(0.123));
    expect(a.toScreenY(0.456)).toBe(b.toScreenY(0.456));
  });

  it("rejects degenerate bounds and viewports instead of producing NaN", () => {
    expect(() => fitTransform({ xMin: 1, xMax: 1, yMin: 0, yMax: 1 }, 100, 100, INSETS)).toThrow(RangeError);
    expect(() => fitTransform({ xMin: 0, xMax: 1, yMin: 0, yMax: 1 }, 50, 30, INSETS)).toThrow(RangeError);
  });
});

describe("ticks and scale bar steps", () => {
  it("chooses 1/2/5 steps", () => {
    expect(niceStep(1.2, 6)).toBe(0.2);
    expect(niceStep(0.25, 5)).toBe(0.05);
    expect(niceStep(80000, 1)).toBe(100000);
    expect(niceStep(28000, 1)).toBe(20000);
  });

  it("produces clean multiples without floating-point noise", () => {
    expect(ticks(-0.11, 0.31, 0.1)).toEqual([-0.1, 0, 0.1, 0.2, 0.3]);
    expect(ticks(0.78, 0.9, 0.05)).toEqual([0.8, 0.85, 0.9]);
  });

  it("bounds sets of points with a margin", () => {
    expect(boundsOf([0, 2], [0, 1], 0.5)).toEqual({ xMin: -1, xMax: 3, yMin: -1, yMax: 2 });
  });
});

describe("scene geometry for the reference run", () => {
  const manifest = parseManifest(publicJson(`/research/${RUN_PATH}/manifest.json`));
  const trajectory = parseTrajectory(publicJson(`/research/${RUN_PATH}/trajectory.json`), manifest);

  it("places L1 where the JPL catalog does", () => {
    // NASA/JPL periodic orbits API, Earth-Moon system: L1 = 0.836915125772357.
    expect(lagrangeL1(manifest.model.mu)).toBeCloseTo(0.836915125772357, 12);
  });

  it("frames Earth, Moon and the orbit in the system view, and the orbit, L1 and Moon in the close-up", () => {
    const system = cameraBounds("system", manifest.model, trajectory);
    expect(system.xMin).toBeLessThan(manifest.model.earth.x - 6371 / manifest.model.length_scale_km);
    expect(system.xMax).toBeGreaterThan(manifest.model.moon.x);
    const region = cameraBounds("region", manifest.model, trajectory);
    expect(region.xMin).toBeLessThan(0.815);
    expect(region.xMax).toBeGreaterThan(manifest.model.moon.x);
    expect(region.yMax).toBeGreaterThan(0.1);
    expect(region.xMax - region.xMin).toBeLessThan(system.xMax - system.xMin);
  });

  it("draws bodies at true radius when large enough, and flags enlargement otherwise", () => {
    const { model } = manifest;
    const close = fitTransform(cameraBounds("region", model, trajectory), 900, 560, INSETS);
    const moon = bodyGlyph(close, model.moon.x, 0, model.moon.radius_km, model.length_scale_km);
    expect(moon.toScale).toBe(true);
    expect(moon.r).toBeCloseTo((1737.1 / model.length_scale_km) * close.scale, 9);

    const tiny = fitTransform(cameraBounds("system", model, trajectory), 300, 300, INSETS);
    const tinyMoon = bodyGlyph(tiny, model.moon.x, 0, model.moon.radius_km, model.length_scale_km);
    expect(tinyMoon.toScale).toBe(false);
    expect(tinyMoon.r).toBe(MIN_BODY_RADIUS_PX);
  });
});
