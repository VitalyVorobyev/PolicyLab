import { describe, expect, it } from "vitest";
import { basePrefix, researchBase, researchUrl } from "./data/paths";
import { normalizePath, stripBase, withBase } from "./router";

describe("base-path helpers", () => {
  it("normalizes base prefixes with and without trailing slashes", () => {
    expect(basePrefix("/")).toBe("");
    expect(basePrefix("")).toBe("");
    expect(basePrefix("/PolicyLab/")).toBe("/PolicyLab");
    expect(basePrefix("/PolicyLab")).toBe("/PolicyLab");
    expect(basePrefix("PolicyLab/")).toBe("/PolicyLab");
  });

  it("derives the research base URL from Vite base", () => {
    expect(researchBase("/")).toBe("/research");
    expect(researchBase("/PolicyLab/")).toBe("/PolicyLab/research");
    expect(researchUrl("catalog.json")).toBe("/research/catalog.json");
  });
});

describe("router path mapping", () => {
  it("normalizes trailing slashes while preserving root", () => {
    expect(normalizePath("/")).toBe("/");
    expect(normalizePath("/citizens/")).toBe("/citizens");
    expect(normalizePath("/gravity///")).toBe("/gravity");
  });

  it("strips default root base cleanly", () => {
    expect(stripBase("/", "/")).toBe("/");
    expect(stripBase("/citizens", "/")).toBe("/citizens");
    expect(stripBase("/gravity/", "/")).toBe("/gravity");
  });

  it("strips the /PolicyLab/ base prefix cleanly without misparsing siblings", () => {
    expect(stripBase("/PolicyLab", "/PolicyLab/")).toBe("/");
    expect(stripBase("/PolicyLab/", "/PolicyLab/")).toBe("/");
    expect(stripBase("/PolicyLab/citizens", "/PolicyLab/")).toBe("/citizens");
    expect(stripBase("/PolicyLab/gravity/", "/PolicyLab/")).toBe("/gravity");
    expect(stripBase("/PolicyLab2/citizens", "/PolicyLab/")).toBe("/PolicyLab2/citizens");
  });

  it("prepends the deployment base prefix for app-relative routes", () => {
    expect(withBase("/", "/")).toBe("/");
    expect(withBase("/citizens", "/")).toBe("/citizens");
    expect(withBase("/", "/PolicyLab/")).toBe("/PolicyLab/");
    expect(withBase("/citizens", "/PolicyLab/")).toBe("/PolicyLab/citizens");
    expect(withBase("/gravity#found", "/PolicyLab/")).toBe("/PolicyLab/gravity#found");
    expect(withBase("#main", "/PolicyLab/")).toBe("#main");
  });
});
