// @vitest-environment node
import { describe, expect, it } from "vitest";
import { publicJson } from "../../test/fixtures";
import { parseCatalog } from "../../lib/data/validate";
import { STUDIES } from "./studies";

describe("gallery copy", () => {
  it("lists the same studies with the same statuses as the published catalog", () => {
    const catalog = parseCatalog(publicJson("/research/catalog.json"));
    expect(STUDIES.map((s) => [s.study_id, s.status])).toEqual(catalog.studies.map((s) => [s.study_id, s.status]));
  });

  it("links only active or complete studies to a page", () => {
    for (const study of STUDIES) {
      expect(study.route !== undefined).toBe(study.status === "active" || study.status === "complete");
    }
  });
});
