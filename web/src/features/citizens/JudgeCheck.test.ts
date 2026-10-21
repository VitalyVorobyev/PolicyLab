import { describe, expect, it } from "vitest";
import { judgeTakeaway, type JudgeScore } from "./JudgeCheck";

const score = (model: string, agreements: number, scorable = 8): JudgeScore => ({ model, scorable, agreements, excluded: 6, notScorable: 3 });

describe("judge takeaway", () => {
  it("says B1 is not the uniquely right judge when the learner models also order every comparison right", () => {
    const text = judgeTakeaway([score("B1", 8), score("B2", 7), score("B3", 8), score("B23", 8)]);
    expect(text).toMatch(/^B1 orders all of them right — but so do the learner models \(B3 and B23\)/);
  });

  it("names B1 alone when no other model of people is perfect", () => {
    expect(judgeTakeaway([score("B1", 8), score("B2", 7), score("B3", 6), score("B23", 7)])).toBe("B1 is the only model of people that orders all of them right.");
  });

  it("says nothing when B1 misses a comparison", () => {
    expect(judgeTakeaway([score("B1", 7), score("B3", 8), score("B23", 8)])).toBeNull();
  });
});
