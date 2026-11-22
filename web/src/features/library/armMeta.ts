/** Visual encoding metadata (color + dash pattern + marker) for The Last Library arms. */

export type MarkerKind = "circle" | "square" | "triangle" | "diamond" | "cross";

export interface ArmMeta {
  arm: string;
  label: string;
  short: string;
  color: string;
  dash: string;
  marker: MarkerKind;
  description: string;
}

export const ARM_META: Record<string, ArmMeta> = {
  oracle: {
    arm: "oracle",
    label: "Oracle (true costs)",
    short: "Oracle",
    color: "#8ea0b8",
    dash: "2 3",
    marker: "circle",
    description: "Knows true latent terrain costs with zero uncertainty (regret lower bound = 0).",
  },
  none: {
    arm: "none",
    label: "None (no inheritance)",
    short: "None",
    color: "#94a3b8",
    dash: "5 4",
    marker: "cross",
    description: "Starts every generation from uniform prior with no cultural memory.",
  },
  unlimited: {
    arm: "unlimited",
    label: "Unlimited posterior",
    short: "Unlimited",
    color: "#38bdf8",
    dash: "7 3",
    marker: "diamond",
    description: "Passes exact uncompressed floating-point posterior across generations.",
  },
  single_lifetime: {
    arm: "single_lifetime",
    label: "Single lifetime control",
    short: "Single life",
    color: "#7dd3fc",
    dash: "3 3",
    marker: "diamond",
    description: "Single continuous learner receiving the same cumulative experience (matches Unlimited).",
  },
  example_random: {
    arm: "example_random",
    label: "Random examples",
    short: "Ex. random",
    color: "#fb923c",
    dash: "4 3",
    marker: "cross",
    description: "Uniformly samples up to floor(B/5) raw observation pairs within the bit budget.",
  },
  example_pedagogical: {
    arm: "example_pedagogical",
    label: "Pedagogical examples",
    short: "Ex. ped.",
    color: "#eb6834",
    dash: "6 2",
    marker: "square",
    description: "Greedily picks up to floor(B/5) observations minimizing KL divergence to sender belief.",
  },
  rule: {
    arm: "rule",
    label: "Abstract rule",
    short: "Rule",
    color: "#e3a72f",
    dash: "",
    marker: "circle",
    description: "Encodes MAP cost levels (19-bit dense ternary packing or 5-bit sparse top terrains) with trust 0.88.",
  },
  rule_confidence: {
    arm: "rule_confidence",
    label: "Confidence-weighted rule",
    short: "Rule+conf",
    color: "#a78bfa",
    dash: "8 2 2 2",
    marker: "triangle",
    description: "Encodes `(level, confidence bin)` using 32-bit dense nibbles or 7-bit sparse entries.",
  },
  rule_hedged: {
    arm: "rule_hedged",
    label: "Hedged rule (receiver h=0.30)",
    short: "Rule+hedge",
    color: "#4fd1b0",
    dash: "",
    marker: "diamond",
    description: "Transmits identical bits as Rule, while the receiver blends 30% uniform hazard hedging.",
  },
  placebo_random: {
    arm: "placebo_random",
    label: "Random bit placebo",
    short: "Placebo rand",
    color: "#f2785c",
    dash: "2 3",
    marker: "cross",
    description: "Random bits passed through the rule decoder (sanity control).",
  },
  placebo_shuffled: {
    arm: "placebo_shuffled",
    label: "Shuffled-seed placebo",
    short: "Placebo shuf",
    color: "#e11d48",
    dash: "3 2 1 2",
    marker: "cross",
    description: "Valid rule encoded from an unrelated seed (sanity control).",
  },
};

export function armInfo(arm: string): ArmMeta {
  return (
    ARM_META[arm] ?? {
      arm,
      label: arm,
      short: arm,
      color: "#a8b5c7",
      dash: "",
      marker: "circle",
      description: "",
    }
  );
}
