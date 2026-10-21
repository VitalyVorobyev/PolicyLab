import type { CitizensDemo, CrossplayCell, GameRecord } from "../../lib/data/citizens";

/** The four citizen models the AI governors were trained and evaluated against (cross-play columns). */
export const MODEL_KEYS = ["B1", "B2", "B3", "B23"] as const;
export type ModelKey = (typeof MODEL_KEYS)[number];

export const PEOPLE = "people";

export interface CitizenInfo {
  key: string;
  name: string;
  /** One line: what the model is. */
  line: string;
  /** CSS colour token of a model (categorical slot, fixed order); none for people and LLM players. */
  color?: string;
}

export const CITIZENS: readonly CitizenInfo[] = [
  {
    key: "B1",
    name: "Neural clone",
    line: "A recurrent neural network trained to imitate what players return, given their offers and what everyone returned before.",
    color: "var(--color-model-b1)",
  },
  {
    key: "B2",
    name: "Conditional cooperators",
    line: "A 12-parameter rule: return more when offered more than an equal share, and when the others returned more.",
    color: "var(--color-model-b2)",
  },
  {
    key: "B3",
    name: "Learners",
    line: "Experience-weighted learning: players drift towards the return shares that paid off, in coins kept and in later offers.",
    color: "var(--color-model-b3)",
  },
  {
    key: "B23",
    name: "Learners who match",
    line: "The B3 learners with a pull towards returning the same share the others returned.",
    color: "var(--color-model-b23)",
  },
  { key: PEOPLE, name: "Real people", line: "Recorded games of the original experiment, four people per game." },
  {
    key: "B4-calibrated",
    name: "LLM players",
    line: "Claude Haiku plays each citizen from the written rules; the mix of player types is calibrated to human play.",
  },
  {
    key: "B4-plain",
    name: "LLM, uncalibrated",
    line: "Claude Haiku plays each citizen from the written rules, with no calibration to human play.",
  },
];

export interface GovernorInfo {
  /** Policy name in the data. */
  key: string;
  short: string;
  name: string;
  line: string;
  kind: "rule" | "ai";
  trainedOn?: ModelKey | "Mix";
}

export const SIMPLE_RULE = "Interpolating k=22";

export const GOVERNORS: readonly GovernorInfo[] = [
  { key: "Equal", short: "Equal", name: "Equal split", line: "Offers every player the same share of the pool.", kind: "rule" },
  {
    key: "Proportional",
    short: "Proportional",
    name: "Proportional",
    line: "Offers each player a share in proportion to what they returned last round.",
    kind: "rule",
  },
  {
    key: SIMPLE_RULE,
    short: "Simple rule",
    name: "The simple rule (k = 22)",
    line: "DeepMind's published, human-tested rule: split equally while the pool is full, reward contributors as it runs low.",
    kind: "rule",
  },
  ...MODEL_KEYS.map(
    (m): GovernorInfo => ({
      key: `Planner[${m}]`,
      short: `AI · ${m}`,
      name: `AI governor trained on ${m}`,
      line: `A neural policy trained by reinforcement learning to maximise the players' total surplus against the ${citizenName(m).toLowerCase()} (${m}).`,
      kind: "ai",
      trainedOn: m,
    }),
  ),
  {
    key: "Planner[Mix]",
    short: "AI · mix",
    name: "AI governor trained on all four",
    line: "Trained the same way against a mixture of the four models, one model drawn per game.",
    kind: "ai",
    trainedOn: "Mix",
  },
  {
    key: "M2",
    short: "DeepMind AI",
    name: "DeepMind's AI governor (M2)",
    line: "The AI mechanism of the original study, as played by real people.",
    kind: "ai",
  },
];

export function citizenName(key: string): string {
  return CITIZENS.find((c) => c.key === key)?.name ?? key;
}

export function citizenInfo(key: string): CitizenInfo | undefined {
  return CITIZENS.find((c) => c.key === key);
}

export function governorInfo(key: string): GovernorInfo | undefined {
  return GOVERNORS.find((g) => g.key === key);
}

export function modelColor(key: string): string {
  return citizenInfo(key)?.color ?? "var(--color-graphite)";
}

/** The recorded games for a (governor, citizens) pair, in game order; [] if none. */
export function gamesFor(demo: CitizensDemo, governor: string, citizens: string): GameRecord[] {
  if (citizens === PEOPLE) return demo.human_games.filter((h) => h.governor === governor);
  return demo.traces.filter((t) => t.governor === governor && t.citizens === citizens).sort((a, b) => a.game - b.game);
}

/** Citizens options that have games in this run, in display order. */
export function citizenOptions(demo: CitizensDemo): CitizenInfo[] {
  return CITIZENS.filter((c) => (c.key === PEOPLE ? demo.human_games.length > 0 : demo.traces.some((t) => t.citizens === c.key)));
}

/** Governors that have games in this run (for any citizens), in display order. */
export function governorOptions(demo: CitizensDemo): GovernorInfo[] {
  return GOVERNORS.filter((g) => demo.traces.some((t) => t.governor === g.key) || demo.human_games.some((h) => h.governor === g.key));
}

export function hasGames(demo: CitizensDemo, governor: string, citizens: string): boolean {
  return gamesFor(demo, governor, citizens).length > 0;
}

/** The governor to fall back to when a citizens choice has no games for the current one. */
export function fallbackGovernor(demo: CitizensDemo, citizens: string, preferred: string): string {
  if (hasGames(demo, preferred, citizens)) return preferred;
  const order = [SIMPLE_RULE, ...GOVERNORS.map((g) => g.key)];
  return order.find((g) => hasGames(demo, g, citizens)) ?? preferred;
}

/** The citizens to fall back to when a governor has no games for the current ones. */
export function fallbackCitizens(demo: CitizensDemo, governor: string, preferred: string): string {
  if (hasGames(demo, governor, preferred)) return preferred;
  return CITIZENS.map((c) => c.key).find((c) => hasGames(demo, governor, c)) ?? preferred;
}

export function cellOf(demo: CitizensDemo, policy: string, column: string): CrossplayCell | undefined {
  return demo.crossplay.cells.find((c) => c.policy === policy && c.column === column);
}

/** Coins with thousands separators and no decimals. */
export function coins(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

const MINUS = "−";

/** A signed whole percentage, e.g. "+98 %" or "−28 %". */
export function signedPercent(relative: number): string {
  const v = Math.round(relative * 100);
  if (v === 0) return "0 %";
  return `${v > 0 ? "+" : MINUS}${Math.abs(v)} %`;
}

export function signedCoins(value: number): string {
  const v = Math.round(value);
  if (v === 0) return "0";
  return `${v > 0 ? "+" : MINUS}${Math.abs(v).toLocaleString("en-US")}`;
}

/** k for display: two significant digits below 10, whole numbers above. */
export function kLabel(k: number): string {
  if (k >= 10) return Math.round(k).toString();
  if (k >= 1) return k.toFixed(1);
  return k.toPrecision(1);
}
