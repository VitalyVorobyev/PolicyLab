/**
 * The plain-language findings of the cross-play matrix. Every number and every list in these
 * sentences is read from the run's cells (descriptive); a sentence whose premise does not hold in
 * the data is left out rather than reworded.
 */
import type { CitizensDemo, CrossplayCell, Distilled, Exams } from "../../lib/data/citizens";
import { cellOf, MODEL_KEYS, signedCoins, SIMPLE_RULE } from "./model";

export interface Finding {
  lead: string;
  text: string;
  /** Detail shown in a tooltip on the lead. */
  tip?: string | undefined;
}

const planner = (m: string): string => `Planner[${m}]`;

function pct(relative: number): number {
  return Math.round(Math.abs(relative) * 100);
}

function range(values: number[]): string {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return lo === hi ? `${lo} %` : `${lo}–${hi} %`;
}

function list(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1] ?? ""}`;
}

/** Models that passed the whole-game exam on the closed-loop regime (R-I). */
export function wholeGameModels(exams: Exams | null): string[] {
  if (exams === null) return [];
  return exams.rows.filter((r) => r.regime === "R-I" && r.pass_outcomes && r.label.includes("closed-loop")).map((r) => r.model);
}

/** Models that passed the next-move exam but not the whole-game exam on R-I. */
export function nextMoveOnlyModels(exams: Exams | null): string[] {
  if (exams === null) return [];
  return exams.rows.filter((r) => r.regime === "R-I" && r.pass_likelihood && !r.pass_outcomes && (MODEL_KEYS as readonly string[]).includes(r.model)).map((r) => r.model);
}

export function crossplayFindings(demo: CitizensDemo): Finding[] {
  const out: Finding[] = [];
  const cell = (p: string, c: string): CrossplayCell | undefined => cellOf(demo, p, c);

  // 1. Home advantage.
  const home = MODEL_KEYS.map((m) => ({ m, c: cell(planner(m), m) }));
  const bestAtHome = home.every((h) => h.c !== undefined && demo.crossplay.cells.filter((c) => c.column === h.m).every((c) => c.total_surplus_mean <= (h.c?.total_surplus_mean ?? 0)));
  if (bestAtHome && home.every((h) => h.c !== undefined && h.c.vs_k_star.relative > 0)) {
    const sorted = [...home].sort((a, b) => (a.c?.vs_k_star.relative ?? 0) - (b.c?.vs_k_star.relative ?? 0));
    const lo = sorted[0];
    const hi = sorted[sorted.length - 1];
    out.push({
      lead: "Each AI governor is the best at home.",
      text: `On the citizens it was trained against, no other governor earns more. Its margin over the simple rule tuned for them ranges from ${pct(lo?.c?.vs_k_star.relative ?? 0)} % (${lo?.m ?? ""}) to ${pct(hi?.c?.vs_k_star.relative ?? 0)} % (${hi?.m ?? ""}).`,
    });
  }

  // 2. Transfer between worlds.
  const wins: [string, string][] = [];
  let away = 0;
  for (const x of MODEL_KEYS) {
    for (const y of MODEL_KEYS) {
      if (x === y) continue;
      const c = cell(planner(x), y);
      if (c === undefined) continue;
      away++;
      if (c.vs_k_star.relative > 0) wins.push([x, y]);
    }
  }
  const learners = ["B3", "B23"];
  const isLearnerPair = (p: [string, string]): boolean => learners.includes(p[0]) && learners.includes(p[1]);
  if (away > 0) {
    if (wins.length === 2 && wins.every(isLearnerPair)) {
      out.push({
        lead: "Transfer stays within a family.",
        text: "Away from home, only the governors trained on the two learning models (B3, B23) beat the simple rule, each on the other's model. No governor transfers between the B1/B2 and B3/B23 worlds.",
      });
    } else {
      out.push({
        lead: "Away from home, most governors lose.",
        text: `${away - wins.length} of the ${away} away pairings earn less than the simple rule tuned for those citizens.`,
      });
    }
  }

  // 3. The best available judge.
  const judges = wholeGameModels(demo.exams);
  const oneStep = nextMoveOnlyModels(demo.exams);
  const judge = judges.length === 1 ? judges[0] : undefined;
  if (judge !== undefined && oneStep.length > 0) {
    const cells = oneStep.map((m) => ({ m, c: cell(planner(m), judge) }));
    if (cells.every((x) => x.c !== undefined && x.c.vs_k_star.relative < 0)) {
      const losses = cells.map((x) => pct(x.c?.vs_k_star.relative ?? 0));
      const worst = Math.max(...losses);
      const most = cells.filter((x) => pct(x.c?.vs_k_star.relative ?? 0) >= worst - 1).map((x) => x.m);
      out.push({
        lead: `On ${judge}, next-move validation was not enough.`,
        text: `${judge} is the only model that also passed the whole-game exam. Every model that passed only the next-move exam (${list(oneStep)}) produced a governor that loses ${range(losses)} of the surplus to the simple rule on ${judge}; ${most.length === 1 ? `the one trained on ${most[0] ?? ""} loses most` : `those trained on ${list(most)} lose most`}.`,
      });
    }
  }

  // 4. The simple rule and the mixture governor.
  const k22 = demo.crossplay.columns.map((c) => cell(SIMPLE_RULE, c)).filter((c): c is CrossplayCell => c !== undefined);
  const parts: string[] = [];
  if (k22.length === demo.crossplay.columns.length && k22.length > 0) {
    parts.push(`The published rule (k = 22) is within about ${Math.max(...k22.map((c) => Math.round(Math.abs(c.vs_k_star.relative) * 100)))} % of the best re-tuned simple rule in every world.`);
  }
  const beatsOthers = ["B1", "B2"].filter((col) => {
    const ref = cell(SIMPLE_RULE, col);
    return ref !== undefined && MODEL_KEYS.filter((m) => m !== col).every((m) => (cell(planner(m), col)?.total_surplus_mean ?? Infinity) < ref.total_surplus_mean);
  });
  if (beatsOthers.length === 2) parts.push("and beats every governor trained on another model on B1 and B2.");
  const learnerGains = learners.flatMap((col) =>
    [...learners.map(planner), "Planner[Mix]"].map((p) => cell(p, col)).filter((c): c is CrossplayCell => c !== undefined),
  );
  if (learnerGains.length > 0 && learnerGains.every((c) => c.vs_k22.relative > 0)) {
    parts.push(
      `On the learner models (B3, B23), the learner-trained and mixture governors beat it by ${range(learnerGains.map((c) => pct(c.vs_k22.relative)))}, gains that may reflect exploiting the model.`,
    );
  }
  const rob = demo.crossplay.robustness;
  const mix = rob["Planner[Mix]"];
  const rule = rob[SIMPLE_RULE];
  let tip: string | undefined;
  if (mix !== undefined && rule !== undefined) {
    parts.push(
      `By its worst world, the governor trained on all four is the more robust: ${Math.round(mix.worst_ratio * 100)} % of what that world's own AI governor earns, against ${Math.round(rule.worst_ratio * 100)} % for the simple rule.`,
    );
    const mixB1 = cell("Planner[Mix]", "B1");
    const ci = mixB1?.vs_k_star.ci ?? null;
    if (mixB1 !== undefined && ci !== null && ci[0] < 0 && ci[1] > 0) {
      tip = `On B1 the mixture governor roughly matches the tuned simple rule: ${signedCoins(mixB1.vs_k_star.difference)} coins per game, 95 % interval ${signedCoins(ci[0])} to ${signedCoins(ci[1])}.`;
    }
  }
  if (parts.length > 0) {
    // "is within … every world." + "and beats …" read as one sentence.
    const text = parts.join(" ").replace(" every world. and beats", " every world and beats");
    out.push({ lead: "The simple rule is a strong default, not the most robust choice.", text, tip });
  }
  return out;
}

/** The B1 judge's narrow pass: game outcomes inside the human range only through the tolerance, and outcomes outside it. */
export function narrowPass(exams: Exams | null, model: string, regime = "R-I"): string | null {
  if (exams === null) return null;
  const rows = exams.metrics.filter((m) => m.model === model && m.regime === regime);
  if (rows.length === 0) return null;
  const NAMES: Record<string, string> = { surplus: "surplus", gini: "inequality (Gini)", depletion_round: "when the pool runs dry", players_active: "players still active" };
  const viaTolerance = rows.filter((m) => m.in_range && (m.sim_mean < m.human_ci[0] || m.sim_mean > m.human_ci[1])).map((m) => NAMES[m.metric] ?? m.metric);
  const outside = rows.filter((m) => !m.in_range).map((m) => NAMES[m.metric] ?? m.metric);
  const parts: string[] = [];
  if (viaTolerance.length > 0) parts.push(`${list(viaTolerance)} sits just outside the human range and counts as inside only through the pre-set tolerance`);
  if (outside.length > 0) parts.push(`${list(outside)} is outside it`);
  return parts.length === 0 ? null : `${parts.join("; ")}.`;
}

/**
 * Whether the one-weight curves explain the AI governors: per planner (seed means, on its own model's
 * distillation games, in sample) the planner's and its curve's surplus minus the tuned simple rule.
 * The caveat is returned only when no curve keeps half of its governor's gain.
 */
export function distillCaveat(distilled: Distilled): { tip: string } | null {
  const mean = (v: number[]): number => v.reduce((s, x) => s + x, 0) / Math.max(1, v.length);
  const rows = distilled.planners.map((p) => ({
    model: p.policy.replace(/^Planner\[(.*)\]$/, "$1"),
    planner: mean(p.planner_minus_k_star_per_seed),
    curve: mean(p.w_rule_minus_k_star_per_seed),
    r2: p.r2_per_seed.filter((v): v is number => v !== null),
  }));
  if (rows.length === 0 || rows.some((r) => r.planner > 0 && r.curve >= 0.5 * r.planner)) return null;
  const r2 = rows.flatMap((r) => r.r2);
  const fmt = (v: number): string => `${v >= 0 ? "+" : "−"}${Math.round(Math.abs(v)).toLocaleString("en-US")}`;
  const parts = rows.map((r) => `AI · ${r.model}: governor ${fmt(r.planner)}, its curve ${fmt(r.curve)}`);
  return {
    tip:
      `Coins per game against the simple rule tuned for the governor's own model, on the games the curves were fitted to (seed means): ${parts.join("; ")}. ` +
      (r2.length > 0 ? `The curves match the individual offers only moderately (R² ${Math.min(...r2).toFixed(2)}–${Math.max(...r2).toFixed(2)}).` : ""),
  };
}

