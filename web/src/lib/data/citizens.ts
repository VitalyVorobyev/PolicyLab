/**
 * Model Citizens web demo run (docs/data-contract.md, "Model Citizens addendum (web demo)"):
 * the manifest (C4 citizens layout; the CR3BP model and solver blocks do not apply) and its single
 * `citizens_demo` artifact. Everything is validated before the page sees it; the page only renders.
 */
import { DataError } from "./errors";
import type { Artifact, Environment, Metric, RunStatus, Source, Validation } from "./types";
import { RUN_STATUSES } from "./types";
import {
  checkRunId,
  parseArtifact,
  parseEnvironment,
  parseMetrics,
  parseSource,
  parseValidation,
  Reader,
  root,
  RUN_ID,
  TIMESTAMP,
} from "./validate";

export const CITIZENS_STUDY_ID = "model-citizens";
export const PLAYERS = 4;

export interface CitizensGame {
  players: number;
  rounds: number;
  max_pool: number;
  growth_factor: number;
  pool_floor: number;
  pool_rule: string;
}

export interface CitizensManifest {
  schema_version: 1;
  run_id: string;
  study_id: typeof CITIZENS_STUDY_ID;
  experiment_id: string;
  status: RunStatus;
  created_at: string;
  source: Source;
  environment: Environment;
  random_seed: number | null;
  artifacts: Artifact[];
  metrics: Record<string, Metric>;
  limitations: string[];
  validation: Validation | null;
}

/** One game: coin values rounded to 2 decimals by the exporter. */
export interface GameRecord {
  rounds: number;
  /** rounds + 1 values: the pool before each round, then after the last. */
  pool: number[];
  /** rounds x 4 */
  offers: number[][];
  returns: number[][];
  kept: number[][];
  /** Running sum of kept coins over players, one per round. */
  cumulative_surplus: number[];
  total_surplus: number;
}

export interface CitizenTrace extends GameRecord {
  governor: string;
  /** "B1" | "B2" | "B3" | "B23" | "B4-<variant>" */
  citizens: string;
  game: number;
  seed: number | null;
  /** LLM games: the C2 held-out run or a C4 run. */
  source: "c2" | "c4" | null;
}

export interface HumanGame extends GameRecord {
  governor: string;
  mechanism: string;
  cohort: string;
  game_id: string;
  split: "training" | "heldout";
}

export interface Difference {
  versus: string;
  difference: number;
  relative: number;
  ci: [number, number] | null;
  seeds_positive: number | null;
  seeds_negative: number | null;
}

export interface CrossplayCell {
  policy: string;
  column: string;
  total_surplus_mean: number;
  total_surplus_per_seed: number[];
  seeds: (number | null)[];
  gini_mean: number;
  vs_k_star: Difference;
  vs_k22: Difference;
  /** Optional per-cell game metrics from a supplementary run (e.g. depletion_round_mean). */
  extra_metrics: Record<string, number> | null;
}

export interface Robustness {
  ratios: Record<string, number>;
  worst_ratio: number;
  worst_column: string;
}

export interface Crossplay {
  columns: string[];
  policies: string[];
  games: number;
  k_star: Record<string, number>;
  cells: CrossplayCell[];
  /** Per policy: mean as a share of each column's own planner; empty when not exported. */
  robustness: Record<string, Robustness>;
}

export interface KTuning {
  ln_k: number[];
  curves: Record<string, number[]>;
  k_star: Record<string, { k: number; ln_k: number }>;
  k_published: number;
  games: number;
}

export interface DistilledPlanner {
  policy: string;
  seeds: number[];
  w_mean: (number | null)[];
  fitted_k_per_seed: number[];
}

export interface Distilled {
  bins: number;
  bin_edges: number[];
  planners: DistilledPlanner[];
  reference: { rule: string; pool_fraction: number[]; w: number[] };
}

export interface ExamRow {
  regime: string;
  model: string;
  delta_ell: number;
  delta_ci: [number, number];
  pass_likelihood: boolean;
  metrics_in_range: number;
  pass_outcomes: boolean;
  label: string;
  units: number;
}

export interface ExamMetric {
  regime: string;
  model: string;
  metric: string;
  human_ci: [number, number];
  tolerance: number;
  sim_mean: number;
  in_range: boolean;
}

export interface Exams {
  regimes: string[];
  closed_loop_tier: Record<string, boolean>;
  rows: ExamRow[];
  metrics: ExamMetric[];
}

export interface B4Record {
  variant: string;
  regime: string;
  games_complete: number;
  metrics_in_range: number;
  pass_outcomes: boolean;
  closed_loop: boolean;
  label: string;
  counts_as_validated: boolean;
}

export interface B4Row {
  governor: string;
  variant: string;
  source: "c2" | "c4";
  games_complete: number;
  games_discarded: number;
  total_surplus_mean: number | null;
  total_surplus_sd: number | null;
  cli_version: string | null;
}

export interface Hypotheses {
  verdicts: { H1: string; H2: string; H2b_confirmed_losses: string[] };
  confirmed: Record<string, string[]>;
  rows: Record<string, { own_model_gain: boolean; possible_exploitation: boolean; optimiser_limited: boolean }>;
  delta: number;
  h2_b4: Record<string, { verdict: string | null; validated_evaluation_population: boolean | null; label: string | null }> | null;
}

export interface E1Table {
  columns: { name: string; unit: string | null; definition: string }[];
  rows: (number | string | null)[][];
}

export interface E1 {
  run_id: string;
  experiment_id: string;
  tables: Record<string, E1Table>;
}

export interface CitizensDemo {
  schema_version: 1;
  run_id: string;
  game: CitizensGame;
  trace_games: number[];
  traces: CitizenTrace[];
  human_games: HumanGame[];
  data_attribution: string;
  crossplay: Crossplay;
  k_tuning: KTuning;
  distilled: Distilled;
  exams: Exams | null;
  b4_c2: { records: B4Record[] } | null;
  b4_rows: B4Row[] | null;
  hypotheses: Hypotheses | null;
  e1: E1 | null;
}

export interface LoadedCitizensRun {
  manifest: CitizensManifest;
  demo: CitizensDemo;
  manifestUrl: string;
  artifactUrl: string;
}

// --- helpers ------------------------------------------------------------------------------------

function numbers(r: Reader, key: string, length?: number): number[] {
  const values = r.array(key);
  if (length !== undefined && values.length !== length) r.fail(key, `an array of ${length} numbers`);
  return values.map((v, i) => {
    if (typeof v !== "number" || !Number.isFinite(v)) r.fail(`${key}[${i}]`, "a finite number", v);
    return v;
  });
}

function nullableNumbers(r: Reader, key: string, length?: number): (number | null)[] {
  const values = r.array(key);
  if (length !== undefined && values.length !== length) r.fail(key, `an array of ${length} values`);
  return values.map((v, i) => {
    if (v === null) return null;
    if (typeof v !== "number" || !Number.isFinite(v)) r.fail(`${key}[${i}]`, "a finite number or null", v);
    return v;
  });
}

function matrix(r: Reader, key: string, rows: number, columns: number): number[][] {
  const values = r.array(key);
  if (values.length !== rows) r.fail(key, `${rows} rows`);
  return values.map((row, i) => {
    if (!Array.isArray(row) || row.length !== columns) r.fail(`${key}[${i}]`, `${columns} numbers`, row);
    return (row as unknown[]).map((v, j) => {
      if (typeof v !== "number" || !Number.isFinite(v)) r.fail(`${key}[${i}][${j}]`, "a finite number", v);
      return v;
    });
  });
}

function pair(r: Reader, key: string): [number, number] {
  const [a, b] = numbers(r, key, 2);
  return [a ?? 0, b ?? 0];
}

function boolField(r: Reader, key: string): boolean {
  return r.boolean(key);
}

function nullableString(r: Reader, key: string): string | null {
  return r.has(key) ? r.nullable(key, (k) => r.string(k)) : null;
}

function record<T>(r: Reader, read: (sub: Reader, key: string) => T): Record<string, T> {
  const out: Record<string, T> = {};
  for (const key of r.keys()) out[key] = read(r, key);
  return out;
}

/** Fields of one game; checks the shapes and the identities the page relies on. */
function parseGameRecord(r: Reader, rounds: number): GameRecord {
  const n = r.integer("rounds", 1, rounds);
  const pool = numbers(r, "pool", n + 1);
  const offers = matrix(r, "offers", n, PLAYERS);
  const returns = matrix(r, "returns", n, PLAYERS);
  const kept = matrix(r, "kept", n, PLAYERS);
  const cumulative = numbers(r, "cumulative_surplus", n);
  const total = r.finite("total_surplus");
  if (pool.some((p) => p < 0)) r.fail("pool", "non-negative pool levels");
  // Rounded values: kept = offer - returned within the two roundings.
  offers.forEach((row, t) => {
    row.forEach((e, i) => {
      const k = kept[t]?.[i] ?? Number.NaN;
      const c = returns[t]?.[i] ?? Number.NaN;
      if (!(Math.abs(k - (e - c)) <= 0.011)) r.fail(`kept[${t}][${i}]`, `offer − returned (${e} − ${c})`, k);
    });
  });
  const last = cumulative[n - 1] ?? Number.NaN;
  if (!(Math.abs(last - total) <= 0.011)) r.fail("cumulative_surplus", `a last value equal to total_surplus ${total}`, last);
  return { rounds: n, pool, offers, returns, kept, cumulative_surplus: cumulative, total_surplus: total };
}

function parseDifference(r: Reader): Difference {
  return {
    versus: r.string("versus"),
    difference: r.finite("difference"),
    relative: r.finite("relative"),
    ci: r.nullable("ci", (k) => pair(r, k)),
    seeds_positive: r.nullable("seeds_positive", (k) => r.integer(k, 0)),
    seeds_negative: r.nullable("seeds_negative", (k) => r.integer(k, 0)),
  };
}

// --- manifest -------------------------------------------------------------------------------------

export function parseCitizensManifest(value: unknown, source = "manifest.json"): CitizensManifest {
  const r = root(source, value, "manifest");
  const manifest: CitizensManifest = {
    schema_version: 1,
    run_id: r.matching("run_id", RUN_ID, "a run id of lowercase letters, digits, '.', '_' and '-' (at most 128)"),
    study_id: r.literal("study_id", CITIZENS_STUDY_ID),
    experiment_id: r.string("experiment_id"),
    status: r.oneOf("status", RUN_STATUSES),
    created_at: r.matching("created_at", TIMESTAMP, 'a UTC time as "YYYY-MM-DDTHH:MM:SSZ"'),
    source: parseSource(r.object("source")),
    environment: parseEnvironment(r.object("environment")),
    random_seed: r.nullable("random_seed", (k) => r.integer(k)),
    artifacts: r.objects("artifacts", 1).map(parseArtifact),
    metrics: parseMetrics(r.object("metrics")),
    limitations: r.strings("limitations", 1),
    validation: r.has("validation") ? parseValidation(r.object("validation")) : null,
  };
  if (manifest.source.git_dirty && !manifest.limitations.some((l) => l.includes("git_dirty"))) {
    throw new DataError("invalid", source, "source.git_dirty is true but no limitation discloses it (expected a limitation mentioning git_dirty).");
  }
  if (manifest.status === "validated") {
    const failed = manifest.validation?.checks.filter((c) => !c.passed) ?? null;
    if (failed === null || manifest.validation?.checks.length === 0 || failed.length > 0) {
      throw new DataError("invalid", source, 'status is "validated" but the manifest does not record passing validation checks.');
    }
  }
  if (manifest.artifacts.length !== 1 || manifest.artifacts[0]?.kind !== "citizens_demo") {
    throw new DataError("invalid", source, 'artifacts: expected exactly one artifact of kind "citizens_demo".');
  }
  return manifest;
}

// --- the demo artifact -------------------------------------------------------------------------------

function parseCrossplay(r: Reader): Crossplay {
  const columns = r.strings("columns", 1);
  const policies = r.strings("policies", 1);
  const cells = r.objects("cells", 1).map(
    (c): CrossplayCell => ({
      policy: c.oneOf("policy", policies),
      column: c.oneOf("column", columns),
      total_surplus_mean: c.finite("total_surplus_mean"),
      total_surplus_per_seed: numbers(c, "total_surplus_per_seed"),
      seeds: c.array("seeds").map((s, i) => {
        if (s !== null && (typeof s !== "number" || !Number.isSafeInteger(s))) c.fail(`seeds[${i}]`, "an integer or null", s);
        return s as number | null;
      }),
      gini_mean: c.finite("gini_mean"),
      vs_k_star: parseDifference(c.object("vs_k_star")),
      vs_k22: parseDifference(c.object("vs_k22")),
      extra_metrics: c.has("extra_metrics") ? c.nullable("extra_metrics", (k) => record(c.object(k), (o, key) => o.finite(key))) : null,
    }),
  );
  for (const p of policies) {
    for (const col of columns) {
      if (!cells.some((c) => c.policy === p && c.column === col)) r.fail("cells", `a cell for ${p} in column ${col}`, undefined);
    }
  }
  const k = r.object("k_star");
  const rob = r.has("robustness") ? r.object("robustness") : null;
  const robustness: Record<string, Robustness> =
    rob === null || !rob.has("policies")
      ? {}
      : record(rob.object("policies"), (o, key) => {
          const p = o.object(key);
          return {
            ratios: record(p.object("ratios"), (q, col) => q.positive(col)),
            worst_ratio: p.positive("worst_ratio"),
            worst_column: p.oneOf("worst_column", columns),
          };
        });
  return { columns, policies, games: r.integer("games", 1), k_star: record(k, (o, key) => o.positive(key)), cells, robustness };
}

function parseKTuning(r: Reader): KTuning {
  const lnK = numbers(r, "ln_k");
  if (lnK.length < 2 || lnK.some((v, i) => i > 0 && v <= (lnK[i - 1] ?? v))) r.fail("ln_k", "at least two increasing grid points");
  const curves = r.object("curves");
  const ks = r.object("k_star");
  return {
    ln_k: lnK,
    curves: record(curves, (o, key) => numbers(o, key, lnK.length)),
    k_star: record(ks, (o, key) => {
      const s = o.object(key);
      return { k: s.positive("k"), ln_k: s.finite("ln_k") };
    }),
    k_published: r.positive("k_published"),
    games: r.integer("games", 1),
  };
}

function parseDistilled(r: Reader): Distilled {
  const bins = r.integer("bins", 1);
  const ref = r.object("reference");
  const fraction = numbers(ref, "pool_fraction");
  return {
    bins,
    bin_edges: numbers(r, "bin_edges", bins + 1),
    planners: r.objects("planners").map((p) => ({
      policy: p.string("policy"),
      seeds: numbers(p, "seeds"),
      w_mean: nullableNumbers(p, "w_mean", bins),
      fitted_k_per_seed: numbers(p, "fitted_k_per_seed"),
    })),
    reference: { rule: ref.string("rule"), pool_fraction: fraction, w: numbers(ref, "w", fraction.length) },
  };
}

function parseExams(r: Reader): Exams {
  const tier = r.object("closed_loop_tier");
  return {
    regimes: r.strings("regimes", 1),
    closed_loop_tier: record(tier, (o, key) => o.boolean(key)),
    rows: r.objects("rows", 1).map((e) => ({
      regime: e.string("regime"),
      model: e.string("model"),
      delta_ell: e.finite("delta_ell"),
      delta_ci: pair(e, "delta_ci"),
      pass_likelihood: boolField(e, "pass_likelihood"),
      metrics_in_range: e.integer("metrics_in_range", 0, 4),
      pass_outcomes: boolField(e, "pass_outcomes"),
      label: e.string("label"),
      units: e.integer("units", 1),
    })),
    metrics: r.objects("metrics").map((m) => ({
      regime: m.string("regime"),
      model: m.string("model"),
      metric: m.string("metric"),
      human_ci: pair(m, "human_ci"),
      tolerance: m.nonNegative("tolerance"),
      sim_mean: m.finite("sim_mean"),
      in_range: m.boolean("in_range"),
    })),
  };
}

function parseB4C2(r: Reader): { records: B4Record[] } {
  return {
    records: r.objects("records").map((b) => ({
      variant: b.string("variant"),
      regime: b.string("regime"),
      games_complete: b.integer("games_complete", 0),
      metrics_in_range: b.integer("metrics_in_range", 0, 4),
      pass_outcomes: b.boolean("pass_outcomes"),
      closed_loop: b.boolean("closed_loop"),
      label: b.string("label"),
      counts_as_validated: b.boolean("counts_as_validated"),
    })),
  };
}

function parseB4Rows(r: Reader, key: string): B4Row[] {
  return r.objects(key).map((b) => ({
    governor: b.string("governor"),
    variant: b.string("variant"),
    source: b.oneOf("source", ["c2", "c4"] as const),
    games_complete: b.integer("games_complete", 0),
    games_discarded: b.integer("games_discarded", 0),
    total_surplus_mean: b.nullable("total_surplus_mean", (k) => b.finite(k)),
    total_surplus_sd: b.nullable("total_surplus_sd", (k) => b.finite(k)),
    cli_version: nullableString(b, "cli_version"),
  }));
}

function parseHypotheses(r: Reader): Hypotheses {
  const v = r.object("verdicts");
  const confirmed = r.object("confirmed");
  const rows = r.object("rows");
  const h2b4 = r.has("h2_b4") ? r.object("h2_b4") : null;
  return {
    verdicts: { H1: v.string("H1"), H2: v.string("H2"), H2b_confirmed_losses: v.has("H2b_confirmed_losses") ? v.strings("H2b_confirmed_losses") : [] },
    confirmed: record(confirmed, (o, key) => o.strings(key)),
    rows: record(rows, (o, key) => {
      const row = o.object(key);
      return {
        own_model_gain: row.boolean("own_model_gain"),
        possible_exploitation: row.boolean("possible_exploitation"),
        optimiser_limited: row.boolean("optimiser_limited"),
      };
    }),
    delta: r.positive("delta"),
    h2_b4:
      h2b4 === null
        ? null
        : record(h2b4, (o, key) => {
            const b = o.object(key);
            return {
              verdict: nullableString(b, "verdict"),
              validated_evaluation_population: b.has("validated_evaluation_population") ? b.nullable("validated_evaluation_population", (k) => b.boolean(k)) : null,
              label: nullableString(b, "label"),
            };
          }),
  };
}

function parseE1(r: Reader): E1 {
  const tables = r.object("tables");
  return {
    run_id: r.string("run_id"),
    experiment_id: r.string("experiment_id"),
    tables: record(tables, (o, key) => {
      const t = o.object(key);
      const columns = t.objects("columns", 1).map((c) => ({
        name: c.string("name"),
        unit: c.nullable("unit", (k) => c.string(k)),
        definition: c.string("definition"),
      }));
      const rows = t.array("rows").map((row, i) => {
        if (!Array.isArray(row) || row.length !== columns.length) t.fail(`rows[${i}]`, `${columns.length} values`, row);
        return (row as unknown[]).map((cell, j) => {
          if (cell !== null && typeof cell !== "string" && !(typeof cell === "number" && Number.isFinite(cell))) {
            t.fail(`rows[${i}][${j}]`, "a number, string or null", cell);
          }
          return cell as number | string | null;
        });
      });
      return { columns, rows };
    }),
  };
}

function optional<T>(r: Reader, key: string, read: (sub: Reader) => T): T | null {
  return r.nullable(key, (k) => read(r.object(k)));
}

export function parseCitizensDemo(value: unknown, manifest: CitizensManifest, source = "citizens_demo.json"): CitizensDemo {
  const r = root(source, value, "Model Citizens demo");
  const runId = checkRunId(r, manifest, source);
  const g = r.object("game");
  const game: CitizensGame = {
    players: g.literal("players", PLAYERS),
    rounds: g.integer("rounds", 1, 1000),
    max_pool: g.positive("max_pool"),
    growth_factor: g.positive("growth_factor"),
    pool_floor: g.nonNegative("pool_floor"),
    pool_rule: g.string("pool_rule"),
  };
  const traces = r.objects("traces", 1).map(
    (t): CitizenTrace => ({
      ...parseGameRecord(t, game.rounds),
      governor: t.string("governor"),
      citizens: t.string("citizens"),
      game: t.integer("game", 0),
      seed: t.has("seed") ? t.nullable("seed", (k) => t.integer(k)) : null,
      source: t.has("source") ? t.oneOf("source", ["c2", "c4"] as const) : null,
    }),
  );
  const human = r.objects("human_games").map(
    (h): HumanGame => ({
      ...parseGameRecord(h, game.rounds),
      governor: h.string("governor"),
      mechanism: h.string("mechanism"),
      cohort: h.string("cohort"),
      game_id: h.string("game_id"),
      split: h.oneOf("split", ["training", "heldout"] as const),
    }),
  );
  return {
    schema_version: 1,
    run_id: runId,
    game,
    trace_games: numbers(r, "trace_games"),
    traces,
    human_games: human,
    data_attribution: r.string("data_attribution"),
    crossplay: parseCrossplay(r.object("crossplay")),
    k_tuning: parseKTuning(r.object("k_tuning")),
    distilled: parseDistilled(r.object("distilled")),
    exams: optional(r, "exams", parseExams),
    b4_c2: optional(r, "b4_c2", parseB4C2),
    b4_rows: r.nullable("b4_rows", (k) => parseB4Rows(r, k)),
    hypotheses: optional(r, "hypotheses", parseHypotheses),
    e1: optional(r, "e1", parseE1),
  };
}
