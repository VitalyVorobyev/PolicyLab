/**
 * The Last Library confirmatory run data & sample lineages (`last-library`).
 * Validated in the client after Web Crypto SHA-256 verification before any chart or grid renders.
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

export const LIBRARY_STUDY_ID = "last-library";
const SHA256_RE = /^[0-9a-f]{64}$/;

export interface LibraryProtocol {
  version: string;
  status: string;
  path: string;
  config_sha256: string;
  confirmed_frozen: boolean;
}

export interface LibraryManifest {
  schema_version: 1;
  run_id: string;
  study_id: typeof LIBRARY_STUDY_ID;
  experiment_id: string;
  status: RunStatus;
  created_at: string;
  source: Source;
  protocol: LibraryProtocol;
  environment: Environment;
  random_seed: number | null;
  artifacts: Artifact[];
  metrics: Record<string, Metric>;
  limitations: string[];
  validation: Validation | null;
}

export interface LibraryHypothesisContrast {
  contrast_id: string;
  family: string;
  label: string;
  higher_expected_arm: string;
  lower_expected_arm: string;
  n_lineages: number;
  delta: number;
  mean_diff: number;
  sd_diff: number;
  se_diff: number;
  ci95_low: number;
  ci95_high: number;
  t_stat: number;
  df: number;
  p_raw: number;
  p_holm: number;
  cohens_dz: number;
  alpha: number;
  passed: boolean;
}

export interface LibraryHypothesisFamily {
  family_id: string;
  title: string;
  n_contrasts: number;
  alpha: number;
  n_passed: number;
  all_passed: boolean;
  contrasts: LibraryHypothesisContrast[];
}

export interface LibraryBudgetPoint {
  budget_bits: number;
  stationary_final_mean: number;
  stationary_final_se: number;
  stationary_bits_mean: number;
  change_at_split_mean: number;
  change_at_split_se: number;
  change_final_mean: number;
  change_final_se: number;
}

export interface LibraryBaselineSummary {
  stationary_final_mean: number;
  stationary_final_se: number;
  change_at_split_mean: number;
  change_at_split_se: number;
  change_final_mean: number;
  change_final_se: number;
}

export interface LibraryArmSummary {
  arm: string;
  budget_bits: number;
  rule_changed: boolean;
  n_lineages: number;
  heldout_regret_mean_by_gen: number[];
  heldout_regret_se_by_gen: number[];
  train_regret_mean_by_gen: number[];
  entropy_bits_mean_by_gen: number[];
  map_accuracy_mean_by_gen: number[];
  bits_used_mean_by_gen: number[];
}

export interface LibraryResults {
  schema_version: 1;
  study_id: typeof LIBRARY_STUDY_ID;
  experiment_id: string;
  run_id: string;
  n_lineages: number;
  config_sha256: string;
  seeds: {
    stationary_start: number;
    stationary_end_inclusive: number;
    rule_change_start: number;
    rule_change_end_inclusive: number;
    bootstrap_seed: number;
    bootstrap_resamples: number;
  };
  config: {
    grid_size: number;
    n_terrains: number;
    n_levels: number;
    level_costs: number[];
    noise_beta: number;
    rule_trust: number;
    hedge_discount: number;
    episodes_per_generation: number;
    heldout_layouts_per_gen: number;
    n_generations: number;
    change_generation: number;
    bit_budgets: number[];
  };
  hypotheses: Record<string, LibraryHypothesisFamily>;
  budget_curves: {
    budgets: number[];
    arms: Record<string, LibraryBudgetPoint[]>;
    baselines: Record<string, LibraryBaselineSummary>;
  };
  stationary: Record<string, LibraryArmSummary>;
  rule_change: Record<string, LibraryArmSummary>;
}

export interface LibrarySampleGenMetric {
  generation: number;
  heldout_regret: number;
  train_regret: number;
  bits_used: number;
  posterior_entropy_bits: number;
  map_accuracy: number;
  record_summary: string[];
}

export interface LibrarySampleArmTrace {
  arm: string;
  budget_bits: number;
  planned_paths_by_key_gen: Record<string, [number, number][]>;
  generations: LibrarySampleGenMetric[];
}

export interface LibrarySampleLayout {
  generation: number;
  heldout_preview: {
    start: [number, number];
    goal: [number, number];
    terrains: number[][];
  };
}

export interface LibrarySampleLineage {
  lineage_index: number;
  seed: number;
  rule_changed: boolean;
  rule_initial: number[];
  rule_after: number[];
  layouts_by_key_gen: LibrarySampleLayout[];
  arms: Record<string, LibrarySampleArmTrace>;
}

export interface LibrarySamples {
  schema_version: 1;
  study_id: typeof LIBRARY_STUDY_ID;
  experiment_id: string;
  run_id: string;
  sample_indices: number[];
  sample_budgets: number[];
  key_generations: number[];
  config: {
    grid_size: number;
    level_costs: number[];
    noise_beta: number;
    n_generations: number;
    change_generation: number;
    hedge_discount: number;
  };
  stationary: LibrarySampleLineage[];
  rule_change: LibrarySampleLineage[];
}

export interface LoadedLibraryRun {
  manifest: LibraryManifest;
  results: LibraryResults;
  samples: LibrarySamples;
  manifestUrl: string;
  resultsUrl: string;
  samplesUrl: string;
}

// --- helpers -------------------------------------------------------------------------------------

function numbers(r: Reader, key: string, length?: number): number[] {
  const values = r.array(key);
  if (length !== undefined && values.length !== length) r.fail(key, `an array of ${length} numbers`);
  return values.map((v, i) => {
    if (typeof v !== "number" || !Number.isFinite(v)) r.fail(`${key}[${i}]`, "a finite number", v);
    return v;
  });
}

function integers(r: Reader, key: string, min?: number, max?: number, length?: number): number[] {
  const values = r.array(key);
  if (length !== undefined && values.length !== length) r.fail(key, `an array of ${length} integers`);
  return values.map((v, i) => {
    if (typeof v !== "number" || !Number.isSafeInteger(v) || (min !== undefined && v < min) || (max !== undefined && v > max)) {
      r.fail(`${key}[${i}]`, "an integer in range", v);
    }
    return v;
  });
}

function coords(r: Reader, key: string, gridSize: number): [number, number] {
  const vals = integers(r, key, 0, gridSize - 1, 2);
  return [vals[0] ?? 0, vals[1] ?? 0];
}

function coordPath(values: unknown[], source: string, key: string, gridSize: number): [number, number][] {
  return values.map((pt, i) => {
    if (!Array.isArray(pt) || pt.length !== 2) {
      throw new DataError("invalid", source, `${key}[${i}]: expected [row, col] pair.`);
    }
    const pairVals = pt as unknown[];
    const r = pairVals[0];
    const c = pairVals[1];
    if (
      typeof r !== "number" ||
      typeof c !== "number" ||
      !Number.isSafeInteger(r) ||
      !Number.isSafeInteger(c) ||
      r < 0 ||
      r >= gridSize ||
      c < 0 ||
      c >= gridSize
    ) {
      throw new DataError("invalid", source, `${key}[${i}]: cell coordinate out of bounds.`);
    }
    return [r, c];
  });
}

function terrainGrid(r: Reader, key: string, gridSize: number): number[][] {
  const rows = r.array(key);
  if (rows.length !== gridSize) r.fail(key, `${gridSize} rows`);
  return rows.map((row, ri) => {
    if (!Array.isArray(row) || row.length !== gridSize) r.fail(`${key}[${ri}]`, `${gridSize} integers`, row);
    return (row as unknown[]).map((v, ci) => {
      if (typeof v !== "number" || !Number.isSafeInteger(v) || v < -1 || v > 31) {
        r.fail(`${key}[${ri}][${ci}]`, "a valid terrain index (-1..31)", v);
      }
      return v;
    });
  });
}

function record<T>(r: Reader, read: (sub: Reader, key: string) => T): Record<string, T> {
  const out: Record<string, T> = {};
  for (const key of r.keys()) out[key] = read(r, key);
  return out;
}

// --- manifest ------------------------------------------------------------------------------------

export function parseLibraryManifest(value: unknown, source = "manifest.json"): LibraryManifest {
  const r = root(source, value, "manifest");
  const pr = r.object("protocol");
  const manifest: LibraryManifest = {
    schema_version: 1,
    run_id: r.matching("run_id", RUN_ID, "a run id"),
    study_id: r.literal("study_id", LIBRARY_STUDY_ID),
    experiment_id: r.string("experiment_id"),
    status: r.oneOf("status", RUN_STATUSES),
    created_at: r.matching("created_at", TIMESTAMP, 'a UTC time as "YYYY-MM-DDTHH:MM:SSZ"'),
    source: parseSource(r.object("source")),
    protocol: {
      version: pr.string("version"),
      status: pr.string("status"),
      path: pr.string("path"),
      config_sha256: pr.matching("config_sha256", SHA256_RE, "a 64-char sha256 digest"),
      confirmed_frozen: pr.boolean("confirmed_frozen"),
    },
    environment: parseEnvironment(r.object("environment")),
    random_seed: r.nullable("random_seed", (k) => r.integer(k)),
    artifacts: r.objects("artifacts", 2).map(parseArtifact),
    metrics: parseMetrics(r.object("metrics")),
    limitations: r.strings("limitations", 1),
    validation: r.has("validation") ? parseValidation(r.object("validation")) : null,
  };
  if (manifest.source.git_dirty && !manifest.limitations.some((l) => l.includes("git_dirty"))) {
    throw new DataError("invalid", source, "source.git_dirty is true without disclosure in limitations.");
  }
  if (manifest.status === "validated") {
    const failed = manifest.validation?.checks.filter((c) => !c.passed) ?? null;
    if (failed === null || manifest.validation?.checks.length === 0 || failed.length > 0) {
      throw new DataError("invalid", source, 'status is "validated" without passing checks.');
    }
  }
  const hasResults = manifest.artifacts.some((a) => a.kind === "confirmatory_results");
  const hasSamples = manifest.artifacts.some((a) => a.kind === "sample_lineages");
  if (!hasResults || !hasSamples) {
    throw new DataError("invalid", source, 'artifacts: expected both "confirmatory_results" and "sample_lineages".');
  }
  return manifest;
}

// --- results.json --------------------------------------------------------------------------------

function parseArmSummary(r: Reader, nGen: number): LibraryArmSummary {
  return {
    arm: r.string("arm"),
    budget_bits: r.integer("budget_bits", 0),
    rule_changed: r.boolean("rule_changed"),
    n_lineages: r.integer("n_lineages", 1),
    heldout_regret_mean_by_gen: numbers(r, "heldout_regret_mean_by_gen", nGen),
    heldout_regret_se_by_gen: numbers(r, "heldout_regret_se_by_gen", nGen),
    train_regret_mean_by_gen: numbers(r, "train_regret_mean_by_gen", nGen),
    entropy_bits_mean_by_gen: numbers(r, "entropy_bits_mean_by_gen", nGen),
    map_accuracy_mean_by_gen: numbers(r, "map_accuracy_mean_by_gen", nGen),
    bits_used_mean_by_gen: numbers(r, "bits_used_mean_by_gen", nGen),
  };
}

export function parseLibraryResults(value: unknown, manifest: LibraryManifest, source = "results.json"): LibraryResults {
  const r = root(source, value, "confirmatory results");
  r.literal("study_id", LIBRARY_STUDY_ID);
  const runId = checkRunId(r, manifest, source);
  const cfg = r.object("config");
  const nGen = cfg.integer("n_generations", 2);
  const seeds = r.object("seeds");
  const curves = r.object("budget_curves");
  const curvesArms = curves.object("arms");
  const curvesBaselines = curves.object("baselines");
  const hypReader = r.object("hypotheses");
  const stReader = r.object("stationary");
  const rcReader = r.object("rule_change");

  const hypotheses = record(hypReader, (h, key): LibraryHypothesisFamily => {
    const f = h.object(key);
    const contrasts = f.objects("contrasts", 1).map(
      (c): LibraryHypothesisContrast => ({
        contrast_id: c.string("contrast_id"),
        family: c.string("family"),
        label: c.string("label"),
        higher_expected_arm: c.string("higher_expected_arm"),
        lower_expected_arm: c.string("lower_expected_arm"),
        n_lineages: c.integer("n_lineages", 2),
        delta: c.finite("delta"),
        mean_diff: c.finite("mean_diff"),
        sd_diff: c.finite("sd_diff"),
        se_diff: c.finite("se_diff"),
        ci95_low: c.finite("ci95_low"),
        ci95_high: c.finite("ci95_high"),
        t_stat: c.finite("t_stat"),
        df: c.integer("df", 1),
        p_raw: c.finite("p_raw"),
        p_holm: c.finite("p_holm"),
        cohens_dz: c.finite("cohens_dz"),
        alpha: c.finite("alpha"),
        passed: c.boolean("passed"),
      }),
    );
    return {
      family_id: f.string("family_id"),
      title: f.string("title"),
      n_contrasts: f.integer("n_contrasts", 1),
      alpha: f.finite("alpha"),
      n_passed: f.integer("n_passed", 0),
      all_passed: f.boolean("all_passed"),
      contrasts,
    };
  });

  return {
    schema_version: 1,
    study_id: LIBRARY_STUDY_ID,
    experiment_id: r.string("experiment_id"),
    run_id: runId,
    n_lineages: r.integer("n_lineages", 2),
    config_sha256: r.matching("config_sha256", SHA256_RE, "a 64-char sha256 digest"),
    seeds: {
      stationary_start: seeds.integer("stationary_start"),
      stationary_end_inclusive: seeds.integer("stationary_end_inclusive"),
      rule_change_start: seeds.integer("rule_change_start"),
      rule_change_end_inclusive: seeds.integer("rule_change_end_inclusive"),
      bootstrap_seed: seeds.integer("bootstrap_seed"),
      bootstrap_resamples: seeds.integer("bootstrap_resamples", 100),
    },
    config: {
      grid_size: cfg.integer("grid_size", 3),
      n_terrains: cfg.integer("n_terrains", 2),
      n_levels: cfg.integer("n_levels", 2),
      level_costs: numbers(cfg, "level_costs"),
      noise_beta: cfg.positive("noise_beta"),
      rule_trust: cfg.finite("rule_trust"),
      hedge_discount: cfg.finite("hedge_discount"),
      episodes_per_generation: cfg.integer("episodes_per_generation", 1),
      heldout_layouts_per_gen: cfg.integer("heldout_layouts_per_gen", 1),
      n_generations: nGen,
      change_generation: cfg.integer("change_generation", 1, nGen - 1),
      bit_budgets: integers(cfg, "bit_budgets", 0),
    },
    hypotheses,
    budget_curves: {
      budgets: integers(curves, "budgets", 0),
      arms: record(curvesArms, (a, key) =>
        a.objects(key, 1).map(
          (p): LibraryBudgetPoint => ({
            budget_bits: p.integer("budget_bits", 0),
            stationary_final_mean: p.finite("stationary_final_mean"),
            stationary_final_se: p.finite("stationary_final_se"),
            stationary_bits_mean: p.finite("stationary_bits_mean"),
            change_at_split_mean: p.finite("change_at_split_mean"),
            change_at_split_se: p.finite("change_at_split_se"),
            change_final_mean: p.finite("change_final_mean"),
            change_final_se: p.finite("change_final_se"),
          }),
        ),
      ),
      baselines: record(curvesBaselines, (b, key): LibraryBaselineSummary => {
        const sub = b.object(key);
        return {
          stationary_final_mean: sub.finite("stationary_final_mean"),
          stationary_final_se: sub.finite("stationary_final_se"),
          change_at_split_mean: sub.finite("change_at_split_mean"),
          change_at_split_se: sub.finite("change_at_split_se"),
          change_final_mean: sub.finite("change_final_mean"),
          change_final_se: sub.finite("change_final_se"),
        };
      }),
    },
    stationary: record(stReader, (s, key) => parseArmSummary(s.object(key), nGen)),
    rule_change: record(rcReader, (s, key) => parseArmSummary(s.object(key), nGen)),
  };
}

// --- samples.json --------------------------------------------------------------------------------

function parseSampleLineages(
  items: Reader[],
  gridSize: number,
  nGen: number,
  source: string,
): LibrarySampleLineage[] {
  return items.map((item) => {
    const layouts = item.objects("layouts_by_key_gen", 1).map((l): LibrarySampleLayout => {
      const preview = l.object("heldout_preview");
      return {
        generation: l.integer("generation", 0, nGen - 1),
        heldout_preview: {
          start: coords(preview, "start", gridSize),
          goal: coords(preview, "goal", gridSize),
          terrains: terrainGrid(preview, "terrains", gridSize),
        },
      };
    });

    const armsObj = item.object("arms");
    const arms = record(armsObj, (a, armKey): LibrarySampleArmTrace => {
      const trace = a.object(armKey);
      const pathsReader = trace.object("planned_paths_by_key_gen");
      const plannedPaths: Record<string, [number, number][]> = {};
      for (const genKey of pathsReader.keys()) {
        plannedPaths[genKey] = coordPath(pathsReader.array(genKey, 2), source, `${armKey}.${genKey}`, gridSize);
      }
      const generations = trace.objects("generations", nGen).map(
        (g): LibrarySampleGenMetric => ({
          generation: g.integer("generation", 0, nGen - 1),
          heldout_regret: g.finite("heldout_regret"),
          train_regret: g.finite("train_regret"),
          bits_used: g.integer("bits_used", 0),
          posterior_entropy_bits: g.finite("posterior_entropy_bits"),
          map_accuracy: g.finite("map_accuracy"),
          record_summary: g.strings("record_summary"),
        }),
      );
      return {
        arm: trace.string("arm"),
        budget_bits: trace.integer("budget_bits", 0),
        planned_paths_by_key_gen: plannedPaths,
        generations,
      };
    });

    return {
      lineage_index: item.integer("lineage_index", 0),
      seed: item.integer("seed"),
      rule_changed: item.boolean("rule_changed"),
      rule_initial: integers(item, "rule_initial", 0),
      rule_after: integers(item, "rule_after", 0),
      layouts_by_key_gen: layouts,
      arms,
    };
  });
}

export function parseLibrarySamples(value: unknown, manifest: LibraryManifest, source = "samples.json"): LibrarySamples {
  const r = root(source, value, "sample lineages");
  r.literal("study_id", LIBRARY_STUDY_ID);
  const runId = checkRunId(r, manifest, source);
  const cfg = r.object("config");
  const gridSize = cfg.integer("grid_size", 3);
  const nGen = cfg.integer("n_generations", 2);

  return {
    schema_version: 1,
    study_id: LIBRARY_STUDY_ID,
    experiment_id: r.string("experiment_id"),
    run_id: runId,
    sample_indices: integers(r, "sample_indices", 0),
    sample_budgets: integers(r, "sample_budgets", 0),
    key_generations: integers(r, "key_generations", 0, nGen - 1),
    config: {
      grid_size: gridSize,
      level_costs: numbers(cfg, "level_costs"),
      noise_beta: cfg.positive("noise_beta"),
      n_generations: nGen,
      change_generation: cfg.integer("change_generation", 1, nGen - 1),
      hedge_discount: cfg.finite("hedge_discount"),
    },
    stationary: parseSampleLineages(r.objects("stationary", 1), gridSize, nGen, source),
    rule_change: parseSampleLineages(r.objects("rule_change", 1), gridSize, nGen, source),
  };
}
