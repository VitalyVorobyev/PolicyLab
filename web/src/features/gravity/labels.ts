import type { ComparisonRole, LengthScaleConvention } from "../../lib/data/types";

/** Plain-language names for the contract's length-scale conventions (exhaustive over the enum). */
export const CONVENTION_LABEL: Record<LengthScaleConvention, string> = {
  nssdca_lunar_semimajor_axis: "NSSDCA lunar semi-major axis (384,400 km)",
  jpl_periodic_orbits_catalog: "JPL three-body periodic-orbit catalog length unit",
};

export const COMPARISON_ROLE_LABEL: Record<ComparisonRole, string> = {
  tight_tolerance: "Tight-tolerance reference",
  independent_integrator: "Independent integrator",
};

/** Plain-language names for known validation checks; unknown checks fall back to their identifier. */
export const CHECK_LABELS: Record<string, string> = {
  periodicity_max_abs_state_error_one_period: "State error after one full period",
  jacobi_max_abs_drift_exported_samples: "Jacobi constant drift across the exported samples",
  jacobi_max_abs_drift_3_periods: "Jacobi constant drift over three periods",
  tight_tolerance_max_abs_state_difference_samples: "Difference from a tighter-tolerance propagation",
  independent_integrator_max_abs_state_difference_samples: "Difference from an independent integrator",
  jpl_catalog_abs_diff_vy0: "Initial vy versus the JPL catalog",
  jpl_catalog_abs_diff_period: "Period versus the JPL catalog",
  jpl_catalog_abs_diff_jacobi: "Jacobi constant versus the JPL catalog",
  jpl_catalog_rel_diff_stability_index: "Stability index (monodromy) versus the JPL catalog",
  non_horizon_terminations_or_collision_events: "Early terminations or collision events",
  non_finite_sample_values: "Non-finite sample values",
  non_increasing_sample_times: "Non-increasing sample times",
};

/** Groups for the checks table, matched by name prefix in order; unmatched checks go last. */
export const CHECK_GROUPS: { title: string; prefixes: string[] }[] = [
  { title: "Periodicity and conservation", prefixes: ["periodicity_", "jacobi_"] },
  { title: "Agreement with other propagations", prefixes: ["tight_tolerance_", "independent_integrator_"] },
  { title: "Comparison with the JPL catalog", prefixes: ["jpl_catalog_"] },
  { title: "Integrity of propagations and samples", prefixes: ["non_"] },
];

/** Short labels for known metrics; the manifest's definition is always shown alongside. */
export const METRIC_LABELS: Record<string, string> = {
  period_normalized: "Period",
  period_days: "Period",
  jacobi_constant: "Jacobi constant",
  initial_x_km: "Initial x",
  initial_vy_km_s: "Initial vy",
  x_extent_km: "Extent in x",
  y_extent_km: "Extent in y",
  min_earth_distance_km: "Closest approach to Earth",
  min_moon_distance_km: "Closest approach to the Moon",
  sample_count: "Samples",
  sample_step_normalized: "Sample step",
  sample_step_hours: "Sample step",
  monodromy_lambda_max: "Largest monodromy eigenvalue",
  monodromy_stability_index: "Stability index (computed)",
  jpl_stability_index: "Stability index (JPL catalog)",
};
