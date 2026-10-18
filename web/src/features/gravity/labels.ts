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
  predicate_failures_run_tight_independent: "Free-return predicates fail, or margins too thin, in any of three integrations",
  tight_tolerance_abs_diff_perilune_altitude_km: "Perilune altitude versus a tighter-tolerance run",
  tight_tolerance_abs_diff_return_time_s: "Return time versus a tighter-tolerance run",
  tight_tolerance_return_position_difference_km: "Shell-crossing position versus a tighter-tolerance run",
  tight_tolerance_abs_diff_entry_speed_km_s: "Entry speed versus a tighter-tolerance run",
  tight_tolerance_abs_diff_entry_flight_path_angle_deg: "Entry flight-path angle versus a tighter-tolerance run",
  tight_tolerance_abs_diff_entry_vacuum_perigee_km: "Entry vacuum perigee versus a tighter-tolerance run",
  independent_integrator_abs_diff_perilune_altitude_km: "Perilune altitude versus an independent integrator",
  independent_integrator_abs_diff_return_time_s: "Return time versus an independent integrator",
  independent_integrator_return_position_difference_km: "Shell-crossing position versus an independent integrator",
  independent_integrator_abs_diff_entry_speed_km_s: "Entry speed versus an independent integrator",
  independent_integrator_abs_diff_entry_flight_path_angle_deg: "Entry flight-path angle versus an independent integrator",
  independent_integrator_abs_diff_entry_vacuum_perigee_km: "Entry vacuum perigee versus an independent integrator",
  jacobi_max_abs_drift_run: "Jacobi constant drift, exported run",
  jacobi_max_abs_drift_tight_tolerance: "Jacobi constant drift, tighter-tolerance run",
  jacobi_max_abs_drift_independent_integrator: "Jacobi constant drift, independent integrator",
  backward_integration_along_track_time_shift_s: "Backward integration: along-track time shift",
  backward_integration_transverse_position_error_m: "Backward integration: transverse position error",
  backward_integration_transverse_velocity_error_mm_s: "Backward integration: transverse velocity error",
  brute_force_abs_diff_perilune_distance_km: "Perilune event versus a brute-force minimum",
  dense_samples_inside_return_shell_or_moon: "Dense samples inside the shell or the Moon (a missed crossing)",
  non_increasing_sample_times: "Non-increasing sample times",
};

/**
 * Groups for the checks table, matched by name prefix in order; unmatched checks go last.
 * `titleWithout` renames a group when none of its checks has the given prefix.
 */
export const CHECK_GROUPS: { title: string; prefixes: string[]; titleWithout?: { prefix: string; title: string } }[] = [
  { title: "Free-return predicates", prefixes: ["predicate_"] },
  { title: "Periodicity and conservation", prefixes: ["periodicity_", "jacobi_"], titleWithout: { prefix: "periodicity_", title: "Conservation" } },
  { title: "Agreement with other propagations, including entry conditions", prefixes: ["tight_tolerance_", "independent_integrator_"] },
  { title: "Backward integration, decomposed into a time shift and transverse errors", prefixes: ["backward_integration_"] },
  { title: "Event detection", prefixes: ["brute_force_", "dense_samples_"] },
  { title: "Comparison with the JPL catalog", prefixes: ["jpl_catalog_"] },
  { title: "Integrity of propagations and samples", prefixes: ["non_"] },
];

/** How a metric is presented in the evidence section, from its key and definition. */
export type MetricRole = "result" | "margin" | "diagnostic" | "superseded";

export function metricRole(key: string, definition: string): MetricRole {
  if (key.startsWith("superseded_")) return "superseded";
  if (key.startsWith("backward_run_") || key.startsWith("backward_tight_")) return "diagnostic";
  if (definition.startsWith("non-gating")) return "margin";
  return "result";
}

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
  departure_phase_deg: "Departure phase θ",
  departure_delta_v_km_s: "TLI Δv",
  impulse_angle_deg: "Impulse angle",
  parking_altitude_km: "Parking orbit altitude",
  return_shell_altitude_km: "Entry-interface shell altitude",
  perilune_band_min_km: "Perilune band, lower edge",
  perilune_band_max_km: "Perilune band, upper edge",
  t_max_days: "Time window",
  perilune_altitude_km: "Perilune altitude",
  lunar_encounter_time_days: "Perilune time",
  lunar_passage_sense: "Lunar passage sense",
  entry_sense: "Entry sense",
  perilune_margin_to_band_min_km: "Perilune margin to the band floor",
  perilune_margin_to_band_max_km: "Perilune margin to the band ceiling",
  return_shell_crossing_time_days: "Entry-interface crossing time",
  return_shell_speed_km_s: "Entry speed",
  return_flight_path_angle_deg: "Entry flight-path angle",
  return_vacuum_perigee_altitude_km: "Entry vacuum perigee",
  earth_periapses_after_encounter: "Earth periapses after the flyby",
  targeting_residual_perilune: "Targeting residual, perilune",
  targeting_residual_perigee: "Targeting residual, perigee",
  targeting_propagations: "Targeting propagations",
  rhs_evaluations: "Right-hand-side evaluations",
  delta_v_margin_below_km_s: "Δv decrease to the nearest infeasible value",
  delta_v_margin_above_km_s: "Δv increase to the nearest infeasible value",
};
