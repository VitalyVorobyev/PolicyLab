# Data Contract — Immutable Static Research Runs (v1)

## Objectives

Allow **scientific Python tools** to export results and **static React** to load, validate and inspect them without a server. File layout can change behind a loader, but schema fields and units should be explicit and versioned.

## Directory model (illustrative)

```text
web/public/research/
  catalog.json
  gravity-atlas/
    benchmark-a/
      manifest.json
      trajectory.json
```

Only small, curated public samples belong in the deployable site. Full research files may live under ignored local `runs/`, a release asset, or an object store. Source code and experiment configurations should remain separately versioned.

## Example `manifest.json` (schema v1)

```json
{
  "schema_version": 1,
  "run_id": "gravity-benchmark-a",
  "study_id": "gravity-atlas",
  "experiment_id": "cr3bp-feasibility-v1",
  "status": "computed_unvalidated",
  "created_at": "2026-10-18T00:00:00Z",
  "source": {
    "repository": "VitalyVorobyev/PolicyLab",
    "git_commit": "<actual-commit-sha>",
    "generator": "phasespace.gravity.export",
    "generator_version": "0.1.0"
  },
  "model": {
    "name": "planar-cr3bp",
    "coordinate_frame": "barycentric_synodic",
    "units": "normalized_cr3bp",
    "mu": 0.0121505856,
    "length_scale_km": 384400.0,
    "time_scale_seconds": 375190.0,
    "assumptions": ["circular_primaries", "massless_spacecraft", "planar"]
  },
  "solver": {"name": "DOP853", "rtol": 1e-11, "atol": 1e-12},
  "random_seed": null,
  "artifacts": [
    {"kind": "trajectory", "path": "trajectory.json", "sha256": "<actual-file-sha256>"}
  ],
  "metrics": {},
  "limitations": ["Example schema only; no physical validity is implied."]
}
```

**Important:** this is a *shape example*, not a valid existing record. Angle brackets are placeholders, `created_at` is illustrative, and `time_scale_seconds` is approximate until calculated from declared constants. Do not commit this literal example as published scientific data. Exporters must fill real source SHA, timestamps, scales, checksums and solver settings.

### `trajectory.json` example

```json
{
  "schema_version": 1,
  "run_id": "gravity-benchmark-a",
  "frame": "barycentric_synodic",
  "time_unit": "normalized_cr3bp",
  "state_order": ["t", "x", "y", "vx", "vy"],
  "samples": [
    [0.0, 0.1, 0.2, 0.0, 0.1],
    [0.01, 0.101, 0.2009, 0.02, 0.08]
  ]
}
```

The rows above are **fabricated data-shape examples, not a CR3BP solution**. The UI must never present them as computed scientific results.

## Status and validation meaning

- `illustrative`: invented or drawn for demo; **not physical evidence**.
- `computed_unvalidated`: produced by a computational procedure but independent scientific validation remains incomplete.
- `validated`: physics/model-specific checks passed and their thresholds are recorded. Does **not** imply a new scientific discovery.

A manifest may include `validation: {checks: [{name, observed, threshold, passed}], validated_by, validated_at}` when `validated`. A run must not be labeled validated without such evidence.

## Mandatory invariants

- Trajectory times strictly increase; all values finite; state vector order explicit; units are not guessed.
- For CR3BP, primaries, coordinate conventions, origin, and epoch/model scales are unambiguous.
- `sha256` matches artifact bytes; published files never overwritten in place.
- Dataset provenance includes precise code commit and the actual model/solver configuration.
- Frontend validates unknown versions, nulls, nonfinite values, length mismatches, wrong paths, empty runs and time ordering; fails clearly.
- No user-supplied path traversal or remote cross-origin scripts through manifest fields.
- Derived metrics carry physical units and definition, e.g. `departure_delta_v_km_s`, `return_shell_crossing_time_days`, `lunar_min_distance_km`; *only* include when actually computed.

## Comparison/protocol artifacts (M4)

Store independently versioned experiment definitions containing: predeclared candidate parameterization, feasibility thresholds, objective and descriptor bounds, budget measured in numerical propagations (and wall-clock as secondary), seed list, held-out perturbation distribution, numerical solver settings, and statistical analysis procedure.

Prefer schema validation on producer and consumer sides (Python + TypeScript). Avoid generating a complicated cross-language framework before first two consumers exist.
