# Show available project commands.
[default]
help:
    @just --list

# Run Python tests.
test:
    uv run pytest

# Run Python lint checks.
lint:
    uv run ruff check .

# Verify Python formatting without modifying files.
format-check:
    uv run ruff format --check .

# Format Python code.
format:
    uv run ruff check --fix-only .
    uv run ruff format .

# Run strict Python type checking.
typecheck:
    uv run basedpyright

# Run all non-mutating Python validation.
check: lint format-check typecheck

# Run the complete Python validation suite.
verify-python: check test

# Run all validation: Python checks and tests, then the web lint, typecheck, tests and build.
verify: verify-python web-verify

# Export the M1 reference run (Earth-Moon L1 Lyapunov orbit) to a new directory under runs/.
gravity-reference run_id="gravity-l1-lyapunov-reference-v2":
    uv run python -m phasespace.gravity.export --runs-dir runs --run-id {{run_id}}

# M3 (S1): dense 2-D free-return feasibility map (0.5 deg x 2 m/s grid, 90 720 propagations) as a new run under runs/.
gravity-m3-map run_id="gravity-free-return-map-2d-grid-v1" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} map --design grid2d --run-id {{run_id}}

# M3 (S4): 3-D scrambled Sobol free-return map (2^17 samples, explicit seed) as a new run under runs/.
gravity-m3-sobol run_id="gravity-free-return-map-3d-sobol17-v1" seed="20261018" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} map --design sobol3d --seed {{seed}} --log2-samples 17 --run-id {{run_id}}

# M3 (S2+S3): refine, validate and export free-return candidates from a 2-D map run (new runs under runs/).
# The M3 cap is 1000 propagations per start (design proposed 200); the search run records it.
gravity-m3-candidates map_run="gravity-free-return-map-2d-grid-v1" version="v1" cap="1000" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} candidates --map-run {{map_run}} --run-prefix gravity-free-return --version {{version}} --max-propagations-per-start {{cap}}

# M3: fine delta-v probe (theta = 229 and 251 deg, 0.02 m/s, 7002 propagations) resolving feasible-band widths.
gravity-m3-probe run_id="gravity-free-return-dv-probe-v1" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} map --design dvprobe --run-id {{run_id}}

# M3: derive the website's compact, lossless outcome grid from a 2-D map run (new run under runs/).
gravity-m3-web-map map_run="gravity-free-return-map-2d-grid-v1" run_id="gravity-free-return-map-2d-grid-v1-web":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs web-map --map-run {{map_run}} --run-id {{run_id}}

# M4 configuration: print the config hash and key constants (no computation; --json for everything).
gravity-m4-config *args:
    uv run python -m phasespace.gravity.m4 config {{args}}

# M4 smoke run (pre-freeze, never comparative): every method at <= 2^8 propagations (seed 9001) plus cost probes.
gravity-m4-smoke run_id="gravity-m4-smoke-v1" budget="256" workers="1" probe_draws="8":
    uv run python -m phasespace.gravity.m4 --runs-dir runs smoke --run-id {{run_id}} --budget {{budget}} --workers {{workers}} --probe-draws {{probe_draws}}

# M4 frozen-protocol stages: each refuses unless protocol.md says `frozen` and --confirm-frozen is passed.
# Tuning grid (seeds 9101-9102; every grid entry, plus D and CC once): just gravity-m4-tune --confirm-frozen
gravity-m4-tune *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs stage --stage tune {{args}}

# Pilot (seeds 9201-9203) with the tuned configurations: just gravity-m4-pilot --configs MS=0,MBH=1,DE=0,T=2,T2=0 --confirm-frozen
gravity-m4-pilot *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs stage --stage pilot {{args}}

# Main comparison (seeds 1001-1010): just gravity-m4-main --configs ... --confirm-frozen
gravity-m4-main *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs stage --stage main {{args}}

# One M4 run: just gravity-m4-run --stage main --method T --config-index 1 --seed 1011 --confirm-frozen
gravity-m4-run *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs search {{args}}

# M4 coverage ground truth (protocol 5.3 a-c, K1 check): just gravity-m4-groundtruth --confirm-frozen
gravity-m4-groundtruth *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs groundtruth {{args}}

# M4 oracle-seeded continuation reference CC*: just gravity-m4-ccstar --confirm-frozen
gravity-m4-ccstar *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs ccstar {{args}}

# M4 re-check + execution-error tolerance of a run under one injection set and its paired Gates sets
# (T_E | H_E_nominal | H_E_wide): just gravity-m4-robustness <run_id> T_E --confirm-frozen
gravity-m4-robustness run errors *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs robustness --run-id {{run}} --errors {{errors}} {{args}}

# M4 step 4: calibrate B on the D tuning catalogs (T_E, primary Gates set; protocol 9.1), reading stored runs.
# just gravity-m4-calibrate --confirm-frozen   (writes gravity-m4-calibration-v1)
gravity-m4-calibrate *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs calibrate {{args}}

# M4 step 5: configuration selection (protocol 6) at the calibrated B; prints the --configs value.
# just gravity-m4-select --confirm-frozen   (writes gravity-m4-selection-v1)
gravity-m4-select *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs select {{args}}

# M4 step 6: after the pilot and its T_E robustness, the seed count n, K3 and the 7.4 re-projection.
# just gravity-m4-pilot-analysis --confirm-frozen   (writes gravity-m4-pilot-analysis-v1)
gravity-m4-pilot-analysis *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs pilot-analysis {{args}}

# M4 step 8: the final analysis (5.3 d, 5.6 LSODA sample, 10-12) as a new run.
# just gravity-m4-analysis --confirm-frozen --workers 10   (writes gravity-m4-analysis-v1)
gravity-m4-analysis *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs analysis {{args}}

# Install the website's pinned dependencies (npm, from web/package-lock.json).
web-install:
    cd web && npm ci

# Start the website dev server.
web-dev:
    cd web && npm run dev

# Lint, typecheck, test and build the website.
web-verify:
    cd web && npm run verify

# Copy a run from runs/ into web/public/research/ after verifying its checksums (never overwrites).
web-publish-run run destination:
    node web/scripts/publish-run.mjs runs/{{run}} {{destination}}

# M4 public demo run (one seed's catalogs and uncorrected error fans; reads stored runs, re-propagates the shown elites).
# just gravity-m4-demo-export --confirm-frozen --workers 10   (writes gravity-m4-demo-atlas-v1)
gravity-m4-demo-export *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs demo-export {{args}}
