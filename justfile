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
