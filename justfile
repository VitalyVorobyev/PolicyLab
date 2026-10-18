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
