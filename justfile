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
verify: check test
