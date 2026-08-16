test:
    uv run pytest

check:
    uv run ruff check .
    uv run pyright
    cd web && bun run lint
    cd web && bun run typecheck

train-smoke:
    uv run python -m trainer.train configs/smoke.yaml

eval-smoke:
    uv run python -m trainer.eval configs/smoke.yaml
