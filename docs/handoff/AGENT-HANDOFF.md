# Coding-Agent Handoff — Read This First

You are continuing work in **PolicyLab**. Do not treat these documents as a request to scaffold a new repository by default.

## Repository context actually inspected (2026-10-18)

- `AGENTS.md`: small changes, stable boundaries, explicit seeds, tests; mandatory `just verify`.
- `docs/architecture.md`: Hex-centric domains (`games`, `agents`, `search`, `training`, `experiments`, `server`) and frontend/client separation.
- `docs/domain-model.md`: currently **defines `Experiment` as a single training run**. This handoff proposes a new broader research meaning; avoid silently changing its semantics in Hex code.
- `docs/training.md`: AlphaZero self-play, checkpoints, held-out evaluation, no early distributed processing.
- `docs/frontend.md`: proposes React/TypeScript/Vite/Tailwind/shadcn and SVG, but its older implementation order is Hex-first.
- `pyproject.toml`: Python >=3.12, `uv`, Ruff, pytest, basedpyright; no runtime dependencies listed when inspected.
- `justfile`: `just verify` runs lint, format check, typecheck, tests.
- `README.md`: empty when inspected; do not assume app scaffolding exists.
- `web/package.json`: not found at the inspected default-branch path. **Verify current checkout**; it may have changed.

These are documentation observations, **not** a comprehensive code audit. Inspect files and history in the local checkout before coding.

## Immediate task: implement milestone M0 / M1 only

1. Inspect the current tree, existing frontend, and Hex routes; record what can be reused. Read root `AGENTS.md` and relevant docs before changes.
2. Implement the smallest **public-ready static React vertical slice** for the research gallery and the Gravity Atlas study page using the frontend spec. If a usable React application already exists, evolve it; otherwise create a minimal `web/` application.
3. Establish versioned **static asset loading** for one small trajectory fixture via the `DATA-CONTRACT.md` conventions. The viewer must actually play, pause, scrub, select trajectories, and expose the state and units. An explicitly labeled illustrative fixture is permissible at this stage; do not claim it is physics.
4. Add targeted UI tests, fixture validation, error/empty states and a simple README on running/building the frontend. Keep Hex reachable if already implemented.
5. Run the existing Python validation (`just verify`) and frontend lint/build/tests as applicable. If a check cannot run, state why.
6. Provide a concise implementation report with changed files, verification, screenshot or preview instructions, and remaining items. **Stop**. Do not implement a new CR3BP solver, MAP-Elites, backend, database, authentication or deployment credentials in the same task.

## Scientific follow-on task (separate change)

Read `research/GRAVITY-ATLAS.md`; implement a validated planar CR3BP reference integrator, Jacobi invariance tests, collision/return event logic and real exported samples. Only then pursue a confirmed free-return benchmark and comparisons. Scientific validation must be independent of frontend behavior.

## Engineering rules

- Use English for UI, code and documentation.
- Prefer existing code, packages and conventions; don't rewrite Hex or change all architectural layers.
- Any schema introduced must be small and versioned; validate input boundaries.
- UI must work with static assets and without Python backend running.
- Keep scientific domain code out of React; browser playback is visualization, not authoritative numerical integration.
- Immutable experimental artifacts; content hashes for checksummed files.
- No fabricated scientific data, claims, references, performance or screenshots.
- If two docs conflict, follow the immediate milestone and current inspected code; explain any required migration in the PR instead of silently changing semantics.

## Definition of done for first PR

- Gallery and Gravity Atlas pages render on desktop and phone; route/navigation works.
- Distinct layout, good typography, purposeful color, responsive visualization panel.
- Sample trajectory (real and validated, or explicitly *illustrative*) loads from a file and has working playback controls with numeric readout.
- Selected trajectory, timeline, units, and legend are synchronized; no arbitrary placeholders passed as metrics.
- Missing or corrupt data fails visibly and accessibly.
- `just verify` and frontend checks pass, or each blocker is clearly documented.
- PR description includes what is **not** implemented yet.

## Copy-paste task for Codex / Claude Code

> Read repository `AGENTS.md`, relevant existing `docs/`, and `handoff/README.md`, `handoff/AGENT-HANDOFF.md`, `handoff/FRONTEND-DESIGN.md`, `handoff/DATA-CONTRACT.md`, and `handoff/ROADMAP.md`. Implement **only milestone M0/M1: the static public research gallery plus Gravity Atlas viewer**. Audit the current checkout first, reuse existing code, preserve Hex functionality, and do not create infrastructure beyond that vertical slice. Support a versioned static fixture, real play/pause/scrubbing/selection, provenance labels, errors, responsive polished UI, and verification. Do not invent valid trajectories, implement the scientific solver, deploy, or start the next milestones. Run `just verify` and frontend validation; report results and limitations.
