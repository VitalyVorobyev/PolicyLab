# System Design — Minimal, Static-First Research Lab

## Architectural objective

Publish a high-quality interactive research site quickly **without** building a production experiment-orchestration service. Local Python creates validated, immutable artifacts; static React consumes them. Reuse PolicyLab's proven Hex components, avoid forcing continuous physical dynamics into turn-based game interfaces.

## Stage-one data flow

```text
Python scientific programs (offline, local)
  -> numerical validation + experiment evaluation
  -> versioned run manifests and result artifacts
  -> curated public data directory / static asset hosting
  -> React / TypeScript research site (static CDN)
        gallery | world-specific visualization | comparisons | articles

existing PolicyLab Hex -> remains usable in its existing architecture
```

Browser interactions with recordings (playback, pan, filter, compare, select) do not require a backend. If a browser-only illustrative scenario is used for M1, it must be visibly identified; the published scientific track switches to Python-generated validated trajectories in M2.

## Separation of concerns

- **Scientific world:** physics/model assumptions, state representation, solver, feasibility, numerical tests, evaluation, domain exports. Independent of web.
- **Study/execution:** run/configuration metadata, content hashes, reproducibility, baselines and aggregate metrics. No implicit global training loop.
- **Presentation:** world-specific React components read normalized artifacts and render controls, plots and text. Never recompute authoritative physics in components.
- **Publication:** human-authored Markdown/MDX study narrative with cited literature and links to exact artifacts/runs; MDX integration can wait until needed.

Do *not* introduce a monolithic polymorphic `World` / `Agent` / `Game` interface. Start with a `RunManifest` schema and view-specific data loaders. Re-evaluate shared abstractions only after two non-game worlds use them.

## Suggested additive layout (adapt after inspecting code)

```text
handoff/                             # these source documents
src/policylab/research/             # optional new Python modules, if appropriate
  gravity/                           # CR3BP, events, optimization, export
  records/                           # minimal manifest validation/helpers
web/                                 # reuse existing frontend if present
  src/features/gallery/
  src/features/gravity/
  src/components/research/
  src/lib/data/
  public/research/gravity-atlas/...   # small versioned sample datasets
research/                            # optional long-form published prose/configs
runs/                                # local research outputs (not auto-committed)
```

Not a mandate to create every directory. Integrate only what is needed for the current milestone.

## Technology choices

- Research: Python >=3.12, NumPy, SciPy (`solve_ivp`), pytest. Quality-diversity library such as `pyribs` may be evaluated in M3 **only if beneficial**. `Z3`, PyTorch or Rust belong to later worlds if justified.
- Web: React, TypeScript, Vite, Tailwind, shadcn/ui, SVG/Canvas for the 2D orbit, Recharts for derived plots. Check installed dependencies before adding any.
- Persistence: immutable JSON manifests and compact trajectory/sample arrays; Parquet optional only when sample volume demands it. No PostgreSQL or SQLite required for the public M1–M3 browsing experience.
- Hosting: Vite static build on Cloudflare Pages or similar. No scientific Python execution on the public host in initial scope.
- CI: preserve `just verify`, add web checks/build; avoid committing large run datasets or proprietary credentials.

## Research run model (beware term collision)

The existing `docs/domain-model.md` calls an **individual training run** an `Experiment`. For new work, use explicit terminology without breaking that contract:

- **Study**: question, datasets/model, baselines, articles and results.
- **Experiment definition**: immutable parameterization and comparison protocol (not necessarily RL).
- **Run**: a particular execution with seed, source commit, solver and artifacts.
- **Artifact**: immutable trace, metric, configuration, image or checkpoint.

Do not rename existing Hex classes until a concrete migration plan exists. The external JSON schema may use `study_id`, `experiment_id` and `run_id` now; adapt code incrementally.

## Versioning and scientific integrity

- `schema_version = 1` with explicit failure on unknown major versions.
- Preserve units, coordinate reference frame, time scale, dynamics parameters, solver tolerances, source commit and actual generation tool.
- Match trajectory frame and reported derived metrics; never interpolate across numerical discontinuities or infer scientific accuracy from pretty animations.
- Immutable published artifacts with SHA-256 checksums; do not silently replace a run after its URL is published.
- Maintain separate statuses: `illustrative`, `computed_unvalidated`, `validated`. Status is **not** a claim of peer-reviewed novelty.

## Deployment strategy

1. Build locally and validate that the static preview works offline/from a local static server (no API dependency).
2. Serve `/research/...` assets with stable links and cache-friendly paths.
3. Configure Cloudflare Pages Git deployment (`web/` as build root if applicable; Vite `dist` output). Confirm project configuration and fallback routing on actual host.
4. Optional custom domain later. No credentials in repository.
5. For publication, use PR-reviewed additions to the curated data directory; don't accept arbitrary uploads to public artifact paths.

## Likely risks & mitigations

| Risk | Mitigation |
|---|---|
| 'Universal platform' scope creep | Keep only one active world until first report |
| Scientific error hidden by UI | Independent ODE invariant/event tests; explicit status in UI |
| Huge static datasets | Downsample for playback, link full artifacts separately |
| Frontend refactor damages Hex | Audit, preserve existing routes, keep domain code independent |
| Overfitting comparisons | Predeclared budget, seeds, held-out perturbations, baseline parity |
| Unclear contribution workflow | Start with GitHub PRs and reproducible local scripts |
