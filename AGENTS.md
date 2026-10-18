# Phase Space Agent Instructions

Phase Space is a small public laboratory of computational research studies. Each study has a falsifiable question, credible baselines, validated models, an interactive viewer and a readable report. The quality bar is publishable and literature-aware work.

Active study: Gravity Atlas (planar Earth–Moon CR3BP, trajectory diversity and robustness).

## Before changing code

Read only the documentation relevant to the task (index: `docs/README.md`):

* `docs/vision.md` — intent, quality bar, non-goals
* `docs/roadmap.md` — current milestone and its gates
* `docs/decisions.md` — decision log; check before re-opening a settled question
* `docs/system-design.md` — architecture and layout
* `docs/data-contract.md` — run manifests and artifact schema
* `docs/research-methodology.md` — fair comparisons, reproducibility
* `docs/frontend-design.md` — viewer and site UX
* `docs/studies/<study>/` — study specification, literature scan, protocol

The docs are a working baseline, not ultimate truth. Raise disagreements explicitly; record accepted changes in `docs/decisions.md`.

Inspect existing code before proposing new abstractions.

## Architecture

Static-first data flow:

```text
Python study code (offline, local)          src/phasespace/<study>/
  -> validation + experiment evaluation
  -> immutable run artifacts (manifest + data, schema v1)
  -> curated public samples                 web/public/research/
  -> static React site                      web/
```

Important rules:

* Study domain code is pure and deterministic given its seed; it does not depend on web or I/O layers.
* Each study keeps its own model; no universal `World` / `Agent` / `Environment` interface until two studies demonstrably share it.
* The browser renders and inspects artifacts; it never recomputes authoritative science.
* No backend, database or accounts until static results plus a local CLI are a demonstrated bottleneck.

## Research integrity

* Never fabricate data, results, metrics, citations or screenshots.
* Every citation must be verified (resolvable DOI or arXiv ID, actually fetched). Mark unverified references as such.
* Every headline claim must trace to an artifact with code commit, configuration, seed and checksums.
* Artifact status is one of `illustrative`, `computed_unvalidated`, `validated`; `validated` requires recorded checks with thresholds.
* Freeze protocols (budgets, seeds, metrics, held-out perturbations) before running comparisons. Never tune on held-out data or cherry-pick seeds.
* Null and negative results are publishable; do not add complexity to rescue a narrative.

## Development rules

* Make small, focused changes. Do not redesign unrelated code.
* Prefer the simplest implementation satisfying the current milestone; no abstractions for hypothetical needs.
* Use explicit random seeds in code involving randomness.
* Never silently change experiment hyperparameters.
* Never overwrite experiment artifacts or published data.
* Add tests for new behavior and regression tests for bugs. Numerical tests must check against independent references (analytical cases, invariants, published values), not against the implementation's own output.
* Never modify a test merely because the implementation fails it; first verify the test contradicts the specification.
* Do not weaken linting, typing or tests to make validation pass.
* Do not commit, push, rewrite history or delete branches unless explicitly asked.
* English for code, docs, UI and reports.

## Validation

Use project commands rather than reconstructing tool invocations:

```text
just test
just check
just verify
just format
```

Before considering a code-changing task complete, run `just verify`. If a required check cannot run, report that explicitly.

## Completion report

At the end of an implementation task, report briefly:

* what changed;
* important design decisions;
* validation performed;
* any unresolved issues.
