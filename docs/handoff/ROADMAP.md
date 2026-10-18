# Roadmap — Sequential, Research-First

Timings below are *planning ranges*, not deadlines or promises. Complete one vertical slice before expanding. No parallel workstreams are required.

## M0 — Audit & technical decisions (first change)

**Goal:** establish an honest baseline of the actual repository, avoid unneeded rewrites.

- Inspect source tree, existing Hex functionality, build and dependency state, tests, CI and frontend status.
- Document keep/adapt/replace choices in a short PR note.
- Confirm exact frontend commands and the smallest package changes.

**Done when:** the first PR can proceed without guessing about existing code or overwriting functioning subsystems. **Stop if:** branch/repository is not the expected checkout; correct the target before changes.

## M1 — Visible public preview (first vertical slice)

**Goal:** recognizable, beautiful public-facing site, *not* yet a scientific result.

- Gallery with four investigation cards (one active, remaining `Planned`), plus retained Hex entry as appropriate.
- Gravity Atlas detail page with one working trajectory playback fixture; explicit `Illustrative` label if nonphysical.
- Time scrub, play/pause, trajectory picker (when >1), legend, coordinate/unit readout, accessible keyboard operation.
- Basic responsive design, meaningful empty/error states, frontend checks and build.
- Document static deployment (Cloudflare Pages or equivalent); deployment requires repository access/credentials and user approval.

**Done when:** a visitor can explore the page from static assets alone. **Not done:** scientifically correct orbital display unless validated data is available.

## M2 — Physically validated Gravity baseline

**Goal:** remove toy data and produce trustworthy computed examples.

- Implement planar normalized CR3BP dynamics in Python; adaptive ODE integration.
- Independent fixtures/checks for invariant drift, cross-frame transforms and Earth/Moon collision events.
- Reproduce one known periodic/transfer test case before searching free-return solutions.
- Specify exact free-return event definition; search for and validate at least one qualifying candidate.
- Export versioned trajectory artifacts and render them in the same viewer.

**Go/no-go:** if no physically valid free-return candidate can be found reproducibly under a reasonable bounded search, publish a negative feasibility note and narrow/revise the scientific task *before* adding MAP-Elites. No synthetic 'successful return'.

## M3 — Comparative Gravity study

**Goal:** answer a bounded, falsifiable question.

- Compare multi-start local search (required), a population baseline (recommended, e.g. differential evolution), and MAP-Elites with matched integration budgets.
- Define archive descriptors before evaluation; run multiple independent random seeds.
- Evaluate trajectory feasibility, quality, archive coverage and perturbation robustness on held-out perturbations.
- Publish report with methods, confidence/dispersion, failed cases, validated replays and downloadable configs.

**Go/no-go:** if simple baselines dominate and the archive adds no robustness or utility, publish this result and stop adding complexity. RL is not mandatory.

## M4 — Second world: The Last Library

**Goal:** demonstrate that the site is a research lab, not a cislunar microsite.

- Implement the narrowly scoped inherited-memory experiment specified in `research/THE-LAST-LIBRARY.md`.
- Reuse run/provenance/report *format*, not gravitational rendering or physics APIs.
- Publish an interactive generational lineage visualization and comparison with no-inheritance baseline.

## Later, only after M4

- Algorithm Forge, then optional Constitution Machine, provided each passes a short feasibility review.
- Community contributions through GitHub issues/PRs and reproducible offline experiments.
- Live backend/worker scheduling only if static results + local CLI become a demonstrable bottleneck.

## MVP exclusions

No shared worker mesh, accounts, user compute, monetization, built-in paper generation, universal plugin SDK, custom deep RL framework, automated academic novelty claims, or broad migration to Rust.

## Cross-milestone completion checklist

- [ ] Scientific correctness and visualization fidelity are independently tested.
- [ ] Baseline and conditions are public and reproducible.
- [ ] Claims distinguish exploration, feasibility, validated result and scientific conclusion.
- [ ] New functionality follows repository conventions; `just verify` passes.
- [ ] The live site can be built and served without an always-on scientific backend.
