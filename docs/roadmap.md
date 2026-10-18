# Roadmap — Sequential, Research-First

Timings below are *planning ranges*, not deadlines or promises. Complete one vertical slice before expanding. No parallel workstreams are required.

## M0 — Workspace reset (done 2026-10-18)

Hex prototype, Hex docs and the Pi local-LLM configuration archived on branch `archive/policylab-hex`; package renamed to `phasespace`; handoff documents promoted to `docs/` (D10, D11).

## S3 — Gravity literature scan and reframing (done 2026-10-18: D16–D18)

**Goal:** confirm the research question is worth asking before writing solver code.

- Focused nearest-neighbor search around Beeson, Li & Sinha (2024) and quality-diversity methods in trajectory design; record queries and dates.
- Confirm or adjust the robustness angle; choose a target venue (or arXiv only).
- Update `studies/gravity-atlas/README.md` with the reframed question and literature positioning.

**Done when:** the study states its nearest prior work and how it differs.

## M1 — Physics core (D13: before any viewer)

**Goal:** trustworthy computed trajectories exist before anything is displayed.

- Implement planar normalized CR3BP dynamics in Python; adaptive ODE integration (`solve_ivp`, DOP853) with collision events.
- Independent checks for Jacobi drift, primary placement, cross-frame transforms, event direction and unit conversions (km, days, km/s).
- Reproduce one published reference case (e.g. an L1 Lyapunov orbit) with recorded tolerances.
- Export one versioned manifest + trajectory per `data-contract.md`, status `validated` with recorded checks.

**Done when:** a validated reference artifact exists and its checks are reproducible with one command.

## M2 — Visible public preview with real data

**Goal:** recognizable, beautiful public-facing site showing only computed data.

- Gallery with four investigation cards (one active, remaining `Planned`).
- Gravity Atlas page rendering the M1 artifact: time scrub, play/pause, trajectory picker (when >1), legend, coordinate/unit readout, accessible keyboard operation.
- Schema validation on load; meaningful empty/error states; responsive design; frontend checks and build.
- Document static deployment (Cloudflare Pages or equivalent); deployment requires repository access/credentials and user approval.

**Done when:** a visitor can explore the page from static assets alone, and every displayed trajectory is computed.

## M3 — Free-return feasibility

**Goal:** a precisely defined free-return-like candidate, or an honest negative result.

- Specify the exact free-return event definition; search for and validate at least one qualifying candidate.
- Export versioned trajectory artifacts and render them in the same viewer.

**Go/no-go:** if no physically valid free-return candidate can be found reproducibly under a reasonable bounded search, publish a negative feasibility note and narrow/revise the scientific task *before* adding MAP-Elites. No synthetic 'successful return'.

## M4 — Comparative Gravity study

**Goal:** answer a bounded, falsifiable question.

- Freeze the protocol first (`study-protocol` skill).
- Compare multi-start local search (required), a population baseline (recommended, e.g. differential evolution), and MAP-Elites with matched integration budgets.
- Define archive descriptors before evaluation; run multiple independent random seeds.
- Evaluate trajectory feasibility, quality, archive coverage and perturbation robustness on held-out perturbations.
- Publish report with methods, confidence/dispersion, failed cases, validated replays and downloadable configs.

**Go/no-go:** if simple baselines dominate and the archive adds no robustness or utility, publish this result and stop adding complexity. RL is not mandatory.

## M5 — Second world: The Last Library

**Goal:** demonstrate that the site is a research lab, not a cislunar microsite.

- Implement the narrowly scoped inherited-memory experiment specified in `studies/last-library/README.md`, after its own literature scan.
- Reuse run/provenance/report *format*, not gravitational rendering or physics APIs.
- Publish an interactive generational lineage visualization and comparison with no-inheritance baseline.

## Later, only after M5

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
