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

## M1 — Physics core (D13: before any viewer) — done 2026-10-18

**Goal:** trustworthy computed trajectories exist before anything is displayed.

- Implement planar normalized CR3BP dynamics in Python; adaptive ODE integration (`solve_ivp`, DOP853) with collision events.
- Independent checks for Jacobi drift, primary placement, cross-frame transforms, event direction and unit conversions (km, days, km/s).
- Reproduce one published reference case (e.g. an L1 Lyapunov orbit) with recorded tolerances.
- Export one versioned manifest + trajectory per `data-contract.md`, status `validated` with recorded checks.

**Done when:** a validated reference artifact exists and its checks are reproducible with one command.

## M2 — Visible public preview with real data — built 2026-10-18 (not yet deployed)

**Goal:** recognizable, beautiful public-facing site showing only computed data.

- Gallery with four investigation cards (one active, remaining `Planned`).
- Gravity Atlas page rendering the M1 artifact: time scrub, play/pause, trajectory picker (when >1), legend, coordinate/unit readout, accessible keyboard operation.
- Schema validation on load; meaningful empty/error states; responsive design; frontend checks and build.
- Document static deployment (Cloudflare Pages or equivalent); deployment requires repository access/credentials and user approval.

**Done when:** a visitor can explore the page from static assets alone, and every displayed trajectory is computed.

## M3 — Free-return feasibility — done 2026-10-18 (GO)

Two validated planar free returns (c1, c2) and a 2¹⁷-sample 3-D Sobol map (243 feasible, 27 of 100 descriptor cells), all from clean commits; see `studies/gravity-atlas/m3-feasibility-design.md` ("M3 results"). Next: M4. Its protocol is drafted in `studies/gravity-atlas/protocol.md` and awaits review and freezing.

**Goal:** a precisely defined free-return-like candidate, or an honest negative result.

- Specify the exact free-return event definition; search for and validate at least one qualifying candidate.
- Export versioned trajectory artifacts and render them in the same viewer.

**Go/no-go:** if no physically valid free-return candidate can be found reproducibly under a reasonable bounded search, publish a negative feasibility note and narrow/revise the scientific task *before* adding MAP-Elites. No synthetic 'successful return'.

## M4 — Comparative Gravity study — done 2026-10-19 (D29)

Frozen protocol run in full (16 seeds, 7 arms); H1 supported against all five baselines through coverage, with less robust individual solutions; the study is closed (D28, D29). A preprint is the owner's call and is not scheduled.

**Goal:** answer a bounded, falsifiable question.

- Freeze the protocol first (`study-protocol` skill).
- Compare multi-start local search (required), a population baseline (recommended, e.g. differential evolution), and MAP-Elites with matched integration budgets.
- Define archive descriptors before evaluation; run multiple independent random seeds.
- Evaluate trajectory feasibility, quality, archive coverage and perturbation robustness on held-out perturbations.
- Publish report with methods, confidence/dispersion, failed cases, validated replays and downloadable configs.

**Go/no-go:** if simple baselines dominate and the archive adds no robustness or utility, publish this result and stop adding complexity. RL is not mandatory.

## Model Citizens (D31) — active

Spec: `studies/model-citizens/README.md`. Code goes in `src/phasespace/citizens/`. Held-out human data are opened only through a guarded accessor after the relevant protocol is frozen.

### S0 — Record, scan and set up

- Record the topic scan (`scouting/2026-10-19-rl-topic-scan.md`), D31, the spec v0 and this roadmap section; retire the Constitution Machine slot.
- Focused `lit-scan` into `studies/model-citizens/literature.md` (Koster 2022/2025 full texts, AI Economist, Lucas critique and structural behavioural models, held-out validation of behavioural simulators, LLM "silicon subjects", robust mechanism design, scoop check).
- Add JAX dependencies without weakening type or lint settings.

**Done when:** the docs above exist and `just verify` passes.

### C1 — Data and game reimplementation (gate G1)

- Download and checksum the public CSV; parse to typed per-game arrays; publish a data-audit run (games per mechanism and cohort, filters, rounds, post-depletion behaviour).
- Pure NumPy game and mechanisms, plus a JAX twin that matches it to 1e-9.

**G1:** every recorded pool transition and Equal / Mixed / Proportional / Interpolating offer is reproduced within print precision; the published Exp 1–4 z-values are reproduced to two decimals. Otherwise stop and reconcile; a documented approximation needs the owner's approval.

### C2 — Citizen models and validation (gate G2)

- Freeze `protocol-validation.md` first (training mechanisms, held-out regimes, mechanism surrogate, validation criterion, seeds and budgets, LLM prompt).
- Fit B1 (GRU clone), B2 (parametric conditional cooperation) and B3 (EWA / Roth–Erev); run B4 (LLM citizens via headless `claude -p`) after a throughput pilot; then evaluate on held-out regimes.

**G2:** at least two model classes, at least one not a GRU, are validated on at least one held-out regime. Otherwise report and ask the owner; all classes failing everywhere is a publishable null.

### C3 — RL planner pipeline (gate G3)

- PPO in JAX with a set-encoder + GRU policy and a 5-share stochastic head; baselines including the re-tuned interpolating k\* per model.
- Correctness on a toy problem and an analytically solvable citizen; determinism for a fixed seed; measured cost per million steps.

**G3:** against B1, the planner's surplus exceeds Interpolating k = 22 (the paper's qualitative in-silico result) within 24 h per training run. Otherwise report and ask the owner (scale down, or backpropagate through the differentiable clone).

### C4 — Main comparison (gate G4)

- Freeze `protocol.md` with owner approval: seeds, cross-play matrix (planners and baselines × validated citizen models, plus LLM citizens), H1 and H2, statistics (common random numbers, bootstrap, Holm), minimum effect size, compute ceiling.
- Runs from clean trees; distil each planner into closed-form rules and compare them across models.

**G4:** all evidence runs carry the frozen config hash and `git_dirty = false`; H1 and H2 are decided as declared, and either outcome is reported.

### C5 — Report and public page

- `studies/model-citizens/report.md` with method, evidence, limitations and reproducibility metadata.
- Story-first public page: replay real human games (CC-BY attribution), choose who the citizens are and who governs, explore the cross-play matrix and the distilled rules.

## Later world: The Last Library (after Model Citizens)

**Goal:** demonstrate that the site is a research lab, not a cislunar microsite.

- Implement the narrowly scoped inherited-memory experiment specified in `studies/last-library/README.md`, after its own literature scan.
- Reuse run/provenance/report *format*, not gravitational rendering or physics APIs.
- Publish an interactive generational lineage visualization and comparison with no-inheritance baseline.

## Later, only after The Last Library

- Algorithm Forge, provided it passes a short feasibility review. (Constitution Machine is retired, superseded by Model Citizens: D31.)
- Community contributions through GitHub issues/PRs and reproducible offline experiments.
- Live backend/worker scheduling only if static results + local CLI become a demonstrable bottleneck.

## MVP exclusions

No shared worker mesh, accounts, user compute, monetization, built-in paper generation, universal plugin SDK, custom general-purpose deep RL framework (a study-local PPO, as in Model Citizens, is not a framework), automated academic novelty claims, or broad migration to Rust.

## Cross-milestone completion checklist

- [ ] Scientific correctness and visualization fidelity are independently tested.
- [ ] Baseline and conditions are public and reproducible.
- [ ] Claims distinguish exploration, feasibility, validated result and scientific conclusion.
- [ ] New functionality follows repository conventions; `just verify` passes.
- [ ] The live site can be built and served without an always-on scientific backend.
