# Architecture Decision Log & Deferred Questions

## Accepted directions (2026-10-18)

**D01 — Research first, not a generic RL platform.** World-specific methods and visualizations; shared only run metadata and presentation conventions.

**D02 — Reuse current PolicyLab.** Audit first; gradual extension preferred to total rewrite. *Superseded by D11 after the audit.*

**D03 — Design starts immediately.** Working viewer and site precede full experiment automation. User explicitly values public presentation and immediate insight.

**D04 — Gravity Atlas first.** Numerical CR3BP + trajectory family/robustness comparison. Baselines before MAP-Elites, no compulsory RL.

**D05 — Static-first public launch.** Publish curated immutable results via React static site; local Python generates data. No initial API/server required for public readers.

**D06 — First four new worlds.** Gravity Atlas, The Last Library, Algorithm Forge, Constitution Machine; others not committed to delivery. (Hex: see D11.)

**D07 — Explicit no-go gates.** If physics baseline fails or comparative hypothesis is unsupported, report/revise instead of endless expansion.

**D08 — English source and research materials.** Code, UI, reports, content and docs in English.

## Deferred, not blocked

- Public domain and visual brand (name decided: D10). Renaming the GitHub repository and local folder is deferred until there is something public to show.
- Choose actual static hosting provider (Cloudflare Pages recommended, alternatives acceptable).
- Full production run schema and database: defer after static-file contracts prove usable.
- Exact method hyperparameters, robust perturbation distributions and orbital shell radii: select and freeze during M3 feasibility, *before* M4 comparison.
- MDX/content pipeline: add when first actual research article is written; simple Markdown is sufficient before that.
- Distribution of code artifacts, dataset size policy, and license: check existing LICENSE and do not assume a license.
- Community contributors and volunteer compute: start with GitHub issues/PRs; no distributed scheduler in MVP.

## Decisions after workspace audit (2026-10-18)

**D09 — Purpose and quality bar.** Personal project held to a publishable, literature-aware standard; public, open to collaborators, and part of the author's public profile. Each study names its nearest prior work before its protocol is frozen.

**D10 — Name: Phase Space.** Python package and docs renamed from `policylab` to `phasespace`. GitHub repository and local folder keep their names for now.

**D11 — Hex and Pi archived.** The audit found the Hex code uncommitted, produced during a local-LLM agent experiment, with 14 of 54 tests failing. Hex code, Hex docs, the Pi configuration and the original handoff are preserved on branch `archive/policylab-hex` and removed from `main`. This resolves the earlier conflicts with `docs/architecture.md`, `docs/domain-model.md` and `docs/frontend.md` (no longer on `main`); the term `Experiment` now follows `system-design.md`.

**D12 — Gravity Atlas first, reframed.** Nearest prior work found so far: Beeson, Li & Sinha (2024), diverse low-thrust CR3BP solutions via generative models compared with multi-start. Proposed angle: robustness under held-out perturbations of a MAP-Elites archive vs multi-start vs a population/generative baseline. To be confirmed by a focused literature scan (roadmap S3).

**D13 — Physics core before viewer.** A validated CR3BP integrator and a reproduced reference orbit precede the website. The viewer only ever renders computed data; the `illustrative` status remains in the schema but is not planned for use.

**D14 — Python first, Rust when profiled.** NumPy/SciPy for the science engine; move hot paths to Rust only after measurement shows a need.

**D15 — World order.** After Gravity Atlas: The Last Library (open angle: a byte budget on inherited memory, compared with Cook et al. 2024), then Algorithm Forge (crowded field; teaching/showcase value), then Constitution Machine (conditional).

**D16 — Gravity Atlas question reframed (after S3 scan, `studies/gravity-atlas/literature.md`).** At matched propagation budgets, does a MAP-Elites archive of planar CR3BP free returns contain more solutions that remain feasible under held-out injection errors (with a declared correction budget) and under held-out constraint changes than descriptor-binned multi-start / monotonic basin hopping / differential evolution, a continuation-family catalog, and a dense quasi-random map? The dense map serves both as coverage ground truth and as a budget-matched baseline. A generative (CVAE/diffusion) baseline is deferred.

**D17 — Robustness split in two.** (1) *Execution-error tolerance*: remains feasible after a declared small correction budget under held-out injection errors. (2) *Changed-constraint availability*: the archive still holds a feasible solution when mission constraints change. Integrator-setting sensitivity is a numerical-validity check, not a robustness metric.

**D18 — Venue: arXiv preprint + interactive site.** No conference target; publish when the evidence is ready. Rigor bar unchanged (D09).

**D19 — Web stack (M2).** Vite + React + strict TypeScript + Tailwind, npm with exact pins, a hand-written history router, self-hosted fonts (no third-party requests), Vitest + Playwright checks. `just verify` runs Python and web checks. Static output deployable to Cloudflare Pages with SPA fallback; deployment itself awaits the author's approval.

**D20 — M3 definitions accepted (2026-10-18; see `studies/gravity-atlas/m3-feasibility-design.md`).** Default DE440 / 384,400 km system; 185 km circular parking orbit; single impulse. Feasible free return: closest lunar approach 100–30,000 km altitude, then an inbound crossing of a 121.92 km altitude Earth entry shell, within 10 days (50,000 km shell only as a declared fallback). Search space: 2-D (departure phase, Δv 3.00–3.25 km/s) for M3 feasibility; 3-D (adding impulse angle ±10°) for the M4 comparison, because a 2-D archive is near-trivial. M4 objective: entry flight-path-angle error; Δv is reported, not optimized. Deployment deferred until there is a free-return result.

**D21 — Backward-integration check redefined after it failed (disclosed).** The M3 design's backward check (re-integrate from the return state; raw normalized state error ≤ 1e-7) failed for both converged candidates (observed 8.3e-7). Decomposition showed a 0.13 ms along-track timing shift with about 6 cm of position error, so the raw criterion was about 1,000× stricter than the design's own 0.1 s return-time check. The check is replaced by a decomposed criterion: along-track timing shift ≤ 1 ms, transverse position ≤ 1 m, velocity ≤ 1 mm/s. This change was made *after* observing the failure. The superseded metric stays in every manifest as a non-gating diagnostic.

**D22 — M3 refinement cap raised from 200 to 1000 propagations per start.** Under 200, 0 of 10 starts converged because the 2 m/s grid cannot resolve the narrow target valley; under 1000, all 10 converged to 2 distinct candidates. The total stays within the M3 budget. M4 budgets are set separately by its frozen protocol.

**D23 — M4 protocol open questions answered (2026-10-18; protocol still `draft`).** Budget N = 2^14 propagations per method per seed; continuation-catalog baseline seeded within budget, with an oracle-seeded copy reported only as an unmatched reference; full M3 search box; correction budget B calibrated on dense-map tuning catalogs from {1, 3, 10, 30} m/s to a median survival in [0.2, 0.8], with survival reported for all B; injection-error scales declared as design values (σ = 0.03 % Δv magnitude, 0.02° direction, 0.1 s timing) plus a held-out 2× set; a pre-registered secondary MAP-Elites arm with local refinement (T2) outside the Holm family. Freezing still requires reading the Du & Li 2026 controllable-set papers and an M4 smoke run. The main comparison must not run before the freeze.

## Conditions for revising the plan

A milestone may change only because of observed source state, physical feasibility, research data, runtime limits or a clear user product choice. Document each meaningful change and its reason. Do not enlarge scope simply because implementing another framework sounds elegant.
