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

**D24 — Freeze gate and correction-execution errors (2026-10-18, before any M4 data).** (1) The protocol may be frozen with Du & Li, AST 179:113609 unread (no open-access copy found). Everything read in the controllable-set lineage (`studies/gravity-atlas/literature.md` §9–§11) characterizes *existence* of feasible returns under nominal dynamics. The report must not claim that robustness of free-return sets is unstudied, only that budget-matched catalog-member robustness across generators was not found; read 113609 before publication if it becomes available. (2) The midcourse correction is no longer applied exactly: it gets Gates-model execution errors with the parameters used by Burnett, Boone & Michelotti (arXiv:2607.03424, 2026) for a planar CR3BP free return. Primary: σs = 5×10⁻³ (proportional), σr = 10 mm/s (fixed, along burn), σp = 3×10⁻⁴ rad (pointing), σa = 0.9 mm/s (fixed, transverse; in-plane only in the planar model). Held-out stress set: σs = 3×10⁻² (their pessimistic value). The departure-burn error scales from D23 are unchanged.

**D25 — Survival definition and compute (2026-10-18, before any M4 data).** Execution-error survival means the trajectory is feasible after the executed correction: commanded |Δv| ≤ B, with Gates execution errors applied. A draw that is feasible without correction counts as surviving at every B. Convergence of the correction solver is recorded only as a diagnostic. Reason: the M4 smoke run on c1 showed that a convergence-based definition makes survival flat and non-monotone in B for definitional reasons. Remaining unspecified implementation details are fixed in the protocol using the implementation's working choices, listed there. Compute is accepted at about 4.3 M propagation-equivalents (≈ 4.9–7.9 h), with the pre-declared reductions if a measured projection exceeds 8 h.

**D26 — M4 protocol v1 frozen (2026-10-18, owner's explicit approval).** Frozen tree: commit `90c57756d809d1bee0c238e7954c43fe147b055e`; config hash `4cfc119543dd0c53d95750d24561badf1e440aea9170db76ffd165553bf3415a`. Confirmed at freeze: (1) the Gates stress set shares the primary set's normal draws (`default_rng((7401, injection seed))`), so the two differ only through σs; (2) Zhu et al., AST 170:111492 stays unread under the D24 claim restriction; (3) Storn & Price (1997) and Leary (2000) are cited from Crossref-verified metadata. Pre-freeze amendment in the same tree: the compute ceiling is 7.8 M PE (8 h at the 3.7 ms/PE reference rate) rather than measured wall time, because the owner chose to run while the machine carries other load; wall time is logged but never triggers a reduction. Every evidence manifest must carry this config hash, `protocol_status = "frozen"` and `git_dirty = false`.

**D27 — Stray test-generated main run quarantined (disclosure).** When the freeze landed, a pre-freeze test that expected the guard to refuse instead executed a main-stage MS search (config 0, seed 1001). It wrote `runs/gravity-m4-main-ms-c0-s1001-v1` at 2026-10-18T15:44:07Z from a dirty tree (`git_dirty = true`, commit 90c5775). The run contains a search catalog only: no robustness or held-out evaluation, and nobody inspected its contents. It is not evidence (dirty tree) and it blocks the run id that step 7 needs. It was moved unread to `runs/_quarantine/`. The tests now exercise the guard against temporary protocol files and temporary run directories (commit bfc1d14). The legitimate seed-1001 MS run will be produced by the protocol's step 7 under the same id.

**D28 — Gravity Atlas scope and the K3 branch (2026-10-18, owner's decision, before the pilot analysis was seen).**
- **Scope.** The owner questioned the study's practical value. Agreed: Gravity Atlas becomes a validated physics demo plus one complete, honestly reported method comparison, and the study then ends. The sensible end point is the frozen protocol's step 8 analysis, run overnight. The lab's next study is chosen from a cross-field opportunity scan.
- **Calibration result.** Step 4 calibration found no B in [0.2, 0.8]. Median survival was 0.9375 at 1, 3 and 10 m/s and 1.0 at 30 m/s. B = 1 m/s by the fallback rule.
- **If K3 passes,** step 7 and step 8 run as frozen.
- **If K3 fires,** the owner chose, before seeing the pilot analysis, a v2 amendment with these rules:
  - execution-error survival (P1) is reported as a descriptive result only, not a confirmatory endpoint;
  - coverage and changed-constraint availability (P2) are the primary endpoints;
  - nothing else changes: seeds, budgets, configurations, the held-out sets and the statistics procedure stay as frozen;
  - the amendment is logged before any main-run data exists.
- **Outcome.** The step 6 pilot analysis (`gravity-m4-pilot-analysis-v1`, clean at c0429bc) found that **K3 is not triggered**, so the protocol runs as frozen and the v2 branch is unused.
  - Pilot median survivals at B = 1 m/s (T_E, primary set): D 0.94, MS 0.78, CC 0.72, DE 0.69, T 0.59, T2 0.59, MBH 0.125. These are pilot values, not evidence.
  - Main seeds: n = 16 (seeds 1001–1016), clamped from 275.
  - The pre-declared normal-approximation power at n = 16 for the smallest declared effect is 0.027. This low power is disclosed; contrasts near the declared minimum effect cannot be resolved.
  - The projection is 5.14 M PE, with no reductions needed; K4 is not triggered.

**D29 — Gravity Atlas M4 result; the study closes (2026-10-19).**
- **Evidence.** `gravity-m4-analysis-v1` (clean, commit c0429bc, frozen config hash). It covers:
  - 112 main runs: 7 arms × seeds 1001–1016;
  - CC\*;
  - 226 held-out robustness runs, H_E-nominal and H_E-wide for every catalog.
- **H1 is supported against all five baselines.** All ten primary contrasts favour T, with Holm-adjusted p ≤ 0.041. The weakest contrast is T − DE on P1: +4.7, 95 % CI [1.1, 8.9].
  - P1, the expected number of surviving solutions at B = 1 m/s: T 20.9, DE 16.2, CC 14.2, D 12.2, MS 1.9, MBH 0.24.
  - P2: T 0.91, CC 0.75, DE 0.68, MS 0.38, D 0.37, MBH 0.09.
- **Interpretation, required alongside any headline.** T wins through coverage. It fills 99.6 % of the 32 reachable 10 × 10 cells, against D 44 %, CC 56 % and DE 68 %. Its individual solutions are *less* robust:
  - mean per-solution survival is T 0.66 vs D 0.88 (paired difference −0.22, CI [−0.24, −0.19]);
  - H2 is weakly supported: Spearman ρ(s_i, selection margin) = −0.089, p = 0.049, over 507 elites.
  - The defensible claim is "a MAP-Elites catalog *contains* more solutions that survive held-out errors". The claim "MAP-Elites *finds* more robust solutions" is not supported.
- **The correction is nearly inert at the calibrated B.** For T, P1 is 20.5 at B = 0 and 20.9 at B = 1 m/s. The Gates stress set changes P1 by at most 0.004 for any arm. So P1 mainly measures uncorrected injection-error tolerance; the s(B) curves to 30 m/s are reported.
- **Stop rules.** K2, K5, K6 and K7 are not triggered, and the DOP853 re-check found 0 failures.
- **LSODA sample.** The 5 % LSODA sample had 4 of 97 elites (4.1 %) fail the full M3 S3 validation. This is reported beside K2, which is defined on the re-check, and must be disclosed.
  - Two are Jacobi-drift threshold failures, one D and one T elite, both in cell (7, 5).
  - Two are T2 elites in cell (8, 9) whose outcome changes under tight tolerance and under LSODA.
- **Other findings.**
  - DE is unstable across seeds: P1 SD 8.5, worst seed 0.
  - T2, which adds local refinement, does not beat T: P1 −0.76, CI [−1.2, −0.31].
  - MS and MBH concentrate on very few cells, with means of 2.6 and 0.5 elites.
- **Execution disclosure.**
  - `stage --stage main` covers only seeds 1001–1010. The extension seeds 1011–1016 were run with per-seed `search` calls, which use the same `_search_task` code at the same commit.
  - The final batch was stopped and resumed twice. Runs are written atomically, so the only work lost was runs in flight, which were redone from scratch; no partial run directories remained.
  - Robustness runs were executed in parallel across catalogs. Each run is serial and deterministic.
- **Close.** As agreed in D28, Gravity Atlas ends here as a validated physics demo plus this comparison. A preprint (M5) is the owner's call and is not scheduled.

### D30. Gravity Atlas public page is an interactive demonstration

- **Decision.** `/gravity` is a story-first demo: the orbit view with the selected free return and its 50 launch-error variants, the 10 × 10 atlas, and a switch over the seven methods, with short problem, method and finding text. Provenance, run ids, checksums, status badges, milestone history and the L1 / feasibility-map / Δv-probe material no longer appear there; the technical record stays in the docs and the repository.
- **Data.** The page reads only `gravity-m4-demo-atlas-v1` (kinds `atlas` and `error_fan`, added to the website validator). The browser draws stored polylines and interpolates positions between vertices; it recomputes nothing. Scores are the 16-seed summaries; the atlas shows seed 1001.
- **Catalog.** Study status `complete` added for finished studies; Gravity Atlas is `complete`. The earlier M1–M3 run directories stay published because the validator tests use them as real-data fixtures.

**D31 — Next study: Model Citizens (2026-10-20, owner's choice).**
- **Brief.** On 2026-10-19 the owner rejected the 2026-10-18 v2 shortlist (fisheries, heat pumps / EVs, insulin, cubature, RealPage, TD vs ANCCR, information engines, RL vs optimal control): RL was peripheral and the fields did not match his interests. New brief: RL central; social behaviour, political-system simulation or algorithm discovery; a real research context anchored to a dataset or disputed regularity; a visualisable result. The search that followed (six literature agents plus a spec extraction) is recorded in [`scouting/2026-10-19-rl-topic-scan.md`](scouting/2026-10-19-rl-topic-scan.md).
- **Question.** Does the mechanism an RL policy designer discovers depend on which *validated* model of people it is trained against? This is an empirical Lucas critique of AI policy design. Anchor: Koster et al. 2022 (Nature Human Behaviour; arXiv 2201.11441) and Koster et al. 2025 (Nature Communications 16:2824, doi 10.1038/s41467-025-58043-7), whose public trial-level human data (CC-BY 4.0) cover baseline, interpolating and RL-designed mechanisms. Behavioural models are fitted on baseline-mechanism play only and validated on human play under mechanisms they never saw; PPO planners are then trained against each validated model and cross-evaluated. Spec: [`studies/model-citizens/README.md`](studies/model-citizens/README.md).
- **RL is central.** The designer is a PPO planner in JAX, as the original designer was a deep-RL agent.
- **LLM citizens.** An LLM-citizen model class (B4) runs through headless `claude -p` on the owner's plan, with no API key. It is a validation and evaluation population only, not a PPO training environment (PPO needs millions of episodes). Model id, prompts and raw responses are archived; sampling cannot be seeded, which is a recorded limitation.
- **Constitution Machine retired.** Model Citizens supersedes the Constitution Machine v0 spec, which asked a close question (do adaptive resource rules survive a change of the governed population?) with synthetic agents only. The slot is retired; its README is kept as history. The Last Library and Algorithm Forge stay planned, after Model Citizens.
- **ChatGPT report held.** The owner-provided report [`chatgpt-deep-research.md`](chatgpt-deep-research.md) was audited. Its #1 (local memory vs the information bound) and #2 (causal emergence under memory completion) are on HOLD: the first is trivially true by time-averaging and its resource claim conflicts with Govern & ten Wolde 2014; the second is largely covered by Rosas et al. 2024 and Geiger & Wu 2016, with a definitional residue. Its #3 (dryland patterning) is not pursued.
- **Gates** (detail in the spec and `roadmap.md`):
  - **G1, data and game:** every recorded pool transition and baseline offer is reproduced from the CSV, and the published Exp 1–4 statistics are reproduced; otherwise stop and reconcile.
  - **G2, citizen models:** at least two model classes, at least one not a GRU, are validated on at least one held-out regime under a validation protocol frozen before held-out data are opened; otherwise report and ask the owner. All classes failing everywhere is a publishable null.
  - **G3, planner pipeline:** against the GRU clone, PPO reproduces the paper's qualitative in-silico result (planner surplus above Interpolating k = 22) within 24 h per training run; otherwise report and ask the owner.
  - **G4, main comparison:** runs only under a frozen, owner-approved `protocol.md`, from clean trees, within its compute ceiling; H1 and H2 are decided as pre-declared and either outcome is reported.

## Conditions for revising the plan

A milestone may change only because of observed source state, physical feasibility, research data, runtime limits or a clear user product choice. Document each meaningful change and its reason. Do not enlarge scope simply because implementing another framework sounds elegant.

**D32 — Model Citizens design amendments from the literature scan (2026-10-20, before any held-out data).** These follow `studies/model-citizens/literature.md`.
1. **Claim restriction.** The prior state is: Koster 2022 validated clones on held-out humans and compared a clone-trained mechanism with a rational-learner-trained one; Tacchetti et al. 2025 framed the Lucas critique. We claim only the empirical multi-model test.
2. **Negative control.** A payoff-maximising "rational" citizen (B0) is added. The validation check must reject it; if it does not, the check is too weak.
3. **Fallback model.** A hybrid of conditional cooperation and EWA is kept as a pre-declared fallback, in case B2 or B3 alone fail validation.
4. **Extrapolation check.** C4 pre-declares a check of whether planners drive play into regions with no training data. The diagnostic is the density of the visited (offer share, pool) states under the training data.
5. **LLM citizens (B4).** Two variants are pre-registered:
   - B4-plain: neutral rules prompt;
   - B4-calibrated: persona or type parameters fitted on baseline games only, after Manning & Horton 2025 and Jackson et al. 2026.

   A memorisation probe for the 2025 paper is logged before any held-out run.

**D33 — Model Citizens G1 split, data-derived game rules and the simulated offer rule (2026-10-20, owner-approved split).** These follow the C1 data audit (`phasespace.citizens.validation`, run `citizens-c1-data-audit-v1-dev2`). Held-out games were touched only through arithmetic-identity residuals and counts.
1. **G1 is split (the owner approved it on 2026-10-20).**
   - **G1a, now:** every training-population check (human Equal / Mixed / Proportional games and all BC games); mechanism-offer reproduction, including the held-out Interpolating rows as an identity; and the four training-only Exp 1 z-values (4.58, 5.28, 5.89, 4.76).
   - **G1b, immediately after `protocol-validation.md` is frozen:** reproduce the published z-values that involve held-out mechanisms, and root-cause the held-out pool anomaly.
   - A C1 audit run is `validated` for the G1a scope only (the Model Citizens addendum in `data-contract.md`). G1 as a whole is not passed until G1b.
2. **Deferred anomaly, bounded and not diagnosed.**
   - In 5 RL Agent (M1) Exp 1 games, 23 held-out pool rows sit about 0.01 coins below the full rule. No other held-out mechanism or cohort fails.
   - Recorded bound: only RL Agent (M1) Exp 1, at most 23 rows in at most 5 games, |residual| ≤ 0.0100 + 1e-4 coins.
   - The strict held-out pool identity is an `xfail(strict=True)` test. A separate regression test enforces the bound. Nothing else about those rows is inspected before the freeze. The earlier "0.01 deduction" fit count was removed as premature diagnosis.
3. **Game rules established from the data (training rows, except where noted).**
   - **Pool floor:** R' = max(0.01, min(200, R − Σe + 1.4Σc)). The floor is not in the paper; every recorded pool is ≥ 0.01.
   - **Withheld amount:** deployed baseline offers are round(shareᵢ·(R − 0.001), 4) in float32. Withholding 0, 0.0005, 0.0015 or 0.002 does not reproduce them (as for k).
   - **Legacy launch_ids:** a frozen list of 7 early Exp 1 games (3 Equal, 4 Proportional) drops the remainder R − Σe: R' = max(0.01, min(200, 1.4Σc)), in every row. Each of the 7 needs this rule, and every other game follows the full rule. Rules are assigned per game, never chosen per row. No held-out game carries a listed id.
   - **Σc = 0:** this occurs only with the pool at the 0.01 floor, and the offer is then an equal split.
   - **Interpolating weight:** quantised as w = round((R/200)^22, 1). The unquantised weight, floor rounding, k = 21 or 23, and k = e^3.1 do not reproduce the BC offers. The quantised weight also reproduces the held-out Interpolating offers as an identity.
   - **Post-depletion rows:** these are kept. Training games record pool 0.01, offers 0.0022 and c = 0 until the game ends. The notebook's metrics include these rows, and so do ours.
   - **`distribute_pool_ratio`:** unidentifiable. It is 1.0 in every human training row and empty in BC rows; the withholding is absolute, not a ratio.
4. **Simulated offer rule.** C2–C4 simulate the *deployed* rule, so simulated games face exactly the offers humans experienced: 0.001 withheld, 4-decimal rounding and float32 shares as reverse-engineered. It is the default (`mechanisms.baseline_offers`, `rule="deployed"`). The paper's idealised formula stays available as `rule="paper"` / `paper_offers`; the two differ by at most about 0.0011 coins per player. Simulations use the full pool rule with the 0.01 floor.
5. **JAX twin moves to C3.** The NumPy reference is the C1 deliverable. The JAX twin, and its 1e-9 agreement check, are built with the PPO pipeline in C3, where they are first needed.

**D34 — Model Citizens C2 validation protocol frozen (2026-10-20, owner-approved).** `studies/model-citizens/protocol-validation.md` v1 is frozen. The freeze commit is `589956d6f289d94041b089dd77db06cffe25e23f`; the frozen code lives at `f8eb3b5cf1414b2d477c6ff8a608f29e0c728154`; `config_sha256` = `75ed1f96fc76ef76ee466c51c7fc553566fe8ed39164ac8465864ad539ac16ee`. Held-out human rows may now be read, only through the post-freeze stages of `phasespace.citizens.heldout`. The owner's decisions, recorded on 2026-10-20:
1. **Freeze v1 now.**
2. **G1b tolerance as drafted:** |Δz| ≤ 0.005, or ≤ 0.05 for the value printed as 2.9.
3. **Surrogate acceptance uses the gating experiments only (pre-freeze amendment A34).** Acceptance on held-out human rows is decided by M1 Exp 1 and Exp 2, and by M2 Exp 3, the only experiment of R-M2. M2 Exp 4 numbers are computed and reported as a diagnostic.
4. **Calibration of record.** The pre-freeze B4 calibration run `citizens-c2-b4-calibration-v2` is accepted as the run of record (weights T1 0.7, T2 0.0, T3 0.3).

**D35 — Model Citizens G2 failed; C4 scope reframed (2026-10-20, owner's decision after the G2 result).**
- **Evidence.** `citizens-c2-heldout-eval-v1`: status `validated`, commit d249ef0, `git_dirty = false`, protocol v1 frozen, `config_sha256` = `75ed1f96…ac16ee`. Its numbers below come from `criterion_heldout.json`, `criterion_in_sample.json` and `surrogate_fidelity_human.json`. G1b passed first, in `citizens-c1-g1b-v1`:
  - all 13 published z-values that involve held-out mechanisms were reproduced within tolerance;
  - the pool anomaly stays within its D33 bound (23 rows, 5 games), and no human game's depletion round changes under it.
- **§9 outcome: "Only B1 validated". G2 fails** (`g2_passed` = 0).
  - **B1 is validated (closed-loop) on R-I.** 3 of 4 metrics are in range. The miss is players active: 2.90, against the human 2.24 [1.96, 2.53]. Δℓ = 0.104 nat, 95 % CI [0.075, 0.134]. Its mean normalised W1 is 0.26, below M0-sim's 0.35. The A22 rule did not apply on R-I, because M0-sim passes only 2 of 4 metrics there.
  - **No other trainable class passes the outcome criterion (a) on any regime.** Metrics in range, R-I / R-M1 / R-M2: B1 3 / 1 / 2, B2 1 / 2 / 2, B3 0 / 0 / 1, B23 0 / 0 / 1.
  - **All four trainable classes pass the likelihood criterion (b) on all three held-out regimes.** Δℓ in nat, with the 95 % CI lower bound in brackets:

    | Class | R-I | R-M1 | R-M2 |
    |---|---|---|---|
    | B1 | 0.104 [0.075] | 0.081 [0.041] | 0.089 [0.037] |
    | B2 | 0.103 [0.058] | 0.212 [0.154] | 0.175 [0.113] |
    | B3 | 0.180 [0.143] | 0.301 [0.243] | 0.302 [0.250] |
    | B23 | 0.174 [0.146] | 0.256 [0.207] | 0.235 [0.179] |

  - **Labels.** On R-M1 all four classes are labelled "validated (one-step)", because the M1 surrogate failed. Only B1's label counts for G2 there, because B1 is validated closed-loop elsewhere (O3). R-I and R-M2 are closed-loop regimes. There, every class except B1 on R-I is "not validated".
  - **The criterion is not too weak by its control.** B0 fails every regime: 0 metrics in range, Δℓ between −2.88 and −2.63. But criterion (b) alone is a weak bar. The diagnostic persistence baseline P0 also passes it everywhere, with Δℓ 0.116 / 0.223 / 0.231, above B1's.
- **The pattern.** B3 and B23 have a higher Δℓ than B1 on every regime. B2 has a higher Δℓ than B1 on R-M1 and R-M2, and the same on R-I (0.103 vs 0.104). Yet the classes with the best next-choice likelihood do worst in closed loop:
  - **B3 and B23 deplete the pool too early.** Their mean depletion round on R-I is 23.4 and 18.5, against the human 35.2 [33.4, 36.9]. On R-M2 it is 31.1 and 25.9, against 36.5 [34.5, 38.3]. Few players stay active.
  - **B2 errs the other way.** On R-I its pool lasts too long (38.1). Its Gini is too low on every regime: 0.12 / 0.14 / 0.16, against the human 0.18 / 0.26 / 0.29.
  - **In-sample check** (training mechanisms, non-gating). B1 fails Equal: 1 of 4 metrics, with a simulated depletion round of 23.5 against the human 13.3. B1 passes Mixed and Proportional. B2 passes Proportional only. B3 and B23 pass Equal only.
- **Surrogates.**
  - **M1 failed** human-row acceptance on the median error alone: 1.18 coins (Exp 1) and 1.04 coins (Exp 2), against a limit of 1.0. Its p90 errors (4.79, 4.66) and exclusion agreement (0.980, 0.980) were within their limits. So R-M1 is one-step only.
  - **M2 was accepted** (Exp 3: median 0.39, p90 2.68, agreement 0.990).
- **B4.** The B4 held-out runs were still running when this entry was written. They never decide G2 (O4) and are reported when they finish.
- **Owner's decision ("Reframe and continue", asked after the G2 result).**
  1. **The G2 failure is a reported result, not something to rescue.** There is no refit, no new or relaxed criterion and no v2 of `protocol-validation.md`. The recorded labels stand.
  2. **C3 and C4 train RL planners against all four trainable classes,** each labelled honestly:
     - B1: validated (closed-loop) on R-I, and passes (b) on all three held-out regimes;
     - B2, B3 and B23: pass (b) on all three held-out regimes, but are rejected in closed loop on every regime where the closed-loop tier counts.

     "Validated" is never used for B2, B3 or B23 without the qualifier "one-step".
  3. **The C4 question becomes:** do citizen models that each pass a different held-out test (next-choice prediction vs whole-game outcomes) lead RL planners to the same mechanism, and how do the planners fare in cross-play? This is an empirical test of whether one-step behavioural validation is enough for policy design (the Lucas critique).
  4. **B4 stays an evaluation-only population.** Its held-out runs continue to completion under the frozen protocol.
- **Disclosure.** The C4 scope was set *after* the G2 results were seen. It was set *before* any planner was trained and before any C4 comparison was run.
- **What fixes C4.** `studies/model-citizens/protocol.md` will fix the hypotheses, seeds and statistics. It is not yet written; it will be frozen with the owner's approval before any C4 evidence run. The hypotheses in the spec are drafts until then. The spec v0 hypotheses are kept there, marked superseded. G3 is unchanged.

**D36 — Model Citizens C4 protocol frozen; B4's C4 role set by its C2 labels (2026-10-20, owner-approved).**
- **B4 held-out evidence.** `citizens-c2-b4-eval-v1`: status `validated`, commit d249ef0, `git_dirty = false`, protocol-validation v1 (frozen, D34). Six runs of 40 games each: `citizens-c2-b4-{plain,calibrated}-{interpolating,m1,m2}-v1`, 26,703 calls in all, no discarded game. Metrics in range and mean normalised W1 (M0-sim's W1 in brackets):

  | Variant | R-I | R-M1 (diagnostic) | R-M2 |
  |---|---|---|---|
  | plain | 0 of 4, 0.54 (0.35): not validated | 4 of 4, 0.20 (0.46) | 0 of 4, 0.69 (0.45): not validated |
  | calibrated | 1 of 4, 0.40 (0.35): not validated | 3 of 4, 0.34 (0.46) | **3 of 4, 0.32 (0.45): validated (closed-loop)** |

  - R-M1 cannot validate B4, because the M1 surrogate failed (D35) and B4 has no likelihood.
  - B4-calibrated's R-M2 pass is narrow. Gini (0.263, against the human [0.261, 0.317]) and depletion (34.5, against [34.5, 38.3]) sit at the edge of the human interval. Players active (2.90, against [2.15, 2.84]) is in range only through the 0.1 tolerance. Surplus is the miss: 10.1, against [7.84, 9.66]. It is the only model of any class validated in closed loop on R-M2.
  - Plain B4 errs in opposite directions on the two closed-loop regimes. On R-I it is too cooperative (no depletion by round 40). On R-M2 it depletes early, with a high Gini.
  - Run `citizens-c2-b4-calibrated-interpolating-v1` was paused and resumed by its built-in resume after a hang. 10 interrupted games were replayed from round 1, and their 190 partial calls are kept in `abandoned.calls.jsonl`. No game was dropped.
- **C4 role by §9.4, with no other judgement.**
  - **B4-calibrated is a validated evaluation population:** H2-B4 is confirmatory for it.
  - **B4-plain is descriptive:** "B4-plain not validated in closed loop (C2)".
- **Owner's decisions at the freeze.**
  1. Freeze now: `protocol.md` v1 frozen at 6917485. The code lives at 67f31a2, `c4_config_sha256` = `47ebefeb…21ed4b8`, C3 `config_sha256` = `4a312211…a1dde`.
  2. Keep both B4 variants (O2 option (ii)), as decided before the labels were known. The descriptive plain arm is not dropped, so the B4 budget is free of selection on the labels.
- **Disclosure.** The B4 labels were read after §9.4 had fixed their mapping, and before any C4 training or evaluation run.
