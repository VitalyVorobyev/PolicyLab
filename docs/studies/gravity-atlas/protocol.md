# Gravity Atlas M4: comparison protocol

## Freeze checklist

The protocol is frozen only on the study owner's explicit approval. Until then it stays `draft`, and every comparative command refuses to run (`require_frozen`).

**Recorded at freeze** (in the §1 header and in a new D-entry in `docs/decisions.md`, written only on the owner's instruction):
1. Protocol version: **v1**.
2. Status `frozen` and the freeze date.
3. The git commit SHA whose tree is frozen (this text and all M4 code). The commit that flips the status row and records the SHA changes nothing else.
4. The configuration hash `config_sha256`, printed by `just gravity-m4-config` at that commit (draft value at the time of writing: `4cfc119543dd0c53d95750d24561badf1e440aea9170db76ffd165553bf3415a`). Every evidence manifest must carry this hash, `protocol_status = "frozen"` and `git_dirty = false`.
5. The owner's approval, stated in the D-entry.

**Still open before the freeze:** nothing. The owner approved the freeze on 2026-10-18 (D26).

**Order of the post-freeze steps.** Each step writes new run directories and never repeats or overwrites one.
1. *Projection check* (§7.4): recount the projected PE total against the PE ceiling. Wall time is logged but does not trigger reductions (D26).
2. *Ground truth* (§5.3 a–c) with the K1 reproducibility check: `just gravity-m4-groundtruth --confirm-frozen`. K1 failure stops the study.
3. *Tuning* (§6): `just gravity-m4-tune --confirm-frozen` (32 runs), then `just gravity-m4-robustness <run> T_E --confirm-frozen` for each run.
4. *Calibration of B* (§9.1) on the D tuning catalogs. B is recorded before step 5.
5. *Configuration selection* (§6) with P1 at the calibrated B under T_E and P2 on the satisfiable tuning sets. Recorded before step 6.
6. *Pilot* (§8): `just gravity-m4-pilot --configs … --confirm-frozen` and its T_E robustness; the K3 check; the seed count n; the re-projection of §7.4. Recorded before step 7.
7. *Main runs*: `just gravity-m4-main --configs … --confirm-frozen` (plus extension seeds with `just gravity-m4-run` if n > 10) and CC\* (`just gravity-m4-ccstar --confirm-frozen`). Then, for every main catalog and CC\*, the held-out robustness: `H_E_nominal` (primary and stress Gates sets) and `H_E_wide`.
8. *Analysis* (§5.3 d, §10, §11, §12), including the LSODA sample of §5.6.

Steps 4, 5, 6 (seed count) and 8 exist as tested functions (`calibrate_budget`, `seed_count`, the statistics in `stats.py`) but not yet as commands (§13). Their commands read stored runs only, and must implement the definitions in this text exactly.

## 1. Header

| | |
|---|---|
| Protocol version | v1 |
| Status | **`frozen`** (v1). Changes only as a versioned, logged amendment (v2). |
| Frozen date | 2026-10-18, on the study owner's explicit approval (D26) |
| Git commit at freeze | `90c57756d809d1bee0c238e7954c43fe147b055e` |
| Config hash at freeze | `4cfc119543dd0c53d95750d24561badf1e440aea9170db76ffd165553bf3415a` (`just gravity-m4-config`) |
| Author of the draft | drafted 2026-10-18 for review by the study owner |
| Inputs | D16–D25 (`docs/decisions.md`); `README.md`; `literature.md`; `m3-feasibility-design.md` (M3 GO, clean runs at `9a721b0` and `ccf326e`); smoke runs `gravity-m4-smoke-v1` (`84fac87`) and `gravity-m4-smoke-v2` (`373717f`) |
| Configuration | `src/phasespace/gravity/m4/config.py` (every constant; hashed into each manifest). Appendix A repeats the key constants; a test checks both agree. |

**Preconditions for freezing** and their status:

1. **Literature: replaced by a claim restriction (D24).** Everything read in the controllable-set lineage (`literature.md` §9–§11) characterizes the *existence* of feasible returns under nominal dynamics. The protocol may be frozen with these unread: Du & Li, AST 179:113609 (no open-access copy; metadata and reference list only) and Zhu et al., AST 170:111492 (metadata and reference list only). Du et al., AST 176:112680 and CMDA 138:46 were read as abstracts. *Restriction:* the report must not claim that robustness of free-return sets is unstudied, only that budget-matched catalog-member robustness across generators was not found. Read 113609 before publication if it becomes available.
2. **Open questions answered.** Done (D23, §15). The D24 and D25 amendments are integrated in §2, §7.4, §9.1, §10–§12 and §14.
3. **Smoke run.** Done: `gravity-m4-smoke-v1` measured the costs that D25 accepted. `gravity-m4-smoke-v2` re-measured the robustness evaluator after the D25 survival definition (§7.4).
4. **Method references.** *Accepted at freeze (D26).* Storn & Price (1997), *J. Global Optim.* 11:341–359, doi:10.1023/A:1008202821328, and Leary (2000), *J. Global Optim.* 18:367–383, doi:10.1023/A:1026500301312, are verified against Crossref metadata (2026-10-18). The owner accepted metadata-only citation of these two standard algorithm references. The implementations used are SciPy's and the definitions in §6.
5. **Record the freeze** as in the freeze checklist.

A frozen protocol is never edited in place. Changes create v2 with a changelog entry and reason, and every result states which version it used.

## 2. Question and hypothesis

**Question (D16).** At matched propagation budgets, does a MAP-Elites archive of planar CR3BP free returns contain more solutions that remain feasible under held-out injection errors, with a declared correction budget, and under held-out constraint changes than five baselines?
- the baselines are descriptor-binned multi-start, monotonic basin hopping, differential evolution, a continuation-family catalog, and a dense quasi-random map;
- robustness is measured as two separate quantities (D17).

**Primary endpoints** (defined in §10):
- **P1, execution-error tolerance.** The expected number of catalog solutions that remain feasible under the held-out injection-error distribution H_E-nominal, either without correction or after one midcourse correction whose commanded size is ≤ B and which is executed with the primary Gates-model errors (§9.1; D24, D25).
- **P2, changed-constraint availability.** The fraction of satisfiable held-out changed-constraint sets for which the catalog holds at least one solution that satisfies them.

**Hypothesis H1.** At N = 2¹⁴ propagations per run, the MAP-Elites catalog (T) has a higher P1 *and* a higher P2 than each of the five baselines.

**What refutes or limits H1.**
- *Refuted* for a baseline X if X's mean is ≥ T's on both P1 and P2.
- *Not supported* for X if T is not significantly better than X (Holm-adjusted, §11) on at least one endpoint.
- *Dense-map or continuation dominance.* If the dense map D, or the continuation catalog CC, is ≥ T on P1, P2 and coverage at equal budget, the study publishes this null result and stops adding methods (§12, D07).

**Pre-registered secondary hypothesis H2** (Flageat & Cully's knife-edge risk, `literature.md` §7). Elites selected by best nominal quality are less robust per solution. Test: within the T catalogs, the per-solution survival s_i is negatively rank-correlated with the elite-selection margin. The margin is the elite's objective improvement over the next-best feasible evaluation in its cell (§11). Also compare the mean per-solution survival of T with that of D, whose cell representatives are chosen by the same objective but from unoptimized samples.

## 3. Model

Unchanged from M3 (D20). Code: `src/phasespace/gravity/` (`system`, `departure`, `propagation`, `free_return`).

- **Dynamics.** Planar CR3BP, barycentric synodic frame, normalized units.
- **System.** `EARTH_MOON`:
  - μ = 0.012150584394709708 (DE440 GMs);
  - length unit 384 400 km and time unit 375 190.2619 s;
  - Earth radius 6371.0 km and Moon radius 1737.4 km;
  - `length_scale_convention = "nssdca_lunar_semimajor_axis"`.
- **Departure.** Circular parking orbit at 185 km altitude about Earth, posigrade. A single impulsive TLI at t₀ = 0 (`departure.tli_state`), then unpowered coast.
- **Propagation.** DOP853 with rtol = atol = 1e-12 and max_evaluations = 1e6.
  - Events: terminal Earth and Moon collisions with grazing detection, and a terminal inbound return shell at R_S = 6492.92 km (121.92 km altitude).
  - Horizon T_MAX = 10 d.
  - Non-terminal Moon and Earth periapsis events.
- **Not modeled.** Lunar eccentricity, out-of-plane motion, ephemeris and solar perturbations, Earth oblateness, finite burns, navigation errors, atmosphere. Every result inherits these limits.

## 4. Candidate parameterization

Decision vector x = (θ, Δv, α). It is identical to the M3 S4 box:

| Variable | Meaning | Bounds |
|---|---|---|
| θ | departure phase, counterclockwise from +x (Earth→Moon), at t₀ = 0 | [0°, 360°) |
| Δv | TLI impulse magnitude | [3.00, 3.25] km/s |
| α | impulse angle from local horizontal (+ outward) | [−10°, +10°] |

- Every method operates in the unit cube u ∈ [0, 1]³, with x = lo + u (hi − lo). Step sizes, finite differences, mutations and perturbations are all in u, and every proposed point is clipped to [0, 1]³.
- θ is treated as bounded, not periodic. M3 found every feasible sample at θ = 220.7–256.8°, far from the 0/360° seam.
- The full θ circle is kept (open question 6). The feasible set is 0.185 % of this box (M3 S4).

## 5. Feasibility, objective, descriptors and numerical gates

### 5.1 Feasibility (reused from M3, D20)

x is **feasible** iff `free_return.classify` returns `feasible` and the gate in §5.2 passes:
- no Moon collision;
- termination at an inbound crossing of R_S at t_ret ≤ 10 d;
- at least one Moon periapsis before the return;
- the closest one has altitude h_p ∈ [100, 30 000] km.

Everything else is infeasible, with the M3 failure reasons recorded.

### 5.2 Scaled Jacobi gate (M3 must-freeze item 1)

For every propagation whose result can enter a catalog or the ground truth:

drift_rel = max_i |C(s_i) − C(s_0)| / S, with S = max_i [x² + y² + 2(1−μ)/r₁ + 2μ/r₂ + vx² + vy²],

where both maxima are taken over the accepted solver steps.

- **Gate:** drift_rel ≤ 1e-11, i.e. 10 × rtol.
- *Reason.* C is a difference of terms of size ≈ 115 each near departure (2(1−μ)/r₁ = 115.8; v² ≈ 114), so S ≈ 230. An absolute threshold ignores this scale. M3's absolute 1e-9 was ≈ 4e-12 relative, below what a 1e-12-relative integrator guarantees over hundreds of steps; c1 passed by a factor of 1.26.
- *What the M3 data imply.* The gate corresponds to ≈ 2.3e-9 absolute. The M3 maxima, 4.3e-10 (S4 feasible samples), 7.5e-10 (S1) and 8.0e-10 (c1), pass it by factors of ≈ 2.9–5.
- *Disclosure.* The order of magnitude was chosen after seeing M3 drift values, but before any M4 data.
- A propagation that fails the gate is classified `numerical_failure`. It is excluded from catalogs and ground truth, and counted per method. S is recorded per propagation.
- A trajectory with a midcourse correction (§9.1) is gated per segment (before and after the impulse, which changes C); the larger drift_rel counts.

### 5.3 Coverage ground truth with adaptive band-edge refinement (M3 must-freeze item 2)

The ground-truth cell set G (on the §5.5 grid) is the set of cells occupied by any feasible propagation from four sources. Each source's contribution is reported.

- **(a) GT map.** Scrambled Sobol, 2²⁰ points, seed 20261018, over the §4 box.
  - The first 2¹⁷ rows must reproduce `runs/gravity-free-return-map-3d-sobol17-v1` exactly: outcomes and h_p, t_ret, γ_S to the stored precision. This is a reproducibility check (K1), and a mismatch stops the study.
  - The map is not charged to any method. At 64 × N it is denser than any run.
- **(b) Level-1 refinement.** Around every feasible GT-map sample: 32 scrambled-Sobol points in a box centred on the sample.
  - The half-widths equal the GT map's mean spacing per dimension, 2^(−20/3) ≈ 0.0098 of each range: θ ± 3.5°, Δv ± 2.5 m/s, α ± 0.20°.
  - This is aimed at sub-spacing windows such as M3 family (B), which is 0.36–0.40 m/s wide at θ = 251°.
- **(c) Levels 2–3.** Around every point of the previous level that is feasible *and* occupies a cell not in G before that level, 32 points in a box half as wide.
- *Seeding of the boxes.* Every box of level L uses its own generator `qmc.Sobol(d=3, scramble=True, rng=default_rng(seed_L))` with seed_L = 20261019, 20261020, 20261021 for L = 1, 2, 3, and draws `random_base2(5)`. Every box of a level therefore has the same relative point pattern, independent of the order of the boxes. Points are centre + (2s − 1) · half-width, clipped to [0, 1]³.
- **(d) Union with all method runs.** Every feasible propagation of every method, from tuning, pilot and main runs: any feasible evaluation proves its cell exists.

**Saturation check.** Report |G| from the 2¹⁹ prefix of (a), from all of (a), from (a)+(b)+(c), and from (a)–(d). If (a)–(d) exceeds the 2¹⁹-prefix count by more than 10 %, all coverage numbers are reported as upper bounds on the true coverage fraction, and coverage is reported only as a secondary metric (K7). M3 evidence: on the 10 × 10 grid the nested prefixes give 20, 27 and 27 cells at 2¹⁵, 2¹⁶ and 2¹⁷ (one-off analysis of the S4 table).

The same sources define which constraint sets are *satisfiable* (§9.2).

### 5.4 Objective (quality)

For feasible x: **e(x) = |γ_S(x) − γ*|**, with γ* = −6.5°, in degrees. Lower is better.
- γ_S is the Earth-relative inertial flight-path angle at the return-shell crossing (D20).
- Δv is recorded, not optimized (D20).
- Ties are broken by the earlier evaluation index.

M3 evidence that quality within a cell is non-trivial: in the S4 cells with ≥ 3 feasible samples, γ_S spans 12–71°. Only 1 of 243 feasible samples lies within 1° of γ*.

### 5.5 Descriptors and binning

| Descriptor | Definition | Bounds | Bins |
|---|---|---|---|
| d₁ | log₁₀(h_p / km), the closest Moon periapsis altitude before the return | [2, log₁₀ 30 000 = 4.4771] | 10 equal |
| d₂ | t_ret, the return-shell crossing time in days after TLI | [5, 10] | 10 equal |

Bin index: k = ⌊10 (d − lo)/(hi − lo)⌋, clipped to [0, 9]. A feasible t_ret < 5 d is clipped into bin 0 and counted. M3 saw none; its minimum was 5.68 d.

**Justification.**
- *Meaning.* Perilune altitude and return time are the classical free-return family parameters: Pavlak continues the planar family in h_p, and the time of flight is a mission choice. Both are readable on the site.
- *Bounds come from the predicates, not from data.* The h_p band is the feasibility band. The t_ret upper bound is T_MAX. The lower bound of 5 d was declared in the M3 design before any data.
- *Log scale for h_p.* The band spans 2.5 decades, and h_p's sensitivity to the departure state grows with h_p.

**Resolution (primary 10 × 10).** The 3-D feasible set is thin (0.185 % of the box). M3 S4 put 243 feasible samples in 27 of 100 cells, with a median of 6 samples per occupied cell and 2 singletons. The cell count was flat from 2¹⁶ to 2¹⁷, so ground truth at 2²⁰ plus refinement is expected to resolve it.
- At the run budget N = 2¹⁴, a dense map occupies about 16 of 27 cells (S4 prefix). The comparison is therefore in the unsaturated regime, where methods can differ.
- A 20 × 20 grid is still growing at 2¹⁷ (35 → 52 → 65 cells at 2¹⁵ → 2¹⁷), so its ground truth is weak. It is reported as a pre-declared **secondary sensitivity** analysis only. All catalogs are re-binned from the stored evaluations, with no new propagations.

**Catalog.** Every method's output is the same structure: per cell, the feasible evaluation with the lowest e among *all* propagations the method made, including finite-difference and line-search evaluations, within its budget. Each elite must pass the §5.6 re-check. Binning every method's evaluations this way ensures that MAP-Elites' diversity is not just an artifact of its data structure (`README.md`, Stage B).

### 5.6 Elite re-check (numerical validity, D17)

Each catalog elite is re-propagated with DOP853 at 1e-13. It must:
- have the same outcome;
- satisfy |Δh_p| ≤ 10 m, |Δt_ret| ≤ 0.1 s and |Δγ_S| ≤ 1e-3° (the M3 thresholds);
- fall in the same cell.

An elite that fails is removed from its catalog and replaced by the next-best evaluation in the cell, which is re-checked in turn. At most 3 replacements are tried per cell (at most 4 re-checks); if all fail, the cell is empty in that catalog. Every re-check is counted and reported.

In addition, a seeded 5 % sample of the pooled main-run elites, plus every elite shown on the site, gets the full M3 S3 validation, including LSODA at 1e-13. The sample is ⌈0.05 · count⌉ indices drawn without replacement by `default_rng(7301).choice` over the elites pooled in the order (method as listed in §6, seed, cell). If more than 1 % of elites fail the re-check, the study stops (K2).

## 6. Methods

All methods share the following:
- the §4 box in unit coordinates;
- a counting, caching evaluator: an exactly repeated x is not re-propagated or re-counted, as in M3;
- the §5.5 catalog construction;
- the run budget N (§7);
- scrambled-Sobol streams from `scipy.stats.qmc.Sobol(d=3, scramble=True, rng=numpy.random.default_rng(seed))`. Methods that start from Sobol points share them under a common seed, which gives common random numbers across methods (§8);
- **method-internal random streams**, separate from the Sobol generator: `numpy.random.default_rng((seed, s))` with s = 1 for MBH (hop perturbations), 2 for T (parents and variation) and 3 for T2 (emitter elite, parents and variation). DE uses SciPy's own generator, `rng = seed + 1000 k` for restart k.

### Residual and merit for the non-QD optimizers

- r₁ = (h_vp − 38 km) / 10 km. h_vp is the vacuum-perigee altitude at the shell, or at the first post-lunar Earth periapsis when the shell is not reached (M3 §4).
- r₂ = [max(0, 2 − log₁₀ h_p) + max(0, log₁₀ h_p − 4.4771)] / 0.1.
- If h_vp or h_p is undefined, r = (1000, 1000), as in M3 `free_return_search`.
- The scalar merit is J(x) = e(x) if x is feasible, and J(x) = 100 + min(‖r‖₂, 1000) otherwise, so that every feasible point ranks above every infeasible one.
- r₁ = 0 corresponds to γ_S ≈ −6.5° at entry speeds near 11 km/s (M3 c1 and c2).

### Local solve (shared by MS, MBH, CC and the T2 emitter)

`scipy.optimize.least_squares` with method trf, `jac="2-point"`, x_scale = 1 and ftol = xtol = gtol = 1e-10, in a box (default [0, 1]³), with a cap on its *new* propagations (cache hits are free, as in the run budget). When least_squares returns, its returned point is the solve's result. When the cap or the run budget stops it, the result is the touched point with the lowest ‖R‖.

### Baselines

**D, dense quasi-random map** (coverage baseline, D16).
- The first N points of the seed's Sobol stream; each is one propagation.
- Binned into the catalog. No hyperparameters.

**MS, descriptor-binned multi-start** (B1, required).
- Starts are taken from the seed's Sobol stream, in order.
- A local solve on r is launched from a start only if r is defined there; otherwise the next start is taken. The start's propagation is counted in N either way, but **not** against c_MS.
- Local solve cap: c_MS new propagations per start.
- Hyperparameter c_MS ∈ {100, 300}.

**MBH, monotonic basin hopping** (after Leary 2000; Crossref metadata verified, see §1 precondition 4).
- The incumbent x* starts as the result of the first local solve from the Sobol stream (MS rule for starts with undefined r).
- Each hop:
  - perturb with u′ = clip(u* + U(−δ, δ)³);
  - run the local solve on r from u′, capped at 200 new propagations (u′ itself included);
  - compare the lowest-J evaluation the solve touched with the incumbent, and accept if J decreases. For a fresh incumbent the start is included among the touched points. Ties keep the earlier evaluation. The returned least_squares point is not used for the comparison.
- After n_fail consecutive rejections, restart from the next Sobol point.
- Hyperparameters (δ, n_fail) ∈ {(0.005, 10), (0.02, 10), (0.02, 30)}.

**DE, differential evolution** (Storn & Price 1997; Crossref metadata verified, see §1 precondition 4).
- `scipy.optimize.differential_evolution` (SciPy 1.18.1) on J in [0, 1]³: strategy `rand1bin`, mutation dithered in (0.5, 1), recombination 0.9, `init="sobol"`, `polish=False`, `tol=0`, **`updating="deferred"`** (stated explicitly; it makes the result independent of the worker count), `rng = seed + 1000 k` for restart k = 0, 1, …
- `popsize` = ⌊NP/3⌋. With `init="sobol"` SciPy rounds the population up to a power of two, so the **effective population** NP_eff is 32, 64 and 128 for NP = 30, 60 and 120.
- maxiter = ⌊R/NP_eff⌋ − 1, where R is the remaining budget. If DE terminates before N is used, it restarts on the remaining budget; a remainder smaller than one population is spent by a restart that the budget interrupts.
- The population is evaluated as one batch (a map-like `workers` callable); this changes wall time only.
- Hyperparameter NP ∈ {30, 60, 120} (NP_eff ∈ {32, 64, 128}).

**CC, continuation-family catalog** (natural-parameter continuation in the descriptors; after Pavlak 2010 and Jesick & Ocampo 2011).
1. *Seeding.* Draw batches of 2¹⁰ points from the seed's Sobol stream. The batch's feasible samples are taken in Sobol order; each one whose cell holds no CC solution at that moment starts a branch, and its own cell counts as solved by it.
2. *Corrector.* For a target cell with centre (c₁, c₂) and widths (w₁, w₂):
   - the local solve on R = [(d₁ − c₁)/w₁, (d₂ − c₂)/w₂, 10 · r₁], box [0, 1]³;
   - when the shell is not reached, t_ret is replaced by the time of the post-lunar Earth periapsis where h_vp is evaluated; if R is undefined, R = (1000, 1000, 1000);
   - cap: 60 new propagations per target.

   With three equations in three unknowns, the solution in each cell is an isolated free return on the entry corridor. When the cell centre lies outside the reachable set, the solve lands at the nearest reachable point, wherever it bins.
3. *Solved.* A target attempt **solves** the cell that the corrector's result (§ local solve) bins into, if that result is feasible. This cell may differ from the target. A cell is solved at most once; later results that bin into an already solved cell are discarded. An infeasible result is a corrector failure.
4. *Predictor.*
   - If two solved cells lie in line with the target in the same row or column, use the secant extrapolation 2a − b of their solutions (a the nearer); candidate lines are tried in row-major order of their nearer cell.
   - Otherwise use the solution of the first solved 4-neighbour in row-major order.
5. *Continuation order.* Breadth-first over the 4-neighbours of solved cells, in row-major order, separately per branch. Each target is attempted at most once per branch, and is skipped if any branch has already solved it. **Solved cells are shared across branches**: the predictor may use any branch's solutions.
6. When every branch is exhausted, return to step 1 until the budget N is used.

Every propagation, from seeding, predictors and correctors, is counted. Pseudo-arclength continuation is not used: folds of the descriptor map show up as corrector failures, which bound the region. This is a declared limitation.
- **CC\*, oracle-seeded reference.** Steps 2–5 only, seeded with the validated M3 candidates c1 and c2 at no seed cost (their two propagations are logged and counted separately), and stopped when both branches are exhausted, with a safety cap of 50 000 propagations. It is deterministic, so it runs once.
  - CC\* is *not* budget-matched, because finding c1 and c2 cost about 96 000 M3 propagations.
  - It is reported as a reference, outside the hypothesis tests (open question 4).

### Treatment

**T, MAP-Elites** (Mouret & Clune 2015) with iso+line-dd ("directional") variation (Vassiliades & Mouret, GECCO 2018, arXiv:1804.03906; abstract page verified).
- *Initialization.* The first n_init = 2¹² points of the seed's Sobol stream. If the archive is still empty, continue the stream in batches of 2¹⁰.
- *Loop.* Batches of 64 offspring:
  - parents are chosen uniformly from the elites (ordered by cell);
  - u′ = clip(u_i + σ_iso N(0, I₃) + σ_line (u_j − u_i) N(0, 1)), with a second uniformly chosen elite j;
  - a feasible offspring replaces the cell elite if the cell is empty or its e is strictly lower (so ties keep the earlier evaluation);
  - infeasible offspring are discarded but counted.
- *Safety stop.* The run ends early after 100 consecutive batches that make no new propagation (all cache hits); the unused budget is reported.
- Archive: the §5.5 grid, identical to the catalog grid.
- Hyperparameters (σ_iso, σ_line) ∈ {(0.002, 0.2), (0.005, 0.2), (0.01, 0.2)}, with n_init fixed. *Reason for the range:* the feasible slab is 0.36 m/s to 11 m/s thick in Δv (M3 probe), that is 0.0014–0.045 in u.

**T2, MAP-Elites with a local corridor emitter** (open question 5). T's loop with one change. Each batch is:
1. the emitter: one uniformly chosen elite gets a local solve on r₁ alone, from the elite, capped at 40 new propagations, inside the box u_elite ± 0.02 (clipped to [0, 1]³); all of its evaluations enter the archive;
2. then 63 variation offspring (T's operator), whose parents are drawn from the archive *after* the emitter's evaluations entered it.

The hyperparameter grid is **T's three (σ_iso, σ_line) entries**; the emitter settings are fixed. T2 is reported as a secondary arm, outside the Holm family. It isolates whether any T–baseline gap is due to the lack of local refinement.

### How hyperparameters are set (tuning set only)

- Each configuration listed above runs with tuning seeds {9101, 9102} at budget N: MS 2, MBH 3, DE 3, T 3 and T2 3 configurations (14). D and CC have no grid but also run once per tuning seed, because B is calibrated on the D tuning catalogs (§9.1) and the pilot dispersion uses the tuning seeds of every method (§8). The tuning stage is 16 × 2 = 32 runs.
- Configurations are scored after the calibration of B, on two quantities:
  - P1 at the calibrated B under the *tuning* error distribution T_E with the primary Gates set (§9.1, K_T = 32 draws);
  - P2 on the *satisfiable tuning* constraint sets (§9.2).
- Selection rule, identical for every method: the lowest sum of the two ranks (average ranks for ties); remaining ties are broken by coverage against G_tune (the cells of GT sources (a)–(c) and of all tuning runs), then by the lower listed index.
- D has no hyperparameters. CC has fixed settings (seeding batch 2¹⁰, cap 60, weight 10), declared and not tuned. *Disclosure:* the tuning effort is therefore unequal, with 0–3 configurations per method.
- Main runs never reuse tuning seeds, and held-out sets are never evaluated during tuning.

## 7. Budget

### 7.1 Unit

The **primary unit is one full propagation**: TLI to termination, at most T_MAX, with the §3 settings. Every objective evaluation is counted, including finite-difference columns, line-search and trust-region trial points, infeasible and undefined points, and seeding and initialization samples. Exact repeats are cached and not recounted.

- **Run budget N = 2¹⁴ = 16 384 propagations per method per seed** (D23).
- A run stops at the first evaluation that would exceed N. The interrupted local solve or batch keeps the evaluations it already made.

### 7.2 Not captured by the unit

- *Internal method work* is not counted but is logged as wall-clock time: SciPy linear algebra, DE bookkeeping, archive updates, least-squares overhead.
- *Unequal propagation lengths are not normalized.* An early collision is cheaper than a 10 d no-return. The right-hand-side evaluation count is logged per run as a secondary cost.
- *Evaluation costs are outside the search budget*: the robustness evaluation (§9.1), the elite re-checks (§5.6) and the ground truth (§5.3). These are the same procedure for every method and are reported separately, in **propagation equivalents (PE)**: right-hand-side evaluations divided by those of c1's full propagation (4106).
  - A smoke timing on 2026-10-18 measured 100 serial propagations of M3 Sobol samples: the departure-to-1 d arc costs 0.29 PE, and a 1 d-to-termination tail about 0.72–0.75 PE.

### 7.3 Anytime checkpoints

Catalog snapshots are taken at 2¹¹, 2¹², 2¹³ and 2¹⁴ propagations. Coverage, quality and P2 are computed at every checkpoint, since they need no new propagations. P1 is computed at 2¹⁴ only. The anytime curves show where methods cross, which answers `literature.md`'s novelty risk 1.

### 7.4 Compute estimate and ceiling

**Ceiling: 7.8 M PE**, defined as 8 h at the reference rate of 3.7 ms per PE (Apple M4 Pro, 10 workers, measured on c1 by the implementer). The ceiling is a propagation count so that concurrent load on the machine cannot change the course of the study (D26). Wall time is logged for every stage and reported, but it never triggers a reduction. The work may be split across sessions.

**Measured costs** (the robustness costs from c1 with 8 smoke error draws, T_E scales, seed 9002, which is not a protocol seed):

| Quantity | Value | Source |
|---|---|---|
| Robustness setup (arc + base + 2 FD columns) | 2.52 PE per solution | `gravity-m4-smoke-v1`, `-v2` |
| One draw, primary Gates set only (T_E, H_E-wide) | 3.65 PE | `gravity-m4-smoke-v2` (after D25) |
| One draw, primary and stress sets (H_E-nominal) | 4.39 PE | `gravity-m4-smoke-v2` |
| Elite re-check (DOP853 1e-13) | ≈ 1.2 PE | `gravity-m4-smoke-v2` (wall time ratio) |
| Wall time per PE, 10 workers | ≈ 3.7 ms | measured on c1 by the implementer (reported with D25, not stored as a run) |

- *Per draw*, the evaluator propagates the arc to t_c (0.29 PE), the uncorrected tail, at most 3 chord tails, and one executed tail per Gates set when a burn is commanded: at most 0.29 + 6 × 0.75 ≈ 4.8 PE for H_E-nominal. The pre-D25 measurement, 3.17 PE (`-v1`), had no executed tail on non-converged draws.
- *Not used:* the wall times of `gravity-m4-smoke-v2` were measured while another project's processes loaded the machine (load average ≈ 17 on 12 cores; serial propagations 2× slower than in `-v1`). The PE ratios above count right-hand-side evaluations and do not depend on load.
- *Planning rates:* 3.7 ms/PE as measured, and the M3 rates 4.1 ms (fine probe) and 6.6 ms (S4, shared machine) as the range.

Per-elite robustness costs, with ≈ 30 elites per catalog:
- T_E run (tuning, pilot): 1.2 + 2.52 + 32 × 3.65 ≈ 120.5 PE.
- Main catalog: H_E-nominal run 1.2 + 2.52 + 50 × 4.39 ≈ 223.2 PE, plus H_E-wide run 1.2 + 2.52 + 50 × 3.65 ≈ 186.2 PE; ≈ 409.4 PE per elite.

| Item | PE |
|---|---|
| GT map 2²⁰ | 1 048 576 |
| GT refinement, levels 1–3 (≈ 1 950 feasible map samples × 32, plus ≤ 20 000) | ≈ 82 000 |
| Tuning search: 16 runs (14 grid configurations + D + CC) × 2 seeds × 2¹⁴ | 524 288 |
| Tuning robustness: 32 catalogs × 30 elites × 120.5 (T_E) | ≈ 116 000 |
| Pilot search: 3 pilot seeds × 7 methods × 2¹⁴ | 344 064 |
| Pilot robustness: 21 catalogs × 30 × 120.5 (the tuning-seed catalogs are already evaluated) | ≈ 76 000 |
| Main search: 7 methods × 10 seeds × 2¹⁴ | 1 146 880 |
| Main robustness: 70 catalogs × 30 elites × 409.4 (H_E-nominal with both Gates sets, H_E-wide) | ≈ 860 000 |
| CC\* reference (one deterministic run, ≲ 6 000) + its held-out robustness | ≈ 18 000 |
| LSODA S3 validation of the 5 % sample and the site elites | ≈ 5 000 |
| **Total** | **≈ 4.22 M PE** |

This is **≈ 4.3 h at the measured 3.7 ms/PE**, and 4.8–7.7 h at the M3 planning rates (4.1–6.6 ms), within the ceiling at every rate. It matches the compute accepted in D25 (≈ 4.3 M PE including the Gates stress set). Each main seed beyond 10 adds ≈ 0.20 M PE (7 × 2¹⁴ search + 210 elites × 409.4), so n = 16 gives ≈ 5.4 M PE (≈ 5.6 h at 3.7 ms/PE, ≈ 9.9 h at 6.6 ms/PE).

**Projection rule.** A projection is the PE total of all stages, counted from the per-stage PE costs in the table above (RHS-evaluation ratios, which do not depend on machine load):
- (a) before the ground-truth stage (freeze checklist step 1);
- (b) after the pilot, with the measured PE per draw of the completed stages and the seed count n.

**Pre-declared reductions**, applied in this order, only while a projection of the total exceeds 7.8 M PE:
1. The GT map shrinks from 2²⁰ to 2¹⁹ (still 32 × N), saving 524 288 PE. Available at projection (a) only.
2. K per held-out error set drops from 50 to 32.
3. T2 is dropped.

If the projection still exceeds 7.8 M PE, stop and revise the protocol (v2) (K4). N is fixed at 2¹⁴ (D23).

## 8. Seeds

| Use | Seeds |
|---|---|
| Tuning | 9101, 9102 |
| Pilot (dispersion only) | 9201, 9202, 9203 |
| Main | 1001–1010; extension 1011–1016 if the count rule asks for more |
| GT map / refinement levels 1–3 | 20261018 / 20261019, 20261020, 20261021 |
| Injection error draws: T_E, H_E-nominal, H_E-wide | 7001, 7101, 7102 |
| MCC execution-error normals (all Gates sets) | `default_rng((7401, injection seed))` |
| Constraint-set generation / tuning–held-out split | 7201 / 7202 |
| Elite LSODA sample | 7301 |
| Bootstrap; H2 permutation test | 4242; 4242 |
| Method-internal streams | `default_rng((seed, s))`, s = 1 MBH, 2 T, 3 T2; DE `rng = seed + 1000 k` |
| Smoke runs (never evidence) | method seed 9001, error seed 9002 |

- **Pairing.** Seed k is used by every method. Methods that read the Sobol stream (D, MS, MBH, CC, T, T2) therefore share their initial points. The paired differences in §11 are paired by seed.
- **Count rule (pilot dispersion).** The pilot runs the selected configuration of every method on the 3 pilot seeds. Together with the 2 tuning seeds of that configuration, this gives 5 seeds per method. The pilot measures P1 at B under T_E (primary Gates set) and P2 on the satisfiable tuning sets, then computes the SD σ_d (n − 1 denominator) of the paired differences T − X for each baseline X and endpoint.
  - The smallest effects of interest are δ_P1 = 2 surviving solutions and δ_P2 = 0.10.
  - The power target is 0.8 at a two-sided Holm-worst-case α of 0.05/10 = 0.005.
  - n = clamp(⌈(2.807 + 0.842)² · max(σ_d/δ)²⌉, 10, 16) = clamp(⌈13.315 · max(σ_d/δ)²⌉, 10, 16), where (2.807 + 0.842)² = 13.315201 is used unrounded.
- The rule depends only on dispersion, not on the pilot's mean differences.
- If the rule asks for more than 16 seeds, run 16 and report the achieved power. The computed n is recorded in the run manifests before the main runs start.
- With 10 seeds, the exact sign-flip test reaches p = 2/1024 ≈ 0.002 < 0.005.

## 9. Held-out conditions

### 9.1 Injection errors, the correction and its execution errors (execution-error tolerance)

**Injection-error model.** It is applied to a catalog solution's TLI, and the three components are independent, with each draw truncated at 3σ (by inversion of `default_rng(seed).random((K, 3))`, columns magnitude, direction, timing):
- magnitude: Δv′ = Δv (1 + ε_m), with ε_m ~ N(0, σ_m²);
- in-plane pointing: α′ = α + ε_d, with ε_d ~ N(0, σ_d²);
- timing: the burn executes δt late, with δt ~ N(0, σ_t²). The CR3BP is autonomous, so this equals a departure phase θ′ = θ + (n_park − 1) δt, with n_park = 446.233 (normalized). That is 0.06799° per second of delay; the parking period is 88.05 min.

**Injection distributions** (open question 3):

| Set | σ_m | σ_d | σ_t | Draws K | Seed | Gates sets | Use |
|---|---|---|---|---|---|---|---|
| T_E (tuning) | 0.03 % (≈ 0.95 m/s) | 0.02° (≈ 1.1 m/s) | 0.1 s | 32 | 7001 | primary | tuning, pilot, B calibration |
| H_E-nominal (held-out, **primary**) | same | same | same | 50 | 7101 | primary (P1), stress | final evaluation only |
| H_E-wide (held-out distribution) | 0.06 % | 0.04° | 0.2 s | 50 | 7102 | primary | final evaluation only |

- The scales are *declared design values* of order 1 m/s per component. They are not cited operational dispersions.
- The draws are common random numbers: draw k is the same (ε_m, ε_d, δt) triple for every solution of every method.

**Correction (declared budget).**
- **One** in-plane impulsive midcourse correction (MCC) at t_c = 1.0 d after TLI. The earliest lunar encounter in M3 S4 was at 2.54 d.
- It retargets the solution's own nominal perilune altitude and vacuum perigee, (h_p, h_vp): a classical MCC that keeps the designed flyby and entry.
- The MCC is a rotating-frame velocity increment (δv_x, δv_y); its magnitude equals the inertial one.
- The trajectory after a correction is the arc 0 → t_c joined to a new propagation from t_c. §5.1 is evaluated on the joined trajectory and §5.2 per segment.

**Commanded correction** (a chord method with a fixed Jacobian, so that the cost is bounded):
1. Per catalog solution, once:
   - J = ∂(h_p, h_vp)/∂δv at the nominal t_c state, by forward differences with 0.1 m/s steps (2.52 PE measured);
   - if J is undefined or its 2-norm condition number exceeds 1e8, **no correction is commanded** for any draw of this solution (reported). Both rows of J are in km and both columns in m/s, so the condition number does not depend on the unit choice.
2. Per draw:
   - propagate the perturbed TLI to t_c; if the trajectory terminates before t_c, nothing is corrected (it is infeasible);
   - evaluate the uncorrected trajectory and R(0) = (h_p, h_vp)(δv = 0) − nominal;
   - iterate δv_{k+1} = δv_k − J⁻¹ R(δv_k), at most 3 times. The iteration stops early when |R_h_p| ≤ 1 km and |R_h_vp| ≤ 1 km, when R is undefined, or when |δv| > 100 m/s (divergence).
3. **The commanded correction δv_cmd is the final iterate.** It is zero when no iteration was made (the uncorrected residual is already within 1 km or undefined, or J is singular).
4. **Execution (D24).** δv_cmd is executed as δv_cmd + η with Gates-model errors, η ~ N(0, Q) (not truncated), V = |δv_cmd|, e = δv_cmd / V:

   Q = (σ_r² + V² σ_s²) e eᵀ + (σ_a² + V² σ_p²) (I − e eᵀ),

   where I − e eᵀ is the single in-plane transverse direction. η = √(σ_r² + V² σ_s²) z₁ e + √(σ_a² + V² σ_p²) z₂ e_⊥ with e_⊥ = (−e_y, e_x) and (z₁, z₂) standard normal.
   - **No commanded burn, no execution error:** if δv_cmd = 0, no burn is executed and the executed trajectory is the uncorrected one.
   - *Normals.* Draw k of an injection set uses the k-th pair of `default_rng((7401, injection seed)).standard_normal((K, 2))`. The pair is shared by every Gates set, every solution and every method (common random numbers).

**Gates sets** (D24; the parameters of Burnett, Boone & Michelotti 2026, `literature.md` §11):

| Set | σ_s (proportional) | σ_r (fixed, along burn) | σ_p (pointing) | σ_a (fixed, transverse) | Use |
|---|---|---|---|---|---|
| primary | 5 × 10⁻³ | 10 mm/s | 3 × 10⁻⁴ rad | 0.9 mm/s | every injection set; P1 |
| stress (held-out) | 3 × 10⁻² | 10 mm/s | 3 × 10⁻⁴ rad | 0.9 mm/s | H_E-nominal only; secondary |

- *Pairing of the stress set.* The stress set is evaluated only with H_E-nominal, on the same injection draws, the same commanded corrections and the same normals as the primary set. The two differ only through σ_s, so the stress result isolates the effect of correction-execution error. Both are evaluated in one pass, which costs one extra tail per draw.

**Survival (D25).** A draw **succeeds at budget B** iff
- the uncorrected trajectory is feasible (§5.1 and §5.2), *or*
- |δv_cmd| ≤ B *and* the trajectory after the **executed** correction is feasible.

A draw that is feasible without correction therefore counts at every B; B applies to the *commanded* |δv|; and for fixed draws and normals, survival is non-decreasing in B. Convergence of the chord iteration (|R| ≤ 1 km per component) is **not** part of the definition. It is recorded only as a diagnostic, together with divergence, the iteration count and feasibility under the exact commanded correction. When the chord diverged, the executed trajectory is not propagated, and the draw fails at every B ≤ 100 m/s, the range over which survival is defined.
- *Reason (D25).* In the M4 smoke run on c1, injection errors moved h_vp by thousands of km while the trajectory often stayed feasible, and 3 chord iterations rarely reached the 1 km tolerance. A convergence-based definition made survival nearly flat and non-monotone in B for definitional reasons.

Per draw the evaluator records: uncorrected feasibility (survival at B = 0), |δv_cmd|, executed feasibility per Gates set, and the diagnostics. s(B) is then computed for every B without new propagations.

**Budget B (open question 2).**
- B is calibrated *before* configuration selection, on the D catalogs of the two tuning seeds, under T_E with the primary Gates set only. It is the smallest B ∈ {1, 3, 10, 30} m/s for which the median per-solution survival s_i(B), pooled over the elites of both catalogs, falls in [0.2, 0.8]. If none does, it is the grid value whose median is closest to 0.5 (ties: the smaller B).
- *Reason:* to avoid a floor or ceiling that removes discriminative power (`literature.md`, novelty risk 3). The calibration uses no treatment data.
- B is then fixed. Survival curves s(B) over the whole grid are reported as secondary results at no extra cost.

### 9.2 Changed constraints (changed-constraint availability)

A constraint set C = (h_lo, h_hi, T′, γ_c, w) adds requirements to §5.1:
- h_p ∈ [h_lo, h_hi];
- t_ret ≤ T′;
- if a corridor is present, |γ_S − γ_c| ≤ w.

Availability is computed from stored catalog records, with no propagation and no correction.

**Families.** 32 sets each:

| Family | Varies | Ranges | Fixed |
|---|---|---|---|
| F1 geometry | perilune sub-band and maximum TOF | log₁₀ h_lo ∈ [2, 4]; width ∈ [0.5, 2] dex (h_hi = min(h_lo·10^width, 30 000)); T′ ∈ [6.5, 10] d | no corridor |
| F2 entry corridor | corridor | γ_c ∈ [−8°, −5°]; w ∈ [0.5°, 2°] | nominal band [100, 30 000] km, T′ = 10 d |
| F3 combined | all of the above | as F1 and F2 | — |

- **Generation.** One generator per family, `qmc.Sobol(d, scramble=True, rng=default_rng(7201))`, each drawing `random_base2(5)` = 32 points: d = 3 for F1 (columns log₁₀ h_lo, width, T′), d = 2 for F2 (γ_c, w) and d = 5 for F3 (log₁₀ h_lo, width, T′, γ_c, w). Each column is scaled affinely to its range.
- **Split.** One generator `default_rng(7202)` draws three permutations of the 32 set indices, in the order F1, F2, F3. In each family, the first 10 permuted sets are *tuning* sets and the remaining 22 are *held-out* sets: 30 tuning and 66 held-out sets.
- Held-out sets are evaluated only in the final analysis.
- **Satisfiability.**
  - A *held-out* set counts in P2 only if some feasible propagation from G's sources (§5.3 a–d) satisfies it.
  - A *tuning* set counts in tuning and pilot P2 only if some feasible propagation from GT sources (a)–(c) or from any tuning-stage run satisfies it. This list is fixed after the tuning stage and uses no pilot, main or held-out data.
  - Unsatisfiable sets are dropped and their number is reported.

## 10. Metrics

Per run (method × seed), on its catalog (§5.5, after §5.6):

| Metric | Definition | Unit |
|---|---|---|
| **P1** | Σ_i s_i, where s_i is the fraction of the 50 H_E-nominal draws that succeed at budget B (§9.1) with the primary Gates set, for solution i | solutions |
| **P2** | fraction of satisfiable held-out constraint sets for which ≥ 1 catalog solution satisfies the set (no correction) | — |
| Coverage | \|catalog cells ∩ G\| / \|G\| (10 × 10; 20 × 20 secondary) | — |
| Quality | mean and median e over elites; QD-score Σ_cells (90° − e) | deg |
| Per-solution robustness | mean s_i; P1 at B = 0; s(B) curves; P1 under H_E-wide; P1 under the Gates stress set (H_E-nominal) | — |
| Worst case | min_i s_i; the 10th percentile of e; the worst seed's P1 and P2 | — |
| Correction diagnostics | fraction of draws with a commanded burn; median \|δv_cmd\|; chord convergence and divergence rates; singular-J solutions | — |
| Feasibility yield | feasible propagations / N; first-feasible index | — |
| Diversity beyond cells | number of distinct (lunar passage sense, entry sense) classes in the catalog | count |
| P2 by family | F1, F2 and F3 separately | — |
| Cost | propagations used (≤ N); RHS evaluations; wall time; numerical-failure count | count, h |

P1 deliberately combines coverage and per-solution robustness, because D16 asks how many robust solutions a catalog *contains*. The mean s_i separates the two.

## 11. Statistical analysis

- **Summaries.** For every method and metric, across main seeds: mean, median, SD, min and max, plus a per-seed table. Anytime curves show mean ± 95 % bootstrap CI at each checkpoint.
- **Primary comparisons.** Ten contrasts: T − X for X ∈ {D, MS, MBH, DE, CC} on P1 and P2, paired by seed.
  - Effect: the mean paired difference, with a 95 % percentile bootstrap CI over seeds (10 000 resamples, seed 4242).
  - Test: an exact two-sided sign-flip permutation test on the paired differences (2^n flips; ties within a relative 1e-12 count as ties).
  - Holm–Bonferroni across the ten contrasts at a family-wise α = 0.05.
- **Secondary analyses** (not in the Holm family, reported as estimates with CIs):
  - T2 vs every arm, and T2 vs T;
  - coverage and quality contrasts;
  - H_E-wide;
  - the Gates stress set (H_E-nominal), and its paired difference from the primary set per method;
  - s(B);
  - the 20 × 20 sensitivity analysis;
  - P2 by family;
  - CC\* vs every arm (unmatched, labelled as such);
  - **H2:** the Spearman correlation (average ranks for ties) of s_i (at B, H_E-nominal, primary Gates set) with the elite-selection margin within T, pooled over main seeds. The margin is e(next-best feasible evaluation in the cell) − e(elite); elites without a next-best evaluation are excluded. Two-sided p from 10 000 permutations of the margins within each seed, generator `default_rng(4242)`, p = (1 + #{|ρ_perm| ≥ |ρ|}) / (1 + 10 000). Also the mean s_i of T vs D.
- **Required reporting.** Failed and diverged local solves, numerical failures, elites rejected by the re-check, unsatisfiable sets, correction diagnostics, and the achieved power when n hits its cap.
- **Not done.** No exclusion of seeds, no post-hoc metrics promoted to primary, and no re-tuning after held-out evaluation.

## 12. Stop / kill criteria

| # | Condition | Action |
|---|---|---|
| K0 | Any §1 precondition unmet | Do not freeze. (The literature precondition is replaced by the D24 claim restriction.) |
| K1 | The GT 2¹⁷ prefix does not reproduce `gravity-free-return-map-3d-sobol17-v1` | Stop; investigate the numerics before any comparison. |
| K2 | > 1 % of elites fail the §5.6 re-check, or > 1 % of any method's propagations fail the §5.2 gate | Stop; investigate. Numerics come before comparison (`research-methodology.md`). |
| K3 | After B calibration, the pilot median s_i of *every* method is < 0.05 or > 0.95 under T_E (primary Gates set) | P1 cannot discriminate; revise to v2 before the main runs. |
| K4 | A §7.4 projection exceeds 7.8 M PE after the reductions | Stop; revise the scope (v2). |
| K5 | **D (or CC) ≥ T on P1, P2 and coverage (point estimates) at equal budget** | Publish the null ("a dense map / a continuation catalog dominates at equal budget"). Stop adding methods. No QD variants are added to rescue the result (D07; roadmap M4 go/no-go). |
| K6 | H1 not supported for any baseline, and T not better than D on either endpoint | Publish as a negative result with the full metric set. |
| K7 | G grows by > 10 % between the 2¹⁹ prefix and all sources | Report coverage as an upper bound and as secondary only; P1 and P2 are unaffected. |

## 13. Reproduction

Every constant is in `src/phasespace/gravity/m4/config.py`; its canonical JSON is hashed (`config_sha256`) into every M4 manifest. Each command writes a new run directory and never overwrites one. Every command except `config` and `smoke` refuses unless the status above is `frozen` and `--confirm-frozen` is given.

```text
just gravity-m4-config                       # config hash and key constants (--json: all)
just gravity-m4-smoke <run_id> 256 10        # pre-freeze smoke and cost probe (seed 9001; never evidence)
just gravity-m4-groundtruth --confirm-frozen # §5.3 (a)–(c) and K1; run gravity-m4-groundtruth-v1
just gravity-m4-tune --confirm-frozen        # §6 tuning stage, seeds 9101–9102 (32 runs);
                                             #   runs gravity-m4-tune-<method>-c<k>-s<seed>-v1
just gravity-m4-pilot --configs MS=…,MBH=…,DE=…,T=…,T2=… --confirm-frozen   # seeds 9201–9203
just gravity-m4-main  --configs … --confirm-frozen                           # seeds 1001–1010
just gravity-m4-run --stage main --method <M> --config-index <k> --seed <s> --confirm-frozen
just gravity-m4-ccstar --confirm-frozen      # CC*; run gravity-m4-ccstar-v1
just gravity-m4-robustness <run_id> <T_E|H_E_nominal|H_E_wide> --confirm-frozen
                                             # §5.6 re-check, then §9.1 with the paired Gates sets;
                                             #   run <run_id>-robustness-<set>
```

**Not yet commands** (tested functions exist; the commands are added before the step runs and implement the definitions above exactly, reading stored runs only): the B calibration (`robustness.calibrate_budget`), configuration selection, the seed count (`stats.seed_count`), satisfiability, and the analysis (`stats`: paired contrasts, sign-flip test, Holm, bootstrap, H2 permutation test).

Every run manifest records the protocol version and status, config hash, commit, `git_dirty` (which must be false for evidence), environment, seeds and budget counters. Evidence comes only from clean commits.

## 14. Risks to validity (known now)

- **Strawman risk for T.** Plain MAP-Elites does not refine e locally, while MS, MBH and CC solve r₁ = 0 directly. T may lose on quality, and therefore on F2/F3 availability, for that reason alone. T2 (open question 5) measures this effect.
- **CC may dominate.** Three equations in three unknowns per cell make continuation very cheap: a full 10 × 10 sweep needs on the order of 10³ propagations once seeded. This is a real, publishable outcome (K5), not a flaw.
- **The injection-error scales are declared, not sourced.** P1's absolute values depend on them. Comparisons between methods are less sensitive, because the draws are common random numbers. The execution-error parameters are sourced (Burnett et al. 2026, after Gates 1963, seen second-hand).
- **The commanded correction is at most 3 chord iterations on a fixed Jacobian.** A full Newton solve, or a correction designed for the dispersed trajectory (Burnett et al.), could command a better burn. Survival is judged by feasibility after the executed burn, not by targeting accuracy, so a cruder command lowers survival in the same way for every method. This is a declared limitation.
- **Execution errors can dominate small corrections.** σ_r = 10 mm/s is fixed: a commanded burn of a few mm/s is executed with a relative error near 100 %. This is the intended realism of D24, and it affects every method equally.
- **The binning and saturation numbers come from a one-off analysis** of the M3 S4 table, not from a recipe. The GT run recomputes them.
- **The ground truth includes the methods' own finds**, so coverage is relative to the union of everything observed.

## 15. Open questions for the user before freezing

**Resolved 2026-10-18 (D23):** the user accepted all six recommendations below.

**Amended 2026-10-18 (D24), before any M4 data:** (1) the Du & Li AST 179:113609 reading precondition is replaced by a claim restriction (§1 precondition 1); (2) the midcourse correction carries Gates-model execution errors (σs = 5×10⁻³, σr = 10 mm/s, σp = 3×10⁻⁴ rad, σa = 0.9 mm/s; held-out stress σs = 3×10⁻²) following Burnett et al. 2026 (§9.1).

**Amended 2026-10-18 (D25), before any M4 data:** survival is feasibility after the executed correction with commanded |δv| ≤ B; a draw feasible without correction counts at every B; chord convergence is a diagnostic only (§9.1). The implementation's working choices were adopted for the details listed below, and the compute estimate was updated (§7.4).

Details fixed with D25 (each in the section named):
- §6: the MS start counts in N but not against c_MS; MBH compares the lowest-J evaluation of a solve; DE uses `updating="deferred"` and the effective populations 32/64/128; CC's "solved" and the sharing of solved cells across branches; T2's grid (T's three entries) and batch composition (emitter + 63 offspring); the method-internal random streams; the T safety stop.
- §5.3, §9.2: the seeding of the refinement boxes, the constraint families and the split; the satisfiability of tuning sets.
- §5.6: elite plus at most 3 replacements.
- §8, §11: 13.315 used unrounded; the H2 permutation count and seed.
- §9.1: the Gates normals seed 7401 shared by both sets; no burn means no execution error; B applies to the commanded |δv|; the stress set pairs with H_E-nominal; cond(J) is unit-independent.
- §6, §7.4: D and CC also run on the tuning seeds.

1. **Run budget N.** *Recommend N = 2¹⁴.* A dense map then covers about 16 of 27 known cells (unsaturated). N = 2¹⁵ would roughly double the search lines and need reductions.
2. **Correction budget B.** *Recommend calibrating B on the dense-map tuning catalogs* to a median survival in [0.2, 0.8], from {1, 3, 10, 30} m/s, with s(B) reported for the whole grid. The alternative is a fixed B = 10 m/s declared now, which risks a floor or ceiling effect.
3. **Injection-error scales.** *Recommend the declared σ = (0.03 %, 0.02°, 0.1 s)* with a 2× held-out set, labelled as design values. The alternative is to first find a citable TLI dispersion source, which delays the freeze.
4. **Continuation baseline seeding.** *Recommend the budget-matched CC* (in-budget Sobol seeding) as the compared baseline, plus the oracle-seeded CC\* (from M3 c1 and c2) as an unmatched reference outside the tests.
5. **Second treatment arm T2** (MAP-Elites with a local corridor emitter). *Recommend including it as a pre-registered secondary arm* outside the Holm family. Without it, a negative T result cannot be told apart from "QD without local refinement".
6. **Search box.** *Recommend keeping the full M3 box* (θ over the full circle). Narrowing it to the region M3 found would leak map knowledge into every method and favour the dense map least.

## Appendix A. Key constants (machine-readable)

These values repeat `src/phasespace/gravity/m4/config.py`. The test `tests/gravity/test_gravity_m4_protocol_constants.py` parses this block and fails if the two disagree. Lengths are km, times days or seconds as named, speeds m/s, angles degrees unless named `_rad`; `u` coordinates are unit-cube fractions.

<!-- m4-constants: begin -->
```toml
version = "v1"
run_budget = 16384
checkpoints = [2048, 4096, 8192, 16384]

[box]
lo = [0.0, 3.0, -10.0]
hi = [360.0, 3.25, 10.0]

[descriptors]
d1_bounds = [2.0, 4.477121254719663]
d2_bounds = [5.0, 10.0]
bins = 10
secondary_bins = 20

[objective]
gamma_target_deg = -6.5
vacuum_perigee_target_km = 38.0
vacuum_perigee_scale_km = 10.0
perilune_log_band = [2.0, 4.4771]
perilune_log_scale = 0.1
undefined_residual = 1000.0
infeasible_merit_offset = 100.0
infeasible_merit_cap = 1000.0

[gate]
jacobi_drift_rel_max = 1e-11
recheck_rtol = 1e-13
recheck_perilune_km = 0.010
recheck_return_time_s = 0.1
recheck_gamma_deg = 1e-3
recheck_max_replacements = 3
lsoda_fraction = 0.05
lsoda_seed = 7301
stop_fraction = 0.01

[ground_truth]
log2_map = 20
map_seed = 20261018
m3_prefix_log2 = 17
saturation_prefix_log2 = 19
refinement_seeds = [20261019, 20261020, 20261021]
refinement_points = 32
saturation_tolerance = 0.10

[local_solve]
method = "trf"
jac = "2-point"
x_scale = 1.0
tolerance = 1e-10

[methods]
ms_max_propagations_per_start = [100, 300]
mbh_delta_n_fail = [[0.005, 10], [0.02, 10], [0.02, 30]]
mbh_local_cap = 200
de_population = [30, 60, 120]
de_effective_population = [32, 64, 128]
de_strategy = "rand1bin"
de_mutation = [0.5, 1.0]
de_recombination = 0.9
de_updating = "deferred"
de_restart_seed_stride = 1000
cc_seed_batch = 1024
cc_cap_per_target = 60
cc_r1_weight = 10.0
cc_star_safety_cap = 50000
t_sigma_iso_line = [[0.002, 0.2], [0.005, 0.2], [0.01, 0.2]]
t_n_init = 4096
t_init_batch = 1024
t_batch = 64
t_max_stalled_batches = 100
t2_emitter_cap = 40
t2_emitter_half_width = 0.02
rng_streams = { MBH = 1, T = 2, T2 = 3 }

[seeds]
tuning = [9101, 9102]
pilot = [9201, 9202, 9203]
main = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
main_extension = [1011, 1012, 1013, 1014, 1015, 1016]
bootstrap = 4242

[errors.T_E]
sigma_magnitude = 3e-4
sigma_direction_deg = 0.02
sigma_timing_s = 0.1
draws = 32
seed = 7001
held_out = false
executions = ["gates_primary"]
truncation_sigmas = 3.0

[errors.H_E_nominal]
sigma_magnitude = 3e-4
sigma_direction_deg = 0.02
sigma_timing_s = 0.1
draws = 50
seed = 7101
held_out = true
executions = ["gates_primary", "gates_stress"]
truncation_sigmas = 3.0

[errors.H_E_wide]
sigma_magnitude = 6e-4
sigma_direction_deg = 0.04
sigma_timing_s = 0.2
draws = 50
seed = 7102
held_out = true
executions = ["gates_primary"]
truncation_sigmas = 3.0

[correction]
t_c_days = 1.0
fd_step_m_s = 0.1
cond_max = 1e8
max_iterations = 3
tolerance_km = 1.0
divergence_m_s = 100.0
budget_grid_m_s = [1.0, 3.0, 10.0, 30.0]
calibration_band = [0.2, 0.8]

[gates]
normals_seed = 7401

[gates.gates_primary]
sigma_s = 5e-3
sigma_r_m_s = 0.010
sigma_p_rad = 3e-4
sigma_a_m_s = 0.0009
held_out = false

[gates.gates_stress]
sigma_s = 3e-2
sigma_r_m_s = 0.010
sigma_p_rad = 3e-4
sigma_a_m_s = 0.0009
held_out = true

[constraints]
sets_per_family = 32
tuning_per_family = 10
generation_seed = 7201
split_seed = 7202
log_h_lo = [2.0, 4.0]
width_dex = [0.5, 2.0]
t_max_days = [6.5, 10.0]
gamma_c_deg = [-8.0, -5.0]
corridor_width_deg = [0.5, 2.0]
h_cap_km = 30000.0

[statistics]
alpha = 0.05
holm_family_size = 10
bootstrap_resamples = 10000
bootstrap_seed = 4242
confidence = 0.95
effect_p1 = 2.0
effect_p2 = 0.10
z_alpha = 2.807
z_power = 0.842
seed_count_bounds = [10, 16]
h2_permutations = 10000
h2_permutation_seed = 4242
```
<!-- m4-constants: end -->
