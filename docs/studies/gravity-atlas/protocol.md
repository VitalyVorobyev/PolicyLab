# Gravity Atlas M4: comparison protocol

## 1. Header

| | |
|---|---|
| Protocol version | v1 |
| Status | **`draft`. Not frozen.** It must not be used to produce comparative results. |
| Frozen date | — |
| Git commit at freeze | — |
| Author of the draft | drafted 2026-10-18 for review by the study owner |
| Inputs | D16–D22 (`docs/decisions.md`); `README.md`; `literature.md`; `m3-feasibility-design.md` (M3 GO, clean runs at `9a721b0` and `ccf326e`) |

**Blocking preconditions for freezing.** All of these must hold before the status changes to `frozen`:

1. **Read the Du & Li 2026 "controllable set" papers and Zhu et al. 2025** (`literature.md`, Unverified leads):
   - AST 179:113609;
   - AST 176:112680;
   - CMDA 138 (doi 10.1007/s10569-026-10321-2);
   - Zhu et al., AST 170:111492.

   They are paywalled and have not been read. A controllable set of free returns may already characterize the set of departure (or post-departure) states from which a feasible return exists. That would be a reachability-style answer to the D17 robustness question. Update `literature.md` with what they do and do not do. If they already answer the D16 question, revise the question (new D-entry) before freezing.
2. The study owner answers the open questions in §15. Each answer either keeps the recommended option or changes this draft.
3. A smoke run of the M4 implementation on this machine re-measures the cost per propagation-equivalent and confirms the §7.4 estimate under the 8 h ceiling. The current estimate comes from M3 runs and a serial smoke timing, not from M4 code.
4. Verify by full text, or replace, the method references whose content has not been read (§6): Leary (2000) and Storn & Price (1997) are metadata-verified only.
5. Record the freeze in `docs/decisions.md` (a new D-entry with the commit), and copy every frozen constant into the config file named in §13.

A frozen protocol is never edited in place. Changes create v2 with a changelog entry and reason, and every result states which version it used.

## 2. Question and hypothesis

**Question (D16).** At matched propagation budgets, does a MAP-Elites archive of planar CR3BP free returns contain more solutions that remain feasible under held-out injection errors, with a declared correction budget, and under held-out constraint changes than five baselines?
- the baselines are descriptor-binned multi-start, monotonic basin hopping, differential evolution, a continuation-family catalog, and a dense quasi-random map;
- robustness is measured as two separate quantities (D17).

**Primary endpoints** (defined in §10):
- **P1, execution-error tolerance.** The expected number of catalog solutions that remain feasible after at most one midcourse correction of size ≤ B, under the held-out injection-error distribution H_E.
- **P2, changed-constraint availability.** The fraction of held-out changed-constraint sets for which the catalog holds at least one solution that satisfies them.

**Hypothesis H1.** At N = 2¹⁴ propagations per run, the MAP-Elites catalog (T) has a higher P1 *and* a higher P2 than each of the five baselines.

**What refutes or limits H1.**
- *Refuted* for a baseline X if X's mean is ≥ T's on both P1 and P2.
- *Not supported* for X if T is not significantly better than X (Holm-adjusted, §11) on at least one endpoint.
- *Dense-map or continuation dominance.* If the dense map D, or the continuation catalog CC, is ≥ T on P1, P2 and coverage at equal budget, the study publishes this null result and stops adding methods (§12, D07).

**Pre-registered secondary hypothesis H2** (Flageat & Cully's knife-edge risk, `literature.md` §7). Elites selected by best nominal quality are less robust per solution. Test: within the T catalogs, the per-solution survival s_i is negatively rank-correlated with the elite-selection margin. The margin is the elite's objective improvement over the next-best feasible sample in its cell. Also compare the mean per-solution survival of T with that of D, whose cell representatives are chosen by the same objective but from unoptimized samples.

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

- Every method operates in the unit cube u ∈ [0, 1]³, with x affine in u. Step sizes, finite differences, mutations and perturbations are all in u.
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
- A propagation that fails the gate is classified `numerical_failure`. It is excluded from catalogs and ground truth, and counted per method. S must be recorded per propagation, which is a code addition.

### 5.3 Coverage ground truth with adaptive band-edge refinement (M3 must-freeze item 2)

The ground-truth cell set G (on the §5.5 grid) is the set of cells occupied by any feasible propagation from four sources. Each source's contribution is reported.

- **(a) GT map.** Scrambled Sobol, 2²⁰ points, seed 20261018, over the §4 box.
  - The first 2¹⁷ rows must reproduce `runs/gravity-free-return-map-3d-sobol17-v1` exactly: outcomes and h_p, t_ret, γ_S to the stored precision. This is a reproducibility check, and a mismatch stops the study.
  - The map is not charged to any method. At 64 × N it is denser than any run.
- **(b) Level-1 refinement.** Around every feasible GT-map sample: 32 scrambled-Sobol points (seed 20261019) in a box centred on the sample.
  - The half-widths equal the GT map's mean spacing per dimension, 2^(−20/3) ≈ 0.0098 of each range: θ ± 3.5°, Δv ± 2.5 m/s, α ± 0.20°.
  - This is aimed at sub-spacing windows such as M3 family (B), which is 0.36–0.40 m/s wide at θ = 251°.
- **(c) Levels 2–3.** Around every refinement point that is feasible *and* occupies a cell not yet in G, 32 points in a box half as wide, up to level 3. Seeds 20261020 and 20261021.
- **(d) Union with all method runs.** Every feasible propagation of every method, from tuning, pilot and main runs: any feasible evaluation proves its cell exists.

**Saturation check.** Report |G| from the 2¹⁹ prefix of (a), from all of (a), from (a)+(b)+(c), and from (a)–(d). If (a)–(d) exceeds the 2¹⁹-prefix count by more than 10 %, all coverage numbers are reported as upper bounds on the true coverage fraction, and coverage is reported only as a secondary metric. M3 evidence: on the 10 × 10 grid the nested prefixes give 20, 27 and 27 cells at 2¹⁵, 2¹⁶ and 2¹⁷ (one-off analysis of the S4 table).

The same procedure, with sources (a)–(d) and the changed constraints of §9.2 applied, defines which constraint sets are *satisfiable* (§10).

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

An elite that fails is removed from its catalog, and the next-best evaluation in the cell is re-checked, up to 3 times.

In addition, a seeded 5 % sample of all elites (seed 7301), plus every elite shown on the site, gets the full M3 S3 validation, including LSODA at 1e-13. If more than 1 % of elites fail the re-check, the study stops (§12).

## 6. Methods

All methods share the following:
- the §4 box in unit coordinates;
- a counting, caching evaluator: an exactly repeated x is not re-propagated or re-counted, as in M3;
- the §5.5 catalog construction;
- the run budget N (§7);
- scrambled-Sobol streams from `scipy.stats.qmc.Sobol(d=3, scramble=True, rng=numpy.random.default_rng(seed))`. Methods that start from Sobol points share them under a common seed, which gives common random numbers across methods (§8).

### Residual and merit for the non-QD optimizers

- r₁ = (h_vp − 38 km) / 10 km. h_vp is the vacuum-perigee altitude at the shell, or at the first post-lunar Earth periapsis when the shell is not reached (M3 §4).
- r₂ = [max(0, 2 − log₁₀ h_p) + max(0, log₁₀ h_p − 4.4771)] / 0.1.
- If h_vp or h_p is undefined, r = (UNDEFINED, UNDEFINED), as in M3 `free_return_search`.
- The scalar merit is J(x) = e(x) if x is feasible, and J(x) = 100 + min(‖r‖₂, 1000) otherwise, so that every feasible point ranks above every infeasible one.
- r₁ = 0 corresponds to γ_S ≈ −6.5° at entry speeds near 11 km/s (M3 c1 and c2).

### Baselines

**D, dense quasi-random map** (coverage baseline, D16).
- The first N points of the seed's Sobol stream; each is one propagation.
- Binned into the catalog. No hyperparameters.

**MS, descriptor-binned multi-start** (B1, required).
- Starts are taken from the seed's Sobol stream, in order.
- A local solve is launched from a start only if r is defined there; otherwise the next start is taken. The start's propagation is counted either way.
- Local solve: `scipy.optimize.least_squares` on r with method trf, `jac="2-point"`, box [0, 1]³, x_scale = 1, ftol = xtol = gtol = 1e-10, and at most c_MS propagations per start.
- Hyperparameter c_MS ∈ {100, 300}.

**MBH, monotonic basin hopping** (after Leary 2000; metadata verified only, see §1 item 4).
- The incumbent x* starts as the first local-solve result from the Sobol stream.
- Each hop:
  - perturb with u′ = clip(u* + U(−δ, δ)³);
  - run the MS local solve, capped at 200 propagations;
  - accept if J decreases.
- After n_fail consecutive rejections, restart from the next Sobol point.
- Hyperparameters (δ, n_fail) ∈ {(0.005, 10), (0.02, 10), (0.02, 30)}.

**DE, differential evolution** (Storn & Price 1997; metadata verified only).
- `scipy.optimize.differential_evolution` (SciPy 1.18.1) on J: strategy `rand1bin`, mutation dithered in (0.5, 1), recombination 0.9, `init="sobol"`, `polish=False`, `tol=0`, `rng=seed`.
- maxiter = ⌊N/NP⌋ − 1. If it terminates before N is used, it restarts with seed + 1000·k on the remaining budget.
- Hyperparameter NP (total population) ∈ {30, 60, 120}, with scipy `popsize` = NP/3.

**CC, continuation-family catalog** (natural-parameter continuation in the descriptors; after Pavlak 2010 and Jesick & Ocampo 2011).
1. *Seeding.* Draw batches of 2¹⁰ points from the seed's Sobol stream. Every feasible sample whose cell holds no CC solution starts a branch.
2. *Corrector.* For a target cell with centre (c₁, c₂) and widths (w₁, w₂):
   - solve `least_squares` (trf, 2-point, box [0, 1]³) on R = [(d₁ − c₁)/w₁, (d₂ − c₂)/w₂, 10 · r₁];
   - when the shell is not reached, t_ret is replaced by the time of the post-lunar Earth periapsis where h_vp is evaluated;
   - cap: 60 propagations per target.

   With three equations in three unknowns, the solution in each cell is an isolated free return on the entry corridor. When the cell centre lies outside the reachable set, the solve lands at the nearest reachable point, wherever it bins.
3. *Predictor.*
   - If two solved cells lie in line with the target in the same row or column, use the secant extrapolation of their solutions.
   - Otherwise use the solution of the nearest solved 4-neighbour.
4. *Continuation order.* Breadth-first over the 4-neighbours of solved cells, in row-major tie order, separately per branch. Each target is attempted once per branch.
5. When every branch is exhausted, return to step 1 until the budget N is used.

Every propagation, from seeding, predictors and correctors, is counted. Pseudo-arclength continuation is not used: folds of the descriptor map show up as corrector failures, which bound the region. This is a declared limitation.
- **CC\*, oracle-seeded reference.** Steps 2–4 only, seeded with the validated M3 candidates c1 and c2 at no seed cost, and stopped when both branches are exhausted. It is deterministic, so it runs once.
  - CC\* is *not* budget-matched, because finding c1 and c2 cost about 96 000 M3 propagations.
  - It is reported as a reference, outside the hypothesis tests (open question 4).

### Treatment

**T, MAP-Elites** (Mouret & Clune 2015) with iso+line-dd ("directional") variation (Vassiliades & Mouret, GECCO 2018, arXiv:1804.03906; abstract page verified).
- *Initialization.* The first n_init = 2¹² points of the seed's Sobol stream. If the archive is still empty, continue the stream in batches of 2¹⁰.
- *Loop.* Batches of 64 offspring:
  - parents are chosen uniformly from the elites;
  - u′ = clip(u_i + σ_iso N(0, I₃) + σ_line (u_j − u_i) N(0, 1)), with a second uniformly chosen elite j;
  - a feasible offspring replaces the cell elite if the cell is empty or its e is lower;
  - infeasible offspring are discarded but counted.
- Archive: the §5.5 grid, identical to the catalog grid.
- Hyperparameters (σ_iso, σ_line) ∈ {(0.002, 0.2), (0.005, 0.2), (0.01, 0.2)}, with n_init fixed. *Reason for the range:* the feasible slab is 0.36 m/s to 11 m/s thick in Δv (M3 probe), that is 0.0014–0.045 in u.

**T2, MAP-Elites with a local corridor emitter** (proposed, open question 5). T's loop with one change: in each batch, one uniformly chosen elite gets a `least_squares` solve on r₁ alone, capped at 40 propagations, inside a box of ±0.02 (in u) around the elite. All of its evaluations enter the archive, and the rest of the batch is T's variation. It is reported as a secondary arm, outside the Holm family. It isolates whether any T–baseline gap is due to the lack of local refinement.

### How hyperparameters are set (tuning set only)

- Each configuration listed above runs with tuning seeds {9101, 9102} at budget N.
- Configurations are scored on two quantities:
  - P1 under the *tuning* error distribution T_E (§9.1, K_T = 32 draws);
  - P2 on the *tuning* constraint sets (§9.2).
- Selection rule, identical for every method: the lowest sum of the two ranks; ties are broken by coverage, then by the lower listed index.
- D has no hyperparameters. CC has fixed settings (seeding batch, cap 60, weight 10), declared and not tuned. *Disclosure:* the tuning effort is therefore unequal, with 0–3 configurations per method.
- Main runs never reuse tuning seeds, and held-out sets are never evaluated during tuning.

## 7. Budget

### 7.1 Unit

The **primary unit is one full propagation**: TLI to termination, at most T_MAX, with the §3 settings. Every objective evaluation is counted, including finite-difference columns, line-search and trust-region trial points, infeasible and undefined points, and seeding and initialization samples. Exact repeats are cached and not recounted.

- **Run budget N = 2¹⁴ = 16 384 propagations per method per seed** (open question 1).
- A run stops at the first evaluation that would exceed N. The interrupted local solve or batch keeps the evaluations it already made.

### 7.2 Not captured by the unit

- *Internal method work* is not counted but is logged as wall-clock time: SciPy linear algebra, DE bookkeeping, archive updates, least-squares overhead.
- *Unequal propagation lengths are not normalized.* An early collision is cheaper than a 10 d no-return. The right-hand-side evaluation count is logged per run as a secondary cost.
- *Evaluation costs are outside the search budget*: the robustness evaluation (§9.1), the elite re-checks (§5.6) and the ground truth (§5.3). These are the same procedure for every method and are reported separately, in **propagation equivalents (PE)**.
  - A smoke timing on 2026-10-18 measured 100 serial propagations of M3 Sobol samples: the departure-to-1 d arc costs 0.29 PE, and the 1 d-to-T_MAX tail costs 0.72 PE.

### 7.3 Anytime checkpoints

Catalog snapshots are taken at 2¹¹, 2¹², 2¹³ and 2¹⁴ propagations. Coverage, quality and P2 are computed at every checkpoint, since they need no new propagations. P1 is computed at 2¹⁴ only. The anytime curves show where methods cross, which answers `literature.md`'s novelty risk 1.

### 7.4 Compute estimate and ceiling

**Ceiling: 8 h of wall time on this machine** (Apple M4 Pro, 10 workers), which may be split across sessions.

Per-PE wall time, measured on clean M3 runs with 10 workers on a shared machine:

| Run | Effective cost |
|---|---|
| Fine probe | 4.1 ms |
| S4 | 6.6 ms |
| S1 | 9.5 ms (under heavy concurrent load) |

The estimate uses 4.1 ms as optimistic and 6.6 ms as conservative.

| Item | PE |
|---|---|
| GT map 2²⁰ | 1 048 576 |
| GT refinement, levels 1–3 (≈ 1 950 feasible map samples × 32, plus ≤ 20 000) | ≈ 82 000 |
| Tuning search: 13 configurations (MS 2, MBH 3, DE 3, T 3, T2 2) × 2 seeds × 2¹⁴ | 425 984 |
| Tuning robustness: 26 catalogs × ≈ 30 elites × (2.45 + 32 × 2.5) | ≈ 64 000 |
| Pilot search: 3 pilot seeds × 7 methods (D, MS, MBH, DE, CC, T, T2) × 2¹⁴ | 344 064 |
| Pilot robustness: 35 catalogs (5 seeds × 7) × 30 × 82.5 | ≈ 87 000 |
| Main search: 7 methods × 10 seeds × 2¹⁴ | 1 146 880 |
| Main robustness: 70 catalogs × 30 elites × (2.45 + 100 draws × 2.5) | ≈ 530 000 |
| CC\* reference (one deterministic run, ≲ 6 000) + its robustness | ≈ 15 000 |
| Elite re-checks (DOP853 1e-13 ≈ 1.5 PE) + LSODA on the 5 % sample | ≈ 5 000 |
| **Total** | **≈ 3.75 M PE** |

This is **≈ 4.3 h at 4.1 ms or ≈ 6.9 h at 6.6 ms**, within the ceiling at both rates. Per draw, the robustness cost is 0.29 for the arc plus 0.72 for each residual evaluation (at most 1 + 3 chord iterations), giving ≤ 3.2 PE and about 2.5 PE expected.

**Pre-declared reductions**, applied in this order, only if the smoke run (§1 item 3), or a seed count above 10 (§8), projects more than 8 h:
1. The GT map shrinks from 2²⁰ to 2¹⁹ (still 32 × N), saving 524 288.
2. K per held-out error set drops from 50 to 32.
3. T2 is dropped.

If the projection still exceeds 8 h, stop and revise the protocol (v2). Increasing N to 2¹⁵ doubles every search line, to ≈ 5.7 M PE (≈ 6.5–10.4 h), and would need reductions 1–2.

## 8. Seeds

| Use | Seeds |
|---|---|
| Tuning | 9101, 9102 |
| Pilot (dispersion only) | 9201, 9202, 9203 |
| Main | 1001–1010; extension 1011–1016 if §8's rule asks for more |
| GT map / refinement levels 1–3 | 20261018 / 20261019, 20261020, 20261021 |
| Error draws: T_E, H_E-nominal, H_E-wide | 7001, 7101, 7102 |
| Constraint-set generation / tuning–held-out split | 7201 / 7202 |
| Elite LSODA sample; bootstrap | 7301; 4242 |

- **Pairing.** Seed k is used by every method. Methods that read the Sobol stream (D, MS, MBH, CC, T, T2) therefore share their initial points. The paired differences in §11 are paired by seed.
- **Count rule (pilot dispersion).** The pilot runs the selected configuration of every method on the 3 pilot seeds. Together with the 2 tuning seeds of that configuration, this gives 5 seeds per method. The pilot measures P1 under T_E and P2 on the tuning sets, then computes the SD σ_d of the paired differences T − X for each baseline X and endpoint.
  - The smallest effects of interest are δ_P1 = 2 surviving solutions and δ_P2 = 0.10.
  - The power target is 0.8 at a two-sided Holm-worst-case α of 0.05/10 = 0.005.
  - n = clamp(⌈(2.807 + 0.842)² · max(σ_d/δ)²⌉, 10, 16) = clamp(⌈13.3 · max(σ_d/δ)²⌉, 10, 16).
- The rule depends only on dispersion, not on the pilot's mean differences.
- If the rule asks for more than 16 seeds, run 16 and report the achieved power. The computed n is recorded in the run manifests before the main runs start.
- With 10 seeds, the exact sign-flip test reaches p = 2/1024 ≈ 0.002 < 0.005.

## 9. Held-out conditions

### 9.1 Injection errors and the correction budget (execution-error tolerance)

**Error model.** It is applied to a catalog solution's TLI, and the three components are independent, with each draw truncated at 3σ:
- magnitude: Δv′ = Δv (1 + ε_m), with ε_m ~ N(0, σ_m²);
- in-plane pointing: α′ = α + ε_d, with ε_d ~ N(0, σ_d²);
- timing: the burn executes δt late, with δt ~ N(0, σ_t²). The CR3BP is autonomous, so this equals a departure phase θ′ = θ + (n_park − 1) δt, with n_park = 446.233 (normalized). That is 0.06799° per second of delay; the parking period is 88.05 min.

**Distributions** (open question 3):

| Set | σ_m | σ_d | σ_t | Draws K | Seed | Use |
|---|---|---|---|---|---|---|
| T_E (tuning) | 0.03 % (≈ 0.95 m/s) | 0.02° (≈ 1.1 m/s) | 0.1 s | 32 | 7001 | tuning, pilot, B calibration |
| H_E-nominal (held-out, **primary**) | same | same | same | 50 | 7101 | final evaluation only |
| H_E-wide (held-out distribution) | 0.06 % | 0.04° | 0.2 s | 50 | 7102 | final evaluation only |

- The scales are *declared design values* of order 1 m/s per component. They are not cited operational dispersions.
- The draws are common random numbers: draw k is the same (ε_m, ε_d, δt) triple for every solution of every method.

**Correction (declared budget).**
- **One** in-plane impulsive midcourse correction (MCC) at t_c = 1.0 d after TLI. The earliest lunar encounter in M3 S4 was at 2.54 d.
- It retargets the solution's own nominal perilune altitude and vacuum perigee, (h_p, h_vp) — a classical MCC that keeps the designed flyby and entry.

**Correction solver** (a chord method with a fixed Jacobian, so that the cost is bounded):
1. Per catalog solution, once:
   - J = ∂(h_p, h_vp)/∂δv at the nominal t_c state, by forward differences with 0.1 m/s steps (≈ 2.45 PE);
   - if cond(J) > 1e8, all draws fail for this solution, and this is reported.
2. Per draw:
   - propagate the perturbed TLI to t_c;
   - evaluate R(0) = (h_p, h_vp)(δv = 0) − nominal;
   - iterate δv_{k+1} = δv_k − J⁻¹ R(δv_k), at most 3 times.
3. Convergence: |R_h_p| ≤ 1 km and |R_h_vp| ≤ 1 km. Divergence: |δv| > 100 m/s.
4. Record per draw:
   - **δv_req** = |δv| at convergence, or ∞;
   - the uncorrected outcome, which gives the B = 0 survival for free.
5. **Success at budget B:** converged, δv_req ≤ B, and the corrected trajectory satisfies every §5.1 predicate and the §5.2 gate.

**Budget B (open question 2).**
- B is calibrated *before* any main run, on the D catalogs of the two tuning seeds under T_E only. It is the smallest B ∈ {1, 3, 10, 30} m/s for which the median per-solution survival falls in [0.2, 0.8], or the grid value closest to 0.5 if none does.
- *Reason:* to avoid a floor or ceiling that removes discriminative power (`literature.md`, novelty risk 3). The calibration uses no treatment data.
- B is then fixed. Because δv_req is stored, survival curves s(B) over the whole grid are reported as secondary results at no extra cost.

### 9.2 Changed constraints (changed-constraint availability)

A constraint set C = (h_lo, h_hi, T′, γ_c, w) adds requirements to §5.1:
- h_p ∈ [h_lo, h_hi];
- t_ret ≤ T′;
- if a corridor is present, |γ_S − γ_c| ≤ w.

Availability is computed from stored catalog records, with no propagation.

**Families.** 32 sets each, generated by scrambled Sobol (seed 7201) over the stated ranges:

| Family | Varies | Ranges | Fixed |
|---|---|---|---|
| F1 geometry | perilune sub-band and maximum TOF | log₁₀ h_lo ∈ [2, 4]; width ∈ [0.5, 2] dex (h_hi = min(h_lo·10^width, 30 000)); T′ ∈ [6.5, 10] d | no corridor |
| F2 entry corridor | corridor | γ_c ∈ [−8°, −5°]; w ∈ [0.5°, 2°] | nominal band, T′ = 10 d |
| F3 combined | all of the above | as F1 and F2 | — |

- **Split.** Within each family, a seeded permutation (seed 7202) assigns ⅓ of the sets (10) to *tuning* and ⅔ (22) to *held-out*. That gives 30 tuning and 66 held-out sets.
- Held-out sets are evaluated only in the final analysis.
- A held-out set counts in P2 only if it is **satisfiable**: some feasible propagation in G's sources (§5.3 a–d) satisfies it. Unsatisfiable sets are dropped and their number is reported.

## 10. Metrics

Per run (method × seed), on its catalog (§5.5, after §5.6):

| Metric | Definition | Unit |
|---|---|---|
| **P1** | Σ_i s_i, where s_i is the fraction of the 50 H_E-nominal draws that succeed at budget B for solution i | solutions |
| **P2** | fraction of satisfiable held-out constraint sets for which ≥ 1 catalog solution satisfies the set (no correction) | — |
| Coverage | \|catalog cells ∩ G\| / \|G\| (10 × 10; 20 × 20 secondary) | — |
| Quality | mean and median e over elites; QD-score Σ_cells (90° − e) | deg |
| Per-solution robustness | mean s_i; P1 at B = 0; s(B) curves; P1 under H_E-wide | — |
| Worst case | min_i s_i; the 10th percentile of e; the worst seed's P1 and P2 | — |
| Feasibility yield | feasible propagations / N; first-feasible index | — |
| Diversity beyond cells | number of distinct (lunar passage sense, entry sense) classes in the catalog | count |
| P2 by family | F1, F2 and F3 separately | — |
| Cost | propagations used (≤ N); RHS evaluations; wall time; numerical-failure count | count, h |

P1 deliberately combines coverage and per-solution robustness, because D16 asks how many robust solutions a catalog *contains*. The mean s_i separates the two.

## 11. Statistical analysis

- **Summaries.** For every method and metric, across main seeds: mean, median, SD, min and max, plus a per-seed table. Anytime curves show mean ± 95 % bootstrap CI at each checkpoint.
- **Primary comparisons.** Ten contrasts: T − X for X ∈ {D, MS, MBH, DE, CC} on P1 and P2, paired by seed.
  - Effect: the mean paired difference, with a 95 % percentile bootstrap CI over seeds (10 000 resamples, seed 4242).
  - Test: an exact two-sided sign-flip permutation test on the paired differences (2^n flips).
  - Holm–Bonferroni across the ten contrasts at a family-wise α = 0.05.
- **Secondary analyses** (not in the Holm family, reported as estimates with CIs):
  - T2 vs every arm, and T2 vs T;
  - coverage and quality contrasts;
  - H_E-wide;
  - s(B);
  - the 20 × 20 sensitivity analysis;
  - P2 by family;
  - CC\* vs every arm (unmatched, labelled as such);
  - H2: the Spearman correlation of s_i with the elite-selection margin within T, pooled over seeds with a seed-stratified permutation test, and mean s_i of T vs D.
- **Required reporting.** Failed and diverged local solves, numerical failures, elites rejected by the re-check, unsatisfiable sets, and the achieved power when n hits its cap.
- **Not done.** No exclusion of seeds, no post-hoc metrics promoted to primary, and no re-tuning after held-out evaluation.

## 12. Stop / kill criteria

| # | Condition | Action |
|---|---|---|
| K0 | Any §1 precondition unmet | Do not freeze. In particular, if Du & Li already answer D16, revise the question. |
| K1 | The GT 2¹⁷ prefix does not reproduce `gravity-free-return-map-3d-sobol17-v1` | Stop; investigate the numerics before any comparison. |
| K2 | > 1 % of elites fail the §5.6 re-check, or > 1 % of any method's propagations fail the §5.2 gate | Stop; investigate. Numerics come before comparison (`research-methodology.md`). |
| K3 | After B calibration, the pilot median s_i of *every* method is < 0.05 or > 0.95 under T_E | P1 cannot discriminate; revise to v2 before the main runs. |
| K4 | The projected wall time exceeds 8 h after the §7.4 reductions | Stop; revise the scope (v2). |
| K5 | **D (or CC) ≥ T on P1, P2 and coverage (point estimates) at equal budget** | Publish the null ("a dense map / a continuation catalog dominates at equal budget"). Stop adding methods. No QD variants are added to rescue the result (D07; roadmap M4 go/no-go). |
| K6 | H1 not supported for any baseline, and T not better than D on either endpoint | Publish as a negative result with the full metric set. |
| K7 | G grows by > 10 % between the 2¹⁹ prefix and all sources | Report coverage as an upper bound and as secondary only; P1 and P2 are unaffected. |

## 13. Reproduction

Placeholders, to be implemented and filled in before freezing. Each command writes a new run directory and never overwrites one.

```text
# configuration: configs/gravity-atlas/m4-protocol-v1.toml   (every constant of §4–§11; hashed into each manifest)
just gravity-m4-groundtruth            # §5.3 (a)–(c); run gravity-m4-groundtruth-v1
just gravity-m4-tune                   # §6 tuning grid, seeds 9101–9102; runs gravity-m4-tune-<method>-<cfg>-s<seed>-v1
just gravity-m4-calibrate-b            # §9.1 B calibration from the D tuning catalogs
just gravity-m4-pilot                  # §8 pilot, seeds 9201–9203; writes the seed count n
just gravity-m4-run <method> <seed>    # main runs; gravity-m4-<method>-s<seed>-v1
just gravity-m4-robustness <run_id>    # §9.1 on H_E-nominal and H_E-wide, plus §5.6 re-checks
just gravity-m4-analysis               # §5.3 (d), §10–§11; run gravity-m4-analysis-v1
```

Every run manifest records the protocol version, config hash, commit, `git_dirty` (which must be false for evidence), environment, seeds and budget counters. Evidence comes only from clean commits.

## 14. Risks to validity (known now)

- **Strawman risk for T.** Plain MAP-Elites does not refine e locally, while MS, MBH and CC solve r₁ = 0 directly. T may lose on quality, and therefore on F2/F3 availability, for that reason alone. T2 (open question 5) measures this effect.
- **CC may dominate.** Three equations in three unknowns per cell make continuation very cheap: a full 10 × 10 sweep needs on the order of 10³ propagations once seeded. This is a real, publishable outcome (K5), not a flaw.
- **The error scales are declared, not sourced.** P1's absolute values depend on them. Comparisons between methods are less sensitive, because the draws are common random numbers.
- **The chord MCC solver can fail where a full Newton solve would succeed.** It undercounts survival in the same way for all methods; this is a declared limitation.
- **The binning and saturation numbers come from a one-off analysis** of the M3 S4 table, not from a recipe. The GT run recomputes them.
- **The ground truth includes the methods' own finds**, so coverage is relative to the union of everything observed.

## 15. Open questions for the user before freezing

1. **Run budget N.** *Recommend N = 2¹⁴.* A dense map then covers about 16 of 27 known cells (unsaturated), and the total is ≈ 4.3–6.9 h. N = 2¹⁵ would cost ≈ 6.5–10.4 h and need reductions.
2. **Correction budget B.** *Recommend calibrating B on the dense-map tuning catalogs* to a median survival in [0.2, 0.8], from {1, 3, 10, 30} m/s, with s(B) reported for the whole grid. The alternative is a fixed B = 10 m/s declared now, which risks a floor or ceiling effect.
3. **Injection-error scales.** *Recommend the declared σ = (0.03 %, 0.02°, 0.1 s)* with a 2× held-out set, labelled as design values. The alternative is to first find a citable TLI dispersion source, which delays the freeze.
4. **Continuation baseline seeding.** *Recommend the budget-matched CC* (in-budget Sobol seeding) as the compared baseline, plus the oracle-seeded CC\* (from M3 c1 and c2) as an unmatched reference outside the tests.
5. **Second treatment arm T2** (MAP-Elites with a local corridor emitter). *Recommend including it as a pre-registered secondary arm* outside the Holm family (cost ≈ 0.38 M PE of the total). Without it, a negative T result cannot be told apart from "QD without local refinement".
6. **Search box.** *Recommend keeping the full M3 box* (θ over the full circle). Narrowing it to the region M3 found would leak map knowledge into every method and favour the dense map least.
