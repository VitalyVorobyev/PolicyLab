# Model Citizens C2: citizen-model validation protocol

Status: draft

Version v1-draft, written 2026-10-20 for the study owner's review. Only the owner freezes it: the freeze changes the word `draft` above to `frozen` (the single status line the held-out guard `phasespace.citizens.data.load_heldout` reads), and fills in §1. Until then no held-out human game is read. A frozen protocol is never edited in place; changes create v2 with a changelog entry and a reason, and every result states the version it used.

**Freeze checklist** (recorded in §1 and in a new D-entry, written only on the owner's instruction):
1. Every pre-freeze task in §10.1 is done and its output recorded (smoke costs, B4 pilot, memorisation probe, `players_kept_active` check, prompt texts).
2. All constants of this text are transcribed into the C2 configuration module; a test checks that Appendix A and the module agree; `config_sha256` is printed.
3. The open questions in §13 are answered and integrated.
4. The freeze commit changes only the status word and §1, on a clean tree.

## 1. Header

| | |
|---|---|
| Protocol version | v1 (draft) |
| Frozen date | — (owner) |
| Git commit at freeze | — |
| Config hash at freeze | — (`config_sha256` of the C2 configuration module) |
| Inputs | D31, D32, D33 (`docs/decisions.md`); `README.md` (spec v0); `literature.md`; C1 audit run `citizens-c1-data-audit-v1` |
| Scope | Milestone C2 and gate G2 only. The planner comparison (C4) has its own `protocol.md`. |

**What was known when this text was written (honest disclosure).** The authors have read Koster et al. 2025, including its *published aggregate results* for the held-out mechanisms (e.g. Exp 3 M2 > Interpolating surplus z = 2.35; Interpolating < M2 Gini z = 5.64) and its figures. The protocol is frozen with that knowledge but **before any computation of ours on held-out human rows**. Our only contact with held-out rows so far is C1's identity checks (pool transition, Interpolating offer reproduction), which return residual statistics and game counts only (one finding: 23 M1 Exp 1 rows whose recorded next pool is 0.01 below the full rule; §10.3). Training aggregates (four Exp 1 z-values, reproduced in C1) have been seen.

## 2. Question and hypothesis

**Question.** Which classes of behavioural model, fitted only on human play under the Equal, Mixed and Proportional rules, predict human play under mechanisms they never saw (Interpolating, M1, M2)?

**Hypothesis H-V (per model class and held-out regime).** The model reproduces the human group-level outcomes and beats a marginal-frequency baseline in held-out per-decision likelihood, by the criterion of §8. It is refuted for a (model, regime) pair when the criterion fails. **Control expectation:** the payoff-maximising control B0 fails every regime; if it passes one, the criterion is declared too weak for that regime (§9).

## 3. Model (game and data)

- **Game.** As in `README.md` and D33, implemented in `phasespace.citizens.game` / `mechanisms`: 4 players, R₀ = 200 = cap, R′ = max(0.01, min(200, R − Σe + 1.4Σc)); depletion when R < 1; a player with e < 1 is excluded and reciprocates 0; human actions are integers c ∈ {0, …, ⌊e⌋}. Exps 1–3: 40 rounds.
- **Deployed offer rule (D33).** Simulated baseline and Interpolating offers use the deployed convention `recorded_offers`: w quantised to one decimal for Interpolating (w = round((R/200)²², 1)), 0.001 coins withheld, offers rounded to 4 decimals. The equal split applies in round 1 and whenever Σc_{t−1} = 0.
- **Data.** `sustainable_behavior.csv` (C1-verified checksums), parsed by `phasespace.citizens.data`. Training rows via `load_training`; held-out rows only via `load_heldout` after the freeze (§10.2).
- **Training set (human, 280 games, 200 independent units).** Exp 1 Equal 40, Mixed 40, Proportional 40; Exp 2 Proportional 40; Exp 4 Proportional 40 groups × 3 consecutive games (25–49 rounds). An Exp 4 group (its 3 games) is one unit.
- **BC rows.** The simulated BC 1/BC 2 cohorts (3,584 games, 7 × 512) are not held out. They are used only (a) to fit the M1/M2 mechanism surrogates from BC 1 M1 and BC 2 M2 mechanism rows (§6), and (b) as a visual reference. They never enter a citizen model or a criterion.
- **Held-out human games** (counts from C1; content unseen): Interpolating Exp 2 = 40, Exp 3 = 80; M1 Exp 1 = 40, Exp 2 = 40; M2 Exp 3 = 80, Exp 4 = 40 groups × 3 games.

## 4. Candidate parameterization: the common likelihood

Every non-LLM model emits, for each decision of player i in round t with e = e_{i,t} ≥ 1 and n = ⌊e⌋, a pmf over the integers c ∈ {0, …, n}, conditioned only on what the player has observed (current four offers with own first, current pool, all previous offers and reciprocations of the game; rotated focal-first as in the CSV). **Forced-zero decisions (e < 1) are excluded** from every likelihood (they include all post-depletion rows).

**Bin-to-integer map** (used by B1, B3, B23 and the baselines). A model that outputs probabilities p_b over fraction bins assigns integer c to bin b(c) and sets P(c) = p_{b(c)} / |{c′ ≤ n : b(c′) = b(c)}|, renormalised over the bins that contain at least one integer. Two binnings:
- *10 floor bins* (B1, M0, P0): b(c) = min(9, ⌊10c/e⌋), i.e. bin k = [k/10, (k+1)/10), the last bin closed.
- *11 level bins* (B3, B23): ℓ(c) = round-half-up(10c/e) ∈ {0, …, 10}, i.e. levels a_j = j/10.
Closed-loop sampling draws c from the same integer pmf, so likelihood and simulation use one object.

**Per-decision score.** ℓ_m = mean log P_m(c_observed) in nats over all non-forced decisions of a regime; Δℓ_m = ℓ_m − ℓ_M0.

## 5. Feasibility of a simulated game

A simulated game is valid if every round satisfies: e ≥ 0 and Σe ≤ R (+1e-9); integer 0 ≤ c ≤ ⌊e⌋ and c = 0 when e < 1; the pool update of §3; finite values throughout. An invalid game is a code bug: the run stops (K2), it is never silently dropped.

## 6. Methods

All fitting uses the training **fit split** only (§6.1); no model is refitted after the unlock (§11).

### 6.1 Split and model selection
- **Units:** the 200 training units, stratified by (mechanism, cohort): Exp 1 Equal, Exp 1 Mixed, Exp 1 Proportional, Exp 2 Proportional, Exp 4 sets (40 units each). Fit/validation = 32/8 per stratum (160/40 units; 224/56 games), drawn with `numpy.random.default_rng(20261020)` permuting each stratum's units sorted by `launch_id` (Exp 4: by the Game 1 `launch_id`).
- The validation split is used only for early stopping (B1, surrogates) and reporting. Final models are those trained on the fit split; there is **no refit on fit + validation**.
- **BC split for surrogates:** BC 1 M1 games and BC 2 M2 games each 410/102 by game, `default_rng(20261021)`.

### 6.2 Reference baselines (likelihood only, not model classes)
- **M0 marginal:** the fit-split frequencies of the 10 floor bins over non-forced decisions, +0.5 per bin, normalised; mapped to integers by §4. Gating reference for Δℓ. Also simulated closed-loop as a no-dynamics control (**M0-sim**, non-gating; §9).
- **P0 persistence** (diagnostic): with probability q the player's previous floor bin (if t > 0 and e_{i,t−1} ≥ 1), else M0; q by MLE on the fit split.

### 6.3 Model classes

**B0, rational negative control** (no parameters). Closed loop: c = 0 always (the myopic payoff maximiser). Likelihood: P(c) = 0.95·1[c = 0] + 0.05/(n+1).

**B1, GRU clone** (Koster et al. 2025, Table S1, BC1 scale; no hyperparameter search).
- Inputs (9, all /200): the four current offers (own first), the four previous reciprocations (own first; zeros in round 1 as recorded), the pool.
- Network: FC encoder (16, 32) → GRU 64 → FC projection (32, 16) → linear 10 logits (10 floor bins); tanh activations; one sequence per (game, player).
- Loss: cross-entropy of the observed floor bin on non-forced decisions (forced steps are fed as inputs, not scored). BPTT over the whole game; Adam, learning rate 5e-4 constant, batch 64 sequences, no regularisation.
- Early stopping: validation NLL every 250 updates, patience 8 evaluations, at most 20,000 updates; the best-validation checkpoint is kept.
- **Ensemble N = 4** members, seeds 101–104 (the paper used 4 checkpoints). Likelihood: the mean of the members' integer pmfs. Closed loop: each simulated player draws one member uniformly with replacement at game start (as the paper's planner did).

**B2, parametric conditional cooperation** (12 parameters; MLE). Features of decision (i, t):
x₁ = min(4, 4e_{i,t}/R_t) (own offer relative to the equal share); x₂ = c_{i,t−1}/e_{i,t−1} (own previous fraction; 0 if t = 0 or e_{i,t−1} < 1); x₃ = Σ_{j≠i}c_{j,t−1} / Σ_{j≠i}e_{j,t−1} (others' previous fraction; 0 if t = 0 or the denominator < 1); x₄ = R_t/200; x₅ = 1[t = 0]; x₆ = 1[t > 0 and e_{i,t−1} < 1].
- Zero hurdle: logit π₀ = γ₀ + γ₁x₂ + γ₂x₃ (3). Full-return spike: logit π_F = δ₀ (1). Interior mean: logit μ = β₀ + Σ_{k=1..6} β_k x_k (7). Precision φ = exp(η) (1).
- pmf: P(0) = π₀. n = 1: P(1) = 1 − π₀. n ≥ 2: P(n) = (1 − π₀)π_F; for 1 ≤ c ≤ n − 1, P(c) = (1 − π₀)(1 − π_F)·BetaBinomial(c − 1; n − 2, μφ, (1 − μ)φ).
- β₃ is the conditional-cooperation slope (Fischbacher & Gächter 2010 under-matching).
- Fit: L-BFGS-B on the fit-split NLL from 10 starts (zeros + N(0, 0.5²) noise, seeds 201–210); best fit-split NLL is kept.

**B3, learning model: EWA with realised forward reinforcement** (6 parameters; MLE; Camerer & Ho 1999). Actions are the 11 levels a_j = j/10.
- Initial attractions A_j(0) = log(p⁰_j)/λ, where p⁰ is the fit-split round-1 level distribution (+0.5 smoothing); this is data-derived, not a free parameter.
- After a non-forced round t with chosen level j* = ℓ(c_{i,t}), at the start of round t + 1 (when e_{i,t+1} is seen): N(t+1) = ρN(t) + 1, N(0) = N₀, and
  A_j(t+1) = [φ N(t) A_j(t) + (δ + (1 − δ)1[j = j*])·e_{i,t}(1 − a_j)/50 + 1[j = j*]·θ·e_{i,t+1}/50] / N(t+1).
  The first payoff term is the myopic surplus (foregone payoffs weighted by δ); the θ term reinforces the chosen level by the next offer received, the institution's realised response. Forced rounds do not update.
- Choice: p_j = softmax_j(λA_j(t)); integer pmf by the 11-level map.
- Bounds: φ, ρ, δ ∈ [0, 1]; λ ∈ [0.01, 50]; N₀ ∈ [0, 20]; θ ∈ [0, 5]. L-BFGS-B, 10 starts uniform in the bounds (seeds 301–310).

**B23, hybrid (pre-declared fallback, D32)** (7 parameters). B3's attractions plus a conditional-cooperation pull: p_j ∝ exp(λA_j(t) − κ(a_j − x₃)²·1[t > 0]), κ ∈ [0, 50]. Fitted and evaluated **always**, before the unlock, with seeds 311–320, so that its use does not depend on held-out outcomes. Its role in C4 is fixed in §9.

**B4, LLM citizens through headless Claude** (closed loop only; no likelihood).
- *Runner.* One fresh `claude -p` call per player per round (160 calls per 40-round game), the four players of a round in parallel, `--output-format json`, a pinned `--model` value; the model id reported in each JSON response is archived. Each call runs in an empty temporary directory with tools disabled and user/project memory and settings excluded; the exact CLI flags are fixed in the configuration at freeze, and the pilot verifies isolation by asking the model to list any instructions in its context.
- *Prompt.* A neutral rules text written from the paper's player instructions (coins, offers, keeping vs returning, the pool and its 1.4 growth, exclusion below 1 coin, round length not disclosed). It never names the paper, DeepMind, a "manager", "AI" or "RL agent". Each call carries the rules, the player's current observation (four offers, own first; pool) and a table of all previous rounds as that player saw them (offers, everyone's reciprocations, pool, own coins). Answer format: a JSON object `{"return": <integer>}`.
- *Parsing.* Valid iff an integer in {0, …, ⌊e⌋}. On failure, up to 2 retries with a one-line format reminder; timeout 120 s per attempt. A game with an unrecoverable failure is discarded and replaced (mirroring the human dropout exclusion); discards are counted. If > 10 % of a regime's games are discarded, B4 is non-gating for that regime.
- *Seeding.* LLM sampling cannot be seeded through the CLI; runs are stochastic and archived (every prompt, raw response, retry, timing and model id). Game order and type draws are seeded (7001).
- **B4-plain:** the neutral prompt.
- **B4-calibrated** (after Manning & Horton 2025; Jackson et al. 2026, simplified): K = 3 pre-written type paragraphs appended to the neutral prompt — T1 "keeps most coins for themselves", T2 "returns about as much as the others returned last round", T3 "returns generously to keep the pool healthy" (exact texts fixed at freeze). Calibration on training mechanisms only: 60 Proportional 40-round games, 4 per type composition (15 compositions of 4 players over 3 types). For weights w on the simplex grid of step 0.1 (66 points), predicted group means are Σ_comp Multinomial(comp; 4, w)·mean(comp). Chosen w minimises Σ_m ((pred_m − h_m)/s_m)² over the four metrics of §7, with h_m, s_m the mean and between-group SD of the fit-split Exp 1 + Exp 2 Proportional games (64 games); ties go to the w nearest uniform. In evaluation, each player's type is drawn i.i.d. from w.
- *Memorisation probe* (pre-freeze, logged): 5 fresh calls each of (a) "What do you know about Koster et al. 2025, Nature Communications, on deep RL in a common-pool resource game?" and (b) the neutral rules followed by "Do you recognise this experiment? Name its source and any results you know." Consequence: if any answer states k = 22, the interpolating rule, or a published result, all B4 results carry the label "contamination risk: high". Not a stop rule.

### 6.4 Mechanism surrogates for M1 and M2
M1/M2 weights are not released, so closed-loop simulation needs a surrogate.
- **Fitting data:** BC 1 M1 rows (for M1) and BC 2 M2 rows (for M2) only — mechanism observation → offers. **Never human rows.**
- **Architecture** (permutation-equivariant, recurrent, like the paper's GNN + GRU): per player, inputs (e_{i,t−1}, c_{i,t−1}, R_t, mean_j e_{j,t−1}, mean_j c_{j,t−1}), all /200 (zeros in round 1) → shared FC 32 (tanh) → shared GRU 32 per player → FC 32 → 1 player logit; pool logit from the mean of the four hidden states → FC 32 → 1. Offers e = R·softmax(5 logits)[players].
- **Loss:** cross-entropy between the recorded share vector (e_i/R, 1 − Σe/R) and the predicted shares, over rounds with R ≥ 1. Adam 1e-3, batch 64 games, validation offer MAE every 250 updates, patience 8, at most 20,000 updates; seeds 401–403 (M1) and 411–413 (M2); the best BC-validation seed is kept.
- **Pre-unlock check (BC validation games):** the acceptance thresholds below must hold. If they fail, one pre-declared capacity step (FC/GRU 64, seeds 404–406 / 414–416); if that also fails, the surrogate is rejected before the unlock.
- **Acceptance on held-out human mechanism rows** (after the unlock; offers predicted teacher-forced from the recorded observation sequence of each human game; rounds with R ≥ 1; per (mechanism, experiment)):
  1. median |ê − e| ≤ 1.0 coin, **and**
  2. 90th percentile |ê − e| ≤ 5.0 coins, **and**
  3. exclusion agreement 1[ê < 1] = 1[e < 1] on ≥ 90 % of player-rounds.
  *Justification.* Humans act in whole coins, so a 1-coin median error is below the action resolution; 5 coins is 10 % of the full-pool equal share (50), so large misallocations such as excluding the wrong player (errors of order R/4) must be rare; exclusion is M1/M2's signature behaviour and drives depletion and players-active.
- **If a surrogate fails acceptance** in any experiment of its mechanism: that mechanism's closed-loop results are reported as diagnostic only, and its validation rests on the one-step likelihood alone (tier "one-step", §8). The surrogate is never refitted after the unlock.

## 7. Budget, seeds and simulation sizes

- **Non-LLM compute ceiling:** 12 h wall-clock on the M4 Pro for all fitting, surrogates and simulation together (B1 ≤ 4 × 20,000 updates, B2/B3/B23 10 starts each, surrogates ≤ 2 × 6 × 20,000 updates). The smoke run (§10.1) measures the actual cost; the measured numbers are recorded here before freezing.
- **LLM call ceiling:** 55,000 calls including retries; wall-clock ≤ 14 days. Planned: pilot 480; probe 10; calibration 9,600; evaluation 2 variants × 3 mechanisms × 40 games × 160 = 38,400 (≈ 48,500). B4 is not run on M2 Exp 4 (3-game sets of variable length), and is run on M1/M2 only if the mechanism's surrogate passed the pre-unlock BC check.
- **Pre-declared reductions** (applied in order only if the pilot's measured throughput projects past a ceiling): (1) calibration 4 → 2 games per composition; (2) evaluation 40 → 30 games per (variant, mechanism); (3) M1/M2 evaluation 30 → 20 games. Below 20 games a regime's B4 result is exploratory and non-gating.
- **Simulation sizes (non-LLM):** S = 1,000 simulated games per (model, mechanism) for Interpolating, M1, M2 and for each training mechanism (in-sample check, §8); for M2 Exp 4, 25 simulated 3-game sets per human set (1,000 sets), each game's length equal to the matched human game's recorded length (lengths are an exogenous lottery).
- **Seeds:** split 20261020; BC split 20261021; B1 101–104; B2 201–210; B3 301–310; B23 311–320; surrogates 401–406, 411–416; simulation seed = 50000 + 100·m + r with m ∈ {B0 0, M0-sim 1, B1 2, B2 3, B3 4, B23 5} and r ∈ {Interpolating 1, M1 2, M2 3, M2 Exp 4 4, Equal 5, Mixed 6, Proportional 7}; bootstrap 9001; B4 game order and type draws 7001. The MC error of a simulated mean with S = 1,000 is ≈ 3 % of the between-game SD, an order below the human CI half-widths expected with 80–120 groups.

## 8. Held-out regimes, metrics and the validation criterion

**Metrics** (notebook definitions, `phasespace.citizens.metrics`; per game; human values from all recorded rows): (1) surplus per player-round; (2) Gini of the players' mean rewards (the paper's "Gini", 0.5·mean|xᵢ − xⱼ|/mean x; not the CSV's per-round `offer_gini`); (3) depletion round (first round with R < 1, 1-based, else 40); (4) players active at the end (`players_kept_active` in the last round; simulated as #{i : e_{i,T} ≥ 1}, subject to the §10.1 check).

**Gating regimes** (one per held-out mechanism; human groups pooled over that mechanism's 40-round experiments, because a model's simulation of a mechanism is the same for every cohort):

| Regime | Human groups | Mechanism in simulation |
|---|---|---|
| R-I | Interpolating Exp 2 (40) + Exp 3 (80) = 120 | exact deployed rule (§3) |
| R-M1 | M1 Exp 1 (40) + Exp 2 (40) = 80 | M1 surrogate |
| R-M2 | M2 Exp 3 (80) | M2 surrogate |

**Diagnostic regimes** (reported, never gating): each (mechanism, experiment) separately; **M2 Exp 4** (40 groups × 3 games; unit = group, group metric = mean over its 3 games). M2 Exp 4 is non-gating because our models carry no state across games, Exp 4 changed the incentive (whole-episode pay) and length, and the notebook's depletion round defaults to 40 for shorter games.

**Human reference.** For each regime and metric, the mean over groups and its 95 % percentile bootstrap CI: B = 10,000 resamples of groups (stratified by experiment in pooled regimes), seed 9001.

**Criterion for model m on regime r** ("validated (closed-loop)"):
- **(a) Outcomes:** for ≥ 3 of the 4 metrics, the simulated mean over S games lies in [lo − τ_k, hi + τ_k], the human CI widened by a resolution tolerance τ = (surplus 0.1 coins, Gini 0.005, depletion 0.5 rounds, players active 0.1) — this matters only when a human CI is degenerate (e.g. no group depleted).
- **(b) Likelihood** (non-LLM models): Δℓ_m > 0 on the regime's non-forced decisions (teacher-forced, every game's recorded history), with the 95 % group-bootstrap CI lower bound of Δℓ_m > 0 (B = 10,000, seed 9001).
- B4: (a) only, with its own simulated games (≈ 40 per regime); the B4 simulated mean's own bootstrap CI is reported.
- If the regime's surrogate failed acceptance (§6.4): (a) is diagnostic and the pair is "validated (one-step)" iff (b) holds.

**In-sample check (pre-unlock, non-gating):** the same criterion on the training mechanisms (Equal, Mixed, Proportional; human groups of the whole training set, 40-round games). Reported to show whether a model fits the data it was trained on.

**Distributional diagnostics (non-gating):** for each metric, the Wasserstein-1 distance between the simulated and human game-level distributions, with a human-vs-human reference (W1 between random halves of the human groups, 1,000 splits, seed 9001); per-round mean reciprocation fraction and pool trajectories; reciprocation fraction vs offer share and vs pool ("fingerprint" plots); Δℓ vs P0; per-experiment results.

## 9. G2 decision rule and pre-declared outcomes

**Classes counted for G2:** B1, B2, B3, B23, B4 (either variant). B0 and M0 never count. **Trainable** classes (usable as PPO environments in C3/C4): B1, B2, B3, B23.

**G2 passes** if ≥ 2 classes, at least one of them not B1, are validated (closed-loop) on at least one gating regime each, B0 fails every gating regime, and ≥ 2 of the validated classes are trainable. (Whether "one-step" validations count is open question O3.)

**Criterion-strength checks.** If **B0 passes** a gating regime: the criterion is declared too weak for that regime; no "validated" label from that regime counts for G2; this is reported and the owner is asked. Any stronger criterion would be post hoc (v2, labelled exploratory). If **M0-sim passes (a)** on a regime, the outcomes criterion alone does not discriminate there and validation in that regime rests on (b); reported, no other consequence.

| Pattern after evaluation | Conclusion reported | Next step |
|---|---|---|
| ≥ 2 trainable classes validated (≥ 1 not B1), B0 fails | Several model classes transfer to new institutions | G2 passes; C3/C4 use exactly the validated trainable classes; B4 (if validated) is an evaluation population |
| Only B1 validated | Only the flexible clone transfers; the cross-model question collapses | G2 fails; report; ask the owner |
| Exactly one non-B1 trainable class validated | One validated model; no cross-model test | G2 fails; report; ask the owner |
| B1 and B4 only | Two validated classes, one trainable | G2 fails as defined; report; ask the owner (option: C4 reduced to B1 planners evaluated on B4) |
| B23 validated, B2 and B3 not | Hybrid needed for transfer | Counts as a non-B1 trainable class |
| No class validated anywhere | Publishable null: no model fitted on baselines predicts behaviour under new institutions | Report; ask the owner before continuing |
| B0 passes any gating regime | The criterion is too weak there | Report; ask the owner |
| A surrogate fails acceptance | M1/M2 closed-loop results are diagnostic | Validation there is one-step only (O3) |
| Models validated only on R-I | Transfer shown for the distilled rule, not for RL mechanisms | Report as such; G2 decided by the rule above |

## 10. Procedure

### 10.1 Before the freeze (training data only)
1. Implement the C2 code and configuration; `just verify` passes.
2. Smoke run: one B1 member for 2,000 updates, one start each of B2/B3/B23, one surrogate seed for 2,000 updates, 100 simulated games; record measured costs in §7.
3. Check on training human rows that `players_kept_active` equals #{i : e_i ≥ 1} in every row; if not, the owner fixes the simulated definition before freezing.
4. B4: write the prompt and type texts; pilot 2–3 games under Proportional (seconds per call, rate limits, parse failures); isolation probe; memorisation probe. Prompt changes are allowed only here.

### 10.2 After the freeze, before the unlock
5. Fit M0, P0, B1, B2, B3, B23 and both surrogates (with their BC checks); in-sample checks; B4 calibration.
6. Write the **fitted-model run** (immutable; checksums of every fitted parameter file, the chosen B4 weights and the surrogate verdicts), then commit it. Everything evaluated later is loaded from this run by checksum.

### 10.3 Unlock and G1b
7. `just citizens-c2-evaluate --confirm-frozen` calls `load_heldout(confirm_frozen=True)`, which refuses unless this file reads `frozen`, is tracked and clean; every held-out-derived manifest records the returned protocol SHA-256 and `config_sha256`.
8. **G1b, first:** (a) reproduce the remaining published Exp 1–4 rank-sum z-values that involve held-out mechanisms to |Δz| ≤ 0.005; (b) root-cause the 23 M1 Exp 1 rows whose recorded next pool is 0.01 below the full rule, and count human games whose depletion round would differ under the full rule. *Consequences:* a z-value that cannot be reproduced makes that mechanism's human reference unverified: its regime does not count for G2 and the owner is asked. Human metrics always use the recorded rows. The simulator keeps the full rule; whatever the anomaly turns out to be, no model or surrogate is refitted, and it is reported as a limitation.
9. Surrogate acceptance (§6.4); one-step likelihoods; closed-loop simulations; B4 held-out games; criterion; G2 (§9).

## 11. Rules after the unlock (forbidden actions)

- No refit, re-initialisation, re-selection of ensemble members or checkpoints, change of hyperparameters, seeds, S, B, thresholds, tolerances, regimes or prompts.
- No extra B4 games beyond the pre-declared count, and no re-running of a regime with new seeds or new LLM samples to change a verdict. Discarded B4 games are replaced only under §6.3's rule.
- Held-out rows are read only through `load_heldout`.
- Any analysis not specified here is labelled **exploratory** wherever it appears, and cannot change a "validated" label or G2.

**Reported regardless of outcome:** the full model × regime × metric table (simulated means, human CIs, pass/fail), Δℓ with CIs for every non-LLM model and P0, B0 and M0-sim results, in-sample checks, surrogate fidelity (BC and human), G1b outcome, B4 parse/discard rates, call counts, probe answers and calibration weights, the distributional diagnostics, and per-experiment results. The run status follows `docs/data-contract.md`: the evaluation run is `validated` only if its recorded checks (G1b, feasibility of every simulated game, checksum match with the fitted-model run) pass; otherwise `computed_unvalidated`.

## 12. Stop / kill criteria

| # | Condition | Action |
|---|---|---|
| K1 | B1's validation NLL does not beat M0 on the validation split | Report a fitting failure; B1 is still evaluated as fitted; ask the owner before C3 |
| K2 | Any simulated game violates §5 | Stop; fix the bug before any further evaluation (no verdict changes) |
| K3 | Non-LLM work projects past 12 h, or LLM work past its ceiling after the reductions | Stop that arm; report; ask the owner |
| K4 | G1b fails for all three mechanisms | Stop; reconcile the data before any validation claim |
| K5 | G2 outcomes as in §9 | As tabled; either outcome is reported |

## 13. Resolved questions (2026-10-20, before freezing)

These were the draft's open questions. The owner chose the B4 model; the rest are pre-freeze choices made by the lead, which the owner reviews at the freeze.

- **O1.** The gating regimes are pooled per mechanism. Results per (mechanism, experiment) are reported as diagnostics.
- **O2.** M2 Exp 4 stays non-gating. Its instructions differ, its game length is geometric, and it was played as 3-game sets.
- **O3.** A pair validated on the one-step likelihood only, after its surrogate failed, is labelled "validated (one-step)". It counts toward G2 only together with a closed-loop validation of the same model on another regime.
- **O4.** G2 needs two classes that are both validated on at least one common regime, and both must be trainable (non-LLM), because planners must be trained against them. B4 can never satisfy G2 alone.
- **O5.** The surrogate thresholds (1 / 5 coins, 90 % exclusion agreement) and the tolerances τ are accepted as drafted.
- **O6.** B4 uses `claude-haiku-5-5`, the owner's choice of 2026-10-20, through headless `claude -p`. The call ceiling of 55,000 is accepted. The calls run with a neutral working directory and no project context, and the exact CLI flags are recorded per run. Calibration uses Proportional only, with K = 3 types.
- **O7.** "Gini" means exactly the metric that reproduced the published Exp 1 Gini z-values in C1 (`phasespace.citizens.metrics`). The CSV column `offer_gini` is not used.
- **O8.** The likelihood criterion requires the CI lower bound to be above 0, as drafted.
- **Pre-freeze check.** On training rows, `players_kept_active` must equal the number of players offered at least 1 coin. This is verified by test before the freeze.

## Appendix A. Key constants

```text
split_seed=20261020 bc_split_seed=20261021 fit/val=32/8 units per stratum (160/40)
B0 eps=0.05 | M0 smoothing=0.5 | bins: 10 floor (B1,M0,P0), 11 levels (B3,B23)
B1 enc=(16,32) gru=64 proj=(32,16) tanh bins=10 adam lr=5e-4 batch=64 eval_every=250 patience=8 max_updates=20000 seeds=101..104
B2 params=12 starts=10 seeds=201..210 | B3 params=6 starts=10 seeds=301..310 | B23 params=7 seeds=311..320
B3 bounds phi,rho,delta in [0,1] lambda in [0.01,50] N0 in [0,20] theta in [0,5]; B23 kappa in [0,50]; payoff scale 50
surrogate fc=32 gru=32 lr=1e-3 batch=64 eval_every=250 patience=8 max_updates=20000 seeds M1 401..403 M2 411..413 (step 2: width 64, 404..406 / 414..416)
surrogate accept: median_abs<=1.0 p90_abs<=5.0 exclusion_agreement>=0.90
S=1000 games per (model, mechanism); M2 Exp 4: 25 sets per human set; sim seed=50000+100m+r
bootstrap B=10000 seed=9001; tolerance tau: surplus 0.1, gini 0.005, depletion 0.5, active 0.1; rule: >=3 of 4 metrics and Delta-ell CI_lo>0
B4: K=3 types, calib 60 games (4 x 15 compositions), grid step 0.1, eval 40 games/(variant, mechanism), retries=2, timeout=120 s, discard limit 10 %, call ceiling 55000, seed 7001
compute ceiling non-LLM 12 h
```
