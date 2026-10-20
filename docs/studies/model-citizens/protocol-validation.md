# Model Citizens C2: citizen-model validation protocol

Status: draft

Version v1-draft, written 2026-10-20 for the study owner's review. Only the owner freezes it: the freeze changes the word `draft` above to `frozen` (the single status line the held-out guard `phasespace.citizens.data.load_heldout` reads), and fills in §1. Until then no held-out human game is read. A frozen protocol is never edited in place; changes create v2 with a changelog entry and a reason, and every result states the version it used.

**Freeze checklist** (recorded in §1 and in a new D-entry, written only on the owner's instruction):
1. Every pre-freeze task in §10.1 is done and its output recorded (smoke costs, B4 pilot, memorisation probe, `players_kept_active` check, prompt texts).
2. All constants of this text are transcribed into the C2 configuration module; a test checks that Appendix A and the module agree; `config_sha256` is printed.
3. The open questions in §13 are answered and integrated.
4. The pre-freeze amendments of §14 are integrated in the sections they name.
5. The freeze commit changes only the status word and §1, on a clean tree.

## 1. Header

| | |
|---|---|
| Protocol version | v1 (draft) |
| Frozen date | — (owner) |
| Git commit at freeze | — |
| Config hash at freeze | — (`config_sha256` of the C2 configuration module) |
| Inputs | D31, D32, D33 (`docs/decisions.md`); `README.md` (spec v0); `literature.md`; C1 audit run `citizens-c1-data-audit-v1` and its C2 count-only extension `citizens-c1-data-audit-v1-dev3` (A29, A30); pre-freeze runs `citizens-c2-smoke-v1-dev`, `citizens-c2-surrogates-pre1`, `citizens-c2-b4-pilot-v1`, `citizens-c2-b4-probes-pilot-v1`, `citizens-c2-b4-pilot-v3`, `citizens-c2-b4-calibration-v2` (superseding `citizens-c2-b4-calibration-v1`, A32) (§14) |
| Scope | Milestone C2 and gate G2 only. The planner comparison (C4) has its own `protocol.md`. |

**What was known when this text was written (honest disclosure).** The authors have read Koster et al. 2025, including its *published aggregate results* for the held-out mechanisms (e.g. Exp 3 M2 > Interpolating surplus z = 2.35; Interpolating < M2 Gini z = 5.64) and its figures. The protocol is frozen with that knowledge but **before any computation of ours on held-out human rows**. Our only contact with held-out rows so far is C1's identity checks (pool transition, Interpolating offer reproduction), which return residual statistics and game counts only (one finding: 23 M1 Exp 1 rows whose recorded next pool is 0.01 below the full rule; §10.3), and two count-only C2 checks through the same path (A29, A30: anomalous recorded actions and `players_kept_active` vs #{e ≥ 1}; both counts are 0). Training aggregates (four Exp 1 z-values, reproduced in C1) have been seen, and so have the pre-freeze runs of §14 (training-regime dry run and in-sample check, surrogate BC fidelity, B4 pilot, probe answers and calibration games); none of them contains a held-out human row.

## 2. Question and hypothesis

**Question.** Which classes of behavioural model, fitted only on human play under the Equal, Mixed and Proportional rules, predict human play under mechanisms they never saw (Interpolating, M1, M2)?

**Hypothesis H-V (per model class and held-out regime).** The model reproduces the human group-level outcomes and beats a marginal-frequency baseline in held-out per-decision likelihood, by the criterion of §8. It is refuted for a (model, regime) pair when the criterion fails. **Control expectation:** the payoff-maximising control B0 fails every regime; if it passes one, the criterion is declared too weak for that regime (§9).

## 3. Model (game and data)

- **Game.** As in `README.md` and D33, implemented in `phasespace.citizens.game` / `mechanisms`: 4 players, R₀ = 200 = cap, R′ = max(0.01, min(200, R − Σe + 1.4Σc)); depletion when R < 1; a player with e < 1 is excluded and reciprocates 0; human actions are integers c ∈ {0, …, ⌊e⌋}. Exps 1–3: 40 rounds.
- **Deployed offer rule (D33).** Simulated baseline and Interpolating offers use the deployed convention `recorded_offers`: w quantised to one decimal for Interpolating (w = round((R/200)²², 1)), 0.001 coins withheld, offers rounded to 4 decimals. The equal split applies in round 1 and whenever Σc_{t−1} = 0.
- **Data.** `sustainable_behavior.csv` (C1-verified checksums), parsed by `phasespace.citizens.data`. Training rows via `load_training`; held-out rows only via `load_heldout` after the freeze (§10.2).
- **Player view (cyclic rotation).** Player i sees the four players in the order i, i+1, i+2, i+3 (mod 4), own values first, as the CSV records them; models and B4 prompts use this order.
- **Arithmetic.** Simulation runs in float64. The deployed software computed in float32, so recorded and simulated quantities can differ by about 4e-4 coin, far below the action resolution of one coin; this is noted, not modelled.
- **Training set (human, 280 games, 200 independent units).** Exp 1 Equal 40, Mixed 40, Proportional 40; Exp 2 Proportional 40; Exp 4 Proportional 40 groups × 3 consecutive games (25–49 rounds). An Exp 4 group (its 3 games) is one unit.
- **BC rows.** The simulated BC 1/BC 2 cohorts (3,584 games, 7 × 512) are not held out. They are used only (a) to fit the M1/M2 mechanism surrogates from BC 1 M1 and BC 2 M2 mechanism rows (§6), and (b) as a visual reference. They never enter a citizen model or a criterion.
- **Held-out human games** (counts from C1; content unseen): Interpolating Exp 2 = 40, Exp 3 = 80; M1 Exp 1 = 40, Exp 2 = 40; M2 Exp 3 = 80, Exp 4 = 40 groups × 3 games.

## 4. Candidate parameterization: the common likelihood

Every non-LLM model emits, for each decision of player i in round t with e = e_{i,t} ≥ 1 and n = ⌊e⌋, a pmf over the integers c ∈ {0, …, n}, conditioned only on what the player has observed (current four offers with own first, current pool, all previous offers and reciprocations of the game; rotated focal-first as in the CSV). **Forced-zero decisions (e < 1) are excluded** from every likelihood (they include all post-depletion rows).

**Declared handling of anomalous rows (A29, fixed before the unlock).** A recorded decision that is not an integer 0 ≤ c ≤ ⌊e⌋ (non-integer c; c > ⌊e⌋ with e ≥ 1; c < 0; c > 0 with e < 1) is excluded from every likelihood and counted; it stays, as recorded, in the history every model observes. A scored decision to which a model assigns P(c) < 10⁻¹² (including P = 0) is scored at log 10⁻¹² and counted per model. Neither stops the evaluation. Counted before the freeze through the C1 identity path (counts only, run `citizens-c1-data-audit-v1-dev3`): 0 anomalous held-out player-entries out of 58,360 (14,590 rows); training rows have none (C1 `action_validity`).

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
- **Units:** the 200 training units, stratified by (mechanism, cohort): Exp 1 Equal, Exp 1 Mixed, Exp 1 Proportional, Exp 2 Proportional, Exp 4 sets (40 units each). Fit/validation = 32/8 per stratum (160/40 units; 224/56 games). One generator `numpy.random.default_rng(20261020)` draws one permutation per stratum, in the order just listed, of the stratum's units sorted by `launch_id` (Exp 4: the `launch_id` its Games 1–3 share); the first 32 permuted units are the fit split, the last 8 the validation split.
- The validation split is used only for early stopping (B1, surrogates) and reporting. Final models are those trained on the fit split; there is **no refit on fit + validation**.
- **BC split for surrogates:** BC 1 M1 games and BC 2 M2 games each 410/102 by game: a fresh `default_rng(20261021)` per mechanism permutes that mechanism's games sorted by `launch_id`; the first 410 are the fit split.

### 6.2 Reference baselines (likelihood only, not model classes)
- **M0 marginal:** the fit-split frequencies of the 10 floor bins over non-forced decisions, +0.5 per bin, normalised; mapped to integers by §4. Gating reference for Δℓ. Also simulated closed-loop as a no-dynamics control (**M0-sim**, non-gating; §9).
- **P0 persistence** (diagnostic): with probability q the player's previous floor bin (if t > 0 and e_{i,t−1} ≥ 1), else M0; q by MLE on the fit split.

### 6.3 Model classes

**B0, rational negative control** (no parameters). Closed loop: c = 0 always (the myopic payoff maximiser). Likelihood: P(c) = 0.95·1[c = 0] + 0.05/(n+1).

**B1, GRU clone** (Koster et al. 2025, Table S1, BC1 scale; no hyperparameter search).
- Inputs (9, all /200): the four current offers (own first), the four previous reciprocations (own first; zeros in round 1 as recorded), the pool.
- Network: FC encoder (16, 32) → GRU 64 → FC projection (32, 16) → linear 10 logits (10 floor bins); tanh activations; one sequence per (game, player).
- Loss: cross-entropy of the observed floor bin on non-forced decisions (forced steps are fed as inputs, not scored). BPTT over the whole game; Adam, learning rate 5e-4 constant, batch 64 sequences, no regularisation. Each epoch is a permutation of the fit-split sequences drawn with `numpy.random.default_rng(seed)`, cut into consecutive batches of 64; the last partial batch of an epoch is dropped. Initial weights: equinox defaults from `jax.random.key(seed)`; float32 throughout.
- Early stopping: the validation score is the per-decision NLL of the integer pmf of §4 (the score Δℓ and K1 use), evaluated every 250 updates, patience 8 evaluations, at most 20,000 updates; the best-validation checkpoint is kept.
- **Ensemble N = 4** members, seeds 101–104 (the paper used 4 checkpoints). Likelihood: the mean of the members' integer pmfs. Closed loop: each simulated player draws one member uniformly with replacement at game start (as the paper's planner did).

**B2, parametric conditional cooperation** (12 parameters; MLE). Features of decision (i, t):
x₁ = min(4, 4e_{i,t}/R_t) (own offer relative to the equal share); x₂ = c_{i,t−1}/e_{i,t−1} (own previous fraction; 0 if t = 0 or e_{i,t−1} < 1); x₃ = Σ_{j≠i}c_{j,t−1} / Σ_{j≠i}e_{j,t−1} (others' previous fraction; 0 if t = 0 or the denominator < 1); x₄ = R_t/200; x₅ = 1[t = 0]; x₆ = 1[t > 0 and e_{i,t−1} < 1].
- Zero hurdle: logit π₀ = γ₀ + γ₁x₂ + γ₂x₃ (3). Full-return spike: logit π_F = δ₀ (1). Interior mean: logit μ = β₀ + Σ_{k=1..6} β_k x_k (7). Precision φ = exp(η) (1).
- pmf: P(0) = π₀. n = 1: P(1) = 1 − π₀. n ≥ 2: P(n) = (1 − π₀)π_F; for 1 ≤ c ≤ n − 1, P(c) = (1 − π₀)(1 − π_F)·BetaBinomial(c − 1; n − 2, μφ, (1 − μ)φ).
- β₃ is the conditional-cooperation slope (Fischbacher & Gächter 2010 under-matching).
- Fit: L-BFGS-B on the fit-split NLL from 10 starts (zeros + N(0, 0.5²) noise, seeds 201–210); best fit-split NLL is kept.
- *L-BFGS-B options* (B2, B3, B23): `scipy.optimize.minimize(method="L-BFGS-B")` with maxiter 2000, maxfun 100,000, ftol 1e-12, gtol 1e-7 and 2-point finite-difference gradients; the objective is the mean NLL per non-forced decision.

**B3, learning model: EWA with realised forward reinforcement** (6 parameters; MLE; Camerer & Ho 1999). Actions are the 11 levels a_j = j/10.
- Initial attractions A_j(0) = log(p⁰_j)/λ, where p⁰ is the fit-split round-1 level distribution (+0.5 smoothing); this is data-derived, not a free parameter.
- After a non-forced round t with chosen level j* = ℓ(c_{i,t}), at the start of round t + 1 (when e_{i,t+1} is seen): N(t+1) = ρN(t) + 1, N(0) = N₀, and
  A_j(t+1) = [φ N(t) A_j(t) + (δ + (1 − δ)1[j = j*])·e_{i,t}(1 − a_j)/50 + 1[j = j*]·θ·e_{i,t+1}/50] / N(t+1).
  The first payoff term is the myopic surplus (foregone payoffs weighted by δ); the θ term reinforces the chosen level by the next offer received, the institution's realised response. Forced rounds do not update.
- Choice: p_j = softmax_j(λA_j(t)); integer pmf by the 11-level map.
- Bounds: φ, ρ, δ ∈ [0, 1]; λ ∈ [0.01, 50]; N₀ ∈ [0, 20]; θ ∈ [0, 5]. L-BFGS-B, 10 starts uniform in the bounds (seeds 301–310).

**B23, hybrid (pre-declared fallback, D32)** (7 parameters). B3's attractions plus a conditional-cooperation pull: p_j ∝ exp(λA_j(t) − κ(a_j − x₃)²·1[t > 0]), κ ∈ [0, 50]. Fitted and evaluated **always**, before the unlock, with 10 starts uniform within the bounds (κ included), seeds 311–320, so that its use does not depend on held-out outcomes. Its role in C4 is fixed in §9.

**B4, LLM citizens through headless Claude** (closed loop only; no likelihood).
- *Runner.* One fresh `claude -p` call per player per round with an offer of at least 1 coin (at most 160 calls per 40-round game; an excluded player returns 0 without a call), the four players of a round in parallel, the prompt on stdin, the model id reported in each JSON response archived. Each call runs in a fresh empty temporary directory, with an environment built from an allowlist (PATH, HOME, USER, LANG, LC_ALL, TMPDIR, SHELL, TERM; A28): no other variable of the caller (an enclosing Claude Code session, API keys, tokens) reaches the call, and the names (never values) of the variables passed are recorded per run. The CLI authenticates through the macOS keychain with these alone (checked 2026-10-20). The CLI version is pinned (Claude Code 2.1.304, A26): every run records `claude --version` and the held-out B4 stage refuses another version. Each response's reported model usage must name exactly `claude-haiku-5-5`; anything else is an infrastructure error. The CLI flags are fixed (`phasespace.citizens.llm.backend.ClaudeCLI.argv`):
  `claude -p --model claude-haiku-5-5 --effort medium --output-format json --system-prompt "<SYSTEM_PROMPT>" --tools "" --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence --safe-mode --max-turns 1`.
  `--effort medium` is pinned explicitly: it is claude-haiku-5-5's default effort in this CLI version (its built-in model catalog and the served catalog both give `default_effort: "medium"`), so the pilot, run before the flag was pinned, stays representative.
- *Prompt* (`b4-prompt-v2`, `phasespace.citizens.llm.prompts`). A neutral rules text written from the paper's player instructions (coins, offers, keeping vs returning, the pool and its 1.4 growth, exclusion below 1 coin, round length not disclosed). Offers are described in the passive voice ("coins from the pool are offered to the four players"); the text never names the paper, DeepMind, a "manager", "AI" or "RL agent". Each call carries the rules, the player's current observation (four offers, own first; pool) and a table of all previous rounds as that player saw them (offers, everyone's reciprocations, pool, own coins). Every coin amount (offers, own and others', current and past; pool; coins kept) is shown truncated, not rounded, to 2 decimals (A32). The question first states the range ("You may return any whole number of coins from 0 to ⌊e⌋."), as the human players' integer slider from 0 to ⌊e⌋ did. Answer format: exactly the JSON object `{"return": N}`, N a whole number from 0 to ⌊e⌋; nothing else is accepted (no extraction from prose).
- *Parsing.* Valid iff an integer in {0, …, ⌊e⌋}. An **invalid answer** (unparseable or out of range) is retried up to 2 times with a one-line format reminder; a decision still invalid makes the game unrecoverable: it is discarded and replaced (mirroring the human dropout exclusion); discards are counted. If > 10 % of a regime's games are discarded (discarded / games played), or fewer than 20 games are complete (§7), B4 is non-gating for that regime. **Infrastructure errors** (A27: timeout of 120 s, non-zero exit, CLI or API error including rate limits, a missing result, a reported model other than the requested one) are not answers: they are retried with the same prompt after waits of 5, 10, 20, 40, 80 and 160 s, never count toward the discard rule, and a decision that exhausts the six retries stops the run, which is resumed with the same command (the interrupted games are replayed from round 1 under their pre-declared index; their calls are archived and counted).
- *Display precision (A32, fixed in `b4-prompt-v2`).* `b4-prompt-v1` rounded coins to 2 decimals, so an offer just below a whole number displayed above its largest valid return (the deployed round-1 offer 49.99975 showed "50.00" with a maximum of 49; e = 1.9996 showed "2.00" with a maximum of 1). In the v1 calibration run every discarded game (4 of 64 played) was of this kind: a player answered 50 in round 1 on all three attempts; all 53 invalid answers of its 6,299 calls were out of range. Humans could not make this error (integer slider from 0 to ⌊e⌋). `b4-prompt-v2` truncates every displayed coin amount to 2 decimals (49.99975 → "49.99", 1.9996 → "1.99"), so a displayed own offer never reaches ⌊e⌋ + 1, and states the allowed range in the question; every other word is unchanged. The v1 calibration run (`citizens-c2-b4-calibration-v1`) is superseded by `citizens-c2-b4-calibration-v2` (v2 prompt) and kept on disk unchanged; the held-out stage refuses it (its `prompt_version` differs).
- *Seeding.* LLM sampling cannot be seeded through the CLI; runs are stochastic and archived (every prompt, raw response, retry, timing and model id). Game order and type draws are seeded (7001).
- *Privacy.* Raw prompt/response records can contain the operator's account e-mail address, which the CLI injects into the model's environment context. They stay local under `runs/` (git-ignored) and are never published; published B4 data are the parsed actions only.
- **B4-plain:** the neutral prompt.
- **B4-calibrated** (after Manning & Horton 2025; Jackson et al. 2026, simplified): K = 3 pre-written type paragraphs appended to the neutral prompt — T1 "keeps most coins for themselves", T2 "returns about as much as the others returned last round", T3 "returns generously to keep the pool healthy" (exact texts fixed at freeze). Calibration on training mechanisms only: 60 Proportional 40-round games, 4 per type composition (15 compositions of 4 players over 3 types), in an order and seat assignment drawn from `default_rng(7001)`; a discarded calibration game is replaced by a game with the same seats and types. For weights w on the simplex grid of step 0.1 (66 points), predicted group means are Σ_comp Multinomial(comp; 4, w)·mean(comp). Chosen w minimises Σ_m ((pred_m − h_m)/s_m)² over the four metrics of §7, with h_m, s_m the mean and between-group SD of the fit-split Exp 1 + Exp 2 Proportional games (64 games); ties go to the w nearest uniform. In evaluation, each player's type is drawn i.i.d. from w with `default_rng([7001, game index])`.
- *Memorisation and isolation probes* (pre-freeze, logged; 16 calls): 5 fresh calls each of (a) "What do you know about Koster et al. 2025, Nature Communications, on deep RL in a common-pool resource game?", (b) the neutral rules followed by "Do you recognise this experiment? Name its source and any results you know." and (c) a described-study question naming no paper ("a four-player common-pool game in which an AI manager redistributes a shared pool of coins … grow by a factor of 1.4"); plus 1 isolation call asking the model to list verbatim every instruction in its context. Consequence: if any answer states k = 22, the interpolating rule, or a published result, all B4 results carry the label "contamination risk: high". Not a stop rule. **Verdict (pre-freeze probe run `citizens-c2-b4-probes-pilot-v1`, A33):** no evidence that `claude-haiku-5-5` knows the 2025 study: every paper-specific answer denies knowledge of it (and none states k = 22, the interpolating rule or a published result), and the keyword flags (10 of 16 answers) fire on those denials, on mentions of the 2022 study and on the generic word "sustainable"; the model does know Koster et al. 2022 ("Democratic AI"), which it names unprompted. Label for every B4 result: **"no contamination evidence for the 2025 study"**.

### 6.4 Mechanism surrogates for M1 and M2
M1/M2 weights are not released, so closed-loop simulation needs a surrogate.
- **Fitting data:** BC 1 M1 rows (for M1) and BC 2 M2 rows (for M2) only — mechanism observation → offers. **Never human rows.**
- **Architecture** (permutation-equivariant, recurrent, like the paper's GNN + GRU): per player, inputs (e_{i,t−1}, c_{i,t−1}, R_t, mean_j e_{j,t−1}, mean_j c_{j,t−1}), all /200 (zeros in round 1) → shared FC 32 (tanh) → shared GRU 32 per player → FC 32 → 1 player logit; pool logit from the mean of the four hidden states → FC 32 → 1. Every hidden width equals the surrogate's width (32; 64 in the capacity step). Offers e = R·softmax(5 logits)[players].
- **Loss:** cross-entropy between the recorded share vector (e_i/R, 1 − Σe/R) and the predicted shares, over rounds with R ≥ 1. Adam 1e-3, batch 64 games (each epoch a permutation of the fit games by `default_rng(seed)`, the last partial batch dropped; initial weights from `jax.random.key(seed)`), validation offer MAE every 250 updates, patience 8, at most 20,000 updates; seeds 401–403 (M1) and 411–413 (M2); the best BC-validation seed is kept.
- **Pre-freeze fit.** Because the surrogates use BC rows only and the fit is deterministic, they may be fitted before the freeze (run `citizens-c2-surrogates-pre1`: BC-validation median / p90 |ê − e| and exclusion agreement M1 0.28 / 1.60 coins / 0.993, M2 0.36 / 2.64 coins / 0.991; both accepted at width 32, kept seeds 403 and 413; every seed ran the full 20,000 updates). The post-freeze step (§10.2) re-runs the fit and must reproduce the parameter files bit for bit (`--reproduce`); a difference is reported to the owner before the unlock.
- **Pre-unlock check (BC validation games):** the acceptance thresholds below must hold. If they fail, one pre-declared capacity step (FC/GRU 64, seeds 404–406 / 414–416); if that also fails, the surrogate is rejected before the unlock.
- **Acceptance on held-out human mechanism rows** (after the unlock; offers predicted teacher-forced from the recorded observation sequence of each human game; rounds with R ≥ 1; per (mechanism, experiment), every experiment of the mechanism including M2 Exp 4):
  1. median |ê − e| ≤ 1.0 coin, **and**
  2. 90th percentile |ê − e| ≤ 5.0 coins, **and**
  3. exclusion agreement 1[ê < 1] = 1[e < 1] on ≥ 90 % of player-rounds.
  *Justification.* Humans act in whole coins, so a 1-coin median error is below the action resolution; 5 coins is 10 % of the full-pool equal share (50), so large misallocations such as excluding the wrong player (errors of order R/4) must be rare; exclusion is M1/M2's signature behaviour and drives depletion and players-active.
- **If a surrogate fails acceptance** in any experiment of its mechanism: that mechanism's closed-loop results are reported as diagnostic only, and its validation rests on the one-step likelihood alone (tier "one-step", §8). The surrogate is never refitted after the unlock.

## 7. Budget, seeds and simulation sizes

- **Non-LLM compute ceiling:** 12 h wall-clock on the M4 Pro for all fitting, surrogates and simulation together (B1 ≤ 4 × 20,000 updates, B2/B3/B23 10 starts each, surrogates ≤ 2 × 6 × 20,000 updates). *Measured before the freeze* (smoke run `citizens-c2-smoke-v1-dev`, protocol settings): all fits 0.11 h; B1 worst case (4 × 20,000 updates) 0.16 h; S = 1,000 simulated games take 0.04–2.9 s per (model, mechanism); the surrogates took 0.36 h for 6 seeds × 20,000 updates (`citizens-c2-surrogates-pre1`). Projected total ≈ 0.7 h, far inside the ceiling.
- **LLM call ceiling:** 55,000 calls including retries; wall-clock ≤ 14 days. Planned: pilot 480; probes 16; calibration 9,600; evaluation 2 variants × 3 mechanisms × 40 games × 160 = 38,400 (≈ 48,500). *Measured before the freeze* (pilot `citizens-c2-b4-pilot-v1`, 2 Proportional games): 106 calls, 0 invalid answers, mean latency 8.5 s, ≈ 1,400 calls/h with 8 concurrent calls; projected 16,000 calls at the measured calls per game, 48,100 in the worst case (160 calls per game), ≤ 1.5 days. *After the display fix (A32)* (pilot `citizens-c2-b4-pilot-v3`, 2 games): 191 calls (95.5 per game), 0 invalid answers, mean latency 6.5 s, ≈ 2,060 calls/h; calibration v2: 6,367 calls (106.1 per game), retry fraction 2.0 % (12 invalid answers, 114 infrastructure errors from a session limit), ≈ 6,600 calls/h. Its runner projection (its own 6,367 calls plus 60 calibration and 240 evaluation games, so the calibration is counted twice) gives 38,202 calls at the measured rate and 55,328 in the worst case. Counted directly: the B4 calls spent before the freeze are 12,979 (probes 16, pilot v1 106, calibration v1 6,299, pilot v3 191, calibration v2 6,367), and the 240 evaluation games need at most 240 × 160 × 1.02 ≈ 39,200 calls, ≈ 52,200 in all, inside the 55,000 ceiling. B4 is not run on M2 Exp 4 (3-game sets of variable length), and is run on M1/M2 only if the mechanism's surrogate passed the pre-unlock BC check.
- **Pre-declared reductions** (applied in order only if the pilot's measured throughput projects past a ceiling): (1) calibration 4 → 2 games per composition; (2) evaluation 40 → 30 games per (variant, mechanism); (3) M1/M2 evaluation 30 → 20 games. Below 20 games a regime's B4 result is exploratory and non-gating.
- **Simulation sizes (non-LLM):** S = 1,000 simulated games per (model, mechanism) for Interpolating, M1, M2 and for each training mechanism (in-sample check, §8); for M2 Exp 4, 25 simulated 3-game sets per human set (1,000 sets), each game's length equal to the matched human game's recorded length (lengths are an exogenous lottery).
- **Seeds:** split 20261020; BC split 20261021; B1 101–104; B2 201–210; B3 301–310; B23 311–320; surrogates 401–406, 411–416; simulation seed = 50000 + 100·m + r with m ∈ {B0 0, M0-sim 1, B1 2, B2 3, B3 4, B23 5} and r ∈ {Interpolating 1, M1 2, M2 3, M2 Exp 4 4, Equal 5, Mixed 6, Proportional 7}; bootstrap 9001; B4 game order and type draws 7001. The MC error of a simulated mean with S = 1,000 is ≈ 3 % of the between-game SD, an order below the human CI half-widths expected with 80–120 groups.

## 8. Held-out regimes, metrics and the validation criterion

**Metrics** (notebook definitions, `phasespace.citizens.metrics`; per game; human values from all recorded rows): (1) surplus per player-round; (2) Gini of the players' mean rewards (the paper's "Gini", 0.5·mean|xᵢ − xⱼ|/mean x; not the CSV's per-round `offer_gini`); (3) depletion round (first round with R < 1, 1-based, else 40); (4) players active at the end, #{i : e_{i,T} ≥ 1} in the last round, for human and simulated games alike (A30; it equals the CSV's `players_kept_active` on every training row and every held-out row: 0 mismatches in 9,832 training and 14,590 held-out rows, counts only).

**Gating regimes** (one per held-out mechanism; human groups pooled over that mechanism's 40-round experiments, because a model's simulation of a mechanism is the same for every cohort):

| Regime | Human groups | Mechanism in simulation |
|---|---|---|
| R-I | Interpolating Exp 2 (40) + Exp 3 (80) = 120 | exact deployed rule (§3) |
| R-M1 | M1 Exp 1 (40) + Exp 2 (40) = 80 | M1 surrogate |
| R-M2 | M2 Exp 3 (80) | M2 surrogate |

**Diagnostic regimes** (reported, never gating): each (mechanism, experiment) separately; **M2 Exp 4** (40 groups × 3 games; unit = group, group metric = mean over its 3 games). M2 Exp 4 is non-gating because our models carry no state across games, Exp 4 changed the incentive (whole-episode pay) and length, and the notebook's depletion round defaults to 40 for shorter games.

**Human reference.** For each regime and metric, the mean over groups and its 95 % percentile bootstrap CI: B = 10,000 resamples of groups (stratified by experiment in pooled regimes, each stratum resampled to its own size), seed 9001. One resample matrix serves every metric and every model's Δℓ of the regime.

**Criterion for model m on regime r** ("validated (closed-loop)"):
- **(a) Outcomes:** for ≥ 3 of the 4 metrics, the simulated mean over S games lies in [lo − τ_k, hi + τ_k], the human CI widened by a resolution tolerance τ = (surplus 0.1 coins, Gini 0.005, depletion 0.5 rounds, players active 0.1) — this matters only when a human CI is degenerate (e.g. no group depleted).
- **(b) Likelihood** (non-LLM models): Δℓ_m > 0 on the regime's non-forced decisions (teacher-forced, every game's recorded history), with the 95 % group-bootstrap CI lower bound of Δℓ_m > 0 (B = 10,000, seed 9001). Each resample recomputes the decision-weighted ratio Σ_u S_u / Σ_u N_u over its groups, with S_u a group's sum of log P_m(c) − log P_M0(c) and N_u its number of non-forced decisions.
- **(c) Must beat marginal** (owner decision 2026-10-20, A22). In a regime where M0-sim passes (a), a model's closed-loop validation there additionally requires its mean normalised Wasserstein-1 distance over the 4 metrics to be strictly lower than M0-sim's. The normalised W1 of a metric is the W1 between the model's simulated game-level values and the human unit-level values, divided by the human between-unit SD of that metric (ddof 1; if that SD is 0, by the metric's τ instead). A model that passes (a) and (b) but not (c) is labelled "not validated (mean normalised W1 not below M0-sim)". `m0_sim_passes_a` and the W1 comparison of every model are recorded in the G2 block of the evaluation run.
- B4: (a) and (c) only, with its own simulated games (≈ 40 per regime), in a separate stage (`just citizens-b4-evaluate`, A24) against the human reference CIs, human unit metrics, W1 scales and M0-sim values recorded by the evaluation run. The B4 simulated mean's own 95 % CI (B = 10,000 resamples of the complete B4 games, seed 9001) is reported. The stage computes each run's discard fraction (discarded / games played) and `b4_gating` = (discard fraction ≤ 0.10 and ≥ 20 complete games).
- If the regime's surrogate failed acceptance (§6.4): (a) and (c) are diagnostic and the pair is "validated (one-step)" iff (b) holds.

**In-sample check (pre-unlock, non-gating):** the same criterion on the training mechanisms (Equal: Exp 1; Mixed: Exp 1; Proportional: Exp 1 + Exp 2; human groups of the whole training set, 40-round games only, so Exp 4 sets are excluded). Reported to show whether a model fits the data it was trained on.

**M2 Exp 4 simulation** (diagnostic): for each human set, 25 simulated sets of three games whose lengths equal the set's recorded Game 1–3 lengths; a simulated set's metric is the mean over its three games, as for human sets, so means, SDs and W1 compare sets with sets.

**Distributional diagnostics (non-gating except through (c)):** for each metric, the Wasserstein-1 distance (also normalised as in (c)) between the simulated and human game-level distributions, with a human-vs-human reference (the median W1 between random halves of the human groups over 1,000 splits, seed 9001); per-round mean reciprocation fraction and pool trajectories; reciprocation fraction vs offer share and vs pool ("fingerprint" plots); Δℓ vs P0; per-experiment results.

## 9. G2 decision rule and pre-declared outcomes

**Classes counted for G2:** B1, B2, B3, B23, B4 (either variant). B0 and M0 never count. **Trainable** classes (usable as PPO environments in C3/C4): B1, B2, B3, B23.

**G2 passes** if B0 fails every gating regime and some gating regime whose human reference is verified (G1b) has ≥ 2 trainable classes, at least one of them not B1, validated there (O4: one common regime). A "validated (closed-loop)" label counts; a "validated (one-step)" label counts only if the same model is validated (closed-loop) on another counting regime (O3). Implemented as `phasespace.citizens.heldout.g2_decision`.

**Criterion-strength checks.** **B0 passes** a gating regime when it passes the part of the criterion that decides validation there (A23): outcome criterion (a) where closed-loop results count, the likelihood criterion (b) where only the one-step tier counts (a rejected surrogate). B0 never needs both (B0's likelihood is a weak point mass at 0, so requiring (a) and (b) would make the control vacuous). If B0 passes: the criterion is declared too weak for that regime; no "validated" label from that regime counts for G2; this is reported and the owner is asked. Any stronger criterion would be post hoc (v2, labelled exploratory). If **M0-sim passes (a)** on a regime, the outcomes criterion alone does not discriminate there: closed-loop validation in that regime then also needs (c), a lower mean normalised W1 than M0-sim (A22).

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
| M0-sim passes (a) on a regime | Outcomes alone do not separate models from the marginal citizen there | Closed-loop validation there also needs (c) (A22) |

## 10. Procedure

### 10.1 Before the freeze (training data only)
1. Implement the C2 code and configuration; `just verify` passes.
2. Smoke run: one B1 member for 2,000 updates, one start each of B2/B3/B23, one surrogate seed for 2,000 updates, 100 simulated games; record measured costs in §7. Simulating the models under the exact Interpolating rule before the freeze is allowed for timing and feasibility only: model-only, no human data, no simulated outcome recorded.
3. Check on training human rows that `players_kept_active` equals #{i : e_i ≥ 1} in every row; if not, the owner fixes the simulated definition before freezing. *Done:* 0 mismatches in 9,832 training rows (test) and, as a count-only identity over held-out rows (A30), 0 mismatches in 14,590 held-out rows (run `citizens-c1-data-audit-v1-dev3`).
4. B4: write the prompt and type texts; pilot 2–3 games under Proportional (seconds per call, rate limits, parse failures); isolation probe; memorisation probe. Prompt changes are allowed only here. On the owner's instruction the B4-calibrated calibration (§6.3) is also run before the freeze, training mechanism only; it stands only if the prompt texts and CLI flags are unchanged at the freeze, otherwise it is re-run after the freeze. *Calibration v1, superseded (A32):* run `citizens-c2-b4-calibration-v1` (`b4-prompt-v1`, finished 2026-10-20: 60 complete games, 4 discarded, 6,299 calls, 0 infrastructure errors) showed rounded coins, and all 4 of its discards were the display artifact of A32; it also predates the environment allowlist and the infrastructure-retry rule (A27, A28: its calls ran with the earlier denylist, the caller's `CLAUDE*` and `AI_AGENT` variables removed, and infrastructure errors counted as invalid answers). It is kept on disk unchanged and is not used: the held-out stage refuses it (`prompt_version`). *Calibration v2, the run of record:* after the display fix, pilot `citizens-c2-b4-pilot-v3` (2 B4-plain Proportional games, `b4-prompt-v2`: 191 calls, 0 invalid answers, 0 infrastructure errors, 0 discards, mean latency 6.5 s, ≈ 2,060 calls/h) and calibration `citizens-c2-b4-calibration-v2` (`b4-prompt-v2`, environment allowlist, A27 retry rule, CLI 2.1.304, effort medium; 2026-10-20): 60 complete games, 1 discarded (61 played), 6,367 calls (6,242 decisions), 12 invalid answers, all out of range (answers above ⌊e⌋ with displayed offers 0.07–0.79 below the next whole number, i.e. not the A32 artifact; the discard is three answers of 47 with ⌊e⌋ = 46, displayed offer 46.93), 114 infrastructure errors (all non-zero exits from an account session limit, HTTP 429, within 6 minutes; all recovered by the A27 backoff), only `claude-haiku-5-5` reported, US$ 3.51, wall time 57.6 min. Its `games.json` passes the A31 verification and its set-up passes the calibration identity check. *Calibrated type weights* (training data only, `just citizens-b4-weights`; grid step 0.1): w = (T1 0.7, T2 0.0, T3 0.3), objective 0.198; predicted group means surplus 7.14, Gini 0.387, depletion round 31.7, players active 1.09 against the fit-split Proportional human means 6.25, 0.350, 34.5, 1.02 (64 games); calibration manifest SHA-256 `26243585486483535b4651246197372153e31d0f409d3a497035384787b9a258`.

### 10.2 After the freeze, before the unlock
5. From a clean tree: fit M0, P0, B1, B2, B3, B23 (`just citizens-c2-fit citizens-c2-fit-v1`) and re-run both surrogates with their BC checks, reproducing the pre-freeze fit bit for bit (`just citizens-c2-surrogates citizens-c2-surrogates-v1 --reproduce runs/citizens-c2-surrogates-pre1`); B4 calibration (§10.1 step 4). The evaluation refuses a surrogate run that did not reproduce (A25: reference parameters recorded, both `surrogate_*_differs_from_reference` checks 0, run status `validated`) unless the owner passes `--owner-override-surrogate-reproduction "<reason>"`, which is recorded in the evaluation manifest and its limitations.
6. The **fitted-model runs** are these immutable runs (fit, surrogates, calibration), each with checksums of every artifact. Later stages load them by checksum and record each one's manifest SHA-256; the B4-calibrated weights are computed deterministically from the calibration run and the fit-split human games and recorded in every B4 held-out run. The calibration `games.json` is verified against its manifest SHA-256 and must hold exactly 4 complete games for each of the 15 compositions (A31).

### 10.3 Unlock and G1b
7. Every post-freeze stage (`phasespace.citizens.heldout`: `just citizens-g1b`, `just citizens-c2-evaluate`, `just citizens-b4-heldout`, `just citizens-b4-evaluate`; and the surrogate's human-row acceptance, `surrogate.human_fidelity`) refuses unless this file reads `frozen`, is tracked and clean, `--confirm-frozen` is passed and the working tree is clean; held-out rows are read through `load_heldout(confirm_frozen=True)`; every manifest records the protocol SHA-256, `config_sha256`, the commit and `git_dirty = false`.
8. **G1b, first:** (a) reproduce the published Exp 1–4 rank-sum surplus and Gini z-values that involve held-out mechanisms (games as units; Exp 4 per game number) to within half a unit of the last printed digit (|Δz| ≤ 0.005; ≤ 0.05 for the Exp 2 M1 > Proportional value printed as 2.9): Exp 1 M1 > Proportional 3.25, > Mixed 4.38, > Equal 6.31, Gini M1 < Proportional 2.78; Exp 2 M1 > Proportional 2.9, Interpolating > Proportional 2.52, Gini Interpolating < Proportional 4.17, Interpolating < M1 3.67; Exp 3 M2 > Interpolating 2.35, Gini 5.64; Exp 4 M2 > Proportional, games 1–3: 2.22, 4.99, 6.08. The other published z-values (players active, pool duration, exclusion length, Exp 4 trends) depend on endpoint definitions the paper does not fix and are not part of G1b; (b) root-cause the 23 M1 Exp 1 rows whose recorded next pool is 0.01 below the full rule, and count human games whose depletion round would differ under the full rule. *Consequences:* a z-value that cannot be reproduced makes that mechanism's human reference unverified: its regime does not count for G2 and the owner is asked. Human metrics always use the recorded rows. The simulator keeps the full rule; whatever the anomaly turns out to be, no model or surrogate is refitted, and it is reported as a limitation.
9. Surrogate acceptance (§6.4); one-step likelihoods; closed-loop simulations; criterion; G2 (§9) — one evidence run (`just citizens-c2-evaluate`). Then the B4 held-out games, one resumable run per (variant, mechanism) (`just citizens-b4-heldout`; `--games` accepts only 40 or the §7 reductions 30, and 20 for M1/M2, A31), and their evaluation against the evaluation run (`just citizens-b4-evaluate`, A24).

## 11. Rules after the unlock (forbidden actions)

- No refit, re-initialisation, re-selection of ensemble members or checkpoints, change of hyperparameters, seeds, S, B, thresholds, tolerances, regimes or prompts.
- No extra B4 games beyond the pre-declared count, and no re-running of a regime with new seeds or new LLM samples to change a verdict. Discarded B4 games are replaced only under §6.3's rule.
- Held-out rows are read only through `load_heldout`.
- Any analysis not specified here is labelled **exploratory** wherever it appears, and cannot change a "validated" label or G2.

**Reported regardless of outcome:** the full model × regime × metric table (simulated means, human CIs, pass/fail), Δℓ with CIs for every non-LLM model and P0, B0 and M0-sim results, in-sample checks, surrogate fidelity (BC and human), G1b outcome, B4 parse/discard rates, infrastructure-error counts, call counts, probe answers and calibration weights, the A22 W1 comparison (`m0_sim_passes_a`, every model's mean normalised W1), the A29 counts (excluded and floored decisions), the distributional diagnostics, and per-experiment results. The run status follows `docs/data-contract.md`: the evaluation run is `validated` only if its recorded checks (G1b, feasibility of every simulated game, checksum match with the fitted-model run) pass; otherwise `computed_unvalidated`.

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

## 14. Pre-freeze amendments (2026-10-20)

Recorded before the freeze, from the C2 implementation and the pre-freeze runs; each is integrated in the section named. None uses a held-out human row, except the count-only identity checks of A29 and A30 (through the C1 identity path; counts, no behavioural value).

| # | Amendment | Section |
|---|---|---|
| A1 | Split mechanics: one generator `default_rng(20261020)`, one permutation per stratum in the §6.1 order, first 32 permuted units = fit | §6.1 |
| A2 | Cyclic rotation: player i sees players i, i+1, i+2, i+3 (mod 4) | §3 |
| A3 | B1: early stopping on the integer-pmf NLL; epoch-permutation batches, partial batch dropped; equinox default initialisation; float32 | §6.3 |
| A4 | L-BFGS-B: maxiter 2000, maxfun 100,000, ftol 1e-12, gtol 1e-7, finite-difference gradients; B23 starts uniform within the bounds | §6.3 |
| A5 | Δℓ bootstrap: decision-weighted ratio over groups; one resample matrix per regime for every metric and model | §8 |
| A6 | W1 human reference: the median over the 1,000 half-splits | §8 |
| A7 | Training regimes (dry run, in-sample check) use 40-round games only | §8 |
| A8 | Simulation in float64; the deployed float32 arithmetic (≈ 4e-4 coin) is noted, not modelled | §3 |
| A9 | Pre-freeze simulation under Interpolating is allowed for timing and feasibility only (model-only, no human data, no outcome recorded) | §10.1 |
| A10 | B4: answer format `{"return": N}`; passive "coins are offered", never "manager"; the exact CLI flags; `--effort medium` pinned (Haiku 5.5's default in Claude Code 2.1.304); probes are 16 calls including the described-study question and the isolation call | §6.3, §7 |
| A11 | B4 privacy: raw prompt/response logs can contain the operator's account e-mail (CLI environment context); they stay local under `runs/` and are never published; published B4 data are parsed actions only | §6.3 |
| A12 | B4-calibrated: seeded order and seats; a discarded calibration game is replaced by the same seats and types; evaluation types from `default_rng([7001, game index])`; calibration run before the freeze on the owner's instruction, valid only with unchanged prompts and flags | §6.3, §10.1 |
| A13 | Surrogate BC split: a fresh `default_rng(20261021)` per mechanism over games sorted by `launch_id`; all hidden widths = width; epoch-permutation batches | §6.1, §6.4 |
| A14 | Surrogates may be fitted before the freeze (BC data only, deterministic fit); the post-freeze step re-runs the fit and must reproduce it bit for bit, otherwise it is reported | §6.4, §10.2 |
| A15 | Pre-freeze facts, surrogate BC-validation fidelity (median / p90 / exclusion agreement): M1 0.28 / 1.60 / 0.993; M2 0.36 / 2.64 / 0.991 | §6.4 |
| A16 | Surrogate acceptance on human rows covers every experiment of the mechanism, including M2 Exp 4 | §6.4 |
| A17 | M2 Exp 4 simulation: per human set 25 simulated 3-game sets of the recorded lengths; set metric = mean over its games | §8 |
| A18 | G2 rule integrates O3 and O4 (one common counting regime) | §9 |
| A19 | G1b: the 13 surplus and Gini z-values listed; tolerance half a unit of the last printed digit | §10.3 |
| A20 | The fitted-model "run" is the set of fit, surrogate and calibration runs, loaded by checksum; post-freeze stages require a clean tree and record `git_dirty = false` | §10.2, §10.3 |
| A21 | Measured costs: non-LLM ≈ 0.7 h projected (ceiling 12 h); B4 pilot ≈ 1,400 calls/h, 0 invalid answers in 106 calls | §7 |
| A22 | **Must beat marginal** (owner decision 2026-10-20): where M0-sim passes (a), closed-loop validation also needs a mean normalised W1 over the 4 metrics strictly below M0-sim's; normalisation = the human between-unit SD (τ if that SD is 0); `m0_sim_passes_a` and the W1 comparison recorded in the G2 block; applies to B4 too | §8, §9 |
| A23 | The B0 negative control is decided by (a) where closed-loop results count and by (b) in one-step regimes (it previously needed (a) and (b), which B0 never meets, so the check was vacuous) | §9 |
| A24 | B4 evaluation stage (`b4-evaluate`): (a) and A22 against the evaluation run's human CIs, unit metrics, W1 scales and M0-sim values; game bootstrap of B4 means (B = 10,000, seed 9001); discard fraction = discarded / games played; `b4_gating` = discard fraction ≤ 0.10 and ≥ 20 complete games; post-freeze guard as every stage | §8, §10.3 |
| A25 | The evaluation refuses a surrogate run that did not reproduce the pre-freeze fit bit for bit (reference recorded, both difference checks 0, status `validated`) unless the owner overrides it with a recorded reason | §10.2 |
| A26 | Claude Code CLI pinned at 2.1.304: version recorded per run, the held-out B4 stage and the calibration check refuse another; a response whose reported model usage is not exactly the requested model is an infrastructure error | §6.3, App. A |
| A27 | Infrastructure errors (timeout, non-zero exit, CLI/API error incl. rate limits, missing result, model mismatch) get their own budget: 6 retries with waits 5–160 s (doubling), never discards; exhausting it stops the run (resumable). Only invalid answers count toward the 10 % discard rule | §6.3, App. A |
| A28 | Subprocess environment from an allowlist (PATH, HOME, USER, LANG, LC_ALL, TMPDIR, SHELL, TERM; names recorded per run). *Disclosed deviation:* the superseded calibration v1 used the earlier denylist (`CLAUDE*`, `AI_AGENT` removed) and the earlier retry rule; calibration v2 (A32) uses the allowlist and A27 | §6.3, §10.1 |
| A29 | Declared before the unlock: anomalous recorded decisions (non-integer c, c > ⌊e⌋, c < 0, c > 0 with e < 1) are excluded from the likelihood and counted; P(c) < 10⁻¹² is scored at log 10⁻¹² and counted. Held-out count (identity path, counts only): 0 of 58,360 player-entries | §4, §8 |
| A30 | Players active = #{i : e_i ≥ 1} in the last round for humans and simulations alike; identity `players_kept_active` = #{e ≥ 1} checked on held-out rows as counts only: 0 mismatches in 14,590 rows | §8, §10.1 |
| A31 | `heldout b4 --games` accepts only 40 or the §7 reductions (30; 20 for M1/M2); the calibration `games.json` is verified against its manifest SHA-256 and must hold exactly 4 complete games per composition | §6.3, §10.2 |
| A32 | Display fix, `b4-prompt-v2`: every coin amount truncated (not rounded) to 2 decimals, so a displayed own offer never reaches ⌊e⌋ + 1 (v1 showed the round-1 offer 49.99975 as "50.00" with a maximum of 49; all 4 v1 calibration discards were answers of 50), and the question first states "You may return any whole number of coins from 0 to ⌊e⌋." (the humans' integer slider); other wording unchanged. Calibration v1 (`citizens-c2-b4-calibration-v1`, old display) is superseded by `citizens-c2-b4-calibration-v2` and kept on disk unchanged; pilot `citizens-c2-b4-pilot-v3` and calibration v2 run with v2 (weights w = (0.7, 0.0, 0.3)) | §6.3, §7, §10.1 |
| A33 | Probe verdict: no evidence that Haiku knows the 2025 study (all paper-specific answers deny knowledge; keyword flags fire on denials); it knows Koster et al. 2022. B4 label: "no contamination evidence for the 2025 study" | §6.3 |

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
B4 CLI: model=claude-haiku-5-5 effort=medium cli_version=2.1.304 env allowlist=PATH,HOME,USER,LANG,LC_ALL,TMPDIR,SHELL,TERM
B4 infrastructure retries=6 backoff=5,10,20,40,80,160 s (never discards) | eval reductions 30 (any), 20 (M1/M2) | gating >=20 complete games
likelihood pmf floor=1e-12 | must beat marginal: mean W1/SD over 4 metrics < M0-sim's where M0-sim passes (a)
L-BFGS-B maxiter=2000 maxfun=100000 ftol=1e-12 gtol=1e-7 (2-point finite differences) | B1 early stopping=integer_nll batching=epoch_permutation
W1 human reference: median over 1000 half-splits | Delta-ell bootstrap: decision-weighted ratio, one resample matrix per regime
compute ceiling non-LLM 12 h
```
