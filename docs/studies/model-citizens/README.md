# Model Citizens — Research Specification (v0)

**Theme:** RL mechanism design, behavioural models of people, social cooperation, model validation.  
**Status:** spec v0, protocol not frozen. Chosen 2026-10-20 (`docs/decisions.md` D31); supersedes the Constitution Machine slot.  
**RL required?** Yes. The policy designer is a PPO planner; RL is the object of the question, not an add-on.

## Question

> Does the mechanism an RL policy designer discovers depend on which validated model of people it is trained against?

An empirical Lucas critique of AI policy design:

1. Fit several classes of "citizen" models only on human play under the ordinary (baseline) rules.
2. Validate each against real human play under rules it never saw: the interpolating rule and the two RL-designed mechanisms M1 and M2.
3. Train PPO planners against each validated model and test whether the planners, and the rules distilled from them, agree.

Both outcomes are publishable: "the discovered policy is robust to the choice of simulator", or "validated simulators disagree, so the data underdetermine the policy".

## Why it matters

- **Who has the problem.** AI-for-policy and computational mechanism design trains policies against simulated people: neural behavioural clones (Koster et al.), synthetic agents (the AI Economist, Zheng et al. 2021, arXiv 2108.02755) and, increasingly, LLM "silicon subjects". The designed policy is only as trustworthy as the simulator, and simulators are usually checked on the data they were fitted to.
- **What changes if we succeed.** A concrete, reproducible test of whether "validated on held-out institutions" is enough for a learned policy to transfer between plausible models of people, and a public instrument for showing the gap.

## Research context

- **Koster et al. 2022**, *Human-centered mechanism design with Democratic AI*, Nature Human Behaviour; arXiv 2201.11441 (verified). A deep-RL mechanism, trained against behavioural clones, redistributes a public-goods fund and wins a majority vote of real players. A related but different one-shot game (10 rounds, no carried-over pool).
- **Koster et al. 2025**, Nature Communications 16:2824, doi 10.1038/s41467-025-58043-7; arXiv 2404.15059 (verified). A deep-RL "manager" redistributes a common pool that can be depleted. It is trained against GRU clones (BC1, BC2) of human players. The mechanisms it found (M1, M2) and a distilled interpolating rule beat baselines with real people (4,952 Prolific participants, UK and US).
- **The gap, refined by the literature scan ([literature.md](literature.md)).** In the 2025 paper the clones were checked only visually ("behavioural fingerprint"), with no held-out likelihood. Its second clone set (BC2) was trained on interpolating and RL-agent games, so it is not a held-out model. Koster et al. 2022 did validate clones on a separate human set, and also trained a second mechanism against "rational" learners, which new humans liked less (57.2 % preferred the clone-trained one). So *that the model of people can matter* has been shown once. Tacchetti et al. 2025 (PNAS) name the Lucas critique and suggest training against a diverse population, but do not test it. **Our contribution is the empirical test.** It requires every model class to pass the same pre-registered check on held-out mechanisms, and evaluates each planner under every other validated model. We must not claim that the framing is new, or that nobody validated clones.

Scan record: [`../../scouting/2026-10-19-rl-topic-scan.md`](../../scouting/2026-10-19-rl-topic-scan.md).

## The game (Koster et al. 2025)

Labels: **[S]** stated in the paper or supplement; **[D]** checked against CSV rows; **UNSPEC** not specified, so the reimplementation must choose and document.

- p = 4 players plus one mechanism. Pool R₀ = 200, which is also the cap. Growth factor 1.4 on reciprocated coins only **[S, D]**.
- Each round t **[S]**:
  1. the mechanism offers eᵢ with Σᵢ eᵢ ≤ R_t;
  2. each player chooses cᵢ ∈ [0, eᵢ]; their surplus (points) is eᵢ − cᵢ;
  3. R_{t+1} = min(200, R_t − Σe + 1.4·Σc) **[S, D]** (e.g. 200 − 199.9988 + 1.4 × 114.417 = 160.185).
- Depletion: the first round with R < 1 (notebook rule). Behaviour and rows after depletion: **UNSPEC** (the notebook assumes 40 rows per game).
- Exclusion: an offer below 1; the player must keep it and reciprocate 0 **[S]**.
- Actions: humans choose integers on a slider 0..eᵢ **[S, D]**; offers are continuous **[D]**. Slider bound for non-integer eᵢ: **UNSPEC** (floor is the natural choice).
- Observation: the four current offers (own first), the pool and everyone's previous reciprocations **[S, D]**.
- Horizon: Exps 1–3, 40 rounds, length not disclosed to players. Exp 4: at least 25 rounds then a 0.2 stopping chance per round; three consecutive games per group with the pool reset **[S]**.
- Pay: £0.008 per point **[S]**; Exps 1–3 pay up to a random round n ≤ 40, Exp 4 the whole episode. Base pay **UNSPEC**.
- Dropouts: a player who times out twice is replaced by a random bot; games with dropouts were excluded from training and analysis (about 10 %) **[S]**.

**Mechanisms.**

- Baselines **[S]**: round 1 equal (e = R/4); after that eᵢ = w·R/4 + (1−w)·R·cᵢ,ₜ₋₁/Σⱼcⱼ,ₜ₋₁. Equal w = 1, Mixed w = 0.5, Proportional w = 0. The Σc = 0 case is **UNSPEC**. Early baseline data kept a residual fraction of the pool (`mechanism_parameters_distribute_pool_ratio`); later mechanisms give out the whole pool.
- Interpolating **[S]**: w = (R/200)^k with k = 22, chosen on a grid log k ∈ [−5, 5] in steps of 0.1 against clones (log base **UNSPEC**; natural log gives e^3.1 ≈ 22.2).
- M1, M2 **[S]**: deterministic GNN policies; softmax over 5 logits (4 players + pool) times R. Weights are **not released**. Excluded players get tiny positive offers **[D]**.

**Resolved from the data in C1 (D33; `phasespace.citizens.validation`, training rows unless noted).**

- **Pool floor.** The recorded software applies R' = max(0.01, min(200, R − Σe + 1.4·Σc)); the pool never falls below 0.01.
- **Withheld amount.** Deployed baseline offers are round(shareᵢ·(R − 0.001), 4), computed in float32. About 0.001 coins stays in the pool every round, independent of R. `distribute_pool_ratio` is constant (1.0 in human rows, empty in BC rows), so its effect is unidentifiable.
- **Legacy games.** A frozen list of 7 early Exp 1 Equal/Proportional games drops the remainder R − Σe, so for those games R' = max(0.01, min(200, 1.4·Σc)). Every other game follows the full rule.
- **Σc = 0.** This happens only at the 0.01 floor; the offer is then an equal split.
- **Interpolating weight.** The deployed weight is w = round((R/200)^22, 1).
- **After depletion.** Rows are kept until the game ends (pool 0.01, offers 0.0022, c = 0), and the metrics include them.
- **Simulated offer rule.** C2–C4 simulate the *deployed* rule above (`mechanisms.baseline_offers`, default `rule="deployed"`), so simulated games face exactly the offers humans experienced. The paper's idealised whole-pool formula stays available as `rule="paper"`; the two differ by at most about 0.0011 coins per player.
- **Open (G1b).** 23 held-out pool rows in 5 RL Agent (M1) Exp 1 games sit about 0.01 coins below the rule. The anomaly is bounded but not yet diagnosed; it is root-caused only after the validation protocol freeze.

Other **UNSPEC** or contradictory points: the planner's policy-gradient estimator and whether gradients flow through the clones; episodes per update; observation ordering for the clones; the clones' train/validation split; M1 GRU size (16 in Methods, 32 in Table S2); Exp 4 group count (40 "sets" in Methods, 80 groups in Results).

## Data

- **Source:** `https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv`, from `github.com/google-deepmind/sustainable_behavior`. Data CC-BY 4.0, code Apache 2.0; attribute with doi 10.1038/s41467-025-58043-7.
- **File:** 214,185,378 bytes; last modified 29 May 2024; MD5 `dMl+sTp8gl3Y7gOvEE1Ymg==` (base64, from the storage Content-MD5 header). C1 also records the SHA-256.
- **Layout:** one row per game-round. A game is (`mech_name_by_player`, `launch_id`); rounds are `round_id` (0-based). `mech_name_by_player` = "<mechanism> <cohort>". Array columns are space-separated strings and need parsing. Player observations are rotated so the focal player comes first **[D]**.
- **Cohorts:** human `Exp 1`–`Exp 4`; simulated clone games `BC 1`, `BC 2` (a diagnostic reference, never a criterion).

| Cohort | Mechanisms (human groups) |
|---|---|
| Exp 1 | Equal, Mixed, Proportional, M1 (40 groups each) |
| Exp 2 | Proportional, Interpolating, M1 (40 each) |
| Exp 3 | M2, Interpolating (80 each) |
| Exp 4 | M2, Proportional; Games 1–3 per group |

- **Filters:** the notebook's `four_usable_players` and `completed` flags; C1 audits games per (mechanism, cohort) after filtering.
- **Training set:** human games under Equal, Mixed and Proportional only (Exp 1, 2 and 4). The clones' Train Set 1 (537 games, incl. pilot RL games) is not labelled in the CSV and is not reconstructed. Expected size: about 200 games or more; confirmed in C1.
- **Published numbers to reproduce** (independent references for G1): Exp 1 equal < proportional surplus z = 5.28, proportional > equal Gini z = 4.76, and the other Exp 1–4 Wilcoxon rank-sum z-values; Exp 3 M2 > interpolating surplus z = 2.35, interpolating < M2 Gini z = 5.64.
- **Metrics (notebook definitions):** surplus per player-round; Gini = 0.5 · mean|xᵢ − xⱼ| / mean(x) over players' mean rewards; depletion round (first R < 1, else 40); players kept active.

## Hypotheses (provisional; fixed in `protocol.md`)

- **H1 (transfer).** A planner trained on any single validated model beats the re-tuned interpolating rule k* on surplus under every *other* validated model. Any reversal in the cross-play matrix refutes H1.
- **H2 (robust planner).** A planner trained on a mixture or ensemble of validated models (optionally minimax over models) is never worse than a pre-declared tolerance below the best planner in each column.

## Citizen model classes

All are fitted only on training-mechanism human games, split by game into fit and early-stopping sets.

- **B1, GRU clone.** The paper's architecture: 9 inputs (four offers, four previous reciprocations, pool; all /200), FC encoder → GRU → FC → 10-bin reciprocation-fraction head; ensemble over seeds.
- **B2, parametric conditional cooperation.** Reciprocation fraction as a logistic/beta regression on own offer share, own and others' previous fractions and pool level, with a zero-inflation / exclusion component; about 10 parameters, maximum likelihood.
- **B3, learning model.** EWA / Roth–Erev over reciprocation-fraction bins with payoff reinforcement; a few parameters, maximum likelihood.
- **B4, LLM citizens.** Four independent headless `claude -p` sessions per game, one call per player per round; pinned and recorded `--model`; a neutral rules prompt written from the player instructions that does not name the paper, DeepMind or "RL agent"; every prompt and raw response archived. A pilot (2–3 games under Proportional) sets throughput; target 40 or more games per held-out regime. B4 is a validation and evaluation population only, never a PPO training environment. Optional B4-distil: a GRU fitted to LLM play, with its fidelity reported.

## Held-out regimes and validation

- **Interpolating** (Exp 2, Exp 3): closed-loop simulation under the exact rule.
- **M1** (Exp 1, 2) and **M2** (Exp 3, 4): one-step teacher-forced log-likelihood of human actions, plus closed-loop simulation under a **mechanism surrogate**. M1/M2 weights are not public, so the surrogate is a regression of their recorded offers on their recorded observations, fitted on mechanism rows only, with its fit error reported. It contains no human-response information beyond what the mechanism itself saw.
- **Criterion (proposed; fixed in `protocol-validation.md` before held-out data are opened):** a model is validated for a regime if its simulated group-level means fall inside the human 95 % bootstrap CI for at least 3 of the 4 metrics, and its one-step log-likelihood beats a marginal-frequency baseline.
- **Guard:** held-out human data are loaded only through an accessor that refuses unless the validation protocol is `frozen` and `--confirm-frozen` is passed; fitting code filters to training mechanisms.

## Planners and baselines

- **PPO planner (JAX).** Observation: previous offers, previous reciprocations and pool (/200). Permutation-equivariant set encoder over players plus a GRU (the analogue of the paper's GNN). Action: 5 shares through a Dirichlet or logistic-normal head; the deterministic mean is used at evaluation. Objective: total surplus over 40 rounds; Gini is reported, not optimised. Four citizens are sampled from the model's ensemble each episode.
- **Baselines.** Equal, Mixed, Proportional, published Interpolating k = 22, and **re-tuned Interpolating k\*** per citizen model on the paper's grid (ln k ∈ [−5, 5], step 0.1). k\* is the fair simple-rule competitor.
- **Correctness checks.** PPO solves a toy problem with a known optimum; against an analytically solvable citizen (fixed reciprocation fraction) it reaches the known optimal regime; fixed seeds reproduce.
- **Distillation.** Fit closed forms (the w(R) family and a small symbolic or tree fit of offers) to each planner and compare the discovered rules across citizen models.

## Gates

- **G1 — data and game (C1).** Every human and BC row satisfies the pool transition within print precision. Equal, Mixed, Proportional and Interpolating offers are reproduced from the recorded reciprocations and pool. The Σc = 0 and residual-pool cases are resolved from data. The published Exp 1–4 z-values are reproduced to two decimals. G1 is split (D33):
  - **G1a**, before the freeze: the training populations, the offer identities (including held-out Interpolating rows) and the four training-only Exp 1 z-values.
  - **G1b**, immediately after the validation protocol freeze: the z-values involving held-out mechanisms, and the root cause of the held-out pool anomaly.

  The NumPy–JAX 1e-9 agreement check moves to C3 (D33). *Stop:* any failure halts work until reconciled; a documented approximation needs the owner's approval.
- **G2 — citizen models (C2).** At least two model classes, at least one not a GRU, are validated on at least one held-out regime. *Stop:* otherwise the cross-model question collapses; report and ask the owner. If every class fails everywhere, that is a publishable null result (no model fitted on baselines predicts behaviour under new institutions); ask the owner before continuing.
- **G3 — planner pipeline (C3).** Against B1, the planner reproduces the paper's qualitative in-silico result (surplus above Interpolating k = 22) within 24 h or less per training run on the M4 Pro. *Stop:* otherwise report and ask the owner whether to scale down or switch to backpropagation through the differentiable clone.
- **G4 — main comparison (C4).** Runs only under a frozen, owner-approved `protocol.md` (seeds from pilot variance, cross-play matrix, paired episodes with common random numbers, bootstrap CIs over seeds, Holm correction across the H1 family, pre-declared minimum effect, compute ceiling), from clean trees carrying the protocol's config hash. *Stop:* exceeding the compute ceiling triggers only the protocol's pre-declared reductions; H1/H2 are decided as declared and either outcome is reported.

## Non-goals

- No new human-subject experiments, and no claim about real political systems or about fairness beyond the reported Gini.
- No exact reproduction of the paper's GNN planner or its unreleased weights.
- No LLM as a training environment; no backend; the browser only replays recorded runs.

## Limitations

- **Population.** All human data are UK/US Prolific participants. "Population shift" here means a change of *model*, not of culture.
- **No new humans.** New planners cannot be tested on new people. Their evaluation is across validated models (and LLM citizens), never against fresh human play.
- **Held-out regimes are few.** Three mechanisms, two of them evaluated through a surrogate whose error propagates into validation.
- **LLM citizens.** Sampling cannot be seeded through the CLI, so runs are stochastic but archived; the model may know the paper (checked with a contamination probe); the owner's plan usage limits cap the number of games.
- **Unspecified rules.** Every UNSPEC choice above is a modelling decision, documented and tested for sensitivity where it matters.

## Milestones

S0 (record, scan, setup), C1 (data and game, G1), C2 (citizen models and validation, G2), C3 (RL planner pipeline, G3), C4 (main comparison, G4), C5 (report and public page). See `docs/roadmap.md`.
