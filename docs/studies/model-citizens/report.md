# Model Citizens v1 — Technical report (draft, verdicts pending)

Status: **draft**. Every number below is read from a run artifact under `runs/` (§12). The cross-play and distillation numbers are **descriptive**. The confirmatory verdicts of `protocol.md` §9 (both tests, Holm per family) come from the hypothesis stage `citizens-c4-hypotheses-v1`, which has not run yet, and the C4 B4 runs (`citizens-c4-b4-*`) are still in progress. Their places are marked `<<TBD: …>>`; §13 lists them.

Coins are totals per game (4 players × 40 rounds) unless stated otherwise. δ = 16 coins is the protocol's minimum effect. "k\*[Y]" is the Interpolating rule re-tuned on model Y. "Planner[X]@Y" is a PPO planner trained on model X and evaluated on model Y.

## 1. Summary

Model Citizens asks whether RL policy designers trained against different behavioural models of people, all fitted to the same human data (Koster et al. 2025), arrive at the same mechanism. Four citizen models fitted only on baseline-mechanism play all pass a pre-registered held-out one-step likelihood test; only the GRU clone B1 also passes the closed-loop outcome test, on the Interpolating regime (§4). In cross-play under B1, the planners trained on the three one-step-only models lose 315–364 coins per game (24–28 %) to B1's re-tuned Interpolating rule, every seed of each, and the two models with the best next-choice likelihood (B3, B23) lose the most (§6, descriptive). Planners transfer between the two sister learning models (B3 ↔ B23) but not otherwise: 7 of the 12 off-diagonal cells have a crossed-bootstrap 95 % CI entirely below −δ. The published Interpolating rule (k = 22) stays within 0.6–2.1 % of each model's re-tuned rule and beats every planner trained on a different model under B1 and B2; under B3 and B23, whose closed-loop realism was rejected, the B3/B23-trained and mixture planners exceed it by 80–122 %. Confirmatory verdicts: H1 `<<TBD: hypotheses-v1 H1 verdict>>`; H2 `<<TBD: hypotheses-v1 H2 verdict (H2a) and H2b>>`; H2-B4 `<<TBD: hypotheses-v1 H2-B4 verdict, B4-calibrated>>`. The judge of every policy is a simulator; no planner was tested with people.

**Working headline, checked against the artifacts.**
- "Models that best predict the next move yield the policies that do worst under the only closed-loop-validated model" holds descriptively for B3 and B23. But Planner[B2] also loses 24 %, and B2's next-choice likelihood equals B1's on R-I. The cleaner statement is: every one-step-only model yields a planner that loses about a quarter of surplus under B1.
- "RL planners don't transfer" needs the B3 ↔ B23 exception.
- "The Interpolating rule is robust across all" holds relative to the re-tuned rules, and relative to cross-trained planners on B1 and B2. It does not hold in the B3 and B23 columns (§6.4).

## 2. Question and context

**Question (D35, `protocol.md` §2).** Do citizen models that each pass a different held-out test (next-choice prediction or whole-game outcomes) lead PPO planners to the same mechanism, and do the planners transfer between models? This tests empirically whether one-step behavioural validation is enough for policy design. It is the Lucas critique (Lucas 1976) applied to AI mechanism design.

**Context.**
- **Koster et al. 2025** (*Nature Communications* 16:2824). A deep-RL manager was trained against GRU behavioural clones of human players in a depletable common-pool game. Its mechanisms (M1, M2) and a distilled Interpolating rule, w = (R/200)^22, beat baselines with 4,952 Prolific participants. The clones were compared with humans visually. The paper did predict the unseen Interpolating rule's success prospectively, with clones that had not seen it. Its second clone set (BC2) was trained on Interpolating and RL-agent games, so it is not a held-out model.
- **Koster et al. 2022** (*Nature Human Behaviour*). LSTM clones were validated on held-out participants and endowment conditions. A second mechanism was trained against unvalidated "rational" learners, and new humans preferred the clone-trained mechanism (57.2 %). So it has been shown once that the model of people can matter, for one validated and one unvalidated model, judged by humans.
- **Tacchetti et al. 2025** (*PNAS*) name the problem: simulated players diverge from humans, imitation fails out of distribution, and the Lucas critique applies. They suggest training against a diverse population, without testing it.

**What the 2025 paper left open.** It had no second model class, no held-out likelihood and no cross-model evaluation of the designed mechanism. It did not test whether its distilled rule is model-invariant. v1 fills these gaps in silico, with four model classes under one frozen validation protocol (`protocol-validation.md` v1, D34) and one frozen cross-play protocol (`protocol.md` v1, D36).

**Scope disclosure (D35).** The original question assumed several closed-loop-validated model classes; G2 found one. The C4 question above was set *after* the G2 result was seen, and *before* any planner was trained or any C4 comparison was run.

## 3. Data and game (C1, G1)

**Source.** `sustainable_behavior.csv` from `github.com/google-deepmind/sustainable_behavior` (data CC BY 4.0, code Apache 2.0; attribution doi:10.1038/s41467-025-58043-7).
- File: 214,185,378 bytes; MD5 (base64) `dMl+sTp8gl3Y7gOvEE1Ymg==`; SHA-256 `9fd30106ef858772a37fbbf1dfa5f9f8553c88ced0e39e0865b76a945b4638d0`.
- Content: 167,782 rows and 4,264 games. These are 280 human training games (Equal, Mixed, Proportional; 9,832 rows), 400 held-out human games (Interpolating, M1, M2) and 3,584 simulated BC games (`citizens-c1-data-audit-v1`).

**Game.** 4 players; pool R₀ = 200, which is also the cap; growth factor 1.4 on reciprocated coins; 40 rounds (Exps 1–3). Rules recovered from the data (D33):
- pool floor: R′ = max(0.01, min(200, R − Σe + 1.4Σc));
- deployed offers withhold 0.001 coins and are rounded to 4 decimals in float32;
- the Interpolating weight is quantised: w = round((R/200)^22, 1);
- 7 early Exp 1 games drop the remainder R − Σe.

Simulations use the deployed offer rule, so simulated players face the offers humans saw.

**G1 replay checks** (`citizens-c1-data-audit-v1`, status `validated` for the G1a scope; `citizens-c1-g1b-v1`, status `validated`):

| Identity (max abs residual, coins) | Training human | BC | Threshold |
|---|---|---|---|
| Pool transition | 2.1e-5 | 4.3e-5 | 1e-4 |
| Equal / Mixed / Proportional offers | 3.8e-6 / 1.03e-4 / 1.8e-5 | 3.8e-6 / 1.07e-4 / 1.07e-4 | 1.5e-4 |
| Interpolating offers | held-out human rows: 9.9e-5 | 1.22e-4 | 1.5e-4 |
| Reward identity | 1.0e-5 | 1.3e-5 | 1e-4 |

**Published z-values reproduced.** We recompute 17 Wilcoxon rank-sum z-values of Koster et al. 2025, and all 17 match the printed values:
- the 4 training-only Exp 1 values in C1 (largest |Δz| 0.0031);
- the 13 values involving held-out mechanisms in G1b (largest |Δz| 0.0041; tolerance 0.005, or 0.05 for the value printed as 2.9).

Examples: Exp 1 Proportional > Equal surplus, 5.28 (reproduced 5.283); Exp 3 M2 > Interpolating surplus, 2.35 (2.351), and Gini, 5.64 (5.638).

**Held-out pool anomaly.** 23 pool rows in 5 RL Agent (M1) Exp 1 games sit 0.0100 coins below the full rule (max |residual| 0.010001). No other game is affected, and no human game's depletion round changes under it (`citizens-c1-g1b-v1`). The run lists the rows; by its manifest, the root cause is "the owner's reading", and no diagnosis is recorded. Human metrics use the recorded rows.

## 4. Citizen models and their two exams (C2)

**Models.** All are fitted on the fit split of the 280 training games only (`citizens-c2-fit-v1`):
- **B1:** the paper's GRU clone architecture, an ensemble of 4 members;
- **B2:** parametric conditional cooperation (12 parameters: zero hurdle, full-return spike, beta-binomial interior);
- **B3:** EWA learning over 11 reciprocation levels (6 parameters; Camerer & Ho 1999);
- **B23:** B3 plus a conditional-cooperation pull (7 parameters).

Controls: B0 (payoff-maximising: always returns 0), M0 (marginal frequencies; M0-sim in closed loop) and P0 (persistence; diagnostic).

**Two exams** (`protocol-validation.md` §8, frozen before held-out data were read):
- **(b) One-step likelihood.** Δℓ is the per-decision log-likelihood gain over M0 on held-out human decisions (teacher-forced). It passes if the 95 % group-bootstrap CI lower bound is above 0.
- **(a) Closed-loop outcomes.** It passes if, for at least 3 of the 4 metrics (surplus, Gini, depletion round, players active), the simulated mean falls inside the human 95 % CI. The CI is widened by a resolution tolerance τ: 0.1 coin, 0.005, 0.5 rounds and 0.1 players.
- **(c) Must beat the marginal model.** Where M0-sim passes (a), a lower mean normalised W1 than M0-sim's is also required. This never applied.
- **Regimes:** R-I (Interpolating, 120 groups, exact rule), R-M1 (80 groups, M1 surrogate) and R-M2 (80 groups, M2 surrogate).

**Results** (`citizens-c2-heldout-eval-v1`, status `validated`; G2 failed, `g2_passed` = 0). Δℓ is in nat, with the 95 % CI lower bound in brackets:

| Model | Δℓ R-I | Δℓ R-M1 | Δℓ R-M2 | In range R-I / R-M1 / R-M2 | Mean norm. W1 R-I / R-M1 / R-M2 | Label |
|---|---|---|---|---|---|---|
| B1 | 0.104 [0.075] | 0.081 [0.041] | 0.089 [0.037] | 3 / 1 / 2 | 0.26 / 0.45 / 0.36 | validated (closed-loop) on R-I; one-step on all three |
| B2 | 0.103 [0.058] | 0.212 [0.154] | 0.175 [0.113] | 1 / 2 / 2 | 0.36 / 0.52 / 0.39 | one-step only |
| B3 | 0.180 [0.143] | 0.301 [0.243] | 0.302 [0.250] | 0 / 0 / 1 | 0.85 / 0.64 / 0.52 | one-step only |
| B23 | 0.174 [0.146] | 0.256 [0.207] | 0.235 [0.179] | 0 / 0 / 1 | 1.13 / 0.94 / 0.83 | one-step only |
| B0 (control) | −2.876 | −2.627 | −2.739 | 0 / 0 / 0 | 2.03 / 2.16 / 2.39 | rejected everywhere, as required |
| P0 (diagnostic) | 0.116 | 0.223 | 0.231 | — | — | passes (b) everywhere |
| M0-sim | — | — | — | 2 / 1 / 1 | 0.35 / 0.46 / 0.45 | control |

**Surrogates.** R-M1 is one-step only for every model, because the M1 surrogate failed acceptance on human rows. Its median offer error was 1.18 and 1.04 coins (Exp 1, Exp 2), against a limit of 1.0. Its p90 errors (4.79, 4.66; limit 5.0) and exclusion agreement (0.980, 0.980) were within their limits. The M2 surrogate was accepted (Exp 3: median 0.39, p90 2.68, agreement 0.990).

**Inversion.** The models that predict the next choice best simulate whole games worst. Closed-loop means on R-I against the human 95 % CI:

| R-I | Surplus / player-round | Gini | Depletion round | Players active |
|---|---|---|---|---|
| Human | 7.98 [7.33, 8.62] | 0.180 [0.160, 0.200] | 35.2 [33.4, 36.9] | 2.24 [1.96, 2.53] |
| B1 | 7.92 | 0.157 (in range only through τ) | 36.6 | 2.90 |
| B2 | 8.58 | 0.121 | 38.1 | 2.75 |
| B3 | 5.14 | 0.153 | 23.4 | 0.41 |
| B23 | 3.87 | 0.140 | 18.5 | 0.08 |
| M0-sim | 7.10 | 0.173 | 35.2 | 1.24 |

- B3 and B23 deplete the pool far too early. On R-M2 too: 31.1 and 25.9, against the human 36.5 [34.5, 38.3].
- B2 errs the other way. It keeps the pool too long on R-I, and its Gini is too low on every regime: 0.12 / 0.14 / 0.16, against the human 0.18 / 0.26 / 0.29.
- **In-sample check** (training mechanisms, non-gating). B1 fails Equal: 1 of 4 metrics, with a simulated depletion round of 23.5 against the human 13.3. B1 passes Mixed and Proportional. B2 passes Proportional only. B3 and B23 pass Equal only.

**Sensitivity of B1's label to τ.** B1's R-I pass counts Gini as in range only because of the tolerance: 0.157, against a lower bound of 0.160, with τ = 0.005. Without τ, B1 would have 2 of 4 metrics on R-I, and no trainable model would be closed-loop validated. τ is part of the frozen criterion, so the label stands. But the protocol's justification for τ ("this matters only when a human CI is degenerate") does not describe this case.

**B4, LLM citizens** (evaluation only; `claude-haiku-5-5` through Claude Code CLI 2.1.304, `b4-prompt-v2`; `citizens-c2-b4-eval-v1`, status `validated`; 6 runs × 40 games, 26,703 calls, no discarded game). Metrics in range and mean normalised W1, with M0-sim's W1 in brackets:

| Variant | R-I | R-M1 (diagnostic) | R-M2 |
|---|---|---|---|
| plain | 0 of 4, 0.54 (0.35): not validated | 4 of 4, 0.20 (0.46) | 0 of 4, 0.69 (0.45): not validated |
| calibrated (type weights 0.7 / 0.0 / 0.3) | 1 of 4, 0.40 (0.35): not validated | 3 of 4, 0.34 (0.46) | **3 of 4, 0.32 (0.45): validated (closed-loop)** |

- **B4-calibrated's R-M2 pass is narrow.** Gini (0.263) is inside the human CI [0.261, 0.317]. Depletion (34.45) is *below* the lower bound 34.46, and players active (2.90) is above the upper bound 2.84; both count as in range only through τ. Surplus misses: 10.1, against [7.84, 9.66].
- **C4 role.** By the rule fixed beforehand (`protocol.md` §9.4, D36), B4-calibrated is a validated evaluation population, so H2-B4 is confirmatory for it. B4-plain is descriptive.
- **Plain B4** errs in opposite directions: on R-I it is too cooperative (no depletion by round 40); on R-M2 it depletes early, with a high Gini.

**Contamination probe** (`citizens-c2-b4-probes-pilot-v1`, 16 calls). Every paper-specific answer denied knowledge of the 2025 study, and none stated k = 22, the Interpolating rule or a published result. The 10 keyword flags fired on those denials, on mentions of the 2022 study (which the model knows) and on the word "sustainable". The label for every B4 result is "no contamination evidence for the 2025 study".

## 5. Planners (C3, G3, C4 training)

**PPO planner** (JAX; `phasespace.citizens.rl`).
- *Observation:* previous offers, previous reciprocations and the pool, all /200.
- *Network:* a permutation-equivariant set encoder with a GRU (hidden 32), and a logistic-normal head over 5 shares (4 players plus pool retention).
- *Hyperparameters:* frozen in C3 and chosen on B1 only: 1,024 games per iteration, 4 epochs, 8 minibatches, learning rate 1e-3, clip 0.2, γ = 1, λ = 0.95, entropy coefficient 0, initial log σ −0.5.
- *Objective:* total surplus. Gini is reported, not optimised. Each game draws 4 citizens from the training model's ensemble.
- *Correctness:* tests in `tests/citizens/test_citizens_rl_ppo.py` cover a toy optimum, a fixed-fraction optimum, determinism, permutation equivariance and bit-identical default training. The JAX twin matches the NumPy reference (`citizens-c3-baselines-v1`): pmf differences ≤ 9.4e-15, and closed-loop max |z| ≤ 2.70 against a limit of 4.5.

**G3 passed** (`citizens-c3-g3-v1`; C3 seeds 1–3, 500 iterations, never C4 inputs). Against B1, the deterministic planner beat Interpolating k = 22 by 286 coins [258, 313] and k\*[B1] by 258 [230, 286]. Its per-seed means were 1558.6, 1540.2 and 1536.0, at 0.46 h per run.

**Baseline rules on each model** (`citizens-c3-baselines-v1`; 1,000 games, key 31000).
- The ranking of the deployed rules already depends on the model. Under B1, Proportional (1124) beats Mixed (881); under B2, Mixed (1084) beats Proportional (918).
- Re-tuned k\* (key 31001; ln k grid, step 0.1): B1 9.03, B2 9.03, B3 27.1, B23 99.5.

**C4 training.** There are 35 runs, all at commit `4b1f1dd` with `git_dirty = false`, `c4_config_sha256` `47ebefeb…` and C3 hash `4a312211…`:
- main rows Planner[B1], [B2], [B3] and [B23], seeds 1001–1005;
- no-retention rows (4-way head, Σe = R − 0.001), seeds 2001–2003;
- the mixture planner (training model drawn uniformly per game), seeds 3001–3003.

Each run has 1,000 iterations (40.96 M planner decisions) and took 0.82–1.17 h, 33.9 h in total. The runs were concurrent (the protocol planned 5 at a time); they finished between 2026-10-20 18:45 and 2026-10-21 00:48 UTC. The non-LLM ceiling was 40 h wall-clock.

**Convergence labels.** By §6.1, a run is "not converged" if the plateau iteration of its monitoring curve exceeds 900, or its tail slope exceeds 16 coins per 100 iterations.
- The training manifests do not record these labels; the hypothesis stage computes them.
- A preview with the same definition (Appendix A) marks 3 of 35 runs: `citizens-c4-train-b2-noret-s2001-v1` (plateau 930), `citizens-c4-train-b23-s1002-v1` (plateau 960) and `citizens-c4-train-mix-s3002-v1` (plateau 920).
- Record: `<<TBD: hypotheses-v1 convergence labels (not converged) per run>>`.
- Optimiser-limited rows: `<<TBD: hypotheses-v1 own-model-limited family>>`.

## 6. C4 cross-play

### 6.1 Protocol (frozen, `protocol.md` v1, D36)

- **Design.** Every row is evaluated on every trainable column, on the same 1,000 games (key 41000; common random numbers within a column), per seed, with the deterministic policy (confirmatory) and the sampled policy.
- **Tests (§9.2).** A crossed bootstrap over seeds and games (B = 10,000, seed 9101) and a seed-level t-test. A directional claim is **confirmed** only if both reject under Holm within its family, at α = 0.05.
- **Families:**
  - H1-reversal and H1-transfer: the 12 off-diagonal cells, θ = ∓δ;
  - H2a-loss and H2a-noninferior: Planner[B2], [B3] and [B23] on B1, against k\*[B1];
  - H2b-loss: the same cells, against Planner[B1];
  - own-model adequacy: the 4 diagonal cells;
  - the secondary S1 and S2 families.
- **δ = 16 coins:** C2's surplus tolerance, and about one B1 seed SD.

The numbers below are point estimates, crossed-bootstrap 95 % CIs and per-seed counts recorded in `citizens-c4-crossplay-v1` (`crossplay.json`). They are **not verdicts**.

### 6.2 Surplus matrix

Deterministic policy; mean over seeds, with the [min, max] of the per-seed means:

| Row | B1 | B2 | B3 | B23 |
|---|---|---|---|---|
| Planner[B1] | **1577** [1568, 1589] | 1195 [1131, 1245] | 660 [568, 756] | 572 [520, 643] |
| Planner[B2] | 986 [949, 1009] | **1418** [1381, 1448] | 716 [569, 852] | 552 [484, 609] |
| Planner[B3] | 938 [903, 1001] | 1031 [1022, 1046] | **1619** [1595, 1637] | 1258 [1223, 1300] |
| Planner[B23] | 937 [908, 981] | 1016 [1002, 1058] | 1543 [1470, 1594] | **1328** [1299, 1357] |
| Planner[Mix] (S2) | 1287 [1179, 1357] | 1306 [1268, 1342] | 1459 [1436, 1487] | 1178 [1156, 1193] |
| Equal | 761 | 850 | 385 | 412 |
| Mixed | 894 | 1093 | 448 | 454 |
| Proportional | 1138 | 945 | 571 | 551 |
| Interpolating k = 22 | 1284 | 1374 | 811 | 598 |
| k\*[B1] = k\*[B2] (k = 9.03) | **1301** | **1397** | 762 | 562 |
| k\*[B3] (k = 27.1) | 1281 | 1371 | **816** | 602 |
| k\*[B23] (k = 99.5) | 1252 | 1343 | 814 | **610** |

### 6.3 Planner minus k\*[Y] in column Y

Deterministic policy. Mean [crossed-bootstrap 95 % CI], with the number of seeds whose difference d_s is above +δ / below −δ (of 5). The diagonal cells are own-model cells.

| Planner | B1 | B2 | B3 | B23 |
|---|---|---|---|---|
| Planner[B1] | +276 [+251, +303] (5/0) | −201 [−242, −165] (0/5) | −156 [−218, −92] (0/5) | −38 [−84, +11] (1/3) |
| Planner[B2] | −315 [−354, −276] (0/5) | +21 [−10, +50] (3/0) | −100 [−197, −7] (1/3) | −59 [−104, −14] (0/3) |
| Planner[B3] | −362 [−406, −315] (0/5) | −366 [−393, −337] (0/5) | +803 [+764, +841] (5/0) | +648 [+607, +688] (5/0) |
| Planner[B23] | −364 [−403, −325] (0/5) | −381 [−413, −348] (0/5) | +727 [+673, +778] (5/0) | +717 [+680, +755] (5/0) |

Relative to k\*[Y]:
- under B1: Planner[B2] −24.2 %, Planner[B3] −27.9 %, Planner[B23] −28.0 %; Planner[B1] +21.3 %;
- under B2: Planner[B1] −14.4 %, Planner[B3] −26.2 %, Planner[B23] −27.3 %.

Descriptive reading:
- **H1 cells.** 7 of the 12 off-diagonal cells have a CI entirely below −δ: Planner[B1]@B2 and @B3; Planner[B2]@B1; Planner[B3]@B1 and @B2; Planner[B23]@B1 and @B2. 2 cells have a CI entirely above +δ: Planner[B3]@B23 and Planner[B23]@B3. This CI leaves 3 cells unresolved: Planner[B1]@B23, Planner[B2]@B3 and Planner[B2]@B23. Verdict: `<<TBD: hypotheses-v1 H1 verdict; confirmed H1-reversal and H1-transfer cells; raw and Holm p-values, both tests>>`.
- **H2 cells** (column B1). Planner[B2], [B3] and [B23] lie 315–364 coins below k\*[B1], and every seed is more than δ below. Verdict: `<<TBD: hypotheses-v1 H2a-loss / H2a-noninferior verdicts and p-values>>`.
- **Own-model cells.** The gains are large for B1 (+276), B3 (+803) and B23 (+717). B2's gain (+21 [−10, +50]) is about δ. Verdict: `<<TBD: hypotheses-v1 own-model-gain and own-model-limited verdicts>>`.

### 6.4 Regret and the Interpolating rule

Regret Planner[Y]@Y − Planner[X]@Y, deterministic, with crossed-bootstrap 95 % CIs. It is descriptive, with no multiplicity correction (§9.3).

| Planner[X] | B1 | B2 | B3 | B23 |
|---|---|---|---|---|
| Planner[B1] | — | +222 [+177, +267] | +959 [+890, +1023] | +756 [+698, +813] |
| Planner[B2] | +591 [+554, +629] | — | +903 [+805, +1003] | +776 [+722, +832] |
| Planner[B3] | +639 [+593, +682] | +386 [+351, +421] | — | +70 [+33, +107] |
| Planner[B23] | +641 [+602, +678] | +402 [+363, +439] | +77 [+38, +121] | — |
| Planner[Mix] | +290 [+215, +391] | +112 [+68, +155] | +160 [+126, +197] | +150 [+116, +185] |

The H2b-loss cells are the column-B1 regrets of Planner[B2], [B3] and [B23]: 591, 639 and 641 coins. Verdict: `<<TBD: hypotheses-v1 H2b-loss verdict>>`.

**Interpolating k = 22.**
- In every column it is within 5–22 coins (0.6–2.1 %) of the column's re-tuned k\* on the C4 games: B1 −16.9, B2 −22.4, B3 −4.6, B23 −13.0.
- Under B1 and B2 it beats every planner trained on another model. On B1 it equals the mixture planner: +4 [−97, +79].
- Under B3 and B23 it lies 661–808 coins below the B3- and B23-trained planners, and 580–648 coins below the mixture planner. Those columns are the closed-loop-rejected models, and the planners' own-model gains there are not validated (§6.6).

"Robust" therefore means robust relative to the re-tuned rules, and relative to cross-trained planners on B1 and B2. By the worst-column ratio to the column's own planner, the mixture planner (0.82) is more robust than k = 22 (0.45) within this matrix.

### 6.5 Gini, kept share and decisions

- **Gini** (deterministic). The planners trained on B3 and B23 are much less equal under B1 (0.394, 0.391) and B2 (0.300, 0.299) than k\*[B1] (0.150, 0.111), and close to Proportional (0.424, 0.409). Under B1, Planner[B1] (0.134) and Planner[B2] (0.098) stay at or below the Interpolating rules.
- **Kept share**, coin-weighted: pooled over rounds and games with R ≥ 1; the endgame is rounds 39–40.
  - Planner[B1] keeps 3.5 % on B1 (endgame 3.0 %).
  - Planner[B2] keeps 5.8 % on B2, and 7.5–14.0 % on the other columns.
  - Planner[B3] and [B23] keep 2.9–3.9 %.
  - The training mechanisms never retain coins, so retention is off the training distribution for every model (`protocol.md` §6.3).
- **Decisions per game**, which drive the number of B4 calls, range from 51 to 156 in the main cells.

### 6.6 Possible exploitation (diagnostics; labels pending)

§9.5 labels a confirmed own-model gain "possible exploitation" if either holds in the planner's own column:
- its mean B1-member total-variation (TV) disagreement exceeds 1.5 × that of k = 22 in the same column;
- (B2 only) its input out-of-range or off-grid fraction exceeds k = 22's by more than 0.05.

| Cell | B1-member TV (× k = 22) | Input support (vs k = 22) |
|---|---|---|
| Planner[B1]@B1 | 0.193 (× 1.13) | — |
| Planner[B2]@B2 | 0.236 (× 1.13) | out-of-range 0.0000 / off-grid 0.0008 (k = 22: 0.0000 / 0.0009) |
| Planner[B3]@B3 | 0.270 (× 1.35) | state novelty: 0.0003 / 0.0011 (k = 22: 0.0000 / 0.0019) |
| Planner[B23]@B23 | 0.214 (× 1.24) | state novelty: 0.0002 / 0.0017 (k = 22: 0.0000 / 0.0015) |

- The largest TV ratio in the whole main matrix is 1.37 (Planner[B2]@B23 and Planner[B3]@B23), and no input-support difference reaches 0.05.
- Labels: `<<TBD: hypotheses-v1 possible-exploitation labels and "on novel states" annotations>>`.
- **Weakness.** The TV index measures disagreement among B1's members. For B3 and B23, the support diagnostics cover inputs those models do not read. Neither can detect a planner exploiting the internal dynamics of B3 or B23. So the absence of a flag on their own-model gains (+98 % and +118 % over k\*) is not evidence against exploitation.

### 6.7 Sampled policy

The sampled policy (noise key 41001) gives the same signs in every main cell. Its means differ from the deterministic ones by at most 14 coins: for example, Planner[B1]@B1 1563 against 1577, and Planner[B2]@B2 +15 [−15, +44] against +21 over k\*[B2]. Secondary verdicts, and any disagreement with the deterministic ones: `<<TBD: hypotheses-v1 sampled-mode families>>`.

### 6.8 S1, no-retention ablation (secondary, 3 seeds)

Planner-noret minus k\*[Y], deterministic, mean [CI]:

| Planner-noret | B1 | B2 | B3 | B23 |
|---|---|---|---|---|
| [B1] | +155 [+121, +188] | −157 [−195, −120] | −96 [−218, −11] | −2 [−72, +52] |
| [B2] | −42 [−65, −19] | +7 [−15, +29] | −170 [−239, −93] | −66 [−110, −15] |
| [B3] | −358 [−398, −323] | −364 [−392, −336] | +734 [+694, +773] | +620 [+585, +655] |
| [B23] | −354 [−403, −302] | −375 [−412, −337] | +720 [+681, +759] | +684 [+643, +726] |

- Retention accounts for most of Planner[B2]'s loss under B1: −315 with retention, −42 without.
- It accounts for part of Planner[B1]'s own-model gain: +276 with retention, +155 without.
- It does not account for the B3/B23 planners' losses under B1 and B2: −354 to −375 without retention.
- *Confound (§9.7):* with entropy coefficient 0, the 4-way head still explores in fewer dimensions than the 5-way head.
- Verdict ("attributable to retention" or not): `<<TBD: hypotheses-v1 S1 families and retention verdict>>`.

### 6.9 S2, robust mixture planner (secondary, 3 seeds)

Planner[Mix] minus k\*[Y]: B1 −13 [−114, +62] (2 seeds above +δ, 1 below −δ); B2 −91 [−130, −54]; B3 +643 [+600, +686]; B23 +567 [+530, +605]. Its regret against each column's own planner is 290, 112, 160 and 150 coins (§6.4).

On the closed-loop-validated model, the mixture planner neither gains nor loses against k\*[B1] beyond its seed spread; its gains are on B3 and B23. Verdicts: `<<TBD: hypotheses-v1 S2-reversal and S2-transfer families>>`.

### 6.10 Planners on LLM citizens (B4)

Rows (`protocol.md` §6.5): Planner[B1], [B2], [B3] and [B23] (seed 1001) and k\*[B1], each on B4-plain and on B4-calibrated, with 40 games per row. The C2 k = 22 runs are reference rows. All C4 B4 runs use the current CLI, under the D37 deviation. None of these runs exists yet in `runs/`.

| Row | B4-calibrated (confirmatory) | B4-plain (descriptive) |
|---|---|---|
| Planner[B1] | `<<TBD: citizens-c4-b4-calibrated-b1-v1 mean surplus [CI], Gini, depletion, players active>>` | `<<TBD: citizens-c4-b4-plain-b1-v1 …>>` |
| Planner[B2] | `<<TBD: …-calibrated-b2-v1>>` | `<<TBD: …-plain-b2-v1>>` |
| Planner[B3] | `<<TBD: …-calibrated-b3-v1>>` | `<<TBD: …-plain-b3-v1>>` |
| Planner[B23] | `<<TBD: …-calibrated-b23-v1>>` | `<<TBD: …-plain-b23-v1>>` |
| k\*[B1] | `<<TBD: …-calibrated-kstar-b1-v1>>` | `<<TBD: …-plain-kstar-b1-v1>>` |
| k = 22 (C2 reference, CLI 2.1.304, R-I) | 9.15 coins per player-round (≈ 1463 per game), `citizens-c2-b4-calibrated-interpolating-v1` | 11.41 (≈ 1825 per game), `citizens-c2-b4-plain-interpolating-v1` |

- **H2-B4** (B4-calibrated; unpaired game bootstrap; Holm over 3 cells; conditional on seed 1001): `<<TBD: hypotheses-v1 H2-B4 B4-loss / B4-noninferior / B4-b verdicts>>`.
- **Run accounting:** B4 call counts, discards, infrastructure errors, and the CLI version per run: `<<TBD: C4 B4 manifests and hypotheses-v1 B4 block>>`.
- **Power (§9.4):** with game SDs of 300–550 coins and 40 games per row, the SE of a difference is about 70–120 coins, so only large effects can be confirmed.

## 7. Mechanism agreement (distillation, §9.6)

Each planner seed is played on its own training model, on 1,000 distillation games (key 41002), and distilled into three rules (`citizens-c4-distill-v1`):
- (i) a binned w(R) rule, with a kept share κ per bin (10 bins of R/200);
- (ii) a one-parameter Interpolating fit, w = (R/200)^k, on the paper's ln k grid;
- (iii) a CART tree of depth 3.

Distilled rules are scored **in sample**, on the games they were fitted on.

**Fidelity and distilled k** (per row; ranges over seeds):

| Row | w-rule R² | Distilled ln k (k) | k-fit R² | Tree R² | Recovery of planner gain |
|---|---|---|---|---|---|
| Planner[B1] | 0.68–0.84 | 5.0 (148.4, grid edge) in 4 of 5 seeds; 1.8 (6.0) in 1 | 0.47–0.60 | 0.58–0.77 | −1.12 to −0.99 |
| Planner[B2] | 0.56–0.71 | 1.1–1.8 (3.0–6.0) | 0.53–0.69 | 0.58–0.60 | unstable (−4.7 to +18.9; planner − k\* is −10 to +55 on these games) |
| Planner[B3] | 0.82–0.87 | 3.1–4.1 (22.2–60.3) | 0.52–0.64 | 0.86–0.89 | 0.10–0.16 |
| Planner[B23] | 0.85–0.91 | 5.0 (148.4, grid edge) in 5 of 5 | 0.39–0.55 | 0.87–0.90 | 0.10–0.15 |
| Planner-noret[B1] | 0.68–0.88 | 5.0 in 2 of 3; 1.6 (5.0) in 1 | 0.45–0.56 | 0.61–0.83 | −2.61 to −1.99 |
| Planner-noret[B2] | 0.61–0.72 | 0.6–1.1 (1.8–3.0) | 0.55–0.66 | 0.55–0.57 | −26.1 to −7.2 |
| Planner-noret[B3] | 0.86–0.88 | 3.2–4.0 (24.5–54.6) | 0.46–0.61 | 0.87–0.91 | 0.18–0.20 |
| Planner-noret[B23] | 0.85–0.91 | 5.0 in 2 of 3; 4.1 in 1 | 0.35–0.50 | 0.87–0.91 | 0.16–0.20 |

Recovery = (S_w-rule − S_k\*[X]) / (S_planner − S_k\*[X]).
- Every w-rule has R² ≥ 0.5, the §9.6 fidelity bar.
- Yet none recovers a meaningful share of its planner's gain. The B1 w-rules score 263–294 coins *below* k\*[B1], and the B3/B23 w-rules recover 10–20 % (main rows 10–16 %).
- The R² of the share fit and the surplus recovery disagree: the planners' gains are not carried by a w(R) rule.

**Seed-mean w(R)**, by bins of R/200 from low to high pool (w = 1 is Equal, w = 0 is Proportional):

| Row | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Planner[B1] | 0.56 | 0.33 | 0.30 | 0.28 | 0.26 | 0.22 | 0.20 | 0.19 | 0.20 | 0.19 |
| Planner[B2] | 0.59 | 0.18 | 0.16 | 0.18 | 0.20 | 0.21 | 0.24 | 0.29 | 0.44 | 0.87 |
| Planner[B3] | 0.35 | 0.11 | 0.08 | 0.06 | 0.04 | 0.01 | 0.01 | 0.03 | 0.14 | 0.32 |
| Planner[B23] | 0.58 | 0.24 | 0.12 | 0.07 | 0.04 | 0.03 | 0.01 | 0.03 | 0.11 | 0.20 |
| k = 22 (deployed, quantised; recovery test) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.03 | 0.65 |

Descriptively, the four planners differ in shape:
- B1-trained planners stay near w ≈ 0.2 over most of the pool range.
- B2-trained planners rise towards Equal near a full pool, which is the Interpolating rule's direction.
- B3- and B23-trained planners are almost purely Proportional at middle pool levels.
- Every planner is more Equal than k = 22 at low pool.

Every bin has at least 1,288 informative rows per seed. Pairwise L1 distances, within-model references and the "same / different / mixed" summary: `<<TBD: hypotheses-v1 mechanism-agreement block (L1 matrix, labels, κ L1, summary sentence)>>`.

**Recovery test (method bias).** On trajectories of the deployed, quantised rules, the distiller recovers:
- k\*[X] exactly, on every model (ln k bias 0.000);
- k = 22 as 22.2, the nearest grid point (ln k bias +0.009);
- k\*[B23] on B23 as 110.0 (+0.1 in ln k, one grid step).

The max |w error| is ≤ 1.6e-7. The method is unbiased on rules from the family. The grid-edge values of Planner[B1] and Planner[B23] therefore reflect misfit of the family (k-fit R² 0.39–0.60), not method bias.

**Distilled-rule cross-play.** Seed means, minus k\*[Y] in column Y; the distilled k is played through the deployed rounding path:

| Planner | B1: w-rule / distilled k | B2 | B3 | B23 |
|---|---|---|---|---|
| Planner[B1] | −280 / −51 | −248 / −47 | −161 / −18 | −53 / −16 |
| Planner[B2] | −396 / −53 | −160 / −9 | −301 / −189 | −157 / −115 |
| Planner[B3] | −100 / −38 | −155 / −39 | +104 / +1 | +46 / −6 |
| Planner[B23] | −127 / −62 | −205 / −59 | +54 / +3 | +85 / −0 |

- No distilled rule beats k\*[Y] in the B1 or B2 column.
- The distilled-k rules are within ±6 coins of k\*[Y] in their own family's columns, and up to 62 coins below k\*[B1] under B1.
- The mechanism that the network carries does not survive distillation into the rule family the 2025 paper used.

## 8. E1: exploratory credibility check

**E1 is exploratory** (`citizens-e1-ranking-v1`; code on branch `worktree-agent-a0b11da95451dae63`, commit `43439b7`). It was designed after the held-out outcomes had been seen. It is not part of any frozen protocol, and it cannot change a C2 label or a C4 verdict.

It asks whether each simulator ranks pairs of human-tested mechanisms in the same direction as the published Wilcoxon comparisons. Pairs involving M1 are excluded (its surrogate was rejected), Exp 4 pairs are not scorable, and B4 has simulations only on held-out regimes.

| Model | Sign agreements / scorable pairs | Held-out pairs |
|---|---|---|
| B0 | 0 / 8 | 0 / 4 |
| M0-sim | 5 / 8 | 3 / 4 |
| B1 | **8 / 8** | 4 / 4 |
| B2 | 7 / 8 | 3 / 4 |
| B3 | **8 / 8** | 4 / 4 |
| B23 | **8 / 8** | 4 / 4 |
| B4-plain | 1 / 2 | 1 / 2 |
| B4-calibrated | 2 / 2 | 2 / 2 |

Only one held-out pair discriminates among the trainable models: Exp 3 M2 > Interpolating surplus (published z = 2.35; human difference +1.02 coins per player-round). B1 (+0.16), B3 (+2.11) and B23 (+1.92) get the sign right; B2 (−0.95) and M0-sim (−0.25) do not.
- So E1 does not separate B1 from B3/B23 at all.
- Its 4 in-sample pairs are between mechanisms the models were fitted on.
- It is weak evidence that the controls behave as expected (B0: 0 of 8). It is not evidence about which model to trust for policy design.

## 9. Relation to the literature

**The central point of v1 is not new.**
- Model comparison by fit must be complemented by simulating the candidate models (Palminteri, Wyart & Koechlin 2017).
- In model-based RL, the likelihood of one-step predictions need not correlate with control performance (Lambert et al. 2020).
- In security games, adversary models trained for decision quality beat models trained for predictive accuracy (Perrault et al. 2020).
- Policies trained against one simulator exploit it and transfer poorly (Yu et al. 2026).
- Over-optimising a learned model of people follows Goodhart's law (Gao, Schulman & Hilton 2022).
- Predictively adequate learner models can disagree on the best policy (RankCert, Kadir 2026, preprint).
- The Lucas-critique framing for AI mechanism design is in Tacchetti et al. 2025.

**What v1 adds** is a pre-registered demonstration on real institutional data. Four citizen models, fitted on baseline mechanisms only, all pass the same held-out-mechanism one-step test. Each trains an RL planner, and the planners are cross-played under a frozen protocol. Under the only closed-loop-validated model, the planners trained on one-step-only models lose about a quarter of surplus to a simple published rule (descriptive; verdict `<<TBD: hypotheses-v1 H2>>`), and planners transfer only between sister models.

**Contrast with SUQR.** In Nguyen et al. 2013, a better-predicting adversary model (SUQR) gave a better defender policy, judged with 547 human subjects. There the decision is one-shot: the attacker best-responds once to a committed strategy, so one-step choice prediction *is* the whole closed loop. Our 40-round game feeds the policy back into behaviour, and that is where one-step fit and policy validity separate. v1 does not claim that predictive fit never helps.

**Partial anticipation.**
- RankCert reports the same disagreement among adequate models, for 8 fixed tutoring policies, with synthetic model families as oracles.
- Yu et al. show simulator exploitation and population co-training (our mixture planner is its analogue), with LLM user simulators that are not validated on held-out behavioural conditions.
- Neither uses human data from held-out institutions, and neither contrasts one-step with closed-loop validation.

**The design we lack.** Hocke et al. 2026 (abstract only) trained RL policies against four fish-behaviour models and deployed them on a robotic fish among live fish, scoring each model by the sim-to-real gap of closed-loop outcomes. That is the "train a policy per model, then judge it on the real system" design. v1 judges with a simulator (B1), because no new human experiments were run. Koster et al. 2022 and Nguyen et al. 2013 also judged designed policies with people.

## 10. Limitations and deviations

- **The judge is a simulator.** No planner was tested with people. v1 shows that one-step validation is insufficient *relative to a closed-loop-validated model*. It does not show that a policy preferred under B1 is better for real players.
- **B1 is a thin anchor.** It is validated in closed loop on one regime (R-I), with 3 of 4 metrics, one of which (Gini) is in range only through τ (§4). It fails the in-sample check under Equal (depletion round 23.5 against 13.3), and it has 2 of 4 metrics on R-M2.
- **G2 failed, and the scope was set post hoc.** The C4 question was reframed after the G2 result (D35), before any planner was trained. The B4 labels were read after §9.4 had fixed their mapping (D36).
- **The one-step bar is weak.** The persistence baseline P0 passes criterion (b) everywhere, with a higher Δℓ than B1 (0.116 / 0.223 / 0.231).
- **The M1 surrogate failed,** so R-M1 is one-step only and E1 cannot use M1 pairs.
- **B4:**
  - LLM sampling cannot be seeded; runs are archived but not reproducible draw for draw.
  - B4-calibrated's closed-loop pass rests on τ for two of its three in-range metrics (§4).
  - The C2 B4 labels were obtained on CLI 2.1.304, while all C4 B4 rows run on a newer CLI under the D37 deviation. D37 was approved by the owner on 2026-10-21, after the cross-play and distillation results were seen and before any C4 B4 result. So the C4 B4 rows and the C2 k = 22 reference rows differ in CLI version.
  - The contamination probe found no evidence that the model knows the 2025 study; it does know the 2022 study.
  - The pre-freeze probe and calibration runs (`citizens-c2-b4-probes-pilot-v1`, `citizens-c2-b4-calibration-v2`) were made from dirty trees (commit `03b8c32`, `git_dirty = true`).
- **Possible exploitation of B3/B23 is undetectable** with the frozen diagnostics (§6.6). Their planners' own-model gains (+803 and +717 coins), and their transfer to each other, may reflect shared model artefacts (both use EWA attractions over the same levels) rather than behaviour.
- **Retention and horizon.** Retention is off the training distribution of every model. The planner can learn the 40-round horizon, while the rules cannot. Both favour planners in every planner-versus-rule comparison.
- **Distillation is in sample,** and the one-parameter k family is misspecified for Planner[B1] and Planner[B23] (grid edge).
- **Cross-play output gap.** `protocol.md` §8 and §9.8 list the depletion round and players active for every cell. `crossplay.json` and `crossplay_series.json` record total surplus, Gini, decisions, kept share and the diagnostics, but not the depletion round or players active.
- **Commit at C4.** The C4 training, cross-play and distillation runs carry commit `4b1f1dd` (the D36 record commit), not the freeze commit `67f31a2` named in `protocol.md` §1 and §6.1. Between the two commits only `docs/` and `tests/` changed; `src/` is identical. The protocol SHA-256 (`319587d2…`) and both configuration hashes match.
- **C3 configuration hash.** `citizens-c3-baselines-v1`, the source of every k\*, records the C3 hash `4b1c3f7c…`, not `4a312211…` (which `protocol.md` §1 calls "unchanged since C3"). The difference is four keys added later for diagnostics and reporting (`diagnostic_baselines`, `frac_bins`, `reproduction_atol`, `sampled_eval_key_seed`). No key used by the baselines or the k\* tuning changed.
- **k\* tuning noise.** k\* was tuned on separate games (key 31001). On the C4 games, k\*[B3] beats k = 22 by only 4.6 coins on B3.
- **Population.** All human data are from UK/US Prolific participants.

## 11. What next: Model Citizens II

v1 suggests that the unit of validation for a simulator used in policy design is the policy ranking it induces, not its per-decision fit. Model Citizens II would test that directly:
- a population of simulators, each scored by whether it ranks a set of policies as a closed-loop-validated judge (ideally people) does;
- a planner trained against the population rather than against one member.

**Candidate test bed.** Alsobay, Rand, Watts & Almaatouq 2026 (*Science* 392) ran 360 public-goods conditions, with 147,618 decisions from 7,100 participants: an unseen test bed with many institutions. Its OSF project (osf.io/2d56w) has **no licence set**, and its README has no licence statement, so reuse needs the authors' permission first.

## 12. Reproducibility

**Runs.** All runs are under `runs/` (git-ignored); their manifests carry SHA-256 checksums of every artifact.

| Run | Status | Commit | Dirty | Config hash | Protocol SHA-256 | Manifest SHA-256 |
|---|---|---|---|---|---|---|
| `citizens-c1-data-audit-v1` | validated (G1a) | `ff36509` | false | — | — | `02826cf3` |
| `citizens-c1-g1b-v1` | validated | `d249ef0` | false | C2 `75ed1f96` | C2 `75aa5bd6` | `163e9aac` |
| `citizens-c2-fit-v1` | computed_unvalidated | `d249ef0` | false | C2 `75ed1f96` | C2 `75aa5bd6` | `1b630d05` |
| `citizens-c2-surrogates-v1` | validated | `d249ef0` | false | C2 `75ed1f96` | C2 `75aa5bd6` | `0e161016` |
| `citizens-c2-heldout-eval-v1` | validated | `d249ef0` | false | C2 `75ed1f96` | C2 `75aa5bd6` | `da9989cf` |
| `citizens-c2-b4-{plain,calibrated}-{interpolating,m1,m2}-v1` (6) | computed_unvalidated | `d249ef0` | false | C2 `75ed1f96` | C2 `75aa5bd6` | in `citizens-c2-b4-eval-v1` inputs |
| `citizens-c2-b4-eval-v1` | validated | `d249ef0` | false | C2 `75ed1f96` | C2 `75aa5bd6` | `1428cadb` |
| `citizens-c2-b4-calibration-v2` | computed_unvalidated | `03b8c32` | **true** | — | prompt `7f58e2e6` | `26243585` |
| `citizens-c2-b4-probes-pilot-v1` | computed_unvalidated | `03b8c32` | **true** | — | — | `51c458e8` |
| `citizens-c3-baselines-v1` | computed_unvalidated | `4cf989b` | false | C3 `4b1c3f7c` | — | `f71b80e2` |
| `citizens-c3-g3-v1` | computed_unvalidated | `203b35c` | false | C3 `4a312211` | — | `ef0bbd83` |
| `citizens-c4-train-*` (35) | computed_unvalidated | `4b1f1dd` | false | C4 `47ebefeb`, C3 `4a312211` | — | recorded in `citizens-c4-crossplay-v1` inputs |
| `citizens-c4-crossplay-v1` | computed_unvalidated | `4b1f1dd` | false | C4 `47ebefeb` | C4 `319587d2` | `87a1dc8e` |
| `citizens-c4-distill-v1` | computed_unvalidated | `4b1f1dd` | false | C4 `47ebefeb` | C4 `319587d2` | `9be7a5c6` |
| `citizens-e1-ranking-v1` (exploratory) | computed_unvalidated | `43439b7` | false | C2 `75ed1f96` | — | `b025e9be` |
| `citizens-c4-b4-*` (10) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| `citizens-c4-hypotheses-v1` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |

**Full hashes.**
- C2 config: `75ed1f96fc76ef76ee466c51c7fc553566fe8ed39164ac8465864ad539ac16ee`.
- `protocol-validation.md` v1: `75aa5bd6c228329c14d234326ea6026ec11ab2a464660db1a2b8cc04edcd6654`.
- C3 config: `4a312211d02c3b5a2f6c98f2cac4d2a5d7db44143f9341babc7c165f971a1dde`.
- C4 config: `47ebefeb6e2d69fc0672963c6b7dd29ac2aeb95b145234d7c9b8a451921ed4b8`.
- `protocol.md` v1: `319587d286b39d253d554a4c46f1bf69832751baae07967941478b0137cf11d0`.

**Seeds and keys.**
- C2: split 20261020, bootstrap 9001, B4 game order 7001.
- C3: evaluation 31000, k\* tuning 31001, monitoring 31002.
- C4: evaluation 41000, sampled noise 41001, distillation 41002, bootstrap 9101.
- Training seeds: 1001–1005 (main), 2001–2003 (no retention), 3001–3003 (mixture).

**Commands** (`justfile`). The held-out and C4 stages refuse to run unless their protocol is frozen and `--confirm-frozen` is passed.

```text
just citizens-data
just citizens-audit citizens-c1-data-audit-v1
just citizens-g1b --confirm-frozen
just citizens-c2-fit citizens-c2-fit-v1
just citizens-c2-surrogates citizens-c2-surrogates-v1 --reproduce runs/citizens-c2-surrogates-pre1
just citizens-c2-evaluate --confirm-frozen --fitted-run runs/citizens-c2-fit-v1 --surrogate-run runs/citizens-c2-surrogates-v1 --g1b-run runs/citizens-c1-g1b-v1
just citizens-b4-heldout --confirm-frozen --variant <plain|calibrated> --mechanism <Interpolating|M1|M2> ... --run-id citizens-c2-b4-<variant>-<mechanism>-v1
just citizens-b4-evaluate --confirm-frozen --evaluation-run runs/citizens-c2-heldout-eval-v1 --b4-run <the six runs>
just citizens-c3-baselines citizens-c3-baselines-v1
just citizens-c3-g3 citizens-c3-g3-v1 runs/citizens-c3-baselines-v1 runs/citizens-c3-train-b1-s1-v1 runs/citizens-c3-train-b1-s2-v1 runs/citizens-c3-train-b1-s3-v1
just citizens-c4-train citizens-c4-train-<model>[-noret]-s<seed>-v1 <model> <seed> 1000 --confirm-frozen [--no-retention]
just citizens-c4-crossplay citizens-c4-crossplay-v1 --confirm-frozen --train-run <the 35 runs>
just citizens-c4-distill citizens-c4-distill-v1 --confirm-frozen --train-run <the 35 runs>
just citizens-c4-b4 citizens-c4-b4-<variant>-<row>-v1 --confirm-frozen ... --cli-version-deviation <version>   (D37)
just citizens-c4-hypotheses citizens-c4-hypotheses-v1 --confirm-frozen --crossplay-run ... --distill-run ... --train-run ... --b4-run ... --b4-eval-run runs/citizens-c2-b4-eval-v1
just citizens-e1   (branch worktree-agent-a0b11da95451dae63; exploratory)
```

The tables of §6, §7 and Appendix A were extracted from the artifacts with read-only scripts. No number in this report was edited by hand.

## 13. Pending (to fill from `citizens-c4-hypotheses-v1` and the C4 B4 runs)

1. §1 and §6.3: the H1 verdict, with the confirmed H1-reversal and H1-transfer cells and raw and Holm p-values for both tests.
2. §1 and §6.3: the H2 verdict (H2a-loss, H2a-noninferior).
3. §6.4: the H2b-loss verdict.
4. §6.3: the own-model-gain and own-model-limited (optimiser-limited) verdicts.
5. §5 and Appendix A: the convergence labels per run, replacing the preview.
6. §6.6: the possible-exploitation labels and "on novel states" annotations.
7. §6.7: the sampled-mode families, and any disagreement with the deterministic verdicts.
8. §6.8: the S1 families and the "attributable to retention" verdict.
9. §6.9: the S2 families.
10. §7: the mechanism-agreement block (L1 matrix, within-model references, labels, κ L1, summary sentence).
11. §1 and §6.10: the H2-B4 verdicts for B4-calibrated; the descriptive B4-plain table; B4 call counts, discards, infrastructure errors, CLI versions and deviations.
12. §12: the manifests of the 10 C4 B4 runs and of `citizens-c4-hypotheses-v1`.

## Appendix A. C4 training runs

- **Eval:** own-model evaluation on the C3 evaluation games (key 31000, deterministic policy). The mixture rows show the mean over the four models.
- **Plateau iteration and tail slope** apply the §6.1 definition (`c4.plateau_iteration`: trailing mean of 5 points; final level = mean of the last 10; tolerance 0.02; tail = the last 20 % of the curve) to the recorded monitoring curve. **This is a preview**; the hypotheses-v1 labels are the record.
- All runs: 1,000 iterations, commit `4b1f1dd`, `git_dirty = false`, status `computed_unvalidated`.

| Run | Eval | Hours | Plateau it. | Tail slope / 100 it. | Final log σ | Not converged (preview) |
|---|---|---|---|---|---|---|
| `citizens-c4-train-b1-s1001-v1` | 1569 | 1.02 | 550 | −1.6 | −2.30 | no |
| `citizens-c4-train-b1-s1002-v1` | 1562 | 1.03 | 360 | +7.3 | −2.24 | no |
| `citizens-c4-train-b1-s1003-v1` | 1553 | 1.01 | 240 | −5.5 | −2.30 | no |
| `citizens-c4-train-b1-s1004-v1` | 1565 | 0.99 | 550 | +1.0 | −2.23 | no |
| `citizens-c4-train-b1-s1005-v1` | 1566 | 0.99 | 390 | +1.5 | −2.19 | no |
| `citizens-c4-train-b2-s1001-v1` | 1416 | 1.05 | 610 | +4.1 | −2.50 | no |
| `citizens-c4-train-b2-s1002-v1` | 1367 | 1.06 | 460 | +1.0 | −2.64 | no |
| `citizens-c4-train-b2-s1003-v1` | 1415 | 1.06 | 570 | +0.7 | −2.50 | no |
| `citizens-c4-train-b2-s1004-v1` | 1426 | 1.01 | 430 | +2.5 | −2.32 | no |
| `citizens-c4-train-b2-s1005-v1` | 1363 | 1.02 | 390 | −2.6 | −2.38 | no |
| `citizens-c4-train-b3-s1001-v1` | 1581 | 0.93 | 800 | −3.6 | −2.28 | no |
| `citizens-c4-train-b3-s1002-v1` | 1631 | 0.94 | 640 | +2.0 | −2.07 | no |
| `citizens-c4-train-b3-s1003-v1` | 1611 | 0.91 | 650 | −0.8 | −2.15 | no |
| `citizens-c4-train-b3-s1004-v1` | 1613 | 0.91 | 550 | +2.0 | −2.15 | no |
| `citizens-c4-train-b3-s1005-v1` | 1575 | 0.91 | 740 | +10.7 | −2.30 | no |
| `citizens-c4-train-b23-s1001-v1` | 1334 | 0.94 | 730 | −6.8 | −2.25 | no |
| `citizens-c4-train-b23-s1002-v1` | 1335 | 0.94 | 960 | +15.8 | −2.26 | **yes** |
| `citizens-c4-train-b23-s1003-v1` | 1300 | 0.91 | 820 | +8.5 | −2.25 | no |
| `citizens-c4-train-b23-s1004-v1` | 1300 | 0.91 | 600 | +3.1 | −2.19 | no |
| `citizens-c4-train-b23-s1005-v1` | 1334 | 0.91 | 830 | +9.5 | −2.36 | no |
| `citizens-c4-train-b1-noret-s2001-v1` | 1399 | 1.00 | 220 | −10.9 | −2.56 | no |
| `citizens-c4-train-b1-noret-s2002-v1` | 1459 | 0.94 | 790 | +6.7 | −2.52 | no |
| `citizens-c4-train-b1-noret-s2003-v1` | 1462 | 0.92 | 250 | −0.1 | −2.56 | no |
| `citizens-c4-train-b2-noret-s2001-v1` | 1372 | 1.06 | 930 | +14.6 | −2.64 | **yes** |
| `citizens-c4-train-b2-noret-s2002-v1` | 1389 | 0.98 | 740 | −0.1 | −2.62 | no |
| `citizens-c4-train-b2-noret-s2003-v1` | 1378 | 0.91 | 230 | +1.0 | −2.65 | no |
| `citizens-c4-train-b3-noret-s2001-v1` | 1527 | 0.87 | 340 | −7.2 | −2.31 | no |
| `citizens-c4-train-b3-noret-s2002-v1` | 1537 | 0.87 | 300 | −0.2 | −2.21 | no |
| `citizens-c4-train-b3-noret-s2003-v1` | 1526 | 0.84 | 210 | +4.5 | −2.34 | no |
| `citizens-c4-train-b23-noret-s2001-v1` | 1254 | 0.87 | 380 | −8.3 | −2.37 | no |
| `citizens-c4-train-b23-noret-s2002-v1` | 1266 | 0.87 | 800 | +7.1 | −2.42 | no |
| `citizens-c4-train-b23-noret-s2003-v1` | 1303 | 0.82 | 750 | +7.6 | −2.36 | no |
| `citizens-c4-train-mix-s3001-v1` | 1289 | 1.16 | 900 | +7.5 | −2.34 | no |
| `citizens-c4-train-mix-s3002-v1` | 1318 | 1.17 | 920 | +10.4 | −2.41 | **yes** |
| `citizens-c4-train-mix-s3003-v1` | 1256 | 1.17 | 240 | −4.3 | −2.48 | no |

## References

All references are marked verified in [`literature.md`](literature.md) (2026-10-21 update; at the time of drafting it lives on branch `worktree-agent-a7171c328dc4f8ab8`). Entries read from the abstract only are cited for nothing beyond their abstract.

- Alsobay, Rand, Watts & Almaatouq 2026. Integrative experiments identify how punishment affects welfare in public goods games. *Science* 392(6794). doi:10.1126/science.aeb5280.
- Camerer & Ho 1999. *Econometrica* 67(4):827–874. doi:10.1111/1468-0262.00054.
- Gao, Schulman & Hilton 2022. Scaling laws for reward model overoptimization. arXiv:2210.10760.
- Hocke, Gerken, Bierbach, Krause & Landgraf 2026. Robots that learn to evaluate models of collective behavior. arXiv:2604.07303 (abstract only).
- Kadir 2026. RankCert: When can simulated learners safely select an AI tutor? Robust decision certification under structural uncertainty. arXiv:2609.26069 (preprint; abstract only).
- Koster et al. 2022. Human-centred mechanism design with Democratic AI. *Nature Human Behaviour*. doi:10.1038/s41562-022-01383-x; arXiv:2201.11441.
- Koster et al. 2025. Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem. *Nature Communications* 16:2824. doi:10.1038/s41467-025-58043-7; arXiv:2404.15059.
- Lambert, Amos, Yadan & Calandra 2020. Objective mismatch in model-based reinforcement learning. *Proc. L4DC*, PMLR 120:761–770. arXiv:2002.04523.
- Lucas 1976. Econometric policy evaluation: A critique. *Carnegie-Rochester Conference Series on Public Policy* 1:19–46. doi:10.1016/S0167-2231(76)80003-6 (cited only as the origin of the critique).
- Nguyen, Yang, Azaria, Kraus & Tambe 2013. Analyzing the effectiveness of adversary modeling in security games. *Proc. AAAI* 27(1):718–724. doi:10.1609/aaai.v27i1.8599.
- Palminteri, Wyart & Koechlin 2017. The importance of falsification in computational cognitive modeling. *Trends in Cognitive Sciences* 21(6):425–433. doi:10.1016/j.tics.2017.03.011.
- Perrault, Wilder, Ewing, Mate, Dilkina & Tambe 2020. End-to-end game-focused learning of adversary behavior in security games. *Proc. AAAI* 34(02):1378–1386. doi:10.1609/aaai.v34i02.5494; arXiv:1903.00958.
- Tacchetti et al. 2025. Deep mechanism design: Learning social and economic policies for human benefit. *PNAS* 122. doi:10.1073/pnas.2319949121.
- Yu et al. 2026. One frozen simulator is not enough: Simulator collapse in multi-agent RL. arXiv:2608.12253 (abstract only).
