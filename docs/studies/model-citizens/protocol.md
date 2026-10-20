# Model Citizens C4: cross-model planner comparison protocol

Status: draft

Version v1, drafted 2026-10-20 for the study owner's review. Only the owner freezes it. The freeze changes the word `draft` above to `frozen`; that is the single status line that the C4 stages (`phasespace.citizens.rl.c4`: `crossplay`, `distill`, `b4`, `hypotheses`; and C4 training, `stage train --milestone c4`) read. The freeze also fills in §1. A frozen protocol is never edited in place: changes create v2, with a changelog entry and a reason, and every result states the version it used.

**This text is frozen only after B4's C2 held-out evaluation is complete.** B4's role in C4 (gating or descriptive) is fixed in §9.4 *before* those labels are known. The drafter did not read any B4 held-out outcome.

**Freeze checklist** (recorded in §1 and in a new D-entry, written only on the owner's instruction):
1. B4's C2 held-out runs and their `b4-evaluate` stage are complete. Each B4 variant's label per regime is recorded in the D-entry and mapped to its C4 role by §9.4, with no other judgement.
2. Every code prerequisite of §14 is done and `just verify` passes. (Done 2026-10-20, §14; re-run at the freeze commit.)
3. The constants of Appendix B are transcribed into `phasespace.citizens.rl.c4_config`. The appendix test (`tests/citizens/test_citizens_c4_protocol.py`) is extended to cover them, and `config_sha256` is printed into §1. (Transcription and test done, P8; the hash is printed at the freeze.)
4. The open decisions of §13 are answered and integrated in the sections they name. (Answered by the owner on 2026-10-20, §13; integrated in §2, §4, §6.5, §7, §9 and Appendix B.)
5. The freeze commit changes only the status word and §1, on a clean tree.

## 1. Header

| | |
|---|---|
| Protocol version | v1 |
| Frozen date | (at freeze) |
| Git commit at freeze | (at freeze) |
| C4 config hash at freeze | (at freeze; `c4_config.config_sha256()`) |
| C3 config hash | (at freeze; `rl.config.config_sha256()`, unchanged since C3) |
| Inputs | D31–D35 (`docs/decisions.md`); `README.md` (spec v0, amended by D35); `protocol-validation.md` v1 (frozen, D34); runs `citizens-c2-fit-v1`, `citizens-c2-heldout-eval-v1`, `citizens-c3-baselines-v1`, `citizens-c3-g3-v1` (with `citizens-c3-train-b1-s{1,2,3}-v1`), pilots `citizens-c4-pilot-b2-s901`, `citizens-c4-pilot-b3-s902`, `citizens-c4-pilot-b23-s903` |
| Scope | Milestone C4 and gate G4: cross-play (C4b) and distillation (C4c). The report and the public page (C5) are out of scope. |

**What was seen before this text was written (honest disclosure).** The C4 question was reframed (D35) after the G2 result, and this protocol is drafted after all of the following were seen:
- **All C2 non-LLM results** (`citizens-c2-heldout-eval-v1`): the labels of §3, Δℓ, metrics in range, the M1 surrogate failure and P0's Δℓ.
- **G3** (`citizens-c3-g3-v1`). Three B1 planners (seeds 1–3, 500 iterations) beat Interpolating k = 22 on B1 by 286 coins [258, 313] and k\*[B1] by 258 coins [230, 286] (deterministic policy, C3 evaluation games, key 31000). Per-seed means were 1558.6, 1540.2 and 1536.0 coins (SD 12.0; sampled policy SD 15.5). On those games the B1 ensemble's mean member TV was 0.194–0.199 for the planner and 0.170 for k = 22.
- **The deployed rules on all four models** (`citizens-c3-baselines-v1`, 1,000 games each, key 31000). Rule rankings already depend on the model. Under B1, Proportional (1124) beats Mixed (881); under B2, Mixed (1084) beats Proportional (918). k\* is 9.03 for B1 and B2, 27.1 for B3 and 99.5 for B23.
- **Own-model pilots** on seeds 901–903, 1,000 iterations each, scored only on their own model's C3 monitoring games (512 games, key 31002):

  | Model | Planner (deterministic) | k = 22 | k\* | Plateau iteration | Tail slope (coins / 100 it.) | Hours |
  |---|---|---|---|---|---|---|
  | B2 | 1372.9 | 1348.9 | 1360.8 | 690 | 5.7 | 0.87 |
  | B3 | 1582.6 | 791.8 | 796.8 | 480 | −2.3 | 0.81 |
  | B23 | 1266.7 | 587.3 | 601.2 | 530 | 6.2 | 0.81 |

  B2's pilot exceeds its own k\* by 12 coins, which is below the minimum effect δ = 16 chosen in §9.1. δ is anchored on C2's surplus tolerance and on the B1 seed SD, not on this pilot (§9.1).

- **After drafting, while implementing §14 (2026-10-20):** a 20-iteration timing of PPO on B1 and on the mixture citizen (test seed 5, not a C4 seed; wall time only, no score read or printed), and the C3 golden test (B1, seed 5, 4 iterations of 64 games; parameter hash only).

**Not seen:**
- any planner evaluated on a model other than its training model;
- any C4 evaluation game (key 41000) or distillation game (key 41002);
- any planner or k\* game with B4;
- B4's C2 held-out outcomes (still running at drafting).

## 2. Question and hypotheses

**Question (D35).** Do citizen models that each pass a different held-out test lead PPO planners to the same mechanism, and do the planners transfer between models? The two tests are next-choice prediction and whole-game outcomes. The question is an empirical test of whether one-step behavioural validation is enough for policy design (the Lucas critique).

All hypotheses compare total surplus per game (coins; §8) under the deterministic planner policy. δ = 16 coins is the minimum effect (§9.1). Exact tests are in §9.

- **H1 (transfer).** For every ordered pair X ≠ Y of the training models {B1, B2, B3, B23} (12 cells), Planner[X] evaluated on Y beats Y's own re-tuned rule k\*[Y].
  - *Refuted* if any cell shows a confirmed reversal larger than δ.
  - *Supported* if all 12 cells show a confirmed advantage larger than δ.
  - Otherwise: "not refuted; transfer not shown in cells …".
  - The regret Planner[Y]@Y − Planner[X]@Y is reported with every cell.
- **H2 (does one-step validation suffice?).** B1 is the only model validated in closed loop. On B1, the planners trained on the closed-loop-rejected models (B2, B3, B23) lose to B1's k\* by more than δ (H2a, which decides H2). They also lose to Planner[B1] by more than δ (H2b, secondary; it includes the home advantage every planner has on its own model).
  - *Supported* (one-step validation was not enough) if any of the three planners shows a confirmed H2a loss.
  - *Refuted* (one-step validation was enough here) if all three are confirmed non-inferior to k\*[B1] within δ.
  - Otherwise: inconclusive.
  - H2a uses the same three cells as H1's column B1, tested as their own family of three.
- **H2-B4 (conditional).** The same as H2a, on a B4 variant. It is confirmatory only if that variant is closed-loop validated in C2; otherwise it is descriptive (§9.4).
- **Mechanism agreement (descriptive, no test; §9.6).** Whether the distilled rules of the four planners agree. The rules are:
  - w(R) curves;
  - kept share by pool level;
  - the fitted Interpolating k;
  - trees.

  The fit R² and the surplus of each distilled rule are reported with them.
- **Secondary analyses (§9.7):**
  - S1, the no-retention ablation: is disagreement driven by retention?
  - S2, the robust mixture planner (owner decision O1 = yes, 2026-10-20).

**Status of the one-step label.** B2, B3 and B23 pass the one-step criterion (b) on every held-out regime. However, the diagnostic persistence baseline P0 passes it too, with a higher Δℓ than B1 (0.116 / 0.223 / 0.231). So "one-step validated" is a weak bar. No C4 statement may lean on it as strong evidence that a model is realistic. H2 asks whether that weak bar is enough for policy design; it does not assume that it is.

## 3. Model: game and citizen populations

- **Game.** As in `protocol-validation.md` §3 and D33 (4 players, R₀ = 200 = cap, R′ = max(0.01, min(200, R − Σe + 1.4Σc)), exclusion below 1 coin, 40 rounds), simulated by the C3 JAX twin (`phasespace.citizens.rl.twin`, `rl.citizens`). The twin matches the NumPy reference (C3 port checks recorded in `citizens-c3-baselines-v1`).
- **Citizen models (columns).** These are the C2 fitted models of `citizens-c2-fit-v1`, unchanged, with their C2 labels (D35). Metrics in range are out of 4. Δℓ is in nat with its 95 % CI lower bound.

  | Model | Label | R-I in range, Δℓ | R-M1 in range, Δℓ | R-M2 in range, Δℓ |
  |---|---|---|---|---|
  | B1 (GRU clone) | validated (closed-loop) on R-I; one-step on all three | 3, 0.104 [0.075] | 1, 0.081 [0.041] | 2, 0.089 [0.037] |
  | B2 (conditional cooperation) | one-step only; rejected in closed loop | 1, 0.103 [0.058] | 2, 0.212 [0.154] | 2, 0.175 [0.113] |
  | B3 (EWA learning) | one-step only; rejected in closed loop | 0, 0.180 [0.143] | 0, 0.301 [0.243] | 1, 0.302 [0.250] |
  | B23 (hybrid) | one-step only; rejected in closed loop | 0, 0.174 [0.146] | 0, 0.256 [0.207] | 1, 0.235 [0.179] |

  "Validated" is never written for B2, B3 or B23 without "one-step".
  - B2 errs one way: on R-I its pool lasts too long (mean depletion round 38.1), and its Gini is too low on every regime.
  - B3 and B23 err the other way: on R-I they deplete early (23.4 and 18.5, against the human 35.2).
- **B4 (LLM citizens, evaluation only).** This is the frozen C2 set-up: `claude-haiku-5-5` through Claude Code CLI 2.1.304, `b4-prompt-v2`, the exact CLI flags, retries and archive of `protocol-validation.md` §6.3. There are two variants:
  - B4-plain;
  - B4-calibrated, with type weights w = (0.7, 0.0, 0.3) from `citizens-c2-b4-calibration-v2`.

  B4 is never a training environment. The probe label is "no contamination evidence for the 2025 study". Sampling cannot be seeded. The game order uses seed 7001.

## 4. Policies (rows)

**Planners.** Every planner row is PPO with the frozen C3 hyperparameters (Appendix A), trained against one citizen model with 4 citizens per game drawn from that model's ensemble, for 1,000 iterations (1,024 games × 40 rounds per iteration; 40.96 M planner decisions). Every planner row shares **one training signature**: the iterations, PPO configuration, overrides, code commit and fit run are identical. Rows differ only in training model, seed and action head (retention).

| Row | Training model | Head | Seeds | Role |
|---|---|---|---|---|
| Planner[B1], Planner[B2], Planner[B3], Planner[B23] | each model | 5-way (4 players + pool retention) | 1001–1005 | main (H1, H2) |
| Planner-noret[X], X ∈ {B1, B2, B3, B23} | each model | 4-way softmax, Σe = R − 0.001 | 2001–2003 | secondary S1 (ablation) |
| Planner[Mix] (O1 = yes) | uniform mixture per game | 5-way | 3001–3003 | secondary S2 |

- **Excluded seeds and runs.** The C3 B1 planners (`citizens-c3-train-b1-s{1,2,3}-v1`; 500 iterations; already inspected in G3) and the pilots (seeds 901–903) are never C4 inputs. The C4 B1 row is trained fresh, with seeds 1001–1005. The C4 stages refuse C3 runs, pilot runs and reserved seeds (`c4.reserved_seeds`, which includes the C3 seeds 1–3). Training under `--milestone c4` refuses any seed not declared for its row, any iteration count other than 1,000, any `--set` override and an unfrozen protocol (P1).
- **Robust mixture planner** (O1 = yes). Each of the 1,024 training games of an iteration draws its citizen model uniformly and independently from {B1, B2, B3, B23} (`citizens.MixtureCitizen`). The draw uses the game's citizen key, which is split from the iteration key of the training seed, so it is fixed by the seed. All four citizens of a game come from that model. Everything else equals the main rows. Monitoring uses the C3 monitoring games, the same 512 games under each model, with the 4-model mean as the learning curve; the final score on the C3 evaluation games is likewise the 4-model mean, with each model's score recorded. Planner[Mix] has no single training model, so it is not distilled (§6.4) and has no own-model cell (§9.3). *Cost:* about 1.0 h per run (§7), so 3.0 h for 3 seeds.
- **Evaluation policy.** The deterministic policy (shares = softmax(μ)) is confirmatory. The sampled policy (one noise key, 41001) is reported for every cell, as is the final log σ of each run.

**Baselines (deployed offer rule, D33):**
- Equal (w = 1), Mixed (w = 0.5), Proportional (w = 0);
- Interpolating k = 22;
- one re-tuned Interpolating k\*[X] per training model, from `citizens-c3-baselines-v1`. k\* was tuned on separate games (key 31001), on the paper's grid of ln k ∈ [−5, 5] in steps of 0.1:
  - B1: 9.03 (ln k 2.2);
  - B2: 9.03 (2.2);
  - B3: 27.1 (3.3);
  - B23: 99.5 (4.6).

  k\*[B1] and k\*[B2] are the same rule.

  "Y's own k\*" is k\*[Y] evaluated in column Y. Every k\* row is evaluated in every column.

## 5. Feasibility

- **Valid games.** A simulated game is valid if every round satisfies:
  - e ≥ 0 and Σe ≤ R (+1e-9);
  - integer 0 ≤ c ≤ ⌊e⌋, and c = 0 when e < 1;
  - the pool update of §3;
  - finite values throughout.

  An invalid game is a code bug: stop (K2), never drop it silently.
- **Planner offers.** A planner's offers must satisfy Σe ≤ R. The no-retention head must satisfy Σe = R − 0.001, within 1e-9.
- **B4 games.** These follow the C2 rules: invalid answers are retried, and unrecoverable games are discarded and replaced. Infrastructure errors are retried and never discarded.

## 6. Methods

### 6.1 Training
- `just citizens-c4-train <run_id> <model> <seed> 1000 --confirm-frozen [--no-retention]` (stage `train`, milestone `c4`; model `Mix` for the mixture row), from a clean tree at the freeze commit, after the freeze. The command refuses before the freeze, and refuses undeclared seeds, other iteration counts, overrides and other run ids.
- Run ids:
  - `citizens-c4-train-<model>-s<seed>-v1`;
  - `citizens-c4-train-<model>-noret-s<seed>-v1`;
  - `citizens-c4-train-mix-s<seed>-v1`.
- No `--set` overrides.
- Each run records:
  - its learning curve on its own model's monitoring games (key 31002, 512 games);
  - its final own-model score on the C3 evaluation games (key 31000);
  - its wall time.

  These own-model numbers are seen during training. They feed only the K4 and K5 labels (§12).
- **Optimisation adequacy (fixed in advance, applied to every model).** The C3 hyperparameters were chosen on B1 only. To keep an optimiser failure from being read as a model difference, each row's learning curves are checked with the plateau definition of `c4_config` (smooth 5 points, final 10 points, tolerance 0.02):
  - a run is "not converged" if its plateau iteration is above 900 or its tail slope is above δ per 100 iterations (16 coins);
  - a row is "optimiser-limited" if its own-model advantage over k\* is a confirmed loss larger than δ (§9.5).

  Both labels are reported next to every number of that row. Neither permits re-tuning (§11).

### 6.2 Cross-play (C4b)
- `just citizens-c4-crossplay citizens-c4-crossplay-v1 --confirm-frozen --train-run <all planner runs>`.
- Every policy row is evaluated against every trainable column on the same **1,000 C4 evaluation games** (citizen key 41000; common random numbers within a column). Planners are evaluated per seed, in both modes.
- The run records:
  - per-game totals and Gini;
  - per-seed means;
  - paired differences;
  - the diagnostics of §6.3.

### 6.3 Extrapolation diagnostics (per cell, per seed; reported, not gating except through the label of §9.5)
- **Kept share, coin-weighted.** Per round t: Σ_games (R_t − Σe_t) / Σ_games R_t over games with R_t ≥ 1, plus the same pooled over all rounds. A mean of per-game ratios is not used: at a depleted pool, R = 0.01 and the 0.001 withholding would read as a 10 % kept share.
  - Reported per round, and for the endgame (rounds 39–40).
  - The training mechanisms (Equal, Mixed, Proportional) never retain coins (Σe ≈ R). Retention is therefore off the training distribution for every model; partial evidence exists only from the held-out M1/M2 human games.
  - The planner can learn the 40-round horizon; the rules cannot. This is a caveat on every planner-versus-rule comparison.
- **B1 ensemble disagreement.** This is the mean and p90 over non-forced decisions of the B1 members' mean pairwise total variation, teacher-forced on the cell's trajectories (`citizens.b1_member_disagreement`), in every column. It measures how far a trajectory's states are from where B1's members agree. For columns other than B1, it is a state-novelty index.
- **Input support.** These are the fractions of non-forced decisions outside the fit-split training support:
  - *range*: outside the marginal [min, max] of any input the model reads;
  - *grid*: off the occupied cells of a joint (x1/4, x4) grid with 20 × 20 bins.

  The support comes from the scored fit-split decisions of the C2 fit; no held-out row is read.

  *Weakness:*
  - The range diagnostic is marginal: a decision can be jointly novel and still inside every range.
  - The grid covers two inputs only.
  - For **B2**, both diagnostics concern inputs B2 reads, so they measure model extrapolation.
  - For **B3 and B23**, the grid's inputs (offer share and pool) are not read by the models. Their range diagnostic covers own offers (and x3 for B23). These numbers are labelled **"state novelty"**, not model extrapolation.

### 6.4 Distillation (C4c)
- `just citizens-c4-distill citizens-c4-distill-v1 --confirm-frozen --train-run <all planner runs>`.
- Each planner seed (main and no-retention rows; not Planner[Mix], which has no single training model) is played on its own training model on the **1,000 distillation games** (key 41002). The fits are:
  - the binned w(R) family, with 10 bins of R/200, a weight w and a kept share κ per bin;
  - the one-parameter fit w = (R/200)^k on the paper's ln k grid;
  - a CART tree of depth 3 with at least 200 rows per leaf, on (R/200, own previous fraction, others' previous fraction, own previous share).

  Each fit's R² is reported. κ per bin is coin-weighted, as in §6.3.
- **Deployed comparison.** These rules are played on the *same* distillation games:
  - the distilled w-rule, the distilled k and the tree;
  - the deployed k\*[X] and k = 22.

  The distilled k and the deployed rules go through the deployed rounding path: w = round((R/200)^k, 1), float32 arithmetic and Σe = R − 0.001. The distilled rules are scored on the games they were fitted on (in-sample); this is stated with every distilled-rule number.
- **Recovery test (method bias).** The distiller is also run on trajectories of the deployed, quantised k = 22 rule and of each k\*[X], on each model's distillation games. The report gives the recovered k and the w(R) error against the true quantised w per bin. This is the bias of the method, reported next to every planner's distilled k.
- **Distilled-rule cross-play.** Each main planner seed's distilled w-rule and distilled k are played on every trainable column on that column's distillation games, next to k\*[Y] and k = 22 on the same games. This shows whether the *mechanism* transfers, not just the network.

### 6.5 Planner versus B4
- `just citizens-c4-b4 citizens-c4-b4-<variant>-<row>-v1 --confirm-frozen --train-run <run> --crossplay-run runs/citizens-c4-crossplay-v1 --variant <variant> [--calibration-run runs/citizens-c2-b4-calibration-v2]` for a planner row (`<row>` = `b1`, `b2`, `b3`, `b23`), and `--rule "Interpolating k*[B1]"` instead of `--train-run` for the rule row (`<row>` = `kstar-b1`), with 40 games per row.
- Rows on B4 (O2 = (ii), O3: k\*[B1] is run on B4 for each variant, not reused from C2): Planner[B1], Planner[B2], Planner[B3] and Planner[B23], seed 1001 each, plus k\*[B1] (k = 9.03), each on **both** variants, B4-plain and B4-calibrated: 10 B4 runs (§14, P6).
- **Guards (P6).** A planner row must be a main row with seed 1001, from a clean tree, not reserved, an input of the cross-play run (made under this protocol, from a clean tree) and identical to its shared training signature, with the baselines run's fit run. The rule row's k must equal the protocol's k\*[B1]. Only the declared variants, game counts (40, or 30 under the reduction) and run ids are accepted.
- **Seed choice.** The seed is the first main seed, 1001, fixed now. There is no selection on any result.
- The C2 B4 held-out Interpolating k = 22 run of the same variant (for example `citizens-c2-b4-plain-interpolating-v1`) is shown as a reference row. It is not part of a test.
- **Statistics.** B4 games cannot be paired across rows (unseeded sampling), so comparisons are unpaired over games (§9.4).

## 7. Budget, seeds and compute

- **Seeds** (Appendix B):
  - main rows: 1001–1005;
  - ablation: 2001–2003;
  - mixture: 3001–3003.

  They are distinct from every reserved key (C3 31000–31003 and 31100+; C4 41000–41002 and 41010), from the C3 seeds 1–3 and from the pilots 901–903.
- **Why 5 and 3 seeds.**
  - The B1 planners' per-seed SD is 12.0 coins (deterministic) and 15.5 (sampled) (`citizens-c3-g3-v1`), so with 5 seeds the seed SE is about 5–7 coins, below δ. The non-B1 models have one pilot seed each, so their seed SD is unmeasured. That is a risk, controlled by the seed-level test of §9.2. With 5 seeds and Holm over 12, that test needs about 2.2 seed SDs beyond δ (t₄ critical value 4.85 at α = 0.05/12).
  - The secondary rows have 3 seeds, because they are not confirmatory. With df = 2 the seed-level test is weak (t₂ critical value 10.9 at 0.05/12), and the secondary verdicts are labelled accordingly.
  - No seeds are ever added after the freeze.
- **Simulation sizes.**
  - 1,000 evaluation games per (row, column, mode);
  - 1,000 distillation games per planner seed and per column for distilled-rule cross-play;
  - bootstrap B = 10,000 with seed 9101;
  - 40 B4 games per (row, variant).
- **Non-LLM compute.** Measured training costs:
  - 0.806–0.867 h per 1,000-iteration run (the three pilots);
  - 0.455–0.457 h per 500-iteration B1 run (C3), about 0.91 h per 1,000;
  - the mixture citizen runs all four models on every stream; a 20-iteration timing with a test seed (5; no evaluation) took 1.09 times as long per iteration as B1, so about 1.0 h per mixture run.

  Planned training, run serially:

  | Training rows | Runs | Hours |
  |---|---|---|
  | Main | 20 | 18.2 |
  | Ablation | 12 | 10.9 |
  | Mixture (O1 = yes) | 3 | 3.0 |
  | Total | 35 | about 32 (31.8 at the B1 rate) |

  - **Parallel runs.** The machine has 8 performance cores (12 cores in all; `sysctl -n hw.perflevel0.physicalcpu hw.ncpu`). One training process uses about 1.4 cores, so **5 training processes run in parallel** (about 7 performance cores), leaving a core for the B4 runs and the system. That is 7 waves of 5 runs, about 6.5 h of wall time if parallel runs do not slow each other; with contention the plan allows up to about 10 h. Each run's own wall time is recorded and checked against the 24 h limit (K1).
  - Cross-play is estimated at about 0.5 h by the C4 prep. Distillation (with the recovery test and the distilled-rule cross-play) and the hypothesis stage are smaller.
  - **Ceiling: 40 h wall-clock** for all C4 non-LLM work.
- **LLM calls (separate ceiling; counted apart from C2's 55,000).** One call is made per player-round with an offer of at least 1 coin. The estimate uses each policy's own-model decisions per game (G3 and pilot runs), × 40 games × (1 + 0.020 retry fraction from `citizens-c2-b4-calibration-v2`):

  | Policy on B4 | Calls |
  |---|---|
  | Planner[B1] | 5,774 |
  | Planner[B2] | 6,243 |
  | Planner[B3] | 5,541 |
  | Planner[B23] | 5,145 |
  | k\*[B1] (123.2 decisions per game on B1) | 5,026 |

  The worst case is 160 decisions per game, or 6,528 calls per row.

  | O2 option | Estimate | Worst case | **Ceiling** |
  |---|---|---|---|
  | (i) 4 planners + k\*[B1] on one variant | 27,700 | 32,640 | 33,000 |
  | **(ii) the same on both variants (chosen, O2)** | ≤ 55,500 | 65,280 | **66,000** |
  | (iii) none | 0 | 0 | 0 |

  - **Chosen: option (ii)** (owner, 2026-10-20). The estimate is 2 × 27,700 ≈ 55,500 calls for the 10 B4 runs. The **C4 LLM call ceiling is 66,000 calls** (`c4_config.CEILING_LLM_CALLS`): about 19 % above the estimate and above the worst case of 65,280, so only retries beyond the calibration's rate can reach it.
  - The C4 prep estimated about 26k for (i) and 46k for (ii); the table above uses the rates measured in the runs.
  - Throughput in the calibration was about 6,600 calls/h, so option (ii) takes about 8–9 h.
  - C4 B4 runs start only after the C2 B4 runs have finished, and planner rows only after the cross-play run (their guard reads it, §6.5).
- **Pre-declared reductions.** These are applied in order, and only if the measured cost projects past a ceiling:
  - *Non-LLM:*
    1. mixture seeds 3 → 2;
    2. ablation seeds 3 → 2.

    Main rows are never reduced. If the ceiling is still exceeded, K3 applies.
  - *LLM:*
    1. B4 games per row 40 → 30;
    2. drop the variant that is not closed-loop validated, or B4-calibrated if both have the same label (option (ii) is chosen, so this step applies);
    3. drop the k\*[B1] B4 row, and H2-B4 becomes descriptive against the C2 k = 22 reference.

    Below 20 complete games, a row is exploratory.

## 8. Metrics

Per game, from `phasespace.citizens.metrics` (notebook definitions):
- **total surplus**, in coins summed over 4 players × 40 rounds (= 160 × surplus per player-round). This is the confirmatory metric.
- **Gini** of the players' mean rewards; reported, not optimised.
- **depletion round**;
- **players active**;
- the diagnostics of §6.3;
- decisions per game, which is the B4 call driver.

## 9. Statistical analysis and decision rules

### 9.1 Minimum effect
δ = **16 coins of total surplus per game**, or 0.1 coin per player-round. The anchors are:
- it equals C2's surplus resolution tolerance τ (`protocol-validation.md` §8);
- it is about one B1 seed SD (12–16 coins).

An effect smaller than this is not distinguishable from training-seed variation. For reference, it is 1.2 % of k\*[B1] on B1 (1287 coins) and 2.6 % of k\*[B23] on B23 (613 coins) (`citizens-c3-baselines-v1`).

### 9.2 Tests (every confirmatory cell)
For a planner row P (S seeds) and a comparator Q in column Y, on the common games j = 1…J (J = 1,000):
- D = mean_{s,j} P_{s,j} − mean_{s',j} Q_{s',j};
- per-seed differences d_s = mean_j (P_{s,j} − Q̄_j), where Q̄ is the mean over Q's seeds, or Q itself for a rule.

1. **Crossed bootstrap** (`crossplay.group_bootstrap`). Each of B = 10,000 resamples (seed 9101) draws, with replacement:
   - S seeds of P;
   - S' seeds of Q (S' = 1 for a rule);
   - J games, the same for both rows.

   One-sided p-values against the threshold:
   - p⁻(θ) = (1 + #{D\* ≥ θ}) / (B + 1), for "D < θ";
   - p⁺(θ) = (1 + #{D\* ≤ θ}) / (B + 1), for "D > θ".
2. **Seed-level t-test.** For a rule comparator: a one-sample t-test on the d_s, with df = S − 1. For two planner rows: Welch's t on the per-seed means of each row on the same games. Same thresholds θ, one-sided.
3. **Per-seed signs.** The number of seeds with d_s < θ (and > θ) is reported for every cell.

**Holm correction.** Holm is applied separately to the bootstrap p-values and to the t p-values within each family, at family-wise α = 0.05: sort p ascending and reject the i-th while p₍ᵢ₎ ≤ α / (m − i + 1).

**Confirmed only if both tests agree.** A cell's directional claim is **confirmed** only if it is rejected under Holm by *both* the crossed bootstrap and the seed-level t-test.

*Why both.*
- The crossed bootstrap captures game sampling, but with 5 seeds its seed-level percentile intervals under-cover.
- The t-test treats the seeds correctly, but it ignores game sampling, which the CRN design makes common to all seeds.

Requiring both is conservative in both directions. Per-seed signs are shown so that a reader can see a split row.

### 9.3 H1 and H2 (main rows, deterministic policy)

| Family | Cells (m) | Comparator | Claim tested | θ |
|---|---|---|---|---|
| H1-reversal | 12 ordered pairs X ≠ Y | k\*[Y] in column Y | D < −δ | −δ |
| H1-transfer | the same 12 | k\*[Y] | D > +δ | +δ |
| H2a-loss | X ∈ {B2, B3, B23}, column B1 (3) | k\*[B1] | D < −δ | −δ |
| H2a-noninferior | the same 3 | k\*[B1] | D > −δ | −δ |
| H2b-loss | the same 3 | Planner[B1] (5 seeds) | D < −δ | −δ |
| Own-model adequacy | X = Y, 4 cells | k\*[X] | D < −δ (optimiser-limited); D > +δ (own-model gain) | ∓δ |

- **H1 verdict.** Refuted if any H1-reversal cell is confirmed. Supported if all 12 H1-transfer cells are confirmed. Otherwise not refuted, and the confirmed and unconfirmed cells are listed.
- **H2 verdict.** Supported if any H2a-loss cell is confirmed. Refuted if all three H2a-noninferior cells are confirmed. Otherwise inconclusive.
- **H2b** is reported with its own verdicts, as the regret of designing on a one-step-only model.
- **Regret.** Planner[Y]@Y − Planner[X]@Y (12 cells) is reported with crossed-bootstrap 95 % CIs. It is descriptive, with no multiplicity correction.
- **Sampled policy.** The same families are computed for the sampled policy and labelled secondary. A disagreement between the deterministic and sampled verdicts is reported.
- **Annotation.** A confirmed H1 cell is annotated "on novel states" if Planner[X]'s diagnostics in column Y exceed the thresholds of §9.5.

### 9.4 B4 (role fixed before B4's C2 labels are known)
- **When a B4 variant is gating.** B4-v is a **validated evaluation population** if, in C2's `b4-evaluate` for that variant, it is "validated (closed-loop)" on at least one closed-loop regime, R-I or R-M2: `b4_gating` is true and criterion (a), plus (c) where it applies, is met. R-M1 cannot validate B4, because the M1 surrogate failed and B4 has no likelihood. This is the same standard as B1's label.
- **If B4-v is a validated evaluation population.** **H2-B4** for v is confirmatory:
  - Families:
    - B4-loss: Planner[X] versus k\*[B1] on B4-v, X ∈ {B2, B3, B23}, claim D < −δ (3 cells);
    - B4-noninferior: the same cells, claim D > −δ;
    - B4-b: Planner[X] versus Planner[B1], claim D < −δ.
  - Each family is Holm-corrected over 3.
  - Test: an unpaired game bootstrap (B = 10,000, seed 9101; games resampled within each row), with p-values as in §9.2. There is no seed-level test, because each row has one seed; verdicts are stated as conditional on seed 1001.
  - The verdict is defined as for H2.
  - *Power:* with game SDs of 300–550 coins (`citizens-c3-baselines-v1`) and 40 games per row, the SE of a difference is about 70–120 coins. Only large effects can be confirmed, and an unconfirmed cell is not evidence of no effect.
- **Otherwise.** All B4 numbers for v are **descriptive**, labelled "B4-v not validated in closed loop (C2)". These are the means and CIs of total surplus, Gini, depletion and players active per row, and the differences.
- Under O2 = (ii) both variants are run; each is mapped to its role separately, so H2-B4 can be confirmatory for one variant and descriptive for the other. The role is read from C2's `b4-evaluate` run (`counts_as_validated` on R-I or R-M2) by the hypothesis stage.
- A row with fewer than 20 complete games is exploratory: its cells count in their Holm family but are never confirmed.
- B4 never enters H1 (it is not a training model) and never changes H1 or H2.

### 9.5 Possible exploitation (label, own-model gains)
If Planner[X] has a confirmed own-model gain (D > +δ over k\*[X] in column X), the gain is labelled **"possible exploitation"** when, in column X, either of the following holds:
- its mean B1 member TV exceeds **1.5 ×** that of Interpolating k = 22 in the same column (for reference, the B1 planner on B1 had a ratio of about 1.15 in G3);
- (X = B2 only) its range or grid fraction exceeds k = 22's in the same column by more than **0.05** (absolute).

For B3 and B23 the input-support numbers are state novelty (§6.3) and are reported without entering the label.

The same flags are reported for every cell. A transfer win in a flagged cell carries the annotation of §9.3.

### 9.6 Mechanism agreement (descriptive)
For the main planners, from the distillation run:
1. **w(R) distance.** L1(X, Y) is the mean over bins b ∈ 𝔅 of |w̄_X(b) − w̄_Y(b)|, where w̄ is the seed mean and 𝔅 is the set of bins with at least 200 informative rows for every seed of both models. The within-model reference is the mean pairwise L1 among X's seeds. The pair is labelled "same w(R)" if L1(X, Y) ≤ max(0.10, 2 × max(within_X, within_Y)), and "different" otherwise.
2. **κ(R) and kept share.** The L1 between the coin-weighted κ curves (same bins), and the endgame kept share. No label.
3. **Fidelity.** The R² of the w-rule, the distilled-k fit and the tree. A planner whose w-rule R² is below 0.5 is "low fidelity", and every w(R) statement about it carries that label.
4. **Distilled k against the method bias.** The distilled k for each seed, next to k\*[X] and the recovery test's bias.
5. **Distilled-rule surplus.** Recovery = (S_w-rule − S_k\*[X]) / (S_planner − S_k\*[X]) on the distillation games (in-sample), and the distilled-rule cross-play matrix (§6.4) relative to k\*[Y].

**Summary sentence.** "Same mechanism" requires all 6 pairs to be "same w(R)" and every planner to be of adequate fidelity. "Different mechanisms" requires at least one pair to be "different" between planners of adequate fidelity. Otherwise the result is "mixed".

### 9.7 Secondary analyses
- **S1, no-retention ablation.**
  - The H1 families are recomputed for the Planner-noret rows (each with its own Holm family of 12; 3 seeds; labelled secondary), together with §9.6 among the no-retention planners.
  - Disagreement is called **"attributable to retention"** if the main rows have at least one confirmed H1 reversal and the no-retention rows have none, or if the mean between-model w(R) L1 among the no-retention planners is less than half that among the main planners.
  - *Confound:* the frozen PPO configuration has entropy coefficient 0, so no entropy bonus enters the loss. The 4-way head, however, explores in 4 dimensions instead of 5, and at equal σ its Gaussian entropy is about 20 % smaller. This is stated wherever S1 is reported.
- **S2, robust mixture planner** (O1 = yes). Planner[Mix] versus k\*[Y] in every column, with reversal and transfer families of 4, Holm, both tests, labelled secondary. Its regret against Planner[Y] in each column is also reported. This is the v0 H2 question, now descriptive.

### 9.8 Reported regardless of outcome
The following are always reported:
- the full matrix: every row × column × mode, with mean, per-seed means, Gini, depletion, players active, decisions per game and every diagnostic;
- every family's p-values (raw and Holm, both tests) and per-seed signs;
- the learning curves, plateau labels and final log σ;
- the distillation outputs and the recovery test;
- the B4 call counts, discards and infrastructure errors.

The run status follows `docs/data-contract.md`.

## 10. Procedure

### 10.1 Before the freeze
1. The code prerequisites of §14 are implemented and tested; `just verify` passes.
2. The C2 B4 held-out runs and `b4-evaluate` are finished; the labels are recorded and mapped by §9.4.
3. The owner answers §13; the answers are integrated.
4. The Appendix B constants go into `c4_config` and the appendix test is extended.
5. Freeze commit.

Allowed before the freeze: own-model pilots on reserved seeds only, code tests on toy citizens, and timing or golden-hash runs of a few iterations on test seeds that read no score. **Not allowed:** any training run with a declared C4 seed, any evaluation on keys 41000 or 41002, and any planner game with B4.

### 10.2 After the freeze
6. From a clean tree at the freeze commit: all training runs of §4 (`just citizens-c4-train …`; any order, concurrent runs allowed). Each run is checked against the shared training signature.
7. `citizens-c4-crossplay-v1` (all planner runs), then `citizens-c4-distill-v1` (all planner runs, including the recovery test and the distilled-rule cross-play).
8. The B4 runs of the chosen O2 option, resumable, one run per (row, variant).
9. The hypothesis stage: §9 decisions computed from the recorded runs only (`just citizens-c4-hypotheses citizens-c4-hypotheses-v1 --confirm-frozen --crossplay-run … --distill-run … --train-run <all planner runs> --b4-run <the 10 C4 B4 runs and the C2 k = 22 reference runs> --b4-eval-run <C2 b4-evaluate>`).

Every C4 evidence manifest records:
- the protocol SHA-256;
- `c4_config.config_sha256`;
- the C3 configuration hash;
- the commit, with `git_dirty = false`;
- the checksums of its input runs.

## 11. Forbidden actions

- **No re-tuning.** Hyperparameters, iterations, seeds, δ, α, thresholds, bins or keys are never changed after any cross-play number is seen: not in v1, and not in a v2 used for confirmatory claims. A v2 that changes them is exploratory.
- **No reuse of seen runs.** Pilot runs and pilot seeds (901–903) are never inputs. The C3 runs (seeds 1–3) are never inputs.
- **No added or dropped runs.** No seeds are added. No run is re-run with a new seed to replace a bad one. No seed, run or game is dropped, except a B4 discard under the C2 rule.
- **No changes to models or B4.** No refit or change of any citizen model, k\*, prompt, CLI flag or calibration weight.
- **No extra B4 games** beyond the declared count and reductions.
- **Exploratory analyses are labelled.** Any analysis not specified here is labelled **exploratory** and cannot change a verdict.

## 12. Stop / kill criteria

| # | Condition | Action |
|---|---|---|
| K1 | A training run exceeds the C3 per-run limit of 24 h | Stop that row; report; ask the owner |
| K2 | Any simulated game violates §5 | Stop; fix the bug before further evaluation; no verdict changes |
| K3 | The non-LLM or LLM projection exceeds its ceiling after the §7 reductions | Stop that arm; report; ask the owner |
| K4 | A row is optimiser-limited (§6.1, §9.3) | Do not stop. Label it; its cells are reported with the label. Re-tuning is only possible as an exploratory v2 (§11) |
| K5 | A run is "not converged" (§6.1) | Do not stop. Label it; training is not extended |
| K6 | The training signature differs between rows | Stop; the cross-play stage refuses; re-train from the freeze commit under new run ids |
| K7 | H1, H2 or H2-B4 outcomes | Reported as declared, whichever way they fall |

## 13. Owner decisions (answered 2026-10-20)

The owner answered O1–O7 on 2026-10-20. The questions and the drafted recommendations are kept for the record; the answers are integrated in the sections named.

- **O1. Robust mixture planner (S2). Answer: yes**, as a secondary analysis with 3 seeds (3001–3003; §4, §7, §9.7). It needs about 3.0 h serial and the mixture citizen port (P7). It is secondary and never decides H1 or H2.
- **O2. B4 budget. Answer: option (ii)**, both variants (B4-plain and B4-calibrated), with the drafted estimate of about 55,500 calls, a C4 LLM ceiling of 66,000 calls and the pre-declared reductions of §7 kept. The drafted options were (i) the 4 main planners plus k\*[B1] on one variant, ≈ 27.7k calls (ceiling 33k); (ii) the same on both variants; (iii) none. The drafted recommendation was (i).
- **O3. k\* row on B4. Answer: k\*[B1] (k = 9.03) is run on B4**, on each variant, not reused from C2 (§6.5, P6). The C2 k = 22 B4 run of the same variant stays a reference row only.
- **O4. Minimum effect. Answer: accepted as drafted**, δ = 16 coins (0.1 coin per player-round; C2's τ; about one B1 seed SD). It was chosen after the pilots were seen (§1).
- **O5. Confirmation rule. Answer: accepted as drafted.** A verdict requires both the crossed bootstrap and the seed-level t / Welch test to reject under Holm (§9.2); per-seed signs are always shown.
- **O6. Seeds and length. Answer: accepted as drafted**: 5 main seeds (1001–1005), 3 ablation seeds (2001–2003) and 1,000 iterations.
- **O7. Exploitation thresholds. Answer: accepted as drafted**: TV ratio 1.5 × k = 22's; input support +0.05 for B2 only (§9.5).

## 14. Code prerequisites (before the freeze)

At the time of drafting (HEAD 3a09e93) these were missing. All are implemented on the branch `model-citizens-c4`, each tested against independent references (analytic cases, NumPy or SciPy references, hand-computed examples); `just verify` passes. The default planner training is bit-identical to C3 (pinned parameter hash `5f6087ee…891394`, `tests/citizens/test_citizens_rl_ppo.py`, commit 61262b5).

| # | Prerequisite | Protocol § | Implemented |
|---|---|---|---|
| P1 | One training signature across all planner rows: iterations, PPO config, overrides, commit, fit run and configuration hashes identical, differing only in model (or mixture), seed and retention (`c4.group_training_runs`, `c4.shared_signature`). C3 runs, pilot runs, reserved seeds, undeclared rows or seeds and incomplete rows are refused. Training under `--milestone c4` refuses undeclared seeds, other iterations, overrides, other run ids and an unfrozen protocol | §4, K6 | 91c14bf |
| P2 | Kept share coin-weighted (Σ kept / Σ R over R ≥ 1) in cross-play, pilots, G3 (a new field; the C3 field `kept_share_per_round` keeps its mean-of-ratios meaning) and distillation κ | §6.3, §6.4 | 91c14bf, 8b93f13 |
| P3 | Distillation:<br>– deployed rules and the distilled k on the same distillation games, through the deployed rounding path;<br>– the recovery test on quantised-rule trajectories;<br>– the distilled-rule cross-play on every column;<br>– the in-sample note in the output | §6.4, §9.6 | 8b93f13 |
| P4 | Hypothesis stage (`rl.hypotheses`, `c4 hypotheses`):<br>– δ-shifted one-sided bootstrap p-values;<br>– seed-level t / Welch tests;<br>– per-seed signs;<br>– Holm per family;<br>– the verdicts of §9.3–9.7 and H2-B4 | §9 | 894ae92 |
| P5 | Input-support output labelled "input out-of-range" (B2, model extrapolation) or "state novelty" (B3, B23); undefined diagnostics are null with a reason, never NaN | §6.3 | 91c14bf |
| P6 | `c4 b4` accepts the deployed rule row k\*[B1] beside a planner run, with the guards of §6.5 | §6.5, O3 | 894ae92 |
| P7 | The mixture citizen (`citizens.MixtureCitizen`; per-game model draw) for training (`--model Mix`) and evaluation, with the 4-model mean for monitoring | §4 | 91c14bf |
| P8 | The Appendix B constants in `c4_config`; the appendix test extended to them | Freeze checklist | 91c14bf |

## Appendix A. Key constants (in code; checked by `tests/citizens/test_citizens_c4_protocol.py`)

```text
C3 PPO (chosen on B1 in C3): games/iteration=1024 rounds=40 epochs=4 minibatches=8 lr=1e-3 clip=0.2 gamma=1 gae_lambda=0.95 value_coef=0.5 entropy_coef=0 max_grad_norm=0.5 reward_scale=0.005 hidden=32 init_log_std=-0.5 chunk=10
C3 keys: evaluation 31000, k* tuning 31001 (ln k in [-5,5] step 0.1), monitoring 31002 (512 games), sampled 31003 | k published=22 | excluded C3 training seeds 1,2,3
C4 evaluation: games=1000 citizen key=41000 sampled noise key=41001 | rules: Equal, Mixed, Proportional, Interpolating k=22, Interpolating k* (one per model)
endgame rounds (0-based)=38,39 | bootstrap B=10000 seed=9101 level=0.95
input support: models B2,B3,B23 | B2 x1,x2,x3,x4 | B3 own_offer,own_prev_offer | B23 own_offer,own_prev_offer,x3 | grid bins=20
distillation: games=1000 key=41002 w bins=10 tree depth=3 min leaf=200 features pool_frac,own_prev_fraction,others_prev_fraction,own_prev_share
plateau: smooth 5 points, final 10 points, tolerance 0.02 | pilots (excluded): B2 901, B3 902, B23 903 (noise key 41010)
B4 planner games=40 | per-run training limit 24 h
```

## Appendix B. Protocol constants (in `c4_config`; checked by `tests/citizens/test_citizens_c4_protocol.py`; P8)

```text
training: iterations=1000 | main seeds 1001..1005 (B1,B2,B3,B23) | no-retention seeds 2001..2003 | mixture (Mix) seeds 3001..3003 | secondary rows reducible to 2 seeds
delta=16 coins total surplus per game (0.1 coin per player-round) | alpha=0.05 family-wise, Holm per family | confirmed = bootstrap and seed-level t both reject
families: H1-reversal 12, H1-transfer 12, H2a-loss 3, H2a-noninferior 3, H2b-loss 3, own-model-limited 4, own-model-gain 4, H2-B4 3, S1-reversal 12, S1-transfer 12, S2-reversal 4, S2-transfer 4 (H2-B4 per family and validated variant)
convergence: plateau iteration <= 900 and tail slope <= 16 coins per 100 iterations
exploitation: B1 mean TV > 1.5 x k=22 (same column), or (B2 only) range/grid fraction > k=22 + 0.05
w(R) agreement: bins with >= 200 informative rows for every seed of both models; same if L1 <= max(0.1, 2 x within-model L1) | fidelity: w-rule R^2 >= 0.5 | S1 retention: no-retention L1 < 0.5 x main L1
B4: variants plain,calibrated | planner seed 1001; comparator k*[B1] (k=9.03); 40 games per row (reduction 30; exploratory below 20); unpaired game bootstrap | validating regimes R-I,R-M2
ceilings: non-LLM 40 h wall-clock; LLM 66000 calls (O2 option ii, both variants)
```
