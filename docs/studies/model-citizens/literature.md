# Model Citizens — Literature scan (2026-10-20)

Scope: a focused nearest-neighbour scan for one question. It is not an open-ended review. A reference appears outside "Unverified leads" only if its arXiv ID, DOI record (Crossref / Europe PMC / Semantic Scholar), publisher or PMC page, or repository was fetched **this session** and the title, authors and year matched. The **Verified** line says what was fetched. Content claims are limited to what was read: an abstract-only entry is cited for nothing beyond its abstract.

No computation was run for this scan.

## Question as scanned

From `README.md` (v0):

> Does the mechanism an RL policy designer discovers depend on which validated model of people it is trained against?

| Component | Value |
|---|---|
| Domain | Koster et al. 2025 common-pool "trust" game: 4 players, pool 200, ×1.4 on reciprocated coins, 40 rounds; public CC-BY trial-level human data (4,952 UK/US participants) |
| Method | Fit citizen models (GRU clone; parametric conditional cooperation; EWA / Roth–Erev learning; LLM citizens) on human play under Equal/Mixed/Proportional only; validate on held-out Interpolating, M1, M2; PPO planners per validated model; cross-play matrix; robust (ensemble/minimax) planner; distil to closed forms |
| Claim type | Transfer / robustness of a learned mechanism across *validated* behaviour models (an empirical Lucas critique of AI policy design) |
| Baselines | Equal, Mixed, Proportional, Interpolating k = 22, re-tuned Interpolating k\* per model |

## Summary of the finding

1. **The nearest neighbour is the DeepMind line itself, and it is closer than the README states.** Democratic AI (Koster et al. 2022) already (a) validated its imitation clones on a **separate held-out human dataset** (n = 384; 288 blocks, including two endowment conditions absent from training) with hyperparameters swept on validation loss, and (b) trained a second mechanism against a **different behaviour-model class** (gradient-learning "rational players"), tested both mechanisms on new humans, and found the clone-trained mechanism preferred (57.2 %). So "the mechanism depends on the model of people" has been shown once — for one validated vs one *unvalidated* model, judged with humans.
2. Koster et al. 2025 compared clones with humans only visually, but it did **prospectively predict an unseen mechanism** (the clones had never seen interpolating-rule games). BC2, however, was trained on data that **included** interpolating and RL-agent games, so BC2 is not an independent held-out model.
3. The group's 2025 PNAS review (Tacchetti et al.) names our exact problem — simulated players diverge from humans, imitation fails out of distribution, cites Lucas (1976) — and suggests "training against a sufficiently diverse AI population" as a remedy. It does **not** do it.
4. **No work found** that fits several behaviour-model classes on the same human data, validates all on held-out mechanisms, trains a planner per model, and reports a cross-play matrix. Closest robustness work (ERMAS / Zhao et al. 2023) uses synthetic bounded-rational RL agents, no human data.
5. LLM-as-subject work has moved to **held-out-setting validation** (Manning & Horton 2025 "General Social Agents"; Ashokkumar et al. 2026 Nature), and to fitting LLM agents to data (Jackson et al. 2026 "type vectors"). Public-goods studies report LLM agents reproduce treatment *directions* (Sreedhar et al. 2025) but not the human strategic process (Teo 2026); a survey-experiment archive shows LLM-predicted effects are systematically too large (Ashokkumar et al. 2026). None compares LLM citizens with learned clones and structural models under held-out institutions.

**Novelty risk: medium-low** (details below). The claim must be narrowed: not "first to validate clones" or "first to show model dependence", but "first cross-model test among models that all pass the same held-out-institution validation".

## Closest works

### 1. Koster et al. 2025 — RL manager for a common-pool resource (the data source)
- R. Koster, M. Pîslar, A. Tacchetti, J. Balaguer, L. Liu, R. Elie, O. P. Hauser, K. Tuyls, M. Botvinick, C. Summerfield, "Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem," *Nature Communications* 16:2824 (2025). https://doi.org/10.1038/s41467-025-58043-7 ; arXiv:2404.15059. Data/code: https://github.com/google-deepmind/sustainable_behavior
- **Verified:** PMC full text (PMC11929920) read for validation, limitations and data statements; arXiv API record; Semantic Scholar citation list; repository (via `gh search code`). Game/clone/planner details: `spec-koster2025-game-data.md` (earlier session) and the study README.
- **What they did (validation of clones):**
  - BC1 trained on Train Set 1 (537 games: random Dirichlet mechanisms, baselines with varying w, 234 pilot RL "M0" games). Checkpoints selected for high surplus with hand-coded baselines and low surplus with the random mechanism — i.e. partly on the outcome of interest.
  - Fig. 2B / S1 / S11: side-by-side distributions, clones vs humans, under Equal, Mixed, Proportional and **M1** (M1's own games not in training; its prototype M0 was). Text: "without exception, the overlap is striking"; clones "performed almost exactly like real human players". No held-out likelihood or numeric fit metric in the text read.
  - Exp. 2: the interpolating rule was predicted to perform well by clones "even though they had not been trained on any data involving the interpolating baseline", then confirmed with humans. This **is** a prospective held-out-mechanism test, at outcome level, for one model.
  - BC2 (990 games) was trained on all earlier data **including** interpolating and RL-agent games (Exp. 2–3); M2 trained on BC2.
- **What they did NOT do:**
  - No second model class; no misspecification or cross-model analysis; M1 (BC1) never evaluated against BC2 or vice versa; robustness checks limited to an M1′ architecture variant, Exp. 4 (novice/incentive effects) and virtual-player manipulations.
  - Limitations stated: UK/US-only cohort, generalisation beyond it unknown; game differs from classic CPR; "no guarantees" in naturalistic environments.
  - Discussion claims "the model of human behaviour makes no assumptions about the structure of the game" — a claim our study can test directly.

### 2. Koster et al. 2022 — Democratic AI (closest on *model dependence*)
- R. Koster, J. Balaguer, A. Tacchetti, A. Weinstein, T. Zhu, O. Hauser, D. Williams, L. Campbell-Gillingham, P. Thacker, M. Botvinick, C. Summerfield, "Human-centred mechanism design with Democratic AI," *Nature Human Behaviour* (2022), doi:10.1038/s41562-022-01383-x ; arXiv:2201.11441.
- **Verified:** arXiv abstract page; full arXiv PDF text read (Methods "Virtual players", "Rational players", "Pilot testing", Supplementary limitations). Journal DOI confirmed via the Semantic Scholar citations endpoint (122 citing papers returned); journal volume/pages not fetched.
- **What they did:**
  - LSTM imitation clones of contributions, trained on > 4,000 pilot participants (4,809 blocks); **evaluated on a separate hold-out dataset** (n = 384; 288 blocks) "that resembled as closely as possible the final conditions"; hyperparameters swept to minimise validation loss; the validation set covered two endowment conditions not in training. Voting model: logistic in relative payout, slope fixed at 1.4 with a sensitivity sweep.
  - **Alternative behaviour model:** "rational players" that learn by gradient ascent on their own return (learning-rate and initial-generosity parameters), with a parameter cross-evaluation among rational-player settings.
  - Exp. 3 (n = 736): humans preferred the clone-trained mechanism (HCRM) over the rational-player-trained mechanism (RM), 57.2 %; RM's radical policy made the head player stop contributing. Under equal endowments HCRM ≈ RM (47.9 %). Authors: "human data may be crucial when using AI for mechanism design."
  - Notes that virtual players had to "generalize beyond the training dataset" because the mechanism under training produced payouts outside the human data.
- **What they did NOT do:** a different game (one-shot public goods with redistribution, 10 rounds, objective = votes); only two model classes, and the rational class was never validated on human data; no cross-play matrix (each mechanism judged by humans, not by the other model); no robust/ensemble planner; held-out set was held-out *participants and endowments*, not held-out *mechanisms*.

### 3. Tacchetti et al. 2025 — "Deep mechanism design" (PNAS review; states our gap)
- A. Tacchetti, R. Koster, J. Balaguer, L. Liu, M. Pîslar, M. M. Botvinick, K. Tuyls, D. C. Parkes, C. Summerfield, "Deep mechanism design: Learning social and economic policies for human benefit," *PNAS* 122 (2025), doi:10.1073/pnas.2319949121 (Review).
- **Verified:** Semantic Scholar record (title, authors, abstract, type); PMC full text (PMC12207456) read for modelling/robustness passages. PNAS page itself returned 403.
- **What they say:** human models are required yet "the mechanism itself is needed to train those models"; imitation "will not account for any equilibrium effects" and performs poorly outside its training distribution; "divergence between simulated (agent-based) players and humans may arise"; cites Lucas's critique (their ref. 60); mechanisms are not guaranteed robust to strategic behaviour; "training against a sufficiently diverse AI population may be an alternative solution". Reports AI Economist human validation as "somewhat equivocal".
- **What they did NOT do:** no ensembles, multiple validated human models or domain randomisation are discussed or tested; no LLM citizens.

### 4. Zhao, Trott, Xiong & Zheng 2023 — robust principal against bounded-rational agents (closest on *robust planner*)
- E. Zhao, A. R. Trott, C. Xiong, S. Zheng, "Learning to Play General-Sum Games Against Multiple Boundedly Rational Agents," AAAI 2023; arXiv:2106.05492 (v1 2021; DeepAI lists the earlier title "ERMAS: Becoming Robust to Reward Function Sim-to-Real Gaps in Multi-Agent Simulations").
- **Verified:** arXiv abstract and comments (AAAI 2023).
- **What they did:** robust principal trained with RL against worst-case (ε-)best responses found by no-regret dynamics, extended to bounded rationality; dynamic tax policy +15 % welfare in a trade-and-barter economy, including against unseen bounded-rational RL taxpayers.
- **What they did NOT do:** no human data; uncertainty set is a neighbourhood of rational responses, not a set of empirically validated human-model classes.

### 5. AI Economist — Zheng et al. 2022 and the 2020/2021 versions
- S. Zheng, A. Trott, S. Srinivasa, D. C. Parkes, R. Socher, "The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning," *Science Advances* 8(18) (2022), doi:10.1126/sciadv.abk2607. Earlier: arXiv:2004.13332 (2020, with MTurk human experiments); arXiv:2108.02755 (verified in the earlier topic scan). Follow-up: A. Trott et al., "Building a foundation for data-driven, interpretable, and robust policy design using the AI Economist," arXiv:2108.02904 (2021).
- **Verified:** Crossref record with abstract (Science Advances); PMC9067926 text read for limitations; arXiv abstracts of 2004.13332 and 2108.02904.
- **What they did:** two-level RL, planner and RL agents co-adapting; Science Advances version says conclusions "are not meant to be applied in the real world", simulations "lack grounding in real-world data" and rational-RL behaviour is less relevant where behavioural considerations matter. The 2020 preprint reports MTurk experiments in which the AI policy gave an equality–productivity trade-off similar to Saez and higher inverse-income-weighted welfare. The 2021 COVID follow-up reports robustness to calibration errors (over/under-estimated infection rates).
- **What they did NOT do:** agents are rational RL learners, not fitted to human play; robustness is to simulator parameters, not to the behaviour-model class.

### 6. Manning & Horton 2025 — General Social Agents (closest on *LLM citizens validated on held-out settings*)
- B. S. Manning, J. J. Horton, "General Social Agents," arXiv:2508.17407 (v5; NBER w34937 per search, not fetched).
- **Verified:** arXiv API record (title, authors, date, abstract).
- **What they did:** LLM agents built from theory-grounded instructions plus human data from a few "seed" games, then evaluated on a population of 883,320 novel games without target-game data; preregistered tests on held-out games against equilibrium predictions and off-the-shelf agents (later versions add cognitive hierarchy; per search snippets, not the fetched abstract).
- **What they did NOT do:** no mechanism/policy design, no RL planner, no comparison with learned clones in a dynamic multi-round institution. Their "fit on seed settings, test on novel settings" design is the right template for our B4.

### 7. Other mechanism-design-against-learned-models work (brief)
- **HCMD-zero** — J. Balaguer et al., arXiv:2202.10122 (2022). Verified: arXiv abstract. Mechanism improved via elections against copies of itself, using human data "sparingly"; abstract claims no assumptions about participant rationality. Full text not read: whether it uses clones is unknown.
- **The Good Shepherd** — J. Balaguer, R. Koster, C. Summerfield, A. Tacchetti, arXiv:2202.10135 (2022). Verified: arXiv abstract. Mechanisms trained against learning participants (nested loops), tested with artificial agents and humans.
- **SWM-AP** — X. Zhang et al., "Social World Model-Augmented Mechanism Design Policy Learning," arXiv:2510.19270 (2025). Verified: arXiv abstract. A trait-inferring world model of agents trains the mechanism policy (tax, team coordination, facility location); no human data or misspecification test in the abstract.
- **Qin, Zou & Wang 2026** — "Exploring cooperation mechanisms via deep reinforcement learning in network common-pool resource games," arXiv:2606.05867. Verified: arXiv abstract. GNN-RL allocator vs equal/proportional on networks; simulated agents only.
- **LLM Economist** — S. Karten et al., arXiv:2507.15815 (2025). Verified: arXiv abstract. Persona-conditioned LLM workers calibrated to US Census statistics; in-context-RL planner sets tax schedules. No human behavioural validation in the abstract.

## Conceptual frame: Lucas critique and held-out-policy validation

- **Lucas 1976.** R. E. Lucas, "Econometric policy evaluation: A critique," *Carnegie-Rochester Conference Series on Public Policy* 1:19–46, doi:10.1016/S0167-2231(76)80003-6. Verified: Crossref record (abstract not available; content not re-read — cite only as the origin of the critique).
- **Heckman 2010.** "Building bridges between structural and program evaluation approaches to evaluating policy," *J. Econ. Lit.* 48(2):356–398, doi:10.1257/jel.48.2.356. Verified: Crossref with abstract. Frame for structural (B2/B3) vs reduced-form/flexible (B1) models.
- **Todd & Wolpin 2006** — the template for our validation gate. "Assessing the impact of a school subsidy program in Mexico: using a social experiment to validate a dynamic behavioral model…," *AER* 96(5):1384–1417, doi:10.1257/aer.96.5.1384. Verified: Crossref with abstract. A structural model estimated **without post-program data** predicts the experimental impacts of a policy it never saw.
- **Prediction competitions on new treatments.** Erev, Ert & Roth 2010, "A choice prediction competition for market entry games," *Games* 1(2):117–136, doi:10.3390/g1020117; Ert, Erev & Roth 2011, "…for social preferences in simple extensive form games," *Games* 2(3):257–276, doi:10.3390/g2030257. Verified: Crossref with abstracts. Estimation experiment and a separate competition experiment on new games drawn from the same distribution.
- **ML vs theory out of sample.** Fudenberg & Liang 2019, "Predicting and understanding initial play," *AER* 109(12):4112–4141, doi:10.1257/aer.20180654 (Crossref + abstract): ML finds regularities; a hybrid model predicts new algorithmically generated games. Fudenberg & Karreskog Rehbinder 2024, "Predicting cooperation with learning models," *AEJ: Micro* 16(1):1–32, doi:10.1257/mic.20220148 (Crossref + abstract): a simple learning model predicts **out-of-sample** cooperation in repeated PD at least as well as models with more parameters and ML. Zhu, Peterson, Enke & Griffiths, arXiv:2408.07865 (2024; abstract): a deep net beats leading theories on 90,000 initial-play decisions, then is distilled into an interpretable model — the same "flexible model → interpretable model" move as our distillation step.

## Behavioural model classes (candidate B2/B3)

- **Conditional cooperation.** Fischbacher, Gächter & Fehr 2001, *Economics Letters* 71(3):397–404, doi:10.1016/S0165-1765(01)00394-9 (Crossref + RePEc abstract): strategy-method elicitation; ~50 % conditional cooperators, about a third free riders. Fischbacher & Gächter 2010, *AER* 100(1):541–556, doi:10.1257/aer.100.1.541 (Crossref + abstract): decline of cooperation driven by *imperfect* conditional cooperation (contributing less than others). → B2 should include an under-matching slope on others' previous reciprocation fraction.
- **Payoff-based learning.** Burton-Chellew & West 2021, "Payoff-based learning best explains the rate of decline in cooperation across 237 public-goods games," *Nat. Hum. Behav.* 5(10):1330–1338, doi:10.1038/s41562-021-01107-7 (Crossref + Europe PMC abstract): meta-data favour "confused learners" over altruism. → supports B3 as a serious rival, not a strawman.
- **EWA / reinforcement.** Camerer & Ho 1999, *Econometrica* 67(4):827–874, doi:10.1111/1468-0262.00054 (Crossref + RePEc abstract): EWA nests reinforcement and weighted fictitious play. Roth & Erev 1995, *GEB* 8(1):164–212, doi:10.1016/S0899-8256(05)80020-X (Crossref + RePEc abstract): simple learning models track intermediate-term play.
- **Hybrid in public goods.** Janssen & Ahn 2006, "Learning, signaling, and social preferences in public-good games," *Ecology and Society* 11(2):21, doi:10.5751/ES-01752-110221 (Crossref + article page): EWA + Charness–Rabin social preferences + signalling fitted by ML to Isaac–Walker data; selection by AIC/BIC on the **same** data — no held-out treatment test. → A ready B2+B3 hybrid; our held-out-mechanism validation is the missing step.
- **CPR lab.** Janssen, Holahan, Lee & Ostrom 2010, "Lab experiments for the study of social-ecological systems," *Science* 328(5978):613–617, doi:10.1126/science.1183532 (Crossref; summary only).

## LLM citizens (B4)

- **Foundations.** Horton, Filippas & Manning, "Large language models as simulated economic agents: What can we learn from Homo silicus?", arXiv:2301.07543 (abstract): qualitatively similar results to classic experiments. Aher, Arriaga & Kalai, arXiv:2208.10264, ICML 2023 (abstract): "Turing Experiments"; replicate Ultimatum etc., with consistent distortions in some. Argyle et al., "Out of one, many," *Political Analysis* 31(3):337–351 (2023), doi:10.1017/pan.2023.2 (Crossref + abstract): "algorithmic fidelity" for subpopulations.
- **Individuals and scale.** Park et al., "LLM agents grounded in self-reports enable general-purpose simulation of individuals," arXiv:2411.10109 (abstract; 1,052 Americans, held-out GSS items). Binz et al., "A foundation model to predict and capture human cognition" (Centaur), *Nature* 644:1002–1009 (2025), doi:10.1038/s41586-025-09215-4 (Crossref + arXiv:2410.20268 abstract): beats cognitive models on held-out participants across 160 experiments. Ashokkumar, Hewitt, Ghezae & Willer, "Large language models can predict the results of social science experiments," *Nature* 656:115–122 (2026) (Stanford GSB publication page; Nature DOI not fetched): r with true effects comparable to human forecasters, **effects systematically overestimated**.
- **Public goods / trust / CPR.** Sreedhar et al., arXiv:2502.12504 (2025; abstract): multi-agent LLMs replicate PGG lab results under priming, transparency and varying endowments, and predicted combinations of treatments. (A search snippet of the full text says effect sizes were often exaggerated — not verified.) Teo, arXiv:2606.27845 (2026; abstract): 360-cell factorial of local LLM settings vs published human distributions in beauty contest and PGG; dispersion partly recoverable via settings, the strategic process not; in the multi-round PGG human contributions decayed while LLM contributions "stayed flat or rose at every scale", with no within-game belief updating. → a direct risk for B4 in a 40-round game, and a concrete validation check (round-by-round reciprocation trend). Xie et al., arXiv:2402.04559, NeurIPS 2024 (abstract): GPT-4 agents show "high behavioral alignment" with humans in trust games. Piatti et al., "Cooperate or collapse" (GovSim), arXiv:2404.16698, NeurIPS 2024 (abstract): LLM societies mostly fail to sustain a commons; no human data. Willis, Zhao, Du & Leibo, arXiv:2602.16662 (2026; abstract): strategy-as-code LLM populations in PGG/CPR games. Gupta et al., arXiv:2510.14401, AAMAS 2026 (abstract): CPR without explicit rewards, norms and social learning.
- **Fitting LLMs to data.** Jackson, Manning, Xie, Yuan & Mei, "Modeling human behavior with type vectors using AI," arXiv:2608.18265 (2026; abstract): trait-intensity prompts fitted to 119,147 game decisions. → a principled way to *fit* B4 on baseline data rather than prompting cold.
- **Critiques.** Bisbee et al., *Political Analysis* 32(4):401–416 (2024), doi:10.1017/pan.2024.5 (Crossref + abstract): means match, variance too low, unstable across prompt wording and time. Wang, Morgenstern & Dickerson, arXiv:2402.01908 (Nature Machine Intelligence per arXiv comment): LLMs misportray and flatten identity groups. Gao, Lee, Burtch & Fazelpour, arXiv:2410.19599 (abstract): fail to replicate 11–20 game distributions.
- **Group link.** Jarrett, Pîslar, Bakker, Tessler, Köster, Balaguer et al., "Language agents as digital representatives in collective decision-making," arXiv:2502.09369 (2025; abstract): the DeepMind group already trains language agents as stand-ins for humans facing a mechanism — the natural route by which they could scoop a B4-style study.

## Robust / ensemble planning (closest 5)

- Iyengar 2005, "Robust dynamic programming," *Math. Oper. Res.* 30(2):257–280, doi:10.1287/moor.1040.0129; Nilim & El Ghaoui 2005, *Oper. Res.* 53(5):780–798, doi:10.1287/opre.1050.0216 (Crossref + abstracts): max–min over rectangular transition-uncertainty sets.
- Rajeswaran et al., EPOpt, arXiv:1610.01283, ICLR 2017 (abstract): ensemble of source domains + CVaR-style adversarial training. → the natural H2 "minimax-over-models" baseline.
- Tobin et al., domain randomisation, arXiv:1703.06907 (abstract). Kurutach et al., ME-TRPO, arXiv:1802.10592 (abstract): policies exploit regions where a learned model lacks data — exactly the risk of a planner exploiting a clone's out-of-distribution behaviour.
- Bergemann & Morris 2005, "Robust mechanism design," *Econometrica* 73(6):1771–1813, doi:10.1111/j.1468-0262.2005.00638.x (Crossref; title only): "robust" in mechanism design usually means robustness to agents' beliefs, a different notion from ours — avoid the bare term.

## MARL on commons with human comparison (brief)

- Perolat et al. 2017, arXiv:1707.06600 (abstract): MARL model of CPR appropriation in a spatio-temporal commons. Hughes et al. 2018, arXiv:1803.08884 (abstract): inequity aversion in sequential social dilemmas. McKee et al. 2020, arXiv:2002.02325, AAMAS (abstract): social value orientation diversity. McKee et al. 2021, arXiv:2103.04982 (abstract): humans cooperate in Clean Up with identifiable partners but not under anonymity, and a MARL reputation model reproduces the difference. Leibo et al. 2021, Melting Pot, arXiv:2107.06857, ICML (abstract): evaluation on held-out co-player populations — the MARL analogue of our cross-play matrix.

## How this study differs

- **Several model classes, one validation gate.** Prior work used one human-fitted model (2025), or one human-fitted model plus one *unvalidated* theory model (2022). We fit ≥ 3 classes on identical training data and admit only those passing the same held-out-mechanism criterion.
- **Held-out mechanisms, not held-out participants.** 2022 held out participants and endowment conditions; 2025 checked M1 visually and predicted the interpolating outcome for one model. We hold out three mechanisms (Interpolating, M1, M2) for *all* models, with quantitative likelihood and outcome-CI criteria fixed before unblinding. Our own GRU (B1) is trained on baselines only, unlike BC2.
- **Cross-play matrix instead of human tests.** Each planner is scored under every other validated model; the question is whether validated models agree on the designed mechanism, which neither paper asked.
- **Robust planner across empirically validated models.** ERMAS-style robustness uses synthetic ε-rational neighbourhoods; EPOpt/DR use parameter ranges. Ours is an ensemble/minimax over *data-validated* human-model classes — the "diverse population" remedy Tacchetti et al. 2025 suggest but did not test.
- **LLM citizens as a fourth, independently built model,** validated by the same gate and compared with learned and structural models in a dynamic institution — not done in the LLM-subject literature found.
- **Distilled rules compared across models.** The 2025 paper distilled one rule (w = (R/200)^22) from one clone; we ask whether the distilled rule is model-invariant.

## Novelty risk

**Medium-low.** No paper found does the cross-model, held-out-validated, cross-play design, in this game or elsewhere. Risk factors:
- **The DeepMind group is best placed** (data, clones, LLM-representative work, PNAS review naming the gap). A follow-up with clone ensembles or LLM citizens would overlap heavily.
- **Partial precedent** (2022 HCRM vs RM) means the *headline* "it depends on the model of people" is not new by itself; the novelty is the restriction to validated models and the systematic matrix.
- **Weak effect sizes in the source data** (M2 vs interpolating surplus with humans: z = 2.35) mean H1 may fail for reasons unrelated to misspecification. That is still a reportable result.

## Baselines implied by prior work

- **Rational / payoff-maximising citizen** (Koster 2022 "rational players"; Burton-Chellew & West). Recommended as a **negative control model**: it should *fail* the validation gate; if it passes, the gate is too weak.
- **Re-tuned interpolating rule k\*** per model (Koster 2025's own distillation route) — already in the README.
- **Published-k = 22 rule and the human outcome table** for the held-out mechanisms (Koster 2025).
- **Marginal-frequency / last-action baselines** for one-step likelihood (standard in Fudenberg-style prediction work).
- **Ensemble/minimax planner** à la EPOpt for H2; a **domain-randomised** planner over B2/B3 parameter posteriors as a cheaper variant.
- **Teacher-forced vs closed-loop validation** both reported (Democratic AI trained mechanisms closed-loop, without teacher forcing).

## Unverified leads

- Hertz, Köster, Janssen & Leibo, "Beyond the matrix: Experimental approaches to studying cognitive agents in social-ecological systems," *Cognition* 254:105993 (2025), doi:10.1016/j.cognition.2024.105993 — metadata verified (Crossref), content **not read**; likely discusses lab CPR models and could hold relevant validation claims.
- Ashokkumar et al. 2026 *Nature* — verified only via the Stanford GSB page; Nature DOI/landing page not fetched.
- "AI Surrogacy in Psychological Research" (D'Alessandro et al.?) — appears in Semantic Scholar citations of both Koster papers with incomplete metadata; not found.
- Zhu et al. 2024 "LLMs as synthetic participants in auction design" — cited by search-result snippet only.
- HCMD-zero full text — whether it uses behavioural clones and any misspecification analysis.
- Manning & Horton NBER w34937 / Econometrica R&R status — from search snippet only.
- Fagiolo et al. and other ABM-validation reviews ("simulation to reality" for agent-based policy models) — not searched in depth.

## Search log (all 2026-10-20)

| # | Query / action | Source | Notable hits |
|---|---|---|---|
| 1 | citations of DOI 10.1038/s41467-025-58043-7 | Semantic Scholar API | 12 citing works; none is a reanalysis or cross-model test (Balepur et al. FAccT 2026 = trust-contingent RL, simulation only; Hauser 2025 essay) |
| 2 | citations of DOI 10.1038/s41562-022-01383-x | Semantic Scholar API | 122; Tacchetti et al. PNAS 2025, HCMD-zero, Good Shepherd, SWM-AP, Social Environment Design, Dizaji 2024 |
| 3 | Koster 2025 full text: validation, limitations, cross-clone | PMC11929920 | quotes above; no cross-clone evaluation |
| 4 | Democratic AI full PDF | arXiv 2201.11441 (pdftotext) | held-out validation set; rational-player mechanism RM; Exp. 3 57.2 % |
| 5 | Deep mechanism design review | Semantic Scholar + PMC12207456 | names sim-to-human divergence, cites Lucas, suggests diverse populations |
| 6 | AI Economist robustness / sim-to-real critique | WebSearch, Crossref, arXiv | Zhao et al. (ERMAS) 2106.05492; Trott et al. 2108.02904; Science Advances limitations |
| 7 | Lucas, Heckman, Todd–Wolpin, Fudenberg–Liang, Fudenberg–Karreskog, Erev–Ert–Roth | Crossref | all metadata + abstracts where present |
| 8 | conditional cooperation, EWA, Roth–Erev, Burton-Chellew, Janssen–Ahn, Janssen et al. 2010 | Crossref, RePEc, Europe PMC, E&S page | Janssen–Ahn: EWA+SP hybrid, no held-out test |
| 9 | LLM agents public goods human validation 2025 | WebSearch (extended) | Sreedhar et al. 2502.12504; Park et al.; Teo 2026 |
| 10 | General Social Agents | WebSearch + arXiv API | 2508.17407 |
| 11 | LLM vs behavioural models on held-out games | WebSearch (extended) | Zhu et al. 2408.07865; Centaur; no direct comparison under held-out institutions |
| 12 | RL planner vs LLM agents in CPR with human validation | WebSearch (extended) | none; Qin et al. 2606.05867; Gupta et al. 2510.14401 |
| 13 | reuse of `sustainable_behavior` data | WebSearch, `gh search code` | only the DeepMind notebook; Zenodo release 2025-01-23 |
| 14 | DeepMind group 2025–26 papers | arXiv API (`au:` queries), WebSearch | Jarrett et al. 2502.09369; no CPR follow-up |
| 15 | arXiv abstract searches: behavioural clones + mechanism; "Lucas critique" + RL; LLM + mechanism design + human | arXiv API | nothing closer than above |
| 16 | robustness to choice of behaviour model, ensembles, planner misspecification | WebSearch (extended) | Jackson et al. 2608.18265 type vectors; Large Legislative Models 2410.08345 (LLM policymaker, synthetic agents) |
| 17 | robust MDP / EPOpt / DR / ME-TRPO / RARL / Bergemann–Morris | Crossref, arXiv API | metadata + abstracts |
| 18 | LLM-subject critiques | Crossref, arXiv API | Bisbee et al. 2024; Wang et al. 2402.01908; Gao et al. 2410.19599 |

## Positioning

**What is new (defensible claims, if results hold):**
1. The first systematic test, to our knowledge, of whether an RL-designed mechanism transfers across **several behaviour models that all pass the same held-out-mechanism validation** on real human data.
2. A public, reproducible cross-play matrix and robust (ensemble/minimax) planner over empirically validated human models in the Koster et al. 2025 environment.
3. A like-for-like comparison of learned (GRU), structural (conditional cooperation, EWA) and LLM citizens under held-out institutions in a dynamic common-pool game.
4. A check of whether the distilled interpolating rule is model-invariant.

**What would make it redundant:**
- A DeepMind (or other) paper that trains planners against multiple clone families or an ensemble *and* reports cross-model evaluation on this or a similar dataset.
- A paper that fits structural or LLM models to `sustainable_behavior.csv` and validates them on M1/M2/Interpolating games — this would take our validation contribution; the planner cross-play would remain.
- A general "mechanism design under behavioural-model uncertainty with human-validated model sets" paper in another game. Our contribution would shrink to a replication in a new domain.

**Claim restrictions (must respect):**
- Do **not** claim that no one validated behavioural clones on held-out data: Koster 2022 used a held-out human set (n = 384) including unseen endowment conditions; Koster 2025 compared clones with humans under M1 (not in BC1 training) and prospectively predicted the unseen interpolating rule. Correct phrasing: *"validation in prior work was visual or on held-out participants/conditions for a single model class; no study validated multiple model classes on held-out mechanisms."*
- Do **not** claim that no one showed mechanism outcomes depend on the behaviour model: Koster 2022 Exp. 3 (HCRM vs rational-trained RM) shows it with humans. Our claim is about **validated** models and cross-model transfer.
- Do **not** claim the first robust AI planner against behavioural uncertainty (Zhao et al. 2023; EPOpt; DR). Claim robustness over **empirically validated human-model classes**.
- Do **not** claim the Lucas-critique framing as new for AI mechanism design: Tacchetti et al. 2025 cite Lucas and discuss imitation's equilibrium blindness. We provide an **empirical test** of it.
- Do **not** call DeepMind's BC2 a held-out model; it saw interpolating and RL-agent games. BC1/BC2 simulated rows are diagnostic only, as the README says.
- The README's "No other model class was tried, and there is no misspecification test" is true of the 2025 paper only; the lineage (2022) did try one. Wording in the README should be adjusted (proposal below).
- No claim about real people under new planners: all planner comparisons are in-silico across validated models (no new human experiments). Population is UK/US Prolific only.
- LLM citizens: report fidelity honestly and expect under-dispersion and exaggerated effects (Bisbee; Ashokkumar). Note **contamination risk**: Koster 2025 has been public since April 2024 (arXiv), so the model may know the paper and its results; the prompt must not name it, and a memorisation probe should be logged.
- Avoid the bare term "robust mechanism design" (Bergemann–Morris meaning differs).

**Design implications to consider (not decided here; propose for `docs/decisions.md`):**
- Add a **rational / payoff-maximising citizen** as a negative-control model that the validation gate is expected to reject (precedent: Koster 2022 RM).
- Fit B4 with a data-driven procedure (seed-setting fitting à la Manning & Horton, or type vectors à la Jackson et al.) on baseline mechanisms only, so "fitted only on baselines" is true for B4 as well.
- Consider a **hybrid B2/B3** (conditional cooperation + EWA), following Janssen & Ahn, if neither alone validates.
- H1 is a strong "every other model" condition with modest human effect sizes behind it. The protocol should pre-declare that an H1 failure caused by planners exploiting a model's out-of-distribution regions (ME-TRPO phenomenon) is diagnosed via state-visitation overlap with human data.

**Suggested venue:** AAMAS or EC (with a workshop version first, e.g. NeurIPS/ICML cooperative-AI or "AI for policy" workshops). If the LLM-citizen comparison is strong, *Nature Human Behaviour* / *PNAS Nexus*–style behavioural-science venues.

## Scoop watch

- Google DeepMind Koster / Tacchetti / Balaguer / Pîslar / Summerfield: follow-ups to the 2025 paper, clone ensembles, LLM representatives (Jarrett et al. 2502.09369 line).
- New citations of DOI 10.1038/s41467-025-58043-7 (12 as of today) and any reuse of `sustainable_behavior.csv` (GitHub code search).
- Manning & Horton (General Social Agents) and Jackson / Manning type-vector work moving into mechanism or policy design.
- Salesforce / Zheng-Trott robust-planner line, and SWM-AP authors (Zhang, Yang et al.) adding human data.
- LLM-economy simulators (LLM Economist; Large Legislative Models; Social Environment Design) adding human validation.
- Centaur-style foundation models applied to multi-player economic games as citizen models.
- Qin, Zou & Wang (network CPR GNN planner) adding human or empirically fitted agents.
- arXiv alerts: "behavioural clones" + "mechanism"; "Lucas critique" + "reinforcement learning"; "LLM agents" + "common-pool" + "human data"; "cross-play" + "social planner".
