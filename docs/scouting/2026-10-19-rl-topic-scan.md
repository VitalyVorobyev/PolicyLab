# RL topic scan after Gravity Atlas (2026-10-19 to 2026-10-20)

Reference record of the search that chose the lab's second study. Outcome: **Model Citizens** (D31, [`../studies/model-citizens/README.md`](../studies/model-citizens/README.md)).

This file is a record, not a citable source. The agents' reports were working notes; any reference below must be re-verified before it is cited in a study document.

## 1. Brief

- **Trigger.** Gravity Atlas closed on 2026-10-19 (D29). The next study had to be chosen.
- **Rejected first attempt.** The 2026-10-18 v2 shortlist (fisheries harvest-control rules, heat pumps / EVs, deep-RL insulin vs OpenAPS, FEM cubature, RealPage remedy, TD vs ANCCR, information engines, RL vs exact optimal control) was rejected by the owner on 2026-10-19. RL was peripheral, and the fields did not match his interests.
- **Owner's brief (2026-10-19).**
  - RL is central, not a decorative arm.
  - Preferred fields: social behaviour, political-system simulation, algorithm discovery.
  - A real research context: anchored to a disputed empirical regularity or a real dataset, so the claim is about the world, not the algorithm. Studies where emergence is "easy to manufacture" are weak.
  - The result must be visualisable.
- **Owner-provided input.** An external report, [`../chatgpt-deep-research.md`](../chatgpt-deep-research.md) (ChatGPT deep research, 2026-10-19), was audited in round 2 (§5).

## 2. Criteria

Each candidate was scored on four axes: **novelty risk** (scoop or crowding), **real need** (a named community with the problem), **RL centrality** (is the learning rule the object or mechanism of the question?), and **visual appeal**. Hard requirements:

1. A falsifiable question with a kill test of two days or less.
2. An external benchmark that the model is not fitted to (public data or published estimates).
3. Laptop compute (Apple M4 Pro).
4. No pure-maths or purely definitional contribution.

## 3. Method

- **Round 1 (2026-10-19), three literature agents**, one per field, each given seed ideas plus freedom to add its own:
  - `r1-political`: political-system simulation;
  - `r1-social`: social behaviour;
  - `r1-algorithm`: algorithm discovery and RL-as-designer.
- **Round 2 (2026-10-19/20), three agents:**
  - `r2-audit-chatgpt1-infobound`: audit of the ChatGPT report's #1 (and its #3);
  - `r2-audit-chatgpt2-causal-emergence`: audit of its #2;
  - `r2-rl-disputed-regularities`: RL candidates anchored to disputed empirical regularities.
- **Spec extraction (2026-10-20):** `spec-koster2025-game-data` read Koster et al. 2025 (full text and supplement), the `sustainable_behavior` repository notebook and byte ranges of the CSV, plus Koster et al. 2022. Its findings are in the Model Citizens spec.
- **Verification rule.** A reference counts as *verified* only if the agent fetched its abstract, DOI landing page, publisher page, arXiv page or repository in the same session and the title, authors and year matched. Everything else is *unverified* (search snippets, memory, blocked pages). Quotes come from fetch-tool extractions and must be checked against the PDF before reuse. Section 8 keeps each report's own split; nothing was upgraded here.

## 4. Candidates and verdicts

Scores are the agents' own: novelty risk · need · RL centrality · visual.

### 4.1 Finalists (ranked)

| # | Candidate | Source | Verdict |
|---|---|---|---|
| 1 | **Model Citizens** (the "clone gap" / simulator problem): does an RL mechanism designer's discovery survive a change of the behavioural model it trains against? Anchor: Koster et al. 2025 public CC-BY data. | r1-algorithm #1 | **Chosen (D31).** Low–med · high · high · med-high. Open question stated in the paper itself; public trial-level human data under held-out mechanisms; no 2024–26 follow-up found. |
| 2 | **Bubbles and experience**: does backward TD propagation explain why lab bubbles crash earlier with experience? Anchor: experimental-asset-market replication (OSF 5ypku, 166 markets). | r2-rl #2 | Runner-up. 3 · 4 · 5 · 5. Risk: humans see only 3–4 markets, so a pre-trained prior may become a free choice; crowded economics literature; period-level repeated-market data not yet confirmed. |
| 3 | **Blind retrospection** as a credit-assignment failure: a fixed-α TD voter fitted on Huber, Hill & Lenz 2012 lab data, tested on field shocks (Graham et al. 2021 data). | r2-rl #1 | Runner-up. 2 · 4 · 4 · 4. Risk: mapping lab rounds to calendar time is arbitrary; only signs and relative decay are testable. |
| 4 | **Learning Duverger**: voter vs entrant learning channels under plurality and runoff, tested against Fujiwara 2011 and Bordignon et al. 2016 RD signs. | r1-political #1 | Runner-up. Med · high · high · high. Risk: "Laver already did this", alliance actions needed for the runoff channel, high 2026 soc-ph scoop rate. |

### 4.2 Also viable (not shortlisted)

| Candidate | Source | One-line verdict |
|---|---|---|
| Choosing to be governed: do self-interested learners migrate into the sanctioning institution (Gürerk et al. 2006; 7-lab replication)? | r1-social #1 | Strong (low–med · med–high · high · high); humans learn faster in 30 rounds than standard RL; an LLM study of the same game exists without human comparison. |
| Planning-depth identifiability: does fitted "depth" rise in AlphaZero agents with fixed search but better values (van Opheusden et al. 2023)? | r2-rl #3 | Strong RL and visuals (3–4 · 4 · 5 · 5); cognitive science rather than social/political; Ma-lab scoop risk; fitting-code availability unconfirmed. |
| Topological vs metric flocking: does RL rediscover the jackdaw context switch (Ling et al. 2019)? | r1-algorithm #2 | Med · med · high · high; Celani/Ginelli group adjacent; result sensitive to observation encoding; data access unconfirmed (403). |
| Producer blind spot of feed experiments: learning producers invisible to consumer-side RCTs. | r1-social #2 | Med · high · med–high · high; key quantity is an unobservable counterfactual; fast-moving industry research; mild dual use. |
| Learning legislators: Baron–Ferejohn learning vs pooled lab data and the formateur bonus. | r1-political #2 | Low–med · med–high · high · med; pooled data access uncertain; lab-to-cabinet link weakest. |
| Discovered RL algorithms (Disco103, DPO, LPO) in the IPD and other social dilemmas. | r1-algorithm #3 | High novelty risk (FLAIR, DeepMind); anchor is a benchmark, not a real user. |
| Wave-smoothing AVs vs human drivers who learn to exploit larger gaps. | r1-algorithm #4 | Med · med-high · med-high · high; crowded transport RL; arbitrary lane-change model; less "social". |
| Turnout habit: outcome-reinforced (Bendor et al. 2003) vs outcome-blind (Fowler 2006), against rain-shock estimates. | r1-political #3 | Med · med · high · med–low; county aggregation blurs individual wins and losses; replication page returned 403. |
| Political budget cycles as two-sided learning (Brender & Drazen 2005). | r1-political #4 | Few elections per agent, so RL collapses toward Bayesian updating; panel data must be rebuilt. |
| Misinformation sharing as reward-learned habit (Ceylan, Anderson & Wood 2023). | r1-social #3 | Med · med · high · med; only 16 test trials may not separate the models. |

### 4.3 Killed or scooped

| Idea | Reason (scooping or blocking reference as reported) |
|---|---|
| Community Notes manipulation | Scooped: Truong et al. 2025 (arXiv 2511.02615); Selvam et al. 2026 (arXiv 2607.01824, with X's Community Notes team); resistant variant arXiv 2604.11224 (unverified). Dual use. |
| Committed-minority tipping | Ashery, Aiello & Baronchelli (arXiv 2410.08948) did the LLM-population version; Centola's memory model already gives the ~25 % threshold. |
| Cooperation on networks | Ezaki et al. 2016 (aspiration RL explains moody conditional cooperation); McKee et al. (arXiv 2103.04982) fit deep MARL to human data. |
| Public-goods decline and restart effect | Burton-Chellew & West 2021 already fit payoff learning to 237 games. |
| Repeated prisoner's dilemma meta-data | Fudenberg & Karreskog Rehbinder 2024 (AEJ Micro) predict cooperation out of sample with a simple learning model. |
| Unravelling in finitely repeated PD | Embrey, Fréchette & Yuksel 2018 already estimate learning models (unverified). |
| Kidney exchange / dynamic matching | Okada et al. 2026 (arXiv 2607.28925): GNN RL; greedy near-optimal. Registry data restricted. |
| Voting manipulation | Holliday, Kristoffersen & Pacuit (AAAI-25, arXiv 2401.16412); one-shot problem, RL decorative. |
| Sokoban planning interpretability | Bush et al. 2025 (arXiv 2504.01871); crowded niche, GPU training. |
| Lévy foraging | Muñoz-Gil et al. 2024 (arXiv 2303.06050); no shared data benchmark. |
| Recommender preference shift (as framed) | Carroll et al. 2022; Evans & Kasirzadeh; Krueger et al. 2020. Kept only via the producer blind spot. |
| Redistricting vs ensemble tests | RL adds nothing over annealing / short bursts (Cannon et al.); Ratliff, Somersille & Veomett 2024; remaining questions are sampler theory; ethics. |
| Causal emergence in RL agents | Pigozzi & Levin 2026 (arXiv 2605.06746). |
| Wisdom vs madness of crowds | Toyokawa & Gaissmaier 2022 already explain their data with RL plus conformity; Konstanz project ongoing. |
| Dropped for other reasons | Surge-pricing collusion (no benchmark, manipulation recipe); polarisation vs DW-NOMINATE (crowded, RL not essential); protest and repression (no benchmark); primaries / top-two (analytically solvable); RCV adoption (folded into Duverger); coalitions vs ParlGov (no portfolios; folded into legislators); rating inflation (proprietary); vaccine scares (imitation equations suffice); Jülich crowds (not social); tax compliance (no pooled data); cumulative culture and population size (no trial data, easy to manufacture); backfire effects (no learning loop); minimum-effort game (no pooled data); Thompson et al. 2022 sorting algorithms (data availability unconfirmed); epidemics (crowded, politically noisy); school choice (not searched in depth). |

## 5. Audit of the ChatGPT report

Source: [`../chatgpt-deep-research.md`](../chatgpt-deep-research.md). Its citations are tool markers (`cite…`), not references; nothing in it was taken as verified. Its own ranking: #1 temporal memory vs the information bound; #2 causal emergence under memory completion; #3 (conditional) eco-evolutionary dryland patterning; tipping diagnosis "do not prioritize".

### 5.1 #1 Temporal memory and the information bound — **HOLD**

- **Paper checked:** Romeo, Martin, Scandolo, Fruchart, Munro, Vitelli, arXiv 2511.01682 (v1 Nov 2025, v2 Feb 2026; unrefereed). The bound, the wave-pinning evasion and the "no memory" scope sentence were verified; the "intermediate-coupling optimum" and the "resources" claims only partly. Code (NicoRomeo/RobustPatterns, MIT) exists but has no figure-to-code mapping.
- **Trivially true.** The theorem constrains the shape of the fate profile. A local memory m_i (time-average then threshold) makes the profile sigmoidal, and PI → 1 bit as τ_m → ∞. This is Berg–Purcell / Gregor-style time averaging, which the authors' "no memory" caveat anticipates.
- **Conflicts with prior work.** ChatGPT's "fungible resource budget" contradicts Govern & ten Wolde 2014 (time, copies and energy are non-substitutable); storing samples also costs energy (Mehta & Schwab 2012).
- **Narrow residual question.** How much local memory keeps PI ≥ x across diffusivities? The agent's own back-of-envelope (unverified) suggests ξ_eff grows only logarithmically with memory, versus O(1) cost for global conservation. Kill criterion: if PI(τ_m) just follows Φ(√(τ_m/τ_c)), it is a footnote.
- **RL role:** decorative. Small exact search spaces suit exhaustive or gradient search.

### 5.2 #2 Causal emergence under memory completion — **HOLD**

- **Facts verified:** NIS+ (Yang et al., Natl Sci Rev 2024) is Markovian by its own statement and disagrees with Rosas et al.'s measure; the SVD theory (Zhang et al., npj Complexity 2025) is Markov-chain based and its code is not public.
- **Largely covered.** Rosas et al. 2024 (arXiv 2402.09090) define closure on the macro variable's full past and treat higher-order Markov lumpings (App. A.7); Geiger & Wu 2016 (arXiv 1608.04637) already pose the state-dimension vs history-depth trade-off with KL costs; Barnett & Seth 2023 build memory completion into dynamical dependence.
- **Residual is definitional.** Hoel-EI on history-augmented macrostates needs a do()-distribution over histories, and that convention likely sets the sign of the result (cf. Eberhardt & Lee 2022). The "minimal hidden-state realisation" is ill-posed. The owner deprioritises definitional contributions.
- **RL role:** decorative; the RL analogue (bisimulation, approximate information states) is already answered.

### 5.3 #3 Dryland eco-evolutionary patterning — **not pursued**

Dong, Richardson & Muneepeerakul, PNAS 123(42) e2536473123 (Oct 2026) exists (metadata and abstract verified; full text behind a challenge page). The paper already varies rainfall variability ("largely alleviated under the variable rainfall typical of drylands"), so the proposed lever is partly taken. No code found; no RL role; outside the owner's fields.

### 5.4 #4 Active tipping diagnosis

ChatGPT itself ranked it "do not prioritize" (crowded). Not audited.

## 6. Decision

The owner chose Model Citizens on 2026-10-20 (D31). It meets every hard requirement: RL is the designer, as in the original work; the anchor is public trial-level human data under mechanisms a model can be held out from; it sits at the meeting point of social cooperation, political mechanism design and RL discovery; and either outcome (robust or underdetermined policy) is publishable. It supersedes the Constitution Machine slot.

## 7. Open follow-ups from the scan

- Runners-up (bubbles, blind retrospection, learning Duverger) stay recorded here for a later study.
- Before any of them is revived: run `lit-scan`, re-verify every reference, and confirm data access (period-level repeated-market data; Dataverse guestbook; TSE portal).

## 8. References, as marked by each report

Not upgraded: a reference unverified in one report and verified in another (e.g. Burton-Chellew & West 2021) keeps each report's mark.

### r1-political

**Verified (fetched by agents):** Kollman, Miller & Page 1992, APSR 86(4), doi 10.2307/1964345 · Laver & Sergenti 2011, Princeton UP · Bendor, Diermeier, Siegel & Ting 2011, Princeton UP · Fujiwara 2011, QJPS 6(3–4), doi 10.1561/100.00010037 · Bordignon, Nannicini & Tabellini 2016, AER 106(8) · Reed 1990, BJPS 20(3), doi 10.1017/S0007123400005871 · Fey 1997, APSR 91(1) · Clough 2007, BJPS 37(2) · Forsythe, Rietz, Myerson & Weber 1996, IJGT 25(3) · Shugart & Taagepera 2017, *Votes from Seats*, CUP (chapter page; formula not displayed) · Tomlinson et al., arXiv 2402.17109 · Campbell & Ackland, arXiv 2605.09784 · Mukherjee, arXiv 2604.19985 · Liu & Lin, arXiv 2609.26568 · Boratyn & Stolicki, arXiv 2602.08160 · Baranski & Morton 2022, Exp. Econ. 25(4), doi 10.1007/s10683-021-09734-7 · Fréchette, Kagel & Lehrer 2003, APSR 97(2) · Fréchette, Kagel & Morelli 2005, GEB 51(2) · Fréchette 2009, J. Econometrics 153(2) · Ansolabehere, Snyder, Strauss & Ting 2005, AJPS 49(3) · Cox 2021, APSR 115(3) · Van Mulders, Bogaert & Van den Poel, arXiv 2607.15095 · Qian et al., arXiv 2509.09071 · Ratliff, Somersille & Veomett, arXiv 2409.17186 · DeFord, Duchin & Solomon, arXiv 1911.05725 · Cannon et al., arXiv 2011.02288 · Gold, arXiv 2606.01333 · Bendor, Diermeier & Ting 2003, APSR 97(2) · Fujiwara, Meng & Vogl 2016, AEJ: Applied 8(4) · Shikano 2018, ECPR paper · Brender & Drazen 2005, JME 52(7) · data portals: TSE open data, ParlGov, Eligendo.

**Unverified:** Laver 2005 APSR · Fowler & Laver 2008 JCR · Fowler 2006 JOP 68(2) · Cox 1997 *Making Votes Count* · Forsythe et al. 1993 SCW 10 · Bouton, Gallego, Llorente-Saguer & Morton 2022 EJ · Warwick & Druckman 2006 EJPR · Seki & Williams 2014 and dataset (doi 10.7910/DVN/0UNUAM, search only) · openICPSR 113627 (403) · Roth & Erev 1995 · Mattingly; Chen & Rodden · 2025 ABM of district magnitude and entry · seat-product formula N_S = (MS)^1/6 · CLEA (403).

### r1-social

**Verified (fetched by agents):** Lo Iacono et al. 2023, PNAS Nexus 2:pgad091 · Burton-Chellew & West 2021, Nat Hum Behav 5:1330, doi 10.1038/s41562-021-01107-7 · Vasconcelos, Dannenberg & Levin 2022, Nat Sustain 5:578 · Sasaki et al. 2012, PNAS 109:1165 · Rand & Nowak 2011, Nat Commun 2:434 · Bühren et al. 2023, PLoS ONE 18:e0282112 · Guzman Piedrahita et al. 2025, arXiv 2506.23276 · McKee et al., arXiv 2103.04982 · Hughes et al. 2018, arXiv 1803.08884 · Köster et al. 2022, PNAS 119:e2106028118 · Vinitsky et al., arXiv 2106.09012 · Herrmann, Thöni & Gächter dataset, Dryad doi 10.5061/dryad.87301 · Brady et al. 2021, Sci Adv 7:eabe5641 · Lindström et al. 2021, Nat Commun 12:1311 · Milli et al. 2025, PNAS Nexus 4:pgaf062 · Piccardi et al. 2025, arXiv 2411.14652 and Dryad package · Nyhan et al. 2023, Nature 620:137 · Huszár et al., arXiv 2110.11010 · Shen et al. 2026, arXiv 2608.29430 · Zheng & Zhao 2025, arXiv 2509.00199 · Larooij & Törnberg 2025, arXiv 2508.03385 · Carroll et al. 2022, arXiv 2204.11966 · Evans & Kasirzadeh, arXiv 2109.04083 · Krueger, Maharaj & Leike 2020, arXiv 2009.09153 · Chen et al., arXiv 1812.02353 · Ceylan, Anderson & Wood 2023, PNAS 120:e2216614120 · Wojcik et al. 2022, arXiv 2210.15723 · Truong et al. 2025, arXiv 2511.02615 · Selvam et al. 2026, arXiv 2607.01824 · Ashery, Aiello & Baronchelli, arXiv 2410.08948 · Ezaki et al. 2016, PLoS Comput Biol 12:e1005034 · Garg & Nazerzadeh, arXiv 1905.07544.

**Unverified:** Gürerk, Irlenbusch & Rockenbach 2006, Science · Gürerk et al. 2014, J Public Econ 120 · Herrmann, Thöni & Gächter 2008, Science 319 · Sigmund et al. 2010, Nature · Guess et al. 2023, Science · Centola et al. 2018, Science · Gracia-Lázaro et al. 2012, PNAS · Rand et al. 2011, PNAS · Brady et al. 2023, TICS 27:947 · 2026 Nat Commun model of reward learning and habits on social media · Embrey, Fréchette & Yuksel 2018, QJE 133 · Tripathy 2023 · arXiv 2604.11224 · Coppolillo et al., arXiv 2409.16478 · Horita et al. 2017 · Vinitsky et al., Collective Intelligence 2023.

### r1-algorithm

**Verified (fetched by agents):** Koster et al. 2022, *Human-centered mechanism design with Democratic AI*, arXiv 2201.11441 · Koster et al. 2025, Nat Commun 16:2824 (PMC11929920), arXiv 2404.15059 · Qin, Zou & Wang 2026, arXiv 2606.05867 · Zheng et al. 2021, AI Economist, arXiv 2108.02755 · Ling et al. 2019, Nat Commun 10:5174 · Ballerini et al. 2008, arXiv 0709.1916 · Young et al. 2013, arXiv 1302.3195 · Durve, Peruani & Celani 2020, arXiv 1911.01697 · Brambati et al., arXiv 2506.15587 · Li, Li & Zhao 2023, arXiv 2308.12624 · Muñoz-Gil et al. 2024, arXiv 2303.06050 · Oh et al. 2020 (LPG), arXiv 2007.08794 · Lu et al. 2022 (DPO), arXiv 2210.05639 · Jackson et al. 2024, arXiv 2402.05828 · Goldie et al. 2024 (OPEN), arXiv 2407.07082 · Foerster et al. (LOLA), arXiv 1709.04326 · Lu et al. 2022 (M-FOS), arXiv 2205.01447 · Khan et al. (Shaper), arXiv 2312.12568 · Duque et al., arXiv 2406.14662 · Meulemans et al. 2024, arXiv 2410.18636 · Weis et al. 2026, arXiv 2602.16301 · Li, Schultz, Hennes & Lanctot 2026, arXiv 2602.16928 · Barfuss & Meylahn 2023, arXiv 2209.01013 · Jang et al. 2024, arXiv 2402.17050 · Stern et al. 2018, arXiv 1705.01693 · Poudel, Li & Heaslip (EnduRL), arXiv 2311.12261 · Alhuraish, Wang & Zhou 2026, arXiv 2605.01485 · Okada et al. 2026, arXiv 2607.28925 · Holliday, Kristoffersen & Pacuit, arXiv 2401.16412 · Truong et al. 2025, arXiv 2511.02615 · Selvam et al. 2026, arXiv 2607.01824 · Bush et al. 2025, arXiv 2504.01871 · Ji-An, Benna & Mattar 2025, Nature (PMC12390849) · repositories google-deepmind/sustainable_behavior, deepmind/hcmd_dai, google-deepmind/disco_rl, ucl-dark/pax.

**Unverified:** Oh et al. 2025, Nature (DiscoRL), doi 10.1038/s41586-025-09761-x (login redirect; repository verified) · Akbarpour, Li & Oveis Gharan 2020, JPE (403) · Zheng et al. 2022, Sci Adv (AI Economist) · Goldie et al. 2025, arXiv 2507.17668 · Camperi et al. 2012 · Dickerson & Sandholm (FutureMatch); Anil & Bao 2021; Taufeeque et al. 2024; Castro et al. 2025 (CogFunSearch); Sugiyama et al. 2008; CIRCLES methodology arXiv 2404.15533 · access to Ling et al. figshare data; I-24 MOTION portal.

### r2-audit-chatgpt1-infobound

**Verified (fetched by agents):** Romeo et al., arXiv 2511.01682, and repo NicoRomeo/RobustPatterns · Koehler et al., arXiv 2601.05907 · Zhang, Mateu Hoyos, Brückner & Tkačik, arXiv 2512.20536 · Montero et al., arXiv 2605.14998 · Kremser, Vercelli & Gerland, bioRxiv 10.1101/2023.12.29.573602 · Berg & Purcell 1977, doi 10.1016/S0006-3495(77)85544-6 · Gregor et al. 2007, doi 10.1016/j.cell.2007.05.025 · Tostevin, ten Wolde & Howard 2007, doi 10.1371/journal.pcbi.0030078 · Dessaud et al. 2007, doi 10.1038/nature06347 · Erdmann, Howard & ten Wolde 2009, doi 10.1103/PhysRevLett.103.258101 · Mehta & Schwab 2012, doi 10.1073/pnas.1207814109 · Dubuis et al. 2013, doi 10.1073/pnas.1315642110 · Siggia & Vergassola 2013, doi 10.1073/pnas.1314081110 · Govern & ten Wolde 2014, doi 10.1073/pnas.1411524111 · Tkačik et al. 2015, doi 10.1534/genetics.114.171850 · Sokolowski & Tkačik 2015, doi 10.1103/PhysRevE.91.062710 · Mugler, Levchenko & Nemenman 2016, doi 10.1073/pnas.1509597112 · Desponds, Vergassola & Walczak 2020, doi 10.7554/eLife.49758 · Tkačik & Gregor 2021, doi 10.1242/dev.176065 · Lammers, Flamholz & Garcia 2023, doi 10.1073/pnas.2211203120 · Hellman & Cover 1970, doi 10.1214/aoms/1177696958 · Berg, Ordentlich & Shayevitz 2020, doi 10.1109/ISIT44484.2020.9174505 · Dong, Richardson & Muneepeerakul 2026, doi 10.1073/pnas.2536473123 (metadata and abstract only).

**Unverified:** Dong et al. model equations, rainfall autocorrelation/shuffle tests and code availability · the exact Hellman–Cover formula and scaling · Kremser et al. eLife version of record · whether Zhang et al. discuss temporal integration · the agent's own ξ_eff and τ_m scalings.

### r2-audit-chatgpt2-causal-emergence

**Verified (fetched by agents):** Yang et al., Natl Sci Rev 12(1):nwae279 (PMC11697982) and NIS+ code repo · Zhang, Tao, Leong, Yang & Yuan, npj Complexity 2:3 (2025) · Rosas et al., arXiv 2402.09090 · Barnett & Seth 2023, PRE 108:014304, arXiv 2106.06511 · Hoel, Albantakis & Tononi 2013, PNAS 110:19790 · Hoel, arXiv 1612.09592 · Hoel, arXiv 2503.13395 · Comolatti & Hoel, arXiv 2202.01854 · Rosas, Mediano et al. 2020, PLoS Comput Biol 16:e1008289 · Shalizi & Moore, arXiv cond-mat/0303625 (Found Phys 55:2) · Geiger & Temmel, arXiv 1212.4375 · Geiger & Wu, arXiv 1608.04637 · Amjad, Blöchl & Geiger, arXiv 1709.05907 · Pfante et al. 2014, Adv Complex Syst 17(2) (abstract only) · Yuan, Zhang et al. 2024, Entropy 26(2):108 · Eberhardt & Lee 2022, Philosophies 7(2):30 · Marshall, Albantakis & Tononi 2018, PLoS Comput Biol 14:e1006114 · Subramanian et al. 2022, JMLR 23 · Pigozzi & Levin, arXiv 2605.06746 · Sartore, Teichmann & Stock, arXiv 2412.08660.

**Unverified:** Bertschinger, Olbrich, Ay & Jost 2006 · Varley 2020, arXiv 2003.13075 · Görnerup & Jacobi 2008 · Givan, Dean & Greig 2003 · Li, Walsh & Littman 2006 · Hoel et al. 2016, Neurosci Conscious · Jansma & Hoel, arXiv 2510.02649 · Huang group qMSM/IGME papers · Entropy 28(8):872 (2026, 403) · "ε-machine of a lumped chain is typically infinite-state" (background knowledge).

### r2-rl-disputed-regularities

**Verified (fetched by agents):** Huber, Hill & Lenz 2012, APSR 106(4); dataset doi 10.60600/YU/DCNMXL · Graham, Huber, Malhotra & Mo 2021, JOP, doi 10.1086/714761; dataset doi 10.60600/YU/1K9F4T · Fowler & Montagnes 2015, PNAS 112(45) · Healy, Malhotra & Mo 2015, PNAS 112(48) (letter) · Fowler & Hall 2018, JOP 80(4) · Ashworth, Bueno de Mesquita & Friedenberg 2018, AJPS 62(1) · Woon 2012, AJPS 56(4) · Huber et al. 2024, I4R DP 190; OSF uxrgk / 5ypku / kw8mn · Füllbrunn, Huber, Eckel & Weitzel 2024, JFQA 59(3) · Dufwenberg, Lindqvist & Moore 2005, AER 95(5) · Hussam, Porter & Smith 2008, AER 98(3) · Kirchler, Huber & Stöckl 2012, AER 102(2) · Kopányi-Peuker & Weber 2021, RFS 34(9) · Henning et al. 2025, arXiv 2502.15800 · van Opheusden et al. 2023, Nature 618; OSF n2xjm · Lin, Lake & Ma 2024, CCN · Connors, Burns & Campitelli 2011, Cognitive Science 35(8) · Jones 2021, arXiv 2104.03113 · Fudenberg & Karreskog Rehbinder 2024, AEJ Micro 16(1) · Toyokawa & Gaissmaier 2022, eLife 11:e75308 and GitHub data.

**Unverified:** Achen & Bartels 2016 · Healy, Malhotra & Mo 2010, PNAS · Esponda & Pouzo 2017, AEJ Micro 9(2) · Hong, Moinas & Pouget 2021, JEBO 185 · Duffy & Ünver 2006 · Gufler, Sangiorgi & Tarantino 2025 · Smith, Suchanek & Williams 1988 · Chabris & Hearst 2003 · Toyokawa, Whalen & Laland 2019 · Thompson, van Opheusden, Sumers & Griffiths 2022 · Cook, Lu, Hughes, Leibo & Foerster 2024 · Derex et al. 2013; Vaesen et al. 2016; Fay et al. 2019 · Burton-Chellew & West 2021 · Le Mens et al. 2023 · Galke & Raviv 2024 · Wu et al. 2025, Nat Commun · Denrell & March 2001.

### spec-koster2025-game-data

**Read and checked against sources:** Koster et al. 2025 full text (PMC11929920) and supplementary PDF; `google-deepmind/sustainable_behavior` README and notebook; `sustainable_behavior.csv` (HEAD plus byte-range reads); Koster et al. 2022 arXiv 2201.11441 (pp. 1–10) and the `hcmd_dai` repository and notebook. Details are in the Model Citizens spec.
