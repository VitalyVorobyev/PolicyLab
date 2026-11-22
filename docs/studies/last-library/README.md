# The Last Library — Active Research Specification (v1, D38)

**Theme:** Cultural evolution, iterated learning, compressed memory, generalization, negative transfer.  
**Status:** Stage L1 started 2026-11-22 (domain implementation + pre-protocol feasibility gate under `docs/decisions.md` D38).

## Sharpened research question (Q1, D38)

After the focused literature scan (`literature.md`), **Prystawski, Arumugam & Goodman (2023, 2025)** were identified as the nearest prior work rather than Cook et al. (2024). Because 256 bytes (2,048 bits) would not bind on a compact latent gridworld rule, the study sweeps exact channel capacity in **bits** around the latent-rule entropy ($H = 16$ bits):

> At a fixed inherited budget of $B$ bits ($B \in \{0, 5, 10, 15, 20, 25, 30, 40, 60\}$), does a rule-encoded record outperform a bit-matched, pedagogically selected example record on **held-out layouts** while the latent rule family is stationary—yet underperform both example records and a `none` (no-inheritance) baseline in the first generations after an **unannounced change** of the latent rule, and does spending part of the budget on calibrated confidence remove that negative-transfer crossover?

**Secondary analysis (Q3):** Does held-out performance under a binding bit budget continue improving beyond generation 2 compared with a single learner given matched cumulative experience (`single_lifetime`), distinguishing multi-generation accumulation from one-shot transfer?

<details><summary>Superseded v0 specification summary (256-byte framing)</summary>

> When does transmitting a strictly limited external memory (at most 256 bytes) between otherwise fresh agents enable cumulative improvement on a family of previously unseen but structurally related tasks? (Ablations 0 / 64 / 256 bytes retired in D38 as non-binding for a 16-bit latent terrain rule.)

</details>

## Domain & world (`src/phasespace/library/`)

- **9 × 9 Voronoi terrain navigation:** Start at $(0, 0)$, goal at $(8, 8)$ (neutral cells with entry cost 1.0 and no observation). Remaining cells take one of $K = 8$ terrain types allocated across 8 Voronoi patches spanning 4 active terrain types per layout.
- **Noisy cost observations:** Each terrain type has an unknown true discrete cost level $l \in \{0, 1, 2, 3\}$ mapping to step costs $(1.0, 3.0, 7.0, 16.0)$ (plus observation noise parameter $\beta = 1.0$: stepping on a tile yields noisy integer observation $y \in \{0, 1, 2, 3\}$ with probability proportional to $\exp(-\beta |l - y|)$). Thus $H = 8 \times \log_2 4 = 16$ bits.
- **Exact Bayesian Thompson learner:** Fresh agents begin from a uniform prior or the decoded predecessor record, run 2 training episodes per generation planning via exact 4-connected Dijkstra on Thompson-sampled terrain levels, update posteriors via exact Bayes rule, and are evaluated on 10 held-out layouts per generation under posterior expected costs.
- **Unannounced rule change:** In change lineages, starting at generation 5 (0-indexed), the latent rule switches via a strict derangement without warning to the successor or changing the inherited record from generation 4.

Note on Stage L1 pilot (`library-feasibility-dev1`): an initial 3-episode, $\beta = 1.4$ design passed F2 and F3 but missed F1 (`none_gen0 = 4.28 < 5.0`) because single lifetimes learned too much of the 8-terrain vector at once. Shortening lifetimes to 2 episodes and lowering $\beta$ to $1.0$ raises the within-generation challenge before protocol freeze.

## Experimental arms (`simulate.ARMS`)

1. `oracle` — plans with true latent expected terrain costs (regret 0 reference).
2. `none` — fresh uniform prior every generation (individual-learning baseline).
3. `unlimited` — passes exact full floating-point posterior across generations (unbounded channel reference).
4. `single_lifetime` — carries exact posterior continuously over cumulative lifetimes ($G \times E$ experience baseline, Cook et al. 2024).
5. `example_random` — packs $m = \lfloor B/5 \rfloor$ randomly subsampled `(terrain, y)` observations at 5 bits/observation.
6. `example_pedagogical` — greedily selects up to $\lfloor B/5 \rfloor$ observations minimizing receiver-to-sender KL divergence.
7. `rule` — compresses MAP terrain rules (`rule_trust = 0.88`): dense base-5 encoding ($19$ bits for all 8 terrains when $B \ge 19$) or 5-bit `(terrain, level)` tuples ordered by lowest entropy when $B < 19$.
8. `rule_confidence` — encodes `(terrain, level, confidence_bin)` (dense $32$ bits when $B \ge 32$, else 7-bit entries) mapped to calibrated trust levels `(0.25, 0.55, 0.78, 0.92)`.
9. `placebo_random` — uniformly random valid bit payload in the `rule` format (prior/format control).
10. `placebo_shuffled` — `rule` record encoded from an independent, unrelated lineage at matching generation depth.

## Feasibility gate (pre-protocol, `feasibility.py`)

Before freezing `protocol.md` and running a full statistical comparison:

- **F1:** `none` leaves a non-trivial held-out regret gap vs. `oracle` ($\ge 5.0$ expected cost units at generation 0).
- **F2:** `unlimited` continues improving on held-out layouts strictly after generation 2 ($\text{regret}_{\text{gen 1}} - \text{regret}_{\text{gen 9}} \ge 1.0$).
- **F3:** Neither `placebo_random` nor `placebo_shuffled` ($B = 20$ bits) beats `none` beyond Monte Carlo margin ($0.5$ cost units).

## Interactive page (later, after M1/protocol runs)

Will publish validated runs showing generation lineage curves, bit-capacity vs. regret curves, human-readable decoded bit-record contents (`summary`), and the stationary vs. rule-change crossover switch.
