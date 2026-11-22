# The Last Library — Confirmatory Study Report

- **Study ID:** `last-library`
- **Question:** Q1 (bit-budgeted abstract rule vs. pedagogical example inheritance under stationary vs. unannounced latent-rule changes, D38) and Q3 (multi-generation cumulative improvement)
- **Evidence run:** `runs/library-main-v1` (published at `web/public/research/last-library/main-v1/`)
- **Status:** `validated` (all 7 automated checks passed; `source.git_dirty = false`)
- **Code commit:** [`3ed302d7`](../../../) (`3ed302d71ef9270a1a2edda07dd1c49c246bc118`)
- **Protocol:** [`protocol.md`](protocol.md) v1 (**frozen**, 2026-11-22, D39; `config_sha256 = 6ef5f432eeb3cfac79d5303d5fba2ca030b6fb5c984333bd6392da1f66c679ee`)
- **Artifact checksums:**
  - `results.json` (277 KB): `db7805447c311830005ddccaaeb4ac6bcf47e45f19747aed6366cd4072bcc55c`
  - `samples.json` (1.0 MB): `02dbba8b0c5b77e8c5be722a5a60e0e9539ef655e2c9e67df489bbdeed168426`

---

## 1. Executive Summary

At exact inherited channel budgets swept across the 16-bit latent entropy of our 8-terrain $\times$ 4-level Voronoi gridworld (`0..60` bits, $N = 200$ stationary seeds `300_000..300_199` and $N = 200$ rule-change seeds `400_000..400_199`), **whether abstract rule inheritance outperforms bit-matched pedagogical examples—and whether it creates harmful lock-in—depends sharply on a capacity threshold at 19 bits and on whether receivers hedge against environmental shifts:**

1. **Bit-capacity phase transition (`H1`, 1/4 contrasts passed):** Below the 19-bit dense rule capacity (`5^8 = 390,625 < 2^19`), sparse `rule` encoding loses to or ties `example_pedagogical` (`B = 10`: `+0.108` regret difference, `p_holm = 0.934`; `B = 15`: `+0.629`, `p_holm = 0.645` vs. $\delta = 0.50$). Right above threshold at **`B = 20` bits**, `rule` drops held-out generation-9 regret from `3.92` (`B = 15`) to **`0.62`**, beating `example_pedagogical@20` (`3.74`) by **`+3.122 [+2.456, +3.838]`** (`t = 7.468`, `p_holm = 5.01e-12`, `d_z = 0.629`). At `B = 40` bits (8 concrete examples), `example_pedagogical` narrows the gap (`1.46` vs. `0.62`, diff `+0.840`, `p_raw = 0.0368`, `p_holm = 0.1103`).
2. **Cumulative ratchet only for dense rules (`H4`, 2/3 contrasts passed):** Across generations `1 → 9` under stationary rules, `unlimited` (`3.14 → 0.11`, drop `+3.031`, `p_holm = 1.02e-15`; exact match to `single_lifetime`) and `rule@20` (`3.15 → 0.62`, drop `+2.530`, `p_holm = 1.03e-9`) show sustained cumulative improvement. By contrast, `example_pedagogical@20` plateaus immediately after generation 1 (`3.48 → 3.74`, diff `-0.256`, `p_holm = 0.950`).
3. **Unannounced rule-change lock-in (`H2a` 1/2, `H2b` 2/2):** At generation `5`, when every terrain mutates to a new cost level, lossless `unlimited` suffers catastrophic lock-in (`31.97` regret at gen 5; still `15.63` at gen 9 vs. `7.72` for fresh learners `none`). At `B = 20`, `rule` spikes to `15.42`, worse than `example_pedagogical@20` (`11.30`, diff `+4.112 [+2.978, +5.252]`, `p_holm = 3.64e-9`) and much worse than `none` (`7.91`, diff `+7.509 [+6.200, +8.826]`, `p_holm = 2.10e-20`). At `B = 40`, however, `example_pedagogical` also accumulates stubborn priors (`14.20`, `H2a_b40` diff `+1.217`, `p_holm = 0.0802`), showing that high-capacity examples converge toward dense rules in both efficiency and brittleness.
4. **Sender confidence fails; receiver hedging succeeds (`H3a` 0/1, `H3b` 3/4):** Encoding graded confidence (`rule_confidence@40`, high-confidence trust `0.92`) *slightly worsens* gen-5 shock (`15.68` vs. `15.42`, diff `-0.265`, `p_holm = 0.991` — pre-registered negative result), because higher certainty makes overriding stale beliefs harder. Conversely, receiver hazard hedging (`rule_hedged`, blending `0.30` uniform toward unknown priors at decode time) cuts gen-5 overshoot by **`+2.552 [+1.592, +3.508]`** (`12.86` vs. `15.42`, `p_holm = 7.14e-5` at `B = 20, 40`) while keeping a strong stationary edge over `example_pedagogical` at `B = 20` (`1.40` vs. `3.74`, diff `+2.339`, `p_holm = 2.81e-7`) and tying it at `B = 40` (`1.40` vs. `1.46`, diff `+0.057`, `p_holm = 0.980`).

---

## 2. Pre-Registered Hypothesis Verdicts

Paired one-sided Student $t$-tests against minimum meaningful difference $\delta = 0.50$ regret units ($\mathrm{df} = 199$), Holm–Bonferroni family-wise correction at $\alpha = 0.05$, and two-sided 95% paired percentile bootstrap confidence intervals ($10{,}000$ resamples, `bootstrap_seed = 9201`):

| Family | Contrast ID | Comparison (regret difference $X - Y$) | Mean diff | 95% Bootstrap CI | $t$-stat | $p_{\text{raw}}$ | $p_{\text{holm}}$ | $d_z$ | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **H1** | `H1_b10` | Stat gen 9: `example_pedagogical@10 − rule@10` | `+0.1077` | `[-0.3953, +0.6245]` | `-1.509` | `9.336e-01` | `9.336e-01` | `0.029` | FAIL |
| **H1** | `H1_b15` | Stat gen 9: `example_pedagogical@15 − rule@15` | `+0.6292` | `[+0.1178, +1.1966]` | `+0.462` | `3.224e-01` | `6.448e-01` | `0.159` | FAIL |
| **H1** | `H1_b20` | Stat gen 9: `example_pedagogical@20 − rule@20` | `+3.1221` | `[+2.4556, +3.8385]` | `+7.468` | `1.253e-12` | `5.011e-12` | `0.629` | **PASS** |
| **H1** | `H1_b40` | Stat gen 9: `example_pedagogical@40 − rule@40` | `+0.8399` | `[+0.4666, +1.2024]` | `+1.799` | `3.677e-02` | `1.103e-01` | `0.314` | FAIL |
| **H2a** | `H2a_b20` | Change gen 5: `rule@20 − example_pedagogical@20` | `+4.1116` | `[+2.9778, +5.2517]` | `+6.176` | `1.820e-09` | `3.641e-09` | `0.497` | **PASS** |
| **H2a** | `H2a_b40` | Change gen 5: `rule@40 − example_pedagogical@40` | `+1.2170` | `[+0.2271, +2.2081]` | `+1.409` | `8.021e-02` | `8.021e-02` | `0.169` | FAIL |
| **H2b** | `H2b_b20` | Change gen 5: `rule@20 − none` | `+7.5087` | `[+6.1996, +8.8255]` | `+10.365` | `1.049e-20` | `2.098e-20` | `0.785` | **PASS** |
| **H2b** | `H2b_b40` | Change gen 5: `rule@40 − none` | `+7.5087` | `[+6.1996, +8.8255]` | `+10.365` | `1.049e-20` | `2.098e-20` | `0.785` | **PASS** |
| **H3a** | `H3a_b40` | Change gen 5: `rule@40 − rule_confidence@40` | `-0.2652` | `[-0.9162, +0.3719]` | `-2.372` | `9.907e-01` | `9.907e-01` | `-0.058` | FAIL |
| **H3b** | `H3b_change_b20` | Change gen 5: `rule@20 − rule_hedged@20` | `+2.5516` | `[+1.5917, +3.5076]` | `+4.159` | `2.379e-05` | `7.137e-05` | `0.366` | **PASS** |
| **H3b** | `H3b_change_b40` | Change gen 5: `rule@40 − rule_hedged@40` | `+2.5516` | `[+1.5917, +3.5076]` | `+4.159` | `2.379e-05` | `7.137e-05` | `0.366` | **PASS** |
| **H3b** | `H3b_stationary_b20` | Stat gen 9: `example_pedagogical@20 − rule_hedged@20` | `+2.3390` | `[+1.7028, +3.0149]` | `+5.460` | `7.034e-08` | `2.814e-07` | `0.491` | **PASS** |
| **H3b** | `H3b_stationary_b40` | Stat gen 9: `example_pedagogical@40 − rule_hedged@40` | `+0.0568` | `[-0.3821, +0.4504]` | `-2.066` | `9.800e-01` | `9.800e-01` | `0.019` | FAIL |
| **H4** | `H4_unlimited` | Stat `unlimited` gen 1 − gen 9 | `+3.0307` | `[+2.4857, +3.6051]` | `+8.795` | `3.404e-16` | `1.021e-15` | `0.745` | **PASS** |
| **H4** | `H4_rule_b20` | Stat `rule@20` gen 1 − gen 9 | `+2.5301` | `[+1.8943, +3.1286]` | `+6.411` | `5.150e-10` | `1.030e-09` | `0.565` | **PASS** |
| **H4** | `H4_example_pedagogical_b20` | Stat `example_pedagogical@20` gen 1 − gen 9 | `-0.2560` | `[-1.1437, +0.6428]` | `-1.653` | `9.500e-01` | `9.500e-01` | `-0.040` | FAIL |

---

## 3. Budget-Sweep Curves and Generational Dynamics

### Fixed Baselines (Mean ± SE Held-Out Regret, $N = 200$)

- `oracle`: `0.00 ± 0.00` throughout.
- `none` ($B = 0$): stationary gen 9 `8.32 ± 0.50`; rule-change gen 5 `7.91 ± 0.44`; rule-change gen 9 `7.72 ± 0.45`.
- `unlimited` & `single_lifetime`: stationary gen 9 `0.11 ± 0.03`; rule-change gen 5 `31.97 ± 1.08`; rule-change gen 9 `15.63 ± 0.72` (exact numerical parity, max diff `0.0 <= 1e-12`).

### Swept Arms Across Exact Bit Budgets (Stationary Gen 9 / Rule-Change Gen 5 / Rule-Change Gen 9)

| Arm | $B=0$ | $B=5$ | $B=10$ | $B=15$ | $B=20$ | $B=25$ | $B=30$ | $B=40$ | $B=60$ |
|---|---|---|---|---|---|---|---|---|---|
| `example_random` | 8.32 / 7.91 / 7.72 | 7.86 / 8.50 / 6.93 | 7.42 / 8.45 / 6.21 | 6.68 / 9.24 / 5.81 | 6.06 / 10.07 / 5.80 | 6.22 / 10.74 / 5.62 | 6.02 / 11.04 / 5.50 | 5.49 / 10.89 / 5.01 | 4.48 / 12.94 / 3.87 |
| `example_pedagogical` | 8.32 / 7.91 / 7.72 | 6.94 / 9.12 / 5.74 | 5.60 / 10.07 / 4.64 | 4.55 / 10.68 / 4.05 | **3.74 / 11.30 / 3.16** | 2.93 / 12.09 / 2.63 | 2.09 / 12.86 / 2.19 | **1.46 / 14.20 / 2.00** | 0.68 / 15.93 / 1.52 |
| `rule` | 8.32 / 7.91 / 7.72 | 6.56 / 9.50 / 6.50 | 5.50 / 10.58 / 5.27 | 3.92 / 11.74 / 4.15 | **0.62 / 15.42 / 2.60** | 0.62 / 15.42 / 2.60 | 0.62 / 15.42 / 2.60 | 0.62 / 15.42 / 2.60 | 0.62 / 15.42 / 2.60 |
| `rule_confidence` | 8.32 / 7.91 / 7.72 | 8.32 / 7.91 / 7.72 | 6.52 / 9.45 / 6.36 | 5.29 / 10.76 / 5.03 | 5.29 / 10.76 / 5.03 | 4.07 / 11.97 / 4.31 | 2.98 / 12.69 / 3.30 | **0.41 / 15.68 / 2.49** | 0.41 / 15.68 / 2.49 |
| `rule_hedged` | 8.32 / 7.91 / 7.72 | 6.96 / 8.55 / 6.36 | 5.76 / 9.23 / 5.19 | 5.09 / 9.77 / 4.45 | **1.40 / 12.86 / 2.19** | 1.40 / 12.86 / 2.19 | 1.40 / 12.86 / 2.19 | 1.40 / 12.86 / 2.19 | 1.40 / 12.86 / 2.19 |
| `placebo_random` | 8.32 / 7.91 / 7.72 | 9.13 / 8.29 / 7.91 | 8.87 / 8.96 / 8.91 | 9.46 / 9.43 / 9.14 | 11.31 / 11.15 / 11.14 | 11.31 / 11.15 / 11.14 | 11.31 / 11.15 / 11.14 | 11.31 / 11.15 / 11.14 | 11.31 / 11.15 / 11.14 |
| `placebo_shuffled` | 8.32 / 7.91 / 7.72 | 8.79 / 8.55 / 8.60 | 9.20 / 9.63 / 9.42 | 9.85 / 10.36 / 9.96 | 11.83 / 12.79 / 11.89 | 11.83 / 12.79 / 11.89 | 11.83 / 12.79 / 11.89 | 11.83 / 12.79 / 11.89 | 11.83 / 12.79 / 11.89 |

Two structural insights emerge cleanly from the sweeps:
1. **Why 19–20 bits is the inflection point:** Five-symbol ternary-packed `rule` requires 19 bits (`5^8`) to pass MAP estimates for all 8 terrains simultaneously. At `B = 10..15`, each generation only communicates the top 2–3 lowest-entropy terrains and forgets the rest, leaving it statistically indistinguishable from sparse pedagogical examples. Once `B >= 20`, `rule` transmits the full library across generations; `rule_confidence` pays 32 bits (`8 * 4`) to encode confidence bins, explaining why `rule_confidence@20` stalls at sparse `2`-terrain capacity (`5.29`) until `B >= 32`.
2. **Sender confidence vs. receiver skepticism:** Sender calibration (`rule_confidence`) assumes environments stay stationary; knowing a rule is `92%` certain makes a sender more persuasive, which actively harms a successor facing an unannounced world shift. Only receiver-side hazard discounting (`rule_hedged`, `h = 0.30`) decouples stationary belief accumulation from successor adaptability—keeping enough guidance to beat `example_pedagogical@20` on stationary worlds (`1.40` vs. `3.74`) while shedding stale priors fast enough to recover to `2.19` after a total rule reversal.

---

## 4. Positioning Against Verified Literature & Limitations

- **Nearest neighbours (`literature.md`):** Prystawski, Arumugam & Goodman (CogSci 2023, `doi:10.31234/osf.io/q4tz8`; 2025, `arXiv:2511.18220`) established rate-distortion thresholds on fixed single-chain tasks; Cook et al. (NeurIPS 2024, `doi:10.52202/079017-1907`) supplied the single-lifetime control; Kirby, Cornish & Smith (PNAS 2008, `doi:10.1073/pnas.0707835105`) and Griffiths & Kalish (CogSci 2007, `doi:10.1080/15326900701326576`) motivated compressibility vs. prior-drift placebo checks; Ammar, Fogarty & Kandler (PLoS Comp. Biol. 2025, `doi:10.1371/journal.pcbi.1013785`) predicted negative transfer from vertical transmission after fast environmental change. Our confirmatory findings extend this literature by pitting exact bit-matched abstract rules against pedagogical examples under both held-out layout transfer and unannounced rule change, and by separating sender confidence calibration (harmful under change) from receiver hazard hedging.
- **Limitations:** Fixed `9 × 9` Voronoi gridworld (`noise_beta = 1.0`, 2 training episodes/generation), fixed pre-registered `hedge_discount = 0.30` (not meta-learned online), and total derangement mutation at `change_generation = 5` rather than gradual drift.
