# The Last Library: Confirmatory Comparison Protocol

- **Protocol version:** v1
- **Status:** **frozen** (2026-11-22, D39)
- **Study ID:** `last-library`
- **Config hash (`config_sha256`):** `6ef5f432eeb3cfac79d5303d5fba2ca030b6fb5c984333bd6392da1f66c679ee`
- **Inputs:** [`README.md`](README.md), [`literature.md`](literature.md), decisions D15 & D38–D39 ([`decisions.md`](../../decisions.md)), passing Stage L1 feasibility run `runs/library-feasibility-dev2` (`status = validated`).

---

## 1. Research Question and Arms

Primary question (**Q1**, D38) and secondary cumulative-learning check (**Q3**):
at exact inherited bit budgets swept around the latent-rule entropy ($H = 16$ bits for $K = 8$ terrain types across $L = 4$ noisy cost levels):

1. Does explicit abstract `rule` inheritance beat bit-matched pedagogically selected concrete examples (`example_pedagogical`) under stationary rules?
2. Does `rule` suffer higher held-out regret than both `example_pedagogical` and `none` (from-scratch learning) on generation $g = 5$ immediately after an unannounced change in active terrain costs?
3. Does **H3a** (sender confidence calibration, `rule_confidence`) or **H3b** (receiver hazard hedging, `rule_hedged` with `hedge_discount = 0.30`) reduce that post-change regret spike while preserving stationary advantage over examples?

### Arms Evaluated (11 total)

| Arm | Budget dependence | Sender output / transmission | Receiver prior |
|---|---|---|---|
| `oracle` | — | — | Exact true terrain costs (lower bound = 0 regret) |
| `none` | $B=0$ | Empty record | Uniform $U(1/4)$ every generation |
| `unlimited` | — | Uncompressed full posterior belief | Exact previous posterior (upper bound) |
| `single_lifetime` | — | Uncompressed posterior across matched episodes | Exact previous posterior (equivalence control to `unlimited` at gen 9) |
| `example_random` | $B \in \mathcal{B}$ | $\lfloor B / 5 \rfloor$ uniformly sampled `(terrain, observed_level)` pairs | Bayes update from uniform over received pairs |
| `example_pedagogical` | $B \in \mathcal{B}$ | Greedy KL-minimizing $\lfloor B / 5 \rfloor$ pairs from pool toward sender belief | Bayes update from uniform over received pairs |
| `rule` | $B \in \mathcal{B}$ | Dense ternary-packed ($\ge 19$ bits) or greedy lowest-entropy sparse ($5$ bits/entry) MAP levels | Fixed `rule_trust = 0.88` on stated terrains, uniform elsewhere |
| `rule_confidence` | $B \in \mathcal{B}$ | Dense nibble-packed ($\ge 32$ bits) or sparse ($7$ bits/entry) `(level, confidence_bin)` | Bin-dependent `confidence_trusts = (0.25, 0.55, 0.78, 0.92)` |
| `rule_hedged` | $B \in \mathcal{B}$ | Identical payload to `rule` | Receiver blends `(1 - h) * rule_prior + h * U(1/4)` (`h = 0.30`) |
| `placebo_random` | $B \in \mathcal{B}$ | Random bit payload | Same decoder as `rule` |
| `placebo_shuffled` | $B \in \mathcal{B}$ | Same bit encoder on an independent unrelated seed rule | Same decoder as `rule` |

---

## 2. Frozen Design Parameters and Budgets

All domain and channel parameters are frozen in `src/phasespace/library/config.py` (`config_sha256 = 6ef5f432eeb3cfac79d5303d5fba2ca030b6fb5c984333bd6392da1f66c679ee`):

- Grid: `9 × 9` deterministic Dijkstra shortest-path navigation from `(0, 0)` to `(8, 8)` with `8` Voronoi patches assigning `4` active terrains per layout.
- Latent rule: `8` terrain types, `4` cost levels `(1.0, 3.0, 7.0, 16.0)`, observation softmax noise `noise_beta = 1.0`.
- Horizon: `n_generations = 10` (`g = 0..9`), `episodes_per_generation = 2` Thompson-sampling training episodes per generation, `heldout_layouts_per_gen = 10` held-out layouts per generation.
- Unannounced rule change: active in `"rule_change"` lineages at `change_generation = 5` (reassigns all `8` terrain cost levels via `mutate_rule`, ensuring no active pre-change level persists trivially).
- Rule trusts & hedging: `rule_trust = 0.88`, `hedge_discount = 0.30`, `confidence_trusts = (0.25, 0.55, 0.78, 0.92)`, `confidence_thresholds = (0.35, 0.65, 0.85)`.
- Swept bit budgets: `bit_budgets = (0, 5, 10, 15, 20, 25, 30, 40, 60)`. Budget-invariant baselines (`oracle`, `none`, `unlimited`, `single_lifetime`) run once at nominal `budget_bits = 0`.
- Paired lineage sample size (disjoint from feasibility `20261122..` and diagnostic query `999..` seeds):
  - **Stationary scenario:** $N = 200$ lineages, seeds `300_000..300_199` (common random numbers across all arms and budgets).
  - **Rule-change scenario:** $N = 200$ lineages, seeds `400_000..400_199` (common random numbers across all arms and budgets).
- Uncertainty quantification: paired one-sample Student $t$-test against minimum meaningful margin $\delta = 0.50$ regret units (`alternative="greater"`, $\mathrm{df} = N - 1 = 199$), family-wise error control via Holm–Bonferroni ($\alpha = 0.05$) within each pre-registered hypothesis family, and two-sided 95% paired percentile bootstrap CIs from $10{,}000$ shared index resamples drawn with `bootstrap_seed = 9201`.

---

## 3. Pre-Registered Confirmatory Hypothesis Families

For each contrast `d_i = X_i - Y_i` (where `X` is hypothesized to have *higher* held-out regret than `Y`), we test $H_0: \mathbb{E}[d] \le \delta$ vs $H_1: \mathbb{E}[d] > \delta$ with $\delta = 0.50$ and reject when Holm-adjusted $p < 0.05$:

### Family H1 — Stationary Efficiency Advantage (4 contrasts)
At stationary final generation $g = 9$, `rule` achieves lower held-out regret than bit-matched `example_pedagogical` by $> \delta$:
- `B ∈ {10, 15, 20, 40}`: $d_i(B) = R^{\text{stat}}_{g=9}(\text{example\_pedagogical}, B)_i - R^{\text{stat}}_{g=9}(\text{rule}, B)_i$.

### Family H2a — Rule-Change Rigid-Rule Vulnerability vs Examples (2 contrasts)
On the first post-change generation $g = 5$, `rule` suffers higher held-out regret than `example_pedagogical` by $> \delta$:
- `B ∈ {20, 40}`: $d_i(B) = R^{\text{change}}_{g=5}(\text{rule}, B)_i - R^{\text{change}}_{g=5}(\text{example\_pedagogical}, B)_i$.

### Family H2b — Rule-Change Rigid-Rule Vulnerability vs No Inheritance (2 contrasts)
On the first post-change generation $g = 5$, `rule` suffers higher held-out regret than `none` ($B = 0$) by $> \delta$:
- `B ∈ {20, 40}`: $d_i(B) = R^{\text{change}}_{g=5}(\text{rule}, B)_i - R^{\text{change}}_{g=5}(\text{none}, 0)_i$.

### Family H3a — Sender Confidence Calibration Intervention (1 contrast)
On the first post-change generation $g = 5$ at dense confidence capacity ($B = 40$), `rule_confidence` reduces post-change regret relative to `rule` by $> \delta$:
- `B = 40`: $d_i = R^{\text{change}}_{g=5}(\text{rule}, 40)_i - R^{\text{change}}_{g=5}(\text{rule\_confidence}, 40)_i$.
*(Pre-registered expectation from Stage L1 feasibility diagnostics: null / negative result if high-confidence weights reinforce stale beliefs.)*

### Family H3b — Receiver Hazard Hedging Intervention & Stationary Retention (4 contrasts)
Receiver hedging (`rule_hedged`, `hedge_discount = 0.30`) BOTH reduces post-change regret relative to `rule` at $g = 5$ AND preserves stationary advantage over `example_pedagogical` at $g = 9$:
- Change reduction (`B ∈ {20, 40}`): $d_i(B) = R^{\text{change}}_{g=5}(\text{rule}, B)_i - R^{\text{change}}_{g=5}(\text{rule\_hedged}, B)_i$.
- Stationary retention (`B ∈ {20, 40}`): $d_i(B) = R^{\text{stat}}_{g=9}(\text{example\_pedagogical}, B)_i - R^{\text{stat}}_{g=9}(\text{rule\_hedged}, B)_i$.

### Family H4 — Cumulative Multi-Generation Learning (Q3 secondary analysis, 3 contrasts)
Under stationary rules, held-out regret improves from generation $g = 1$ to final generation $g = 9$ by $> \delta$:
- `unlimited`: $R^{\text{stat}}_{g=1}(\text{unlimited})_i - R^{\text{stat}}_{g=9}(\text{unlimited})_i$.
- `rule` ($B = 20$): $R^{\text{stat}}_{g=1}(\text{rule}, 20)_i - R^{\text{stat}}_{g=9}(\text{rule}, 20)_i$.
- `example_pedagogical` ($B = 20$): $R^{\text{stat}}_{g=1}(\text{example\_pedagogical}, 20)_i - R^{\text{stat}}_{g=9}(\text{example\_pedagogical}, 20)_i$.

---

## 4. Automated Validation Checks and Export Requirements

A confirmatory run (`just library-run <run_id> --confirm-frozen --workers 8`) is accepted as `status = "validated"` only if:
1. Protocol header states `**frozen**`, `config_sha256` matches `config_sha256()`, and the git working tree is clean (`git_dirty = false`).
2. `oracle_zero_regret_max <= 1e-12`.
3. `single_lifetime_unlimited_abs_diff_max <= 1e-12`.
4. `none_gen0_regret_deficit <= 0.0` (confirmatory F1 parity: `none` gen 0 held-out regret $\ge 5.0$).
5. `unlimited_post_gen2_drop_deficit <= 0.0` (confirmatory F2 parity: `unlimited` gen $1 \to 9$ drop $\ge 1.0$).
6. `placebo_advantage_over_none_max <= 0.5` (confirmatory F3 parity: random and shuffled placebos at $B = 20$ do not beat `none`).
7. `bit_budget_violation_max <= 0.0` (no inherited record exceeds its configured bit budget).

Artifacts written to `runs/<run_id>/`: `manifest.json`, `results.json` (full budget-sweep curves, generation trajectories, and all family hypothesis statistics), and `samples.json` (non-cherry-picked lineage indices `0, 1, 2` at `B = 10, 20, 40`).
