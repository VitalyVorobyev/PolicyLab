# Constitution Machine — Planned, Conditional Research Specification (v0)

**Theme:** Multi-agent decision-making, resource governance, evolutionary game theory.  
**Priority:** fourth world only if bandwidth remains. This is *not* a claim to simulate human political systems.

## Research question

> Do adaptive resource-governance policies retain their effectiveness when the behavioral mix of the governed agents changes, compared to simple fixed rules?

Avoid normative claims of fairness from arbitrary synthetic payoff functions. The first environment is a stylized mathematical system, not a socioeconomic model of real people.

## Minimal simulation

- 4–6 agents use one renewable shared resource ('lake'), with a deterministic or controlled-stochastic replenishment function and a small number of declared behavioral policies.
- Actions per period: harvest amount, optional conservation/investment if needed for the hypothesis. Individual payoff from consumption and long-term survival.
- A central rule controller may set a harvest cap or tax/penalty within declared legal bounds every `K` steps; fixed enforcement cost (if included) is explicitly modeled.
- Seeded episodes with fixed horizon and identical resource/population starting conditions per comparison.

## Baselines & candidate treatments

- Open access / no caps.
- Simple fixed harvest quota tuned on training populations.
- Proportional/feedback cap based on observed resource stock.
- Optional optimized adaptive policy (bandit, evolutionary optimization, RL) **only after** feedback baseline is established.

Training populations are mixtures of several predeclared fixed behavior types: short-term extraction, cautious conservation, quota compliance and opportunistic rule violation. Held-out evaluation changes mixture weights and introduces an out-of-distribution policy not used for tuning.

## Metrics

Resource extinction episodes, cumulative individual and total utilities, stock variability, enforcement cost, distribution of payoffs, and worst-case performance under population shift. Report tradeoffs, not a single hand-crafted 'fairness' scalar.

## Interactive page

A tiny isometric or minimalist 2D settlement around a lake; live **replay** of harvest, quotas and resource renewal; stock and utility charts; population-mixture selector on precomputed runs; explanations of why a policy helped or failed. The visualization is not the economic model.

## Gates and risks

**Feasibility:** show a real commons dilemma exists in the declared dynamics; demonstrate that free access can be harmful under some regimes without baking a win for the intended adaptive method into the reward. **Research:** robustness-to-population-shift on held-out mixtures with competent feedback and fixed-rule baselines. **Stop:** if results are artifacts of arbitrary payoff choices or a simple feedback policy dominates, publish the limitations instead of multiplying agents and incentives.

## Background

- AI Economist: https://www.salesforceairesearch.com/projects/the-ai-economist
- Melting Pot (multi-agent generalization): https://github.com/google-deepmind/meltingpot
