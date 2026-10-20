# Vision, Positioning, and Scope

## Thesis

Phase Space is a collection of **a few deep, distinct, public computational investigations** at the intersection of search, learning, self-organization, and adaptive intelligence. Each investigation should be enjoyable to explore *and* scientifically honest. The research question is central; the UI makes the reasoning and evidence inspectable.

This is **not** a toolkit marketing project, an algorithm zoo, a universal RL playground, or a perpetual simulation game.

## MVP research portfolio

| Priority | World | Core unknown | First method families | Status |
|---|---|---|---|---|
| 1 | Gravity Atlas | Can a diverse archive of Earth–Moon free-return candidates make trajectory selection under changing conditions more robust/useful than conventional multi-start optimization? | Numerical integration, local/multi-start optimization, MAP-Elites | Closed (D29) |
| 2 | Model Citizens | Does the mechanism an RL policy designer discovers depend on which model of people it is trained against, when the models pass different held-out tests (next-choice prediction vs whole-game outcomes)? | Behavioural models fitted to public human data (GRU clone, parametric, learning, LLM citizens), PPO planners | **Active** (D31, D35) |
| 3 | The Last Library | When does a strictly bounded inherited memory allow cumulative improvement across generations on related unseen tasks? | Iterated learning, compact symbolic memory, evolutionary selection | Planned, after Model Citizens |
| 4 | Algorithm Forge | When does population-based program search outperform simple structured search under fixed execution budgets? | Genetic programming, beam search, small exhaustive baselines | Planned |
| Retired | Constitution Machine | Do adaptive shared-resource rules generalize to populations whose behavior changes? | Agent-based simulation, evolutionary game theory, feedback rules | Superseded by Model Citizens (D31) |
| Archived | Hex Arena | Self-play, search and agent strength in a reproducible game | MCTS, AlphaZero-style learning | Archived on branch `archive/policylab-hex` (D11) |

Not in the initial portfolio: Curious Scientist (weak first hypothesis in its earlier form), Living Matter (interesting but outside the four-study cap), calibration/robotics and distributed community training. They are not forbidden later.

## Definition of one completed research world

A complete world has all of the following:

1. A falsifiable research question and at least one credible baseline.
2. A deterministic or controlled-stochastic scientific model with validation checks.
3. A bounded compute budget and a written termination/kill criterion.
4. Several independent runs where randomness matters; uncertainty is visible.
5. A bespoke, useful, aesthetically coherent interactive visualization.
6. An English-language, readable research article with method, evidence, limitations, links, and reproducibility metadata.

## Public experience

The bar is publishable, literature-aware work (D09): each study names its nearest prior work before its protocol is frozen.

**Gallery → Study page → Interactive experiment → Evidence/replay → Article.**

Visitors can browse and manipulate *recorded* scenarios without authentication or a live Python service. Real-time user-triggered training is not necessary for launch. A later local CLI allows others to reproduce experiments and contribute results through GitHub pull requests; a trusted server/worker network is a distinct future project.

## Design commitments

- The visual interface is a scientific instrument from day one, not a deferred dashboard.
- Publish a useful static preview early; label unfinished research honestly.
- Prefer narrow, novel *questions* and well-controlled comparisons over invented complexity.
- Keep worlds autonomous: do not force every world into Gymnasium, RL, or a single `Environment` interface.
- Share only the essential cross-cutting pieces: content metadata, run records, metrics, artifact loading, and design primitives.
- Do not promise peer-reviewed novelty; document null results and negative findings.

## Release quality bar

The site should look intentional on laptop and mobile, support keyboard and reduced motion, have no dead controls, make units/coordinate frames legible, and clearly distinguish example data, computed results, and validated results.
