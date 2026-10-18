# The Last Library — Planned Research Specification (v0)

**Theme:** Cultural evolution, iterated learning, compressed memory, generalization.  
**Priority:** second world, *after* Gravity Atlas has a public validated investigation.

## Narrow research question

> When does transmitting a strictly limited external memory between otherwise fresh agents enable cumulative improvement on a family of previously unseen but structurally related tasks?

Not an open-ended artificial civilization. Do not start with LLM-driven societies, language evolution at scale or neural civilization simulations.

## First world and task

- A family of small procedurally generated 2D grid-navigation tasks with recurring hidden regularities (e.g. color-coded terrain costs, symbolic door/key rules or reusable hazard cues).
- Each generation gets a controlled trial budget on sampled training worlds; new generations face *held-out layouts with the same latent rule family*.
- An agent learns local task properties during its lifespan. On retirement it may write **at most 256 bytes** into an inherited record readable by the next generation.
- The next generation starts with no learned weights and no private experience; receives only the permitted inherited record. This makes memory transmission explicit.
- A deliberately interpretable small symbolic/probabilistic agent is preferred over a large neural network for baseline research.

## Methods / baselines

A. **No inheritance:** new learner every generation, same lifetime interaction budget.  
B. **Example memory:** retain compact raw observation/action examples inside 256 bytes.  
C. **Rule memory:** transmit compressed hypotheses/estimated task parameters within 256 bytes.  
D. **Control:** random or shuffled inherited bytes / unrelated predecessor memory.

Learning/memory selection may later use evolutionary search, Bayesian updating, or RL. RL is not an assumption of the problem.

## Metrics and tests

- Held-out episode success rate and regret versus an oracle with correct latent rules.
- Learning speed on unseen worlds as a function of generation index.
- Bits/bytes used, retention of useful hypotheses, and transfer advantage over no-inheritance and example-memory baselines.
- Negative transfer when the latent rule family *changes*; memory corruption and bottleneck-size ablations (0 / 64 / 256 bytes).
- Fixed, independent procedural seeds; identical lifetime evaluation budgets; no leak of future task seeds into memory.

## Interactive page

Generation lineage, current memory content (human-readable interpretation), mini-map playback, per-generation competence curves, and a rule-change switch selecting **precomputed evaluation scenarios**. Show failure and negative transfer, not merely an attractive progression.

## Gates / likely pitfalls

**Feasibility:** prove the task contains transferable latent structure; compare to a genuine oracle and a from-scratch baseline. If there is no transfer signal or the task has a trivial constant solution, redesign the task before more complex learning.

**Complete:** reproducible finite experiment with four controls, multiple seeds, memory-size ablation, readable report and interactive replay. **Stop:** if apparent cumulative knowledge disappears on held-out worlds, publish that finding rather than simulating more generations indefinitely.

## Starting references

- Iterated learning models and cultural evolution: see `REFERENCES.md` for topic-oriented reading, to be narrowed and verified when this world becomes active.
