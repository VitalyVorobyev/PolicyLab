---
description: Design, run, or compare a reproducible PolicyLab RL experiment
argument-hint: "<experiment question or task>"
---

Work on the following PolicyLab RL experiment:

$@

Before changing configuration, code, or starting training:

1. Read `AGENTS.md`.
2. Read `.pi/skills/rl-experiment/SKILL.md` and follow it.
3. Read `docs/training.md`.
4. Inspect the relevant baseline configuration, code, checkpoints, and evaluation setup.
5. State the concrete experimental question.
6. Identify the baseline and the controlled change.
7. Ensure the experiment has an explicit seed and reproducible configuration.

Do not mutate or overwrite historical experiments, configurations, or checkpoints.

Prefer changing one important experimental variable at a time.

Before any substantial training run:

1. run the appropriate smoke configuration;
2. verify that self-play, training, checkpointing, and evaluation work;
3. run `just verify` if code was changed.

Evaluate playing strength under fixed conditions. Do not use training loss alone as evidence of improvement.

When comparing agents or checkpoints, verify that board size, search budget, side assignment, evaluation game count, and other relevant conditions are comparable.

Finish with the report defined by the `rl-experiment` skill:

* Question
* Baseline
* Change
* Execution
* Evaluation
* Result
* Interpretation
* Artifacts
* Issues

Distinguish observed results from interpretation and speculation.
