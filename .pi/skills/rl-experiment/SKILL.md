---
name: rl-experiment
description: Design, configure, run, or compare PolicyLab reinforcement-learning and self-play experiments. Use for training runs, hyperparameter changes, checkpoint evaluation, ablations, and experiment analysis. Do not use for ordinary feature implementation or code review.
---

# RL Experiment

Work with PolicyLab experiments in a reproducible and comparison-oriented way.

Follow `AGENTS.md` and `docs/training.md`.

## 1. Identify the experiment question

Before changing configuration or running training, state the concrete question being tested.

Examples:

* Does a larger MCTS budget improve playing strength?
* Does this network architecture learn faster?
* Does canonical state encoding improve sample efficiency?
* Does a larger replay buffer improve stability?

Avoid experiments whose only goal is "try these settings."

The experiment should have a measurable expected outcome.

## 2. Inspect the baseline

Identify the experiment or configuration being used as the baseline.

Inspect:

* complete configuration;
* game and board size;
* model architecture;
* MCTS settings;
* replay-buffer settings;
* optimizer settings;
* training duration;
* evaluation setup;
* random seed.

Do not assume defaults that are not visible in configuration.

## 3. Never mutate historical experiments

Existing experiment records, configurations, checkpoints, and results are immutable historical artifacts.

When changing experiment behavior:

1. create a new experiment configuration;
2. preserve the previous configuration;
3. record the baseline or parent experiment when supported.

Never overwrite a checkpoint or previous experiment output.

Never reuse an experiment identifier for a different configuration.

## 4. Prefer one controlled change

For comparisons, change one important variable at a time where practical.

Good:

```text
baseline:
mcts_simulations = 200

candidate:
mcts_simulations = 400
```

Poor:

```text
change network depth,
learning rate,
MCTS budget,
replay size,
and exploration simultaneously
```

Multiple coordinated changes are acceptable only when they form one inseparable design change. State that explicitly.

## 5. Record randomness

Every experiment must have an explicit seed.

Ensure randomness is controlled where practical for:

* model initialization;
* self-play exploration;
* action sampling;
* replay-buffer sampling;
* evaluation scheduling or side assignment.

Do not add hidden unseeded randomness.

A single seed does not prove a result is robust. For important comparisons, recommend or run multiple seeds when resources allow.

## 6. Run a smoke configuration first

Before expensive training, verify the pipeline using the smallest meaningful smoke run.

The smoke run should confirm:

* game generation completes;
* MCTS completes;
* training samples are produced;
* replay-buffer insertion and sampling work;
* optimization steps run;
* losses are finite;
* checkpoint save/load works;
* evaluation executes.

A smoke run establishes correctness, not learning quality.

Do not interpret smoke-run strength as an experimental result.

## 7. Preserve configuration

All meaningful experiment behavior must be represented by explicit configuration.

Do not hide experimental constants inside code when they belong in configuration.

Examples include:

* board size;
* network architecture;
* MCTS simulation budget;
* PUCT parameters;
* exploration settings;
* replay-buffer capacity;
* batch size;
* optimizer;
* learning rate;
* training schedule;
* checkpoint cadence;
* evaluation opponents;
* evaluation game count.

If code changes are necessary for the experiment, identify them separately from configuration changes.

## 8. Check experiment identity

Before a real run, ensure the experiment can be identified by at least:

* unique experiment ID;
* complete configuration;
* seed;
* Git commit;
* model configuration.

Where supported, also record:

* parent/baseline experiment;
* start time;
* hardware/device;
* software versions.

Do not proceed with a long run if the output cannot later be associated with the code and configuration that produced it.

## 9. Evaluate playing strength

Training loss is diagnostic information.

It is not the primary measure of success.

Evaluate checkpoints under fixed conditions against appropriate opponents such as:

* random agent;
* heuristic agent;
* fixed-budget MCTS;
* selected historical checkpoints;
* previous candidate/baseline checkpoint.

Record both sides where first-player advantage may matter.

Keep evaluation exploration disabled unless the experiment explicitly concerns stochastic evaluation.

Use the same evaluation conditions when comparing candidates.

## 10. Inspect useful diagnostics

Depending on the question, inspect metrics such as:

### Playing strength

* win rate;
* pairwise match results;
* rating;
* historical-checkpoint performance.

### Training

* policy loss;
* value loss;
* total loss;
* learning rate;
* replay-buffer size.

### Policy behavior

* policy entropy;
* illegal-action probability before masking, if useful;
* search-policy concentration.

### Search

* root value;
* visit distribution;
* MCTS simulations;
* search throughput;
* neural-network evaluations.

### System performance

* self-play games per second;
* positions per second;
* training batches per second;
* device utilization where available.

Collect metrics because they answer a question, not simply because they are easy to log.

## 11. Compare fairly

When comparing experiments, verify that differences in results are not explained by unintended changes in:

* board size;
* number of training positions;
* evaluation budget;
* side assignment;
* model size;
* MCTS budget;
* random seed;
* stopping condition.

If runs use different compute budgets, state that explicitly.

Do not claim one method is stronger from an unfair evaluation setup.

## 12. Interpret conservatively

Distinguish:

* observed result;
* likely explanation;
* speculation.

Examples:

Good:

```text
The candidate won 63 of 100 games against the baseline under the same
MCTS budget. This suggests improved playing strength, but one 100-game
evaluation is not enough to establish a robust improvement.
```

Bad:

```text
The lower value loss proves the agent is stronger.
```

Do not overinterpret small differences.

## 13. Validate code changes

If the experiment required code changes, use the normal implementation validation process.

Before considering those changes complete:

```text
just verify
```

For experiment-only configuration changes, validate configuration parsing and run the smoke experiment.

## 14. Completion report

Finish with a concise structured report.

### Question

What the experiment tests.

### Baseline

The reference experiment/configuration.

### Change

The controlled variable or design change.

### Execution

What was actually run, including seed and relevant budget.

### Evaluation

How playing strength was measured.

### Result

Observed measurements only.

### Interpretation

What the result supports and what remains uncertain.

### Artifacts

New configurations, experiment IDs, and checkpoints created.

### Issues

Failures, limitations, or unresolved questions.

Do not claim an experiment succeeded if training completed but meaningful evaluation was not performed.
