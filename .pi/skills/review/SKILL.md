---

name: review
description: Review PolicyLab code or a diff without modifying files. Use for correctness review, architecture review, test review, reproducibility review, or pre-commit inspection. Report actionable findings ordered by severity. Do not edit unless explicitly asked afterward.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Review

Review the requested code, diff, or feature without modifying files.

Follow `AGENTS.md`.

The default mode is **read-only**.

## 1. Determine review scope

Identify exactly what should be reviewed.

Possible scopes include:

* current working-tree diff;
* staged diff;
* specific files;
* one module;
* one feature implementation;
* changes since a given commit.

Do not expand the review into unrelated parts of the repository unless they are necessary to understand the change.

## 2. Inspect context selectively

Read:

* the changed code;
* directly related interfaces;
* relevant tests;
* only the documentation needed to verify the change.

Useful documentation:

* `docs/architecture.md` — dependency boundaries
* `docs/domain-model.md` — terminology
* `docs/hex.md` — Hex correctness
* `docs/training.md` — RL and self-play invariants
* `docs/game-session.md` — match/session behavior
* `docs/frontend.md` — frontend boundaries

Do not read every document automatically.

## 3. Review correctness first

Look for behavior that is wrong or incomplete.

Examples:

* incorrect game rules;
* invalid state transitions;
* off-by-one errors;
* wrong player perspective;
* incorrect terminal handling;
* illegal actions being accepted;
* stale state being used;
* inconsistent value-sign conventions;
* MCTS statistics propagated incorrectly;
* model outputs used without legal-action masking;
* replay reconstruction differing from live execution.

Prioritize concrete correctness problems over stylistic preferences.

## 4. Review architecture

Verify that the change respects dependency direction.

In particular:

* `games` remains independent of higher layers;
* `search` does not depend on training;
* agents do not own match progression;
* training concerns do not leak into game rules;
* server handlers do not contain domain logic;
* React does not duplicate authoritative game rules;
* no circular dependency is introduced.

Flag unnecessary abstractions when they materially increase complexity.

Do not object to simple code merely because a more generic abstraction could theoretically exist.

## 5. Review state and determinism

Look for:

* hidden mutable global state;
* accidental mutation of logically immutable game state;
* uncontrolled randomness;
* implicit dependence on collection iteration order;
* state shared incorrectly between matches or experiments;
* concurrency races affecting turns or sessions.

For randomized behavior, verify that explicit seeds or RNG objects are used where reproducibility requires them.

## 6. Review RL-specific correctness

When reviewing training, MCTS, or evaluation code, check especially:

### Perspective

Verify that player perspective is consistent across:

* state encoding;
* neural value output;
* MCTS backup;
* terminal values;
* training outcomes;
* evaluation results.

Perspective/sign bugs are high severity.

### Policy

Verify that:

* illegal actions are masked;
* policy targets represent the intended search distribution;
* probability normalization is correct;
* terminal states do not request meaningless policies.

### MCTS

Verify where applicable:

* selection uses the intended statistics;
* visit counts are updated correctly;
* values are backed up with correct perspective/sign;
* terminal nodes are evaluated correctly;
* root exploration is applied only where intended;
* evaluation play does not accidentally use training exploration.

### Training

Verify:

* targets align with model outputs;
* losses use correct dimensions;
* replay samples cannot refer to the wrong outcome perspective;
* checkpoint save/load restores equivalent model behavior;
* experiment settings are not silently hard-coded.

## 7. Review reproducibility

For experiment-related changes, look for:

* missing explicit seeds;
* mutable historical configuration;
* checkpoint overwrite risk;
* undocumented hyperparameter changes;
* experiment identity not tied to configuration/code;
* evaluation settings differing unintentionally between compared agents.

Training completion alone is not evidence of improved playing strength.

## 8. Review tests

Check whether tests cover the behavior changed.

Look for:

* missing regression tests;
* tests that only verify implementation details;
* missing failure cases;
* missing edge cases;
* tests that pass without exercising the changed code;
* over-mocking that hides integration errors.

For game logic, prefer small deterministic examples.

Do not demand excessive tests for trivial code with no meaningful behavior.

## 9. Review API and protocol changes

Check whether the change unintentionally modifies:

* public Python interfaces;
* serialized game state;
* WebSocket events;
* API response shapes;
* persisted experiment formats;
* frontend/backend terminology.

If an intentional protocol change lacks corresponding consumer updates, flag it.

## 10. Review scope quality

Look for unrelated changes such as:

* opportunistic refactors;
* mass formatting;
* unrelated renaming;
* dead code;
* debugging output;
* generated artifacts;
* large files;
* dependency additions unrelated to the task.

A focused patch is preferred.

## 11. Run validation when appropriate

For code review, run non-mutating validation when feasible:

```text
just verify
```

Do not run `just format` during a read-only review because it modifies files.

Additional focused tests may be run when needed to verify a suspected problem.

Never modify production code or tests merely to determine whether a finding is valid.

## 12. Severity

Classify findings using:

### Critical

The change can corrupt experiments/data, produce fundamentally invalid results, or create a severe safety/security problem.

### High

A real correctness bug affecting normal behavior, training validity, game results, or important APIs.

### Medium

A meaningful defect that affects edge cases, maintainability, reproducibility, or future correctness.

### Low

Minor issue with limited practical impact.

Do not inflate severity.

Style preferences generally do not deserve findings unless they cause concrete maintainability or correctness problems.

## 13. Output format

Report findings first, ordered by severity.

For each finding include:

```text
[Severity] Short title

Location:
path/to/file.py:line

Problem:
Concise explanation of what is wrong.

Impact:
What can fail or become incorrect.

Suggested fix:
The smallest reasonable correction.
```

Where useful, include a concrete example demonstrating the failure.

After findings, include:

### Validation

Commands actually run and their results.

### Summary

A short assessment of the change.

If no actionable findings exist, say explicitly:

```text
No actionable findings found.
```

Do not invent findings simply to make the review look thorough.

## 14. Do not edit

Do not modify files during this skill unless the user explicitly asks to fix selected findings afterward.

A request such as "review this" means review only.
