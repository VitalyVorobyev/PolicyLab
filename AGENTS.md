# PolicyLab Agent Instructions

PolicyLab is a research workbench for training, evaluating, inspecting, and playing against self-learning game agents.

Initial game: Hex.
Initial learning approach: AlphaZero-style self-play with MCTS and a policy/value network.

## Before changing code

Read only the documentation relevant to the task:

* `docs/architecture.md` — module boundaries and dependency direction
* `docs/domain-model.md` — shared terminology
* `docs/hex.md` — Hex rules and representation
* `docs/training.md` — self-play and learning design
* `docs/game-session.md` — matches, sessions, events, replay
* `docs/frontend.md` — React architecture and UI boundaries

Inspect existing code before proposing new abstractions.

## Architecture

Dependencies point downward:

```text
web
 ↓
server
 ↓
experiments / training / agents
              ↓
            search
              ↓
            games
```

Important rules:

* `games` contains pure deterministic game logic.
* `games` must not depend on ML, search, server, persistence, or UI.
* `search` depends on games, not training.
* agents choose actions; they do not run matches.
* training may depend on games, search, and agents.
* React never owns authoritative game rules.
* FastAPI handlers should orchestrate, not contain domain logic.

Do not introduce circular dependencies.

## Development rules

* Make small, focused changes.
* Do not redesign unrelated code.
* Prefer the simplest implementation satisfying the current requirement.
* Do not add abstractions for hypothetical future needs.
* Preserve deterministic behavior where practical.
* Use explicit random seeds in code involving randomness.
* Never silently change experiment hyperparameters.
* Never overwrite experiment artifacts or checkpoints.
* Add tests for new behavior and regression tests for bugs when practical.
* Do not weaken linting, typing, or tests merely to make validation pass.

## Validation

Use project commands rather than reconstructing tool invocations manually.

```text
just test
just check
just verify
just format
```

Before considering a code-changing task complete, run:

```text
just verify
```

If a required check cannot run, report that explicitly.

## Working style

For implementation tasks:

1. inspect the relevant code and docs;
2. identify the smallest appropriate change;
3. implement it;
4. add or update tests;
5. run validation;
6. review the final diff for accidental changes.

Do not continue into adjacent features unless explicitly requested.

## Completion report

At the end of an implementation task, report briefly:

* what changed;
* important design decisions;
* validation performed;
* any unresolved issues.
