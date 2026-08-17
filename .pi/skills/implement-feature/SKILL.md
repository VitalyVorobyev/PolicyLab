---
name: implement-feature
description: Implement or modify PolicyLab code with a small, tested, architecture-consistent change. Use when asked to add a feature, fix a bug, refactor code, or change behavior. Do not use for review-only tasks or for designing/running RL experiments.
---

# Implement Feature

Implement the requested change with the smallest correct patch.

Follow `AGENTS.md` at all times.

## 1. Understand the task

Before editing:

1. Identify exactly what behavior is requested.
2. Determine which architectural layer owns that behavior.
3. Inspect the relevant existing code.
4. Read only the documentation needed for this task.

Use documentation selectively:

* game rules or state → `docs/hex.md`
* terminology → `docs/domain-model.md`
* dependency boundaries → `docs/architecture.md`
* matches, sessions, replay → `docs/game-session.md`
* training/self-play → `docs/training.md`
* React/UI → `docs/frontend.md`

Do not read every document automatically.

## 2. Inspect before designing

Search the repository for:

* existing interfaces;
* nearby implementations;
* relevant tests;
* existing naming conventions;
* utilities already solving part of the problem.

Prefer extending an existing pattern over creating a parallel one.

Do not invent an abstraction before checking whether one already exists.

## 3. Define the minimal change

Before editing, determine:

* files that need to change;
* behavior that needs tests;
* architectural dependencies involved.

Keep the scope narrow.

Do not:

* implement adjacent features;
* perform unrelated cleanup;
* rename unrelated APIs;
* reorganize directories without need;
* introduce infrastructure for hypothetical future requirements.
* manually reformat code when the project formatter can do it automatically;

If the requested implementation would violate `docs/architecture.md`, do not silently work around the architecture. Report the conflict.

## 4. Implement at the correct layer

Put behavior in the lowest appropriate layer.

Examples:

* Hex rules → `games`
* generic search behavior → `search`
* action-selection strategy → `agents`
* learning behavior → `training`
* experiment metadata → `experiments`
* session/network orchestration → `server`
* rendering and interaction → `web`

Never move domain logic into FastAPI handlers or React components merely because that is convenient.

Preserve dependency direction.

## 5. Preserve deterministic behavior

For deterministic code:

* avoid hidden global state;
* avoid implicit randomness;
* avoid dependence on iteration order where it affects behavior.

For randomized code:

* accept or propagate an explicit random generator or seed where appropriate;
* do not introduce unseeded randomness into reproducible workflows.

Do not silently change experiment parameters.

## 6. Add tests

Every new behavior should have an appropriate test.

For a bug fix, add a regression test when practical.

Tests should verify behavior rather than implementation details.

Prefer:

* small deterministic examples;
* explicit expected values;
* boundary cases;
* failure cases.

For game logic, test the pure game layer directly rather than through the server or UI.

Do not weaken or delete existing tests to make a change pass unless the requested behavior intentionally invalidates them.

## 7. Validate incrementally

During implementation, run the narrowest useful checks first.

Examples:

```text
uv run pytest tests/path/to/relevant_test.py
uv run pytest tests/path/to/relevant_test.py -k specific_test
```

After the implementation is functionally complete:

1. Run:

   ```text
   just format
   ```

   Use the project formatter and automatic safe lint fixes instead of manually reformatting code.

2. Run:

   ```text
   just verify
   ```

   `just verify` is the authoritative completion check for Python code.

3. If `just verify` fails:
   - fix only the remaining real issues;
   - do not manually fight formatting that `just format` can handle;
   - rerun the relevant focused check;
   - rerun `just verify`.

4. If frontend validation commands exist for the affected code, run those as well.

Do not claim validation succeeded unless the command actually completed successfully.

Do not suppress lint, typing, formatting, or test failures merely to finish the task.

## 8. Review the diff

Before finishing, inspect the complete diff.

Check for:

* accidental edits;
* debug code;
* dead code;
* unrelated formatting changes;
* architecture violations;
* missing tests;
* public API changes not required by the task;
* generated or large files that should not be committed.

Remove accidental changes.

## 9. Completion report

Finish with a concise report containing:

### Changed

What behavior was implemented and the main files affected.

### Design

Only mention design decisions that are important for understanding the implementation.

### Validation

List the commands actually run and whether they passed.

### Issues

List unresolved problems or limitations.

If there are none, say:

```text
Issues: none.
```

Do not propose or implement a next feature unless asked.
