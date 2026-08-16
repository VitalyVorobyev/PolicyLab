---
description: Review PolicyLab changes without modifying files
argument-hint: "[scope]"
---

Review PolicyLab code without modifying files.

Scope:

$@

If no scope is provided, review the current working-tree diff.

Before reviewing:

1. Read `AGENTS.md`.
2. Read `.pi/skills/review/SKILL.md` and follow it.
3. Inspect the requested diff/files and directly related code.
4. Read only the documentation needed to verify correctness.

Prioritize:

* correctness;
* architecture violations;
* game-rule errors;
* player-perspective/sign errors;
* MCTS or training errors;
* reproducibility problems;
* missing or ineffective tests;
* unintended API/protocol changes;
* accidental scope expansion.

Run non-mutating validation when useful:

`just verify`

Do not run commands that modify files.

Report actionable findings first, ordered by severity, using the format defined in the review skill.

If no actionable findings exist, say so explicitly.

Do not fix findings unless asked in a separate task.
