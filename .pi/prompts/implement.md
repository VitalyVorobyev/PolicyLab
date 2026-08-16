---
description: Implement one focused PolicyLab task using the project implementation workflow
argument-hint: "<task>"
---

Implement the following PolicyLab task:

$@

Before editing any files:

1. Read `AGENTS.md`.
2. Read `.pi/skills/implement-feature/SKILL.md` and follow it for this task.
3. Inspect the relevant existing code.
4. Read only the project documentation relevant to the requested behavior.

Keep the task narrowly scoped.

Do not implement adjacent features, perform unrelated refactors, or redesign existing abstractions unless the requested behavior genuinely requires it.

If the request conflicts with the documented architecture, stop and explain the conflict instead of silently violating it.

After implementation:

1. add or update appropriate tests;
2. run focused validation while working;
3. run `just verify`;
4. inspect the final diff for accidental changes.

Do not consider the task complete unless validation succeeds, or clearly report why a required check could not succeed.

Finish with the completion report required by the `implement-feature` skill.

Do not start the next feature.
