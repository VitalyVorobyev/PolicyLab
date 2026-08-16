---
description: Inspect PolicyLab and produce a focused implementation plan without changing files
argument-hint: "<task>"
---

Plan the following PolicyLab task:

$@

Do not modify any files.

## Process

1. Read `AGENTS.md`.
2. Inspect the existing code relevant to the task.
3. Read only the project documentation needed to understand the task.
4. Identify the architectural layer that owns the requested behavior.
5. Find existing interfaces, tests, and patterns that should be reused.
6. Determine the smallest implementation that satisfies the request.
7. Identify tests required to verify the behavior.
8. Identify any ambiguity, architectural conflict, or significant risk.

Do not design adjacent features.

Do not introduce abstractions for hypothetical future requirements.

Do not produce code unless a very small signature or data shape is necessary to make the plan unambiguous.

## Output

Return:

### Goal

One or two sentences describing the exact behavior to implement.

### Relevant code

The existing files and abstractions involved.

### Approach

A short ordered implementation plan.

### Tests

The specific behaviors that should be tested.

### Files

Files expected to be created or modified.

### Risks

Only concrete uncertainties, architectural conflicts, or important implementation risks.

If there are none, write:

`Risks: none.`

Keep the plan concise enough that it can be used directly as the specification for one implementation task.
