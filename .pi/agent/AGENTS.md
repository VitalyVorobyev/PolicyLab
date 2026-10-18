# Coding instructions

- Inspect the relevant code before modifying it.
- Make the smallest coherent change that solves the task.
- Do not invent APIs, types, files, or project conventions; inspect them first.
- Prefer existing project patterns over introducing new abstractions.
- Run relevant formatting, linting, builds, and tests after changes.
- Fix failures caused by your changes before finishing.
- Do not commit, push, rewrite git history, or delete branches unless explicitly requested.
- Do not modify secrets, credentials, deployment configuration, or production data unless explicitly requested.
- Keep explanations concise.
- For non-trivial tasks, form a short plan before editing.
- Use `just format` for formatting and safe automatic lint fixes; do not spend agent iterations manually reformatting code.
- Never modify a test merely because the implementation fails it. First verify that the test contradicts the specification.
