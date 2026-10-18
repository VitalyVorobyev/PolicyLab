---
name: study-protocol
description: Design and freeze a Phase Space study protocol before running any comparison — question, hypothesis, baselines, budgets, seeds, held-out perturbations, metrics, statistics and stop criteria. Use when a study moves from feasibility to comparison, when asked to "set up the experiment", "plan the comparison", "freeze the protocol", or when changing a frozen protocol. Produces docs/studies/<study>/protocol.md.
---

# Study protocol

A protocol is written and frozen **before** comparative results are seen. Follow `AGENTS.md` and `docs/research-methodology.md`.

## Preconditions

- `docs/studies/<study>/literature.md` exists (run `lit-scan` first otherwise).
- The feasibility gate for the study has passed (e.g. Gravity Atlas M3), or the protocol is explicitly a feasibility protocol.

## Required sections of `protocol.md`

1. **Header** — protocol version, status (`draft` / `frozen`), frozen date, git commit at freeze.
2. **Question and hypothesis** — falsifiable; state what result would refute it.
3. **Model** — exact model, parameters, units, conventions; link the code module.
4. **Candidate parameterization** — decision variables and bounds.
5. **Feasibility definition** — exact predicates and thresholds.
6. **Methods** — baselines first (at least one competent, unsurprising baseline), then the treatment. Hyperparameters for each, and how they were chosen (tuning set only).
7. **Budget** — primary unit (e.g. ODE propagations / objective evaluations), equal across methods; wall-clock as secondary. Note internal work not captured by the budget unit.
8. **Seeds** — explicit list; count justified by observed dispersion, not a magic number.
9. **Held-out conditions** — perturbation distributions or task instances never used during tuning.
10. **Metrics** — success rate, conditional quality, cost, worst case; units and definitions.
11. **Statistical analysis** — summaries, dispersion/intervals, comparisons to report.
12. **Stop / kill criteria** — what result ends the study or forces a revision.
13. **Reproduction** — exact commands and config files.

## Rules

- One controlled change per comparison where practical.
- A frozen protocol is never edited in place. Changes create a new version with a changelog entry and reason; results state which version they used.
- Never reuse an experiment or run identifier for a different configuration.
- Record the decision to freeze in `docs/decisions.md`.

## Report back

Summarize the frozen design, open risks to validity, and the estimated compute cost on the local machine (from a smoke run, not a guess).
