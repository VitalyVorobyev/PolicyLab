# PolicyLab — Research Laboratory Handoff

**Status:** approved direction; implementation has not started as a result of this handoff.  
**Prepared:** 2026-10-18  
**Repository used for context:** [VitalyVorobyev/PolicyLab](https://github.com/VitalyVorobyev/PolicyLab) (default-branch documentation inspected).  
**Audience:** Codex, Claude Code, and human collaborators.  
**Language:** English for code, documentation, UI copy, and published research.

## Mission

Build a **small, public, visually excellent computational research laboratory**, not a general-purpose reinforcement-learning framework. Each world is a bounded investigation with a question, reproducible baselines, interpretable results, an interactive viewer, and a readable report. Methods are chosen by the problem: numerical optimization, quality-diversity, evolutionary computation, reinforcement learning, etc.

**First research:** Gravity Atlas. **First public artifact:** a beautiful, working, static research page that plays back genuine or explicitly labeled example data. **First scientific gate:** validate the Earth–Moon planar CR3BP and produce a physically consistent benchmark trajectory before running comparisons.

## Index — read in this order

| File | Purpose |
|---|---|
| [`AGENT-HANDOFF.md`](AGENT-HANDOFF.md) | Instructions and immediate first coding task for Codex / Claude Code |
| [`VISION-AND-SCOPE.md`](VISION-AND-SCOPE.md) | Product intent, quality bar, strict non-goals |
| [`ROADMAP.md`](ROADMAP.md) | Milestones, dependencies, acceptance criteria, kill gates |
| [`SYSTEM-DESIGN.md`](SYSTEM-DESIGN.md) | Minimal architecture and repository integration |
| [`FRONTEND-DESIGN.md`](FRONTEND-DESIGN.md) | UX, information architecture, visual style, interaction requirements |
| [`DATA-CONTRACT.md`](DATA-CONTRACT.md) | Run manifests, trajectory files, units, versioning, provenance |
| [`RESEARCH-METHODOLOGY.md`](RESEARCH-METHODOLOGY.md) | Scientific quality, reproducibility, benchmarking and publications |
| [`research/GRAVITY-ATLAS.md`](research/GRAVITY-ATLAS.md) | **Active** investigation, mathematical model and first study |
| [`research/THE-LAST-LIBRARY.md`](research/THE-LAST-LIBRARY.md) | Planned investigation into inherited knowledge |
| [`research/ALGORITHM-FORGE.md`](research/ALGORITHM-FORGE.md) | Planned investigation into synthesis of short programs |
| [`research/CONSTITUTION-MACHINE.md`](research/CONSTITUTION-MACHINE.md) | Planned investigation into robust social institutions |
| [`research/HEX-ARENA.md`](research/HEX-ARENA.md) | Existing research world: preservation and optional integration |
| [`DECISIONS-AND-QUESTIONS.md`](DECISIONS-AND-QUESTIONS.md) | Decisions, explicit deferrals and genuinely open issues |
| [`REFERENCES.md`](REFERENCES.md) | Reference links and existing repository documents |

## Installation into repository

This archive contains **only additive Markdown documents** under `handoff/`. Unpack at repository root; do **not** overwrite root `README.md`, `AGENTS.md`, or existing `docs/` files. Treat this handoff as a planning baseline, not evidence that code or science has already been implemented.

```bash
# From repository root; inspect the archive before extracting.
unzip -l policylab-handoff.zip
unzip policylab-handoff.zip
# Give your coding agent the contents of handoff/AGENT-HANDOFF.md.
```

## First visible milestone (vertical slice)

1. Existing code audited; scope of reuse recorded.
2. Public-ready React research gallery and Gravity Atlas page, with a truthful data-state label.
3. Versioned local trajectory fixture and working play/pause/scrub/inspect UI.
4. Build, basic automated checks, and documented static deployment procedure.

The follow-on scientific milestone validates the integrator and publishes **real computed** trajectory data. A working viewer alone is **not** a research result.

## Important boundaries

- No distributed compute, user accounts, collaborative editing, custom plugin platform, database server or training scheduler in the MVP.
- Do not migrate the whole repository, rename it, or delete working Hex capabilities without an explicit reason.
- No claims of novel scientific findings in advance.
- Any synthetic trajectory must be visibly marked `Illustrative — not a computed physical trajectory`. Never present it as a valid CR3BP solution.
- Run the repository's mandatory `just verify` after code changes, plus frontend checks appropriate to the existing tooling.
