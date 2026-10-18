# Architecture Decision Log & Deferred Questions

## Accepted directions (2026-10-18)

**D01 — Research first, not a generic RL platform.** World-specific methods and visualizations; shared only run metadata and presentation conventions.

**D02 — Reuse current PolicyLab.** Audit first; gradual extension preferred to total rewrite. No code has been replaced by this handoff.

**D03 — Design starts immediately.** Working viewer and site precede full experiment automation. User explicitly values public presentation and immediate insight.

**D04 — Gravity Atlas first.** Numerical CR3BP + trajectory family/robustness comparison. Baselines before MAP-Elites, no compulsory RL.

**D05 — Static-first public launch.** Publish curated immutable results via React static site; local Python generates data. No initial API/server required for public readers.

**D06 — First four new worlds.** Gravity Atlas, The Last Library, Algorithm Forge, Constitution Machine; Hex preserved, others not committed to delivery.

**D07 — Explicit no-go gates.** If physics baseline fails or comparative hypothesis is unsupported, report/revise instead of endless expansion.

**D08 — English source and research materials.** Code, UI, reports, content and docs in English.

## Deferred, not blocked

- Final public product name/domain/visual brand (continue with `PolicyLab` unless author changes it).
- Choose actual static hosting provider (Cloudflare Pages recommended, alternatives acceptable).
- Full production run schema and database: defer after static-file contracts prove usable.
- Python vs Rust science engine: Python default; profile before changing.
- Exact method hyperparameters, robust perturbation distributions and orbital shell radii: select and freeze during M2 feasibility, *before* M3 comparison.
- MDX/content pipeline: add when first actual research article is written; simple Markdown is sufficient before that.
- Distribution of code artifacts, dataset size policy, and license: check existing LICENSE and do not assume a license.
- Community contributors and volunteer compute: start with GitHub issues/PRs; no distributed scheduler in MVP.

## Conflicts to reconcile with existing documentation

| Prior docs | Handoff intent | Safe action |
|---|---|---|
| `docs/architecture.md` is Hex-/game-centric | Multiple independent research worlds | Add independent `research/` feature(s); don't invert existing dependencies |
| `docs/domain-model.md`: Experiment = single training run | Generic study → definition → run | Keep Hex meaning until explicitly migrated; use unambiguous JSON terms |
| `docs/frontend.md` proposes Hex-first frontend order | Gravity UI from first iteration | This roadmap supersedes frontend *implementation order* only, not its UI separation rules |
| `README.md` currently empty in inspected branch | Public-facing lab identity | Add README content in a dedicated later code PR, not from ZIP overwrite |

## Conditions for revising the plan

A milestone may change only because of observed source state, physical feasibility, research data, runtime limits or a clear user product choice. Document each meaningful change and its reason. Do not enlarge scope simply because implementing another framework sounds elegant.
