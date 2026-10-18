---
name: numerics-review
description: Read-only review of Phase Space scientific Python code — numerical correctness, invariants, units and frames, tolerances, event handling, reproducibility, and test independence. Use after writing or changing study code (integrators, dynamics, events, optimizers, exporters), or when asked "is this correct", "review the physics", "check the numerics". Does not edit files unless asked afterward.
---

# Numerics review

Review the requested scope (diff, files or module) **without modifying files**. Follow `AGENTS.md`. Read the study spec in `docs/studies/<study>/README.md` and `docs/data-contract.md` as needed.

## Checklist, in priority order

1. **Equations** — compare implemented equations term by term against the spec and an independent source (e.g. CR3BP: Coriolis signs, `r1`/`r2` definitions, primary positions at `-mu` and `1-mu`).
2. **Invariants** — conserved quantities (e.g. Jacobi constant) computed independently of the integrator and monitored on coast arcs; drift thresholds declared.
3. **Units and frames** — every quantity has a declared unit and frame; conversions (normalized ↔ km, days, km/s; inertial ↔ rotating velocity) tested against hand-computed values.
4. **Integration** — solver and tolerances declared and recorded; results checked under tighter tolerances or a second integrator; no silent fixed-step fallback.
5. **Events** — direction, terminal flag, initial-condition false triggers (e.g. departure shell), grazing encounters, collision surfaces, horizon termination.
6. **Numerical health** — non-finite states detected and reported; singularities near primaries handled; failure is explicit, not a silently truncated trajectory.
7. **Reproducibility** — explicit seeds threaded through; no global RNG; configuration and solver settings flow into exported manifests.
8. **Tests** — expected values come from independent references (analytic cases, invariants, published orbits), not from the implementation's own output; tolerances justified; edge cases covered.
9. **Export** — artifacts match `docs/data-contract.md`: schema version, strictly increasing time, state order, checksums, status honest (`validated` only with recorded checks).

## Output

Findings ordered by severity: **Critical** (wrong science), **Major** (unverified or fragile), **Minor** (clarity, missing test). Each finding: file:line, the problem, a concrete failure scenario, and a suggested fix. End with what was verified as correct and what could not be checked.
