# Research Methodology — Small Studies, Real Evidence

## Standard study outline

Each research study should have: (1) question and testable hypothesis, (2) precise model, (3) baselines, (4) objective and measures, (5) bounded experimental protocol, (6) independent validation, (7) results with uncertainty, (8) failure modes, (9) reproducibility package, (10) limitations and next question.

## Before the main experiment

- Conduct a **targeted literature scan** for the narrow experiment, not an open-ended literature review.
- Reproduce at least one trusted reference or analytical special case. State model simplifications plainly.
- Define success/failure, evaluation episodes and parameter bounds *before* seeing comparative results.
- Choose at least one unsurprising competent baseline. A fancy method beating a random policy is insufficient.
- Freeze a manageable compute budget on the developer's local MacBook (M4 Pro, 24 GB memory is an upper planning target, not a guarantee of timings).

## Fair comparisons

- Prefer equal counts of expensive **objective evaluations / ODE propagations** across optimizers; also log wall-clock and CPU details.
- Use independent seeds for stochastic methods; make repetitions high enough to reveal dispersion and describe the uncertainty (do not choose an arbitrary magic sample size).
- Hold out some task instances or perturbations from method tuning. No seed cherry-picking.
- Distinguish success rate (feasibility), conditional quality among successes, end-to-end cost, and worst-case behavior.
- State if an optimizer's internal work is not captured by a simple evaluation-count metric.
- Null findings, nonconvergence and parameter sensitivity are publishable *research results* in this project.

## Reproducibility

An experiment must retain its code version, package lock information, full configuration, seed, solver tolerances, independent check settings, data hashes, machine/OS details relevant to performance and exact CLI reproduction commands. Published runs are immutable.

Numerical exact-bit reproducibility across architectures is generally unrealistic; instead define tolerances and compare the scientific observables.

## Numerical rigor (especially Gravity Atlas)

- Use independently checked equations and invariants.
- Monitor Jacobi integral drift on coast segments away from collision singularities.
- Correctly handle Earth/Moon collision surfaces, grazing encounters, crossings and max flight duration.
- Distinguish numerical feasibility from mission-grade feasibility; CR3BP ignores lunar eccentricity, ephemeris errors, 3D effects, perturbations and operational constraints.
- Validate sensitive results using tighter tolerances or independent integration; check optimizer winners rather than blindly trusting their stored scores.

## Publication template

1. **Question**: short and falsifiable.
2. **Why it matters**: connection to established literature and known limits.
3. **Interactive illustration**: clearly separate example scenario from result.
4. **Setup and baselines**: precise but comprehensible.
5. **Evidence**: distributions across seeds, failures, confidence, comparisons and plots.
6. **Interpretation**: what appears supported, what is not.
7. **Reproduce**: links to immutable configs, run artifacts, code commit and environment.
8. **Further reading**: stable links, checked metadata, clearly attributed prior work.

Never use a generated prose summary as an unverified scientific conclusion. LLMs may assist editing; computed claims must trace to artifact data.

## Review gates

- **Feasibility gate:** the model and baseline work at all.
- **Comparison gate:** methods have matched, well-defined budgets and objective.
- **Publication gate:** a competent reader can inspect and reproduce every headline claim.

If any gate fails, stop and document the blocker. Do not add more sophisticated methods to mask an unresolved modeling issue.
