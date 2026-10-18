# Gravity Atlas — Active Research Specification (v0)

**Theme:** Dynamical systems, trajectory optimization, quality-diversity, robustness.  
**Status:** first study; experiment design provisional until the feasibility milestone completes.  
**RL required?** No. Explicitly *not* part of the first comparison.

## Research question

> Does maintaining a diverse archive of feasible Earth–Moon free-return-like trajectories improve the availability of useful, robust solutions when mission constraints change, relative to strong multi-start conventional optimization under matched compute budgets?

The terms **free-return-like** and **robust** require precise operational definitions below. Do not imply NASA mission certification or publish novelty claims in advance.

## Scientific model / conventions

Initial model: **planar Circular Restricted Three-Body Problem (CR3BP)** in a barycentric uniformly rotating frame and conventional nondimensional units.

Mass parameter `mu = m_moon / (m_earth + m_moon) ≈ 0.0121505856`. Earth at `(-mu, 0)`, Moon at `(1-mu, 0)`. Distance between primaries `1`, angular rate `1`; time unit is inverse mean angular rate. State `s = (x, y, vx, vy)`.

With `r1 = sqrt((x+mu)^2+y^2)` and `r2 = sqrt((x-1+mu)^2+y^2)`:

```text
xdot  = vx
ydot  = vy
vxdot =  2*vy + x - (1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3
vydot = -2*vx + y - (1-mu)*y/r1^3       - mu*y/r2^3
```

Jacobi constant for unforced coast: `C = x² + y² + 2(1-mu)/r1 + 2mu/r2 - vx² - vy²`. Integrate adaptively (`solve_ivp`, e.g. DOP853) with declared tolerances, collision stopping events, and careful event-direction selection.

Use a documented Earth–Moon distance and gravitational parameter convention; report scale conversions in every artifact. Near-body circular parking orbits in CR3BP are approximations, not exact two-body orbits; convert Earth-centric *inertial* tangential velocity correctly into rotating-frame velocity before injection.

## Small, precise initial challenge

1. Place a spacecraft on an idealized Earth-centered circular-parking initialization (configured radius, phase) and apply **one** departure impulse. Then coast, unpowered.
2. Search over a small bounded vector of departure parameters (e.g. Earth-relative departure phase and two impulse components). **Freeze the actual parameterization and bounds before optimizer comparison.**
3. A candidate is feasible if it: leaves the Earth departure vicinity, passes a predeclared Moon approach region, avoids colliding with either body, and makes a subsequent **inbound** crossing of an Earth encounter shell inside a fixed flight-time window.
4. Record minimum Moon distance, return shell velocity and encounter geometry, time of flight, departure `delta-v`, and numerical health. Do not describe an arbitrary Earth-shell crossing as safe atmospheric re-entry or closed-orbit insertion.
5. Guard against false returns: don't count the initial departure shell crossing, a near-Earth transient, or a collision as mission completion. Moon encounter must precede qualifying return.

**First gate:** produce and independently validate at least one physically coherent candidate under this model. Find a reliable reference solution or use a reproducible numerical shooting procedure; do not claim one exists for arbitrary initial conditions or tolerances. If a full free-return is too narrow, formally revise the target to a *lunar encounter then Earth vicinity return* and label the change honestly.

## Stage A — validate physics BEFORE optimization

- Check Earth/Moon primary placement, nondimensional conventions, circular-frame transforms.
- Confirm Jacobi drift stays under a declared tolerance on representative noncollision coast segments. Compare two tolerance sets or an independent integrator.
- Test event direction, repeatability, nonfinite states, collision detection and horizon termination.
- Visualize at least one validated reference orbit/trajectory; compare numerical outputs, not screenshots alone.
- Include unit tests for conversion to kilometers, days, and km/s.

## Stage B — methods and baselines

Compare at equal propagation/evaluation budgets:

- **B1 (required):** multi-start local optimization / shooting with penalized feasibility objective.
- **B2 (recommended):** differential evolution (or another credible population search baseline).
- **T (candidate):** MAP-Elites / quality-diversity archive search.

Choose MAP-Elites descriptors with **scientific meaning** and variation in feasible trajectories, provisionally `(lunar minimum distance, time of flight)`; avoid picking descriptors after seeing results. An archive cell stores one best *feasible* candidate under the declared objective; report infeasible samples separately. Compare with a descriptor-binned archive of B1/B2 outputs too, otherwise diversity of MAP-Elites may be a trivial artifact of its data structure.

## Metrics and statistics

- Fraction of random seeds yielding at least one feasible return; feasible evaluations / total integrations.
- Best departure `delta-v`, time of flight, and return-state constraints among feasible candidates (show Pareto tradeoffs where appropriate).
- Descriptor coverage, archive quality, distinct physically meaningful families (don't equate adjacent grid cells with independent families).
- Robustness on **held-out**, fixed perturbations: departure impulse magnitude/direction, initial phase (specify distributions, e.g. percent and degrees) and numerical integration settings. Fraction of trajectories retaining encounter/return feasibility.
- Compute budget by ODE propagations, plus runtime and hardware. Multiple independent seeds; sensitivity to feasibility tolerances.

**Expected nontrivial outcomes:** MAP-Elites may create a more transferable catalog at extra cost; multi-start may match it; most discovered nominal solutions may collapse under perturbations. All are interpretable. Do not require a 'positive' result.

## Interactive scientific visualization

- Earth–Moon synodic 2D animation with selectable *validated* trajectories and synchronized time slider.
- Accurate physical scale, viewer orbit trails, event markers and labels for departure, Moon approach, return shell.
- Trajectory archive/coverage plot; select a point/cell to show the exact orbit and numerical evidence.
- Perturbation robustness exploration uses precomputed physically simulated variants, not visual jitter masquerading as dynamics.
- Method-comparison charts show budget, seeds, mean/median/dispersion, failed solutions; report link includes exact exported data.

## Acceptance / kill gates

**M2 complete:** a tested integrator; documented physical assumptions; Jacobi and event checks; at least one independently validated trajectory/return candidate *or* an explicit negative feasibility report and revised task. No fake data.

**M3 complete:** matched-budget B1 versus T (and ideally B2), predeclared metrics, several seeds, robustness tests, published interactive report and reproducible configurations.

**Stop or revise:** if baseline already yields sufficient catalog diversity and robustness at much lower cost, publish the negative finding and stop; do not bolt on RL to rescue the narrative.

## Relevant literature / reference starting points

- CR3BP equations and examples: https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
- Mouret & Clune, *Illuminating search spaces by mapping elites* (2015): https://arxiv.org/abs/1504.04909
- Nunes et al., *Designing trajectories in the Earth–Moon system: A Levenberg–Marquardt approach* (2026, higher fidelity and CR3BP transitions): https://doi.org/10.1016/j.actaastro.2026.07.020

This literature is **background, not proof of a research gap**. Do a focused nearest-neighbor literature search and cite specific related work before finalizing the study protocol.
