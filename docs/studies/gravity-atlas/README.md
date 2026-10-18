# Gravity Atlas — Active Research Specification (v0)

**Theme:** Dynamical systems, trajectory optimization, quality-diversity, robustness.  
**Status:** first study; experiment design provisional until the feasibility milestone completes.  
**RL required?** No. Explicitly *not* part of the first comparison.

## Nearest prior work

See [`literature.md`](literature.md) (S3 scan, 2026-10-18). In short: no quality-diversity / MAP-Elites work on spacecraft trajectories was found. The closest work generates diverse CR3BP solutions with generative models (Beeson, Li & Sinha 2024 and follow-ups) or builds robustness into single solutions (Sinha & Beeson 2025). Classical free-return families come from continuation (Pavlak 2010; Jesick & Ocampo 2011). Still to read before the protocol is frozen: the 2026 Du & Li "controllable set" papers (paywalled).

## Research question (D16, adopted 2026-10-18)

> At matched propagation budgets, does a MAP-Elites archive of planar CR3BP free returns contain more solutions that remain feasible under held-out injection errors (with a declared correction budget) and under held-out constraint changes than descriptor-binned multi-start / monotonic basin hopping / differential evolution, a continuation-family catalog, and a dense quasi-random map?

Robustness is measured as two separate quantities (D17): *execution-error tolerance* and *changed-constraint availability*. Integrator-setting sensitivity is a numerical-validity check only. Publication target: arXiv preprint + interactive site (D18).

The terms **free-return-like** and **robust** require precise operational definitions below. Do not imply NASA mission certification or publish novelty claims in advance.

<details><summary>Original handoff question (superseded)</summary>

> Does maintaining a diverse archive of feasible Earth–Moon free-return-like trajectories improve the availability of useful, robust solutions when mission constraints change, relative to strong multi-start conventional optimization under matched compute budgets?

</details>

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

Baselines adopted in D16 (supersede the list below where they differ): dense quasi-random map of the decision space (coverage ground truth and budget-matched baseline), descriptor-binned multi-start, monotonic basin hopping, differential evolution, and a continuation-family catalog. The generative baseline is deferred.

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

**M1 + M3 complete:** a tested integrator; documented physical assumptions; Jacobi and event checks; at least one independently validated trajectory/return candidate *or* an explicit negative feasibility report and revised task. No fake data.

**M4 complete:** matched-budget B1 versus T (and ideally B2), predeclared metrics, several seeds, robustness tests, published interactive report and reproducible configurations.

**Stop or revise:** if baseline already yields sufficient catalog diversity and robustness at much lower cost, publish the negative finding and stop; do not bolt on RL to rescue the narrative.

## Relevant literature / reference starting points

- CR3BP equations and examples: https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
- Mouret & Clune, *Illuminating search spaces by mapping elites* (2015): https://arxiv.org/abs/1504.04909
- Nunes et al., *Designing trajectories in the Earth–Moon system: A Levenberg–Marquardt approach* (2026, higher fidelity and CR3BP transitions): https://doi.org/10.1016/j.actaastro.2026.07.020

This literature is **background, not proof of a research gap**. Do a focused nearest-neighbor literature search and cite specific related work before finalizing the study protocol.

## M1 implementation notes

Code: `src/phasespace/gravity/` (`system`, `dynamics`, `units`, `propagation`, `periodic`, `reference`, `export`); tests: `tests/gravity/`.

**Constants.** The system is always an explicit `EarthMoonSystem` argument.

- Project default `EARTH_MOON`: GM_earth = 398600.435507 and GM_moon = 4902.800118 km³/s² (JPL DE440, https://ssd.jpl.nasa.gov/astro_par.html), length unit 384 400 km (NASA NSSDCA Moon Fact Sheet semimajor axis), time unit derived as `sqrt(L³/(GM_earth+GM_moon))` = 375 190.26 s, so mu = 0.0121505844 and the velocity unit is 1.024547 km/s. Collision radii are the NSSDCA volumetric mean radii, 6371.0 km and 1737.4 km.
- `JPL_EARTH_MOON` reproduces the JPL Three-Body Periodic Orbits catalog exactly: mu = 1.215058560962404e-2, lunit = 389 703.264829278 km, tunit = 382 981.289129055 s, Moon radius 1737.1 km (Earth radius: 6371.0 km, which the catalog does not give). Its lunit is not 384 400 km. Its mu differs from the DE440 ratio by about 1e-7 relative, and lunit³/tunit² matches the DE440 GM sum to better than 1e-8.

**Reference.** Three Earth–Moon L1 Lyapunov orbits (rows 2400, 2600 and 2800, Jacobi 3.097 to 3.178, stability indices 535 to 1215) from https://ssd-api.jpl.nasa.gov/periodic_orbits.api?sys=earth-moon&family=lyapunov&libr=1, fetched 2026-10-18 and stored verbatim in `tests/gravity/fixtures/jpl_earth_moon_l1_lyapunov.json` together with the SHA-256 of the full response. Tests never use the network.

**Numerics and thresholds** (normalized units; a check passes when `observed <= threshold`):

- Propagation uses DOP853 with rtol = atol = 1e-12. The comparison set uses 1e-10. Differential correction uses 1e-13 and stops at |vx| < 1e-11 at the half-period crossing.
- Collision events are terminal and fire only on inward crossings (direction -1). Termination is one of `horizon`, `earth_collision`, `moon_collision`, `non_finite`, `evaluation_limit` (1e6 RHS evaluations by default) or `solver_failure`.
- Periodicity after one period: 1e-9. Observed values are 2e-11 to 9e-11 for the JPL states and 1.6e-11 for the export. These orbits amplify errors roughly by their stability index (about 10³) per period, so 1e-9 leaves a 10–60× margin over the observed error and is still far below any visible effect.
- Jacobi drift: 1e-10 over 3 periods. Observed values are at most about 2.5e-11 for the fixtures and 2.8e-12 for the export. The run stops at 3 periods because, by about the 4th period, the unstable orbit has amplified errors enough to leave the orbit.
- Agreement between the two tolerance sets: 1e-7 on the final state after one period (observed about 2e-9).
- Agreement with JPL: |Δvy0| and |Δperiod| ≤ 1e-9 in the export and 1e-10 in the tests, |ΔC| ≤ 1e-10. Observed differences are 4e-14, 7e-14 and 2e-14. The catalog lists 15–17 significant digits. The orbit is computed without catalog velocities, from a linear L1 guess plus 10-step continuation in x0; only x0 is taken from the catalog, to select the family member.
- L1, L2 and L3 positions agree with the catalog to 1e-14 and with an independent `numpy.roots` solution of the classical quintics to 1e-12.

**Reproduce.** `just verify` (all checks, about 2 s) and `just gravity-reference` (or `uv run python -m phasespace.gravity.export --out runs/<run_id>`). The export writes `manifest.json` and `trajectory.json` (schema v1, 1001 samples at a uniform step of T/1000, about 0.30 h) into a new directory and refuses to reuse an existing one. Its status is `validated` only if every recorded check passes. `source.git_dirty` records uncommitted changes, so regenerate after committing for a clean provenance record.

**Not yet validated against an independent reference:** L2 Lyapunov orbits (the only check is self-consistent periodicity) and propagation near the primaries other than the collision-event geometry.
