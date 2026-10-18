# Gravity Atlas M3: free-return feasibility design (proposal)

**Status:** accepted by the user on 2026-10-18 (D20). It is not a frozen protocol; the M4 protocol freezes bounds, descriptors and budgets. Every number below is either a proposal (with its reason) or a measurement on this machine (labelled "measured"). The M4 protocol (`study-protocol`) freezes the bounds, descriptors and budgets. Nothing here is M3 evidence yet.

**Scope (roadmap M3):** define the free-return event exactly, find and validate at least one qualifying candidate (or publish a negative note), export it in the M1 format, and produce the dense map that D16 needs as coverage ground truth.

## 1. System constants

**Recommendation: `EARTH_MOON`** (DE440 GMs, 384 400 km length unit, mean radii 6371.0 and 1737.4 km; `system.py`).

- Every M3 quantity is physical: parking and perilune altitudes, entry-interface altitude, Δv in km/s, days. These are compared with mission-scale values. `EARTH_MOON` uses the NSSDCA mean Earth–Moon distance, and its constants are documented.
- `JPL_EARTH_MOON` exists only to reproduce the periodic-orbit catalog. Its length unit (389 703 km) is 1.4 % larger than 384 400 km. The same normalized orbit would then map to different km altitudes, and M3 has no catalog orbit to reproduce.
- Every artifact records `length_scale_convention = "nssdca_lunar_semimajor_axis"`, as M1 already does.

## 2. Departure model

**Parking orbit:** circular, about Earth, **altitude 185 km** (r₀ = 6556 km).

- Pavlak (2010, §3.3.3) constrains his whole planar free-return family to an Earth departure altitude of 185 km and a flight-path angle of 0° (verified in the full text).
- Apollo 11 entered a 100.7 × 99.2 nmi parking orbit (≈ 186 × 184 km). Source: Apollo 11 Mission Report (MSC-00171), §3; the report states that distances are in nautical miles.
- The parking orbit is used only to generate the TLI state at t₀. It is never propagated, so the CR3BP non-closure of a "circular" orbit plays no role.
- At r₀, the lunar tidal acceleration relative to Earth's gravity is ≈ 2μ r₀³/(1−μ) ≈ 1.2e-7. This is computed, not cited.

**Construction** (new helper, e.g. `phasespace.gravity.departure.tli_state`). Without loss of generality t₀ = 0, because the CR3BP is autonomous. At t₀ = 0 the rotating axes coincide with the inertial axes, so the departure phase θ *is* the departure epoch relative to the Earth–Moon line. The construction:

1. Earth-relative position: `r = r₀ (cos θ, sin θ)`, normalized, with θ measured counterclockwise from +x (Earth→Moon).
2. Earth-relative inertial velocity before the burn: posigrade, `v_c = sqrt((1−μ)/r₀)` along `t̂ = (−sin θ, cos θ)`. This is the same sense as the Moon's motion, as for Apollo and Pavlak's "prograde departures".
3. Impulse: `Δv (cos α · t̂ + sin α · r̂)`, where α is the impulse angle from local horizontal.
4. Rotating-frame state: `units.inertial_to_rotating(state, t=0.0, mu, origin="earth")`. This applies v_rot = v_in − ω × (r − r_E). The Earth-centered frame is pseudo-inertial (`units.py` docstring), and the instantaneous transform is exact.

A unit test checks the round trip and checks |v_in| = v_c + Δv for α = 0 against a hand-written formula, not the inverse function.

**Parameterizations.** The rationale for the bounds:

- 3.00 km/s is about 130 m/s below the two-body Δv to an apogee at lunar distance. That value is ≈ 3.137 km/s (computed).
- 3.25 km/s is above two-body escape from 185 km, 3.230 km/s (computed).
- Apollo 11's TLI Δv was 10 441.0 ft/s = 3.182 km/s (Mission Report Table 7-III; a finite burn).
- θ covers the full circle, with no assumption about where the solutions lie.

| | Variables | Bounds (proposed) | Role |
|---|---|---|---|
| **2-D (primary for M3)** | θ, Δv; α ≡ 0 (tangential) | θ ∈ [0°, 360°), Δv ∈ [3.00, 3.25] km/s | find and validate the first candidate; 2-D map figure |
| **3-D (primary for M4)** | θ, Δv, α | as 2-D, plus α ∈ [−10°, +10°] | dense map and the optimizer comparison |

**Why 3-D for M4:** in 2-D the provisional descriptors (perilune altitude, time of flight) are locally almost a one-to-one function of (θ, Δv). Each descriptor cell then holds essentially one solution, and "quality within a cell" is empty, which makes a QD archive trivially equal to a dense map. The third variable leaves a one-parameter set of solutions per cell for the objective to choose from. The α bound is a design choice, not a feasibility limit: the pilot found returns across the whole ±10° range.

## 3. Feasibility predicates

**Constants (proposed):**

- Perilune band `H_P = [100, 30 000] km`: Pavlak's planar family spans perilune altitudes "from 100 to 30,000 kilometers".
- Return shell `R_S = 6371.0 + 121.92 = 6492.92 km`: Apollo entry interface at 400 000 ft (Mission Report Table 7-VII), taken over a spherical Earth of mean radius. This is an *entry-interface-like* shell, not a re-entry model.
- Flight-time window `t_ret ≤ T_MAX = 10 d` (= 2.30283 normalized). Pavlak gives one-way times of 2.86–4.91 d across the family (≈ 5.7–9.8 d round trip). Apollo 11's post-TLI free return reached entry interface 6.64 d after TLI (Table 7-IV). Propagation horizon = T_MAX.

**Propagation:** `propagate(s0, T_MAX, EARTH_MOON, solver=DOP853 1e-12, collision=CollisionRadii.of(system), return_shell_km=R_S)`.

`return_shell_km` is a proposed addition (§9). It adds a terminal inward crossing of r₁ = R_S with the same grazing detection as collisions: an Earth periapsis inside R_S is truncated at the first crossing. It is reported as a new termination, `earth_return_shell`. The physical Earth radius stays enabled; it cannot fire first because R_S > R_E.

```python
def classify(res, sys) -> Verdict:            # pure function of a PropagationResult
    if res.termination in {NON_FINITE, EVALUATION_LIMIT, SOLVER_FAILURE}: return fail("numerical")
    if res.termination is MOON_COLLISION:      return fail("moon_collision")
    if res.termination is HORIZON:             return fail("no_return_within_T_MAX")
    assert res.termination is EARTH_RETURN_SHELL     # Earth surface is unreachable before the shell
    t_ret, s_ret = res.t_final, res.final_state
    assert (s_ret[:2] - r_earth) @ s_ret[2:] < 0     # inbound: radial velocity < 0 (event direction -1)
    lunar = [p for p in res.periapses if p.name == "moon_periapsis" and p.t < t_ret]
    if not lunar:                              return fail("return_before_lunar_encounter")
    p = min(lunar, key=lambda p: r2(p.state))  # closest lunar approach before the return
    h_p = r2(p.state) * L_km - R_moon_km
    if not (100.0 <= h_p <= 30_000.0):         return fail("perilune_out_of_band")
    return ok(h_p=h_p, t_moon=p.t, t_ret=t_ret, ...)
```

**Guards against false returns.**

- *Departure shell crossing.* r₀ (185 km altitude) > R_S (121.92 km altitude), so the start lies outside the shell. With α = 0 the start is an Earth periapsis, and r₁ only grows initially. A departure crossing therefore cannot trigger the return event. A unit test asserts `h_park > h_shell`.
- *Near-Earth transient.* With α < 0 the first motion is inward. Such a trajectory can cross R_S before reaching the Moon, and it then fails the "lunar periapsis precedes return" predicate. The Earth periapsis event that fires at t₀ for α = 0 is never used.
- *Collision as completion.* Moon collisions terminate first, and grazing passes are covered by the M1 periapsis events. Earth collision cannot precede the shell.
- *"Lunar encounter".* Only a Moon periapsis inside the band counts. Moon-relative periapses far from the Moon, e.g. near Earth, are excluded by the 30 000 km upper bound. 30 000 km is inside the Laplace sphere of influence, a(μ_M/μ_E)^(2/5) ≈ 66 000 km (computed).
- *Recorded, not constrained:*
  - the number of Moon periapses in the band;
  - the number of Earth periapses between the lunar encounter and the return;
  - the lunar passage sense: sign of the Moon-relative angular momentum at perilune (circumlunar vs cislunar).

**Recorded per trajectory:**

- θ, Δv, α;
- h_p (km), t_moon and t_ret (d);
- at the shell: Earth-relative inertial speed (km/s) and flight-path angle γ_S (deg);
- the osculating two-body vacuum perigee altitude h_vp (km) of the Earth-relative state at the shell;
- Jacobi constant, nfev and termination.

## 4. Objectives and descriptors

**M3 local targeting (2-D, 2 equations, 2 unknowns).** The residual is

F(θ, Δv) = [(h_p − 1000 km)/1000 km, (h_vp − 38 km)/10 km].

- h_vp is used instead of γ_S because it stays defined and continuous when the trajectory just misses the shell. When the trajectory does not reach the shell, h_vp is evaluated at the first post-lunar Earth periapsis.
- 38 km is the vacuum perigee our spherical model gives for Apollo 11's entry state: 11.032 km/s, −6.48° at 400 000 ft, from Table 7-VII (computed). Apollo targeted the same quantity: a nominal transearth perigee altitude of 20.4 nmi = 37.8 km (§7.4.8).
- h_p* = 1000 km is a round, in-band value of the same magnitude as Apollo 11's post-TLI pericynthion: predicted 718.9 nmi = 1331 km, achieved 896.3 nmi = 1660 km.
- The target is a declared design point. It does not reproduce Apollo.

**M4 (provisional, to be frozen in the protocol before any comparison):**

- *Descriptors:* log₁₀ h_p on [log₁₀ 100, log₁₀ 30 000] and t_ret on [5, 10] d. These are the README's provisional pair, with bounds taken from the predicates above.
- *Objective:* |γ_S − γ*| with γ* = −6.5° (equivalently |h_vp − 38 km|). Δv is a recorded secondary.
- Minimizing Δv would push every elite toward α ≈ 0, i.e. back onto the 2-D family, which would make the 3-D archive degenerate.
- The entry corridor (γ*, its width) and the perilune band are natural *changed constraints* for D17.

## 5. M3 search and validation procedure

All propagation uses DOP853 with rtol = atol = 1e-12 and max_evaluations = 1e6, the M1 run settings. A looser screening tolerance is not worth it: 1e-10 was only 1.4× faster (measured, §6).

| Stage | What | Propagations | Wall time (10 workers, measured rate) |
|---|---|---|---|
| S1 | Deterministic 2-D grid: θ step 0.5° (720) × Δv step 2 m/s (126); `random_seed = null` | 90 720 | ≈ 6–8 min |
| S2 | From the 10 feasible grid points with the smallest ‖F‖: `scipy.optimize.least_squares` (trf, box bounds, forward differences); cap 200 propagations per start | ≤ 2 000 | < 1 min |
| S3 | Validate each converged candidate (checks below) | ≈ 6 each | seconds |
| S4 | 3-D scrambled Sobol map, 2¹⁷ points, explicit seed: feasible fraction and descriptor coverage for the M4 go decision | 131 072 | ≈ 9–11 min |

**Validation checks.** Each check passes when `observed ≤ threshold`, in the M1 `Check` style. The observed values in the table come from a pilot sample (h_p ≈ 293 km, t_ret ≈ 5.8 d).

| Check | Threshold (proposed) | Observed on pilot sample |
|---|---|---|
| All predicates hold under the run, tight (DOP853 1e-13) and independent (LSODA 1e-13) integrations, with margins ≥ 1 km to the h_p band edges and ≥ 1 h to T_MAX | boolean, all three | — |
| \|Δh_p\| vs tight / vs LSODA | 10 m | 1.2 mm / 2.4 mm |
| \|Δt_ret\| vs tight / vs LSODA | 0.1 s | 3e-5 s / 7e-5 s |
| Shell-crossing position difference vs tight / vs LSODA | 10 m | 3.7 cm / 7.6 cm |
| Max \|C(t) − C(0)\| over ≥ 10⁵ dense samples | 1e-9 | 3.5e-10 (1e-12 run), 4e-11 (1e-13), 1.4e-10 (LSODA) |
| Backward DOP853 1e-13 integration from the shell state to t = 0: max \|Δs₀\| | 1e-7 | 4.4e-9 (7.5 mm) |
| Perilune from brute dense sampling of r₂(t) (≥ 10⁵ points + bounded scalar minimization) vs the event | 1 m | not yet run |
| min r₁ on [0, t_ret) > R_S and min r₂ > R_moon by dense sampling (no missed earlier crossing) | boolean | not yet run |

The M1 Jacobi threshold of 1e-10 is **not** usable here. Near the departure point the terms of C are ≈ 2(1−μ)/r₁ ≈ 116, so 1e-9 is ≈ 1e-11 relative. The run integration at 1e-12 already reaches 3.5e-10. As in M1, Jacobi drift is a conservation check only; accuracy rests on the integrator agreement and reversibility checks.

**Plausibility against Apollo 11 (not validation).** Apollo 11 differs in four ways: a 3-D ephemeris, a finite burn, an oblate Earth, and a different epoch distance. Agreement to within several percent would be expected, and a mismatch flags a modeling error rather than disproving anything. The Apollo 11 post-TLI values (Mission Report Tables 7-III and 7-IV):

- Δv 3.182 km/s;
- pericynthion 3.01 d after TLI;
- entry interface 6.64 d after TLI, at 10.996 km/s with a −64.06° flight-path angle (the pure free return before midcourse corrections).

**Pilot observations (exploratory, uncontrolled, not M3 evidence).** These come from throwaway scripts run on 2026-10-18, with uniform random samples, a 12 d horizon, and the shell implemented by temporarily setting the Earth collision radius to R_S.

- Shell returns after a lunar periapsis in [0, 20 000] km made up 0.10 % of the full 2-D box (n = 4000) and 6.8 % of a focused box, θ ∈ [215°, 245°] × Δv ∈ [3.12, 3.20] km/s.
- In 3-D (α ∈ ±10°, θ ∈ [200°, 260°], Δv ∈ [3.10, 3.25] km/s, n = 8000) the fraction was 1.9 %.
- Returns with h_p of 240–990 km had Δv of 3.16–3.18 km/s, t_moon of 2.6–3.0 d, t_ret of 5.8–6.3 d, shell speed of 11.0 km/s and γ_S between −17° and −89°. These magnitudes match Apollo 11 and Pavlak, so S1–S3 are expected to succeed. This expectation must not replace them.

## 6. Dense quasi-random map: measured cost

**Setup (measured 2026-10-18):**

- Machine: Apple M4 Pro, 12 cores (8 performance + 4 efficiency).
- Software: Python 3.12.12, NumPy 2.5.4, SciPy 1.18.1.
- Propagation: existing `propagate`, DOP853 1e-12, 12 d horizon, Moon and shell events.
- Scripts: `/private/tmp/claude-501/-Users-vitalyvorobyev-nonvision-policylab/52cc2435-3873-458c-a6aa-6ae6d604ae99/scratchpad/m3/timing.py` and `/private/tmp/claude-501/-Users-vitalyvorobyev-nonvision-policylab/52cc2435-3873-458c-a6aa-6ae6d604ae99/scratchpad/m3/pilot.py`.

**Results:**

- Serial, n = 300, full 2-D box: mean **34.7 ms** per propagation (median 26.5, p95 84, max 133 ms). Mean nfev was 3053, ≈ 11.4 µs per right-hand-side evaluation.
- Serial, n = 20: 43 ms mean at 1e-12 and 30 ms at 1e-10.
- `ProcessPoolExecutor`, 10 workers, n = 4000–8000: an effective **3.9–5.1 ms** of wall time per propagation. The per-call mean rose to 37–49 ms from contention and the efficiency cores.

| Samples | Serial (35–45 ms) | 10 workers (3.9–5.1 ms) | Sobol size |
|---|---|---|---|
| 10⁴ | 6–8 min | 40–50 s | 2¹⁴ = 16 384 |
| 10⁵ | 1.0–1.3 h | 6.5–8.5 min | 2¹⁷ = 131 072 |
| 10⁶ | 10–13 h | 65–85 min | 2²⁰ = 1 048 576 |

The 10 d horizon makes these estimates slightly conservative.

**Recommendation:**

- M3 runs 2¹⁷ in 3-D (S4).
- M4's ground-truth map is 2²⁰ in the frozen 3-D box, about 1.5 h on 10 workers. It should be at least 8× larger than the largest per-method budget, so that ground truth is denser than any method.
- Rust (D14) is not needed at this scale.
- Results are stored as scalars per sample, ≈ 100 B each, so 2²⁰ samples take ≈ 100 MB in `.npz`.

## 7. Go/no-go for M3

**GO (proceed to the M4 protocol) when all of these hold:**

1. At least one candidate passes every §5 check.
2. Re-running the one-command export reproduces its observables within the same thresholds.
3. The S4 map has feasible samples in at least 10 cells of a provisional 10 × 10 descriptor grid, so that there is something to illuminate.
4. The S1–S4 budget stayed within 2.5 × 10⁵ propagations.

**Honest negative outcomes**, each published as a short note with the S1 termination map, the smallest ‖F‖ reached, and a statement of the checks that failed:

- (a) *No candidate:* no feasible point in S1, or no S2 start converges. The task is then revised, as the README allows, to "lunar encounter, then Earth-vicinity return". The return shell would be enlarged (proposal: r₁ = 50 000 km) and the change labelled.
- (b) *Candidates exist but fail integrator agreement:* the results are numerically unreliable, and nothing is exported as `validated` until this is understood.
- (c) *The feasible set is too thin* (criterion 3 fails): a QD comparison would be trivial. The question must be revised (D07) before M4. For example, widen α or the band, or add a second impulse. No synthetic or interpolated "returns" are allowed under any outcome.

## 8. Exported artifacts and the data contract

**Candidate runs:**

- One run per validated candidate, e.g. `gravity-free-return-c1-v1`: `manifest.json` + `trajectory.json` (schema v1), with tight and LSODA runs recorded in `solver.comparisons`.
- Uniform-time sampling under-resolves the 11 km/s near-Earth arcs. Proposal: export the union of the accepted solver steps and a uniform grid of 2000 points.
- Proposed metric keys:
  - `departure_delta_v_km_s`, `departure_phase_deg`, `impulse_angle_deg`, `parking_altitude_km`;
  - `perilune_altitude_km`, `lunar_encounter_time_days`;
  - `return_shell_crossing_time_days`, `return_shell_speed_km_s`, `return_flight_path_angle_deg`, `return_vacuum_perigee_altitude_km`;
  - `jacobi_constant`.

**Map run:**

- e.g. `gravity-free-return-map-3d-sobol17-v1`, with `random_seed` set to the Sobol seed.
- The full table stays local under `runs/`. Only a binned summary (feasible fraction per descriptor cell) is published to `web/public/research/`.

**Contract changes needed** (proposed, not made):

1. `MetricUnit` lacks angles. Add `"deg"`.
2. `Solver` needs a `return_shell` block (`radius_km`, `function: "distance_minus_radius"`, `direction: "decreasing"`, `terminal: true`, same `grazing_detection`). The manifest also needs a `termination` field, so that "horizon" and "return" cannot be confused.
3. `Trajectory` needs an optional `events: {name, t, state}[]` array (departure, perilune, return-shell crossing) for the viewer's event markers, which the README requires.
4. A `Cr3bpModel.departure` block, or an experiment-config artifact: parking altitude, sense, parameterization, bounds and predicates. This is the M4 "protocol artifact" from the contract.
5. A new `Artifact.kind`, e.g. `"sample_table"` (columnar JSON: column names, units, rows), for map summaries. `.npz` is used only for unpublished local data.

## 9. Code additions M3 needs (small, before S1)

1. `departure.tli_state(...)` (§2), with tests.
2. `propagate(..., return_shell_km=None)`, reusing the `_Body` collision and periapsis machinery, plus `Termination.EARTH_RETURN_SHELL`. Tests: an inbound crossing fires, an outbound crossing does not, and a grazing dip below the shell between steps is caught.
3. **Robustness bug found while measuring.**
   - Observed with SciPy 1.18.1: `propagate` raises an uncaught `ValueError("f(a) and f(b) must have different signs")` from scipy's event root-finding. It happens with LSODA at rtol = atol = 1e-12 on a tangential departure, where the Earth periapsis function is exactly 0 at t₀ and the Earth events are enabled.
   - DOP853 and α ≠ 0 did not trigger it.
   - Fix: map exceptions from event location to `solver_failure` (or a dedicated reason) and add a regression test. One uncaught exception would otherwise abort a 10⁶-sample batch.
4. `free_return.classify` (§3), an export CLI, and `just` recipes. Use a deterministic ordered parallel map with a fixed chunk order, so results are independent of worker scheduling.

## 10. Open decisions for the user

1. **Constants:** `EARTH_MOON` (DE440 / 384 400 km) vs `JPL_EARTH_MOON`. *Recommend `EARTH_MOON`.*
2. **Parameterization:** 2-D tangential (θ, Δv) for M3 and 3-D (θ, Δv, α ∈ ±10°) for M4, vs 3-D throughout. *Recommend 2-D for M3, 3-D for M4*, because the 2-D descriptor cells collapse to single solutions.
3. **Feasible set:** perilune band [100, 30 000] km with t_ret ≤ 10 d (Pavlak's family range), vs a narrower Apollo-like [100, 5 000] km. *Recommend the Pavlak range*, which leaves the descriptors room to vary.
4. **Return shell:** entry-interface-like R_E + 121.92 km, with γ_S recorded but not constrained in M3, vs a larger "Earth-vicinity" shell. *Recommend 121.92 km.* The 50 000 km shell is kept only as the declared fallback in §7a.
5. **M4 objective:** entry-angle error |γ_S + 6.5°| (equivalently vacuum perigee vs 38 km) vs departure Δv. *Recommend the entry-angle error*, with Δv recorded. Minimizing Δv collapses the 3-D archive onto α ≈ 0.

## Sources used here

- Pavlak (2010) thesis, §3.3.3, full text read 2026-10-18 (already in `literature.md`).
- NASA, *Apollo 11 Mission Report*, MSC-00171 (1969): https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf, downloaded 2026-10-18. Values used: parking orbit (§3); Tables 7-III and 7-IV (TLI Δv, pericynthion, free-return entry conditions); §7.4.8 (perigee targeting); Table 7-VII (entry interface).
- Unit conversions: 1 nmi = 1.852 km; 1 ft = 0.3048 m.
