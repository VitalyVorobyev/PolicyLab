# Gravity Atlas M3: free-return feasibility design (proposal)

**Status:** accepted by the user on 2026-10-18 (D20). **M3 complete: GO (2026-10-18)**, on clean-commit evidence (commits `9a721b0` and `ccf326e`; see "M3 results" at the end). Sections 1–10 are the design as accepted, and every number in them is either a proposal (with its reason) or a pilot measurement (labelled "measured"); the evidence is in "M3 results". This is not a frozen protocol: the M4 protocol (`protocol.md`, `study-protocol`) freezes the bounds, descriptors and budgets.

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
- Our spherical model gives a vacuum perigee of 38.5 km for Apollo 11's entry state: 11.032 km/s, −6.48° at 400 000 ft, from Table 7-VII (computed; corrected from "38 km" during M3). Apollo targeted the same quantity: a nominal transearth perigee altitude of 20.4 nmi = 37.8 km (§7.4.8). The declared target stays at the round value of 38 km.
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

## M3 results (clean-commit evidence)

**Status: complete, GO (2026-10-18).** Every evidence run below was generated from a clean working tree (`source.git_dirty = false`). Machine: Apple M4 Pro, 12 cores, 10 workers. The machine was shared with unrelated jobs during the runs, so wall times are upper bounds.

| Stage | Evidence run | Commit | Status |
|---|---|---|---|
| S1 2-D grid | `runs/gravity-free-return-map-2d-grid-v1` | `9a721b0` | `computed_unvalidated` (a map; samples are not individually validated) |
| S2 targeting | `runs/gravity-free-return-search-v2` | `ccf326e` | `computed_unvalidated` |
| S3 candidate c1 | `runs/gravity-free-return-c1-v2` | `ccf326e` | `validated` |
| S3 candidate c2 | `runs/gravity-free-return-c2-v2` | `ccf326e` | `validated` |
| S4 3-D Sobol | `runs/gravity-free-return-map-3d-sobol17-v1` (seed 20261018) | `9a721b0` | `computed_unvalidated` |
| Fine Δv probe | `runs/gravity-free-return-dv-probe-v1` | `ccf326e` | `computed_unvalidated` |

The candidate runs `-v2` supersede `gravity-free-return-{search,c1,c2}-v1` (clean, commit `9a721b0`). Commit `ccf326e` added the entry-agreement checks (speed, flight-path angle, vacuum perigee) and the non-gating Δv and perilune margins after the numerics review. The `-v1` candidate manifests lack them. The maps did not need regenerating. `ccf326e` changed no propagation, evaluation or classification code: in `free_return.py` and `free_return_map.py` it added only a docstring, a new helper and the probe design. The S1 table is the input of the `-v2` search.

**Superseded history, not evidence.** The preliminary runs `…-pre1` to `…-pre4` (including `…-pre2-cap1000` and `…-rerun`) were generated from uncommitted trees. `pre1`–`pre3` sit on base commit `3ac073f`, and `pre4` on `9a721b0`. An earlier version of this section cited them. Every scientific number they gave is reproduced exactly by the clean runs. The S1 and S4 sample tables match row for row: same outcomes, h_p, t_ret, h_vp and nfev. The candidate metrics and check values are identical, and `trajectory.json` samples are bit-identical across `-pre4`, `-v1` and `-v2`. The **differences from the earlier text** are listed here. They are transcription and summary errors, not changes in the data:

| Quantity | Earlier text | Clean runs (and identical in the `pre` tables) |
|---|---|---|
| Family (A) Δv range on the S1 grid | 3.146–3.178 km/s | **3.146–3.180 km/s** |
| Family (A) h_p and t_ret | 188 km "up to 30 000 km"; 5.7–10 d | **188–29 118 km; 5.70–9.96 d** |
| Family (B) h_p and t_ret | 28 000–30 000 km; 9.8–10 d | 28 068–29 981 km; 9.83–10.00 d (consistent, now exact) |
| θ columns with 6 feasible Δv cells | θ = 227–231° | **θ = 227–229.5° and 231°** (230° and 230.5° have fewer) |
| Margins of the integrator-agreement checks | "10³–10⁴" | **20–1.2 × 10⁵.** The tightest is the shell-crossing position against LSODA: 0.49 m against 10 m for c1 (factor 20.5) and 0.13 m for c2 (factor 76). The tight-tolerance position check has factors 76 (c1) and 203 (c2). |
| Base commit of the preliminary runs | all `3ac073f` | `pre4` runs: `9a721b0` (dirty) |
| Wall time, 10 workers (machine load differed) | S1 650 s, S4 935 s, probe 106 s | S1 858 s, S4 870 s, probe 29 s |

**Code.** All in `src/phasespace/gravity/`:
- `departure.py`: TLI state, §2.
- `propagation.py`: `return_shell_km` and the terminations `earth_return_shell` / `start_inside_return_shell`, §9.2.
- `free_return.py`: definition, predicates and record, §3–4.
- `free_return_map.py`: S1 grid, S4 Sobol, ordered parallel evaluation, `sample_table`.
- `free_return_search.py`: S2.
- `free_return_validation.py`: S3.
- `free_return_export.py`: CLI and runs.

Tests are in `tests/gravity/test_gravity_{departure,return_shell,free_return,free_return_pipeline,backward_decomposition}.py`. The data contract was extended additively; see "M3 additions" in `docs/data-contract.md`.

### S1: 2-D grid map

Run `gravity-free-return-map-2d-grid-v1`: 720 × 126 = 90 720 tangential departures, one propagation each. 262 381 794 right-hand-side evaluations in total; 858 s wall time.

| Outcome | Count | Share |
|---|---|---|
| Feasible | 161 | 0.177 % |
| No return within 10 d | 64 681 | 71.3 % |
| Return, perilune out of band¹ | 22 281 | 24.6 % |
| Return before lunar encounter | 3 314 | 3.7 % |
| Moon collision | 283 | 0.3 % |
| Earth collision, invalid start, numerical failure | 0 | |

¹ "Perilune out of band" also covers returns with no real lunar flyby. In those, the closest Moon periapsis is only a far Moon-relative distance minimum, for example while the spacecraft is near the Earth.

The maximum Jacobi drift at accepted steps over the feasible samples is 7.5e-10.

**Structure: at least two families at this resolution.** The feasible set is a thin band along the high-Δv edge of the Moon-collision boundary. The 2 m/s grid under-resolves it and probably misses thinner feasible regions altogether.

- **(A) Lunar passage sense −1** (156 grid points):
  - θ = 225.5–247.0°, Δv = 3.146–3.180 km/s;
  - h_p = 188–29 118 km, t_ret = 5.70–9.96 d;
  - 1–6 feasible Δv cells per θ column (6 at θ = 227–229.5° and 231°).
- **(B) Lunar passage sense +1** (5 grid points):
  - θ = 248.0–254.0°, Δv = 3.130–3.132 km/s;
  - h_p = 28 068–29 981 km and t_ret = 9.83–10.00 d, at the band and T_MAX edges.

**Fine Δv probe.** Run `gravity-free-return-dv-probe-v1`: 0.02 m/s steps over Δv = 3.120–3.190 km/s at θ = 229° and 251°, 7 002 propagations.
- At θ = 229°, family (A) is one contiguous band, Δv = 3.15724–3.16860 km/s. That is 569 points, a width of 11.4 m/s, consistent with the 6 grid cells.
- At θ = 251°, family (B) exists only in a window Δv = 3.13078–3.13114 km/s. That is 19 points: 0.36 m/s between the outermost feasible samples, 0.40 m/s between the bracketing infeasible ones.
- The grid's 5 (B) points are therefore aliasing hits on a sub-cell window. The grid cannot measure (B)'s extent.

The D16 coverage ground truth therefore needs adaptive refinement at band edges. The M4 protocol defines it (`protocol.md` §5.3).

### S2: targeting

Starts were the 10 feasible grid points with the smallest ‖F‖. The method was trf with forward differences, x_scale = 1, box bounds and ftol = xtol = gtol = 1e-12. Convergence means max|F_i| ≤ 1e-5 and the point is feasible.

- **Cap deviation, 200 → 1000 (D22).** The design's 200 propagations per start was a proposal that had not been tested on S2. In the preliminary run `gravity-free-return-search-pre2` (dirty tree, 2 000 propagations), 0 of 10 starts converged under that cap.
  - The cause is the problem's geometry, not a missing solution. The grid cannot resolve the h_vp = 38 km valley: ∂h_vp/∂Δv ≈ −1.0e7 km per km/s, so one 2 m/s cell spans about 20 000 km of h_vp.
  - The starts with the smallest ‖F‖ therefore lie on the valley floor but far from h_p = 1000 km. trf then crawls along the curved valley in steps of about 0.03–0.06°.
  - A probe of x_scale = "jac", x_scale = (1, 0.0044), dogbox and the exact tr_solver converged at most 1 of 10 starts.
  - The cap is now 1000 by default (`SearchSettings`, CLI flag `--max-propagations-per-start`, recipe parameter `cap`). Every search run records the cap and the design's 200 in `limitations`.
- **Evidence (`gravity-free-return-search-v2`, cap 1000).** All 10 starts converged, using 257–648 propagations each and 5 355 in total. They reached 2 distinct candidates with max|F| ≤ 5.0e-9. Start ranks 2 and 8 are the exported representatives.

### S3: candidates and validation

| | c1 (`gravity-free-return-c1-v2`) | c2 (`gravity-free-return-c2-v2`) |
|---|---|---|
| θ, Δv | 229.142241°, 3.1567704 km/s | 225.816259°, 3.1780111 km/s |
| h_p (target 1000 km), t_moon | 1000.000 km, 3.060 d | 1000.000 km, 2.607 d |
| t_ret, speed at the shell, γ_S | 6.1217 d, 11.0078 km/s, −6.485° | 6.4311 d, 10.9945 km/s, −6.477° |
| Vacuum perigee (target 38 km) | 38.000 km | 38.000 km |
| Family; lunar passage sense | (A); −1 | (A); −1 |
| Entry sense (Earth-relative inertial angular momentum at the shell) | +1, posigrade | −1, retrograde |
| Δv margin to the nearest infeasible Δv (fixed θ), non-gating | below 9.25 mm/s (beyond: no return, i.e. skip-out); above 11.3 m/s (no return) | below 2.26 m/s (perilune out of band); above 30.8 mm/s (no return) |
| h_p margin to the band edges, non-gating | 900 km / 29 000 km | 900 km / 29 000 km |
| Targeting propagations (representative start) | 644 | 258 |

**Checks passed, for both candidates** (the observed value is the maximum over c1 and c2 and over both references):
- predicates hold, with margins, under the run, DOP853 1e-13 and LSODA 1e-13;
- |Δh_p| ≤ 5.0e-6 km, against a threshold of 1e-2 km;
- |Δt_ret| ≤ 1.8e-4 s, against 0.1 s;
- shell-crossing position difference ≤ 4.9e-4 km, against 1e-2 km. This is the check with the **smallest margin among the agreement checks** (factor 20.5, c1 vs LSODA);
- entry agreement with the tight and LSODA runs:
  - speed ≤ 7.9e-11 km/s, against 1e-6 km/s (1 mm/s);
  - flight-path angle ≤ 2.1e-6°, against 1e-3°;
  - vacuum perigee ≤ 5.5e-5 km, against 0.05 km.

  These checks were added after review (`ccf326e`). The thresholds come from integrator-agreement physics, not from the observed values: Jacobi conservation bounds the speed, and the 10 m crossing-point separation bounds the angle through the local-horizontal rotation and the path curvature. The two-body sensitivities of a near-parabolic entry (≈ 25 km/deg, ≈ 1.2 km per m/s) then bound the perigee. The derivations are in the `free_return_validation.py` constants;
- Jacobi drift over 10⁵ dense samples ≤ 8.0e-10 (run), 8.0e-11 (tight) and 1.4e-10 (LSODA), against 1e-9. **The margin is small:** c1's run passes by only a factor of 1.26 (7.96e-10), c2's by 1.75 (5.71e-10). The M3 threshold is not changed after the fact;
- brute-force perilune vs the event ≤ 2.3e-11 km, against 1e-3 km;
- no dense sample inside the shell or the Moon;
- samples finite and strictly increasing.

Across all the agreement checks the margins (threshold / observed) run from 20.5 to 1.2 × 10⁵. The backward-integration checks pass by factors of 7.9–19 and the Jacobi checks by 1.26–17.

**Backward-integration check: redefined (D21, user decision 2026-10-18).**

*Why the original check was replaced.* Under the original single max-norm criterion of §5 (max|Δs₀| ≤ 1e-7, normalized), both candidates fail: 8.3e-7 (c1) and 6.6e-7 (c2). These values are recorded in the `-v2` manifests. The failure is understood and is not a modelling error:
- The error is dominated by velocity, because |ds/dt| ≈ 3400 at the parking orbit. The 1e-7 threshold therefore amounted to a ~0.1 ms timing test. That is 1000× stricter than the 0.1 s return-time check, while 1e-7 in position (38 m) is looser than the 10 m checks.
- The backward pass from the tight (1e-13) run's shell state gives |Δs₀| = 1.3e-8 (c1). So the raw error is the run's own 1e-12 shell-state error (13 cm, consistent with the 1.3e-4 km tight-agreement check) amplified backward through the flyby.
- The §5 threshold came from a single pilot sample with h_p = 293 km.

*New definition.* The backward DOP853 1e-13 integration starts from the **exported run's** return-shell state, so the check covers the exported trajectory, not the tight one. It runs to t = 0 and gives Δs₀ = s₀,back − s₀. With f = ds/dt at s₀ (normalized rotating frame):
- τ = (Δs₀·f)/(f·f), the time-shift-like component;
- r = Δs₀ − τ f, the remainder.

τ is a projection in the unweighted normalized 4-D metric. At the parking orbit f = (v, a) with |a| ≈ 3.4e3 ≫ |v| ≈ 10.6, so in effect τ ≈ (Δv·a)/|a|², the velocity error along the local, nearly radial acceleration divided by |a|. A pure time shift and such a velocity error cannot be told apart. Position-only errors land almost entirely in the remainder.

Three gating checks replace the max-norm. All three are **reversibility bounds, not accuracy bounds on the observables**. Departure errors are amplified through the flyby: 1 mm/s at departure moves h_p by ≈ 0.6 km and h_vp by ≈ 10 km at the candidates' sensitivities. Observable accuracy rests on the forward agreement checks.

| Check | Threshold | Meaning | c1 | c2 |
|---|---|---|---|---|
| \|τ\| | 1 ms | Admits a velocity error along a of up to \|a\|·1 ms ≈ 9 mm/s, or the equivalent time shift; 100× below the 0.1 s return-time threshold | 0.127 ms | 0.096 ms |
| \|r_pos\| | 1 m | Departure-position reversibility | 6.2 cm | 10.5 cm |
| \|r_vel\| (rotating frame) | 1 mm/s | Departure-velocity reversibility transverse to a; ≈ 1e-7 of the departure speed | 0.053 mm/s | 0.089 mm/s |

The old criterion stays in each manifest as the non-gating metric `superseded_check_backward_max_abs_initial_state_error`, with its threshold and verdict. The backward diagnostics from the tight run are also kept as metrics. The decomposition is tested against an independent RK4 time shift with a known transverse offset (`tests/gravity/test_gravity_backward_decomposition.py`).

**Result.** All checks pass for both candidates, and `gravity-free-return-c1-v2` and `gravity-free-return-c2-v2` are `validated` from clean commit `ccf326e`.

**Thin margins (disclosed).**
- c1 sits 9.25 mm/s in Δv from skip-out.
- c2 sits 30.8 mm/s from a no-return edge.
- c1's Jacobi gate passes by a factor of only 1.26.
- The tightest integrator-agreement check (shell-crossing position vs LSODA) passes by a factor of 20.5.

**M4 protocol must-freeze.** Both items are defined in the M4 protocol draft (`protocol.md` §5.2 and §5.3) and must be frozen before any M4 data is produced:
- a scaled Jacobi criterion, for example relative to the magnitude of the terms of C along the trajectory or tied to integrator tolerance, instead of the absolute 1e-9;
- adaptive band-edge refinement for the coverage ground truth.

**Plausibility vs Apollo 11 (not validation).**

| | c1 | Apollo 11 |
|---|---|---|
| Δv | 3.157 km/s (−0.8 %) | 3.182 km/s |
| Pericynthion time | 3.06 d | 3.01 d |
| Entry interface | 6.12 d (−7.8 %) | 6.64 d |
| Entry speed | 11.008 km/s | 10.996 km/s |

γ_S was targeted at the Apollo entry value through h_vp, so it is not comparable with Apollo's pure free-return −64°. The magnitudes agree to within several percent, as §5 expected.

**Reproducibility (criterion 2).** Three independent exports of the candidates agree bit-for-bit in metrics, check values and `trajectory.json` samples: `-pre4` (dirty), `-v1` (clean, `9a721b0`) and `-v2` (clean, `ccf326e`). The search tables of `search-v1` and `search-v2` agree in every numeric column. The clean S1 and S4 tables agree row for row with their dirty-tree predecessors.

### S4: 3-D Sobol map

Run `gravity-free-return-map-3d-sobol17-v1`: 2¹⁷ = 131 072 points, scrambled Sobol with seed 20261018. 404 346 727 right-hand-side evaluations in total; 870 s wall time (6.6 ms per propagation, effective, with 10 workers on a shared machine).

- 243 feasible (0.185 %), 0 numerical failures. The other outcomes: 89 741 no return, 35 704 perilune out of band, 5 024 return before lunar encounter, 360 Moon collisions.
- Feasible θ = 220.7–256.8°, Δv = 3.138–3.204 km/s, with α across the whole ±10° range (−9.95° to +9.99°).
- h_p = 143–29 997 km, t_ret = 5.68–9.99 d, t_moon = 2.54–5.56 d.
- Passage sense: 238 at −1, 5 at +1.
- Max Jacobi drift at accepted steps over the feasible samples: 4.3e-10.

The feasible samples occupy **27 of 100 cells** of the provisional 10 × 10 descriptor grid. The grid is log₁₀ h_p on [2, log₁₀ 30 000] and t_ret on [5, 10] d in equal bins, computed by a one-off binning of `samples.json`; the M4 protocol freezes the binning. Coverage is concentrated along a curved h_p–t_ret ridge. Facts the M4 protocol uses, from the same one-off analysis:
- The nested Sobol prefixes occupy 9, 16, 20, 27 and 27 cells at 2¹³, 2¹⁴, 2¹⁵, 2¹⁶ and 2¹⁷ samples. A 20 × 20 grid is still growing at 2¹⁷: 35, 52 and 65 cells at 2¹⁵, 2¹⁶ and 2¹⁷.
- Within the occupied cells, γ_S spans 12–71° (cells with ≥ 3 samples). Only 1 of the 243 feasible samples lies within 1° of γ* = −6.5°, and the median γ_S is −55°. Quality within a cell is therefore not trivial, which confirms the reason for the 3-D parameterization (§2).

### Go/no-go (§7): GO

| Criterion | Result | Evidence |
|---|---|---|
| 1. A candidate passes every §5 check | **Met**, for c1 and c2, under the redefined backward check (D21) and with the added entry checks. Under the original max-norm it is not met; that value is still recorded. | `gravity-free-return-c1-v2`, `-c2-v2` (`ccf326e`), status `validated` |
| 2. Re-export reproduces observables | **Met**, bit-identical across three exports, two of them clean. | `-pre4`, `-v1` (`9a721b0`), `-v2` (`ccf326e`) |
| 3. S4 occupies ≥ 10 of 100 cells | **Met**, 27 cells. | `gravity-free-return-map-3d-sobol17-v1` (`9a721b0`) |
| 4. Budget ≤ 2.5 × 10⁵ propagations | **Met.** S1 90 720 + S2 5 355 + S4 131 072 = 227 147. Adding the S3 validation (a few propagation equivalents per candidate) and the Δv-margin diagnostics (50 for c1, 48 for c2) gives ≈ 227 260. Counting the failed cap-200 attempt (2 000) gives ≈ 229 260. The fine probe (7 002) is an extra diagnostic. Regenerations and superseded runs (`pre1`–`pre4`, `-rerun`, the `-v1` candidates) are not part of the procedure's budget. | run manifests |

**Assessment: GO; M3 is complete.** Two changes to the design were made, and both are disclosed above:
- the backward check was redefined (D21);
- the S2 cap was raised from 200 to 1000 (D22).

None of the negative outcomes (a)–(c) applies:
- (a) Two candidates exist.
- (b) The integrators agree, with a smallest margin of 20.5.
- (c) The feasible set is thin (0.185 % of the box), but it is not too thin for the criterion: 27 cells.

The Jacobi gate (factor 1.26 for c1) and the physical Δv margins (9–31 mm/s) are thin, as disclosed above. The M4 draft protocol is `protocol.md`.

### Reproduce

From a clean checkout of the stated commit, with recipes as committed in `ccf326e`. Each command writes a **new** run directory and refuses to overwrite an existing one, so first move or rename the evidence run, or pass a new run id and compare.

```text
# commit 9a721b0 (or later; map code unchanged in ccf326e)
just gravity-m3-map gravity-free-return-map-2d-grid-v1                      # S1, 90 720 propagations
just gravity-m3-sobol gravity-free-return-map-3d-sobol17-v1 20261018        # S4, 131 072 propagations
# commit ccf326e
just gravity-m3-candidates gravity-free-return-map-2d-grid-v1 v2            # S2 + S3; cap 1000 is the default
just gravity-m3-probe gravity-free-return-dv-probe-v1                       # fine Δv probe, 7 002 propagations
```

The candidate command writes `gravity-free-return-search-v2` and `gravity-free-return-c{1,2}-v2`. The full tables (`samples.json`: 19.8 MB for S1, 34.3 MB for S4) stay local under `runs/`. `just verify` runs the tests.
