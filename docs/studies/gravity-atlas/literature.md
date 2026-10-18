# Gravity Atlas — Literature scan (2026-10-18)

Scope: roadmap step S3, a focused nearest-neighbor scan for one study question. This is not an open-ended review. A reference appears under "Closest works", "How this study differs" or "Baselines implied" only if its arXiv ID, DOI landing page, publisher record or full text was fetched and the title, authors and year matched. The **Verified** line on each entry says what was fetched. References whose identity was confirmed but whose abstract could not be read (paywalls) are listed under "Unverified leads" and must not be cited in a report until their content has been checked.

## Question as scanned

> In the planar Earth–Moon CR3BP, with a single departure impulse from a circular Earth parking orbit, does a quality-diversity (MAP-Elites) archive of free-return-like trajectories (lunar encounter, then an inbound Earth-shell crossing) give more useful and more robust solutions under held-out perturbations or changed constraints than multi-start local optimization, a population baseline (DE), or a generative baseline at matched propagation budgets?

Components:

| Component | Value |
|---|---|
| Domain | Earth–Moon CR3BP (planar), impulsive, free-return / circumlunar return trajectories |
| Method | Quality-diversity / MAP-Elites illumination (synonyms: novelty search, illumination, behavior repertoire) |
| Claim type | Robustness / availability of solutions under held-out perturbations and changed constraints ("archive for adaptation") |
| Baselines | Multi-start local optimization; DE / population search; generative model; (implied below) continuation catalogs and dense maps of the solution domain |

## Closest works

### 1. Beeson, Li & Sinha (2024): generative models for diverse global search in the CR3BP. Closest overall.
- R. Beeson, A. Li, A. Sinha, "Global Search of Optimal Spacecraft Trajectories using Amortization and Deep Generative Models," arXiv:2412.20023 (2024). https://arxiv.org/abs/2412.20023. Initial content appeared as AAS 23-352 (AAS/AIAA Astrodynamics Specialist Conference, 2023).
- Precursor: A. Li, A. Sinha, R. Beeson, "Amortized Global Search for Efficient Preliminary Trajectory Design with Deep Generative Models," arXiv:2308.03960 (2023). https://arxiv.org/abs/2308.03960
- **Verified:** both arXiv abstract pages were fetched. The full text of 2412.20023 was downloaded, and its introduction, problem setup, conclusion and reference list were read.
- **What they did:**
  - They frame preliminary trajectory design as a *parameter-dependent global search* whose goal is a set of solutions that are both high-quality and diverse. Diversity is formalized through a designer's "hidden" diversity metric, with Euclidean distance in control space used as a proxy.
  - They learn a conditional distribution over the basins of good solutions with a CVAE (AmorGS) and use samples from it to warm-start a gradient-based solver.
  - The benchmark is a minimum-fuel low-thrust transfer in the *spatial* Earth–Moon CR3BP, from the end of a GTO spiral to a stable-manifold arc of an L1 halo orbit. The conditioning parameter is a thrust scale α ∈ [0.1, 1].
  - Against a uniform multi-start baseline and simpler ML baselines, they report large speed-ups. They also analyze the multi-funnel structure of the landscape and discuss basin hopping as a level-2 search.
- **What they did NOT do:**
  - No quality-diversity or MAP-Elites method. Their reference list cites no QD, novelty-search or illumination work.
  - Not impulsive, not a free return, not planar.
  - No evaluation of solution robustness to execution or phase errors.
  - The "changed condition" they handle is a *continuous design parameter seen in training* (thrust level). They do not test held-out perturbations of a fixed design.

### 2. Graebner & Beeson (2025) and follow-ups: diffusion models for low-thrust CR3BP global search and Pareto-front coverage
- J. Graebner, R. Beeson, "Global Search for Optimal Low Thrust Spacecraft Trajectories using Diffusion Models and the Indirect Method," arXiv:2501.07005 (2025). Journal of the Astronautical Sciences, https://doi.org/10.1007/s40295-025-00535-1.
- J. Graebner, R. Beeson, "Self-supervised diffusion model fine-tuning for costate initialization using Markov chain Monte Carlo," arXiv:2510.02527 (2025).
- J. Graebner, R. Beeson, "Gradient-Informed Monte Carlo Fine-Tuning of Diffusion Models for Low-Thrust Trajectory Design," arXiv:2512.08705 (2025).
- **Verified:** all three arXiv abstract pages were fetched. The JAS DOI is listed on the arXiv page, and OpenAlex confirms it as the journal record.
- **What they did:**
  - They train diffusion models to warm-start indirect (costate) solvers in the CR3BP. They report one to two orders of magnitude more solutions per minute on unseen thrust values, compared with uniform sampling.
  - The follow-ups use MCMC with reward-weighted fine-tuning to complete or densify Pareto fronts. Their systems are Jupiter–Europa and Saturn–Titan. One follow-up reports "denser, more diverse Pareto front coverage," with feasibility rising from 17% to 63% (MALA versus random-walk Metropolis).
- **What they did NOT do:** no QD archive, no robustness of individual solutions to perturbations, no impulsive free returns. The diversity they target is front coverage in objective space, not coverage of a behavioral descriptor space.

### 3. Sinha & Beeson (2025): global search for *robust* low-thrust trajectories (missed-thrust events)
- A. Sinha, R. Beeson, "Initial Guess Generation for Low-Thrust Trajectory Design with Robustness to Missed-Thrust-Events," arXiv:2501.06694 (2025). Submitted to JGCD. https://arxiv.org/abs/2501.06694
- **Verified:** arXiv abstract page fetched.
- **What they did:**
  - Robust low-thrust design for a cislunar case (Lunar Gateway PPE) that accounts for missed-thrust events.
  - They compare a non-conditional global-search initializer with a conditional one that seeds from solutions of less-robust problems. The conditional initializer improves convergence rate and solution quality.
- **What they did NOT do:** robustness is built into each solution through a robust problem formulation. They do not ask whether a diverse *catalog* contains solutions that survive held-out perturbations, and they use no QD archive. Their result does show that "robustness plus global search in cislunar design" is an active topic, so the Gravity Atlas angle must be stated precisely.

### 4. Smith & Bosanac (2023) and Miceli et al. (2024): motion primitives for geometrically distinct CR3BP transfers
- T. R. Smith, N. Bosanac, "Motion Primitive Approach to Spacecraft Trajectory Design in a Multi-body System," Journal of the Astronautical Sciences (2023). https://doi.org/10.1007/s40295-023-00395-7
- G. E. Miceli, N. Bosanac, J. R. Stuart, F. Alibay, "Motion Primitive Approach to Spacecraft Trajectory Design in the Neptune-Triton System," AIAA SciTech 2024 Forum. https://doi.org/10.2514/6.2024-1280
- **Verified:** for Smith & Bosanac, Crossref and OpenAlex metadata plus the abstract. For Miceli et al., the full-text PDF from the authors' university page (title, authors and abstract match Crossref).
- **What they did:**
  - They cluster arcs of periodic orbits and their manifolds into motion primitives, then search a graph of the primitives to form sequences.
  - Correcting those sequences gives arrays of *geometrically distinct* transfers. For Smith & Bosanac these are impulsive transfers between libration point orbits in the Earth–Moon CR3BP.
- **What they did NOT do:** no QD, no budget-matched comparison with stochastic optimizers, no robustness evaluation. These papers are the dynamical-systems community's answer to "produce a diverse set of transfers," and a reviewer from that community will expect a comparison with this style of approach.

### 5. Free-return families by continuation in the CR3BP (Pavlak 2010; Jesick & Ocampo 2011; Luo et al. 2012)
- T. A. Pavlak, *Mission Design Applications in the Earth-Moon System: Transfer Trajectories and Stationkeeping*, M.S. thesis, Purdue University (2010). https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Masters/2010_Pavlak.pdf
- M. Jesick, C. Ocampo, "Automated Generation of Symmetric Lunar Free-Return Trajectories," JGCD 34(1):98–106 (2011). https://doi.org/10.2514/1.50550
- Q. Luo, J. Yin, C. Han, "Design of Earth–Moon Free-Return Trajectories," JGCD 36:263 (2013 volume; OpenAlex lists 2012 online). https://doi.org/10.2514/1.55910
- **Verified:** Pavlak's full text was downloaded and Section 3.3 was read. For Jesick & Ocampo and Luo et al., Crossref/OpenAlex metadata and abstracts were read (the AIAA landing page returned 403).
- **What they did:**
  - **Pavlak:** builds patched-conic initial guesses (after Jesick & Ocampo), then uses multiple shooting with perilune-altitude, departure-altitude and flight-path-angle constraints. Single-parameter and pseudo-arclength continuation then produce *families* of planar and out-of-plane symmetric circumlunar free returns in the Earth–Moon CR3BP. He notes that symmetry alone can admit lunar-impact solutions, so explicit altitude constraints are needed. This parallels the study's "false return" guards.
  - **Jesick & Ocampo:** automated free-return generation without trial and error, covering posigrade and retrograde departures, circumlunar and cislunar passages, and in-plane and out-of-plane cases, targeted to entry-interface conditions.
  - **Luo et al.:** an improved differential-correction method with a larger convergence domain. They run "a large number of numerical examples" to study the global features of free-return flybys.
- **What they did NOT do:** no stochastic or QD global search, no budget accounting, no robustness of the family members to execution errors. They are nonetheless the conventional way to produce a *catalog* of free returns, and that catalog is a strong diversity baseline.

### 6. He, Li & Zhou (2017): solution-domain analysis of free-return orbits
- B. He, H. Li, J. Zhou, "Solution Domain Analysis of Earth-Moon Quasi-Symmetric Free-Return Orbits," Trans. JSASS 60:195 (2017). https://doi.org/10.2322/tjsass.60.195
- **Verified:** OpenAlex metadata and abstract.
- **What they did:** a direct SQP design method in an ephemeris model, followed by systematic mapping of the *solution domain* (cross-profiles) of four kinds of quasi-symmetric free returns, constrained by TLI altitude, lunar swing-by altitude and re-entry angle.
- **What they did NOT do:** no QD or comparison of optimizers, no perturbation robustness. This is evidence that *mapping the whole feasible domain* of free returns is established practice. In a low-dimensional problem, "coverage" is therefore not a novel output by itself.

### 7. Cully et al. (2015) and QD for adaptation and uncertainty: the conceptual analogue and its known failure mode
- A. Cully, J. Clune, D. Tarapore, J.-B. Mouret, "Robots that can adapt like animals," Nature 521:503–507 (2015). https://doi.org/10.1038/nature14422; arXiv:1407.3501
- J.-B. Mouret, G. Maguire, "Quality Diversity for Multi-task Optimization," GECCO 2020. arXiv:2003.04407
- M. Flageat, A. Cully, "Uncertain Quality-Diversity: Evaluation methodology and new methods for Quality-Diversity in Uncertain Domains," arXiv:2302.00463 (2023)
- **Verified:** arXiv abstract pages fetched for all three. The Nature DOI was confirmed through Crossref.
- **What they did:**
  - **Cully et al.:** build a MAP-Elites behavior–performance map before deployment, then use it after damage to find a compensating behavior quickly. This is the canonical "diverse archive helps when conditions change" result.
  - **Mouret & Maguire:** one archive serves many related tasks.
  - **Flageat & Cully:** show that MAP-Elites performs *poorly* under uncertainty, because elites are selected on single noisy evaluations, and propose re-evaluation-based variants.
- **What they did NOT do:** no trajectory or astrodynamics application. No scan in this round found this argument applied to spacecraft trajectory design. The Flageat & Cully result is a concrete risk: elites chosen for best nominal quality may sit on knife-edges and be *less* robust per solution. That is a falsifiable prediction the protocol should test, not hide.

### 8. Vinkó & Izzo (2008): standard global-optimizer benchmarking in trajectory design
- T. Vinkó, D. Izzo, "Global Optimisation Heuristics and Test Problems for Preliminary Spacecraft Trajectory Design," ESA ACT Technical Report ACT-TNT-MAD-GOHTPPSTD (September 2008). https://www.esa.int/gsp/ACT/doc/INF/pub/ACT-TNT-INF-2008-GOHTPPSTD.pdf
- **Verified:** full-text PDF fetched; the abstract and introduction were read.
- **What they did:** released black-box MGA and MGA-1DSM test problems (the GTOP lineage) and benchmarked DE, PSO, GA and SA with default settings. These failed to solve complex problems quickly. Simple "cooperative" combinations of algorithms improved results.
- **What they did NOT do:** they measured best-objective performance only, not diversity, robustness or archives. This report does **not** include basin hopping. MBH comparisons are elsewhere (see Unverified leads: Addis et al. 2011; Vasile et al. 2010).

### 9. Zhu et al. (2025): learned initial guesses over the free-return "solution space" (NUDT group)
- 朱彬羽, 李海阳, 杨震, 何俊华, 陆林, 张宇航 (B. Zhu, H. Li, Z. Yang, J. He, L. Lu, Y. Zhang), "融合动力学特征的自由返回轨道双路网络学习方法" ("Dual-path neural network learning method for free-return orbit integrating dynamic characteristics"), 国防科技大学学报 (Journal of National University of Defense Technology) 47(4):64–75 (2025). https://doi.org/10.11887/j.issn.1001-2486.25030028. In Chinese.
- **Verified:** the full-text PDF (free from journal.nudt.edu.cn / sciopen) was read in full on 2026-10-18. The title, authors, volume and DOI match the PDF header. The PDF itself carries no licence statement.
- **What they did:**
  - *Model.* The free return is an ephemeris problem in an Earth-centred J2000 frame. A tangential TLI impulse from a circular parking orbit is followed by a free coast to perilune and then a return to a vacuum perigee (VCP). The re-entry condition (122 km, −6°) is converted to a VCP altitude of about 51 km. Convergence is done in ATK (NUDT's in-house "Aerospace Tool Kit") with a 21×21 JGM3 Earth field plus Sun and Moon third-body gravity. The training database comes from the group's pseudo-perilune-parameter method (Yang et al. 2024, below).
  - *Problem, Eq. (1).* The decision variables are the transfer-orbit RAAN Ω, argument of perigee ω and TLI impulse v. The TLI epoch, altitude and inclination are fixed by the launch segment. The constraints are perilune altitude inside a band, VCP altitude equal to its target and return inclination equal to its target.
  - *"Solution space".* They traverse TLI epoch, transfer inclination and return inclination with correction (perilune band printed as "[120.5 km, 119.5 km]", i.e. about 120 km). They plot the converged solutions as ω and Ω against the Moon's orbital phase (Fig. 2). Solutions fall into two disjoint, non-overlapping families: ascending-node and descending-node departures. These form continuous arcs and parallel slanted bands, with angle wrap-arounds that hurt a single network. This is a *nominal existence map* under equality constraints.
  - *Network.* Two separate fully connected paths (ascending and descending). Each has 8 hidden layers [32…256…16] with RMSProp and Leaky-ReLU, a learning rate of 1e-3 and 100 epochs. Inputs are the Moon's J2000 elements at TLI plus h_TLI, i_LTO, h_VCP and i_RTO. Outputs are (Ω, ω, v). A piecewise-linear "parameter-correlation transform" (Eqs. 9–12) unwraps the angle discontinuities in each branch.
  - *Data.* 36,963 converged free returns, with perilune epochs from 2029-08-01 to 2029-11-01, i_LTO and i_RTO over [0°, 180°] and both node branches. Generation took about 208,674 s. The split is 80/20.
  - *Results.*
    - Test RMSE is about 6–7° in Ω, about 4° in ω and about 2 m/s in v.
    - On 2,000 random free returns, the mean absolute error is about 3.6–4.2°, 1.9–2.1° and 1.7–1.8 m/s.
    - Worked case (2028-06-24, i = 21°, perilune 200 km, VCP 50 km, i_RTO = 43°): the descending branch reproduces the reference solution of Peng et al. (2011) at 3162.1 m/s in 2.5 s versus 66.2 s. The ascending branch gives a second, new solution at 3165.0 m/s.
    - Across six cases: with no initial guess, nothing converged. The single-path predecessor of Yang et al. (2024) missed one branch in four of the six. The dual-path guess converged both branches in all six, four of which lie outside the training epoch range.
- **What they did NOT do:**
  - No robustness, dispersion, execution-error or midcourse-correction analysis, and no sensitivity study.
  - No "controllable set" or reachability computation. The paper cites none of the Du & Li controllable-set papers. Its only reachability reference is Yang et al. (2024), described as giving fast estimates of the perilune *reachable domain*.
  - No comparison of search or optimisation paradigms at matched budgets.
  - "Diversity" means only the two node branches. Finding the second branch is presented as "expanding the high-precision solution space".
- **Relevance:** this is the nearest *ML-for-free-return* work. It is an amortised initial-guess generator over mission constraints, so it fits under the deferred generative baseline (Beeson lineage), not under robustness. Its Fig. 2 confirms that the feasible departure set splits into a few disjoint, thin, continuous families. That matches the M3 finding of a thin feasible band (0.185 % of the box).
- **Note:** this is *not* the Zhu et al. AST 170:111492 paper named in the protocol preconditions. That paper is still unread (Unverified leads). Its Crossref reference list overlaps heavily with this one, and it also cites Lu et al. (2023, controllable set) and Wen et al. (2014, reachable domain). So it may contain more on reachability than this paper does.

### 10. Reachable domains and controllable sets of Earth–Moon return trajectories (NUDT / China Astronaut Research and Training Center lineage)
This is the lineage that the Du & Li 2026 papers belong to. Read in order of directness:
- **Full text read:**
  - L. Yang, H. Li, J. Zhang, Y. Zhu, "Fast solution to the free return orbit's reachable domain of the manned lunar mission by deep neural network," J. Syst. Eng. Electron. 35(2):495–508 (2024). https://doi.org/10.23919/JSEE.2023.000117. Free PDF from jseepub.com.
    - Their Eq. (9) defines the reachable domain (RD) as the set of terminal characteristic parameters reachable from an initial set under admissible controls. For free returns, the RD is the perilune (inclination, RAAN) pair, which maps to reachable lunar sites. The departure and return constraints (h_TLI = 170 km, i_TLI = 28°, h_VCP = 50 km, i_VCP = 42°) are fixed equalities. The perilune is at 100 km.
    - Free returns are split into 8 types by ascending or descending phase at TLI, VCP and perilune. The RD of each type is a 1-D curve in (i, Ω), and the types do not overlap.
    - The database is 87,261 solutions from pseudo-perilune parameters solved with SQP. The classifier is >99 % accurate and the regression MSE is <0.01°.
    - The surrogate takes 0.15 s for 875 orbits, against about 1.2 h with the "double two-body" (patched-conic) design.
    - **No dispersion or robustness analysis.**
  - Q. Peng (彭祺擘), L. Lu (陆林), H. Zhang (张海联), "基于区间数的应急返回轨道可达域求解方法" ("A reachable domain calculation method of contingency return trajectories based on interval number"), 中国空间科学技术 (Chinese Space Science and Technology) 46(2):36–47 (2026). https://doi.org/10.16708/j.cnki.1000-758X.2026.0022. Free PDF from the journal site. In Chinese.
    - *Setting.* A single non-coplanar impulse from a 100 km circular lunar orbit at an arbitrary epoch. The reachable domain is the set of re-entry (inclination, longitude, latitude) values that satisfy the re-entry altitude and angle with flight time ≤ T_max.
    - *Model.* Simplified on purpose because high-fidelity is too slow: a Lambert arc inside the lunar sphere of influence plus a Kepler arc to Earth, with DE430 Moon ephemeris. Each candidate is a minimum-Δv SQP (SNOPT) solve.
    - *Method.* An "improved interval branch-and-bound": the SOI exit longitude and latitude are split into sub-boxes. The elimination test checks whether solutions exist on each sub-box's *boundaries*. If they do, the box is bisected at the midpoint down to a minimum width, and the step returns *all* solutions rather than an optimum. This is not a rigorous interval inclusion.
    - *Results.* 47.4–96.5 % less time than brute-force traversal. The domain characteristics repeat with a period of about 28 d.
    - **No uncertainty or robustness analysis.**
- **Abstract read** (publisher full text paywalled):
  - L. Lu, J. Zhou, H. Li, H. Zhang, "Investigation on reachable domain of contingency return trajectories in the circumlunar flight phase based on interval analysis," Adv. Space Res. 72(9):3770–3786 (2023). https://doi.org/10.1016/j.asr.2023.07.023. The abstract is that of the CC BY Research Square preprint rs-2307021 (via Semantic Scholar). The preprint PDF returned 403 and was not read. Its content matches the 2026 CSST paper above: an improved interval branch-and-bound reachable domain, checked against brute force.
  - Z. Du, H. Li, L. Lu, B. Zhu, X. Bu, "A fast solution to impulse controllable set of direct abort orbits from low lunar orbit based on deep learning," IEEE TAES 61(6):19215–19238 (2025). https://doi.org/10.1109/TAES.2025.3618994. Abstract via Semantic Scholar.
    - The paper defines and formulates a "controllable set" (CS) for direct aborts from low lunar orbit.
    - It builds CS databases with pseudo-perilune parameters and a two-segment double two-body model.
    - It fits fully connected networks with "equivalent accuracy" at much lower cost.
    - It analyses how constraints change the CS "topological and numerical characteristics".
  - Z. Du, H. Li, H. Wang, H. Jing, "Time-dependent controllable set for contingency return trajectories under spatiotemporal constraints in CR3BP," Aerosp. Sci. Technol. 176:112680 (2026). https://doi.org/10.1016/j.ast.2026.112680. Abstract of the SSRN preprint (10.2139/ssrn.6169467) via Crossref; the SSRN page itself returned 403.
    - The set is the time-dependent controllable set (TDCS) of contingency returns that must reach a designated landing site at a prescribed epoch or window.
    - They propose a "definition and mathematical formulation of the controllable set" and a re-entry pseudo-parameter method. Backward integration in the CR3BP plus differential correction refines the terminal state.
    - Large numerical sweeps give the set's "topological characteristics". Within the departure TDCS, perilune, vacuum-perigee longitude and latitude, and Jacobi integral form a "tube-like structure".
  - Z. Du, H. Jing, H. Li, H. Wang, "Fast approximation and analysis for time-dependent controllable set of Earth–Moon transfer orbits via velocity surfaces method," CMDA 138:46 (2026). https://doi.org/10.1007/s10569-026-10321-2. The abstract was read on the Springer page (paraphrased there, not copied).
    - It defines a *departure* TDCS for Earth–Moon transfer orbits and estimates it quickly with a "velocity surfaces" method that handles terminal-time constraints.
    - It is checked against high-fidelity dynamics and supplies initial guesses for high-fidelity correction.
    - It analyses how constraints affect the TDCS parameters. Appendix A uses two-body (conic) relations.
- **Not read, so not cited for content:** Du & Li, AST 179:113609, is the free-return controllable-set paper itself. Only its metadata and Crossref reference list were obtained; see Unverified leads for what those imply.
- **What the lineage does:** it computes the *set* of departure (or abort) conditions, or reachable terminal parameters, for which a constraint-satisfying return exists. It works under nominal dynamics, mostly in patched-conic or pseudo-perilune models, with some CR3BP and ephemeris verification. Brute-force traversal, interval branch-and-bound, DNN surrogates and velocity surfaces serve as fast enumerators. The purpose is mission decision support: is this landing site, epoch or abort point covered?
- **What it does NOT do (in everything read):**
  - No per-solution robustness to injection or execution errors.
  - No midcourse-correction budget.
  - No catalog of individual solutions judged by quality or diversity.
  - No comparison of search paradigms at matched cost.
  - "Controllable" here means that a nominal solution *exists*, with a manoeuvre for aborts. It is not tolerance to perturbations of a chosen solution.
- **Relevance:** this lineage is the established answer to the question "what is the feasible departure set?". In the planar 3-parameter M4 box, the same object is the dense map plus band-edge refinement (protocol §5.3). Mapping the feasible set is therefore *not* a contribution of Gravity Atlas, and the report must cite this lineage when it presents coverage ground truth. The lineage does not measure or compare what D16/D17 measure: survival of catalog members under held-out errors with one correction, and availability under changed constraints, compared across generators.

### 11. Dispersion of free returns and midcourse correction (context for the P1 error model)
- C. Huo, H. Wang, Z. Li, J. He, "Uncertainty propagation of the lunar free return orbits and ATK simulation," J. Phys.: Conf. Ser. 3109:012011 (2025). https://doi.org/10.1088/1742-6596/3109/1/012011. CC BY 4.0, **full text read**.
  - High-fidelity ATK propagation: JGM3 6×6 Earth field, Sun and point-mass Moon, RK8. The nominal free return departs a 200 km, i = 28° parking orbit with TLI Δv = 3166.3 m/s and returns to a 50 km VCP at i = 43°.
  - Monte Carlo with 5,000 samples. The Gaussian injection-state errors are σ_r = 100 m and σ_v = 0.01 m/s per axis.
  - Position variance grows by about 10³ after two days, 10⁶ at perilune and 10¹⁰ at return. The distribution is non-Gaussian by perilune.
  - With σ_r = 1 km and σ_v = 0.2 m/s per axis, 172 of 5,000 samples (3.44 %) impact the Moon.
  - No correction manoeuvre is modelled. The error magnitudes are design values without a cited source.
- Q. Peng, B. He, H. Zhang, "地月转移自由返回轨道偏差传播分析" ("Analysis of deviation propagation for translunar free return orbit"), 深空探测学报 (J. Deep Space Explor.) 3(1) (2016). https://doi.org/10.15982/j.issn.2095-7777.2016.01.009. **Abstract only** (bilingual, from the journal page).
  - They derive a deviation-propagation equation in the synodic frame and show that deviations grow quickly after the lunar flyby, to the point of no return.
  - They conclude that a "strict free return orbit in engineering is not existed" and that midcourse correction is necessary.
- E. R. Burnett, S. Boone, N. Michelotti, "An Efficient Non-Gaussian Chance Constraint Method for Stochastic Nonlinear Problems in Spaceflight," arXiv:2607.03424 (2026), AAS 26-899 preprint. **Full text read**.
  - Their worked example is a *planar CR3BP* analogue of the Artemis II free return, corrected by two post-flyby midcourse manoeuvres. It uses navigation errors and the Gates (1963) execution-error model (proportional, resolution, pointing and autopilot terms; 3 % pessimistic, later 0.5 % proportional), with 3σ chance constraints on entry altitude and flight-path angle.
  - Retargeting the first correction (0.25 d after flyby, about 6.5 m/s) naively violates the constraints in 62.9 % of 5,000 Monte Carlo samples. The LinCov policy (3.35 m/s) violates them in 0.56 %, and their non-Gaussian "banana" policy (2.87 m/s) in 0.2 %.
  - This is the closest found precedent for "planar CR3BP free return plus execution errors plus correction".
  - It designs the correction for *one* nominal trajectory and does not compare catalogs or generators.
- **Relevance:**
  - These works support D17: robustness has to be defined against a correction budget, because uncorrected free returns are not robust.
  - They give literature points for the injection-error scale question. Huo et al.'s 0.2 m/s per axis is smaller than the declared 0.03 % × about 3.1 km/s ≈ 0.9 m/s magnitude σ; that comparison is made here, not by Huo et al.
  - Gates (1963), cited by Burnett et al., is the standard execution-error structure. It has been seen only second-hand here.
  - None of the three works asks which of many candidate free returns tolerates errors best, or whether a diverse catalog helps.

## How this study differs

- **Method × domain:** no published application of MAP-Elites, quality-diversity or novelty search to spacecraft trajectory design was found in any dynamics model (CR3BP, MGA, low-thrust, GTOC). The searches covered the arXiv API, the web and citation hops. This is still a *negative search result*, not proof: GECCO and CEC proceedings and AAS conference papers are poorly indexed.
- **Diversity target:** prior astrodynamics diversity work produces diversity through learned generative samplers (Beeson group), motion-primitive graphs (Bosanac group), continuation families (Pavlak; Jesick & Ocampo) or domain mapping (He et al.). Gravity Atlas would compare a *descriptor-space archive* against these at a matched propagation budget. None of the works above report budget-matched comparisons across these paradigms.
- **Set characterization vs. catalog robustness:** the reachable-domain and controllable-set lineage (§10: Yang et al. 2024; Lu et al. 2023; Peng et al. 2026; Du et al. 2025–2026) already computes the set of conditions for which a free or contingency return *exists*. It uses interval branch-and-bound, DNN surrogates or velocity surfaces under nominal dynamics. Gravity Atlas does not claim to map that set; its dense map with band-edge refinement is the planar-CR3BP analogue, used as ground truth. What none of the works read does is evaluate how *individual catalog members* tolerate held-out injection errors with one bounded correction, or compare catalog generators at matched cost. P1 is, in effect, a sampled test of whether each perturbed state still lies in the one-correction controllable set. The report should say so and cite the lineage.
- **Claim type:** robust-design papers (Sinha & Beeson 2025; Kumagai & Oguri 2025) optimize *each* solution for robustness. Gravity Atlas asks a different question: whether a *catalog* produced without robustness in the objective still contains usable solutions when execution errors or constraints change after the search. This is the Cully et al. argument transplanted to trajectories, and it was not found applied there.
- **Problem class:** impulsive, ballistic free returns. All the generative-model and diffusion work above is low-thrust optimal control in high-dimensional transcriptions.
- **Honest evaluation of QD's known weakness:** per-solution fragility of elites (Flageat & Cully) can be tested directly with held-out perturbations. A null or negative result is informative to both communities.

## Novelty risk

**Medium.** This is unchanged after the second pass on the controllable-set literature (§9–§11). The *coverage or feasible-set* framing is now known to be well-trodden. The *catalog-robustness under a correction budget, across generators* framing is still unmatched in everything read. Du & Li, AST 179:113609, remains unread.

- *Lower risk:* no QD-for-trajectories paper was found. The archive-for-adaptation argument has not been tested in astrodynamics. The closest ML work (Beeson group) targets diversity with generative models and does not evaluate held-out perturbation robustness of a catalog.
- *Higher risk 1, low dimension:* the provisional search space has about 3 parameters (departure phase and two impulse components). In 3-D, a dense quasi-random or grid map of the whole feasible domain is affordable (compare He et al. 2017; Luo et al.'s large-sample global-feature study). An expert reviewer will ask why illumination is needed when the domain can be enumerated. Unless the comparison includes a budget-matched dense map and reports where the curves cross, the "QD gives better coverage" result may be trivial.
- *Higher risk 2, conventional catalogs:* continuation families of free returns (Pavlak; Jesick & Ocampo) are cheap and well understood. An astrodynamics reviewer will treat them as the default diversity baseline.
- *Higher risk 3, the robustness claim may be trivially negative:* ballistic free returns are known to be very sensitive to injection errors, and operational practice relies on midcourse corrections. If "robust" means "remains feasible with no correction," almost every solution may fail, and the metric loses discriminative power. Robustness must be defined against a declared correction budget or margin, with that definition frozen before results are seen.
- *Risk from a fast-moving neighbor:* the Beeson group publishes regularly on diversity in CR3BP global search. A QD comparison from them is plausible.
- *Risk 4, the reachability lineage (added after the second pass):* a dense Chinese-language and Elsevier literature already maps feasible free-return and abort sets (§10). Coverage or "solution-space" claims must therefore be framed as ground truth, not as a contribution. The one paper not read in any form beyond metadata, Du & Li, AST 179:113609, might add uncertainty to the free-return controllable set: it cites Long et al. 2015 on controllable sets "under model uncertainties" and an interval-uncertainty trajectory paper. Until it is read, the report must not claim that robustness of free-return sets is unstudied, only that robustness of *catalog members under a correction budget, compared across generators* was not found.
- *Risk 5, a nearby planar-CR3BP correction study:* Burnett et al. (2026) already use a planar CR3BP free return with execution errors and midcourse corrections, for single-trajectory stochastic guidance. Reviewers will expect P1's error model to be comparable, e.g. Gates-style proportional and pointing terms.

## Baselines implied by prior work

1. **Multi-start local optimization** (Beeson et al. use uniform multi-start as the reference). Already B1.
2. **Monotonic basin hopping (MBH).** This is the standard trajectory global optimizer in the astrodynamics literature, and Beeson et al. discuss funnel structure in its terms. It is arguably a stronger conventional baseline than DE for this audience. Recommend B2 = DE (Vinkó & Izzo lineage) *plus* MBH, or justify the choice of one.
3. **Dense quasi-random / grid map of the 3-D domain** (domain-mapping practice, He et al. 2017). This baseline is needed twice:
   - (a) as *ground truth* for the feasible set, so coverage can be measured as a fraction of truly feasible descriptor cells;
   - (b) as a budget-matched baseline.
4. **Continuation-family catalog:** seed with a few converged free returns and continue them in perilune altitude and time of flight (Pavlak; Jesick & Ocampo). Bin the result into the same descriptor grid, and count the propagations it costs.
5. **Descriptor-binned archives of B1, B2 and grid outputs** (already in the README). This is required so that MAP-Elites' diversity is not just a by-product of its data structure.
6. **Generative baseline (CVAE / diffusion).** It is optional, and the protocol must decide whether training-data generation counts toward the budget. Prior work amortizes over a *family* of parameter values, which only pays off if the study includes a changed-constraint family. Recommend deferring this baseline unless the changed-constraint test is adopted.
7. **QD variants robust to noise (Flageat & Cully).** These are not required in v1, but they are the obvious follow-up if MAP-Elites elites prove fragile. Do not add them after seeing results unless the protocol is revised and logged.
8. **Set-based (reachability) computation, not added as a baseline.** Interval branch-and-bound and surrogate methods (§10) target the same object as the dense map in a low-dimensional box: the feasible set. They produce no ranked catalog and need no budget comparison. In the 3-D M4 box, the dense map with band-edge refinement already plays this role. The protocol should cite §10 when it defines coverage ground truth. A reachability-inspired *diagnostic* is optional. One example is each solution's distance in u-space to the edge of the feasible band, as a predictor of P1 survival. It may be added only if pre-registered before the freeze.

## Unverified leads

The items marked "metadata verified" have DOIs that resolve to the stated title, authors and year (Crossref/OpenAlex), but their abstract or full text could not be read. They must not be cited for content until it has been checked.

- A. Miele, T. Wang, S. Mancuso, "Optimal Free-Return Trajectories for Moon Missions and Mars Missions," J. Astronaut. Sci. 48:183–206 (2000), https://doi.org/10.1007/BF03546276. Metadata verified. A secondary source says it uses the restricted three-body model and the sequential gradient-restoration algorithm.
- A. Miele, S. Mancuso, "Optimal Trajectories for Earth–Moon–Earth Flight," Acta Astronautica 49:59–71 (2001), https://doi.org/10.1016/S0094-5765(01)00007-8. Metadata verified.
- B. Addis, A. Cassioli, M. Locatelli, F. Schoen, "A global optimization method for the design of space trajectories," Comput. Optim. Appl. 48:635–652 (2011), https://doi.org/10.1007/s10589-009-9261-6. Metadata verified. This is reportedly the MBH reference for GTOP problems.
- M. Vasile, E. Minisci, M. Locatelli, "Analysis of Some Global Optimization Algorithms for Space Trajectory Design," J. Spacecraft Rockets 47:334–344 (2010), https://doi.org/10.2514/1.45742. Metadata verified.
- C. H. Yam, D. Di Lorenzo, D. Izzo, "Low-thrust trajectory design as a constrained global optimization problem" (2011), SAGE Proc. IMechE G. Seen only in search results. Reportedly uses MBH.
- A. Englander, J. Englander, M. Carter, "Hopping with an Adaptive Hop Probability Distribution" (AAS/AIAA). Seen only in the Beeson et al. reference list.
- K. Oshima, S. Campagnola, T. Yanao, "Global search for low-thrust transfers to the Moon in the planar circular restricted three-body problem," CMDA 128:303–322 (2017), https://doi.org/10.1007/s10569-016-9748-2. Seen only in the Beeson et al. reference list. Potentially relevant as a planar Earth–Moon global-map precedent.
- F. Topputo, "On optimal two-impulse Earth–Moon transfers in a four-body model," CMDA 117:279–313 (2013), https://doi.org/10.1007/s10569-013-9513-8. Metadata verified. Reportedly a global Δt–Δv map of transfers: a "map the whole domain" precedent.
- D. C. Folta, N. Bosanac, D. Guzzetti, K. C. Howell, "An Earth–Moon system trajectory design reference catalog," Acta Astronautica 110:341–353 (2015), https://doi.org/10.1016/j.actaastro.2014.07.037. Metadata verified. The *conference* version (IAA-DyCoSS2, NTRS 20140010042) was read: it describes an interactive catalog of CR3BP periodic and quasi-periodic orbits (libration point orbits, DROs, DPOs, resonant orbits). It does not include free returns.
- D. Guzzetti, N. Bosanac, A. Haapala, K. C. Howell, D. Folta, "Rapid trajectory design in the Earth–Moon ephemeris system via an interactive catalog of periodic and quasi-periodic orbits," Acta Astronautica 126:439–455 (2016), https://doi.org/10.1016/j.actaastro.2016.06.029. Metadata verified.
- NASA/JPL Three-Body Periodic Orbits API (Poincaré catalog), https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html. The documentation page was fetched: it covers Earth–Moon and other systems, with Lyapunov, halo, DRO, DPO, resonant and other families, and gives Jacobi constant, period and stability index. It is a catalog of *periodic* orbits, not of free returns. It is useful for validation (Jacobi checks) and as a precedent for a public catalog, but no formal citation is given.
- S. Kelly, D. K. Geller, "Robust Cislunar Trajectory Optimization in the Presence of Stochastic Errors," J. Astronaut. Sci. 71 (2024), https://doi.org/10.1007/s40295-024-00450-x. Metadata verified. Per the search summary: impulsive CR3BP transfers with a ΔV + 3σ correction objective under injection, execution and navigation errors.
- N. Kumagai, K. Oguri, "Robust Cislunar Low-Thrust Trajectory Optimization under Uncertainties via Sequential Covariance Steering," arXiv:2502.01907 (to appear in JGCD). The arXiv abstract *was* fetched, so the reference itself is verified, but it was read only at abstract level. It is relevant as robust-design background (Earth–Moon CR3BP, chance-constrained covariance steering).
- Z. Du (Zhenjiang Du, NUDT), H. Li, "A fast calculation method for the controllable set of Earth–Moon free-return trajectories based on interval analysis," Aerosp. Sci. Technol. 179:113609 (Dec 2026), https://doi.org/10.1016/j.ast.2026.113609.
  - **Status after the 2026-10-18 second pass:** metadata and the Crossref reference list were verified. No abstract or full text was obtainable: Crossref, OpenAlex and Semantic Scholar have no abstract and report it closed, and ScienceDirect returned 403. No preprint was found.
  - What is known is second-hand, from the title, the reference list and the sibling papers in §10. Most likely it is an interval branch-and-bound computation of the departure set for constrained free returns, like Lu et al. 2023 and Peng et al. 2026.
  - Besides the two papers named in the next bullet, its reference list cites: Lu et al. 2023 (both), Yang et al. 2024 JSEE, Zhu et al. AST 170, Benito & Mease 2010, Wen et al. 2014, Komendera et al. 2015, interval-analysis optimisation work (van Kampen 2010; Chen et al. 2013; Ma et al. 2015/2016), SNOPT, and no QD or illumination work.
  - It cites Long et al. 2015, "Controllable set analysis for planetary landing under model uncertainties," and Yang et al. 2024 (Proc. IMechE G), "interval programming … under interval uncertainty." So an uncertainty layer cannot be ruled out.
  - **Still the one item to read before any novelty statement about robustness.**
  - One route is the author manuscript. The corresponding author's e-mail, lihaiyang@nudt.edu.cn, is printed in Zhu et al. 2025.
- Z. Du, H. Li, H. Wang, H. Jing, "Time-dependent controllable set for contingency return trajectories under spatiotemporal constraints in CR3BP," AST 176:112680 (2026), https://doi.org/10.1016/j.ast.2026.112680. **Moved to §10:** the abstract was read from the SSRN preprint record (10.2139/ssrn.6169467) via Crossref.
- Z. Du, H. Jing, H. Li, H. Wang, "Fast approximation and analysis for time-dependent controllable set of Earth–Moon transfer orbits via velocity surfaces method," CMDA 138:46 (2026), https://doi.org/10.1007/s10569-026-10321-2. **Moved to §10:** the abstract and keywords were read on the Springer page. The full text is paywalled.
- B. Zhu, H. Li, L. Lu, Z. Du, L. Yang, "Data and knowledge dual-driven design method for Earth–Moon free-return orbit," AST 170:111492 (Mar 2026), https://doi.org/10.1016/j.ast.2025.111492.
  - Metadata and the Crossref reference list were verified. No abstract was obtainable.
  - Its reference list overlaps heavily with the open-access Zhu et al. 2025 JNUDT paper (§9); it shares about ten references. It also adds Lu et al. 2023 (controllable set), Lu et al. 2023 ASR (reachable domain), Wen et al. 2014 (single-impulse reachability), Arslantaş et al. 2016 (reachability) and He et al. 2017.
  - It is probably an extended English version of the same line of work (an ML initial-guess generator). That is inferred, not verified.
- Z. Du, H. Li, L. Lu, B. Zhu, X. Bu, "A fast solution to impulse controllable set of direct abort orbits from low lunar orbit based on deep learning," IEEE TAES 61(6):19215–19238 (2025), https://doi.org/10.1109/TAES.2025.3618994. Abstract read (§10).
- H. Jing, Z. Du, H. Li, H. Wang, J. He, C. Huo, "Study on the reachable domain of Earth–Moon transfer orbits with terminal constraints in the CR3BP," Acta Astronautica (2026), https://doi.org/10.1016/j.actaastro.2026.03.046. Semantic Scholar metadata only.
- Z. Du, H. Jing, H. Li, H. Wang, "Design and analysis of controllable domain for point return orbits using velocity surfaces," Adv. Space Res. 78(2):1254–1273 (2026), https://doi.org/10.1016/j.asr.2026.04.059. Seen only in the CMDA reference list.
- Z. Du et al., "Data-driven surrogate modeling for fast reachability envelope prediction of direct abort orbits for manned lunar exploration missions," Acta Astronautica (2026), https://doi.org/10.1016/j.actaastro.2026.06.056. Seen only in the AST 113609 reference list.
- Q. Wang, L. Lu, H. Li, L. Yang, "Global sensitivity analysis of point return orbit based on controllable domain," 系统工程与电子技术 (Syst. Eng. Electron.) 45(11):3606–3615 (2023). Seen only in the CMDA reference list.
  - It is potentially the closest to a "sensitivity of a controllable set" study and is in a Chinese journal that often hosts free PDFs.
  - A web search did not find it on sys-ele.com.
- Yang et al., "An interval programming method of Earth–Moon trajectory for manned lunar exploration mission under interval uncertainty," Proc. IMechE G (2024), https://doi.org/10.1177/09544100241254156. Seen only in the AST 113609 reference list. It is robust (interval-uncertainty) Earth–Moon trajectory design, so read it if the robustness framing is challenged.
- M. Long et al., "Controllable set analysis for planetary landing under model uncertainties," Adv. Space Res. (2015), https://doi.org/10.1016/j.asr.2015.04.019, and J. Benito, K. D. Mease, "Reachable and controllable sets for planetary entry and landing," JGCD (2010), https://doi.org/10.2514/1.47577. Both are seen only in the AST 113609 reference list. They are the probable origin of the "controllable set" terminology: the set of initial states that admissible controls can steer to the target set.
- E. A. Komendera et al., "Efficiently evaluating reachable sets in the circular restricted 3-body problem," IEEE TAES (2015), https://doi.org/10.1109/TAES.2014.130781. Seen only in the AST 113609 reference list.
- D. Woffinden, R. Eckman, S. Robinson, "Optimized Trajectory Correction Burn Placement for the NASA Artemis II Mission," AAS 23-062 (2023). Seen only in the Burnett et al. reference list, where it is described as a NASA LinCov study of Artemis II stochastic guidance. It is a candidate operational source for injection and correction error scales.
- C. R. Gates, "A Simplified Model of Midcourse Maneuver Execution Errors," JPL TR 32-504 (1963). Known through Burnett et al. 2026 (§11); not retrieved.
- B. He (贺波勇), "Analysis of transfer orbit deviation propagation and mid-course correction for manned lunar landing," PhD thesis, NUDT (2013), and "Analysis approaches for precision reachable sets of manned lunar orbits using numerical continuation theory" (NUDT, 2017; cited as a dissertation). Seen only in the Huo et al. and Guo et al. reference lists.
- X. Guo, Z. Du, X. Guo, H. Li, "High-Fidelity Lunar Free-Return Trajectory Optimization via Neural Network and ATK Software," J. Phys.: Conf. Ser. 3109:012038 (2025), https://doi.org/10.1088/1742-6596/3109/1/012038. CC BY; **full text read**.
  - A low- / medium- / high-fidelity cascade: patched conic, then Earth–Moon point masses with Earth non-spherical terms, then ATK.
  - An MLP predicts departure parameters for high-fidelity correction. Its convergence rate is 73–90 %, against 13–63 % for patched-conic guesses and 95–100 % for medium-fidelity guesses.
  - It is another amortised initial-guess generator with no robustness content, so it is listed here rather than in Closest works.
- T. Dong, Q. Luo, C. Han, "Bayesian Optimization of crewed lunar free return abort trajectory," Acta Astronautica 201:288 (2022), https://doi.org/10.1016/j.actaastro.2022.08.052. Metadata verified.
- H. Zhang, C. Han, Q. Luo, J. Li, "A UKF based design method for lunar free-return trajectories," AST 48:272 (2015), https://doi.org/10.1016/j.ast.2015.11.007. Metadata verified.
- L. Lu, J. Zhou, H. Li, Q. Peng, W. Zhou, "Study on controllable set of contingency return trajectories for crewed lunar missions based on interval analysis," Acta Astronautica 210:322–338 (2023), https://doi.org/10.1016/j.actaastro.2023.05.028. Metadata verified (Crossref, OpenAlex); no abstract obtainable. A search summary says it computes the controllable set for returns departing low lunar orbit with interval analysis and a modified branch-and-bound, verified against brute force. That summary is second-hand.
- W. Litteri, Á. F. Gil, M. Vasile, V. Rodríguez-Fernández, D. Camacho, "Generation of periodic orbits in the restricted three-body problem with a variational autoencoder," CMDA 138 (2026), https://doi.org/10.1007/s10569-026-10299-x. The OpenAlex abstract was read (VAE plus latent-space continuation to generate members of periodic-orbit families). It is a generative-catalog precedent, not for free returns. It cites Beeson et al.
- J. Briden, B. J. Johnson, R. Linares, A. Cauligi, "Diffusion Policies for Generative Modeling of Spacecraft Trajectories," AIAA SciTech 2025, https://doi.org/10.2514/6.2025-2775. The OpenAlex abstract was read: compositional diffusion for 6-DoF powered descent, adapting to changed constraints few-shot. Conceptually relevant to "conditions change," but out of domain.
- Y. Takubo et al., "Towards Robust Spacecraft Trajectory Optimization via Transformers," arXiv:2410.05585. The arXiv abstract was fetched: it covers LEO rendezvous with chance constraints. Out of domain.
- H. Schwaniger (1963) and Penzo (1963): classical CR3BP and patched-conic free-return work. Known only through Pavlak's thesis, not retrieved.
- B. Nunes, Brás, Batista, Xavier, "Designing trajectories in the Earth–Moon system: A Levenberg–Marquardt approach," Acta Astronautica 249:337–357 (2026), https://doi.org/10.1016/j.actaastro.2026.07.020. Already cited in the README. Crossref metadata matches, but the abstract was not read.
- GitHub issues PINNeAPPle-Labs/PINNeAPPle #354 and #397 propose MAP-Elites for gravity-assist and low-thrust design. They are plans, not results, and are not citable literature. They do signal that others are thinking along these lines.
- A hobbyist simulation claims that a 0.3% TLI overburn shifts perilune by about 10,000 km (T-Lind/ss.jl). Unvalidated. Do not use it. The study should measure sensitivity itself.

## Search log

All searches ran on 2026-10-18. "Web (ext.)" means WebSearch in extended mode.

| # | Query | Source | Notable hits |
|---|---|---|---|
| 1 | MAP-Elites quality-diversity spacecraft trajectory design | Web (ext.) | No space application. Only a GitHub issue proposal (PINNeAPPle #397) |
| 2 | quality diversity illumination three-body problem trajectory optimization | Web (ext.) | No QD × CR3BP |
| 3 | novelty search interplanetary trajectory multiple gravity assist diverse solutions | Web (ext.) | No novelty search for MGA. Vasile & De Pascale multimodality (secondary) |
| 4 | arXiv 2412.20023 (direct) | arXiv | Beeson, Li & Sinha 2024. Full text downloaded |
| 5 | Li Beeson Sinha diffusion model warm start low-thrust trajectory CR3BP | Web (ext.) | AmorGS 2308.03960; Graebner & Beeson 2501.07005; 2510.02527; 2512.08705; Sinha & Beeson 2501.06694 |
| 6 | Vinkó Izzo global optimisation heuristics spacecraft trajectory benchmark differential evolution basin hopping | Web (ext.) | ESA ACT report (fetched); GTOP; GTOPX; Yam et al. 2011 (lead) |
| 7 | free-return trajectory Earth-Moon circular restricted three-body problem families design | Web (ext.) | Pavlak 2010 thesis; NASA Artemis abort paper (2024, not fetched) |
| 8 | "quality-diversity" OR "MAP-Elites" orbit OR spacecraft OR "space mission" OR astrodynamics | Web (ext.) | No space application |
| 9 | niching multimodal optimization spacecraft trajectory multiple distinct solutions evolutionary | Web (ext.) | Only a generic multimodal-DE paper; no niching for trajectory diversity |
| 10 | Cully Clune Tarapore Mouret 2015 robots that can adapt like animals Nature | Web | arXiv 1407.3501 / Nature DOI |
| 11 | lunar free return trajectory design method restricted three-body problem journal article | Web (ext.) | Pseudostate methods; Miele et al.; multisegment free returns |
| 12 | Miele Wang Mancuso optimal free-return trajectories Moon missions Mars missions restricted three-body | Web (ext.) | Miele et al. 2000 (lead) |
| 13 | Apollo free return trajectory family circumlunar planar restricted three-body periodic symmetric orbits catalog | Web (ext.) | Schwaniger 1963 (secondary); lunar periodic-orbit database (lead) |
| 14 | robust cislunar trajectory design uncertainty maneuver execution errors optimization families | Web (ext.) | Kelly & Geller 2024; Kumagai & Oguri 2025; NTRS midcourse correction |
| 15 | Crossref bibliographic queries (Jesick & Ocampo; Miele; Addis et al.; Vasile et al.; Guzzetti et al.; Folta et al.; Topputo; Nunes et al.) | api.crossref.org | DOIs and metadata confirmed (see entries) |
| 16 | OpenAlex abstract lookups and "cites:" for 2412.20023, 2308.03960 and Jesick & Ocampo 2011 | api.openalex.org | Citing works: Graebner & Beeson JAS; Litteri et al. VAE; Briden et al.; about 40 free-return papers incl. He et al. 2017, Luo et al. 2012, Du & Li controllable-set papers 2026, Zhu et al. 2025 |
| 17 | uncertain quality-diversity robustness MAP-Elites noisy evaluation reproducibility archive | Web (ext.) | Flageat & Cully 2302.00463; deep-grid MAP-Elites; Extract-QD |
| 18 | behavior repertoire archive adaptation spacecraft OR satellite OR orbital "intelligent trial and error" OR "quality diversity" | Web (ext.) | Robotics only. No space application of ITE / repertoires |
| 19 | Topputo optimal low-energy Earth-Moon transfers database Pareto 2013 | Web (ext.) | Topputo 2013; Oshima et al. (leads) |
| 20 | JPL three-body periodic orbits tool and API docs | ssd.jpl.nasa.gov, ssd-api.jpl.nasa.gov | API documentation fetched |
| 21 | arXiv API: "quality-diversity" AND (spacecraft OR orbit OR astrodynamics OR lunar); "MAP-Elites" AND (spacecraft OR orbital OR satellite); "novelty search" AND (spacecraft OR orbital); "three-body" AND (diversity OR diverse) AND trajectory; "free-return"/"free return" AND (lunar OR moon) | export.arxiv.org | 0 relevant QD × space hits. The three-body × diversity query returns the Beeson group papers only |
| 22 | GECCO quality diversity aerospace design space exploration orbit transfer illumination | Web (ext.) | QD for model rockets (GECCO'25, arXiv 2504.02177, not fetched); SAIL. No orbit transfer |
| 23 | diverse trajectory catalog robustness to perturbation "solution diversity" spacecraft trajectory design held-out | Web (ext.) | Takubo et al. 2410.05585; Miceli dissertation (Bosanac group). No catalog-robustness study |
| 24 | Izzo machine learning spacecraft trajectory design survey neural network optimal trajectories 2019 | Web | Izzo, Sprague & Tailor 1802.00180 (background, not fetched) |
| 25 | Miceli Bosanac motion primitives clustering CR3BP diverse trajectory design | Web (ext.) | Miceli et al. SciTech 2024 (full text fetched); Smith & Bosanac 2023 |
| 26 | "quality diversity" cislunar OR "Earth-Moon" OR "gravity assist" OR "low-thrust" trajectory archive | Web (ext.) | None relevant |
| 27 | free return trajectory sensitivity translunar injection errors Monte Carlo dispersion perilune altitude | Web (ext.) | No quantitative dispersion study found. Hobbyist claim (unverified); NASA nodal-cycle paper (not fetched) |
| 28 | "controllable set" Earth–Moon free-return trajectories interval analysis Du Li 2026; "Data and knowledge dual-driven design method for Earth–Moon free-return orbit" | Web (ext.) | Du & Li 2026 papers; Lu et al. 2023; Zhu et al. (abstracts not retrievable) |
| 29 | Direct fetches: arXiv 1504.04909, 2012.04322, 2003.04407, 2302.00463, 2410.05585, 2502.01907, 2606.26367 | arXiv | Mouret & Clune 2015; Chatzilygeroudis et al. QD review 2020; Mouret & Maguire 2020; others as noted. 2606.26367 (cislunar primer) does not cover free returns, so it was dropped |

Second pass (controllable-set freeze blocker), also on 2026-10-18:

| # | Query | Source | Notable hits |
|---|---|---|---|
| 30 | Full read of Zhu et al. 2025 JNUDT (local PDF, pdftotext and page renders) plus its reference list | local | §9. No Du & Li citation. Reachability reference: Yang et al. 2024 JSEE |
| 31 | 自由返回轨道 可控集 区间分析 | Web (ext.) | Peng, Lu & Zhang 2026 CSST (interval-number reachable domain; PDF downloaded and read); Zhu 2025 JNUDT; Li Z. et al. 2023 Acta Aeronaut. Astronaut. Sin. (adaptive Newton–Raphson, not opened) |
| 32 | Du Zhenyu Li Haiyang controllable set free-return trajectories interval analysis | Web (ext.) | Lu et al. 2023 Acta Astronautica (metadata). The first author of the Du papers is *Zhenjiang* Du |
| 33 | OpenAlex / Crossref / Semantic Scholar lookups for AST 113609, AST 112680, CMDA 10321-2, AST 111492, Acta 2023.05.028, ASR 2023.07.023, JSEE 2023.000117, TAES 2025.3618994, Acta 2026.03.046 | api.openalex.org, api.crossref.org, api.semanticscholar.org | AST 112680 abstract (SSRN preprint record); TAES 2025 abstract; ASR 2023 abstract (Research Square, CC BY); JSEE marked gold OA. Reference lists of 113609, CMDA and 111492 harvested. No abstract for 113609 or 111492 |
| 34 | SSRN 6169467; Research Square rs-2307021 PDF; ScienceDirect PII pages | ssrn.com, researchsquare.com, sciencedirect.com | All returned 403 (Research Square API metadata did load: CC BY 4.0, posted 2022-11-28) |
| 35 | jseepub "Fast solution to the free return orbit's reachable domain" | Web; jseepub.com | Yang et al. 2024 PDF downloaded from the journal and read |
| 36 | 基于可控域的定点返回轨道全局灵敏度分析 系统工程与电子技术 2023 | Web (ext.) | Not found online (lead kept) |
| 37 | 杜振江 李海阳 可控集 自由返回轨道; 地月自由返回轨道 可控集 快速计算 区间分析 国防科技大学 李海阳 | Web (ext.) | No Chinese-language Du paper found. CSST 2026 papers (Peng et al.; Wang et al., re-entry region, not opened) |
| 38 | "Zhenjiang Du" OR "Du Zhenjiang" lunar controllable set free-return | Web (ext.) | Guo, Du et al. 2025 JPCS (CC BY, read); "Uncertainty propagation of the lunar free return orbits and ATK simulation" (Huo et al., found via the IOP related-article list) |
| 39 | "Uncertainty propagation of the lunar free return orbits and ATK simulation" | Web; api.crossref.org; iopscience | Huo et al. 2025 (read); Burnett et al. arXiv 2607.03424 (planar-CR3BP Artemis II-like free return with MCC; read); Peng, He & Zhang 2016 JDSE (abstract read) |
| 40 | Exact title of AST 113609; "velocity surfaces" controllable set … 2026 abstract; Springer page for CMDA 10321-2 | Web (ext.); link.springer.com | No abstract for 113609. CMDA abstract and keywords read on the Springer page |

Second-pass backward hops: the Crossref reference lists of AST 113609, CMDA 10321-2 and AST 111492, and the printed reference lists of Zhu 2025, Yang 2024, Guo 2025, Huo 2025 and Burnett 2026, were checked. New leads are listed under Unverified leads. None of them cites QD, novelty search or illumination work.

Backward hop: the reference list of 2412.20023 was checked. It contains no QD, novelty-search or illumination citations. Its global-search references are basin hopping, GA, DE, PSO and MBH variants (Englander et al.; Addis et al.), plus Russell 2007 and Oshima et al. 2017 for PCR3BP global trade studies.
