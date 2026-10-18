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

## How this study differs

- **Method × domain:** no published application of MAP-Elites, quality-diversity or novelty search to spacecraft trajectory design was found in any dynamics model (CR3BP, MGA, low-thrust, GTOC). The searches covered the arXiv API, the web and citation hops. This is still a *negative search result*, not proof: GECCO and CEC proceedings and AAS conference papers are poorly indexed.
- **Diversity target:** prior astrodynamics diversity work produces diversity through learned generative samplers (Beeson group), motion-primitive graphs (Bosanac group), continuation families (Pavlak; Jesick & Ocampo) or domain mapping (He et al.). Gravity Atlas would compare a *descriptor-space archive* against these at a matched propagation budget. None of the works above report budget-matched comparisons across these paradigms.
- **Claim type:** robust-design papers (Sinha & Beeson 2025; Kumagai & Oguri 2025) optimize *each* solution for robustness. Gravity Atlas asks a different question: whether a *catalog* produced without robustness in the objective still contains usable solutions when execution errors or constraints change after the search. This is the Cully et al. argument transplanted to trajectories, and it was not found applied there.
- **Problem class:** impulsive, ballistic free returns. All the generative-model and diffusion work above is low-thrust optimal control in high-dimensional transcriptions.
- **Honest evaluation of QD's known weakness:** per-solution fragility of elites (Flageat & Cully) can be tested directly with held-out perturbations. A null or negative result is informative to both communities.

## Novelty risk

**Medium.**

- *Lower risk:* no QD-for-trajectories paper was found. The archive-for-adaptation argument has not been tested in astrodynamics. The closest ML work (Beeson group) targets diversity with generative models and does not evaluate held-out perturbation robustness of a catalog.
- *Higher risk 1, low dimension:* the provisional search space has about 3 parameters (departure phase and two impulse components). In 3-D, a dense quasi-random or grid map of the whole feasible domain is affordable (compare He et al. 2017; Luo et al.'s large-sample global-feature study). An expert reviewer will ask why illumination is needed when the domain can be enumerated. Unless the comparison includes a budget-matched dense map and reports where the curves cross, the "QD gives better coverage" result may be trivial.
- *Higher risk 2, conventional catalogs:* continuation families of free returns (Pavlak; Jesick & Ocampo) are cheap and well understood. An astrodynamics reviewer will treat them as the default diversity baseline.
- *Higher risk 3, the robustness claim may be trivially negative:* ballistic free returns are known to be very sensitive to injection errors, and operational practice relies on midcourse corrections. If "robust" means "remains feasible with no correction," almost every solution may fail, and the metric loses discriminative power. Robustness must be defined against a declared correction budget or margin, with that definition frozen before results are seen.
- *Risk from a fast-moving neighbor:* the Beeson group publishes regularly on diversity in CR3BP global search. A QD comparison from them is plausible.

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
- Z. Du, H. Li, "A fast calculation method for the controllable set of Earth–Moon free-return trajectories based on interval analysis," Aerosp. Sci. Technol. 179:113609 (2026), https://doi.org/10.1016/j.ast.2026.113609; Z. Du, H. Li, H. Wang, H. Jing, "Time-dependent controllable set for contingency return trajectories under spatiotemporal constraints in CR3BP," AST 176:112680 (2026), https://doi.org/10.1016/j.ast.2026.112680; Z. Du, H. Jing, H. Li, H. Wang, "Fast approximation and analysis for time-dependent controllable set of Earth–Moon transfer orbits via velocity surfaces method," CMDA 138 (2026), https://doi.org/10.1007/s10569-026-10321-2. Metadata verified for all three. **High priority to read:** "controllable sets" of free returns may already characterize the set of departure conditions from which a feasible return exists. That is a reachability-style competitor to a robustness-of-catalog metric.
- B. Zhu, H. Li, L. Lu, Z. Du, L. Yang, "Data and knowledge dual-driven design method for Earth–Moon free-return orbit," AST 170:111492 (2025/2026), https://doi.org/10.1016/j.ast.2025.111492. Metadata verified. An ML-assisted free-return design, possibly close to the generative-baseline idea. Read before freezing the protocol.
- T. Dong, Q. Luo, C. Han, "Bayesian Optimization of crewed lunar free return abort trajectory," Acta Astronautica 201:288 (2022), https://doi.org/10.1016/j.actaastro.2022.08.052. Metadata verified.
- H. Zhang, C. Han, Q. Luo, J. Li, "A UKF based design method for lunar free-return trajectories," AST 48:272 (2015), https://doi.org/10.1016/j.ast.2015.11.007. Metadata verified.
- Lu et al., "Study on controllable set of contingency return trajectories for crewed lunar missions based on interval analysis," Acta Astronautica 210:322 (2023). Seen only in search results.
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

Backward hop: the reference list of 2412.20023 was checked. It contains no QD, novelty-search or illumination citations. Its global-search references are basin hopping, GA, DE, PSO and MBH variants (Englander et al.; Addis et al.), plus Russell 2007 and Oshima et al. 2017 for PCR3BP global trade studies.
