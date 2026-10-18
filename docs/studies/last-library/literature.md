# The Last Library — Literature scan (2026-10-18)

Scope: roadmap step S3 for the second study, a focused nearest-neighbor scan for one question. It is not an open-ended review. A reference appears under "Closest works", "How this study differs" or "Baselines implied" only if its arXiv ID, DOI landing page, publisher record, OpenAlex/Crossref record with abstract, or full text was fetched and the title, authors and year matched. The **Verified** line on each entry says what was fetched. Items whose identity was found but whose content was not read are under "Unverified leads" and must not be cited for content.

No computation was run for this scan.

## Question as scanned

From `README.md` (v0):

> When does transmitting a strictly limited external memory between otherwise fresh agents enable cumulative improvement on a family of previously unseen but structurally related tasks?

Decision D15 adds the angle: *a byte budget on inherited memory, compared with Cook et al. 2024.*

| Component | Value |
|---|---|
| Domain | Procedurally generated 2D grid navigation with a hidden latent rule family (terrain costs, door/key rules, hazard cues); held-out layouts per generation; possible rule change |
| Method | Small interpretable (symbolic / Bayesian) learner per lifetime; at retirement writes ≤ 256 bytes; the next generation starts fresh except for that record |
| Claim type | Cumulative (multi-generation) improvement on held-out tasks as a function of memory budget and memory encoding; negative transfer under rule change |
| Baselines | No inheritance; example memory; rule memory; random / shuffled / unrelated-predecessor memory; oracle with the true rule |

## Summary of the finding

The nearest neighbor is **not** Cook et al. (2024). It is the Stanford line of work by **Prystawski, Arumugam & Goodman**: cumulative culture as Bayesian RL with a **rate-limited** channel between generations (CogSci 2023), and an exact rate-distortion follow-up that sweeps the channel rate in bits (arXiv 2025). The 2023 paper already compares **abstract (rule-like) vs. concrete (example-like) messages under a message-size limit** across generations, in a small symbolic task with a Bayesian agent. That is close to the README's arms B and C. What remains open is narrower than the README suggests (see "How this study differs").

## Closest works

### 1. Prystawski, Arumugam & Goodman (2023): cultural RL on a limited channel. Closest overall.
- B. Prystawski, D. Arumugam, N. D. Goodman, "Cultural reinforcement learning: a framework for modeling cumulative culture on a limited channel," Proc. 45th Annual Conference of the Cognitive Science Society (CogSci 2023). Preprint: PsyArXiv, https://doi.org/10.31234/osf.io/q4tz8. Code: https://github.com/benpry/cultural-rl (seen in search results; not inspected).
- **Verified:** OpenAlex record (DOI, authors, year, abstract) and the OSF preprint API record. Full text downloaded from OSF and read completely (method, results, discussion, references). The CogSci 2023 venue is from the first author's publication page (fetched).
- **What they did:**
  - Framed cumulative culture as Bayesian RL in which each generation passes a message under a rate limit. The message is chosen to minimize the next agent's expected regret ("cultural regret") subject to I(L_k, E_k; L_{k+1}) ≤ R, i.e. a distortion-rate problem.
  - Task: a crafting game with unknown deterministic recipe dynamics. Items have color and shape, and the generated games have abstract structure (only color, or only shape, determines the output).
  - Agent: a PSRL (Thompson sampling) learner with BFS planning. Knowledge is a set of DSL statements; abstract statements use "any" wildcards.
  - Rate limit operationalized as a **statement capacity** of 1, 3, 5, 7, 10 or unlimited. This follows from bounding mutual information by maximum message surprisal under a uniform prior over statements.
  - Compression heuristic: anti-unification (least general generalization) of statement pairs, else drop the most specific and most redundant statement.
  - Design: 5 episodes per generation, 10 generations, 100 chains per game on 100 generated games (10,000 chains).
  - Results: abstraction beats concrete-only messages; low capacity plateaus at a low level; a sawtooth (learn within a lifetime, lose at transmission) appears at low capacity; **capacity 10 with abstraction reaches optimal performance faster than an unlimited channel**, which never abstracts. Selective social learning on learning-phase scores slows learning (an explore–exploit confound).
- **What they did NOT do:**
  - Each chain plays **one fixed game**: the dynamics are fixed within a chain, and only start states and goals vary between episodes. There is no held-out *game* drawn from the same rule family, and no rule change or negative-transfer test.
  - Capacity is counted in **statements, not bits**. Abstract and concrete messages are compared at equal statement count, not equal information, so the "abstraction wins" result is not bit-matched.
  - The example arm is not pedagogically selected; it drops the most specific statements.
  - No random or shuffled-message control, no corruption ablation, no oracle-regret curve.
  - The domain is crafting, not navigation.

### 2. Prystawski, Arumugam & Goodman (2025): lossy communication constrains iterated learning
- B. Prystawski, D. Arumugam, N. D. Goodman, "Lossy communication constrains iterated learning," arXiv:2511.18220 (v1 22 Nov 2025, v2 1 Dec 2025). https://arxiv.org/abs/2511.18220. Code: https://github.com/benpry/lossy-iterated-learning (seen in search results; not inspected).
- **Verified:** arXiv abstract page fetched; full HTML (v2) read for model, tasks, controls, results and limitations.
- **What they did:**
  - Bayesian learners estimate a categorical distribution (Dirichlet–Categorical, K = 3; one observation per learner). Each generation passes its posterior through a channel computed **exactly with Blahut–Arimoto**, with rate measured in **bits** and swept via β.
  - Performance is nonlinear in rate. Very low rates leave chains worse than the first generation, because they produce misinformative priors. Roughly 5–7 bits are needed to beat individual learning; above about 9–10 bits, chains approach perfect posterior passing.
  - Theorem: a Fano-type bound on the final learner's error in terms of information transmitted and information gained from new observations. At zero rate, no number of generations recovers progress.
  - Appendix: selective social learning partially compensates for low rates.
- **What they did NOT do:** no sequential decision task, no held-out tasks (the same true distribution persists along a chain), no environmental change, no examples-vs-abstractions comparison, no random-message control. The authors themselves list a small discrete hypothesis space and the lack of open-ended domains as limitations. They do not cite Cook et al. or emergent-communication work.

### 3. Cook, Lu, Hughes, Leibo & Foerster (2024): cultural accumulation in RL (named in D15)
- J. Cook, C. Lu, E. Hughes, J. Z. Leibo, J. Foerster, "Artificial Generational Intelligence: Cultural Accumulation in Reinforcement Learning," NeurIPS 2024 (Advances in Neural Information Processing Systems 37), https://doi.org/10.52202/079017-1907. arXiv:2406.00392, https://arxiv.org/abs/2406.00392.
- **Verified:** arXiv abstract page fetched; Crossref record of the NeurIPS DOI matched title, authors and year; full HTML (v2) read for environments, transmission mechanism, baselines and limitations.
- **What they did:**
  - Two notions of a generation. *In-context*: meta-RL (RL²-style) agents observe the best agent of the previous generation acting in their environment, with observation probability annealed from 1 to 0 across trials. *In-weights*: a freshly initialized network is trained while observing the previous generation.
  - Environments: Memory Sequence, Goal Sequence (7×7 gridworld with a hidden goal order, 3×3 egocentric view), and a partially observable TSP.
  - Accumulating agents beat single-lifetime baselines with the same cumulative experience. Too-accurate oracles in training hurt accumulation (over-reliance on social learning).
- **What they did NOT do:**
  - Transmission is **behavioral observation**, not an explicit record. There is no bit or byte budget; the only bandwidth-like device is noise on the training oracle, which they relate to Prystawski et al. (2023) as an "information-rate limit".
  - No interpretable memory content, no example vs. rule encoding, no held-out rule change. Selection of the best previous agent is external.
- **Relevance:** Cook et al. supply the *matched-cumulative-experience single-lifetime baseline*, which Last Library should copy. They are not the nearest neighbor for the *budgeted record* question.

### 4. Iterated learning and posterior passing in cognitive science: Beppu & Griffiths (2009); Whalen, Maurits, Pacer & Griffiths (2014)
- A. Beppu, T. L. Griffiths, "Iterated learning and the cultural ratchet," Proc. 31st Annual Conference of the Cognitive Science Society (2009). https://escholarship.org/uc/item/24w0t8c9
- A. Whalen, L. Maurits, M. Pacer, T. L. Griffiths, "Cultural evolution with sparse testimony: when does the cultural ratchet slip?," Proc. CogSci 2014. https://escholarship.org/uc/item/6rx8k9xq
- **Verified:** OpenAlex records with abstracts for both (title, authors, year, eScholarship landing page). eScholarship itself returned HTTP 403, so full texts were **not** read. The Beppu & Griffiths result is also summarized in the Prystawski et al. (2023) full text, which was read.
- **What they did:**
  - Beppu & Griffiths: rational Bayesian analysis plus human experiments on function learning. Merely observing other learners' behavior (example points) does not produce cumulative evolution, but communicating the plausibility of hypotheses (posterior passing) does.
  - Whalen et al.: relax full posterior passing. Much sparser testimony can still let learners improve over generations, with limits on which concepts can be learned. Tested with an iterated category-learning experiment.
- **What they did NOT do:** no sequential decision or navigation task, no explicit byte budget, no held-out task family or rule change.
- **Relevance:** this is the origin of the README's "example memory vs. rule memory" contrast. That contrast is settled in principle (rules/posteriors > examples) for stationary tasks. The study must not present it as new.

### 5. Iterated learning, compression and the transmission bottleneck: Kirby et al. (2008, 2015); Griffiths & Kalish (2007); Imel et al. (2025)
- S. Kirby, H. Cornish, K. Smith, "Cumulative cultural evolution in the laboratory: An experimental approach to the origins of structure in human language," PNAS 105 (2008). https://doi.org/10.1073/pnas.0707835105
- S. Kirby, M. Tamariz, H. Cornish, K. Smith, "Compression and communication in the cultural evolution of linguistic structure," Cognition (2015). https://doi.org/10.1016/j.cognition.2015.03.016
- T. L. Griffiths, M. L. Kalish, "Language Evolution by Iterated Learning With Bayesian Agents," Cognitive Science 31 (2007). https://doi.org/10.1080/15326900701326576
- N. Imel, J. Culbertson, S. Kirby, N. Zaslavsky, "Iterated language learning is shaped by a drive for optimizing lossy compression," bioRxiv (2025). https://doi.org/10.1101/2025.05.23.655237
- **Verified:** Crossref metadata (Kirby 2008, Kirby 2015, Griffiths & Kalish) and OpenAlex abstracts (all four) fetched and read.
- **What they did:** transmission through a bottleneck makes cultural systems more learnable and more structured (Kirby 2008). Structure arises from a trade-off between compressibility (learning) and expressivity (communication) (Kirby 2015). With posterior-sampling Bayesian learners, iterated learning converges to the prior; with MAP learners, the outcome also depends on how much information passes between generations (Griffiths & Kalish). Iterated learning in the lab drives semantic systems toward the information-theoretic limit of efficient lossy compression (Imel et al.).
- **What they did NOT do:** the transmitted object is a *language or semantic system*, not task knowledge for a decision problem. Success is measured as structure and learnability, not as regret on held-out tasks.
- **Relevance:** these works supply the theoretical vocabulary (bottleneck, compression vs. expressivity, convergence to the prior). They also warn that "cumulative improvement" in an iterated chain can be driven by learner priors, not by accumulated evidence. The no-inheritance and shuffled controls are there to separate the two.

### 6. Human transmission chains with free-text messages in games: Tessler et al. (2021); Colas et al. (2025, ICLR 2026)
- M. H. Tessler, J. Madeano, P. A. Tsividis, B. Harper, N. D. Goodman, J. B. Tenenbaum, "Learning to solve complex tasks by growing knowledge culturally across generations," arXiv:2107.13377 (2021; CogSci 2021 and NeurIPS 2021 Cooperative AI workshop). https://arxiv.org/abs/2107.13377
- C. Colas, T. Mills, B. Prystawski, M. H. Tessler, N. Goodman, J. Andreas, J. Tenenbaum, "Language and Experience: a computational model of social learning in complex tasks," arXiv:2509.00074 (2025). https://arxiv.org/abs/2509.00074. ICLR 2026 per the proceedings listing seen in search results (not fetched).
- **Verified:** arXiv abstract page of 2107.13377 (v3) fetched. HTML of 2509.00074v2 fetched and read for model, iterated-learning design, controls and limitations.
- **What they did:**
  - Tessler et al.: humans play minimalist video games with two lives each and leave a written message for the next player. Multigenerational learning tracks the trajectory of individuals with unlimited lives. Messages convey dynamics, goals, risks and strategies.
  - Colas et al.: Bayesian inference over VGDL program world models, combining experience with language advice. An LLM is turned into a model of how advisors speak. Iterated chains of 10 generations on each of 10 games improve, and knowledge transfers between humans and models. Occasional regressions occur in games with several viable strategies.
- **What they did NOT do:** **no length or bit limit** on messages (none stated in what was read); each chain stays on the same game; no rule change; no random-message control.
- **Relevance:** the strongest evidence that "rules in a record" accumulate in game-like tasks. Colas et al. is the obvious bridge to a future LLM version of Last Library. It does not impose the budget.

### 7. Memory, vertical transmission and environmental change: Ammar, Fogarty & Kandler (2025)
- M. Ammar, L. Fogarty, A. Kandler, "Memory, innovation and vertical learning," PLoS Computational Biology 21(12): e1013785 (2025). https://doi.org/10.1371/journal.pcbi.1013785
- **Verified:** PLOS article page fetched (title, authors, date, DOI, model summary and findings); OpenAlex abstract matched.
- **What they did:** agent-based model with repertoires of up to 500 cultural variants, use-dependent forgetting, and an environment switching between two states. Offspring inherit the parent's whole repertoire (without usage history). Vertical learning hurts under very fast environmental change, because repertoires become large and poorly structured, and helps in stable environments, where it preserves variants adapted to the other state. Its benefit peaks at intermediate change rates.
- **What they did NOT do:** capacity is counted in variants, not bits; there is no task learning, no held-out tasks, and no encoding choice (examples vs. rules).
- **Relevance:** independent theoretical support for the hypothesis that inherited memory can cause **negative transfer when the environment changes**, and that its value depends on the change rate. This is the part of Last Library least covered by work 1–2.

### 8. Bounded memory in LLM agents: retention vs. consolidation under a token budget (Kang et al. 2026)
- Q. Kang, M. Liu, S. Kai, K. Liang, Z. Tang, Y. Cui, T. Zhong, "Retain or Consolidate? Budget-Dependent Operator Selection for Language Agent Memory," arXiv:2607.17545 (2026). https://arxiv.org/abs/2607.17545
- **Verified:** arXiv API record (title, authors, date, abstract) fetched. Abstract only.
- **What they did:** under a context-window budget, *retention* (keep raw records: example memory) and *consolidation* (merge, abstract or rewrite: rule memory) trade coverage per token against loss of query-critical details. The preferred operator changes with relative budget pressure. They learn a lightweight selector.
- **What they did NOT do:** not intergenerational; no fresh successor agents, no latent rule family, no rule change.
- **Relevance:** an LLM-agent echo of the README's B-vs-C contrast at a budget. The "consolidate when tight, retain when loose" pattern is a prediction Last Library can test cleanly in a small exact setting.

## Other verified works (context, not nearest neighbors)

- **Shuvaev, Lachi, Koulakov & Zador**, "Encoding innate ability through a genomic bottleneck," PNAS 121(38): e2409160121 (2024), https://doi.org/10.1073/pnas.2409160121. Verified: Crossref and OpenAlex abstract. Inherited *weights* are compressed by orders of magnitude through a "genome". For complex tasks, the bottleneck acts as a regularizer that improves transfer to novel tasks. This is the neural, evolutionary counterpart of a byte budget on inheritance, and supports the idea that a bottleneck can *help* held-out transfer.
- **Thompson, van Opheusden, Sumers & Griffiths**, "Complex cognitive algorithms preserved by selective social learning in experimental populations," Science (2022), https://doi.org/10.1126/science.abn0915. Verified: Crossref and OpenAlex abstract. Large human transmission experiment: complex sorting algorithms persist only with selective social learning, otherwise chains converge to transmissible but weaker algorithms.
- **Tennie, Call & Tomasello (2009)**, "Ratcheting up the ratchet," Phil. Trans. R. Soc. B, https://doi.org/10.1098/rstb.2009.0052; **Mesoudi & Thornton (2018)**, "What is cumulative cultural evolution?," Proc. R. Soc. B, https://doi.org/10.1098/rspb.2018.0712. Verified: Crossref metadata only. Cite only for the term "cumulative cultural evolution" until the abstracts are read.
- **Dalmaijer**, "Cumulative culture spontaneously emerges in artificial navigators who are social and memory-guided," arXiv:2206.06281 (2022). Verified: arXiv abstract. Cumulative route efficiency emerges from minimal agents. A warning that "cumulative improvement" can arise from very simple mechanisms, so the claim needs controls.
- **Roberts-Gaal, Callaway & Cushman**, "Ratchet or hatchet? Modeling the cultural evolution of simplified construals," CogSci 2026 (eScholarship https://escholarship.org/uc/item/6hx4q2zs). Verified: OpenAlex record with abstract (full text blocked, HTTP 403). Maps used for navigation evolve under a utility–complexity trade-off tied to representational capacity. The evolutionarily stable map is often more or less complex than the optimal one. Close in spirit (navigation, capacity-limited representations), but it is about map complexity under selection, not a budgeted record and held-out regret. **Read the full text before the protocol freeze.**
- **Neural iterated learning:** Ren, Guo, Labeau, Cohen & Kirby, "Compositional Languages Emerge in a Neural Iterated Learning Model," ICLR 2020, arXiv:2002.01365; Vani, Schwarzer, Lu, Dhekane & Courville, "Iterated learning for emergent systematicity in VQA," ICLR 2021, arXiv:2105.01119. Verified: arXiv API abstracts. The bottleneck as a pressure toward compositional, generalizable representations in neural agents.
- **Emergent communication with channel constraints:** Kharitonov, Chaabouni, Bouchacourt & Baroni, "Entropy Minimization In Emergent Languages," ICML 2020, arXiv:1905.13687; Tucker, Shah, Levy & Zaslavsky, "Towards Human-Agent Communication via the Information Bottleneck Principle," arXiv:2207.00088 (2022). Verified: arXiv API abstracts. Within-generation agent-to-agent channels; relevant only as information-bottleneck methodology.
- **LLM cultural and generational work:** Vallinder & Hughes, "Cultural Evolution of Cooperation among LLM Agents," arXiv:2412.10270 (2024); Perez et al., "Cultural evolution in populations of Large Language Models," arXiv:2403.08882 (2024); Perez et al., "When LLMs Play the Telephone Game," arXiv:2407.04503 (2024); Jha, Kleiman-Weiner & Jaques, "From Solo to Social Learning: Characterizing Recursive Social Improvement in LLMs," arXiv:2609.38516 (2026); Paolo et al., "TerraLingua," arXiv:2603.16910 (2026); Guo, Wu & Yiu, "Model Collapse as Cultural Evolution," arXiv:2605.23054 (CoNLL 2026). Verified: arXiv abstract pages or API records. Jha et al. is the closest LLM work: populations revise "skill files" under one shared token budget. Copying did not beat independent learners at equal cost and concentrated the population on fewer discoveries. None of these imposes a size limit on the inherited record and tests held-out rule families.
- **Skill and experience libraries:** Wang et al., "Voyager," arXiv:2305.16291 (2023); Wang, Mao, Fried & Neubig, "Agent Workflow Memory," arXiv:2409.07429 (2024); Zhao et al., "ExpeL," arXiv:2308.10144 (AAAI 2024); Mohtashami et al., "Social Learning: Towards Collaborative Learning with Large Language Models," arXiv:2312.11441 (2023); Reddy, "Selective Memory Retention for Long-Horizon LLM Agents," arXiv:2606.29178 (ICML 2026). Verified: arXiv abstracts. These libraries grow without a hard size limit, except Reddy's bounded memory (50 entries, eviction policies). Mohtashami et al. compare transferring *instructions* vs. *synthetic examples* between LLMs, an LLM echo of rules vs. examples, but not across generations or under a bit budget.
- **Inheritance in evolutionary RL and robotics:** Feng et al., "Genes in Intelligent Agents," arXiv:2306.10225; de Bruin, Glette & Ellefsen, "Lamarckian Inheritance in Dynamic Environments," arXiv:2605.15769 (2026). Verified: arXiv abstracts. In the second, Lamarckian inheritance underperforms only when environmental changes are both conflicting and unpredictable. This parallels Ammar et al. and the rule-change test.
- **Information bottleneck for transfer in RL:** Goyal et al., "InfoBot," ICLR 2019, arXiv:1901.10902; Teh et al., "Distral," arXiv:1707.04175. Verified: arXiv abstracts. Bottlenecked or distilled priors for transfer across tasks, within one learner. Background only.
- **Capacity trade-offs:** Turner, Arumugam, Nelson & Griffiths, "Trade-Offs Between Tasks Induced by Capacity Constraints Bound the Scope of Intelligence," OSF preprint (2025), https://doi.org/10.31219/osf.io/ft3e6_v2. Verified: OpenAlex abstract. Found as a forward citation of work 1. Rate-distortion RL for capacity-induced trade-offs across tasks; relevant to the rule-change analysis.

## How this study differs

Only the differences that survive the scan are listed.

- **Held-out tasks from a latent rule family, not one fixed task per chain.** Prystawski et al. (2023, 2025), Tessler et al. and Colas et al. keep a single task per chain. Last Library would test each generation on *new layouts* governed by the same latent rule. Inherited knowledge must then be *rule-level* to help, and example-level memory can be specific to a single layout. This is a real but modest difference.
- **Budget in exact bits, with bit-matched encodings.** Prystawski (2023) matches statement counts, not bits. Prystawski (2025) uses exact bits but has no encoding choice. Last Library can compare example, rule and (optionally) rate-distortion-optimal codes **at an identical bit count**, with a pedagogically selected example arm (in the style of Shafto et al.; see leads) so that the rule-over-example advantage is tested against the strongest example baseline.
- **Rule change and negative transfer, measured on held-out tasks.** None of works 1–6 changes the latent rules. Ammar et al. (2025) and de Bruin et al. (2026) show, in other model classes, that inheritance can hurt under change. Combining a bit budget, encoding type and an unannounced rule change in a sequential decision task was not found.
- **Explicit placebo controls.** Random, shuffled and unrelated-predecessor records are not used in works 1–6. They are cheap, and they separate "information transferred" from "prior or format effects", the iterated-learning confound of Griffiths & Kalish (2007).
- **Navigation domain and public, inspectable memory.** This is a presentation difference, not a scientific one, and must not be sold as novelty.
- **Versus Cook et al. (D15):** Cook et al. transmit through observed behavior with no budget. Last Library uses an explicit, inspectable, budgeted record. The difference is real, but Cook et al. is the wrong primary comparison: the budgeted-record framing already exists in Prystawski et al.

## Novelty risk

**High for the README question as written; medium for the sharpened question recommended below.**

- *High:* "Does a strictly limited inherited record enable cumulative improvement, and is rule memory better than example memory?" is answered in principle by Beppu & Griffiths (2009), Whalen et al. (2014) and Prystawski et al. (2023). "How does performance scale with channel capacity?" is answered, with an exact rate-distortion treatment and a Fano-type bound, by Prystawski et al. (2025). A reviewer from computational cognitive science would see the v0 study as a replication in a gridworld.
- *Medium, after sharpening:* the combination of (i) held-out tasks from a latent family, (ii) bit-matched encodings, and (iii) unannounced rule change with negative-transfer measurement was not found in any work read. Ammar et al. and de Bruin et al. make the rule-change hypothesis plausible but test it without task learning or bit budgets.
- *Fast-moving neighbors:* the Stanford group (Prystawski, Arumugam, Goodman; Colas, Tessler, Tenenbaum) is actively extending this line, and Colas et al. already pairs it with LLMs. A held-out or rule-change extension from them is plausible within a year. On the LLM side, Jha et al. (2026) and the memory-budget papers (Kang et al. 2026; Reddy 2026) are converging on budgeted, copyable skill files.
- *Design risk that could make results trivial:* 256 bytes is 2,048 bits. A gridworld latent rule (e.g. 6 terrain colors × 4 cost levels ≈ 12 bits) fits in 2 bytes. At 256 bytes **the budget would not bind**, and every encoding that carries the rule would tie. Following Prystawski (2025), where the action is at about 5–10 bits, the budget sweep must be placed *around the entropy of the latent rule*, not at 0 / 64 / 256 bytes.
- *Second triviality risk:* if one lifetime can identify the whole rule, inheritance gives one-shot transfer, not *cumulative* improvement. Generation 2 saturates and "cumulative" is unfalsifiable. The task must make single-lifetime identification partial (Cook et al.'s matched-experience baseline exposes this).

## Baselines implied by prior work

1. **No inheritance** (README A): every generation starts from the prior.
2. **Single learner with matched cumulative experience** (Cook et al.): one agent with G lifetimes of experience. This separates "culture" from "more data".
3. **Unlimited posterior passing** (Beppu & Griffiths; Prystawski 2023/2025): the upper bound for any record. In a small discrete hypothesis space it is exact.
4. **Oracle with the true latent rule** (README): the regret reference.
5. **Example memory, two variants** (README B): (a) recency- or random-selected examples; (b) examples selected to maximize the receiver's posterior on the true hypothesis (pedagogical selection). Without (b), "rules beat examples" is a strawman.
6. **Rule memory** (README C): a quantized posterior summary or MAP rule, bit-matched to the examples.
7. **Rate-distortion-optimal code** (Prystawski 2025): Blahut–Arimoto over the hypothesis space, feasible only if that space is small (≲ 10⁴). This is the information-theoretic frontier against which the hand-designed encodings are scored. Optional but cheap in a small space.
8. **Placebo records** (README D): random bytes, a shuffled record from another lineage, an unrelated predecessor (a different rule family).
9. **Abstraction without a budget vs. with a budget** (Prystawski 2023 found capacity 10 beats unlimited). Check whether a budget-induced compression gain replicates on held-out layouts.

## Unverified leads

- P. Shafto, N. D. Goodman, T. L. Griffiths, "A rational account of pedagogical reasoning: Teaching by, and learning from, examples," Cognitive Psychology (2014), https://doi.org/10.1016/j.cogpsych.2013.12.004. Metadata verified (OpenAlex); no abstract available. Needed for the pedagogical-example baseline.
- M. Ammar, L. Fogarty, A. Kandler, "Social learning and memory," PNAS (2023), https://doi.org/10.1073/pnas.2310033120. Metadata verified (OpenAlex search); the content is known only from a search summary (memory and forgetting reverse the classic stability result for social learning).
- D. Arumugam, B. Van Roy, "Deciding What to Model: Value-Equivalent Sampling for Reinforcement Learning," NeurIPS 2022. Seen only in the reference list of work 1. It is the rate-distortion-in-Bayesian-RL basis of CultuRL.
- Krafft et al. (2021), "Bayesian collective learning emerges from heuristic social learning," Cognition 212:104469. Seen only in reference lists (works 1, 2).
- Tucker, Levy, Shah, Zaslavsky, "Trading off Utility, Informativeness, and Complexity in Emergent Communication," NeurIPS 2022. Seen in search results; it appears to be the conference version of arXiv:2207.00088 (verified above under a different title). Confirm before citing under the NeurIPS title.
- Colas et al. (2025) as ICLR 2026: the proceedings listing was seen in search results only; the arXiv version is verified.
- Bhoopchand et al., "Learning few-shot imitation as cultural transmission," Nature Communications (2023), https://doi.org/10.1038/s41467-023-42875-2. OpenAlex abstract read: real-time imitation as cultural transmission in RL agents. Background; it imposes no budget and has no generations of records.
- "Selection Is Not Learning: Choosing Model Futures," arXiv:2610.22029 (2026), and Marincat, "Portable Semantics, Private Dialects," arXiv:2609.11365 (2026). Abstracts read via the arXiv API. Inherited parameters or interfaces in self-evolving or LM societies, including interface-level negative transfer. Tangential; listed for completeness.
- Forward-citation coverage is **incomplete**. Semantic Scholar rate-limited most requests (HTTP 429), and OpenAlex citation coverage for these CogSci and arXiv items is sparse. The Cook et al. citation list from Semantic Scholar (14 items) was retrieved once and scanned. Citations of Prystawski 2023 and 2025 came only through OpenAlex (Turner et al. 2025) and web search. **Repeat a forward-citation pass on both Prystawski papers via Google Scholar before any novelty claim.**

## Search log

All searches ran on 2026-10-18. "Web (ext.)" means WebSearch in extended mode. "API" means direct arXiv, OpenAlex, Crossref or Semantic Scholar queries.

| # | Query / action | Source | Notable hits |
|---|---|---|---|
| 1 | arXiv 2406.00392 (direct) + HTML v2 | arXiv | Cook et al. 2024; read in full. Backward hop: Prystawski et al. 2023 ("information-rate limit") |
| 2 | cultural accumulation reinforcement learning generations agents 2025 | Web (ext.) | Only Cook et al. and summaries |
| 3 | intergenerational knowledge transfer limited memory budget agents bottleneck cumulative culture simulation | Web (ext.) | Dalmaijer 2022; Ammar et al. 2025; TerraLingua 2026 |
| 4 | cultural accumulation large language model agents generations notes passed to successor | Web (ext.) | Vallinder & Hughes 2024 |
| 5 | Cook Foerster cultural accumulation follow-up 2025 arXiv generational transmission agents | Web (ext.) | No direct follow-up found |
| 6 | iterated learning neural agents compositionality transmission bottleneck reinforcement learning | Web (ext.) | Ren et al. 2020; Vani et al. 2021; Kaszyński 2026 |
| 7 | Prystawski Arumugam Goodman cultural reinforcement learning limited channel cumulative culture | Web | CultuRL (CogSci 2023); eScholarship blocked (403) |
| 8 | OpenAlex search "cultural reinforcement learning limited channel" | API | PsyArXiv DOI 10.31234/osf.io/q4tz8; NeurIPS DOI for Cook et al. |
| 9 | OSF download q4tz8 | OSF | Full text of CultuRL read |
| 10 | OpenAlex cites: W4377138583 (CultuRL), W4415797509 (Cook) | API | Turner, Arumugam, Nelson & Griffiths 2025; little else (sparse coverage) |
| 11 | Semantic Scholar citations of arXiv:2406.00392 | API | Jha et al. 2026 (2609.38516); Cook et al. 2026; Perez et al. 2024 telephone game |
| 12 | "cultural reinforcement learning" Prystawski cited rate-distortion transmission chain generations 2024 2025 | Web (ext.) | Pointer to "Lossy communication constrains iterated learning" |
| 13 | Prystawski "communication capacity constrains cultural learnability" | Web | Stanford talk 2024 only |
| 14 | "lossy communication" iterated learning Prystawski Goodman | Web | arXiv:2511.18220; read in full (HTML) |
| 15 | LLM agents generational memory inheritance successor agent notes cumulative improvement held-out tasks 2026 arXiv | Web (ext.) | Agent-memory surveys; nothing generational with a budget |
| 16 | benprystawski.com/publications | Fetch | Confirms CogSci 2023 venue |
| 17 | genomic bottleneck encoding innate ability Shuvaev Koulakov Zador PNAS | Web | Shuvaev et al. 2024 |
| 18 | Tessler … arXiv 2107.13377 | Web + arXiv | Tessler et al. 2021 |
| 19 | Prystawski 2025 "language and experience" social learning … arXiv | Web | Colas et al. 2025 (2509.00074); read |
| 20 | arXiv 2609.38516, 2412.10270, 2305.16291, 2409.07429 (direct) | arXiv | Jha et al.; Vallinder & Hughes; Voyager; AWM |
| 21 | Crossref DOIs (Kirby 2008/2015, Griffiths & Kalish, Shuvaev, Thompson 2022, Cook NeurIPS, Tennie, Mesoudi & Thornton) | API | All matched |
| 22 | arXiv API batch (Ren, Kharitonov, Mohtashami, Perez, ExpeL, TerraLingua, Guo, Kaszyński, Cook 2026, Dalmaijer) | API | Abstracts read |
| 23 | "cultural accumulation" agents arXiv 2026 | Web (ext.) | No new budgeted-record work; De Marzo et al. 2026 (copying among AI agents) |
| 24 | bounded inherited message bits next generation learner transfer held-out environments communication budget cultural transmission computational model | Web (ext.) | Nothing matching; tangential self-evolving-LM papers |
| 25 | Tucker Levy Shah Zaslavsky trading off utility informativeness complexity emergent communication NeurIPS 2022 | Web | arXiv:2207.00088 |
| 26 | Beppu Griffiths 2009 "Iterated learning and the cultural ratchet" | Web + OpenAlex | Abstract read via OpenAlex; full text blocked |
| 27 | OpenAlex DOI lookups (Kirby, Griffiths & Kalish, Shuvaev, Thompson, Bhoopchand) + search "iterated learning and the cultural ratchet" | API | Whalen et al. 2014; Roberts-Gaal, Callaway & Cushman 2026 |
| 28 | Roberts-Gaal Callaway Cushman "ratchet or hatchet" … | Web | Seminar page only; OpenAlex record confirms CogSci 2026 eScholarship item |
| 29 | cumulative culture gridworld agents transmit compressed knowledge new environments latent rules generations simulation memory capacity | Web (ext.) | Cook et al.; JaxLife; nothing closer |
| 30 | "Lossy communication constrains iterated learning" | Web (ext.) | Code repo; Imel et al. 2025 bioRxiv; Yang et al. 2025 (MARL comm priors, tangential) |
| 31 | "Prystawski" "cultural reinforcement learning" cites … 2025 2026 | Web (ext.) | Yaman, Tian & Lindström 2026 PNAS (cites CultuRL; semantic knowledge and innovation) |
| 32 | Ammar Fogarty Kandler memory forgetting vertical learning … | Web | PLoS CB 2025 (read); PNAS 2023 (lead) |
| 33 | RL agents leave messages for future generations … Lamarckian inheritance bounded | Web (ext.) | Learngenes; de Bruin et al. 2026 |
| 34 | LLM agent memory size budget ablation insights transfer new agent cross-task held-out "memory budget" | Web | Kang et al. 2026 (retain vs. consolidate); Reddy 2026 |
| 35 | arXiv API batch (2607.17545, 2606.29178, 2306.10225, 2605.15769, 2512.03528, 1901.10902, 1707.04175, 2409.00853, 2508.04828, 2609.09150, 2207.00088, 2105.01119, 2610.22029, 2609.11365, 2412.14500, 2407.04503) | API | Abstracts read |
| 36 | Semantic Scholar citations of arXiv:2511.18220 | API | One unparseable CogSci proceedings entry; mostly rate-limited |

## Candidate sharpened research questions

Each question below survives the scan, in the sense that no work read answers it. Each one sets the budget around the entropy of the latent rule (in bits), not at 256 bytes, because of the triviality risk above. Compute estimates assume an exact Bayesian learner over a small discrete hypothesis space (≲ 10⁴ hypotheses), Dijkstra planning on ~9×9 grids, and pure NumPy. They are order-of-magnitude guesses, not measurements.

### Q1 (recommended): Does a budgeted rule record turn into a liability when the latent rule changes, and does the ranking of encodings flip?

> At a fixed inherited budget of B bits, swept from below to above the latent-rule entropy H, does a rule-encoded record outperform a bit-matched, pedagogically selected example record on held-out layouts while the rule family is stationary, and underperform it, and the no-inheritance baseline, in the first generations after an unannounced change of the latent rule?

- **Falsifiable predictions (to pre-register):**
  - P1, stationary: rule ≥ example at every B, with the gap largest at intermediate B. This replicates Prystawski 2023 on held-out layouts at matched bits.
  - P2, after the change: rule-record lineages have higher held-out regret than example-record lineages in generation c+1, and higher than no inheritance.
  - P3: a variant that spends some of its bits on evidence counts or confidence removes the P2 crossover.
  - Null: no crossover. Rules are at least as good after the change, because a Bayesian receiver discounts a contradicted prior quickly. That would be a publishable negative result on "culture as liability".
- **Why it survives:** works 1–6 never change the rule. Ammar et al. (2025) and de Bruin et al. (2026) show inheritance hurting under change, but without task learning, bit budgets or encoding choice.
- **Cheapest credible first experiment (hours of CPU at most):**
  - Terrain-cost world: about 6–8 terrain types, each with a cost in {1, 2, 4, lethal}, giving H ≈ 12–16 bits. Costs are noisy, so a single step does not reveal a rule, and an example record differs from a rule record.
  - A lifetime of a few episodes on new random layouts. Only a subset of types is identified per lifetime, so accumulation is possible.
  - Arms: none / examples (random and pedagogical) / rule / rule + confidence / shuffled / random, plus the unlimited-posterior and oracle references.
  - Budgets B ∈ {0, 4, 8, 12, 16, 24, 32, 64} bits.
  - 10 generations, with the rule changed at generation 6 in half of the lineages; about 200 lineages per cell.
  - Roughly 10⁵–10⁶ short lifetimes of Dijkstra on 81 cells. Minutes to an hour in NumPy on one core.
- **Feasibility gate (before any comparison):**
  - a from-scratch lifetime must leave a pre-registered gap to the oracle;
  - the unlimited-posterior chain must improve beyond generation 2 on held-out layouts;
  - shuffled records must not beat no inheritance.
  - If any gate fails, redesign the task (README gate).

### Q2: How close do simple interpretable encodings come to the rate-distortion frontier for held-out regret?

> In a sequential navigation task with held-out layouts, what fraction of the regret reduction achievable at rate R (computed by Blahut–Arimoto over the hypothesis space, with held-out regret as the distortion) do example and rule records achieve at the same R, and is the gap largest where Prystawski (2025) found the nonlinear threshold?

- **Why it survives:** Prystawski (2025) computes exact channels only for a static estimation task; Prystawski (2023) uses a heuristic code in a sequential task. Nobody has scored hand-designed codes against the exact frontier in a decision task.
- **Cheapest first experiment:** the same world as Q1, without the rule change. The cost is dominated by estimating the distortion matrix (expected regret for each true-rule × believed-rule pair, ≈ 10⁴ × 10⁴ entries). Feasible only if the hypothesis space is cut to about 10³ (e.g. 5 types × 4 levels), which takes minutes. Risk: largely a methods paper, closer to the Stanford group's agenda, so it has higher scoop risk.

### Q3: Is the ratchet real, or one-shot transfer?

> Under a binding budget, does held-out performance keep improving after generation 2, beyond a single learner with matched cumulative experience, or does a budgeted record give one-step transfer that then plateaus?

- **Why it survives:** Cook et al. show accumulation without a budget; Prystawski et al. show plateaus at low capacity on fixed tasks. No work read separates "cumulative" from "one-shot" on held-out tasks at a bit budget.
- **Cheapest first experiment:** this is a sub-analysis of Q1's stationary arm (generation-indexed curves, with the slope over generations 2–10 as the pre-registered statistic). No extra compute.

**Recommendation: Q1, with Q3 as its pre-registered secondary analysis.** Q1 is the only framing in which Last Library adds a result to the Prystawski line rather than replicating it. It fits the README's emphasis on showing negative transfer and failure. Its null outcome is informative. It needs no neural networks and runs on a laptop. Q2 is a good follow-up only if Q1's world turns out to have a small enough hypothesis space.

**Proposed decision for `docs/decisions.md` (not applied):** "D15 amended after the S3 scan (`studies/last-library/literature.md`): the nearest neighbor is Prystawski, Arumugam & Goodman (2023, 2025), not Cook et al. (2024). The Last Library question is sharpened to Q1 (rule vs. example records at matched bits under rule change, on held-out layouts). The budget is swept in bits around the latent-rule entropy. The 0 / 64 / 256-byte ablation is retired as non-binding."
