# Algorithm Forge — Planned Research Specification (v0)

**Theme:** Search in program space, genetic programming, superoptimization, formal correctness.  
**Priority:** third new world, after Gravity Atlas and The Last Library. No large language model required.

## Research question

> Under a fixed evaluation budget, when does an evolutionary search over very short bit-vector programs beat simple structured search at finding correct and compact implementations?

Program superoptimization is an established field; no claims that the idea itself is new. The target is a careful, interpretable comparison in a constrained toy ISA.

## Bounded virtual machine

- Pure deterministic **unsigned 8-bit** semantics (wraparound modulo 256), one 8-bit input register, up to 3–4 working registers, limited immediates.
- Operations selected from `MOV`, `AND`, `OR`, `XOR`, `ADD`, `SUB`, `SHL`, `SHR` (select a minimal fixed subset at study freeze; masks/shift semantics defined exactly).
- Straight-line programs of maximum **12 instructions** for initial tasks; no loops, jumps, memory access, I/O, undefined behavior or arbitrary Python execution.
- Fixed starter corpus of tractable unary bit functions, e.g. parity, population count, bit reversal, low-bit isolation and rotate by a constant. Check feasibility against instruction set and limits first; remove tasks which provably cannot be represented in the chosen ISA.

## Methods

- Uniform random program sampling: sanity baseline.
- Beam search with partial scoring and bounded beam width.
- Genetic / mutation search with elitism, length penalties and multiple independent seeds.
- Bounded exhaustive search **only for tractable subspaces**, explicitly documenting where full enumeration is impractical.
- Optional later: Z3 bit-vector constraints for synthesis/pruning or independent proof on larger input spaces.

## Evaluation & correctness

For unary 8-bit functions, test **all 256 inputs**, not a small train/test split. This is exact finite-domain equivalence (given correct VM semantics), not approximate test accuracy. Report program length and a consistent execution-cost measure only for *fully correct* programs. Invalid or partially correct candidates must never be ranked as successful implementations.

Measure first-correct discovery probability by budget, best correct length and cost, search steps, evaluation count and wall time. Vary instruction set composition and maximum length as predeclared ablations. Keep search budget parity and ensure executable VM semantics are identical across algorithms.

## Visualization

An interactive instruction-by-instruction execution trace for a selected program; visual input/output truth-table checker; evolutionary lineage and best-cost history; side-by-side comparison of discovered equivalent programs. Never label an unproven random candidate as correct.

## Gates

**Feasibility:** reference interpreter passes exhaustive truth tables for hand-written known programs; an exact correctness checker exists; small benchmark tasks have valid implementations. **Research completion:** beam versus evolutionary vs appropriate exhaustive baseline across multiple seeds and tasks; report with null outcomes and reproducible programs. **Stop:** if novelty or benefit is negligible, report and do not artificially increase program complexity.

## Background

- AlphaDev: https://www.nature.com/articles/s41586-023-06004-9
- STOKE: https://stoke.stanford.edu/
- Z3: https://github.com/Z3Prover/z3
