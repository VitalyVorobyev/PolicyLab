---
name: lit-scan
description: Focused nearest-neighbor literature scan for one Phase Space study question. Use before freezing a study protocol, when reframing a research question, when asked "has this been done", "what is the closest work", "is this novel", or before making any novelty claim in a report. Produces docs/studies/<study>/literature.md with verified citations only. Not an open-ended literature review.
---

# Literature scan

Goal: find the work closest to **one** study question, state how this study differs, and rate novelty risk honestly. Follow `AGENTS.md` research-integrity rules.

## 1. Pin the question

Read `docs/studies/<study>/README.md`. Write down the question in one sentence and split it into its components: domain (e.g. CR3BP trajectories), method (e.g. MAP-Elites), claim type (e.g. robustness under perturbation), baseline.

## 2. Search systematically

- Run at least 6 queries covering each component pair (domain × method, method × claim, domain × claim) plus synonyms (quality-diversity / illumination / novelty search; free-return / cislunar / three-body).
- Search both general web and scholarly sources (arXiv, Google Scholar, Semantic Scholar, NASA ADS for astrodynamics, ACM DL for GECCO).
- Follow citations one hop: who cites the closest paper, and what does it cite?
- Log every query with date and the useful hits.

## 3. Verify every citation

A reference enters the document only if its DOI or arXiv ID was fetched and the title, authors and year match. Read at least the abstract; read method/results sections of the 3 closest works. Never cite from memory or from a search-summary alone. If a source cannot be verified, list it under "Unverified leads".

## 4. Write `docs/studies/<study>/literature.md`

```markdown
# <Study> — Literature scan (<date>)

## Question as scanned
## Closest works            (3–8; per work: citation, link, what they did, what they did NOT do)
## How this study differs   (concrete; one bullet per difference)
## Novelty risk             (low / medium / high, with reasoning)
## Baselines implied by prior work
## Unverified leads
## Search log               (query, date, source, notable hits)
```

## 5. Report back

Summarize: closest work, the remaining gap (or that there is none), recommended reframing, and suggested target venue. If the gap is gone, say so plainly and propose alternatives; do not stretch the claim. Propose any resulting decision for `docs/decisions.md` rather than silently editing the study question.
