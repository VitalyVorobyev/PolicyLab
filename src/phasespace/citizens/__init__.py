"""Model Citizens: the Koster et al. (2025) common-pool game and its published dataset.

Pure domain code: `game` (pool dynamics and move validity), `mechanisms` (baseline,
Interpolating and Random allocation rules), `metrics` (the published notebook's game
metrics) and `records` (per-game arrays). I/O: `source` (verified download), `data`
(parsing, storage and the held-out guard) and `export` (the C1 data-audit run).
`validation` holds the C1 identity and reproduction checks.
"""
