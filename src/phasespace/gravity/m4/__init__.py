"""Gravity Atlas M4: budget-matched comparison machinery (protocol v1, DRAFT).

Implements docs/studies/gravity-atlas/protocol.md: the common evaluator and budget
counter (`evaluation`), catalogs and elite re-checks (`catalog`), the compared methods
(`methods`), the robustness evaluators (`robustness`), the coverage ground truth
(`groundtruth`), per-run metrics (`metrics`), the statistics (`stats`) and the runs and
command line (`runs`, `cli`). Every comparative command refuses to run until the
protocol's status line says `frozen` and `--confirm-frozen` is given (`config`).
"""
