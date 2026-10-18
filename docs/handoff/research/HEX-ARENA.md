# Hex Arena — Existing World (Preserve)

**Status:** existing PolicyLab core; not a new development milestone in the current handoff.

## What exists as repository design

`docs/hex.md`, `docs/training.md`, `docs/game-session.md`, `docs/frontend.md` and `docs/architecture.md` specify deterministic Hex rules, matches, MCTS, AlphaZero-style self-play, event-driven sessions, evaluation and replay. The inspected documentation does **not** prove that every subsystem is implemented; audit current source and tests.

## Role in the broader laboratory

- Preserve Hex as a game/research world and an integration/regression test for the shared site, if it is functional.
- Optionally expose an existing Hex playback or a link from the gallery; don't invest in Hex-specific new RL capabilities before Gravity M1–M3.
- Do not fit continuous dynamics into `GameState`, Hex turn rules or MCTS interfaces.

## Optional bounded future research question

> How reliable are apparent self-play improvements when evaluated against frozen historical opponents, fixed-budget MCTS and unseen search budgets?

Methods and outcome metrics are already mostly described in `docs/training.md`. Only promote this to an active study if there is a known working training loop and a concrete comparative protocol.

## Compatibility constraints

- Preserve public module semantics and existing matches/tests.
- `Experiment` currently means one training run in `docs/domain-model.md`; don't silently repurpose it.
- Follow repository `AGENTS.md` and `just verify`.
