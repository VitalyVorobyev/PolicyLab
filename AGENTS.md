# Selfplay

Selfplay is an RL research workbench for training and evaluating
game-playing agents through self-play.

Initial game: Hex.

## Architecture

- `engine/`: pure game mechanics; no ML or networking
- `trainer/`: self-play, MCTS, policy/value training
- `server/`: FastAPI API and live game sessions
- `web/`: React/TypeScript UI

Dependencies point inward:

web -> server -> trainer -> engine

The engine must not depend on any other component.

## Development rules

- Make small, focused changes.
- Do not redesign unrelated code.
- Read relevant existing code before editing.
- Never silently change experiment hyperparameters.
- Every game-rule change requires tests.
- Every bug fix should have a regression test when practical.
- Run the relevant checks before declaring a task complete.

## Commands

just test
just check
just train-smoke
just web-test
just dev

## RL reproducibility

Every experiment records:
- complete configuration
- random seed
- git commit
- model architecture
- training steps
- evaluation results

Do not overwrite experiment artifacts.
