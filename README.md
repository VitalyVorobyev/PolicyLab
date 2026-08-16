# PolicyLab

PolicyLab is an experimental workbench for training, evaluating, inspecting, and playing against self-learning game agents.

The project starts with **Hex** and an **AlphaZero-style** learning system based on:

* Monte Carlo Tree Search (MCTS);
* policy/value neural networks;
* self-play;
* historical checkpoints;
* agent evaluation and league play.

The longer-term goal is a game-independent environment for exploring how agents learn, comparing different strategies and training configurations, and inspecting their decisions through an interactive UI.

> **Status:** early development. The architecture and development environment are being established before the first game implementation.

## What PolicyLab aims to provide

### Train

Run reproducible self-play experiments with explicit configurations, seeds, checkpoints, and evaluation.

### Evaluate

Compare agents against:

* fixed baselines;
* search agents;
* historical checkpoints;
* each other in leagues and tournaments.

Playing strength, rather than training loss alone, is the primary measure of progress.

### Inspect

Explore what an agent is doing during search:

* policy probabilities;
* MCTS visit counts;
* Q values;
* estimated position value;
* search evolution.

### Play

Use the React UI to:

* play against trained agents;
* watch agent-vs-agent games;
* replay historical matches;
* inspect search while a game is running.

## First environment: Hex

Hex is a deterministic two-player connection game with simple rules and deep strategy.

It is a useful first environment because it provides:

* perfect information;
* deterministic transitions;
* no draws;
* a compact action space;
* straightforward game simulation;
* non-trivial search and learning behavior.

The initial implementation will progress from small boards to larger standard Hex boards.

## Planned architecture

```text
React UI
   │
   ▼
FastAPI server
   │
   ├── game sessions
   ├── experiments
   └── training
          │
          ▼
        agents
          │
        search
          │
          ▼
         games
```

The game layer remains independent from search, machine learning, networking, and UI concerns.

See [`docs/architecture.md`](docs/architecture.md) for the architectural boundaries.

## Planned agents

PolicyLab will initially include:

* random agent;
* heuristic Hex agent;
* vanilla MCTS agent;
* neural policy agent;
* AlphaZero-style MCTS agent;
* frozen checkpoint agents;
* human player support.

Historical checkpoints will remain playable and evaluable so that the evolution of an agent can be inspected over time.

## Training

The initial training loop is:

```text
self-play
   ↓
MCTS search policies
   ↓
replay buffer
   ↓
policy/value training
   ↓
checkpoint
   ↓
evaluation
   └────────→ repeat
```

See [`docs/training.md`](docs/training.md) for the intended training design.

## Technology

Backend and RL:

* Python 3.12
* PyTorch
* FastAPI
* SQLite
* pytest
* Ruff
* BasedPyright
* uv

Frontend:

* React
* TypeScript
* Vite
* Tailwind CSS
* shadcn/ui
* SVG game rendering

Project commands are managed with [`just`](https://github.com/casey/just).

## Development setup

Install:

* Python 3.12 or `uv` with Python management;
* [`uv`](https://docs.astral.sh/uv/);
* [`just`](https://github.com/casey/just).

Then:

```bash
uv sync
just verify
```

List available project commands:

```bash
just
```

At the current stage, `just verify` runs Python linting, formatting checks, strict type checking, and tests.

## Project documentation

* [`architecture.md`](docs/architecture.md) — module boundaries and dependencies
* [`domain-model.md`](docs/domain-model.md) — shared terminology
* [`hex.md`](docs/hex.md) — Hex specification
* [`training.md`](docs/training.md) — AlphaZero-style training design
* [`game-session.md`](docs/game-session.md) — match and session model
* [`frontend.md`](docs/frontend.md) — frontend design

## Development philosophy

PolicyLab favors:

* correctness before optimization;
* small, testable changes;
* deterministic game logic;
* explicit experiment configuration;
* reproducible evaluation;
* clear boundaries between game rules, search, learning, server, and UI.

Optimization, multiprocessing, native extensions, and distributed training will be introduced only when profiling or experiment scale justifies them.

## Roadmap

The first milestones are:

1. pure Hex engine with comprehensive tests;
2. random agent and match driver;
3. Human vs Random through the React UI;
4. heuristic and MCTS agents;
5. MCTS inspection in the UI;
6. policy/value network;
7. self-play training;
8. checkpoints and evaluation;
9. historical agent league;
10. experiment and training dashboard.

Additional games may follow once the Hex pipeline is mature.

## License

PolicyLab is dual-licensed under either:

- the MIT License (`LICENSE-MIT`), or
- the Apache License, Version 2.0 (`LICENSE-APACHE`),

at your option.
