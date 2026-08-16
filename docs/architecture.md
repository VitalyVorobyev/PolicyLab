# PolicyLab Architecture

## Purpose

PolicyLab is a research workbench for training, evaluating, inspecting, and playing against game-playing agents.

The first game is Hex, but the architecture should remain game-independent wherever practical.

The initial learning approach is AlphaZero-style self-play using MCTS and a policy/value network.

## Design principles

### Pure game logic

Game implementations contain only state and rules.

They must not depend on:

* PyTorch or neural networks;
* MCTS or training;
* FastAPI or databases;
* UI or networking.

Given the same state and action, the result must always be deterministic.

### Agents choose actions

Agents receive a game state and select an action.

They do not control match progression, persistence, networking, or UI sessions.

### Search is independent from training

MCTS belongs in a reusable search layer.

It may use different evaluators such as heuristics or policy/value networks, but the search implementation must not depend on the training loop.

### Training depends downward

Training may depend on games, search, and agents.

Game and search layers must never depend on training.

### Frontend is a client

React renders state and sends actions.

The backend owns authoritative game state and validates all moves.

### Reproducibility is required

Experiments must record enough information to reproduce and compare results, including configuration, seed, Git commit, model definition, checkpoints, and evaluation results.

Experiment artifacts must not be silently overwritten.

## Repository layout

```text
policylab/
├── .pi/
│   ├── prompts/
│   └── skills/
├── configs/
├── docs/
├── src/
│   └── policylab/
│       ├── games/
│       ├── agents/
│       ├── search/
│       ├── training/
│       ├── experiments/
│       └── server/
├── tests/
├── web/
├── runs/
├── AGENTS.md
├── justfile
└── pyproject.toml
```

## Modules

### `games`

Defines generic game abstractions and concrete game implementations.

Responsibilities:

* state representation;
* actions;
* legal actions;
* state transitions;
* current player;
* terminal detection;
* result;
* optional symmetries and encodings.

The generic API must not contain Hex-specific concepts.

### `games.hex`

Implements Hex rules and representation.

This is the first concrete game and must be thoroughly tested before search or RL code depends on it.

### `agents`

Contains action-selection strategies.

Expected agents include:

* random;
* heuristic;
* MCTS;
* policy-only;
* AlphaZero;
* checkpoint-backed;
* human adapter.

Agents may use `search`, but must not own the match loop.

### `search`

Contains reusable search algorithms.

Initial focus:

* MCTS;
* PUCT;
* evaluator interfaces;
* search statistics.

Search should expose analysis data such as visit counts, priors, Q values, and root value where practical.

### `training`

Contains learning-specific functionality:

* policy/value network;
* replay buffer;
* self-play;
* losses and optimization;
* checkpoints;
* evaluation scheduling.

### `experiments`

Contains experiment configuration, metadata, metrics, checkpoint records, and reproducibility infrastructure.

Experiment behavior should be driven by explicit configuration rather than hidden constants.

### `server`

Exposes backend functionality to the frontend.

Responsibilities include:

* HTTP API;
* WebSockets;
* game sessions;
* human-vs-agent games;
* agent-vs-agent games;
* replay access;
* experiment and training telemetry.

The server should orchestrate domain components rather than contain game or training logic itself.

### `web`

Contains the React application.

The frontend must not duplicate authoritative game rules.

The Hex board will initially use SVG and should support live games, replays, and agent-analysis overlays.

## Dependency direction

Dependencies should point downward:

```text
web
 ↓
server
 ↓
experiments / training / agents
              ↓
            search
              ↓
            games
```

More precisely:

```text
games        -> no higher-level modules
search       -> games
agents       -> games, optionally search
training     -> games, search, agents
experiments  -> training and experiment metadata
server       -> required backend modules
web          -> server protocol only
```

Circular dependencies are architectural defects.

## Cross-cutting rules

When implementing functionality:

1. Put behavior in the lowest appropriate layer.
2. Do not introduce upward dependencies to solve local problems.
3. Keep game and training logic out of FastAPI handlers and React components.
4. Prefer explicit data flow over hidden global state.
5. Preserve deterministic behavior wherever practical.
6. Add abstractions only when they solve a concrete current problem.
7. Prefer correctness, observability, and tests before optimization.
