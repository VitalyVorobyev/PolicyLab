# PolicyLab Domain Model

## Game

A `Game` defines the rules of a deterministic, turn-based game.

It provides operations such as:

* create an initial state;
* determine legal actions;
* apply an action;
* determine the current player;
* detect terminal states;
* return the final outcome.

A game contains no agent, training, UI, or persistence logic.

## State

A `State` is the complete immutable description of a game position.

It must contain everything required to determine:

* legal actions;
* current player;
* whether the game is terminal;
* the result if terminal.

Applying an action produces a new state rather than modifying the existing state.

## Action

An `Action` represents one legal decision a player can make.

Actions are game-specific.

For Hex, an action is initially a board coordinate where a stone is placed.

## Player

A `Player` identifies one side in a game.

For two-player zero-sum games, PolicyLab uses two logical players.

Game results are always expressed relative to an explicitly specified player where ambiguity is possible.

## Agent

An `Agent` is an automated strategy that selects an action from a game state.

Examples:

* random agent;
* heuristic agent;
* MCTS agent;
* neural policy agent;
* AlphaZero agent;
* frozen checkpoint agent.

An agent does not control the match loop.

## Human Player

A human player is treated as another source of actions.

The match system should not require special game logic for human participation.

The difference is only that the action arrives asynchronously from the UI.

## Match

A `Match` is one complete game between two players or agents.

It contains:

* game configuration;
* participants;
* initial state;
* ordered actions;
* final outcome;
* optional analysis metadata.

The match driver owns turn progression and validates actions.

## Game Session

A `GameSession` is the runtime representation of an active or replayed match exposed to external clients.

It may manage:

* connected UI clients;
* asynchronous human input;
* agent execution;
* emitted game events;
* live search analysis.

Game sessions do not define game rules.

## Search

A `Search` explores possible future game states to improve action selection.

The initial implementation is Monte Carlo Tree Search.

Search may use an evaluator but is independent from the training loop.

## Evaluator

An `Evaluator` estimates a game state for search.

Depending on the implementation, it may provide:

* a scalar value estimate;
* action priors;
* both.

Examples:

* heuristic evaluator;
* policy/value neural network.

## Policy

A `Policy` is a probability distribution or preference over legal actions.

During AlphaZero training we distinguish:

* network policy: output directly by the neural network;
* search policy: distribution derived from MCTS visit counts.

## Value

A `Value` estimates the expected game outcome from a defined player's perspective.

The perspective must always be explicit or fixed by convention.

For AlphaZero-style training, values are normalized to:

```text
-1  loss
+1  win
```

Hex has no draws.

## Self-Play Game

A `SelfPlayGame` is a match generated for training.

It records training examples derived from positions encountered during the game.

Typical examples are:

```text
(state, search_policy, final_outcome)
```

## Training Sample

A `TrainingSample` contains the information needed to train the policy/value model for one position.

Initially:

* encoded state;
* target search policy;
* target game outcome.

## Replay Buffer

A `ReplayBuffer` stores recent training samples used by the learner.

It is training infrastructure and is distinct from a user-visible game replay.

## Checkpoint

A `Checkpoint` is an immutable saved version of a trained model plus identifying metadata.

A checkpoint can be loaded as an agent for:

* evaluation;
* league matches;
* human play;
* historical comparison.

Existing checkpoints must never be silently modified.

## Experiment

An `Experiment` is one reproducible training run.

It owns or references:

* configuration;
* seed;
* source Git commit;
* progress;
* metrics;
* checkpoints;
* evaluations.

An experiment has a stable unique identifier.

## Evaluation

An `Evaluation` measures an agent or checkpoint against defined opponents under fixed conditions.

It records at least:

* participants;
* game configuration;
* number of games;
* side assignment;
* results;
* relevant search budgets.

Training loss alone is not considered an evaluation of playing strength.

## League

A `League` is a collection of agents or checkpoints evaluated against each other.

It can contain:

* historical checkpoints;
* fixed baseline agents;
* current agents.

League results may be represented through win rates, ratings, and pairwise match matrices.

## Replay

A `Replay` is a deterministic reconstruction of a completed match from its recorded actions and game configuration.

A replay is not the same thing as a replay buffer.

## Terminology Rules

Use these terms consistently:

* `Game` — rules;
* `State` — one position;
* `Action` — one move;
* `Agent` — automated action selector;
* `Match` — one complete game;
* `GameSession` — runtime/UI-facing match execution;
* `Search` — planning over future states;
* `Checkpoint` — immutable trained model snapshot;
* `Experiment` — one training run;
* `Evaluation` — measured playing-strength comparison;
* `League` — collection of competing agents/checkpoints;
* `Replay` — recorded match reconstruction;
* `ReplayBuffer` — training sample storage.

Do not introduce synonyms for these concepts without a concrete reason.
