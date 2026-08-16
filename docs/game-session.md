# PolicyLab Game Session Design

## Purpose

A game session connects game logic, players, agents, and external clients.

It is responsible for running matches, accepting human actions, exposing live state, and emitting events.

Game sessions do not define game rules.

## Match Driver

The match driver owns the lifecycle of one match.

It is responsible for:

* creating or receiving the initial state;
* determining whose turn it is;
* requesting an action from the active player;
* validating the action;
* applying the action;
* recording move history;
* detecting match completion;
* producing the final result.

Conceptually:

```text
state
  ↓
current player
  ↓
request action
  ↓
validate
  ↓
apply
  ↓
emit event
  ↓
terminal?
```

The match driver must work the same way for:

* human vs agent;
* agent vs agent;
* checkpoint vs checkpoint;
* evaluation matches.

## Player Interface

A player is a source of actions.

Conceptually:

```text
choose_action(state, context) -> action
```

The concrete interface may be asynchronous.

Automated players may return immediately or after computation.

Human players wait until an action arrives from the UI.

The match driver must not contain separate Hex-specific logic for human and automated players.

## Action Validation

The backend is authoritative.

Every submitted action must be validated against the current game state.

Invalid actions must be rejected explicitly.

Examples:

* action outside the board;
* action on an occupied cell;
* action submitted by the wrong player;
* action submitted after the game has ended;
* stale action for an earlier turn.

The frontend may disable illegal interactions for convenience, but frontend validation is never authoritative.

## Game Session

A `GameSession` wraps a running or replayed match for external consumers.

A live session may contain:

* match identifier;
* game configuration;
* participants;
* current state;
* move history;
* session status;
* pending human input;
* optional agent analysis;
* connected clients.

The session should expose immutable snapshots rather than allowing clients to mutate internal state.

## Session Status

A session should have an explicit lifecycle such as:

```text
created
running
finished
failed
```

Additional states should only be introduced when needed.

## Game Snapshot

A `GameSnapshot` represents the current authoritative game state exposed to clients.

It should contain enough information to render the game, including:

* match/session identifier;
* game type;
* game configuration;
* serialized state;
* current player;
* legal interaction state;
* move number;
* terminal status;
* winner when terminal.

The snapshot should not contain MCTS implementation details.

## Analysis Snapshot

Agent analysis is separate from game state.

An `AnalysisSnapshot` may contain:

* active agent;
* search progress;
* root value;
* policy priors;
* visit counts;
* Q values;
* selected action;
* principal variation.

Analysis may be absent.

The same game UI must work when no analysis information is available.

## Events

Sessions communicate changes through typed events.

The initial event vocabulary should remain small.

Expected events include:

```text
game_started
state_changed
your_turn
agent_thinking
search_updated
move_played
game_finished
error
```

Event payloads should be explicit and versionable.

Do not encode protocol meaning in human-readable strings.

## Human Input

When a human player is active:

1. the session enters a state waiting for human input;
2. the client receives the current game snapshot;
3. the client submits an action;
4. the backend validates it;
5. the match continues.

The UI must not submit actions for an outdated move number.

Including a turn or state revision in action requests is preferred so stale actions can be detected.

## Agent Turns

For an automated agent:

1. the match driver requests an action;
2. the session may emit `agent_thinking`;
3. the agent performs search or inference;
4. optional analysis updates may be emitted;
5. the chosen action is returned;
6. the match driver applies it.

Search telemetry must not affect game correctness.

Failure to produce an action should fail the session explicitly rather than silently selecting another move.

## Live Analysis

Longer searches may emit intermediate analysis updates.

These updates are observational only.

Clients may use them to render:

* policy overlays;
* MCTS visit counts;
* win estimates;
* search progress.

The final authoritative move still comes from the match driver.

## Replay

A replay reconstructs a completed match from:

* game configuration;
* ordered actions;
* optional recorded metadata.

Replays should not store frontend state.

Given the same game configuration and actions, replay reconstruction must produce the same sequence of game states.

## Replay Driver

The frontend should consume replayed games through the same state representation used for live matches.

Replay controls may include:

* first position;
* previous move;
* next move;
* last position;
* jump to move;
* autoplay.

Replay navigation is not part of the game engine.

## Persistence

Completed matches should eventually persist:

* participants;
* game configuration;
* ordered actions;
* outcome;
* timestamps;
* optional analysis metadata.

Persistence is not required for the first playable prototype.

The in-memory session implementation should therefore avoid assumptions that prevent later persistence.

## Concurrency

The first implementation should favor simple execution.

Each live match may be represented by one session object with serialized state transitions.

Do not introduce distributed session management or complex concurrency before a single-process implementation works correctly.

Concurrent requests must never cause two actions to be applied to the same turn.

## Initial Backend Protocol

The first frontend/backend integration only needs enough protocol to support:

1. create a Hex match;
2. retrieve the current snapshot;
3. receive live events;
4. submit a human action;
5. finish the match.

Agent analysis can be added after basic gameplay is reliable.

## Initial Implementation Order

Build this layer only after the pure Hex engine exists.

Recommended sequence:

1. player interface;
2. random agent;
3. match driver;
4. match history;
5. in-memory game session;
6. human action adapter;
7. minimal FastAPI endpoints;
8. WebSocket events;
9. React integration;
10. replay support;
11. search-analysis events.

Keep the first playable path simple:

```text
Human
  ↓
React
  ↓
GameSession
  ↓
MatchDriver
  ↓
HexGame
  ↑
RandomAgent
```

Only after this works should MCTS or neural agents be connected to the UI.
