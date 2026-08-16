# PolicyLab Frontend Design

## Purpose

The PolicyLab frontend is a React application for:

* playing games against agents;
* watching agent-vs-agent matches;
* inspecting search behavior;
* monitoring training;
* browsing experiments and checkpoints;
* comparing agents and league results;
* replaying historical games.

The frontend is a client. It does not own authoritative game or training logic.

## Technology

Initial stack:

* React;
* TypeScript;
* Vite;
* Tailwind CSS;
* shadcn/ui;
* SVG for game boards and overlays.

Prefer standard React patterns and small components over custom framework abstractions.

## Application Areas

The initial navigation should grow toward:

```text
Arena
Experiments
Agents
League
Replays
```

Not all sections need to exist in the first version.

### Arena

Used to:

* create a match;
* choose participants;
* play against an agent;
* watch agent-vs-agent games;
* inspect agent analysis.

### Experiments

Used to:

* view training runs;
* inspect configuration;
* see training progress;
* browse generated checkpoints;
* compare evaluation results.

### Agents

Used to inspect:

* fixed baseline agents;
* trained checkpoints;
* model metadata;
* evaluation history.

### League

Used for:

* pairwise win rates;
* ratings;
* historical progress;
* checkpoint-vs-checkpoint comparison.

### Replays

Used to inspect completed games independently of live sessions.

## Frontend Boundaries

React components must not implement authoritative:

* legal-move logic;
* winner detection;
* turn progression;
* MCTS;
* training logic.

The backend provides authoritative snapshots and validates actions.

Frontend logic may include:

* rendering;
* local interaction state;
* navigation;
* formatting;
* optimistic visual feedback where safe;
* replay navigation over already supplied states.

## Data Flow

Prefer explicit one-way data flow:

```text
backend
   ↓
API / WebSocket
   ↓
client state
   ↓
page/component
   ↓
user action
   ↓
API / WebSocket
```

Do not create hidden global mutable state.

Server state and temporary UI state should remain conceptually separate.

## Suggested Source Structure

```text
web/src/
├── app/
├── components/
│   ├── game/
│   ├── agents/
│   ├── experiments/
│   └── ui/
├── features/
│   ├── arena/
│   ├── experiments/
│   ├── agents/
│   ├── league/
│   └── replays/
├── api/
├── hooks/
├── lib/
└── types/
```

Exact structure may evolve as real code appears.

Avoid creating directories or abstractions before they are needed.

## API Types

Backend protocol types should have a clear TypeScript representation.

Important concepts include:

* `GameSnapshot`;
* `AnalysisSnapshot`;
* `GameEvent`;
* `AgentSummary`;
* `ExperimentSummary`;
* `CheckpointSummary`.

Do not create separate frontend terminology for concepts already defined by the backend domain model.

Where practical, protocol types should be generated or validated from a shared schema rather than maintained independently forever.

## Arena State

The Arena should distinguish:

```text
authoritative game state
analysis state
temporary UI state
```

For example:

### Game state

* board contents;
* current player;
* move number;
* winner;
* legal interaction status.

### Analysis state

* policy;
* MCTS visits;
* Q values;
* root value;
* search progress.

### UI state

* selected overlay;
* sidebar visibility;
* replay position;
* hover state.

These should not be merged into one large state object.

## Hex Board

The initial Hex board should be rendered using SVG.

Reasons:

* small board size;
* precise geometry;
* direct pointer interaction;
* easy overlays;
* easy animation;
* straightforward debugging.

The board component should receive data rather than fetch it itself.

Conceptually:

```text
<HexBoard
  state={...}
  analysis={...}
  interaction={...}
  onAction={...}
/>
```

The board should not know whether the state comes from:

* a live human match;
* agent-vs-agent play;
* training inspection;
* replay.

## Board Layers

Prefer separate visual layers:

```text
board background
cells
stones
last-move marker
winning path
interaction highlights
analysis overlay
labels/debug information
```

Analysis overlays may later include:

* policy probabilities;
* MCTS visits;
* Q values;
* principal variation.

Only one primary analysis overlay needs to be visible at a time.

## Components

Prefer small components with clear responsibilities.

Examples:

```text
HexBoard
HexCell
GameToolbar
PlayerPanel
MoveHistory
AnalysisPanel
PolicyOverlay
VisitOverlay
MatchSetupDialog
```

Do not create a single large `Arena` component containing board geometry, networking, match logic, and analysis rendering.

## Networking

Use HTTP for ordinary request/response operations such as:

* loading experiments;
* loading checkpoints;
* creating matches;
* retrieving metadata.

Use WebSockets for live match and training updates where streaming is useful.

WebSocket handling should be isolated from visual components.

Components should consume typed state/events rather than manipulate socket messages directly.

## Loading and Error States

Every remote view should handle:

* loading;
* empty result;
* error;
* normal content.

Failures should be visible and actionable rather than silently ignored.

Game protocol errors, such as rejected moves or lost sessions, should be shown distinctly from ordinary UI errors.

## Training UI

Training views should prioritize meaningful learning signals.

Useful metrics include:

* evaluation win rates;
* checkpoint strength;
* league results;
* self-play throughput;
* replay-buffer state;
* policy entropy;
* value loss;
* policy loss.

Training loss should not dominate the UI as the primary measure of agent quality.

## Experiment Configuration

Experiment configuration should be inspectable as immutable run metadata.

The frontend may later support creating new configurations, but it should not silently mutate an existing experiment.

## Responsive Design

Desktop is the initial primary target.

The interface should still use responsive layout practices, but mobile-specific optimization is not an initial requirement.

The game board should preserve its aspect ratio and remain usable at different window sizes.

## Styling

Use shared design primitives from shadcn/ui and Tailwind.

Avoid:

* one-off button styles;
* duplicated spacing systems;
* inline style proliferation;
* visually unrelated component patterns.

The UI should feel like one research application rather than separate dashboards assembled independently.

## Initial Implementation Order

Build the frontend only after the backend can run a Human-vs-Random match.

Recommended sequence:

1. initialize React/Vite;
2. establish shared layout and UI primitives;
3. define backend protocol types;
4. implement Hex board;
5. create Arena page;
6. connect match creation;
7. connect human actions;
8. consume live session events;
9. show game result;
10. add replay;
11. add MCTS analysis overlays;
12. add experiments and training views.

The first frontend milestone is simply:

```text
open Arena
→ create Human vs Random Hex game
→ play complete game
→ see final result
```

Everything else builds on that path.
