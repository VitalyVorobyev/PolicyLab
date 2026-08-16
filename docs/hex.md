# Hex Game Specification

## Scope

Hex is the first game implemented in PolicyLab.

The initial implementation supports standard two-player Hex on an `N × N` board.

The pie/swap rule is **not** supported initially.

## Coordinates

Board coordinates are zero-based:

```text
(row, column)
```

with:

```text
0 <= row < N
0 <= column < N
```

Rows increase downward and columns increase to the right.

Example for a 3×3 board:

```text
(0,0) (0,1) (0,2)
   (1,0) (1,1) (1,2)
      (2,0) (2,1) (2,2)
```

The visual indentation is only a rendering property. Logically the board is an `N × N` array.

## Players

There are two players:

* Player 0 connects **left to right**.
* Player 1 connects **top to bottom**.

Players alternate turns.

Player 0 moves first.

## Board Cells

Each cell is exactly one of:

* empty;
* occupied by Player 0;
* occupied by Player 1.

A move places the current player's stone into one empty cell.

Stones are never moved or removed.

## Hex Neighbors

Each cell has up to six adjacent cells.

For `(row, column)`, the possible neighbor offsets are:

```text
(-1,  0)
(-1, +1)
( 0, -1)
( 0, +1)
(+1, -1)
(+1,  0)
```

Neighbors outside the board are ignored.

This convention must be used consistently by game logic and UI geometry.

## Legal Actions

An action is a board coordinate.

An action is legal if:

1. the game is not terminal;
2. the coordinate is inside the board;
3. the cell is empty.

Applying a legal action:

1. places the current player's stone;
2. advances the turn to the other player;
3. produces a new immutable state.

Applying an illegal action must fail explicitly. It must never silently modify or ignore the action.

## Winning Condition

Player 0 wins when there exists a connected path of Player 0 stones from the left edge:

```text
column = 0
```

to the right edge:

```text
column = N - 1
```

Player 1 wins when there exists a connected path of Player 1 stones from the top edge:

```text
row = 0
```

to the bottom edge:

```text
row = N - 1
```

Connectivity uses the six Hex neighbors defined above.

The game becomes terminal immediately when a winning connection exists.

No actions are legal after the game becomes terminal.

## Draws

Standard Hex has no draws.

The PolicyLab Hex implementation therefore has only two terminal outcomes:

```text
Player 0 wins
Player 1 wins
```

A draw result must not be introduced into Hex-specific logic.

Generic PolicyLab game abstractions may still support games that can draw in the future.

## State

A Hex state must contain enough information to determine:

* board size;
* board contents;
* current player;
* legal actions;
* terminal status;
* winner.

The state is logically immutable.

Implementation details such as tuples, arrays, or compact bit representations are not specified here and may evolve later.

Derived values such as connectivity structures may be cached only if doing so preserves the observable immutability and deterministic behavior of the state.

## Initial State

For board size `N`:

* all cells are empty;
* Player 0 is the current player;
* the game is non-terminal;
* there is no winner.

Board size must be positive.

## Action Index

For algorithms that require a flat action space, coordinates use the stable mapping:

```text
index = row * N + column
```

and:

```text
row = index // N
column = index % N
```

Therefore an `N × N` Hex game has exactly:

```text
N²
```

possible action indices.

Illegal or occupied cells are masked by higher-level policies/search rather than changing the action-space size.

## Symmetry

Hex has a useful player-swapping symmetry.

Transposing the board:

```text
(row, column) -> (column, row)
```

and swapping Player 0 with Player 1 converts:

* left-right connectivity into top-bottom connectivity;
* top-bottom connectivity into left-right connectivity.

This symmetry will later be useful for canonical neural-network representations and data augmentation.

The pure game implementation should expose enough information to implement such transformations, but neural-network canonicalization does not belong in the Hex rules layer.

## Winning Path

The engine may expose one concrete winning path for visualization or debugging.

A winning path:

* is optional metadata;
* must not affect game correctness;
* consists only of stones belonging to the winner;
* connects the winner's two target edges.

The boolean winner result remains authoritative.

## Initial Testing Requirements

Before any MCTS or learning implementation depends on Hex, tests must cover at least:

* initial state;
* legal moves;
* illegal occupied-cell moves;
* out-of-range moves;
* turn alternation;
* all six neighbor directions;
* Player 0 left-to-right wins;
* Player 1 top-to-bottom wins;
* non-winning disconnected stones;
* terminal-state behavior;
* action-index round trips;
* representative edge and corner cases.

Small boards such as 1×1, 2×2, and 3×3 should be used heavily because expected outcomes can be inspected directly.
