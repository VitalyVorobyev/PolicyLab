from dataclasses import dataclass

Action = tuple[int, int]

EMPTY = 0
PLAYER_0 = 1
PLAYER_1 = 2

Stone = int


@dataclass(frozen=True)
class HexState:
    size: int
    board: tuple[tuple[int, ...], ...]
    current_player: int

    def neighbors(self, action: Action) -> list[Action]:
        """
        Get valid neighbor coordinates for an action.
        Uses the standard Hex neighbor offsets:
        (-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0)
        """
        return neighbors(action[0], action[1], self.size)

    def board_copy(self) -> list[list[int]]:
        """Return a mutable copy of the board."""
        return [[c for c in row] for row in self.board]


def create_initial_state(size: int) -> HexState:
    """
    Create an initial Hex game state with an empty board.

    Args:
        size: Board dimension (must be positive).

    Returns:
        A HexState with all cells empty and current_player=0.
    """
    if size <= 0:
        raise ValueError("Board size must be positive")
    board: list[list[int]] = [[EMPTY for _ in range(size)] for _ in range(size)]
    return HexState(
        size=size,
        board=tuple(tuple(int(x) for x in row) for row in board),
        current_player=0,
    )


def is_legal_action(state: HexState, action: Action) -> bool:
    """
    Check if an action is legal in the given state.

    An action is legal if:
    - The game is not terminal (no winner yet).
    - The coordinate is inside the board.
    - The cell is empty.

    Args:
        state: Current game state.
        action: Proposed move (row, col).

    Returns:
        True if the action is legal, False otherwise.
    """
    # Check if game is terminal
    winner = get_winner(state)
    if winner is not None:
        return False

    if not (0 <= action[0] < state.size and 0 <= action[1] < state.size):
        return False
    return state.board[action[0]][action[1]] == EMPTY


def legal_actions(state: HexState) -> list[Action]:
    """
    Get all legal actions in the current state.

    Args:
        state: Current game state.

    Returns:
        List of all legal action coordinates, or empty list if terminal.
    """
    winner = get_winner(state)
    if winner is not None:
        return []

    actions: list[Action] = []
    for row in range(state.size):
        for col in range(state.size):
            if state.board[row][col] == EMPTY:
                actions.append((row, col))
    return actions


def apply_action(state: HexState, action: Action) -> HexState:
    """
    Apply a legal action to produce a new game state.

    Args:
        state: Current game state.
        action: Legal move coordinate.

    Returns:
        New state with stone placed and turn advanced.

    Raises:
        ValueError: If action is not legal.
    """
    player = state.current_player
    if not is_legal_action(state, action):
        raise ValueError(f"Action {action} is not legal")

    board: list[list[int]] = state.board_copy()
    board[action[0]][action[1]] = player

    if get_winner(board):
        return HexState(
            size=state.size,
            board=tuple(tuple(x for x in row) for row in board),
            current_player=player,
        )
    return HexState(
        size=state.size,
        board=tuple(tuple(x for x in row) for row in board),
        current_player=1 - state.current_player,
    )


def _get_neighbors(action: Action, size: int) -> list[Action]:
    """
    Get all valid neighbors for an action on a board of size.

    Args:
        action: Board coordinate (row, col).
        size: Board dimension.

    Returns:
        List of valid neighbor coordinates.
    """
    row, col = action
    offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    result: list[Action] = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            result.append((nr, nc))
    return result


def neighbors(action: Action, size: int) -> list[Action]:
    """
    Get all valid neighbor coordinates for an action on a board of size.

    Args:
        action: Board coordinate (row, col).
        size: Board dimension.

    Returns:
        List of valid neighbor coordinates.
    """
    return _get_neighbors(action, size)


def has_connection(state: HexState, player: int) -> bool:
    """
    Check if the given player has a connected winning path.

    Args:
        state: Current game state.
        player: Player to check (0 or 1).

    Returns:
        True if the player has a winning path, False otherwise.
    """
    board: list[list[int]] = [[int(x) for x in row] for row in state.board]
    return _has_connection_internal(board, player, state.size)


def _has_connection_internal(board: list[list[int]], player: int, size: int) -> bool:
    """
    Internal helper for has_connection. Checks a mutable board.

    Args:
        board: The mutable game board.
        player: Player to check (0 or 1).
        size: Board dimension.

    Returns:
        True if the player has a winning path, False otherwise.
    """
    if player == 0:
        for row in range(size):
            if board[row][0] == PLAYER_0 and _bfs_to_edge(
                board, (row, 0), size - 1, size - 1, PLAYER_0
            ):
                return True
    else:
        for col in range(size):
            if board[0][col] == PLAYER_1 and _bfs_to_edge(
                board, (0, col), size - 1, 0, PLAYER_1
            ):
                return True

    return False


def _bfs_to_edge(
    board: list[list[int]],
    start: tuple[int, int],
    target_row: int,
    target_col: int,
    player: int,
) -> bool:
    """
    BFS from start to check if we can reach the opposite edge.

    Args:
        board: The game board.
        start: Starting coordinate.
        target_row: Target row if going top-down, else -1
        target_col: Target col if going left-right, else -1
        player: The player to check.

    Returns:
        True if a winning path exists.
    """
    row, col = start

    if target_col != -1 and col == target_col:
        return True
    if target_row != -1 and row == target_row:
        return True

    visited = set([(row, col)])
    stack = [(row, col)]

    while stack:
        curr_r, curr_c = stack.pop()
        for nr, nc in _get_neighbors((curr_r, curr_c), len(board)):
            if (nr, nc) not in visited:
                if target_col != -1 and nc == target_col:
                    return True
                if target_row != -1 and nr == target_row:
                    return True
                if board[nr][nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

    return False


def get_winner(state: HexState | list[list[int]]) -> Stone:
    """
    Get the winner of the current state.

    Args:
        state: HexState or mutable board.

    Returns:
        The winning player (0 or 1), or None if no winner (non-terminal).
    """
    board = state if isinstance(state, list) else state.board

    for row in range(len(board)):
        if board[row][0] == PLAYER_0 and _bfs_to_edge(
            board, (row, 0), len(board) - 1, -1, PLAYER_0
        ):
            return PLAYER_0

    for col in range(len(board[0]) if board else 0):
        if board[0][col] == PLAYER_1 and _bfs_to_edge(
            board, (0, col), -1, len(board) - 1, PLAYER_1
        ):
            return PLAYER_1

    return None


def coord_to_action_index(row: int, col: int, size: int) -> int:
    """
    Convert board coordinates to a flat action index.

    Args:
        row: Row coordinate.
        col: Column coordinate.
        size: Board dimension.

    Returns:
        Flat index = row * size + col.

    Raises:
        ValueError: If coordinates are out of bounds.
    """
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"Invalid coordinates: ({row}, {col}) for size {size}")
    return row * size + col


def action_index_to_coord(index: int, size: int) -> tuple[int, int]:
    """
    Convert a flat action index to board coordinates.

    Args:
        index: Flat action index.
        size: Board dimension.

    Returns:
        Tuple (row, col).

    Raises:
        ValueError: If index is out of bounds.
    """
    if not (0 <= index < size * size):
        raise ValueError(f"Invalid index {index} for size {size}")
    return index // size, index % size
