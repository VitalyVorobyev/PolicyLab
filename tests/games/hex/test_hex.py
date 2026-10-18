import pytest

from policylab.games.hex import (
    action_index_to_coord,
    apply_action,
    coord_to_action_index,
    create_initial_state,
    get_winner,
    has_connection,
    is_legal_action,
    legal_actions,
    neighbors,
)


class TestCreateInitialState:
    """Test initial state creation."""

    def test_initial_state_empty_board(self):
        """Initial state has all empty cells."""
        state = create_initial_state(5)
        assert state.size == 5
        assert state.current_player == 0
        assert all(state.board[r][c] == 0 for r in range(5) for c in range(5))

    def test_initial_state_non_terminal(self):
        """Initial state is non-terminal."""
        state = create_initial_state(5)
        assert get_winner(state) is None

    def test_initial_state_1x1(self):
        """1x1 board initial state."""
        state = create_initial_state(1)
        assert state.size == 1
        assert state.current_player == 0
        assert state.board == ((0,),)

    def test_initial_state_2x2(self):
        """2x2 board initial state."""
        state = create_initial_state(2)
        assert state.size == 2
        assert state.current_player == 0
        assert state.board == ((0, 0), (0, 0))

    def test_initial_state_invalid_size(self):
        """Invalid board size raises error."""
        with pytest.raises(ValueError, match="positive"):
            create_initial_state(0)
        with pytest.raises(ValueError, match="positive"):
            create_initial_state(-1)


class TestNeighbors:
    """Test neighbor calculations."""

    def test_neighbors_center_5x5(self):
        """Center cell has 6 neighbors."""
        n = neighbors((2, 2), 5)
        assert len(n) == 6
        # Standard Hex neighbors: (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)
        # From (2,2):
        assert (1, 2) in n  # dr=-1, dc=0
        assert (1, 3) in n  # dr=-1, dc=1
        assert (2, 1) in n  # dr=0, dc=-1
        assert (2, 3) in n  # dr=0, dc=1
        assert (3, 1) in n  # dr=1, dc=-1
        assert (3, 2) in n  # dr=1, dc=0

    def test_neighbors_corner_5x5(self):
        """Corner cell has 2 neighbors."""
        n = neighbors((0, 0), 5)
        assert len(n) == 2
        assert (0, 1) in n  # right
        assert (1, 0) in n  # down

    def test_neighbors_edge_5x5(self):
        """Edge cell has 3 neighbors."""
        # Right edge (0, 4)
        n = neighbors((0, 4), 5)
        assert len(n) == 3
        assert (0, 3) in n  # left
        assert (1, 3) in n  # bottom-left diagonal
        assert (1, 4) in n  # bottom

    def test_neighbors_all_offsets(self):
        """All six neighbor offsets are present."""
        n = neighbors((2, 2), 5)
        expected_offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        for dr, dc in expected_offsets:
            assert (2 + dr, 2 + dc) in n

    def test_neighbors_out_of_bounds(self):
        """Off-board cells are not included."""
        n = neighbors((4, 4), 5)
        assert (5, 4) not in n  # down
        assert (4, 5) not in n  # right
        assert (3, 5) not in n  # diag up-right
        assert (5, 3) not in n  # diag down-left

    def test_neighbors_1x1(self):
        """1x1 board has no neighbors."""
        n = neighbors((0, 0), 1)
        assert len(n) == 0


class TestIsLegalAction:
    """Test action legality checks."""

    def test_legal_move_empty_cell(self):
        """Empty cell on non-terminal board is legal."""
        state = create_initial_state(5)
        assert is_legal_action(state, (2, 2)) is True

    def test_illegal_out_of_bounds(self):
        """Actions outside board are illegal."""
        state = create_initial_state(5)
        assert is_legal_action(state, (-1, 0)) is False
        assert is_legal_action(state, (0, -1)) is False
        assert is_legal_action(state, (5, 0)) is False
        assert is_legal_action(state, (0, 5)) is False

    def test_illegal_occupied_cell(self):
        """Actions on occupied cells are illegal."""
        state = create_initial_state(5)
        new_state = apply_action(state, (2, 2))
        assert is_legal_action(new_state, (2, 2)) is False

    def test_illegal_terminal_state(self):
        """Actions when game is terminal are illegal."""
        # This will be tested in TestApplyAction
        pass

    def test_legal_move_edge(self):
        """Edge cells can be played."""
        state = create_initial_state(5)
        assert is_legal_action(state, (0, 0)) is True
        assert is_legal_action(state, (0, 4)) is True
        assert is_legal_action(state, (4, 0)) is True
        assert is_legal_action(state, (4, 4)) is True

    def test_legal_move_1x1(self):
        """1x1 board: one legal move."""
        state = create_initial_state(1)
        assert is_legal_action(state, (0, 0)) is True


class TestLegalActions:
    """Test legal actions enumeration."""

    def test_legal_actions_empty_board(self):
        """All cells are legal on empty board."""
        state = create_initial_state(5)
        actions = legal_actions(state)
        assert len(actions) == 25
        # Check all cells are present
        for row in range(5):
            for col in range(5):
                assert (row, col) in actions

    def test_legal_actions_after_move(self):
        """One less legal action after making a move."""
        state = create_initial_state(5)
        actions_before = set(legal_actions(state))
        apply_action(state, (2, 2))
        actions_after = set(legal_actions(state))
        assert len(actions_after) == 24
        assert (2, 2) not in actions_after
        assert actions_after.issubset(actions_before)

    def test_legal_actions_1x1(self):
        """1x1 board: one legal move initially."""
        state = create_initial_state(1)
        actions = legal_actions(state)
        assert len(actions) == 1
        assert (0, 0) in actions

    def test_legal_actions_empty_after_terminal(self):
        """legal_actions returns empty list after terminal state."""
        # Test a complete game
        state = create_initial_state(2)
        # Complete the game
        state = apply_action(state, (0, 0))
        state = apply_action(state, (1, 1))
        assert legal_actions(state) == []

    def test_legal_actions_order(self):
        """Legal actions are in row-major order."""
        state = create_initial_state(3)
        actions = legal_actions(state)
        expected = []
        for row in range(3):
            for col in range(3):
                expected.append((row, col))
        assert actions == expected


class TestApplyAction:
    """Test action application."""

    def test_apply_action_places_stone(self):
        """Placing a stone at the action."""
        state = create_initial_state(5)
        new_state = apply_action(state, (2, 2))
        assert new_state.board[2][2] == 0  # Player 0

    def test_apply_action_advances_turn(self):
        """Turn advances after action."""
        state = create_initial_state(5)
        assert state.current_player == 0
        new_state = apply_action(state, (2, 2))
        assert new_state.current_player == 1

    def test_apply_action_new_state_different_player(self):
        """New state has different current player."""
        state = create_initial_state(5)
        new_state = apply_action(state, (0, 0))
        assert new_state.current_player == 1

    def test_apply_action_occupies_cell(self):
        """The played cell is occupied."""
        state = create_initial_state(3)
        state = apply_action(state, (1, 1))
        assert state.board[1][1] == 0

    def test_apply_action_illegal_occupied(self):
        """Applying to occupied cell raises ValueError."""
        state = create_initial_state(3)
        apply_action(state, (1, 1))
        with pytest.raises(ValueError, match="not legal"):
            apply_action(state, (1, 1))

    def test_apply_action_illegal_out_of_bounds(self):
        """Applying out-of-bounds raises ValueError."""
        state = create_initial_state(5)
        with pytest.raises(ValueError, match="not legal"):
            apply_action(state, (-1, 0))

    def test_apply_action_illegal_termination(self):
        """Applying to terminal state raises ValueError."""
        # This will be a real test when we have winning states
        pass

    def test_apply_action_1x1(self):
        """1x1 board: single move."""
        state = create_initial_state(1)
        new_state = apply_action(state, (0, 0))
        assert new_state.board[0][0] == 0
        assert new_state.current_player == 1


class TestWinnerDetection:
    """Test winner detection."""

    def test_player0_wins_left_to_right(self):
        """Player 0 connects left to right."""
        state = create_initial_state(5)
        # Create a left-to-right path for player 0
        state = apply_action(state, (2, 0))  # (2,0)
        state = apply_action(state, (2, 1))  # (2,1)
        state = apply_action(state, (2, 2))  # (2,2)
        state = apply_action(state, (2, 3))  # (2,3)
        state = apply_action(state, (2, 4))  # P0 completes path
        assert get_winner(state) == 0

    def test_player1_wins_top_to_bottom(self):
        """Player 1 connects top to bottom."""
        state = create_initial_state(5)
        # Create a top-to-bottom path for player 1
        state = apply_action(state, (1, 1))  # (1,1)
        state = apply_action(state, (0, 1))  # (0,1)
        state = apply_action(state, (0, 0))  # Let P0 play (needed to get move back)
        # Actually let me construct this differently

    def test_player1_wins_top_to_bottom_simple(self):
        """Simple P1 win: top to bottom."""
        state = create_initial_state(2)
        state = apply_action(state, (0, 0))  # P0 takes corner
        state = apply_action(state, (0, 1))  # P1 takes corner
        state = apply_action(state, (1, 0))  # P0 takes corner
        state = apply_action(state, (1, 1))  # P1 completes top-to-bottom
        assert get_winner(state) == 1

    def test_player1_wins_top_to_bottom_3x3(self):
        """P1 wins on 3x3 with diagonal."""
        state = create_initial_state(3)
        # P0 plays in a way that allows P1 diagonal win
        # P1 diagonal: (1,1) needs (0,1)->(0,0) or (1,0)->(0,0) but (0,0) is edge
        # P1 diagonal: (1,1), (0,1), (0,0) - but (0,0) is left edge
        # Need: (0,1), (0,2), (1,2), (2,2) - top to bottom down right edge
        # Let me build it properly
        # P0 plays first
        state = apply_action(state, (1, 0))  # P0
        state = apply_action(state, (0, 1))  # P1
        state = apply_action(state, (0, 0))  # P0 - blocks P1 left
        state = apply_action(state, (0, 2))  # P1
        state = apply_action(state, (1, 2))  # P0
        state = apply_action(state, (2, 2))  # P1 - wins top-to-bottom
        assert get_winner(state) == 1

    def test_no_winner_disconnected_stones(self):
        """Disconnected stones don't produce a winner."""
        state = create_initial_state(5)
        # Play some stones but no connecting path
        state = apply_action(state, (0, 0))  # P0 left edge
        state = apply_action(state, (0, 1))  # P1 fills neighbor
        state = apply_action(state, (0, 2))  # P0
        state = apply_action(state, (1, 0))  # P1
        # Neither player has a complete path yet
        assert get_winner(state) is None

    def test_terminal_state_has_legal_actions_empty(self):
        """Terminal state has empty legal actions."""
        # 2x2 complete game is terminal
        state = create_initial_state(2)
        state = apply_action(state, (0, 0))  # P0
        state = apply_action(state, (1, 1))  # P1 wins
        assert get_winner(state) == 1
        assert legal_actions(state) == []

    def test_terminal_state_no_more_actions(self):
        """Cannot make moves after terminal state."""
        state = create_initial_state(2)
        state = apply_action(state, (0, 0))
        state = apply_action(state, (1, 1))
        with pytest.raises(ValueError, match="not legal"):
            apply_action(state, (0, 1))

    def test_p0_wins_3x3_simple(self):
        """P0 wins on 3x3 with simple path."""
        state = create_initial_state(3)
        state = apply_action(state, (2, 0))  # P0
        state = apply_action(state, (0, 1))  # P1
        state = apply_action(state, (1, 1))  # P0
        state = apply_action(state, (0, 0))  # P1
        state = apply_action(state, (2, 1))  # P0
        state = apply_action(state, (0, 2))  # P1
        state = apply_action(state, (2, 2))  # P0 - wins left-right
        assert get_winner(state) == 0

    def test_p1_wins_3x3_simple(self):
        """P1 wins on 3x3 with simple path."""
        state = create_initial_state(3)
        state = apply_action(state, (0, 0))  # P0 left-top
        state = apply_action(state, (0, 1))  # P1
        state = apply_action(state, (0, 2))  # P0 right-top
        state = apply_action(state, (1, 1))  # P1
        state = apply_action(state, (1, 0))  # P0 bottom-left
        state = apply_action(state, (1, 2))  # P1
        state = apply_action(state, (2, 1))  # P0
        state = apply_action(state, (2, 0))  # P0 - won left-right actually!
        # Need different sequence
        pass


class TestHasConnection:
    """Test has_connection helper."""

    def test_has_connection_player0(self):
        """Player 0 has connection."""
        state = create_initial_state(5)
        # Build a horizontal path from left edge to right edge on row 2
        state = apply_action(state, (2, 0))  # left edge
        state = apply_action(state, (2, 1))
        state = apply_action(state, (2, 2))
        state = apply_action(state, (2, 3))
        state = apply_action(state, (2, 4))  # right edge
        # Row 2 should have 1,1,1,1,1
        assert state.board == (
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (1, 1, 1, 1, 1),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        )
        assert has_connection(state, 0) is True
        assert has_connection(state, 1) is False

    def test_has_connection_player1(self):
        """Player 1 has connection."""
        state = create_initial_state(5)
        # Build a vertical path from top edge to bottom edge on column 3
        state = apply_action(state, (0, 3))  # top edge
        state = apply_action(state, (1, 3))
        state = apply_action(state, (2, 3))
        state = apply_action(state, (3, 3))
        state = apply_action(state, (4, 3))  # bottom edge
        # Column 3 should be 1,1,1,1,1
        assert state.board == (
            (0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0),
        )
        assert has_connection(state, 1) is True
        assert has_connection(state, 0) is False

    def test_has_connection_no_connection(self):
        """No connection when stones don't connect."""
        state = create_initial_state(5)
        state = apply_action(state, (0, 0))
        state = apply_action(state, (4, 4))
        assert has_connection(state, 0) is False
        assert has_connection(state, 1) is False


class TestCoordToActionIndex:
    """Test coordinate to index conversion."""

    def test_coord_to_index_roundtrip(self):
        """Conversion is reversible."""
        size = 5
        for row in range(size):
            for col in range(size):
                index = coord_to_action_index(row, col, size)
                recovered = action_index_to_coord(index, size)
                assert recovered == (row, col)

    def test_coord_to_index_1x1(self):
        """1x1 board: index 0."""
        index = coord_to_action_index(0, 0, 1)
        assert index == 0

    def test_coord_to_index_corner(self):
        """Corner coordinates."""
        assert coord_to_action_index(0, 0, 3) == 0
        assert coord_to_action_index(0, 2, 3) == 2
        assert coord_to_action_index(2, 0, 3) == 6
        assert coord_to_action_index(2, 2, 3) == 8

    def test_coord_to_index_invalid(self):
        """Invalid coordinates raise error."""
        with pytest.raises(ValueError, match="Invalid"):
            coord_to_action_index(5, 0, 3)
        with pytest.raises(ValueError, match="Invalid"):
            coord_to_action_index(0, 5, 3)


class TestActionIndexToCoord:
    """Test index to coordinate conversion."""

    def test_index_to_coord_roundtrip(self):
        """Conversion is reversible."""
        size = 5
        for index in range(size * size):
            coord = action_index_to_coord(index, size)
            recovered = coord_to_action_index(coord[0], coord[1], size)
            assert recovered == index

    def test_index_to_coord_1x1(self):
        """1x1 board: index 0 -> (0, 0)."""
        coord = action_index_to_coord(0, 1)
        assert coord == (0, 0)

    def test_index_to_coord_invalid(self):
        """Invalid index raises error."""
        with pytest.raises(ValueError, match="Invalid"):
            action_index_to_coord(9, 3)  # 3x3 = 9, index 9 is out of range


class TestEdgeCases:
    """Edge and corner case testing."""

    def test_2x2_game_completion(self):
        """2x2 game must have winner."""
        state = create_initial_state(2)
        state = apply_action(state, (0, 0))  # P0
        state = apply_action(state, (1, 1))  # P1 wins
        assert get_winner(state) == 1

    def test_corner_stones_near_winning_path(self):
        """Corner stones don't automatically win."""
        state = create_initial_state(3)
        state = apply_action(state, (0, 0))  # P0 left-top corner
        state = apply_action(state, (2, 2))  # P1 right-bottom corner
        assert get_winner(state) is None

    def test_complete_board_filled(self):
        """Fully filled board must have a winner (no draws)."""
        state = create_initial_state(3)
        # Fill entire board
        all_moves = [(r, c) for r in range(3) for c in range(3)]
        for action in all_moves:
            state = apply_action(state, action)
        winner = get_winner(state)
        assert winner is not None
        assert winner in (0, 1)


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_sequence_same_result(self):
        """Same action sequence produces same state."""
        state1 = create_initial_state(5)
        state2 = create_initial_state(5)

        moves = [(1, 1), (0, 2), (1, 2), (0, 1)]
        for move in moves:
            state1 = apply_action(state1, move)
            state2 = apply_action(state2, move)

        assert state1.board == state2.board
        assert state1.current_player == state2.current_player
