from typing import Optional

import numpy as np
import pygame


EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
INACTIVE = 2
DRAW = 3


class IllegalMoveError(ValueError):
    """Raised when an action is not legal in the current game state."""


class UTTTGame:
    """Pure Ultimate Tic-Tac-Toe rules plus a small pygame viewer.

    Public state kept for compatibility:
    - grid: 9x9 array with 0 for empty, 1 for X, -1 for O.
    - current_player: player to move, either 1 or -1.
    - mini_board_states: compatibility view where 0 means active, 2 means
      inactive/unclaimed, 1/-1 mean won, and 3 means drawn.
    """

    def __init__(self):
        self.grid = np.zeros((9, 9), dtype=np.int8)
        self.board_results = np.zeros(9, dtype=np.int8)
        self.active_boards = np.ones(9, dtype=bool)
        self.current_player = PLAYER_X
        self.winner = EMPTY
        self.is_terminal = False

        self.rng = np.random.default_rng()

        self.window_size = 540
        self.window = None
        self.clock = None
        self.render_fps = 30

    @property
    def mini_board_states(self) -> np.ndarray:
        """Compatibility view used by the older env/wrapper code."""
        states = np.full(9, INACTIVE, dtype=np.int8)
        states[self.active_boards] = EMPTY
        finished = self.board_results != EMPTY
        states[finished] = self.board_results[finished]
        return states

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid.fill(EMPTY)
        self.board_results.fill(EMPTY)
        self.active_boards.fill(True)
        self.current_player = int(self.rng.choice([PLAYER_X, PLAYER_O]))
        self.winner = EMPTY
        self.is_terminal = False

    def apply_action(self, action: int) -> bool:
        """Apply one legal action and return whether the game is over."""
        if self.is_terminal:
            raise IllegalMoveError("Cannot play after the game has ended.")

        row, col = self._int_to_entry(action)
        if not self._is_legal_entry(row, col):
            raise IllegalMoveError(f"Action {action} is illegal for the current state.")

        self.grid[row, col] = self.current_player
        played_board = self._action_entry_to_board_num((row, col))
        self.board_results[played_board] = self._check_3x3_state(
            self._board_num_to_3x3(played_board)
        )

        self.winner = self._check_3x3_state(self.board_results.reshape(3, 3))
        self.is_terminal = self.winner != EMPTY

        if not self.is_terminal:
            self.active_boards = self._next_active_boards(row, col)
            self.current_player *= -1

        return self.is_terminal

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / 9

        for board_num in np.flatnonzero(self.active_boards):
            row = (board_num // 3) * 3
            col = (board_num % 3) * 3
            rect = pygame.Rect(
                col * pix_square_size,
                row * pix_square_size,
                pix_square_size * 3,
                pix_square_size * 3,
            )
            pygame.draw.rect(canvas, (255, 255, 200), rect)

        for i in range(1, 9):
            thickness = 4 if i % 3 == 0 else 1
            color = (0, 0, 0) if i % 3 == 0 else (200, 200, 200)
            pygame.draw.line(canvas, color, (0, i * pix_square_size), (self.window_size, i * pix_square_size), thickness)
            pygame.draw.line(canvas, color, (i * pix_square_size, 0), (i * pix_square_size, self.window_size), thickness)

        for row in range(9):
            for col in range(9):
                self._draw_cell_mark(canvas, row, col, pix_square_size)

        for board_num, result in enumerate(self.board_results):
            if result in (PLAYER_X, PLAYER_O):
                self._draw_board_result(canvas, board_num, int(result), pix_square_size)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self._get_legal_moves().ravel())

    def _get_legal_moves(self) -> np.ndarray:
        active_board_grid = self._boards_to_grid(self.active_boards.astype(np.float32))
        empty_squares = (self.grid == EMPTY).astype(np.float32)
        return active_board_grid * empty_squares

    def _get_info(self):
        return {
            "board_states": self.mini_board_states.copy(),
            "board_results": self.board_results.copy(),
            "active_boards": self.active_boards.copy(),
            "winner": self.winner,
            "num_player_1_won_boards": np.sum(self.board_results == PLAYER_X),
            "num_player_2_won_boards": np.sum(self.board_results == PLAYER_O),
            "legal_moves": self._get_legal_moves(),
        }

    def _get_grid_with_condition(self, condition: int) -> np.ndarray:
        return self._boards_to_grid((self.mini_board_states == condition).astype(np.float32))

    def _check_3x3_state(self, board_slice) -> int:
        for player in (PLAYER_X, PLAYER_O):
            if np.any(np.all(board_slice == player, axis=0)):
                return player
            if np.any(np.all(board_slice == player, axis=1)):
                return player
            if np.all(np.diag(board_slice) == player):
                return player
            if np.all(np.diag(np.fliplr(board_slice)) == player):
                return player

        if not np.any((board_slice == EMPTY) | (board_slice == INACTIVE)):
            return DRAW

        return EMPTY

    def _get_new_active_board(self, action_entry):
        row, col = action_entry
        return np.flatnonzero(self._next_active_boards(row, col))

    def _int_to_entry(self, tile_num: int):
        if not 0 <= int(tile_num) < 81:
            raise IllegalMoveError(f"Action must be in [0, 80], got {tile_num}.")
        return np.array(divmod(int(tile_num), 9), dtype=np.int8)

    def _entry_to_int(self, row: int, col: int):
        return (9 * int(row)) + int(col)

    def _board_num_to_3x3(self, board_num: int):
        row = (int(board_num) // 3) * 3
        col = (int(board_num) % 3) * 3
        return self.grid[row:row + 3, col:col + 3]

    def _action_entry_to_board_num(self, action_entry):
        row, col = action_entry
        return ((int(row) // 3) * 3) + (int(col) // 3)

    def _is_legal_entry(self, row: int, col: int) -> bool:
        board_num = self._action_entry_to_board_num((row, col))
        return (
            self.grid[row, col] == EMPTY
            and self.active_boards[board_num]
            and self.board_results[board_num] == EMPTY
        )

    def _next_active_boards(self, row: int, col: int) -> np.ndarray:
        target_board = (int(row) % 3) * 3 + (int(col) % 3)
        available_boards = self.board_results == EMPTY

        if available_boards[target_board]:
            active_boards = np.zeros(9, dtype=bool)
            active_boards[target_board] = True
            return active_boards

        return available_boards

    def _boards_to_grid(self, board_values: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(board_values.reshape(3, 3), 3, axis=0), 3, axis=1)

    def _draw_cell_mark(self, canvas, row: int, col: int, cell_size: float):
        value = self.grid[row, col]
        if value == PLAYER_X:
            start_pos = (col * cell_size + 10, row * cell_size + 10)
            end_pos = ((col + 1) * cell_size - 10, (row + 1) * cell_size - 10)
            pygame.draw.line(canvas, (255, 0, 0), start_pos, end_pos, 3)
            pygame.draw.line(canvas, (255, 0, 0), (start_pos[0], end_pos[1]), (end_pos[0], start_pos[1]), 3)
        elif value == PLAYER_O:
            center = (int((col + 0.5) * cell_size), int((row + 0.5) * cell_size))
            pygame.draw.circle(canvas, (0, 0, 255), center, int(cell_size / 2 - 8), 3)

    def _draw_board_result(self, canvas, board_num: int, result: int, cell_size: float):
        row = (board_num // 3) * 3
        col = (board_num % 3) * 3
        center_x = (col + 1.5) * cell_size
        center_y = (row + 1.5) * cell_size

        if result == PLAYER_X:
            pygame.draw.line(canvas, (255, 0, 0), (center_x - 60, center_y - 60), (center_x + 60, center_y + 60), 10)
            pygame.draw.line(canvas, (255, 0, 0), (center_x - 60, center_y + 60), (center_x + 60, center_y - 60), 10)
        elif result == PLAYER_O:
            pygame.draw.circle(canvas, (0, 0, 255), (int(center_x), int(center_y)), 70, 10)
