import numpy as np
import pygame
from typing import Optional

class UTTTGame():
    def __init__(self):
        # ======================== Game Variables ========================
        self.grid = np.zeros((9, 9)) # 0 for empty, 1 for player 1 (X's), -1 for player 2 (O's)
        self.current_player = 1 # 1 for player 1 (X's), -1 for player 2 (O's)

        # 0: playable, 1: won by player1, -1: won by player2, 2 inactive, 3: tie
        self.mini_board_states = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # Each index represents a board, i.e. idx 2 = top right board

        self.rng = np.random.default_rng()

        # Rendering variables
        self.window_size = 540  # 60 pixels per small cell
        self.window = None
        self.clock = None
        self.render_fps = 30

    def reset(self, seed: Optional[int] = None):
        """Resets the board."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.grid.fill(0)
        self.mini_board_states.fill(0)
        self.current_player = self.rng.choice([1, -1])

    def apply_action(self, action: int) -> bool:
        """
        Purely updates the internal state of the game board and players

        Args:
            action: an integer [0, 80] representing the index where the mark should be placed

        Returns:
            bool: whether or not the game is over after the move is applied
        """
        legal_moves = self._get_legal_moves().flatten()
        if legal_moves[action] == 0:
            return False
        game_over = False
        
        # Place new mark
        action_entry = self._int_to_entry(action)
        self.grid[action_entry[0]][action_entry[1]] = self.current_player

        # Check if the mini board was newly won
        board_that_got_played_in = self._action_entry_to_board_num(action_entry)
        new_board_state = self._check_3x3_state(self._board_num_to_3x3(board_that_got_played_in))
        self.mini_board_states[board_that_got_played_in] = new_board_state

        if new_board_state != 0:
            game_over = self._check_3x3_state(self.mini_board_states.reshape(3, 3)) != 0

        # For next turn, set active player and active boards
        new_active_boards = self._get_new_active_board(action_entry)
        self.mini_board_states = np.where(np.isin(self.mini_board_states, [1, -1, 3]), self.mini_board_states, 2) # Set all non-won boards to inactive
        self.mini_board_states[new_active_boards] = 0 # Set new active boards to playable
        self.current_player *= -1 # Switch who's turn it is

        return game_over

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # White background

        pix_square_size = self.window_size / 9

        # 1. Highlight Active Boards (Light Yellow)
        for i in range(9):
            if self.mini_board_states[i] == 0: # 0 means active/playable
                row = (i // 3) * 3
                col = (i % 3) * 3
                rect = pygame.Rect(col * pix_square_size, row * pix_square_size, pix_square_size * 3, pix_square_size * 3)
                pygame.draw.rect(canvas, (255, 255, 200), rect)

        # 2. Draw Grid Lines
        for i in range(1, 9):
            # Thin grey lines for small cells
            thickness = 1
            color = (200, 200, 200)
            
            # Thick black lines for macro board
            if i % 3 == 0:
                thickness = 4
                color = (0, 0, 0)
            
            # Horizontal
            pygame.draw.line(canvas, color, (0, i * pix_square_size), (self.window_size, i * pix_square_size), thickness)
            # Vertical
            pygame.draw.line(canvas, color, (i * pix_square_size, 0), (i * pix_square_size, self.window_size), thickness)

        # 3. Draw X's and O's (Small)
        for r in range(9):
            for c in range(9):
                if self.grid[r, c] == 1:
                    color = (255, 0, 0) # Red X
                    start_pos = (c * pix_square_size + 10, r * pix_square_size + 10)
                    end_pos = ((c + 1) * pix_square_size - 10, (r + 1) * pix_square_size - 10)
                    pygame.draw.line(canvas, color, start_pos, end_pos, 3)
                    pygame.draw.line(canvas, color, (start_pos[0], end_pos[1]), (end_pos[0], start_pos[1]), 3)
                elif self.grid[r, c] == -1:
                    color = (0, 0, 255) # Blue O
                    center = (int((c + 0.5) * pix_square_size), int((r + 0.5) * pix_square_size))
                    pygame.draw.circle(canvas, color, center, int(pix_square_size / 2 - 8), 3)

        # 4. Draw Big X's and O's for won mini-boards
        for i in range(9):
            state = self.mini_board_states[i]
            if state == 1 or state == -1:
                row = (i // 3) * 3
                col = (i % 3) * 3
                center_x = (col + 1.5) * pix_square_size
                center_y = (row + 1.5) * pix_square_size
                
                if state == 1: # Big Red X
                    pygame.draw.line(canvas, (255, 0, 0), (center_x - 60, center_y - 60), (center_x + 60, center_y + 60), 10)
                    pygame.draw.line(canvas, (255, 0, 0), (center_x - 60, center_y + 60), (center_x + 60, center_y - 60), 10)
                elif state == -1: # Big Blue O
                    pygame.draw.circle(canvas, (0, 0, 255), (int(center_x), int(center_y)), 70, 10)

        if self.window is not None:
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)


    # ================= Helper Functions =================  
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_legal_moves(self):
        active_board_grid = self._get_grid_with_condition(0)
        empty_squares = (self.grid == 0).astype(np.float32)
        legal_moves_grid = active_board_grid * empty_squares # This is element wise mult, so 1's where it's empty and on active board
        return legal_moves_grid

    def _get_info(self):
        """
        Give human readable info for debugging

        Returns:
            dict: dictionary containing human readable info about game state
        """
        return {
            "board_states": self.mini_board_states.copy(),
            "num_player_1_won_boards": np.sum(self.mini_board_states == 1),
            "num_player_2_won_boards": np.sum(self.mini_board_states == -1),
            "legal_moves": self._get_legal_moves()
            }
  
    def _get_grid_with_condition(self, condition: int):
        """
        Returns the grid with 1's where condition is met, and 0's elsewhere.
        
        int: condition is a number representing a condition: 
            0: currently playable
            1: Won by player1
            -1: Won by player2
            2: unclaimed but currently unplayable
            3: tied
        """
        mini_board_states_with_cond = (self.mini_board_states == condition).astype(np.float32)
        boards_matrix = mini_board_states_with_cond.reshape(3, 3) # 3x3
        return np.kron(boards_matrix, np.ones((3, 3), dtype=np.float32)) # 9x9

    def _check_3x3_state(self, board_slice):
        """
        Args:
            board_slice: A 3x3 NumPy array (either a mini-board or the mega-board)
        Returns:
            int: 1 if X wins, -1 if O wins, 0 if no winner yet, 3 if tie
        """
        # Check rows and cols
        for p in [1, -1]:
            if np.any(np.all(board_slice == p, axis=0)) or np.any(np.all(board_slice == p, axis=1)):
                return p
        
        # Check diagonals
        if np.all(np.diag(board_slice) == 1) or np.all(np.diag(np.fliplr(board_slice)) == 1):
            return 1
        if np.all(np.diag(board_slice) == -1) or np.all(np.diag(np.fliplr(board_slice)) == -1):
            return -1
        
        # Check for tie (no 0s for empty, no 2s for inactive-but-playable)
        if not np.any(board_slice == 0) and not np.any(board_slice == 2):
            return 3
        
        return 0
    
    def _get_new_active_board(self, action_entry):
        """
        Given the last action entry, return the new active board number
        If that board is unplayable, return all playable boards
        """
        row = action_entry[0] % 3
        col = action_entry[1] % 3
        board_num = (row * 3) + col
        if self.mini_board_states[board_num] not in [1, -1, 3]:
            return board_num
        else:
            return np.where(np.isin(self.mini_board_states, [0, 2]))[0]



    def _int_to_entry(self, tile_num: int):
        return np.array(divmod(tile_num, 9))

    def _entry_to_int(self, x: int, y: int):
        return (9 * y) + x

    def _board_num_to_3x3(self, board_num: int):
        """
        Return a 3x3 slice of the grid representing one board

        Args:
            int: the board number, [0, 8] where 0 is top left, 2 is top right

        Returns:
            np array: 3x3 slice of the grid representing the mini board corresponding to input int
        """
        row = (board_num // 3) * 3
        col = (board_num % 3) * 3

        return self.grid[row:row+3, col:col+3]
        
    def _action_entry_to_board_num(self, action_entry):
        return ((action_entry[0] // 3) * 3) + (action_entry[1] // 3)

   
        