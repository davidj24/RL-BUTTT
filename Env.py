import gymnasium as gym
import numpy as np
import pygame
from typing import Optional

class UTTTEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        # ======================== Game Variables ========================
        self.grid = np.zeros((9, 9)) # 0 for empty, 1 for player 1 (X's), -1 for player 2 (O's)
        self.current_player = 1 # 1 for player 1 (X's), -1 for player 2 (O's)

        # 0: playable, 1: won by player1, -1: won by player2, 2 inactive, 3: tie
        self.mini_board_states = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # Each index represents a board, i.e. idx 2 = top right board



        # ======================== Agent variables ========================
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, 9, 9), dtype=np.float32) # 7 channels, and a 9x9 grid
        self.action_space = gym.spaces.Discrete(81)

        # Rendering variables
        self.window_size = 540  # 60 pixels per small cell
        self.window = None
        self.clock = None

    def reset(self, seed:Optional[int]=None, options: Optional[dict]=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)
        self.grid.fill(0)
        self.mini_board_states.fill(0)
        self.current_player = self.np_random.choice([1, -1])

        observations = self._get_obs()
        info = self._get_info()

        return observations, info

    def step(self, action):
        action_mask = self._get_action_mask_grid().flatten()
        if action_mask[action] == 0:
            return
        terminated = False

        old_board_states = self.mini_board_states.copy()
        
        # Place new mark
        action_entry = self._int_to_entry(action)
        self.grid[action_entry[0]][action_entry[1]] = self.current_player

        # Check if the mini board was newly won
        board_that_got_played_in = self._action_entry_to_board_num(action_entry)
        new_board_state = self._check_3x3_state(self._board_num_to_3x3(board_that_got_played_in))
        self.mini_board_states[board_that_got_played_in] = new_board_state

        if new_board_state != 0:
            terminated = self._check_3x3_state(self.mini_board_states.reshape(3, 3)) != 0
            

        # For next turn, set active player and active boards
        new_active_boards = self._get_new_active_board(action_entry)
        print(f"New active boards: {new_active_boards}")
        self.mini_board_states = np.where(np.isin(self.mini_board_states, [1, -1, 3]), self.mini_board_states, 2) # Set all non-won boards to inactive
        self.mini_board_states[new_active_boards] = 0 # Set new active boards to playable

        rewards = self.reward_func(old_board_states)

        self.current_player *= -1

        return self._get_obs(), rewards, terminated, False, self._get_info()

    def reward_func(self, old_board_states):
        # Sparse rewards: only give reward at end of game
        macro_state = self.mini_board_states.reshape(3, 3)
        winner = self._check_3x3_state(macro_state)
        if winner == self.current_player: return 1
        elif winner == -self.current_player: return -1
        elif winner == 3: return 0 # Tie
        
        # Dense event-based rewards: small reward for winning mini-boards
        reward = 0
        if np.any((old_board_states != self.mini_board_states) & (self.mini_board_states == self.current_player)):
            reward += 0.1 # Won a mini-board

        time_penalty = 0
        reward += time_penalty

        return reward

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
            self.clock.tick(self.metadata["render_fps"])

    # ================= Helper Functions =================  
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_action_mask_grid(self):
        active_board_grid = self._get_grid_with_condition(0)
        empty_squares = (self.grid == 0).astype(np.float32)
        action_mask_grid = active_board_grid * empty_squares # This is element wise mult, so 1's where it's empty and on active board
        return action_mask_grid

    def _get_obs(self):
        """ Converts the internal state into the correct format for agent observations
            Returns:
                np array: 3x9x9 tensor as specified in the observation space definition
        """
        chan0 = (self.grid == self.current_player).astype(np.float32)
        chan1 = (self.grid == -self.current_player).astype(np.float32)
        chan2 = self._get_grid_with_condition(0) # Active Board(s)
        chan3 = self._get_action_mask_grid() # Currently playable tiles
        chan4 = self._get_grid_with_condition(1) # Boards won by agent
        chan5 = self._get_grid_with_condition(-1) # Boards won by opponent
        chan6 = self._get_grid_with_condition(3) # Boards tied

        obs = np.stack([chan0, chan1, chan2, chan3, chan4, chan5, chan6], axis=0)
        return obs

    def _get_info(self):
        """
        Give human readable info for debugging

        Returns:
            dict: Info with 
        """
        return {
            "board_states": self.mini_board_states.copy(),
            "agent_won_boards": np.sum(self.mini_board_states == 1),
            "opponent_won_boards": np.sum(self.mini_board_states == -1),
            }
  
    def _get_grid_with_condition(self, condition: int):
        """
        Returns the grid with 1's where condition is met, and 0's elsewhere.
        
        int: condition is a number representing a condition: 
            0: currently playable
            1: Won by agent
            -1: Won by opponent
            2: unclaimed but currently unplayable
            3: tied
        """
        mini_board_states_won = (self.mini_board_states == condition).astype(np.float32)
        boards_matrix = mini_board_states_won.reshape(3, 3) # 3x3
        return np.kron(boards_matrix, np.ones((3, 3))) # 9x9

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

   
        