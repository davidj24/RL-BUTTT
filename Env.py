import gymnasium as gym
import numpy as np
from typing import Optional

class UTTTEnv(gym.Env):

    def __init__(self):
        # ======================== Game Variables ========================
        self.grid = np.zeros((9, 9)) # 0 for empty, 1 for player 1 (X's), -1 for player 2 (O's)
        self.current_player = 1 # 1 for player 1 (X's), -1 for player 2 (O's)

        # 0: playable, 1: won by player1, -1: won by player2, 2 inactive, 3: tie
        self.mini_board_states = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # Each index represents a board, i.e. idx 2 = top right board



        # ======================== Agent variables ========================
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, 9, 9), dtype=np.float32) # 7 channels, and a 9x9 grid
        self.action_space = gym.spaces.Discrete(81)

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
        self.current_player *= -1
        new_active_boards = self._get_new_active_board(action_entry)
        self.mini_board_states = np.where(np.isin(self.mini_board_states, [1, -1, 3]), self.mini_board_states, 2) # Set all non-won boards to inactive
        self.mini_board_states[new_active_boards] = 0 # Set new active boards to playable


        rewards = self.reward_func()

        return self._get_obs(), rewards, terminated, False, self._get_info()

    def reward_func(self):
        return 0


    # ================= Helper Functions =================  
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

    def _get_new_active_board(self, action_entry):
        """
        Given the last action entry, return the new active board number
        If that board is unplayable, return all playable boards
        """
        row = action_entry[0] % 3
        col = action_entry[1] % 3
        board_num = (row * 3) + col
        if self.mini_board_states[board_num] == 0:
            return board_num
        else:
            return np.where(self.mini_board_states == 0)[0]


        