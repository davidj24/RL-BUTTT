import gymnasium as gym
import numpy as np
from typing import Optional

class UTTTEnv(gym.Env):

    def __init__(self):
        self.grid = np.zeros((9, 9))

        
        # 0: playable, 1: won by current player, -1: won by opponent, 2 unplayable but unclaimed, 3: tie
        self.mini_board_states = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # Each index represents a board, i.e. idx 2 = top right board

        self.current_player = 1 # 1 for player 1 (X's), -1 for player 2 (O's)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, 9, 9), dtype=np.int32) # 3 channels, and a 9x9 grid
        self.action_space = gym.spaces.Discrete(81)




    def _get_obs(self):
        """ Converts the internal state into the correct format for agent observations
            Returns:
                np array: 3x9x9 tensor as specified in the observation space definition
        """
        chan0 = (self.grid == self.current_player).astype(np.float32)
        chan1 = (self.grid == -self.current_player).astype(np.float32)
        chan2 = self._get_boards(0)
        chan3 = self._get_boards(1) # Boards won by agent
        chan4 = self._get_boards(-1) # Boards won by opponent
        chan5 = self._get_boards(3) # Boards tied

        obs = np.stack([chan0, chan1, chan2, chan3, chan4, chan5], axis=0)
        return obs




    # ================= Helper Functions =================    
    def _get_boards(self, condition: int):
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

    def _int_to_grid(tile_num: int):
        x = tile_num % 9
        y = tile_num // 9
        return np.array([x, y])

    def _grid_to_int(x: int, y: int):
        return (9 * y) + x




        