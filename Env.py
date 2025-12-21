import gymnasium as gym
import numpy as np
from typing import Optional

class UTTTEnv(gym.Env):

    def __init__(self):
        self.grid = np.zeros((9, 9))

        # Each index represents a board, i.e. idx 2 = top right board
        # 0 : playable, 1: won by X, -1: won by O, 2 unplayable but unclaimed, 3: tie
        self.mini_board_states = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.current_player = 1 # 1 for player 1 (X's), -1 for player 2 (O's)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 9, 9), dtype=np.int32) # 3 channels, and a 9x9 grid
        self.action_space = gym.spaces.Discrete(81)




    def _get_obs(self):
        """ Converts the internal state into the correct format for agent observations
            Returns:
                np array: 3x9x9 tensor as specified in the observation space definition
        """
        obs = np.zeros((3, 9, 9))
        obs[0] = (self.grid == self.current_player).astype(np.float32)
        obs[1] = (self.grid == -self.current_player).astype(np.float32)
        obs[2] = self.get_playable_board_mask()




    # ================= Helper Functions =================
    def _get_playable_board_mask(self):
        playable_status = self.mini_board_states.reshape(3, 3)
        playable_grid = np.kron(playable_status, np.ones((3, 3)))


    def _int_to_grid(tile_num: int):
        x = tile_num % 9
        y = tile_num // 9
        return np.array([x, y])

    def _grid_to_int(x: int, y: int):
        return (9 * y) + x




        