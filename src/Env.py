import gymnasium as gym
import numpy as np
from UTTTGame import UTTTGame
from typing import Optional

class UTTTEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        self.game = UTTTGame()
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
        self.game.reset(seed=seed)

        observations = self._get_obs()
        info = self._get_info()

        return observations, info

    def step(self, action):
        terminated = self.game.apply_action(action)
        rewards = 0 # Base environment provides no reward signal, handled by wrapper

        return self._get_obs(), rewards, terminated, False, self._get_info()

    def render(self):
        return self.game.render()

    # ================= Helper Functions =================  
    def close(self):
        self.game.close()

    def _get_obs(self):
        """ Converts the internal state into the correct format for agent observations
            Returns:
                np array: 3x9x9 tensor as specified in the observation space definition
        """
        chan0 = (self.game.grid == self.game.current_player).astype(np.float32)
        chan1 = (self.game.grid == -self.game.current_player).astype(np.float32)
        chan2 = self.game._get_grid_with_condition(0) # Active Board(s)
        chan3 = self.game._get_legal_moves() # Currently playable tiles
        chan4 = self.game._get_grid_with_condition(1) # Boards won by agent
        chan5 = self.game._get_grid_with_condition(-1) # Boards won by opponent
        chan6 = self.game._get_grid_with_condition(3) # Boards tied

        obs = np.stack([chan0, chan1, chan2, chan3, chan4, chan5, chan6], axis=0)
        return obs

    def _get_info(self) -> dict:
        """
        Give human readable info for debugging
        """
        return self.game._get_info()
  
