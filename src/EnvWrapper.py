import numpy as np
import gymnasium as gym
from typing import Optional

class SingleAgentTrainingWrapper(gym.Wrapper):
    def __init__(self, env, opponent):
        super().__init__(env)
        self.learning_agent = 1 # THis can be 1 or -1, doesn't really matter
        self.opponent = opponent

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        obs, info = self.env.reset(seed)

        if self.env.current_player != self.learning_agent:
            action_mask = obs[3, :, :].flatten()
            action = self.opponent.pick_action(obs, action_mask)
            self.env.apply_action(action)
        
        return self.env.get_obs(), self.env._get_info()

    def step(self, action):
        old_board_states = self.env.mini_board_states.copy()
        if self.env.apply_action(action): # If learning agent's move just won the game
            return self.env._get_obs(), 1, True, False, self.env._get_info()

        obs = self.env._get_obs()
        action_mask = obs[3, :, :].flatten()
        action = self.opponent.pick_action(obs, action_mask)
        terminated = self.env.apply_action(action)

        rewards = self.reward_func(old_board_states, terminated)
        return self.env._get_obs(), rewards, terminated, False, self.env._get_info()
            
    def reward_func(self, old_board_states, terminated):
        if terminated:
            # Sparse rewards: only give reward at end of game
            macro_state = self.env.mini_board_states.reshape(3, 3)
            winner = self.env._check_3x3_state(macro_state)
            if winner == -self.learning_agent: return -1
            elif winner == 3: return 0 # Tie
        
        # Dense event-based rewards: small reward for winning mini-boards
        reward = 0
        if np.any((old_board_states != self.env.mini_board_states) & (self.env.mini_board_states == self.learning_agent)):
            reward += 0.1 # Won a mini-board
        if np.any((old_board_states != self.env.mini_board_states) & (self.env.mini_board_states == -self.learning_agent)):
            reward -= 0.1 # Lost a mini-board

        return reward