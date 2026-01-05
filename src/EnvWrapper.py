import numpy as np
import gymnasium as gym
import os
import sys
import torch
import random
from src.Opponent import Opponent
from src.Opponent import FrozenAgentOpponent
from src.Agent import Agent
from typing import Optional

class SingleAgentTrainingWrapper(gym.Wrapper):
    def __init__(self, env, opponent_pool_paths, swap_to_newest_model_prob: float=0.8):
        super().__init__(env)
        self.learning_agent = 1 # This can be 1 or -1, doesn't really matter
        self.opponent_pool_paths = opponent_pool_paths
        self.newest_model_prob = swap_to_newest_model_prob

        if self.opponent_pool_paths:
            self.opponent = FrozenAgentOpponent(name="opponent_container", path_to_model=self.opponent_pool_paths[-1])
        else:
            self.opponent = FrozenAgentOpponent(name="opponent_container")

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        obs, info = self.unwrapped.reset(seed=seed, options=options)

        if len(self.opponent_pool_paths) > 1:
            model = self.opponent_pool_paths[-1] if random.random() <= self.newest_model_prob else random.choice(self.opponent_pool_paths)
            self.opponent.load(path_to_model=model)
            print(f"Opponent is now: {model}")

        if self.unwrapped.game.current_player != self.learning_agent: # if game starts with opponent's turn
            action_mask = obs[3].flatten()
            action = self.opponent.pick_action(obs, action_mask)
            self.unwrapped.game.apply_action(action)
            return self.unwrapped._get_obs(), self.unwrapped._get_info()
        
        return obs, info

    def step(self, action):
        old_board_states = self.unwrapped.game.mini_board_states.copy()
        if self.unwrapped.game.apply_action(action): # If learning agent's move just won the game
            if self.unwrapped.game._check_3x3_state(self.unwrapped.game.mini_board_states.reshape(3, 3)) == self.learning_agent:
                return self.unwrapped._get_obs(), 1, True, False, self.unwrapped._get_info() # Agent won the game
            else:
                return self.unwrapped._get_obs(), 0, True, False, self.unwrapped._get_info() # Agent's move tied the game

        obs = self.unwrapped._get_obs()
        action_mask = obs[3].flatten()
        action = self.opponent.pick_action(obs, action_mask)
        terminated = self.unwrapped.game.apply_action(action)

        rewards = self.reward_func(old_board_states, terminated)
        return self.unwrapped._get_obs(), rewards, terminated, False, self.unwrapped._get_info()
            
    def reward_func(self, old_board_states, terminated):
        if terminated:
            # Sparse rewards: only give reward at end of game
            macro_state = self.unwrapped.game.mini_board_states.reshape(3, 3)
            winner = self.unwrapped.game._check_3x3_state(macro_state)
            if winner == -self.learning_agent: return -1
            elif winner == 3: return 0 # Tie
        
        # Dense event-based rewards: small reward for winning mini-boards
        reward = 0
        if np.any((old_board_states != self.unwrapped.game.mini_board_states) & (self.unwrapped.game.mini_board_states == self.learning_agent)):
            reward += 0.1 # Won a mini-board
        if np.any((old_board_states != self.unwrapped.game.mini_board_states) & (self.unwrapped.game.mini_board_states == -self.learning_agent)):
            reward -= 0.1 # Lost a mini-board

        return reward

    def add_new_opponent(self, opponent_path: str):
        self.opponent_pool_paths.append(opponent_path)


