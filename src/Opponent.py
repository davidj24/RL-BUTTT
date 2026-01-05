import torch
import numpy as np
import random
from src.Agent import Agent
from abc import ABC, abstractmethod


class Opponent(ABC):
    """
    Abstract base class for opponents in single training loop for UTTT
    """
    @abstractmethod
    def pick_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """
        The core method that takes a board state and returns a move.

        Args:
            observation: The 7-channel 9x9 egocentric board state (for the opponent's turn)
            action_mask: A flattened (81,) array of 1's and 0's representing legal and illegal moves respectively

        Returns:
            The integer index (0-80) of the chosen move.
        """
        pass


    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the opponent type (e.g., 'RandomAgent', 'PPO-V1', 'Human)."""
        pass

class RandomOpponent(Opponent):
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"Random (Seed: {self.seed})"

    def pick_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        legal_actions = np.where(action_mask == 1)[0]
        return self.rng.choice(legal_actions)
    
class FrozenAgentOpponent(Opponent):
    def __init__(self, name: str, path_to_model: str=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.display_name = name
        self.agent = Agent()
        self.agent.to(self.device)
        self.agent.eval()

        if path_to_model is not None:
            self.load(path_to_model)

    @property
    def name(self) -> str:
        return self.display_name
    
    def pick_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.agent.get_action(obs_tensor)
        return action
    
    def load(self, path_to_model: str):
        state_dict = torch.load(path_to_model, map_location=self.device)
        self.agent.load_state_dict(state_dict)
