import numpy as np
from abc import ABC, abstractmethod

class Opponent(ABC):
    """
    Abstract base class for opponents in single training loop for UTTT
    """
    @abstractmethod
    def take_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """
        The core method that takes a board state and returns a move.

        Args:
            observation: The 7-channel 9x9 egocentric board state.
            action_mask: A flattened (81,) array of legal moves.

        Returns:
            The integer index (0-80) of the chosen move.
        """
        pass


    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the opponent type (e.g., 'RandomAgent', 'PPO-V1', 'Human)."""
        pass

        