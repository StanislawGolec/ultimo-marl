"""Abstract base interface for Ultimo agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all Ultimo agents (Random, Heuristic, RL, etc.)."""

    def __init__(self, name: str = "BaseAgent") -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Agent display name."""
        return self._name

    @abstractmethod
    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Select a discrete action given the observation vector and legal action mask.

        Parameters
        ----------
        obs : np.ndarray
            Normalized float32 observation vector under Fog of War.
        action_mask : np.ndarray
            Boolean or int8 array of shape (61,) where 1/True represents a legal move.

        Returns
        -------
        int
            Selected action ID in [0, 60]. Must satisfy action_mask[action] > 0.
        """
        ...

    def reset(self) -> None:
        """Reset internal agent memory / recurrent states between matches."""
        # Optional lifecycle hook; stateless agents do nothing.
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"
