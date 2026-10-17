"""Random baseline agent sampling uniformly from legal actions."""

from __future__ import annotations

import random

import numpy as np

from ultimo_marl.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Baseline agent that uniformly samples among valid actions defined by action_mask."""

    def __init__(self, name: str = "RandomAgent", seed: int | None = None) -> None:
        super().__init__(name=name)
        self._rng = random.Random(seed)

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Sample uniformly from legal actions."""
        legal_actions = np.where(action_mask > 0)[0]
        if len(legal_actions) == 0:
            # Fallback to action 60 (DRAW) if mask was empty
            return 60

        chosen = self._rng.choice(legal_actions.tolist())
        return int(chosen)
