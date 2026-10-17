"""Neural DQN policy agent conforming to the BaseAgent interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultimo_marl.agents.base import BaseAgent
from ultimo_marl.env.action_mapping import NUM_ACTIONS
from ultimo_marl.env.observation import OBSERVATION_DIM
from ultimo_marl.training.q_network import DuelingQNetwork


class DQNAgent(BaseAgent):
    """Deep Q-Network agent powered by DuelingQNetwork and action masking.

    Parameters
    ----------
    network : DuelingQNetwork | None
        Underlying PyTorch DuelingQNetwork. If None, initialized with default dimensions or loaded from checkpoint.
    checkpoint_path : str | Path | None
        Optional path to a trained PyTorch model checkpoint (.pt) to load.
    hidden_dim : int
        Hidden dimension when creating network from scratch (default: 256).
    epsilon : float
        Exploration rate in [0.0, 1.0] (default: 0.0 for pure greedy exploitation).
    name : str
        Agent identifier string.
    device : torch.device | None
        Target computing device for neural inference.
    """

    def __init__(
        self,
        network: DuelingQNetwork | None = None,
        checkpoint_path: str | Path | None = None,
        hidden_dim: int = 256,
        epsilon: float = 0.0,
        name: str = "DQNAgent",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(name=name)
        self.device = device if device is not None else torch.device("cpu")
        self.epsilon = epsilon

        if network is not None:
            self.network = network.to(self.device)
            if checkpoint_path is not None:
                self.load_checkpoint(checkpoint_path)
        elif checkpoint_path is not None:
            self.network = self._load_from_checkpoint(checkpoint_path)
        else:
            self.network = DuelingQNetwork(hidden_dim=hidden_dim).to(self.device)

        self.network.eval()

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Select discrete action using the neural policy under Fog of War."""
        return self.network.act(obs, action_mask, epsilon=self.epsilon)

    def _load_from_checkpoint(self, checkpoint_path: str | Path) -> DuelingQNetwork:
        """Instantiate DuelingQNetwork matching checkpoint dimensions and load weights."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")

        data = torch.load(path, map_location=self.device)
        state_dict: dict[str, Any]
        if isinstance(data, dict) and "online_net" in data:
            state_dict = data["online_net"]
        elif isinstance(data, dict) and "state_dict" in data:
            state_dict = data["state_dict"]
        elif isinstance(data, dict):
            state_dict = data
        else:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        # Automatically deduce dimensions from parameter shapes
        if "feature_trunk.0.weight" in state_dict:
            hidden_dim = int(state_dict["feature_trunk.0.weight"].shape[0])
            obs_dim = int(state_dict["feature_trunk.0.weight"].shape[1])
        else:
            hidden_dim = data.get("hidden_dim", 256) if isinstance(data, dict) else 256
            obs_dim = (
                data.get("obs_dim", OBSERVATION_DIM) if isinstance(data, dict) else OBSERVATION_DIM
            )

        if "advantage_stream.2.weight" in state_dict:
            action_dim = int(state_dict["advantage_stream.2.weight"].shape[0])
        else:
            action_dim = (
                data.get("action_dim", NUM_ACTIONS) if isinstance(data, dict) else NUM_ACTIONS
            )

        net = DuelingQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        net.load_state_dict(state_dict)
        return net

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load neural network weights into existing network from disk checkpoint."""
        self.network = self._load_from_checkpoint(checkpoint_path)
        self.network.eval()
