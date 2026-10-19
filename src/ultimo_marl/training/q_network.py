"""Dueling Q-Network architecture with action masking for Ultimo MARL."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from ultimo_marl.env.action_mapping import NUM_ACTIONS
from ultimo_marl.env.observation import OBSERVATION_DIM

if TYPE_CHECKING:
    pass

MASK_PENALTY = -1e9


def mask_q_values(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
    penalty_value: float = MASK_PENALTY,
) -> torch.Tensor:
    """Apply action mask to Q-values, assigning large negative values to illegal actions.

    Parameters
    ----------
    q_values : torch.Tensor
        Raw Q-values tensor of shape (..., action_dim).
    action_mask : torch.Tensor
        Binary action mask tensor of shape (..., action_dim) where 1 indicates legal,
        and 0 indicates illegal move.
    penalty_value : float
        Negative constant assigned to illegal actions (default: -1e9).

    Returns
    -------
    torch.Tensor
        Masked Q-values tensor of shape (..., action_dim).
    """
    mask_bool = action_mask.bool()
    penalty_tensor = torch.full_like(q_values, penalty_value)
    return torch.where(mask_bool, q_values, penalty_tensor)


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network (Wang et al., 2016) with action masking.

    Decomposes state-action value Q(s, a) into state value V(s) and advantage A(s, a):
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality (default: OBSERVATION_DIM = 90).
    action_dim : int
        Discrete action space dimensionality (default: NUM_ACTIONS = 61).
    hidden_dim : int
        Number of neurons in hidden layers (default: 256).
    """

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared feature trunk: 2-layer MLP with LayerNorm and ReLU
        self.feature_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # State Value stream V(s): outputs scalar (batch_size, 1)
        sub_hidden = max(hidden_dim // 2, 64)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, sub_hidden),
            nn.ReLU(),
            nn.Linear(sub_hidden, 1),
        )

        # Action Advantage stream A(s, a): outputs (batch_size, action_dim)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, sub_hidden),
            nn.ReLU(),
            nn.Linear(sub_hidden, action_dim),
        )

    def adapt_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Adapt legacy 78-dimensional observations to 90-dimensional trunk input (or vice versa)."""
        last_dim = obs.shape[-1]
        if last_dim == self.obs_dim:
            return obs

        if last_dim == 78 and self.obs_dim == 90:
            # 78-dim layout: [0:72] top+hand, [72:75] 3 opponents, [75:78] 3 dynamics
            # 90-dim layout: [0:72] top+hand, [72:87] 15 opponents, [87:90] 3 dynamics
            pad_shape = (*obs.shape[:-1], 12)
            opp_padding = torch.zeros(pad_shape, dtype=obs.dtype, device=obs.device)
            return torch.cat([obs[..., :75], opp_padding, obs[..., 75:78]], dim=-1)

        if last_dim == 90 and self.obs_dim == 78:
            return torch.cat([obs[..., :75], obs[..., 87:90]], dim=-1)

        if last_dim < self.obs_dim:
            return nn.functional.pad(obs, (0, self.obs_dim - last_dim), value=0.0)

        return obs[..., : self.obs_dim]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute unmasked Q-values for all actions.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor of shape (batch_size, obs_dim) or (obs_dim,).

        Returns
        -------
        torch.Tensor
            Q-values tensor of shape (batch_size, action_dim).
        """
        adapted_obs = self.adapt_observation(obs)
        features: torch.Tensor = self.feature_trunk(adapted_obs)
        values: torch.Tensor = self.value_stream(features)
        advantages: torch.Tensor = self.advantage_stream(features)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_vals: torch.Tensor = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_vals

    def evaluate_streams(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state value V(s) and action advantage A(s, a) streams separately.

        Useful for telemetry and explainability inspection.
        """
        adapted_obs = self.adapt_observation(obs)
        features: torch.Tensor = self.feature_trunk(adapted_obs)
        values: torch.Tensor = self.value_stream(features)
        advantages: torch.Tensor = self.advantage_stream(features)
        return values, advantages

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """Select action using epsilon-greedy policy with strict action masking.

        Parameters
        ----------
        obs : np.ndarray | torch.Tensor
            Observation vector of shape (obs_dim,).
        action_mask : np.ndarray | torch.Tensor
            Binary legal action mask of shape (action_dim,).
        epsilon : float
            Exploration probability in [0.0, 1.0]. When exploring, samples uniformly
            among strictly legal actions.

        Returns
        -------
        int
            Selected legal action ID in [0, action_dim - 1].
        """
        if isinstance(action_mask, torch.Tensor):
            mask_np = action_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = action_mask.astype(bool)

        legal_actions = np.where(mask_np)[0]
        if len(legal_actions) == 0:
            # Fallback if somehow no actions are legal
            return 0

        # Epsilon-greedy exploration: sample uniformly among legal actions
        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice(legal_actions))

        # Greedy exploitation: argmax over masked Q-values
        device = next(self.parameters()).device
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        else:
            obs_tensor = obs.float().unsqueeze(0).to(device) if obs.dim() == 1 else obs.to(device)

        if isinstance(action_mask, np.ndarray):
            mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(device)
        else:
            mask_tensor = (
                action_mask.unsqueeze(0).to(device)
                if action_mask.dim() == 1
                else action_mask.to(device)
            )

        q_vals = self.forward(obs_tensor)
        masked_q = mask_q_values(q_vals, mask_tensor)
        selected = int(torch.argmax(masked_q, dim=-1).item())

        # Defensive sanity assertion: chosen action MUST be legal
        if not mask_np[selected]:
            return int(random.choice(legal_actions))

        return selected
