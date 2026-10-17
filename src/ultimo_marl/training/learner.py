"""Double Dueling Deep Q-Network (DDQN) learner with Polyak updates and gradient clipping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from ultimo_marl.training.q_network import DuelingQNetwork, mask_q_values
from ultimo_marl.training.replay_buffer import TransitionBatch


class DDQNLearner:
    """Double Dueling Deep Q-Network learner.

    Implements:
    - Decoupled action selection (online network) and action evaluation (target network).
    - Polyak soft target network sync (tau parameter) or hard periodic sync.
    - Importance-weighted SmoothL1 (Huber) loss.
    - Gradient norm clipping to prevent gradient explosion (max_norm=1.0).

    Parameters
    ----------
    online_net : DuelingQNetwork
        Active policy network being updated via gradients.
    target_net : DuelingQNetwork | None
        Target network for stable temporal difference bootstrap. If None, clones online_net.
    lr : float
        Learning rate for AdamW optimizer (default: 1e-4).
    weight_decay : float
        Weight decay L2 regularization (default: 1e-5).
    gamma : float
        Discount factor for future rewards in [0.0, 1.0] (default: 0.99).
    tau : float
        Polyak update rate for soft target sync (default: 0.005).
    max_grad_norm : float
        Maximum gradient norm for gradient clipping (default: 1.0).
    device : torch.device | None
        Device on which computations and tensors are placed.
    """

    def __init__(
        self,
        online_net: DuelingQNetwork,
        target_net: DuelingQNetwork | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else next(online_net.parameters()).device
        self.online_net = online_net.to(self.device)

        if target_net is None:
            self.target_net = DuelingQNetwork(
                obs_dim=online_net.obs_dim,
                action_dim=online_net.action_dim,
                hidden_dim=online_net.hidden_dim,
            ).to(self.device)
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            self.target_net = target_net.to(self.device)

        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        self.optimizer = AdamW(
            self.online_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def sync_target_network(self, soft: bool = True) -> None:
        """Synchronize target network weights with online network.

        Parameters
        ----------
        soft : bool
            If True, performs Polyak soft update: theta_target = tau * theta_online + (1 - tau) * theta_target.
            If False, performs hard copy: theta_target = theta_online.
        """
        if soft:
            for target_param, online_param in zip(
                self.target_net.parameters(),
                self.online_net.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def compute_loss(
        self,
        batch: TransitionBatch,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Double-DQN Huber loss and transition TD-errors.

        Double DQN Algorithm:
        1. Online net predicts next greedy action:
           a* = argmax_a' [ Q_online(s', a') with action_mask ]
        2. Target net computes value of that action:
           Q_target(s', a*)
        3. Target value: y = r + (1 - done) * gamma * Q_target(s', a*)
        4. Current value: Q_online(s, a)
        5. Loss: mean(weights * Huber(y - Q_online(s, a)))

        Parameters
        ----------
        batch : TransitionBatch
            Batch of states, actions, rewards, next_states, dones, next_masks.
        weights : torch.Tensor
            Importance-sampling weights from PER buffer of shape (batch_size,).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - scalar_loss : Weighted Huber loss for backprop.
            - td_errors : Tensor of absolute TD errors (|y - Q(s, a)|) for PER priority updates.
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device).unsqueeze(-1)
        rewards = batch.rewards.to(self.device).unsqueeze(-1)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device).unsqueeze(-1)
        next_masks = batch.next_action_masks.to(self.device)
        is_weights = weights.to(self.device).unsqueeze(-1)

        # 1. Current Q(s, a) using online network
        current_q = self.online_net(states).gather(1, actions)

        # 2. Double-DQN target evaluation:
        with torch.no_grad():
            # Greedy action selection using online network + action masking
            next_q_online = self.online_net(next_states)
            masked_next_q_online = mask_q_values(next_q_online, next_masks)
            next_actions = masked_next_q_online.argmax(dim=-1, keepdim=True)

            # Action evaluation using target network
            next_q_target = self.target_net(next_states).gather(1, next_actions)

            # Target Q-value: y = r + (1 - done) * gamma * Q_target(s', a*)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        # 3. Absolute TD errors for PER update
        td_errors = (target_q - current_q).abs().detach()

        # 4. Importance-weighted Huber loss
        elementwise_loss = self.loss_fn(current_q, target_q)
        weighted_loss = (is_weights * elementwise_loss).mean()

        return weighted_loss, td_errors

    def train_step(
        self,
        batch: TransitionBatch,
        weights: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        """Perform a single DDQN gradient update step.

        Parameters
        ----------
        batch : TransitionBatch
            Batch of transitions sampled from PER.
        weights : torch.Tensor
            Importance-sampling weights from PER buffer.

        Returns
        -------
        tuple[float, np.ndarray]
            - loss_value : Scalar loss as Python float.
            - td_errors_np : Absolute TD errors as numpy array for PER priority update.
        """
        self.online_net.train()
        loss, td_errors = self.compute_loss(batch, weights)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norm clipping for numerical stability (Agent 09 audit requirement)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()

        # Soft Polyak target network update
        self.sync_target_network(soft=True)

        return float(loss.item()), td_errors.squeeze(-1).cpu().numpy()

    def save(self, filepath: str | Path) -> None:
        """Save learner checkpoint to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.online_net.obs_dim,
            "action_dim": self.online_net.action_dim,
            "hidden_dim": self.online_net.hidden_dim,
        }
        torch.save(checkpoint, path)

    def load(self, filepath: str | Path) -> None:
        """Load learner checkpoint from disk."""
        path = Path(filepath)
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
