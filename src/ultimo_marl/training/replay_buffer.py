"""Prioritized Experience Replay (PER) buffer with binary SumTree."""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np
import torch

from ultimo_marl.env.action_mapping import NUM_ACTIONS
from ultimo_marl.env.observation import OBSERVATION_DIM


class TransitionBatch(NamedTuple):
    """Batched PyTorch tensors for neural DDQN parameter updates."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    next_action_masks: torch.Tensor


class SumTree:
    """Binary SumTree data structure for O(log N) prioritized sampling and priority updates.

    Parameters
    ----------
    capacity : int
        Maximum number of leaf nodes (transitions) to store.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Total nodes in a complete binary tree with `capacity` leaves is 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        # Circular pointer for adding new leaves
        self.write_idx = 0
        self.size = 0

    @property
    def total_priority(self) -> float:
        """Root node value representing the total sum of all leaf priorities."""
        return float(self.tree[0])

    def update(self, tree_idx: int, priority: float) -> None:
        """Update leaf node priority and propagate the delta up to the root in O(log N).

        Parameters
        ----------
        tree_idx : int
            Absolute node index in tree array.
        priority : float
            New positive priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change upward to the root
        idx = tree_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, priority: float) -> int:
        """Add new priority at current circular leaf index.

        Returns
        -------
        int
            Absolute tree index of the newly updated leaf node.
        """
        tree_idx = self.write_idx + self.capacity - 1
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

        return tree_idx

    def get(self, value: float) -> tuple[int, float, int]:
        """Search the tree down to the leaf node corresponding to cumulative priority `value`.

        Parameters
        ----------
        value : float
            Cumulative priority search scalar in [0.0, total_priority].

        Returns
        -------
        tuple[int, float, int]
            (tree_idx, leaf_priority, data_idx)
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1

            # Leaf reached
            if left >= len(self.tree):
                leaf_idx = parent
                break

            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                if right >= len(self.tree) or self.tree[right] == 0.0:
                    leaf_idx = left
                    break
                parent = right

        data_idx = leaf_idx - self.capacity + 1
        # Defensive clamp to valid data indices
        data_idx = max(0, min(data_idx, self.capacity - 1))
        return leaf_idx, float(self.tree[leaf_idx]), data_idx


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer (Schaul et al., 2015).

    Samples transitions proportionally to their TD-error priorities:
        P(i) = p_i^alpha / sum_k(p_k^alpha)

    Corrects for sampling bias via importance-sampling weights:
        w_i = (N * P(i))^(-beta) / max_k(w_k)

    Parameters
    ----------
    capacity : int
        Maximum transitions stored in circular buffer (default: 100,000).
    obs_dim : int
        Observation dimension (default: 78).
    action_dim : int
        Action dimension (default: 61).
    alpha : float
        Priority exponent determining degree of prioritization (0 = uniform, 1 = full PER).
    beta_start : float
        Initial importance-sampling exponent (default: 0.4).
    beta_end : float
        Final annealed importance-sampling exponent (default: 1.0).
    beta_frames : int
        Number of steps to anneal beta from beta_start to beta_end.
    epsilon : float
        Small positive constant added to TD-errors to prevent 0 priority.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = NUM_ACTIONS,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.beta_step = (beta_end - beta_start) / max(beta_frames, 1)
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        # Preallocated contiguous numpy arrays for zero-copy memory efficiency
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, action_dim), dtype=np.int8)

    def __len__(self) -> int:
        return self.tree.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        """Push a new transition tuple into the prioritized circular buffer."""
        idx = self.tree.write_idx

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0
        self.next_action_masks[idx] = next_action_mask

        # Assign maximal priority to new experiences so they are sampled at least once
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority)

    def sample(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[TransitionBatch, np.ndarray, torch.Tensor]:
        """Sample a batch of transitions using stratified priority intervals.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        device : torch.device, optional
            Target PyTorch device for returned tensors.

        Returns
        -------
        tuple[TransitionBatch, np.ndarray, torch.Tensor]
            - batch : TransitionBatch containing states, actions, rewards, next_states, dones, next_masks.
            - tree_indices : Array of tree node indices to update after computing TD errors.
            - is_weights : Importance-sampling weights tensor of shape (batch_size,).
        """
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty PrioritizedReplayBuffer.")

        total_p = self.tree.total_priority
        segment = total_p / batch_size

        tree_indices = np.empty(batch_size, dtype=np.int32)
        data_indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            val = random.uniform(low, high)
            tree_idx, priority, data_idx = self.tree.get(val)

            # Defensive fallback if uninitialized leaf sampled
            if priority <= 0.0 or data_idx >= len(self):
                data_idx = random.randint(0, len(self) - 1)
                tree_idx = data_idx + self.capacity - 1
                priority = max(self.tree.tree[tree_idx], self.epsilon)

            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i] = priority

        # Compute Importance-Sampling (IS) weights
        probs = priorities / total_p
        # w_i = (N * P(i))^(-beta)
        weights = (len(self) * probs) ** (-self.beta)
        # Normalize by max weight for stability
        weights /= np.max(weights)

        # Anneal beta towards beta_end
        self.beta = min(self.beta_end, self.beta + self.beta_step)

        # Build PyTorch batch tensors
        dev = device if device is not None else torch.device("cpu")
        batch = TransitionBatch(
            states=torch.from_numpy(self.states[data_indices]).to(dev),
            actions=torch.from_numpy(self.actions[data_indices]).to(dev),
            rewards=torch.from_numpy(self.rewards[data_indices]).to(dev),
            next_states=torch.from_numpy(self.next_states[data_indices]).to(dev),
            dones=torch.from_numpy(self.dones[data_indices]).to(dev),
            next_action_masks=torch.from_numpy(self.next_action_masks[data_indices]).to(dev),
        )
        is_weights_tensor = torch.from_numpy(weights.astype(np.float32)).to(dev)

        return batch, tree_indices, is_weights_tensor

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray | torch.Tensor,
    ) -> None:
        """Update priorities in SumTree using absolute TD errors.

        Parameters
        ----------
        tree_indices : np.ndarray
            Tree indices returned from `sample()`.
        td_errors : np.ndarray | torch.Tensor
            Absolute or raw TD-errors (|Q_target - Q_curr|).
        """
        if isinstance(td_errors, torch.Tensor):
            errors = td_errors.detach().cpu().abs().numpy().flatten()
        else:
            errors = np.abs(td_errors).flatten()

        for tree_idx, error in zip(tree_indices, errors, strict=False):
            clamped_error = float(error) + self.epsilon
            self.max_priority = max(self.max_priority, clamped_error)
            priority = clamped_error**self.alpha
            self.tree.update(int(tree_idx), priority)
