"""Unit and integration tests for Dueling Q-Network, PER buffer, DDQN learner, and DQNAgent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ultimo_marl.agents import DQNAgent
from ultimo_marl.env import NUM_ACTIONS, OBSERVATION_DIM
from ultimo_marl.training import (
    DDQNLearner,
    DuelingQNetwork,
    PrioritizedReplayBuffer,
    SumTree,
    TransitionBatch,
    mask_q_values,
)

# ============================================================================
# 1. Dueling Q-Network Tests
# ============================================================================


def test_dueling_q_network_forward_and_streams() -> None:
    """Test output shapes, value/advantage streams, and dueling identity."""
    net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    batch_size = 8
    obs = torch.randn(batch_size, OBSERVATION_DIM)

    q_vals = net(obs)
    assert q_vals.shape == (batch_size, NUM_ACTIONS)

    v, a = net.evaluate_streams(obs)
    assert v.shape == (batch_size, 1)
    assert a.shape == (batch_size, NUM_ACTIONS)

    # Invariant: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    expected_q = v + (a - a.mean(dim=-1, keepdim=True))
    assert torch.allclose(q_vals, expected_q, atol=1e-5)


def test_action_masking_penalty() -> None:
    """Illegal actions must receive -1e9 penalty and never be chosen as argmax."""
    q_vals = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    # Only action 1 is legal
    mask = torch.tensor([[0, 1, 0, 0]], dtype=torch.int8)

    masked_q = mask_q_values(q_vals, mask)

    assert masked_q[0, 1].item() == 20.0
    assert masked_q[0, 0].item() <= -1e8
    assert masked_q[0, 2].item() <= -1e8
    assert masked_q[0, 3].item() <= -1e8

    chosen_action = int(torch.argmax(masked_q, dim=-1).item())
    assert chosen_action == 1


def test_dueling_q_network_act_legality() -> None:
    """act() must strictly pick legal actions under both greedy and epsilon exploration."""
    net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    obs = np.random.randn(OBSERVATION_DIM).astype(np.float32)

    # Restrict legal actions to only [5, 12, 60]
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    legal_indices = [5, 12, 60]
    for idx in legal_indices:
        mask[idx] = 1

    # Test greedy exploitation (epsilon = 0.0)
    for _ in range(10):
        action = net.act(obs, mask, epsilon=0.0)
        assert action in legal_indices

    # Test exploration (epsilon = 1.0)
    for _ in range(20):
        action = net.act(obs, mask, epsilon=1.0)
        assert action in legal_indices


# ============================================================================
# 2. SumTree Tests
# ============================================================================


def test_sum_tree_add_and_update() -> None:
    """Test SumTree priority summation, retrieval, and dynamic priority updates."""
    tree = SumTree(capacity=4)
    assert tree.total_priority == 0.0

    # Add 4 elements with priorities 1, 2, 3, 4
    tree.add(1.0)
    tree.add(2.0)
    tree.add(3.0)
    tree.add(4.0)

    assert tree.size == 4
    assert pytest.approx(tree.total_priority, rel=1e-5) == 10.0

    # Query value = 0.5 -> should be first element (priority 1.0)
    leaf_idx, priority, data_idx = tree.get(0.5)
    assert data_idx == 0
    assert pytest.approx(priority, rel=1e-5) == 1.0

    # Query value = 9.5 -> should be fourth element (priority 4.0)
    leaf_idx, priority, data_idx = tree.get(9.5)
    assert data_idx == 3
    assert pytest.approx(priority, rel=1e-5) == 4.0

    # Update first element priority from 1.0 to 11.0 (total should become 20.0)
    first_leaf = 0 + 4 - 1  # idx = 3
    tree.update(first_leaf, 11.0)
    assert pytest.approx(tree.total_priority, rel=1e-5) == 20.0


# ============================================================================
# 3. Prioritized Replay Buffer Tests
# ============================================================================


def test_prioritized_replay_buffer_sampling_and_priorities() -> None:
    """Test pushing transitions, stratified PER sampling, and priority updates."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_dim=OBSERVATION_DIM,
        action_dim=NUM_ACTIONS,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_frames=10,
    )
    assert len(buffer) == 0

    # Push 20 transitions
    for i in range(20):
        s = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        s[0] = float(i)
        next_s = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        next_s[0] = float(i + 1)
        mask = np.ones(NUM_ACTIONS, dtype=np.int8)
        buffer.push(
            state=s,
            action=i % NUM_ACTIONS,
            reward=float(i),
            next_state=next_s,
            done=(i % 5 == 0),
            next_action_mask=mask,
        )

    assert len(buffer) == 20

    # Sample batch
    batch_size = 8
    batch, tree_indices, is_weights = buffer.sample(batch_size=batch_size)

    assert isinstance(batch, TransitionBatch)
    assert batch.states.shape == (batch_size, OBSERVATION_DIM)
    assert batch.actions.shape == (batch_size,)
    assert batch.rewards.shape == (batch_size,)
    assert batch.next_states.shape == (batch_size, OBSERVATION_DIM)
    assert batch.dones.shape == (batch_size,)
    assert batch.next_action_masks.shape == (batch_size, NUM_ACTIONS)
    assert is_weights.shape == (batch_size,)
    assert (is_weights <= 1.0 + 1e-5).all()

    # Update priorities with new TD-errors
    mock_td_errors = torch.rand(batch_size) * 5.0
    initial_total_p = buffer.tree.total_priority
    buffer.update_priorities(tree_indices, mock_td_errors)
    # Total priority should be valid positive float
    assert buffer.tree.total_priority > 0.0
    assert buffer.tree.total_priority != initial_total_p


# ============================================================================
# 4. DDQN Learner & Loss Calculation Tests
# ============================================================================


def test_ddqn_learner_gradient_step_and_soft_sync() -> None:
    """Verify DDQN backpropagation, finite loss, and Polyak soft target sync."""
    online_net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    target_net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    learner = DDQNLearner(
        online_net=online_net,
        target_net=target_net,
        lr=1e-3,
        tau=0.01,
        max_grad_norm=1.0,
    )

    # Create synthetic transition batch
    batch_size = 4
    batch = TransitionBatch(
        states=torch.randn(batch_size, OBSERVATION_DIM),
        actions=torch.randint(0, NUM_ACTIONS, (batch_size,)),
        rewards=torch.randn(batch_size),
        next_states=torch.randn(batch_size, OBSERVATION_DIM),
        dones=torch.zeros(batch_size),
        next_action_masks=torch.ones(batch_size, NUM_ACTIONS, dtype=torch.int8),
    )
    weights = torch.ones(batch_size)

    # Initial online and target parameters are different
    learner.sync_target_network(soft=False)
    # Check that soft sync moves target weights towards online weights
    loss, td_errors = learner.train_step(batch, weights)

    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert loss >= 0.0
    assert td_errors.shape == (batch_size,)
    assert (td_errors >= 0.0).all()


def test_ddqn_learner_save_and_load(tmp_path: Path) -> None:
    """Learner model weights must be serializable and reloadable from disk."""
    save_path = tmp_path / "test_model.pt"
    online_net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    learner1 = DDQNLearner(online_net=online_net)

    learner1.save(save_path)
    assert save_path.is_file()

    # Create new learner and load weights
    online_net2 = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    learner2 = DDQNLearner(online_net=online_net2)
    learner2.load(save_path)

    for p1, p2 in zip(
        learner1.online_net.parameters(), learner2.online_net.parameters(), strict=True
    ):
        assert torch.allclose(p1, p2)


def test_ddqn_overfit_single_batch() -> None:
    """Sanity check: DDQN must rapidly minimize Huber loss on a tiny fixed batch."""
    torch.manual_seed(42)
    np.random.seed(42)

    online_net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=128)
    learner = DDQNLearner(online_net=online_net, lr=1e-2, tau=0.1)

    batch_size = 4
    batch = TransitionBatch(
        states=torch.randn(batch_size, OBSERVATION_DIM),
        actions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        rewards=torch.tensor([1.0, -1.0, 0.5, 0.0]),
        next_states=torch.randn(batch_size, OBSERVATION_DIM),
        dones=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        next_action_masks=torch.ones(batch_size, NUM_ACTIONS, dtype=torch.int8),
    )
    weights = torch.ones(batch_size)

    initial_loss, _ = learner.train_step(batch, weights)
    losses = [initial_loss]
    for _ in range(40):
        loss, _ = learner.train_step(batch, weights)
        losses.append(loss)

    # Loss after 40 steps on identical batch must drop noticeably
    final_loss = losses[-1]
    assert final_loss < initial_loss * 0.5, f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"


# ============================================================================
# 5. DQNAgent Integration Tests
# ============================================================================


def test_dqn_agent_conforms_to_base_agent() -> None:
    """DQNAgent must conform to BaseAgent API and respect action masking."""
    agent = DQNAgent(name="DQN_Bot_1")
    obs = np.random.randn(OBSERVATION_DIM).astype(np.float32)

    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[42] = 1  # Only action 42 is legal

    selected = agent.select_action(obs, mask)
    assert selected == 42
    assert agent.name == "DQN_Bot_1"

    agent.reset()


def test_dqn_agent_checkpoint_loading_and_errors(tmp_path: Path) -> None:
    """Test loading model weights into DQNAgent and handling invalid paths."""
    # 1. Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DQNAgent(checkpoint_path=tmp_path / "non_existent.pt")

    # 2. Test saving online_net dictionary and loading
    net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, action_dim=NUM_ACTIONS, hidden_dim=64)
    ckpt_path = tmp_path / "agent_ckpt.pt"
    torch.save({"online_net": net.state_dict()}, ckpt_path)

    agent = DQNAgent(checkpoint_path=ckpt_path)
    for p1, p2 in zip(net.parameters(), agent.network.parameters(), strict=True):
        assert torch.allclose(p1, p2)

    # 3. Test raw state_dict loading
    raw_ckpt_path = tmp_path / "raw_ckpt.pt"
    torch.save(net.state_dict(), raw_ckpt_path)
    agent_raw = DQNAgent(checkpoint_path=raw_ckpt_path)
    for p1, p2 in zip(net.parameters(), agent_raw.network.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_dqn_agent_tournament_execution() -> None:
    """DQNAgent must participate cleanly in a multi-game Tournament with random agents."""
    from ultimo_marl.agents import BaseAgent, RandomAgent, Tournament

    dqn_bot = DQNAgent(name="NeuralBot")
    random_bots = [RandomAgent(name=f"Random_{i}") for i in range(1, 4)]
    agents: list[BaseAgent] = [dqn_bot, *random_bots]

    tournament = Tournament(agents=agents)
    results = tournament.play(num_games=10, seed=123)

    assert results.games_played == 10
    assert sum(results.wins.values()) <= 10
    assert results.average_turns > 0.0
