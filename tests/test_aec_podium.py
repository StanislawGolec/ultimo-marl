"""Tests for Last-Man-Standing PettingZoo AEC loop and multi-tier podium rewards."""

from __future__ import annotations

import random

import numpy as np

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.env.action_mapping import ACTION_DRAW, card_to_action_id
from ultimo_marl.env.ultimo_aec_env import UltimoAECEnv, get_podium_reward


def test_podium_reward_formulation() -> None:
    """Verify exact multi-tier podium values across 2, 4, and 16 players."""
    # 4-player standard table
    assert get_podium_reward(1, 4) == 1.0
    assert get_podium_reward(2, 4) == 0.5
    assert get_podium_reward(3, 4) == 0.2
    assert get_podium_reward(4, 4) == -1.0

    # 16-player large table
    assert get_podium_reward(1, 16) == 1.0
    assert get_podium_reward(2, 16) == 0.5
    assert get_podium_reward(3, 16) == 0.2
    for mid_rank in range(4, 16):
        assert get_podium_reward(mid_rank, 16) == 0.0
    assert get_podium_reward(16, 16) == -1.0

    # 2-player duel
    assert get_podium_reward(1, 2) == 1.0
    assert get_podium_reward(2, 2) == -1.0


def test_aec_podium_four_player_hundred_games() -> None:
    """Run 100 4-player games to completion; verify all games assign ranks 1..4 without deadlock."""
    env = UltimoAECEnv(num_players=4, max_turns=600)
    rng = random.Random(42)

    for game_idx in range(100):
        env.reset(seed=1000 + game_idx)

        step_count = 0
        while env.agents and step_count < 2500:
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                env.step(None)
            else:
                mask = info["action_mask"]
                card_actions = np.where(mask[:60])[0]
                if len(card_actions) > 0:
                    action = int(rng.choice(card_actions))
                else:
                    action = ACTION_DRAW

                env.step(action)

            step_count += 1

        # Assert no deadlock: match finished cleanly
        assert not env.agents, f"Game {game_idx} failed to terminate within step limit"
        assert len(env.placements) == 4, f"Game {game_idx} did not place all 4 agents"

        # Assert exact ranks 1..4 assigned
        assigned_ranks = set(env.placements.values())
        assert assigned_ranks == {1, 2, 3, 4}, (
            f"Game {game_idx} assigned invalid rank set: {assigned_ranks}"
        )


def test_aec_podium_two_player_duel() -> None:
    """Verify 2-player duel finishes with ranks 1 and 2."""
    env = UltimoAECEnv(num_players=2, max_turns=400)
    rng = random.Random(77)

    env.reset(seed=77)
    while env.agents:
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
        else:
            mask = info["action_mask"]
            card_actions = np.where(mask[:60])[0]
            if len(card_actions) > 0:
                action = int(rng.choice(card_actions))
            else:
                action = ACTION_DRAW
            env.step(action)

    assert set(env.placements.values()) == {1, 2}


def test_aec_podium_sixteen_player_finish() -> None:
    """Verify 16-player match finishes with all 16 distinct ranks (1..16)."""
    env = UltimoAECEnv(num_players=16, max_turns=2000)
    rng = random.Random(999)

    env.reset(seed=999)
    while env.agents:
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
        else:
            mask = info["action_mask"]
            card_actions = np.where(mask[:60])[0]
            if len(card_actions) > 0:
                action = int(rng.choice(card_actions))
            else:
                action = ACTION_DRAW
            env.step(action)

    assert len(env.placements) == 16
    assert set(env.placements.values()) == set(range(1, 17))


def test_adversarial_rewards_deflection_and_stacking() -> None:
    """Test adversarial defense bonuses (+0.2 for Skip deflection and Draw stacking)."""
    env = UltimoAECEnv(num_players=4, max_turns=100)
    env.reset(seed=42)

    # Force player 0 to face a Skip challenge with 2 cards (so hand isn't emptied)
    env._sim.pending_skip_deflection = True
    p0 = env._sim.get_player(0)
    p0.hand.clear()
    skip_card = Card(CardColor.RED, CardValue.SKIP)
    p0.hand.add_card(skip_card)
    p0.hand.add_card(Card(CardColor.RED, CardValue.FIVE))

    action_id = card_to_action_id(skip_card)
    env.step(action_id)

    # Assert Player 0 received +0.2 deflection reward (-0.01 step cost = +0.19)
    assert np.isclose(env._cumulative_rewards["player_0"], 0.19)


def test_adversarial_rewards_catch_ultimo() -> None:
    """Test +0.3 catch bonus and -0.5 penalty for failing to declare ULTIMO."""
    env = UltimoAECEnv(num_players=4, max_turns=100)
    env.reset(seed=42)

    # Set player 1 as a pending violator (holding 1 card, did not declare Ultimo)
    p1 = env._sim.get_player(1)
    p1.hand.clear()
    p1.hand.add_card(Card(CardColor.BLUE, CardValue.NINE))
    env._sim._pending_ultimo_violator = 1

    # Player 0 takes turn by drawing card, catching player 1
    env.step(ACTION_DRAW)

    # Player 0 should gain +0.3 catch bonus (minus 0.01 step cost = 0.29)
    assert np.isclose(env._cumulative_rewards["player_0"], 0.29)
    # Player 1 should receive -0.5 penalty and 6 cards
    assert env.rewards["player_1"] == -0.5
    assert p1.card_count == 7
