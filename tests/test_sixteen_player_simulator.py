"""Unit, invariant, and performance tests for 16-player multi-deck simulation and 90-dim observations."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from ultimo_marl.domain import (
    Card,
    CardColor,
    CardValue,
    Deck,
    Direction,
    UltimoSimulator,
    get_playable_cards,
)
from ultimo_marl.env import (
    ACTION_DRAW,
    NUM_ACTIONS,
    OBSERVATION_DIM,
    UltimoAECEnv,
    encode_observation,
)
from ultimo_marl.training import DuelingQNetwork


def test_multi_deck_creation_and_card_counts() -> None:
    """Verify Deck.create_standard scales card counts by 108 * num_decks."""
    deck_single = Deck.create_standard(num_decks=1)
    assert len(deck_single) == 108
    assert deck_single.num_decks == 1

    deck_double = Deck.create_standard(num_decks=2)
    assert len(deck_double) == 216
    assert deck_double.num_decks == 2

    cards: list[Card] = []
    while not deck_double.is_empty:
        card = deck_double.draw()
        assert card is not None
        cards.append(card)

    assert len(cards) == 216
    assert sum(1 for c in cards if c.value == CardValue.WILD) == 8
    assert sum(1 for c in cards if c.value == CardValue.WILD_DRAW_FOUR) == 8

    for color in CardColor.standard_colors():
        color_cards = [c for c in cards if c.color == color]
        assert len(color_cards) == 50
        assert sum(1 for c in color_cards if c.value == CardValue.ZERO) == 2
        assert sum(1 for c in color_cards if c.value == CardValue.SKIP) == 4
        assert sum(1 for c in color_cards if c.value == CardValue.REVERSE) == 4
        assert sum(1 for c in color_cards if c.value == CardValue.DRAW_TWO) == 4

    with pytest.raises(ValueError, match="num_decks must be >= 1"):
        Deck.create_standard(num_decks=0)


@pytest.mark.parametrize(
    ("num_players", "expected_decks", "expected_total_cards"),
    [
        (2, 1, 108),
        (4, 1, 108),
        (8, 1, 108),
        (9, 2, 216),
        (12, 2, 216),
        (16, 2, 216),
    ],
)
def test_simulator_player_scaling_and_deck_initialization(
    num_players: int,
    expected_decks: int,
    expected_total_cards: int,
) -> None:
    """Verify automatic 2-deck initialization when num_players > 8."""
    sim = UltimoSimulator(num_players=num_players, seed=42)
    sim.reset()

    assert sim.num_players == num_players
    assert sim.num_decks == expected_decks
    assert sim.total_cards == expected_total_cards
    sim.assert_card_conservation()

    for idx in range(num_players):
        assert sim.get_player(idx).card_count == 7

    assert sim.discard_pile_size == 1
    assert sim.deck_size == expected_total_cards - (7 * num_players) - 1


def test_simulator_invalid_player_bounds() -> None:
    """Verify simulator rejects player counts outside [2, 16]."""
    with pytest.raises(ValueError, match="Player count must be between 2 and 16"):
        UltimoSimulator(num_players=1)

    with pytest.raises(ValueError, match="Player count must be between 2 and 16"):
        UltimoSimulator(num_players=17)


def test_sixteen_player_card_conservation_across_turns() -> None:
    """Verify total cards == 216 at every turn across 16-player simulated matches."""
    sim = UltimoSimulator(num_players=16, seed=101)

    total_steps = 0
    for game_seed in range(10):
        sim.reset(seed=100 + game_seed)
        assert sim.total_cards == 216
        sim.assert_card_conservation()

        steps = 0
        while not sim.is_game_over and steps < 300:
            curr = sim.current_player_idx
            playable = get_playable_cards(
                sim.get_player(curr).hand,
                sim.top_card,
                sim.active_color,
                pending_skip_deflection=sim.pending_skip_deflection,
                pending_draw_penalty=sim.pending_draw_penalty,
            )
            if playable:
                chosen = playable[0]
                wild_color = CardColor.BLUE if chosen.is_wild else None
                res = sim.play_cards(curr, [chosen], chosen_wild_color=wild_color)
                assert res.success is True
            else:
                sim.draw_card(curr)
                sim.pass_turn(curr)

            assert sim.total_cards == 216
            sim.assert_card_conservation()
            steps += 1
            total_steps += 1

    assert total_steps >= 500


def test_90_dim_observation_vector_16_players() -> None:
    """Verify 90-dimensional observation vector structure and normalization for 16 players."""
    assert OBSERVATION_DIM == 90

    sim = UltimoSimulator(num_players=16, seed=77)
    sim.reset()

    obs = encode_observation(sim, observing_player_idx=0)
    assert obs.shape == (90,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)

    # 1. Top card features (0..18)
    assert np.sum(obs[0:4]) == 1.0
    assert np.sum(obs[4:17]) == 1.0
    assert obs[17] == 0.0

    # 2. Hand histogram (18..72)
    assert np.sum(obs[18:72]) > 0.0

    # 3. Opponent hand counts (72..87): all 15 opponents have 7 cards -> 0.7
    for idx in range(72, 87):
        assert np.isclose(obs[idx], 0.7)

    # 4. Game and defense dynamics (87..90)
    assert obs[87] == 1.0  # Clockwise
    assert obs[88] == 0.0  # pending_skip_deflection
    assert obs[89] == 0.0  # pending_draw_penalty

    # Verify defense dynamics encoding
    sim._turn_manager.reverse_direction()
    sim.pending_skip_deflection = True
    sim.pending_draw_penalty = 8
    obs_def = encode_observation(sim, observing_player_idx=0)
    assert obs_def[87] == 0.0  # Counter-clockwise
    assert obs_def[88] == 1.0  # Skip deflection active
    assert np.isclose(obs_def[89], 8.0 / 16.0)


def test_dueling_q_network_90_dim_and_78_dim_adapter() -> None:
    """Verify DuelingQNetwork defaults to obs_dim=90 and adapts legacy 78-dim inputs."""
    net = DuelingQNetwork(hidden_dim=64)
    assert net.obs_dim == 90
    assert net.action_dim == NUM_ACTIONS

    # 90-dim batch forward
    obs_90 = torch.rand(4, 90)
    q_90 = net(obs_90)
    assert q_90.shape == (4, NUM_ACTIONS)

    # 78-dim legacy batch forward via backward-compatibility adapter
    obs_78 = torch.rand(4, 78)
    q_78 = net(obs_78)
    assert q_78.shape == (4, NUM_ACTIONS)

    v_78, a_78 = net.evaluate_streams(obs_78)
    assert v_78.shape == (4, 1)
    assert a_78.shape == (4, NUM_ACTIONS)

    # 78-dim single observation in act()
    single_78 = np.zeros(78, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[ACTION_DRAW] = 1
    chosen = net.act(single_78, mask, epsilon=0.0)
    assert chosen == ACTION_DRAW


def test_sixteen_player_aec_environment() -> None:
    """Verify UltimoAECEnv runs cleanly with 16 players and 90-dim observation space."""
    aec_env = UltimoAECEnv(num_players=16, max_turns=60)
    aec_env.reset(seed=42)

    assert len(aec_env.possible_agents) == 16
    for _ in range(40):
        obs, _reward, term, trunc, info = aec_env.last()
        if term or trunc:
            aec_env.step(None)
        else:
            assert obs.shape == (90,)
            mask = info["action_mask"]
            legal = np.where(mask > 0)[0]
            aec_env.step(int(legal[0]))
        assert aec_env._sim.total_cards == 216


def test_sixteen_player_throughput_benchmark() -> None:
    """Benchmark 16-player simulation throughput."""
    sim = UltimoSimulator(num_players=16, seed=42)
    sim.reset()

    steps = 0
    target_steps = 10_000
    start = time.perf_counter()

    while steps < target_steps:
        curr = sim.current_player_idx
        hand = sim.get_player(curr).hand
        playable = get_playable_cards(
            hand,
            sim.top_card,
            sim.active_color,
            pending_skip_deflection=sim.pending_skip_deflection,
            pending_draw_penalty=sim.pending_draw_penalty,
        )
        if playable:
            chosen = playable[0]
            wild_color = CardColor.RED if chosen.is_wild else None
            sim.play_cards(curr, [chosen], chosen_wild_color=wild_color)
        else:
            sim.draw_card(curr)
            sim.pass_turn(curr)

        steps += 1
        if sim.is_game_over:
            sim.reset()

    elapsed = time.perf_counter() - start
    steps_per_sec = steps / elapsed
    assert steps_per_sec > 10_000
    assert sim.direction in (Direction.CLOCKWISE, Direction.COUNTER_CLOCKWISE)
