"""Unit, invariant, and PettingZoo API compliance tests for UltimoAECEnv."""

import numpy as np
import pytest
from pettingzoo.test import api_test

from ultimo_marl.domain import Card, CardColor, CardValue, PlayerHand, UltimoSimulator
from ultimo_marl.env import (
    ACTION_DRAW,
    NUM_ACTIONS,
    OBSERVATION_DIM,
    UltimoAECEnv,
    action_id_to_card_and_color,
    card_to_action_id,
    compute_action_mask,
    encode_observation,
    env,
)

# ============================================================================
# 1. Action Mapping Tests
# ============================================================================


def test_action_mapping_standard_cards_roundtrip() -> None:
    """Verify standard cards map to unique IDs [0, 51] and decode back accurately."""
    seen_ids: set[int] = set()

    for color in CardColor.standard_colors():
        for val in (
            CardValue.ZERO,
            CardValue.ONE,
            CardValue.TWO,
            CardValue.THREE,
            CardValue.FOUR,
            CardValue.FIVE,
            CardValue.SIX,
            CardValue.SEVEN,
            CardValue.EIGHT,
            CardValue.NINE,
            CardValue.SKIP,
            CardValue.REVERSE,
            CardValue.DRAW_TWO,
        ):
            card = Card(color, val)
            action_id = card_to_action_id(card)
            assert 0 <= action_id < 52
            assert action_id not in seen_ids
            seen_ids.add(action_id)

            decoded_card, decoded_color = action_id_to_card_and_color(action_id)
            assert decoded_card == card
            assert decoded_color is None

    assert len(seen_ids) == 52


def test_action_mapping_wilds_roundtrip() -> None:
    """Verify regular Wilds map to [52, 55] and Wild Draw Fours map to [56, 59]."""
    for i, color in enumerate(CardColor.standard_colors()):
        # Regular Wild
        w = Card(CardColor.WILD, CardValue.WILD)
        w_id = card_to_action_id(w, chosen_wild_color=color)
        assert w_id == 52 + i
        dec_w, dec_color = action_id_to_card_and_color(w_id)
        assert dec_w == w
        assert dec_color == color

        # Wild Draw Four
        w4 = Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)
        w4_id = card_to_action_id(w4, chosen_wild_color=color)
        assert w4_id == 56 + i
        dec_w4, dec_w4_color = action_id_to_card_and_color(w4_id)
        assert dec_w4 == w4
        assert dec_w4_color == color

    # DRAW action
    dec_draw, dec_draw_color = action_id_to_card_and_color(ACTION_DRAW)
    assert dec_draw is None
    assert dec_draw_color is None

    # Invalid action ID
    with pytest.raises(ValueError, match="out of bounds"):
        action_id_to_card_and_color(999)


# ============================================================================
# 2. Action Mask Computation Tests
# ============================================================================


def test_compute_action_mask_invariants() -> None:
    """Assert action mask has shape (61,), DRAW is always legal, and masks illegal cards."""
    hand = PlayerHand()
    red_five = Card(CardColor.RED, CardValue.FIVE)
    blue_nine = Card(CardColor.BLUE, CardValue.NINE)
    wild = Card(CardColor.WILD, CardValue.WILD)
    hand.add_card(red_five)
    hand.add_card(blue_nine)
    hand.add_card(wild)

    top = Card(CardColor.RED, CardValue.TWO)
    active = CardColor.RED

    mask = compute_action_mask(hand, top, active)
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == np.int8

    # DRAW is always legal
    assert mask[ACTION_DRAW] == 1

    # RED_5 matches active color RED -> Legal
    red_5_id = card_to_action_id(red_five)
    assert mask[red_5_id] == 1

    # BLUE_9 does not match RED or 2 -> Illegal
    blue_9_id = card_to_action_id(blue_nine)
    assert mask[blue_9_id] == 0

    # Holding Wild enables all 4 wild choices (52..55)
    for w_idx in range(52, 56):
        assert mask[w_idx] == 1

    # Player does not hold Wild Draw Four -> (56..59) must be False (0)
    for w4_idx in range(56, 60):
        assert mask[w4_idx] == 0

    # Invariant: at least 1 action is always legal
    assert np.sum(mask) >= 1


# ============================================================================
# 3. Observation Space & Fog of War Tests
# ============================================================================


def test_observation_vectorizer_fog_of_war() -> None:
    """Verify observation vector size 78, [0.0, 1.0] bounds, and strict Fog of War."""
    sim = UltimoSimulator(num_players=4, seed=42)
    sim.reset()

    obs_p0 = encode_observation(sim, observing_player_idx=0)
    assert obs_p0.shape == (OBSERVATION_DIM,)
    assert obs_p0.dtype == np.float32

    # Check bounds
    assert np.all(obs_p0 >= 0.0)
    assert np.all(obs_p0 <= 1.0)

    # Top card active color is one-hot
    assert np.sum(obs_p0[0:4]) == 1.0

    # Opponent card counts (offsets 72..75):
    # Starting hands have 7 cards -> 7 / 10.0 = 0.7
    for opp_offset in range(72, 75):
        assert np.isclose(obs_p0[opp_offset], 0.7)

    # Direction offset 75: 1.0 for clockwise
    assert obs_p0[75] == 1.0


# ============================================================================
# 4. PettingZoo AEC Environment Lifecycle Tests
# ============================================================================


def test_ultimo_aec_env_lifecycle() -> None:
    """Verify reset, observe, step, and termination progression."""
    environment = UltimoAECEnv(num_players=4, max_turns=50)
    environment.reset(seed=123)

    assert len(environment.possible_agents) == 4
    assert environment.agent_selection == "player_0"

    step_count = 0
    while environment.agents and step_count < 100:
        obs, reward, terminated, truncated, info = environment.last()
        if terminated or truncated:
            environment.step(None)
        else:
            assert obs is not None
            assert obs.shape == (OBSERVATION_DIM,)
            mask = info["action_mask"]
            assert mask[ACTION_DRAW] == 1

            # Greedily choose first legal action
            legal_actions = np.where(mask)[0]
            chosen_action = int(legal_actions[0])
            environment.step(chosen_action)
        step_count += 1

    assert step_count > 0


def test_pettingzoo_official_api_test() -> None:
    """Run official PettingZoo AEC API test suite."""
    ae_env = env(num_players=4, max_turns=30)
    api_test(ae_env, num_cycles=5)
