"""Observation space vectorizer encoding game state under strict Fog of War."""

from __future__ import annotations

import numpy as np

from ultimo_marl.domain.card import CardColor, CardValue
from ultimo_marl.domain.simulator import UltimoSimulator
from ultimo_marl.domain.turn_manager import Direction
from ultimo_marl.env.action_mapping import STANDARD_SUITS, STANDARD_VALUES

# Observation vector dimension:
# 18 top-card features + 54 hand features + 3 opponent count features + 3 dynamics features = 78
OBSERVATION_DIM: int = 78


def encode_observation(
    sim: UltimoSimulator,
    observing_player_idx: int,
) -> np.ndarray:
    """Encode current game state into a 78-dimensional float32 vector for an observing agent.

    Fog of War Guarantee:
    - Observing agent has access only to visible table cards, their own hand,
      and scalar counts of opponents' hands.
    - Zero information about opponents' card identities or draw deck order is leaked.
    - Output values are strictly bounded in [0.0, 1.0].
    """
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    offset = 0

    # 1. Top Card State (18 floats)
    active_color = sim.active_color
    top_card = sim.top_card

    if active_color is not None and active_color.is_standard_color():
        color_idx = STANDARD_SUITS.index(active_color)
        obs[offset + color_idx] = 1.0
    offset += 4

    if top_card is not None and not top_card.is_wild:
        if top_card.value in STANDARD_VALUES:
            val_idx = STANDARD_VALUES.index(top_card.value)
            obs[offset + val_idx] = 1.0
    offset += 13

    if top_card is not None and top_card.is_wild:
        obs[offset] = 1.0
    offset += 1

    # 2. Observing Player's Hand Histogram (54 floats)
    hand = sim.get_player(observing_player_idx).hand
    hand_counts = hand.card_counts()

    # 52 standard card slots (normalized by 2.0, since deck contains max 2 copies)
    for card, count in hand_counts.items():
        if not card.is_wild:
            suit_idx = STANDARD_SUITS.index(card.color)
            val_idx = STANDARD_VALUES.index(card.value)
            slot = offset + (suit_idx * 13 + val_idx)
            obs[slot] = min(float(count) / 2.0, 1.0)
    offset += 52

    # Wild and Wild Draw Four counts (normalized by 4.0)
    wild_count = sum(1 for c in hand.cards if c.value == CardValue.WILD)
    w4_count = sum(1 for c in hand.cards if c.value == CardValue.WILD_DRAW_FOUR)
    obs[offset] = min(float(wild_count) / 4.0, 1.0)
    obs[offset + 1] = min(float(w4_count) / 4.0, 1.0)
    offset += 2

    # 3. Opponent Hand Profiles (3 floats)
    # Normalized card counts for the 3 opponents in cyclic table order
    num_players = sim.num_players
    for step in range(1, num_players):
        opp_idx = (observing_player_idx + step) % num_players
        opp_count = sim.get_player(opp_idx).card_count
        # Normalize: 10+ cards saturates at 1.0
        obs[offset] = min(float(opp_count) / 10.0, 1.0)
        offset += 1

    # 4. Game Dynamics (3 floats)
    # Direction: 1.0 for CLOCKWISE, 0.0 for COUNTER_CLOCKWISE
    obs[offset] = 1.0 if sim.direction == Direction.CLOCKWISE else 0.0
    offset += 1

    # Opponent Ultimo warning: 1.0 if any opponent has <= 1 card remaining
    any_opp_ultimo = any(
        sim.get_player((observing_player_idx + s) % num_players).card_count <= 1
        for s in range(1, num_players)
    )
    obs[offset] = 1.0 if any_opp_ultimo else 0.0
    offset += 1

    # Deck remaining ratio: deck_size / 108.0
    obs[offset] = min(float(sim.deck_size) / 108.0, 1.0)
    offset += 1

    assert offset == OBSERVATION_DIM, f"Expected {OBSERVATION_DIM} floats, got {offset}"
    return obs
