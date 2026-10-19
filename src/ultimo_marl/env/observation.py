"""Observation space vectorizer encoding game state under strict Fog of War."""

from __future__ import annotations

import numpy as np

from ultimo_marl.domain.card import CardValue
from ultimo_marl.domain.simulator import UltimoSimulator
from ultimo_marl.domain.turn_manager import Direction
from ultimo_marl.env.action_mapping import _SUIT_TO_IDX, _VALUE_TO_IDX

# Observation vector dimension:
# 18 top-card features + 54 hand features + 15 opponent count features + 3 dynamics features = 90
OBSERVATION_DIM: int = 90
LEGACY_OBSERVATION_DIM: int = 78
MAX_OPPONENTS: int = 15


def encode_observation(
    sim: UltimoSimulator,
    observing_player_idx: int,
) -> np.ndarray:
    """Encode current game state into a 90-dimensional float32 vector for an observing agent.

    Fog of War Guarantee:
    - Observing agent has access only to visible table cards, their own hand,
      scalar counts of opponents' hands (up to 15 opponents), and active defense dynamics.
    - Zero information about opponents' card identities or draw deck order is leaked.
    - Output values are strictly bounded in [0.0, 1.0].
    """
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    offset = 0

    # 1. Top Card State (18 floats)
    active_color = sim.active_color
    top_card = sim.top_card

    if active_color is not None and active_color.is_standard_color():
        color_idx = _SUIT_TO_IDX[active_color]
        obs[offset + color_idx] = 1.0
    offset += 4

    if top_card is not None and not top_card.is_wild:
        val_idx = _VALUE_TO_IDX.get(top_card.value)
        if val_idx is not None:
            obs[offset + val_idx] = 1.0
    offset += 13

    if top_card is not None and top_card.is_wild:
        obs[offset] = 1.0
    offset += 1

    # 2. Observing Player's Hand Histogram (54 floats)
    hand = sim.get_player(observing_player_idx).hand
    wild_count = 0
    w4_count = 0

    # 52 standard card slots (normalized by 2.0) and 2 wild slots (normalized by 4.0)
    for card in hand._cards:
        if card.value == CardValue.WILD:
            wild_count += 1
        elif card.value == CardValue.WILD_DRAW_FOUR:
            w4_count += 1
        else:
            suit_idx = _SUIT_TO_IDX[card.color]
            val_idx = _VALUE_TO_IDX[card.value]
            slot = offset + (suit_idx * 13 + val_idx)
            obs[slot] = min(obs[slot] + 0.5, 1.0)
    offset += 52

    obs[offset] = min(float(wild_count) / 4.0, 1.0)
    obs[offset + 1] = min(float(w4_count) / 4.0, 1.0)
    offset += 2

    # 3. Opponent Hand Profiles (15 floats, padded with 0.0 for unused seats up to 16 players)
    num_players = sim.num_players
    max_steps = min(num_players, MAX_OPPONENTS + 1)
    for step in range(1, max_steps):
        opp_idx = (observing_player_idx + step) % num_players
        opp_count = sim.get_player(opp_idx).card_count
        obs[offset + step - 1] = min(float(opp_count) / 10.0, 1.0)
    offset += MAX_OPPONENTS

    # 4. Game & Defense Dynamics (3 floats)
    # Turn direction: 1.0 for CLOCKWISE, 0.0 for COUNTER_CLOCKWISE
    obs[offset] = 1.0 if sim.direction == Direction.CLOCKWISE else 0.0
    offset += 1

    # Pending Skip deflection flag: 1.0 if active, 0.0 otherwise
    obs[offset] = 1.0 if sim.pending_skip_deflection else 0.0
    offset += 1

    # Pending Draw penalty accumulator: normalized by 16.0
    obs[offset] = min(float(sim.pending_draw_penalty) / 16.0, 1.0)
    offset += 1

    assert offset == OBSERVATION_DIM, f"Expected {OBSERVATION_DIM} floats, got {offset}"
    return obs
