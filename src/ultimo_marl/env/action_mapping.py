"""Action space mapping, serialization, and bit-level action mask calculation."""

from __future__ import annotations

import numpy as np

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.domain.move_validator import is_valid_play
from ultimo_marl.domain.player import PlayerHand

# Total action space size:
# 52 standard card plays (4 suits * 13 values)
# + 4 Wild card declarations (WILD with RED, YELLOW, GREEN, BLUE)
# + 4 Wild Draw Four declarations (WILD_DRAW_FOUR with RED, YELLOW, GREEN, BLUE)
# + 1 Draw action
# Total = 61 actions
NUM_ACTIONS: int = 61

STANDARD_SUITS: tuple[CardColor, ...] = CardColor.standard_colors()
STANDARD_VALUES: tuple[CardValue, ...] = (
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
)

_SUIT_TO_IDX: dict[CardColor, int] = {suit: idx for idx, suit in enumerate(STANDARD_SUITS)}
_VALUE_TO_IDX: dict[CardValue, int] = {val: idx for idx, val in enumerate(STANDARD_VALUES)}

ACTION_DRAW: int = 60
WILD_START_IDX: int = 52
WILD_DRAW4_START_IDX: int = 56


def card_to_action_id(card: Card, chosen_wild_color: CardColor | None = None) -> int:
    """Encode a Card and optional wild color declaration into a discrete action integer."""
    if card.value == CardValue.WILD:
        if chosen_wild_color is None or not chosen_wild_color.is_standard_color():
            raise ValueError(f"WILD card requires a valid standard color, got {chosen_wild_color}")
        color_idx = _SUIT_TO_IDX[chosen_wild_color]
        return WILD_START_IDX + color_idx

    if card.value == CardValue.WILD_DRAW_FOUR:
        if chosen_wild_color is None or not chosen_wild_color.is_standard_color():
            raise ValueError(
                f"WILD_DRAW_FOUR requires a valid standard color, got {chosen_wild_color}"
            )
        color_idx = _SUIT_TO_IDX[chosen_wild_color]
        return WILD_DRAW4_START_IDX + color_idx

    suit_idx = _SUIT_TO_IDX[card.color]
    val_idx = _VALUE_TO_IDX[card.value]
    return suit_idx * 13 + val_idx


def action_id_to_card_and_color(
    action_id: int,
) -> tuple[Card | None, CardColor | None]:
    """Decode a discrete action integer back to a Card and optional chosen wild color.

    Returns (None, None) for ACTION_DRAW.
    """
    if action_id == ACTION_DRAW:
        return None, None

    if WILD_START_IDX <= action_id < WILD_DRAW4_START_IDX:
        color_idx = action_id - WILD_START_IDX
        chosen_color = STANDARD_SUITS[color_idx]
        return Card(CardColor.WILD, CardValue.WILD), chosen_color

    if WILD_DRAW4_START_IDX <= action_id < ACTION_DRAW:
        color_idx = action_id - WILD_DRAW4_START_IDX
        chosen_color = STANDARD_SUITS[color_idx]
        return Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR), chosen_color

    if 0 <= action_id < 52:
        suit_idx = action_id // 13
        val_idx = action_id % 13
        return Card(STANDARD_SUITS[suit_idx], STANDARD_VALUES[val_idx]), None

    raise ValueError(f"Action ID {action_id} out of bounds [0, {NUM_ACTIONS - 1}]")


def compute_action_mask(
    hand: PlayerHand,
    top_card: Card | None = None,
    active_color: CardColor | None = None,
    pending_skip_deflection: bool = False,
    pending_draw_penalty: int = 0,
) -> np.ndarray:
    """Compute legal action mask of shape (61,) with dtype np.int8 (1=legal, 0=illegal).

    Invariant: At least one action (at minimum, DRAW) is always legal (sum(mask) >= 1).
    Gymnasium Discrete.sample(mask) requires dtype np.int8.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)

    # DRAW action is always available on your turn (or to accept active defense penalty)
    mask[ACTION_DRAW] = 1

    # 1. Tactical Skip Deflection Masking Rule:
    # Only Skip cards in hand and ACTION_DRAW are legal.
    if pending_skip_deflection:
        for card in hand._cards:
            if card.value == CardValue.SKIP:
                mask[card_to_action_id(card)] = 1
        return mask

    # 2. Tactical Draw Stacking Masking Rule:
    # Only matching Draw Two / Wild Draw Four cards in hand and ACTION_DRAW are legal.
    if pending_draw_penalty > 0:
        has_w4 = False
        for card in hand._cards:
            if card.value == CardValue.DRAW_TWO:
                mask[card_to_action_id(card)] = 1
            elif card.value == CardValue.WILD_DRAW_FOUR:
                has_w4 = True
        if has_w4:
            for idx in range(WILD_DRAW4_START_IDX, ACTION_DRAW):
                mask[idx] = 1
        return mask

    if top_card is None or active_color is None:
        return mask

    # 3. Standard Turn Legality Check
    has_regular_wild = False
    has_wild_draw4 = False

    for card in hand._cards:
        if card.value == CardValue.WILD:
            has_regular_wild = True
        elif card.value == CardValue.WILD_DRAW_FOUR:
            has_wild_draw4 = True
        elif is_valid_play(card, top_card, active_color):
            action_id = card_to_action_id(card)
            mask[action_id] = 1

    # If player holds regular wild, any of the 4 color choices is legal
    if has_regular_wild:
        for idx in range(WILD_START_IDX, WILD_DRAW4_START_IDX):
            mask[idx] = 1

    # If player holds wild draw four, any of the 4 color choices is legal
    if has_wild_draw4:
        for idx in range(WILD_DRAW4_START_IDX, ACTION_DRAW):
            mask[idx] = 1

    return mask
