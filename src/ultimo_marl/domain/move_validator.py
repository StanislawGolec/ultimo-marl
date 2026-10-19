"""Pure domain validator enforcing official Ultimo move legality rules.

Replicates MoveValidator.java from the authoritative server 1:1.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.domain.player import PlayerHand


def is_valid_play(
    candidate_card: Card | None,
    top_card: Card | None,
    active_color: CardColor | None,
) -> bool:
    """Check if a card is legally playable against the active table state.

    A play is valid if:
    1. The candidate card is a WILD or WILD_DRAW_FOUR, OR
    2. The candidate card matches the currently active color, OR
    3. The candidate card matches the face value of the current top card.
    """
    if candidate_card is None or top_card is None or active_color is None:
        return False

    if candidate_card.is_wild:
        return True

    return candidate_card.color == active_color or candidate_card.value == top_card.value


def validate_single_move(
    hand: PlayerHand,
    card: Card,
    top_card: Card | None,
    active_color: CardColor | None,
    chosen_wild_color: CardColor | None = None,
    pending_skip_deflection: bool = False,
    pending_draw_penalty: int = 0,
) -> str | None:
    """Validate an attempted single-card move by a player.

    Returns None if the move is valid, or a descriptive error reason string.
    """
    if not hand.contains(card):
        return f"You do not hold card: {card}"

    if pending_skip_deflection:
        if card.value != CardValue.SKIP:
            return f"Must play a SKIP card to deflect pending skip, got {card}"
        return None

    if pending_draw_penalty > 0:
        if card.value not in (CardValue.DRAW_TWO, CardValue.WILD_DRAW_FOUR):
            return f"Must play a DRAW_TWO or WILD_DRAW_FOUR to stack penalty, got {card}"
        if card.is_wild and (
            chosen_wild_color is None or not chosen_wild_color.is_standard_color()
        ):
            return "Must select a valid color (RED, YELLOW, GREEN, BLUE) for wild card"
        return None

    if card.is_wild:
        if chosen_wild_color is None or not chosen_wild_color.is_standard_color():
            return "Must select a valid color (RED, YELLOW, GREEN, BLUE) for wild card"
    elif not is_valid_play(card, top_card, active_color):
        return f"Card {card} cannot be played on {top_card} (active color: {active_color})"

    return None


def validate_multi_move(
    hand: PlayerHand,
    cards: Sequence[Card],
    top_card: Card | None,
    active_color: CardColor | None,
    chosen_wild_color: CardColor | None = None,
    pending_skip_deflection: bool = False,
    pending_draw_penalty: int = 0,
) -> str | None:
    """Validate an attempted multi-card combo move by a player.

    Rules for combos:
    1. Cards sequence must not be empty.
    2. The player must hold all specified cards in hand (accounting for duplicates).
    3. All cards in a combo must share the same face value (e.g. three SEVENs).
    4. At least one card in the combo must match the current table state (or defense state).
    5. If any card in the combo is wild, a valid standard wild color must be declared.

    Returns None if valid, or a descriptive error reason string.
    """
    if not cards:
        return "No cards specified to play"

    if len(cards) == 1:
        return validate_single_move(
            hand,
            cards[0],
            top_card,
            active_color,
            chosen_wild_color,
            pending_skip_deflection=pending_skip_deflection,
            pending_draw_penalty=pending_draw_penalty,
        )

    # Check that hand holds sufficient quantities of each card
    required_counts = Counter(cards)
    hand_counts = hand.card_counts()
    for card, count in required_counts.items():
        if hand_counts.get(card, 0) < count:
            return f"You do not hold enough copies of card: {card}"

    # All cards in a combo must share the same face value
    first_value = cards[0].value
    if not all(c.value == first_value for c in cards):
        return "All cards in a combo must share the same face value"

    if pending_skip_deflection:
        if first_value != CardValue.SKIP:
            return "Must play SKIP cards to deflect pending skip"
        return None

    if pending_draw_penalty > 0:
        if first_value not in (CardValue.DRAW_TWO, CardValue.WILD_DRAW_FOUR):
            return "Must play DRAW_TWO or WILD_DRAW_FOUR cards to stack penalty"
        if any(c.is_wild for c in cards) and (
            chosen_wild_color is None or not chosen_wild_color.is_standard_color()
        ):
            return "Must select a valid color (RED, YELLOW, GREEN, BLUE) for wild card"
        return None

    # At least one card in the combo must match the current table
    can_match_table = any(is_valid_play(c, top_card, active_color) for c in cards)
    if not can_match_table:
        return f"None of the cards in combo ({cards}) can be played on {top_card} (active: {active_color})"

    # If any card is wild, ensure valid color chosen
    if any(c.is_wild for c in cards):
        if chosen_wild_color is None or not chosen_wild_color.is_standard_color():
            return "Must select a valid color (RED, YELLOW, GREEN, BLUE) for wild card"

    return None


def get_playable_cards(
    hand: PlayerHand,
    top_card: Card | None,
    active_color: CardColor | None,
    pending_skip_deflection: bool = False,
    pending_draw_penalty: int = 0,
) -> list[Card]:
    """Return all cards in hand that are legally playable individually on table."""
    if pending_skip_deflection:
        return [c for c in hand._cards if c.value == CardValue.SKIP]
    if pending_draw_penalty > 0:
        return [c for c in hand._cards if c.value in (CardValue.DRAW_TWO, CardValue.WILD_DRAW_FOUR)]
    if top_card is None or active_color is None:
        return []
    return [c for c in hand._cards if is_valid_play(c, top_card, active_color)]
