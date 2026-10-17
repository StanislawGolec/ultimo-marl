"""Deck and DiscardPile domain implementations matching official Ultimo rules."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence

from ultimo_marl.domain.card import Card, CardColor, CardValue


class Deck:
    """Represents the draw deck containing standard 108 Ultimo cards.

    Standard deck layout:
    - 19 Blue cards (one 0, two of 1-9)
    - 19 Green cards (one 0, two of 1-9)
    - 19 Red cards (one 0, two of 1-9)
    - 19 Yellow cards (one 0, two of 1-9)
    - 8 Draw Two cards (two in each color)
    - 8 Reverse cards (two in each color)
    - 8 Skip cards (two in each color)
    - 4 Wild cards
    - 4 Wild Draw Four cards
    Total = 108 cards.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._cards: deque[Card] = deque()

    @classmethod
    def create_standard_deck(cls, rng: random.Random | None = None) -> Deck:
        """Create and shuffle a standard 108-card Ultimo deck."""
        deck = cls(rng=rng)
        cards: list[Card] = []

        for color in CardColor.standard_colors():
            # One 0 per color
            cards.append(Card(color, CardValue.ZERO))

            # Two of each 1-9 per color
            number_values = (
                CardValue.ONE,
                CardValue.TWO,
                CardValue.THREE,
                CardValue.FOUR,
                CardValue.FIVE,
                CardValue.SIX,
                CardValue.SEVEN,
                CardValue.EIGHT,
                CardValue.NINE,
            )
            for val in number_values:
                cards.append(Card(color, val))
                cards.append(Card(color, val))

            # Two of each action card per color
            action_values = (CardValue.SKIP, CardValue.REVERSE, CardValue.DRAW_TWO)
            for val in action_values:
                cards.append(Card(color, val))
                cards.append(Card(color, val))

        # Four Wild and four Wild Draw Four cards
        for _ in range(4):
            cards.append(Card(CardColor.WILD, CardValue.WILD))
            cards.append(Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR))

        deck._rng.shuffle(cards)
        deck._cards.extend(cards)
        return deck

    def shuffle(self) -> None:
        """Shuffle remaining cards in the deck in-place."""
        card_list = list(self._cards)
        self._rng.shuffle(card_list)
        self._cards = deque(card_list)

    def draw(self) -> Card | None:
        """Draw one card from the top of the deck, or None if empty."""
        if not self._cards:
            return None
        return self._cards.popleft()

    def replenish_from_discard(self, recycled_cards: Sequence[Card]) -> None:
        """Replenish deck from discard pile cards, shuffling them."""
        if not recycled_cards:
            return
        to_shuffle = list(recycled_cards)
        self._rng.shuffle(to_shuffle)
        self._cards.extend(to_shuffle)

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def is_empty(self) -> bool:
        """Return True if deck has no remaining cards."""
        return len(self._cards) == 0


class DiscardPile:
    """Manages the played cards on the table, top card, and declared active color."""

    def __init__(self) -> None:
        self._played_cards: deque[Card] = deque()
        self._active_color: CardColor | None = None

    def play_card(self, card: Card, chosen_color: CardColor | None = None) -> None:
        """Place a card onto the discard pile and update the active table color."""
        if card.is_wild:
            if chosen_color is None or not chosen_color.is_standard_color():
                raise ValueError("Must declare a valid standard color when playing a wild card")
            self._active_color = chosen_color
        else:
            self._active_color = card.color

        self._played_cards.appendleft(card)

    @property
    def top_card(self) -> Card | None:
        """Return the current top card on the discard pile."""
        if not self._played_cards:
            return None
        return self._played_cards[0]

    @property
    def active_color(self) -> CardColor | None:
        """Return the currently required color for matching."""
        return self._active_color

    def extract_cards_for_replenish(self) -> list[Card]:
        """Extract all cards except the top card for recycling back into the deck."""
        if len(self._played_cards) <= 1:
            return []

        top = self._played_cards.popleft()
        recycled = list(self._played_cards)
        self._played_cards.clear()
        self._played_cards.appendleft(top)
        return recycled

    def __len__(self) -> int:
        return len(self._played_cards)
