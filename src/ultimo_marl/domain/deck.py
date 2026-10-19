"""Deck and DiscardPile domain implementations matching official Ultimo rules."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence

from ultimo_marl.domain.card import Card, CardColor, CardValue


def _build_standard_108_cards() -> tuple[Card, ...]:
    """Build the canonical immutable 108-card Ultimo deck template once."""
    cards: list[Card] = []
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
    action_values = (CardValue.SKIP, CardValue.REVERSE, CardValue.DRAW_TWO)

    for color in CardColor.standard_colors():
        cards.append(Card(color, CardValue.ZERO))
        for val in number_values:
            cards.append(Card(color, val))
            cards.append(Card(color, val))
        for val in action_values:
            cards.append(Card(color, val))
            cards.append(Card(color, val))

    for _ in range(4):
        cards.append(Card(CardColor.WILD, CardValue.WILD))
        cards.append(Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR))

    return tuple(cards)


_STANDARD_108_CARDS: tuple[Card, ...] = _build_standard_108_cards()


class Deck:
    """Represents the draw deck containing standard Ultimo cards (108 per deck).

    Standard single-deck layout:
    - 19 Blue cards (one 0, two of 1-9)
    - 19 Green cards (one 0, two of 1-9)
    - 19 Red cards (one 0, two of 1-9)
    - 19 Yellow cards (one 0, two of 1-9)
    - 8 Draw Two cards (two in each color)
    - 8 Reverse cards (two in each color)
    - 8 Skip cards (two in each color)
    - 4 Wild cards
    - 4 Wild Draw Four cards
    Total = 108 * num_decks cards.
    """

    def __init__(self, rng: random.Random | None = None, num_decks: int = 1) -> None:
        if num_decks < 1:
            raise ValueError(f"num_decks must be >= 1, got {num_decks}")
        self._rng = rng if rng is not None else random.Random()
        self._num_decks = num_decks
        self._cards: deque[Card] = deque()

    @property
    def num_decks(self) -> int:
        """Number of standard 108-card decks combined in this draw deck."""
        return self._num_decks

    @classmethod
    def create_standard(
        cls,
        num_decks: int = 1,
        rng: random.Random | None = None,
    ) -> Deck:
        """Create and shuffle a multi-deck Ultimo draw pile (108 * num_decks cards)."""
        if num_decks < 1:
            raise ValueError(f"num_decks must be >= 1, got {num_decks}")
        deck = cls(rng=rng, num_decks=num_decks)
        cards: list[Card] = list(_STANDARD_108_CARDS * num_decks)
        deck._rng.shuffle(cards)
        deck._cards.extend(cards)
        return deck

    @classmethod
    def create_standard_deck(
        cls,
        rng: random.Random | None = None,
        num_decks: int = 1,
    ) -> Deck:
        """Create and shuffle a standard Ultimo deck (alias accepting optional num_decks)."""
        return cls.create_standard(num_decks=num_decks, rng=rng)

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
