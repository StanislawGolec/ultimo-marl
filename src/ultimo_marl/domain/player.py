"""Player and PlayerHand domain models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ultimo_marl.domain.card import Card


@dataclass
class PlayerHand:
    """Encapsulates the collection of cards held by an Ultimo player."""

    _cards: list[Card] = field(default_factory=list)

    def add_card(self, card: Card) -> None:
        """Add a card to the hand."""
        self._cards.append(card)

    def remove_card(self, card: Card) -> bool:
        """Remove a card from the hand if present. Returns True if removed."""
        try:
            self._cards.remove(card)
            return True
        except ValueError:
            return False

    def contains(self, card: Card) -> bool:
        """Return True if this hand holds the specified card."""
        return card in self._cards

    @property
    def cards(self) -> list[Card]:
        """Return a copy of the cards held in hand."""
        return list(self._cards)

    def calculate_total_score(self) -> int:
        """Sum official point values of all cards currently held in hand."""
        return sum(card.point_value for card in self._cards)

    def card_counts(self) -> Counter[Card]:
        """Return frequency of each distinct card in hand (handles duplicates)."""
        return Counter(self._cards)

    def clear(self) -> None:
        """Clear all cards in hand."""
        self._cards.clear()

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def is_empty(self) -> bool:
        """Return True if hand has no cards (win condition)."""
        return len(self._cards) == 0

    @property
    def is_ultimo(self) -> bool:
        """Return True if hand has exactly 1 card remaining ('Ultimo!')."""
        return len(self._cards) == 1


@dataclass
class Player:
    """Represents an active player in a match."""

    player_id: int
    name: str
    hand: PlayerHand = field(default_factory=PlayerHand)
    placement_rank: int = 0

    @property
    def card_count(self) -> int:
        """Number of cards currently held."""
        return len(self.hand)

    @property
    def is_ultimo(self) -> bool:
        """Return True if player has exactly 1 card remaining ('Ultimo!')."""
        return len(self.hand) == 1

    @property
    def is_finished(self) -> bool:
        """Return True if player has emptied their hand and claimed a podium rank."""
        return self.placement_rank > 0
