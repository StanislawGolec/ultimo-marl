"""Domain models for Ultimo cards, colors, and values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, unique


@unique
class CardColor(Enum):
    """The four standard Ultimo suits plus special WILD color."""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    BLUE = "BLUE"
    WILD = "WILD"

    def is_standard_color(self) -> bool:
        """Return True if this color is one of the four playable suits."""
        return self != CardColor.WILD

    @classmethod
    def standard_colors(cls) -> tuple[CardColor, ...]:
        """Return all four standard playable suits."""
        return (cls.RED, cls.YELLOW, cls.GREEN, cls.BLUE)


@unique
class CardValue(IntEnum):
    """Face values and special action types with official point valuations."""

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    SKIP = 20
    REVERSE = 21
    DRAW_TWO = 22
    WILD = 50
    WILD_DRAW_FOUR = 51

    @property
    def point_value(self) -> int:
        """Official point value awarded to opponents upon round completion."""
        if self <= CardValue.NINE:
            return int(self.value)
        if self in (CardValue.SKIP, CardValue.REVERSE, CardValue.DRAW_TWO):
            return 20
        return 50  # WILD and WILD_DRAW_FOUR

    def is_number(self) -> bool:
        """Return True if this card has a numeric face value (0-9)."""
        return self <= CardValue.NINE

    def is_action(self) -> bool:
        """Return True if this card triggers a special action (Skip, Reverse, Draw Two)."""
        return self in (CardValue.SKIP, CardValue.REVERSE, CardValue.DRAW_TWO)

    def is_wild(self) -> bool:
        """Return True if this card is a Wild or Wild Draw Four."""
        return self in (CardValue.WILD, CardValue.WILD_DRAW_FOUR)


@dataclass(frozen=True, slots=True)
class Card:
    """Immutable representation of a single Ultimo card.

    Invariants matching Java authoritative server:
    - Wild cards must have CardColor.WILD.
    - Standard cards must have a standard color (RED, YELLOW, GREEN, BLUE).
    """

    color: CardColor
    value: CardValue

    def __post_init__(self) -> None:
        """Enforce domain invariants matching the authoritative Java server."""
        if self.value.is_wild() and self.color != CardColor.WILD:
            raise ValueError(f"Wild cards must have WILD color, got {self.color}")
        if not self.value.is_wild() and self.color == CardColor.WILD:
            raise ValueError(f"Standard cards cannot have WILD color, got {self.value}")

    @property
    def is_wild(self) -> bool:
        """Return True if this card is a wild card."""
        return self.value.is_wild()

    @property
    def is_action(self) -> bool:
        """Return True if this card is a special action card."""
        return self.value.is_action()

    @property
    def is_number(self) -> bool:
        """Return True if this card is a number card."""
        return self.value.is_number()

    @property
    def point_value(self) -> int:
        """Return point value for scoring."""
        return self.value.point_value

    def __str__(self) -> str:
        """Wire format token representation matching Java ProtocolCodec."""
        if self.value == CardValue.WILD:
            return "WILD"
        if self.value == CardValue.WILD_DRAW_FOUR:
            return "WILD_DRAW_FOUR"
        return f"{self.color.name}_{self.value.name}"

    def __repr__(self) -> str:
        return f"Card({self.color.name}, {self.value.name})"
