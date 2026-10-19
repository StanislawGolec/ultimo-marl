"""High-speed headless game simulator for Ultimo.

Replicates the authoritative Java 21 UltimoEngine rules and turn progression,
optimized for zero I/O and maximum multi-agent simulation throughput (>100k steps/sec).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.domain.deck import Deck, DiscardPile
from ultimo_marl.domain.move_validator import (
    is_valid_play,
    validate_multi_move,
)
from ultimo_marl.domain.player import Player
from ultimo_marl.domain.turn_manager import Direction, TurnManager


@dataclass(frozen=True, slots=True)
class PlayResult:
    """Outcome of attempting to play cards."""

    success: bool
    error_message: str | None
    played_cards: tuple[Card, ...]
    is_ultimo: bool
    is_game_over: bool
    winner_idx: int | None

    @classmethod
    def failure(cls, error: str) -> PlayResult:
        """Construct an invalid move result."""
        return cls(
            success=False,
            error_message=error,
            played_cards=(),
            is_ultimo=False,
            is_game_over=False,
            winner_idx=None,
        )

    @classmethod
    def successful(
        cls,
        cards: Sequence[Card],
        is_ultimo: bool,
        is_game_over: bool,
        winner_idx: int | None,
    ) -> PlayResult:
        """Construct a valid move result."""
        return cls(
            success=True,
            error_message=None,
            played_cards=tuple(cards),
            is_ultimo=is_ultimo,
            is_game_over=is_game_over,
            winner_idx=winner_idx,
        )


class UltimoSimulator:
    """Headless simulation engine for Ultimo matches."""

    STARTING_HAND_SIZE: int = 7

    def __init__(self, num_players: int = 4, seed: int | None = None) -> None:
        if num_players < 2 or num_players > 16:
            raise ValueError(f"Player count must be between 2 and 16, got {num_players}")
        self._num_players = num_players
        self._num_decks = 2 if num_players > 8 else 1
        self._rng = random.Random(seed)
        self._players: list[Player] = [
            Player(player_id=i, name=f"Player_{i}") for i in range(num_players)
        ]
        self._deck: Deck = Deck(rng=self._rng, num_decks=self._num_decks)
        self._discard_pile: DiscardPile = DiscardPile()
        self._turn_manager: TurnManager = TurnManager(num_players)
        self._is_game_started: bool = False
        self._is_game_over: bool = False
        self._has_drawn_this_turn: bool = False
        self._winner_idx: int | None = None

    def reset(self, seed: int | None = None) -> None:
        """Reset and initialize a fresh match."""
        if seed is not None:
            self._rng = random.Random(seed)

        self._num_decks = 2 if self._num_players > 8 else 1
        self._deck = Deck.create_standard(num_decks=self._num_decks, rng=self._rng)
        self._discard_pile = DiscardPile()
        self._turn_manager = TurnManager(self._num_players)

        # Clear hands and deal 7 cards each
        for player in self._players:
            player.hand.clear()
            for _ in range(self.STARTING_HAND_SIZE):
                card = self._draw_from_deck()
                if card is not None:
                    player.hand.add_card(card)

        # Flip starting card from deck (must be a number card per authoritative rules)
        starting_card: Card | None = None
        while starting_card is None:
            card = self._draw_from_deck()
            if card is None:
                raise RuntimeError("Deck depleted during setup initialization")
            if card.is_number:
                starting_card = card
            else:
                self._deck.replenish_from_discard([card])

        self._discard_pile.play_card(starting_card, starting_card.color)
        self._is_game_started = True
        self._is_game_over = False
        self._has_drawn_this_turn = False
        self._winner_idx = None

        self.assert_card_conservation()

    def assert_card_conservation(self) -> None:
        """Assert that total cards across hands, deck, and discard pile equal 108 * num_decks."""
        expected = 108 * self._num_decks
        actual = self.total_cards
        assert actual == expected, (
            f"Card conservation invariant breached: total cards {actual} != {expected}"
        )

    def play_cards(
        self,
        player_idx: int,
        cards: Sequence[Card],
        chosen_wild_color: CardColor | None = None,
    ) -> PlayResult:
        """Attempt to play one or more cards from player's hand."""
        if not self._is_game_started:
            return PlayResult.failure("Game has not started yet")
        if self._is_game_over:
            return PlayResult.failure("Game is already over")
        if player_idx != self._turn_manager.current_index:
            return PlayResult.failure(f"It is not player {player_idx}'s turn")
        if not cards:
            return PlayResult.failure("No cards specified to play")

        cards_before = self.total_cards
        expected_cards = 108 * self._num_decks

        player = self._players[player_idx]
        top = self._discard_pile.top_card
        active_color = self._discard_pile.active_color

        validation_err = validate_multi_move(
            hand=player.hand,
            cards=cards,
            top_card=top,
            active_color=active_color,
            chosen_wild_color=chosen_wild_color,
            pending_skip_deflection=self._turn_manager.pending_skip_deflection,
            pending_draw_penalty=self._turn_manager.pending_draw_penalty,
        )
        if validation_err is not None:
            return PlayResult.failure(validation_err)

        # Ensure initiator card matching table is placed first onto discard pile
        ordered_cards = list(cards)
        if (
            not self._turn_manager.pending_skip_deflection
            and self._turn_manager.pending_draw_penalty == 0
            and not is_valid_play(ordered_cards[0], top, active_color)
        ):
            for i in range(1, len(ordered_cards)):
                if is_valid_play(ordered_cards[i], top, active_color):
                    initiator = ordered_cards.pop(i)
                    ordered_cards.insert(0, initiator)
                    break

        # Remove cards from hand and place on discard pile
        for card in ordered_cards:
            player.hand.remove_card(card)
            effective_color = chosen_wild_color if card.is_wild else card.color
            self._discard_pile.play_card(card, effective_color)

        # Check for win condition
        if player.hand.is_empty:
            self._turn_manager.pending_skip_deflection = False
            self._turn_manager.pending_draw_penalty = 0
            self._is_game_over = True
            self._winner_idx = player_idx
            if cards_before == expected_cards:
                self.assert_card_conservation()
            return PlayResult.successful(
                cards=ordered_cards,
                is_ultimo=False,
                is_game_over=True,
                winner_idx=player_idx,
            )

        is_ultimo = player.hand.is_ultimo

        # Resolve card action effects
        self._apply_combo_action(ordered_cards)
        self._has_drawn_this_turn = False

        if cards_before == expected_cards:
            self.assert_card_conservation()

        return PlayResult.successful(
            cards=ordered_cards,
            is_ultimo=is_ultimo,
            is_game_over=False,
            winner_idx=None,
        )

    def draw_card(self, player_idx: int) -> Card | None:
        """Draw card(s) from the draw deck or resolve active defense penalties."""
        if not self._is_game_started or self._is_game_over:
            return None
        if player_idx != self._turn_manager.current_index:
            return None

        cards_before = self.total_cards
        expected_cards = 108 * self._num_decks

        # If facing a deflectable Skip and choosing not to deflect, accept skip and forfeit turn
        if self._turn_manager.pending_skip_deflection:
            self._turn_manager.pending_skip_deflection = False
            self._has_drawn_this_turn = False
            self._turn_manager.advance_turn(1)
            if cards_before == expected_cards:
                self.assert_card_conservation()
            return None

        # If facing an accumulated Draw penalty, draw full penalty and forfeit turn
        if self._turn_manager.pending_draw_penalty > 0:
            penalty = self._turn_manager.pending_draw_penalty
            last_drawn: Card | None = None
            target_hand = self._players[player_idx].hand
            for _ in range(penalty):
                card = self._draw_from_deck()
                if card is not None:
                    target_hand.add_card(card)
                    last_drawn = card
            self._turn_manager.pending_draw_penalty = 0
            self._has_drawn_this_turn = False
            self._turn_manager.advance_turn(1)
            if cards_before == expected_cards:
                self.assert_card_conservation()
            return last_drawn

        if self._has_drawn_this_turn:
            return None  # Maximum 1 draw per normal turn

        drawn = self._draw_from_deck()
        if drawn is not None:
            self._players[player_idx].hand.add_card(drawn)
            self._has_drawn_this_turn = True

        if cards_before == expected_cards:
            self.assert_card_conservation()

        return drawn

    def pass_turn(self, player_idx: int) -> bool:
        """Pass turn after drawing a card or when accepting a pending skip."""
        if not self._is_game_started or self._is_game_over:
            return False
        if player_idx != self._turn_manager.current_index:
            return False

        if self._turn_manager.pending_skip_deflection:
            self._turn_manager.pending_skip_deflection = False
            self._has_drawn_this_turn = False
            self._turn_manager.advance_turn(1)
            return True

        if not self._has_drawn_this_turn:
            return False  # In Ultimo, players may only pass after drawing

        self._turn_manager.advance_turn(1)
        self._has_drawn_this_turn = False
        return True

    def calculate_scores(self) -> dict[int, int]:
        """Calculate final match scores.

        Winner receives sum of all remaining cards in all opponents' hands.
        Opponents receive 0.
        """
        scores = {p.player_id: 0 for p in self._players}
        if not self._is_game_over or self._winner_idx is None:
            return scores

        winner_points = sum(
            p.hand.calculate_total_score() for p in self._players if p.player_id != self._winner_idx
        )
        scores[self._winner_idx] = winner_points
        return scores

    def _apply_combo_action(self, cards: Sequence[Card]) -> None:
        count = len(cards)
        last_card = cards[-1]

        if last_card.value == CardValue.SKIP:
            self._turn_manager.pending_skip_deflection = True
            self._turn_manager.advance_turn(1)
        elif last_card.value == CardValue.REVERSE:
            if self._turn_manager.is_two_player_game:
                if count % 2 == 1:
                    self._turn_manager.advance_turn(2)  # Odd reverses act as Skip in 2-player
                else:
                    self._turn_manager.advance_turn(1)  # Even reverses cancel out
            else:
                for _ in range(count):
                    self._turn_manager.reverse_direction()
                self._turn_manager.advance_turn(1)
        elif last_card.value == CardValue.DRAW_TWO:
            self._turn_manager.pending_draw_penalty += count * 2
            self._turn_manager.advance_turn(1)
        elif last_card.value == CardValue.WILD_DRAW_FOUR:
            self._turn_manager.pending_draw_penalty += count * 4
            self._turn_manager.advance_turn(1)
        else:
            # Number cards and regular Wild advance by 1
            self._turn_manager.advance_turn(1)

    def _draw_cards_for(self, player_idx: int, count: int) -> None:
        target = self._players[player_idx]
        for _ in range(count):
            card = self._draw_from_deck()
            if card is not None:
                target.hand.add_card(card)

    def _draw_from_deck(self) -> Card | None:
        if self._deck.is_empty:
            recycled = self._discard_pile.extract_cards_for_replenish()
            self._deck.replenish_from_discard(recycled)
        return self._deck.draw()

    # Read-only and Defense Inspection Properties
    @property
    def current_player_idx(self) -> int:
        """Active player index."""
        return self._turn_manager.current_index

    @property
    def direction(self) -> Direction:
        """Current table direction."""
        return self._turn_manager.direction

    @property
    def num_players(self) -> int:
        """Number of players."""
        return self._num_players

    @property
    def num_decks(self) -> int:
        """Number of 108-card standard decks used in match."""
        return self._num_decks

    @property
    def total_cards(self) -> int:
        """Total cards currently across all hands, draw deck, and discard pile."""
        return (
            sum(len(p.hand._cards) for p in self._players)
            + len(self._deck)
            + len(self._discard_pile)
        )

    @property
    def pending_skip_deflection(self) -> bool:
        """Whether current player is targeted by a deflectable Skip."""
        return self._turn_manager.pending_skip_deflection

    @pending_skip_deflection.setter
    def pending_skip_deflection(self, value: bool) -> None:
        self._turn_manager.pending_skip_deflection = value

    @property
    def pending_draw_penalty(self) -> int:
        """Accumulated draw penalty targeting the current player."""
        return self._turn_manager.pending_draw_penalty

    @pending_draw_penalty.setter
    def pending_draw_penalty(self, value: int) -> None:
        self._turn_manager.pending_draw_penalty = value

    @property
    def top_card(self) -> Card | None:
        """Top card on the discard pile."""
        return self._discard_pile.top_card

    @property
    def active_color(self) -> CardColor | None:
        """Current required color on table."""
        return self._discard_pile.active_color

    @property
    def deck_size(self) -> int:
        """Number of cards in draw deck."""
        return len(self._deck)

    @property
    def discard_pile_size(self) -> int:
        """Number of cards in discard pile."""
        return len(self._discard_pile)

    @property
    def is_game_started(self) -> bool:
        """Whether match has started."""
        return self._is_game_started

    @property
    def is_game_over(self) -> bool:
        """Whether match is complete."""
        return self._is_game_over

    @property
    def winner_idx(self) -> int | None:
        """Winner player index if game is over."""
        return self._winner_idx

    @property
    def has_drawn_this_turn(self) -> bool:
        """Whether current player has drawn a card this turn."""
        return self._has_drawn_this_turn

    def get_player(self, player_idx: int) -> Player:
        """Get Player instance by index."""
        return self._players[player_idx]

    def get_player_card_counts(self) -> list[int]:
        """Return list of card counts for all players."""
        return [p.card_count for p in self._players]
