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

    def __init__(
        self,
        num_players: int = 4,
        seed: int | None = None,
        *,
        last_man_standing: bool = False,
    ) -> None:
        if num_players < 2 or num_players > 16:
            raise ValueError(f"Player count must be between 2 and 16, got {num_players}")
        self._num_players = num_players
        self._num_decks = 2 if num_players > 8 else 1
        self._last_man_standing = last_man_standing
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
        self._finished_players: list[int] = []
        self._declared_ultimo: set[int] = set()
        self._pending_ultimo_violator: int | None = None

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
            player.placement_rank = 0
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
        self._finished_players = []
        self._declared_ultimo = set()
        self._pending_ultimo_violator = None

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

        # Any move closes the previous catch window
        self._pending_ultimo_violator = None

        # Remove cards from hand and place on discard pile
        for card in ordered_cards:
            player.hand.remove_card(card)
            effective_color = chosen_wild_color if card.is_wild else card.color
            self._discard_pile.play_card(card, effective_color)

        # Check for win condition
        if player.hand.is_empty:
            if self._last_man_standing:
                rank = len(self._finished_players) + 1
                player.placement_rank = rank
                self._finished_players.append(player_idx)
                self._turn_manager.mark_player_finished(player_idx)

                if self._winner_idx is None:
                    self._winner_idx = player_idx

                # Resolve combo action effects onto next active player
                self._apply_combo_action(ordered_cards)
                self._has_drawn_this_turn = False

                if self._turn_manager.active_player_count <= 1:
                    for idx, remaining in enumerate(self._players):
                        if not remaining.is_finished:
                            final_rank = len(self._finished_players) + 1
                            remaining.placement_rank = final_rank
                            self._finished_players.append(idx)
                            self._turn_manager.mark_player_finished(idx)
                            break
                    self._is_game_over = True
                    self._turn_manager.pending_skip_deflection = False
                    self._turn_manager.pending_draw_penalty = 0
                    if cards_before == expected_cards:
                        self.assert_card_conservation()
                    return PlayResult.successful(
                        cards=ordered_cards,
                        is_ultimo=False,
                        is_game_over=True,
                        winner_idx=self._winner_idx,
                    )
                else:
                    if cards_before == expected_cards:
                        self.assert_card_conservation()
                    return PlayResult.successful(
                        cards=ordered_cards,
                        is_ultimo=False,
                        is_game_over=False,
                        winner_idx=None,
                    )
            else:
                self._turn_manager.pending_skip_deflection = False
                self._turn_manager.pending_draw_penalty = 0
                self._is_game_over = True
                self._winner_idx = player_idx
                player.placement_rank = 1
                self._finished_players.append(player_idx)
                if cards_before == expected_cards:
                    self.assert_card_conservation()
                return PlayResult.successful(
                    cards=ordered_cards,
                    is_ultimo=False,
                    is_game_over=True,
                    winner_idx=player_idx,
                )

        is_ultimo = player.hand.is_ultimo
        if is_ultimo:
            if player_idx not in self._declared_ultimo:
                self._pending_ultimo_violator = player_idx
        else:
            self._declared_ultimo.discard(player_idx)

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

        # Any move closes the previous catch window
        self._pending_ultimo_violator = None

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

    def declare_ultimo(self, player_idx: int) -> bool:
        """Register that a player declared ULTIMO (called before/alongside playing down to 1 card)."""
        self._declared_ultimo.add(player_idx)
        if self._pending_ultimo_violator == player_idx:
            self._pending_ultimo_violator = None
        return True

    def catch_ultimo(self, challenger_idx: int, target_idx: int) -> tuple[bool, int]:
        """Attempt to catch an opponent who failed to declare ULTIMO.

        If target is caught, target draws 6 cards and returns (True, 6).
        Otherwise returns (False, 0).
        """
        if self._pending_ultimo_violator != target_idx:
            return False, 0
        target_player = self._players[target_idx]
        if target_player.card_count != 1 or target_idx in self._declared_ultimo:
            return False, 0

        self._draw_cards_for(target_idx, 6)
        self._pending_ultimo_violator = None
        self._declared_ultimo.discard(target_idx)
        return True, 6

    @property
    def pending_ultimo_violator(self) -> int | None:
        """0-indexed player who currently holds 1 card and failed to declare ULTIMO."""
        return self._pending_ultimo_violator

    @property
    def placements(self) -> dict[int, int]:
        """Mapping from player_id to awarded placement rank (1..N)."""
        return {p.player_id: p.placement_rank for p in self._players if p.is_finished}

    @property
    def match_standings(self) -> list[tuple[int, int]]:
        """List of (rank, player_id) placements in finish order."""
        return [(self._players[i].placement_rank, i) for i in self._finished_players]

    @property
    def is_last_man_standing(self) -> bool:
        """Whether this simulator is configured for Last-Man-Standing continuation."""
        return self._last_man_standing

    def get_player(self, player_idx: int) -> Player:
        """Get Player instance by index."""
        return self._players[player_idx]

    def get_player_card_counts(self) -> list[int]:
        """Return list of card counts for all players."""
        return [p.card_count for p in self._players]
