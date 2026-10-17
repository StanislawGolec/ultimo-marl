"""Internal board state representation maintained purely from wire protocol events."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.domain.player import PlayerHand
from ultimo_marl.env.action_mapping import (
    ACTION_DRAW,
    STANDARD_SUITS,
    STANDARD_VALUES,
    compute_action_mask,
)
from ultimo_marl.env.observation import OBSERVATION_DIM
from ultimo_marl.protocol.events import (
    DrawResult,
    ErrorEvent,
    GameOver,
    HandUpdate,
    OpponentUpdate,
    RoomStatus,
    ServerEvent,
    TopCard,
    TurnAnnounced,
    Welcome,
)

logger = logging.getLogger(__name__)


@dataclass
class LiveGameState:
    """Tracks the live match state under Fog of War reconstructed from TCP server events."""

    bot_name: str
    player_id: str | None = None
    room_id: str | None = None
    hand: PlayerHand = field(default_factory=PlayerHand)
    top_card: Card | None = None
    active_color: CardColor | None = None
    opponent_counts: dict[str, int] = field(default_factory=dict)
    player_names: list[str] = field(default_factory=list)
    current_turn_player: str | None = None
    has_drawn_this_turn: bool = False
    is_game_over: bool = False
    winner: str | None = None
    turn_count: int = 0

    def apply_event(self, event: ServerEvent) -> bool:
        """Process an inbound server event and update internal state.

        Returns True if this event announces that it is the bot's turn to act.
        """
        match event:
            case Welcome(assigned_player_id=pid):
                self.player_id = pid
                logger.info("Bot registered with server ID: %s", pid)

            case RoomStatus(room_id=rid, player_names=names):
                self.room_id = rid
                self.player_names = list(names)
                for name in names:
                    if name != self.bot_name and name not in self.opponent_counts:
                        self.opponent_counts[name] = 7  # Standard initial hand size

            case HandUpdate(cards=cards):
                self.hand = PlayerHand(cards)
                logger.debug("Hand updated: %d cards held", len(cards))

            case TopCard(card=c, active_color=color):
                self.top_card = c
                self.active_color = color
                logger.debug("Top card on table: %s (active color: %s)", c, color)

            case OpponentUpdate(opponent_name=oname, card_count=cnt):
                self.opponent_counts[oname] = cnt
                logger.debug("Opponent '%s' has %d cards", oname, cnt)

            case DrawResult(drawn_card=c):
                self.hand.add_card(c)
                self.has_drawn_this_turn = True
                logger.debug("Drawn card: %s", c)

            case TurnAnnounced(active_player_name=pname):
                self.current_turn_player = pname
                self.turn_count += 1
                is_my_turn = pname == self.bot_name
                if is_my_turn:
                    self.has_drawn_this_turn = False
                    logger.info(
                        "Turn announced: Bot '%s' is acting (turn #%d)", pname, self.turn_count
                    )
                    return True
                logger.debug("Turn announced: Player '%s' is acting", pname)

            case GameOver(winner_name=w):
                self.is_game_over = True
                self.winner = w
                logger.info("Match concluded! Winner: %s", w)

            case ErrorEvent(error_message=err):
                logger.warning("Server error event: %s", err)

            case _:
                pass

        return False

    def encode_observation(self) -> np.ndarray:
        """Construct the 78-dimensional Fog-of-War float32 observation vector.

        Matches the exact features and bounds expected by the neural policy and heuristic agent:
        - 18 top-card features (4 color one-hot, 13 value one-hot, 1 wild flag)
        - 54 hand features (52 standard card histogram normalized by 2.0, 2 wild counts by 4.0)
        - 3 opponent count profiles in cyclic relative seat order (normalized by 10.0)
        - 3 game dynamics features (direction, opponent ultimo alert, deck remaining ratio)
        """
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        offset = 0

        # 1. Top Card State (18 floats)
        if self.active_color is not None and self.active_color.is_standard_color():
            color_idx = STANDARD_SUITS.index(self.active_color)
            obs[offset + color_idx] = 1.0
        offset += 4

        if self.top_card is not None and not self.top_card.is_wild:
            if self.top_card.value in STANDARD_VALUES:
                val_idx = STANDARD_VALUES.index(self.top_card.value)
                obs[offset + val_idx] = 1.0
        offset += 13

        if self.top_card is not None and self.top_card.is_wild:
            obs[offset] = 1.0
        offset += 1

        # 2. Hand Histogram (54 floats)
        hand_counts = self.hand.card_counts()
        for card, count in hand_counts.items():
            if not card.is_wild:
                suit_idx = STANDARD_SUITS.index(card.color)
                val_idx = STANDARD_VALUES.index(card.value)
                slot = offset + (suit_idx * 13 + val_idx)
                obs[slot] = min(float(count) / 2.0, 1.0)
        offset += 52

        wild_count = sum(1 for c in self.hand.cards if c.value == CardValue.WILD)
        w4_count = sum(1 for c in self.hand.cards if c.value == CardValue.WILD_DRAW_FOUR)
        obs[offset] = min(float(wild_count) / 4.0, 1.0)
        obs[offset + 1] = min(float(w4_count) / 4.0, 1.0)
        offset += 2

        # 3. Opponent Hand Profiles (3 floats)
        # Order opponents cyclic from bot's position in player_names if known
        opp_names: list[str] = []
        if self.bot_name in self.player_names and len(self.player_names) > 1:
            my_idx = self.player_names.index(self.bot_name)
            for i in range(1, len(self.player_names)):
                opp_names.append(self.player_names[(my_idx + i) % len(self.player_names)])
        else:
            opp_names = [name for name in self.opponent_counts if name != self.bot_name]

        for i in range(3):
            if i < len(opp_names):
                opp_count = self.opponent_counts.get(opp_names[i], 7)
                obs[offset + i] = min(float(opp_count) / 10.0, 1.0)
            else:
                obs[offset + i] = 0.7  # Default estimated 7 cards
        offset += 3

        # 4. Game Dynamics (3 floats)
        # Clockwise direction scalar = 1.0
        obs[offset] = 1.0
        offset += 1

        # Opponent Ultimo warning: any opponent holds <= 1 card
        any_opp_ultimo = any(cnt <= 1 for cnt in self.opponent_counts.values())
        obs[offset] = 1.0 if any_opp_ultimo else 0.0
        offset += 1

        # Estimated deck remaining ratio
        obs[offset] = 0.5
        offset += 1

        assert offset == OBSERVATION_DIM, f"Expected {OBSERVATION_DIM} floats, got {offset}"
        return obs

    def compute_action_mask(self) -> np.ndarray:
        """Calculate bit-level legal action mask (np.int8, shape (61,))."""
        mask = compute_action_mask(
            hand=self.hand,
            top_card=self.top_card,
            active_color=self.active_color,
        )

        # If bot has already drawn a card this turn, it cannot draw again
        if self.has_drawn_this_turn:
            mask[ACTION_DRAW] = 0

        return mask
