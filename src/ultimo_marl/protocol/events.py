"""Strongly-typed protocol command and event definitions for Ultimo text wire protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ultimo_marl.domain.card import Card, CardColor


class ProtocolError(Exception):
    """Raised when protocol encoding, decoding, or wire transmission fails."""


ProtocolException = ProtocolError


class CommandType(StrEnum):
    """Commands initiated by client and transmitted over TCP to server."""

    JOIN = "JOIN"
    PLAY = "PLAY"
    DRAW = "DRAW"
    PASS = "PASS"
    CHAT = "CHAT"
    PING = "PING"
    START = "START"
    ULTIMO = "ULTIMO"
    CATCH_ULTIMO = "CATCH_ULTIMO"
    CHALLENGE_WILD_FOUR = "CHALLENGE_WILD_FOUR"


class EventType(StrEnum):
    """Events emitted by authoritative server to clients."""

    WELCOME = "WELCOME"
    ROOM_STATUS = "ROOM_STATUS"
    HAND_UPDATE = "HAND_UPDATE"
    TOP_CARD = "TOP_CARD"
    TURN_ANNOUNCED = "TURN_ANNOUNCED"
    OPPONENT_UPDATE = "OPPONENT_UPDATE"
    DRAW_RESULT = "DRAW_RESULT"
    ERROR = "ERROR"
    CHAT_BROADCAST = "CHAT_BROADCAST"
    GAME_OVER = "GAME_OVER"
    PONG = "PONG"
    SERVER_SHUTDOWN = "SERVER_SHUTDOWN"
    ULTIMO_DECLARED = "ULTIMO_DECLARED"
    ULTIMO_PENALTY = "ULTIMO_PENALTY"
    CHALLENGE_RESULT = "CHALLENGE_RESULT"
    ROUND_OVER = "ROUND_OVER"


# ============================================================================
# Client Commands (Outbound from Agent to Server)
# ============================================================================


@dataclass(frozen=True)
class ClientCommand:
    """Base class for all outbound client commands."""

    @property
    def command_type(self) -> CommandType:
        raise NotImplementedError


@dataclass(frozen=True)
class Join(ClientCommand):
    """Register player in room, optionally specifying target room ID."""

    username: str
    room_id: str | None = None

    @property
    def command_type(self) -> CommandType:
        return CommandType.JOIN


@dataclass(frozen=True)
class Play(ClientCommand):
    """Play one or more cards to the discard pile."""

    cards: list[Card]
    chosen_wild_color: CardColor | None = None

    @property
    def command_type(self) -> CommandType:
        return CommandType.PLAY

    @property
    def card(self) -> Card | None:
        return self.cards[0] if self.cards else None


@dataclass(frozen=True)
class Draw(ClientCommand):
    """Draw a card from the deck."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.DRAW


@dataclass(frozen=True)
class Pass(ClientCommand):
    """Pass turn after drawing a card."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.PASS


@dataclass(frozen=True)
class Chat(ClientCommand):
    """Broadcast text chat message to room."""

    message: str

    @property
    def command_type(self) -> CommandType:
        return CommandType.CHAT


@dataclass(frozen=True)
class Ping(ClientCommand):
    """Heartbeat ping checking connection liveness."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.PING


@dataclass(frozen=True)
class Start(ClientCommand):
    """Host command to initiate match."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.START


@dataclass(frozen=True)
class Ultimo(ClientCommand):
    """Declare 'Ultimo!' when holding one card."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.ULTIMO


@dataclass(frozen=True)
class CatchUltimo(ClientCommand):
    """Catch an opponent who failed to declare 'Ultimo!'."""

    target_username: str

    @property
    def command_type(self) -> CommandType:
        return CommandType.CATCH_ULTIMO


@dataclass(frozen=True)
class ChallengeWildFour(ClientCommand):
    """Challenge whether a played Wild Draw Four card was legally played."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.CHALLENGE_WILD_FOUR


# ============================================================================
# Server Events (Inbound from Server to Agent)
# ============================================================================


@dataclass(frozen=True)
class ServerEvent:
    """Base class for all inbound server notifications and state updates."""

    @property
    def event_type(self) -> EventType:
        raise NotImplementedError


@dataclass(frozen=True)
class Welcome(ServerEvent):
    """Initial connection acknowledgment assigning client UUID/ID."""

    assigned_player_id: str
    message: str = ""

    @property
    def event_type(self) -> EventType:
        return EventType.WELCOME


@dataclass(frozen=True)
class RoomStatus(ServerEvent):
    """Lobby occupancy and player list metadata."""

    room_id: str
    current_players: int
    max_players: int
    player_names: list[str] = field(default_factory=list)

    @property
    def event_type(self) -> EventType:
        return EventType.ROOM_STATUS


@dataclass(frozen=True)
class HandUpdate(ServerEvent):
    """Private hand state synchronization."""

    cards: list[Card] = field(default_factory=list)

    @property
    def event_type(self) -> EventType:
        return EventType.HAND_UPDATE


@dataclass(frozen=True)
class TopCard(ServerEvent):
    """Active top card and declared wild color on discard pile."""

    card: Card
    active_color: CardColor

    @property
    def event_type(self) -> EventType:
        return EventType.TOP_CARD


@dataclass(frozen=True)
class TurnAnnounced(ServerEvent):
    """Turn rotation notification indicating which player must act."""

    active_player_name: str

    @property
    def event_type(self) -> EventType:
        return EventType.TURN_ANNOUNCED


@dataclass(frozen=True)
class OpponentUpdate(ServerEvent):
    """Public card count update for an opponent."""

    opponent_name: str
    card_count: int

    @property
    def event_type(self) -> EventType:
        return EventType.OPPONENT_UPDATE


@dataclass(frozen=True)
class DrawResult(ServerEvent):
    """Private notification of the card drawn from the deck."""

    drawn_card: Card

    @property
    def event_type(self) -> EventType:
        return EventType.DRAW_RESULT


@dataclass(frozen=True)
class ErrorEvent(ServerEvent):
    """Server-side validation rejection or exception."""

    error_message: str

    @property
    def event_type(self) -> EventType:
        return EventType.ERROR


@dataclass(frozen=True)
class ChatBroadcast(ServerEvent):
    """In-game chat message from player."""

    sender: str
    message: str

    @property
    def event_type(self) -> EventType:
        return EventType.CHAT_BROADCAST


@dataclass(frozen=True)
class GameOver(ServerEvent):
    """Match terminal outcome announcing round winner."""

    winner_name: str

    @property
    def event_type(self) -> EventType:
        return EventType.GAME_OVER


@dataclass(frozen=True)
class Pong(ServerEvent):
    """Server heartbeat pong response."""

    @property
    def event_type(self) -> EventType:
        return EventType.PONG


@dataclass(frozen=True)
class ServerShutdown(ServerEvent):
    """Server notifies clients that graceful shutdown / draining has initiated."""

    drain_seconds: int = 5

    @property
    def event_type(self) -> EventType:
        return EventType.SERVER_SHUTDOWN


@dataclass(frozen=True)
class UltimoDeclared(ServerEvent):
    """Broadcast indicating a player declared 'Ultimo!'."""

    player_name: str

    @property
    def event_type(self) -> EventType:
        return EventType.ULTIMO_DECLARED


@dataclass(frozen=True)
class UltimoPenalty(ServerEvent):
    """Broadcast indicating a player was penalized for failing to declare 'Ultimo!'."""

    penalized_player: str
    challenger: str
    penalty_cards: int

    @property
    def event_type(self) -> EventType:
        return EventType.ULTIMO_PENALTY


@dataclass(frozen=True)
class ChallengeResult(ServerEvent):
    """Broadcast resolution of a Wild Draw Four challenge."""

    challenger: str
    offender: str
    is_guilty: bool
    penalized_player: str
    penalty_cards: int

    @property
    def event_type(self) -> EventType:
        return EventType.CHALLENGE_RESULT


@dataclass(frozen=True)
class RoundOver(ServerEvent):
    """Broadcast indicating a single tournament round concluded."""

    round_winner: str
    points_earned: int
    is_tournament_won: bool
    tournament_scores: dict[str, int] = field(default_factory=dict)

    @property
    def event_type(self) -> EventType:
        return EventType.ROUND_OVER
