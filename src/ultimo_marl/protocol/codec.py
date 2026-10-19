"""Line-based text wire protocol codec for Ultimo game engine.

Converts between line-delimited ASCII wire representation (terminated with '\\n')
and strongly-typed domain commands and server events.
100% compatible with authoritative Java 21 Ultimo-Engine-Protocol.
"""

from __future__ import annotations

from typing import Any

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.protocol.events import (
    CatchUltimo,
    ChallengeResult,
    ChallengeWildFour,
    Chat,
    ChatBroadcast,
    ClientCommand,
    CommandType,
    CreateRoom,
    Draw,
    DrawResult,
    ErrorEvent,
    EventType,
    GameOver,
    HandUpdate,
    Join,
    Kick,
    ListRooms,
    OpponentUpdate,
    Pass,
    Ping,
    Play,
    Pong,
    ProtocolException,
    RoomList,
    RoomOwnerChanged,
    RoomStatus,
    RoundOver,
    ServerEvent,
    ServerShutdown,
    Start,
    TopCard,
    TransferOwnership,
    TurnAnnounced,
    Ultimo,
    UltimoDeclared,
    UltimoPenalty,
    Welcome,
)

# Mapping of string tokens to CardValue for resilient wire parsing
_STRING_TO_VALUE: dict[str, CardValue] = {
    "0": CardValue.ZERO,
    "ZERO": CardValue.ZERO,
    "1": CardValue.ONE,
    "ONE": CardValue.ONE,
    "2": CardValue.TWO,
    "TWO": CardValue.TWO,
    "3": CardValue.THREE,
    "THREE": CardValue.THREE,
    "4": CardValue.FOUR,
    "FOUR": CardValue.FOUR,
    "5": CardValue.FIVE,
    "FIVE": CardValue.FIVE,
    "6": CardValue.SIX,
    "SIX": CardValue.SIX,
    "7": CardValue.SEVEN,
    "SEVEN": CardValue.SEVEN,
    "8": CardValue.EIGHT,
    "EIGHT": CardValue.EIGHT,
    "9": CardValue.NINE,
    "NINE": CardValue.NINE,
    "SKIP": CardValue.SKIP,
    "REVERSE": CardValue.REVERSE,
    "DRAW_TWO": CardValue.DRAW_TWO,
    "DRAW2": CardValue.DRAW_TWO,
    "DRAW_2": CardValue.DRAW_TWO,
    "WILD": CardValue.WILD,
    "WILD_DRAW_FOUR": CardValue.WILD_DRAW_FOUR,
    "WILD_DRAW4": CardValue.WILD_DRAW_FOUR,
}


class ProtocolCodec:
    """Encodes and decodes line-based wire protocol messages."""

    # ------------------------------------------------------------------------
    # Card formatting and parsing
    # ------------------------------------------------------------------------

    @staticmethod
    def format_card(card: Card) -> str:
        """Format a Card instance into its wire representation.

        Wild cards are formatted as 'WILD' or 'WILD_DRAW_FOUR'.
        Standard suit cards are formatted as '<COLOR>_<VALUE>' (e.g. 'RED_SEVEN').
        """
        if card.value == CardValue.WILD:
            return "WILD"
        if card.value == CardValue.WILD_DRAW_FOUR:
            return "WILD_DRAW_FOUR"
        return f"{card.color.name}_{card.value.name}"

    @staticmethod
    def parse_card(token: str) -> Card:
        """Parse a card token from wire representation into a Card instance.

        Supports variations:
        - "WILD", "WILD_WILD" -> Card(WILD, WILD)
        - "WILD_DRAW_FOUR", "WILD_DRAW4", "WILD_WILD_DRAW_FOUR" -> Card(WILD, WILD_DRAW_FOUR)
        - "RED_7", "RED_SEVEN", "BLUE_SKIP", "GREEN_DRAW2", "YELLOW_DRAW_TWO"
        """
        if not token or not token.strip():
            raise ProtocolException("Empty card token")

        normalized = token.strip().upper()

        if normalized in ("WILD", "WILD_WILD"):
            return Card(CardColor.WILD, CardValue.WILD)
        if normalized in ("WILD_DRAW_FOUR", "WILD_DRAW4", "WILD_WILD_DRAW_FOUR"):
            return Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)

        separator = normalized.find("_")
        if separator <= 0 or separator >= len(normalized) - 1:
            raise ProtocolException(f"Invalid card format (expected COLOR_VALUE, got: '{token}')")

        color_part = normalized[:separator]
        value_part = normalized[separator + 1 :]

        color = ProtocolCodec.parse_color(color_part)
        value = ProtocolCodec.parse_value(value_part)

        return Card(color, value)

    @staticmethod
    def parse_color(token: str) -> CardColor:
        """Parse color token into CardColor enum."""
        normalized = token.strip().upper()
        try:
            return CardColor[normalized]
        except KeyError as err:
            raise ProtocolException(f"Unknown card color: '{token}'") from err

    @staticmethod
    def parse_value(token: str) -> CardValue:
        """Parse value token into CardValue enum."""
        normalized = token.strip().upper()
        if normalized in _STRING_TO_VALUE:
            return _STRING_TO_VALUE[normalized]
        raise ProtocolException(f"Unknown card value: '{token}'")

    # ------------------------------------------------------------------------
    # Client Commands (Encoding & Decoding)
    # ------------------------------------------------------------------------

    @staticmethod
    def encode_command(command: ClientCommand) -> str:
        """Serialize a ClientCommand into a wire-format string terminated by '\\n'."""
        match command:
            case Join(
                username=uname,
                room_id=rid,
                access_code=code,
                max_players=max_p,
            ):
                tokens = ["JOIN", uname]
                if rid is not None:
                    tokens.append(rid)
                if code is not None:
                    tokens.append(code)
                if max_p is not None:
                    tokens.append(str(max_p))
                return " ".join(tokens) + "\n"
            case Play(cards=cards, chosen_wild_color=wild_color):
                cards_str = ",".join(ProtocolCodec.format_card(c) for c in cards)
                has_wild = any(c.is_wild for c in cards)
                if has_wild and wild_color is not None:
                    return f"PLAY {cards_str} {wild_color.name}\n"
                return f"PLAY {cards_str}\n"
            case Draw():
                return "DRAW\n"
            case Pass():
                return "PASS\n"
            case Chat(message=msg):
                return f"CHAT {msg}\n"
            case Ping():
                return "PING\n"
            case Start():
                return "START\n"
            case Ultimo():
                return "ULTIMO\n"
            case CatchUltimo(target_username=tgt):
                return f"CATCH_ULTIMO {tgt}\n"
            case ChallengeWildFour():
                return "CHALLENGE_WILD_FOUR\n"
            case CreateRoom(
                name=rname,
                is_public=pub,
                access_code=code,
                max_players=max_p,
            ):
                visibility = "PUBLIC" if pub else "PRIVATE"
                tokens = ["CREATE_ROOM", rname, visibility]
                if code is not None:
                    tokens.append(code)
                if max_p is not None:
                    tokens.append(str(max_p))
                return " ".join(tokens) + "\n"
            case ListRooms():
                return "LIST_ROOMS\n"
            case Kick(target_username=tgt):
                return f"KICK {tgt}\n"
            case TransferOwnership(target_username=tgt):
                return f"TRANSFER_OWNERSHIP {tgt}\n"
            case _:
                raise ProtocolException(f"Unsupported client command type: {type(command)}")

    @staticmethod
    def decode_command(raw_line: str) -> ClientCommand:
        """Parse a raw wire line into a strongly typed ClientCommand."""
        if not raw_line or not raw_line.strip():
            raise ProtocolException("Empty or null command line received")

        line = raw_line.strip()
        parts = line.split()
        command_name = parts[0].upper()

        try:
            if command_name == CommandType.JOIN.value:
                if len(parts) < 2:
                    raise ProtocolException(
                        "JOIN command requires a username: JOIN <username> [room_id] [access_code] [max_players]"
                    )
                username = parts[1]
                room_id: str | None = None
                access_code: str | None = None
                max_players: int | None = None

                if len(parts) == 3:
                    if parts[2].isdigit() and 2 <= int(parts[2]) <= 16:
                        max_players = int(parts[2])
                    else:
                        room_id = parts[2]
                elif len(parts) == 4:
                    room_id = parts[2]
                    if parts[3].isdigit() and 2 <= int(parts[3]) <= 16:
                        max_players = int(parts[3])
                    else:
                        access_code = parts[3]
                elif len(parts) >= 5:
                    room_id = parts[2]
                    access_code = parts[3]
                    max_players = int(parts[4])

                return Join(
                    username=username,
                    room_id=room_id,
                    access_code=access_code,
                    max_players=max_players,
                )

            if command_name == CommandType.PLAY.value:
                if len(parts) < 2:
                    raise ProtocolException("PLAY command requires card: PLAY <card> [chosenColor]")

                chosen_wild_color: CardColor | None = None
                card_end_index = len(parts)

                # Check if last token is a declared standard suit color
                if len(parts) > 2:
                    try:
                        candidate = ProtocolCodec.parse_color(parts[-1])
                        if candidate.is_standard_color():
                            chosen_wild_color = candidate
                            card_end_index = len(parts) - 1
                    except ProtocolException:
                        pass  # Not a color, part of card tokens

                cards: list[Card] = []
                for i in range(1, card_end_index):
                    token = parts[i]
                    if "," in token:
                        for sub_token in token.split(","):
                            if sub_token.strip():
                                cards.append(ProtocolCodec.parse_card(sub_token.strip()))
                    else:
                        cards.append(ProtocolCodec.parse_card(token))

                if not cards:
                    raise ProtocolException("PLAY command requires card: PLAY <card> [chosenColor]")

                has_wild = any(c.is_wild for c in cards)
                if has_wild and chosen_wild_color is None:
                    raise ProtocolException(
                        f"Wild cards require a chosen color: PLAY {parts[1]} <RED|YELLOW|GREEN|BLUE>"
                    )

                return Play(cards=cards, chosen_wild_color=chosen_wild_color)

            if command_name == CommandType.DRAW.value:
                return Draw()

            if command_name == CommandType.PASS.value:
                return Pass()

            if command_name == CommandType.CHAT.value:
                message = line[line.index(" ") + 1 :].strip() if len(parts) > 1 else ""
                return Chat(message=message)

            if command_name == CommandType.PING.value:
                return Ping()

            if command_name == CommandType.START.value:
                return Start()

            if command_name == CommandType.ULTIMO.value:
                return Ultimo()

            if command_name == CommandType.CATCH_ULTIMO.value:
                if len(parts) < 2:
                    raise ProtocolException(
                        "CATCH_ULTIMO requires target username: CATCH_ULTIMO <username>"
                    )
                return CatchUltimo(target_username=parts[1])

            if command_name == CommandType.CHALLENGE_WILD_FOUR.value:
                return ChallengeWildFour()

            if command_name == CommandType.CREATE_ROOM.value:
                if len(parts) < 3:
                    raise ProtocolException(
                        "CREATE_ROOM requires name and visibility: CREATE_ROOM <name> <PUBLIC|PRIVATE> [code] [max]"
                    )
                name = parts[1]
                vis_str = parts[2].upper()
                if vis_str not in ("PUBLIC", "PRIVATE"):
                    raise ProtocolException(
                        f"CREATE_ROOM visibility must be PUBLIC or PRIVATE, got: '{parts[2]}'"
                    )
                is_public = vis_str == "PUBLIC"
                cr_code: str | None = None
                cr_max = 4
                rest = parts[3:]
                if len(rest) == 1:
                    if rest[0].isdigit() and (is_public or 2 <= int(rest[0]) <= 16):
                        cr_max = int(rest[0])
                    else:
                        cr_code = rest[0]
                elif len(rest) >= 2:
                    cr_code = rest[0]
                    cr_max = int(rest[1])
                return CreateRoom(
                    name=name,
                    is_public=is_public,
                    access_code=cr_code,
                    max_players=cr_max,
                )

            if command_name == CommandType.LIST_ROOMS.value:
                return ListRooms()

            if command_name == CommandType.KICK.value:
                if len(parts) < 2:
                    raise ProtocolException("KICK requires target username: KICK <username>")
                return Kick(target_username=parts[1])

            if command_name == CommandType.TRANSFER_OWNERSHIP.value:
                if len(parts) < 2:
                    raise ProtocolException(
                        "TRANSFER_OWNERSHIP requires target username: TRANSFER_OWNERSHIP <username>"
                    )
                return TransferOwnership(target_username=parts[1])

            raise ProtocolException(f"Unknown command: '{command_name}'")

        except ProtocolException:
            raise
        except Exception as e:
            raise ProtocolException(f"Failed to decode command '{line}': {e}") from e

    # ------------------------------------------------------------------------
    # Server Events (Encoding & Decoding)
    # ------------------------------------------------------------------------

    @staticmethod
    def _parse_room_token(token: str) -> dict[str, Any]:
        """Parse a single room descriptor token from ROOM_LIST."""
        segments = token.split(":")
        if len(segments) >= 4:
            if "/" in segments[2]:
                cur_str, max_str = segments[2].split("/", 1)
                return {
                    "room_id": segments[0],
                    "name": segments[1],
                    "current_players": int(cur_str),
                    "max_players": int(max_str),
                    "status": segments[3],
                }
            result: dict[str, Any] = {
                "room_id": segments[0],
                "name": segments[1],
                "current_players": int(segments[2]),
                "max_players": int(segments[3]),
            }
            if len(segments) >= 5:
                result["status"] = segments[4]
            return result
        if len(segments) == 3:
            if "/" in segments[2]:
                cur_str, max_str = segments[2].split("/", 1)
                return {
                    "room_id": segments[0],
                    "name": segments[1],
                    "current_players": int(cur_str),
                    "max_players": int(max_str),
                }
            return {
                "room_id": segments[0],
                "current_players": int(segments[1]),
                "max_players": int(segments[2]),
            }
        if len(segments) == 2 and "/" in segments[1]:
            cur_str, max_str = segments[1].split("/", 1)
            return {
                "room_id": segments[0],
                "current_players": int(cur_str),
                "max_players": int(max_str),
            }
        return {"room_id": token}

    @staticmethod
    def encode_event(event: ServerEvent) -> str:
        """Serialize a ServerEvent into a wire-format string terminated by '\\n'."""
        match event:
            case Welcome(assigned_player_id=pid, message=msg):
                return f"WELCOME {pid} {msg}\n"
            case RoomStatus(
                room_id=rid,
                current_players=cur,
                max_players=max_p,
                player_names=names,
            ):
                plist = ",".join(names) if names else "-"
                return f"ROOM_STATUS {rid} {cur} {max_p} {plist}\n"
            case HandUpdate(cards=cards):
                if not cards:
                    return "HAND_UPDATE\n"
                cards_str = " ".join(ProtocolCodec.format_card(c) for c in cards)
                return f"HAND_UPDATE {cards_str}\n"
            case TopCard(card=c, active_color=color):
                return f"TOP_CARD {ProtocolCodec.format_card(c)} {color.name}\n"
            case TurnAnnounced(active_player_name=pname):
                return f"TURN_ANNOUNCED {pname}\n"
            case OpponentUpdate(opponent_name=oname, card_count=cnt):
                return f"OPPONENT_UPDATE {oname} {cnt}\n"
            case DrawResult(drawn_card=c):
                return f"DRAW_RESULT {ProtocolCodec.format_card(c)}\n"
            case ErrorEvent(error_message=err):
                return f"ERROR {err}\n"
            case ChatBroadcast(sender=s, message=m):
                return f"CHAT_BROADCAST {s} {m}\n"
            case GameOver(winner_name=w):
                return f"GAME_OVER {w}\n"
            case Pong():
                return "PONG\n"
            case ServerShutdown(drain_seconds=drain):
                return f"SERVER_SHUTDOWN {drain}\n"
            case UltimoDeclared(player_name=pname):
                return f"ULTIMO_DECLARED {pname}\n"
            case UltimoPenalty(
                penalized_player=penalized,
                challenger=challenger,
                penalty_cards=cards_cnt,
            ):
                return f"ULTIMO_PENALTY {penalized} {challenger} {cards_cnt}\n"
            case ChallengeResult(
                challenger=challenger,
                offender=offender,
                is_guilty=guilty,
                penalized_player=penalized,
                penalty_cards=cards_cnt,
            ):
                g_str = "GUILTY" if guilty else "INNOCENT"
                return f"CHALLENGE_RESULT {challenger} {offender} {g_str} {penalized} {cards_cnt}\n"
            case RoundOver(
                round_winner=winner,
                points_earned=pts,
                is_tournament_won=won,
                tournament_scores=scores,
            ):
                w_str = "WON" if won else "ONGOING"
                s_str = ",".join(f"{k}:{v}" for k, v in scores.items()) if scores else "-"
                return f"ROUND_OVER {winner} {pts} {w_str} {s_str}\n"
            case RoomList(rooms=rooms):
                if not rooms:
                    return "ROOM_LIST 0\n"
                encoded_rooms: list[str] = []
                for r in rooms:
                    rid = str(r.get("room_id", r.get("id", "unknown")))
                    cur = int(r.get("current_players", 0))
                    max_p = int(r.get("max_players", 4))
                    if "status" in r:
                        rname = str(r.get("name", rid)).replace(":", "_").replace(" ", "_")
                        status = str(r["status"])
                        encoded_rooms.append(f"{rid}:{rname}:{cur}/{max_p}:{status}")
                    elif "name" in r:
                        rname = str(r["name"]).replace(":", "_").replace(" ", "_")
                        encoded_rooms.append(f"{rid}:{rname}:{cur}/{max_p}")
                    else:
                        encoded_rooms.append(f"{rid}:{cur}/{max_p}")
                return f"ROOM_LIST {len(rooms)} {' '.join(encoded_rooms)}\n"
            case RoomOwnerChanged(new_owner_username=new_owner):
                return f"ROOM_OWNER_CHANGED {new_owner}\n"
            case _:
                raise ProtocolException(f"Unsupported server event type: {type(event)}")

    @staticmethod
    def decode_event(raw_line: str) -> ServerEvent:
        """Parse a raw wire line into a strongly typed ServerEvent."""
        if not raw_line or not raw_line.strip():
            raise ProtocolException("Empty or null event line received")

        line = raw_line.strip()
        parts = line.split()
        event_name = parts[0].upper()

        try:
            if event_name == EventType.WELCOME.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed WELCOME event: '{line}'")
                pid = parts[1]
                msg = line[line.find(parts[2]) :].strip() if len(parts) > 2 else ""
                return Welcome(assigned_player_id=pid, message=msg)

            if event_name == EventType.ROOM_STATUS.value:
                if len(parts) < 4:
                    raise ProtocolException(f"Malformed ROOM_STATUS event: '{line}'")
                room_id = parts[1]
                current = int(parts[2])
                max_p = int(parts[3])
                names: list[str] = []
                if len(parts) > 4 and parts[4] != "-":
                    names = [p for p in parts[4].split(",") if p.strip()]
                return RoomStatus(
                    room_id=room_id,
                    current_players=current,
                    max_players=max_p,
                    player_names=names,
                )

            if event_name == EventType.HAND_UPDATE.value:
                cards = [ProtocolCodec.parse_card(p) for p in parts[1:]]
                return HandUpdate(cards=cards)

            if event_name == EventType.TOP_CARD.value:
                if len(parts) < 3:
                    raise ProtocolException(f"Malformed TOP_CARD event: '{line}'")
                card = ProtocolCodec.parse_card(parts[1])
                color = ProtocolCodec.parse_color(parts[2])
                return TopCard(card=card, active_color=color)

            if event_name == EventType.TURN_ANNOUNCED.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed TURN_ANNOUNCED event: '{line}'")
                return TurnAnnounced(active_player_name=parts[1])

            if event_name == EventType.OPPONENT_UPDATE.value:
                if len(parts) < 3:
                    raise ProtocolException(f"Malformed OPPONENT_UPDATE event: '{line}'")
                return OpponentUpdate(opponent_name=parts[1], card_count=int(parts[2]))

            if event_name == EventType.DRAW_RESULT.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed DRAW_RESULT event: '{line}'")
                return DrawResult(drawn_card=ProtocolCodec.parse_card(parts[1]))

            if event_name == EventType.ERROR.value:
                msg = line[line.index(" ") + 1 :].strip() if len(parts) > 1 else "Unknown error"
                return ErrorEvent(error_message=msg)

            if event_name == EventType.CHAT_BROADCAST.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed CHAT_BROADCAST event: '{line}'")
                sender = parts[1]
                msg = line[line.find(parts[2]) :].strip() if len(parts) > 2 else ""
                return ChatBroadcast(sender=sender, message=msg)

            if event_name == EventType.GAME_OVER.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed GAME_OVER event: '{line}'")
                return GameOver(winner_name=parts[1])

            if event_name == EventType.PONG.value:
                return Pong()

            if event_name == EventType.SERVER_SHUTDOWN.value:
                drain = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                return ServerShutdown(drain_seconds=drain)

            if event_name == EventType.ULTIMO_DECLARED.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed ULTIMO_DECLARED event: '{line}'")
                return UltimoDeclared(player_name=parts[1])

            if event_name == EventType.ULTIMO_PENALTY.value:
                if len(parts) < 4:
                    raise ProtocolException(f"Malformed ULTIMO_PENALTY event: '{line}'")
                return UltimoPenalty(
                    penalized_player=parts[1],
                    challenger=parts[2],
                    penalty_cards=int(parts[3]),
                )

            if event_name == EventType.CHALLENGE_RESULT.value:
                if len(parts) < 6:
                    raise ProtocolException(f"Malformed CHALLENGE_RESULT event: '{line}'")
                return ChallengeResult(
                    challenger=parts[1],
                    offender=parts[2],
                    is_guilty=(parts[3].upper() == "GUILTY"),
                    penalized_player=parts[4],
                    penalty_cards=int(parts[5]),
                )

            if event_name == EventType.ROUND_OVER.value:
                if len(parts) < 4:
                    raise ProtocolException(f"Malformed ROUND_OVER event: '{line}'")
                winner = parts[1]
                points = int(parts[2])
                won = parts[3].upper() == "WON"
                scores: dict[str, int] = {}
                if len(parts) > 4 and parts[4] != "-":
                    for pair in parts[4].split(","):
                        kv = pair.split(":")
                        if len(kv) == 2:
                            scores[kv[0]] = int(kv[1])
                return RoundOver(
                    round_winner=winner,
                    points_earned=points,
                    is_tournament_won=won,
                    tournament_scores=scores,
                )

            if event_name == EventType.ROOM_LIST.value:
                if len(parts) <= 1 or parts[1] == "-":
                    return RoomList(rooms=[])
                if parts[1].isdigit():
                    count = int(parts[1])
                    if count == 0:
                        return RoomList(rooms=[])
                    room_tokens = [t for t in parts[2:] if t != "-"]
                else:
                    room_tokens = [t for t in parts[1:] if t != "-"]
                rooms = [ProtocolCodec._parse_room_token(t) for t in room_tokens]
                return RoomList(rooms=rooms)

            if event_name == EventType.ROOM_OWNER_CHANGED.value:
                if len(parts) < 2:
                    raise ProtocolException(f"Malformed ROOM_OWNER_CHANGED event: '{line}'")
                return RoomOwnerChanged(new_owner_username=parts[1])

            raise ProtocolException(f"Unknown server event: '{event_name}'")

        except ProtocolException:
            raise
        except Exception as e:
            raise ProtocolException(f"Failed to decode event '{line}': {e}") from e
