"""Unit and integration test suite for Ultimo text wire protocol codec and TCP client."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.codec import ProtocolCodec
from ultimo_marl.protocol.events import (
    Chat,
    ChatBroadcast,
    ClientCommand,
    CommandType,
    Draw,
    DrawResult,
    ErrorEvent,
    EventType,
    GameOver,
    HandUpdate,
    Join,
    OpponentUpdate,
    Pass,
    Ping,
    Play,
    Pong,
    ProtocolException,
    RoomStatus,
    ServerEvent,
    Start,
    TopCard,
    TurnAnnounced,
    Welcome,
)

# ============================================================================
# Card Formatting & Parsing Tests
# ============================================================================


def test_card_formatting_standard() -> None:
    """Standard suit cards should format as COLOR_VALUE."""
    c1 = Card(CardColor.RED, CardValue.SEVEN)
    assert ProtocolCodec.format_card(c1) == "RED_SEVEN"

    c2 = Card(CardColor.BLUE, CardValue.SKIP)
    assert ProtocolCodec.format_card(c2) == "BLUE_SKIP"

    c3 = Card(CardColor.GREEN, CardValue.DRAW_TWO)
    assert ProtocolCodec.format_card(c3) == "GREEN_DRAW_TWO"


def test_card_formatting_wild() -> None:
    """Wild cards format directly as WILD or WILD_DRAW_FOUR."""
    w1 = Card(CardColor.WILD, CardValue.WILD)
    assert ProtocolCodec.format_card(w1) == "WILD"

    w2 = Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)
    assert ProtocolCodec.format_card(w2) == "WILD_DRAW_FOUR"


def test_card_parsing_variations() -> None:
    """Parser should handle numeric, textual, and shorthand representations."""
    assert ProtocolCodec.parse_card("RED_7") == Card(CardColor.RED, CardValue.SEVEN)
    assert ProtocolCodec.parse_card("RED_SEVEN") == Card(CardColor.RED, CardValue.SEVEN)
    assert ProtocolCodec.parse_card("blue_0") == Card(CardColor.BLUE, CardValue.ZERO)
    assert ProtocolCodec.parse_card("GREEN_DRAW2") == Card(CardColor.GREEN, CardValue.DRAW_TWO)
    assert ProtocolCodec.parse_card("GREEN_DRAW_2") == Card(CardColor.GREEN, CardValue.DRAW_TWO)
    assert ProtocolCodec.parse_card("WILD") == Card(CardColor.WILD, CardValue.WILD)
    assert ProtocolCodec.parse_card("WILD_WILD") == Card(CardColor.WILD, CardValue.WILD)
    assert ProtocolCodec.parse_card("WILD_DRAW_FOUR") == Card(
        CardColor.WILD, CardValue.WILD_DRAW_FOUR
    )
    assert ProtocolCodec.parse_card("WILD_DRAW4") == Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR)


def test_card_parsing_invalid() -> None:
    """Malformed card tokens must raise ProtocolException."""
    with pytest.raises(ProtocolException):
        ProtocolCodec.parse_card("")
    with pytest.raises(ProtocolException):
        ProtocolCodec.parse_card("   ")
    with pytest.raises(ProtocolException):
        ProtocolCodec.parse_card("INVALID")
    with pytest.raises(ProtocolException):
        ProtocolCodec.parse_card("PURPLE_SEVEN")
    with pytest.raises(ProtocolException):
        ProtocolCodec.parse_card("RED_UNKNOWN")


# ============================================================================
# Client Command Encoding & Decoding Round-Trip Tests
# ============================================================================


def test_encode_decode_join() -> None:
    cmd = Join(username="DeepMind_Bot")
    line = ProtocolCodec.encode_command(cmd)
    assert line == "JOIN DeepMind_Bot\n"

    decoded = ProtocolCodec.decode_command(line)
    assert decoded == cmd
    assert decoded.command_type == CommandType.JOIN


def test_encode_decode_play_single() -> None:
    card = Card(CardColor.YELLOW, CardValue.NINE)
    cmd = Play(cards=[card])
    assert cmd.card == card
    line = ProtocolCodec.encode_command(cmd)
    assert line == "PLAY YELLOW_NINE\n"

    decoded = ProtocolCodec.decode_command(line)
    assert decoded == cmd


def test_encode_decode_play_combo() -> None:
    c1 = Card(CardColor.RED, CardValue.FOUR)
    c2 = Card(CardColor.BLUE, CardValue.FOUR)
    cmd = Play(cards=[c1, c2])
    line = ProtocolCodec.encode_command(cmd)
    assert line == "PLAY RED_FOUR,BLUE_FOUR\n"

    decoded = ProtocolCodec.decode_command(line)
    assert decoded == cmd


def test_encode_decode_play_wild() -> None:
    wild = Card(CardColor.WILD, CardValue.WILD)
    cmd = Play(cards=[wild], chosen_wild_color=CardColor.GREEN)
    line = ProtocolCodec.encode_command(cmd)
    assert line == "PLAY WILD GREEN\n"

    decoded = ProtocolCodec.decode_command(line)
    assert decoded == cmd


def test_decode_play_wild_missing_color_raises() -> None:
    with pytest.raises(ProtocolException, match="Wild cards require a chosen color"):
        ProtocolCodec.decode_command("PLAY WILD\n")


def test_encode_decode_draw_pass_ping_start() -> None:
    commands: list[ClientCommand] = [Draw(), Pass(), Ping(), Start()]
    for cmd in commands:
        line = ProtocolCodec.encode_command(cmd)
        decoded = ProtocolCodec.decode_command(line)
        assert decoded == cmd


def test_encode_decode_chat() -> None:
    cmd = Chat(message="Good game everyone!")
    line = ProtocolCodec.encode_command(cmd)
    assert line == "CHAT Good game everyone!\n"

    decoded = ProtocolCodec.decode_command(line)
    assert decoded == cmd


def test_decode_command_invalid() -> None:
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_command("")
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_command("JOIN")
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_command("UNKNOWN_CMD")


# ============================================================================
# Server Event Encoding & Decoding Round-Trip Tests
# ============================================================================


def test_encode_decode_welcome() -> None:
    event = Welcome(assigned_player_id="player-99", message="Welcome to the room!")
    line = ProtocolCodec.encode_event(event)
    assert line == "WELCOME player-99 Welcome to the room!\n"

    decoded = ProtocolCodec.decode_event(line)
    assert decoded == event
    assert decoded.event_type == EventType.WELCOME


def test_encode_decode_room_status() -> None:
    event1 = RoomStatus(
        room_id="lobby-1",
        current_players=3,
        max_players=4,
        player_names=["Alice", "Bob", "Charlie"],
    )
    line1 = ProtocolCodec.encode_event(event1)
    assert line1 == "ROOM_STATUS lobby-1 3 4 Alice,Bob,Charlie\n"
    assert ProtocolCodec.decode_event(line1) == event1

    event2 = RoomStatus(room_id="lobby-2", current_players=0, max_players=4, player_names=[])
    line2 = ProtocolCodec.encode_event(event2)
    assert line2 == "ROOM_STATUS lobby-2 0 4 -\n"
    assert ProtocolCodec.decode_event(line2) == event2


def test_encode_decode_hand_update() -> None:
    cards = [
        Card(CardColor.RED, CardValue.ZERO),
        Card(CardColor.GREEN, CardValue.SKIP),
        Card(CardColor.WILD, CardValue.WILD),
    ]
    event = HandUpdate(cards=cards)
    line = ProtocolCodec.encode_event(event)
    assert line == "HAND_UPDATE RED_ZERO GREEN_SKIP WILD\n"

    decoded = ProtocolCodec.decode_event(line)
    assert decoded == event

    empty_hand = HandUpdate(cards=[])
    line_empty = ProtocolCodec.encode_event(empty_hand)
    assert line_empty == "HAND_UPDATE\n"
    assert ProtocolCodec.decode_event(line_empty) == empty_hand


def test_encode_decode_top_card() -> None:
    event = TopCard(card=Card(CardColor.BLUE, CardValue.TWO), active_color=CardColor.BLUE)
    line = ProtocolCodec.encode_event(event)
    assert line == "TOP_CARD BLUE_TWO BLUE\n"

    decoded = ProtocolCodec.decode_event(line)
    assert decoded == event


def test_encode_decode_turn_and_opponent() -> None:
    turn = TurnAnnounced(active_player_name="Bot_A")
    assert ProtocolCodec.decode_event(ProtocolCodec.encode_event(turn)) == turn

    opp = OpponentUpdate(opponent_name="Bot_B", card_count=3)
    assert ProtocolCodec.decode_event(ProtocolCodec.encode_event(opp)) == opp


def test_encode_decode_draw_result_error_chat_gameover_pong() -> None:
    events: list[ServerEvent] = [
        DrawResult(drawn_card=Card(CardColor.YELLOW, CardValue.FIVE)),
        ErrorEvent(error_message="Card cannot be played on current discard"),
        ChatBroadcast(sender="Charlie", message="Checkmate!"),
        GameOver(winner_name="Alice"),
        Pong(),
    ]
    for ev in events:
        line = ProtocolCodec.encode_event(ev)
        decoded = ProtocolCodec.decode_event(line)
        assert decoded == ev


def test_decode_event_malformed() -> None:
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_event("")
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_event("WELCOME")
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_event("ROOM_STATUS room1 1")
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_event("TOP_CARD RED_1")
    with pytest.raises(ProtocolException):
        ProtocolCodec.decode_event("UNKNOWN_EVENT 123")


# ============================================================================
# Live TCP Client Integration Tests
# ============================================================================


class MockServer:
    """Lightweight in-process TCP server for testing socket client resilience."""

    def __init__(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(("127.0.0.1", 0))
        self.server_sock.listen(5)
        self.host, self.port = self.server_sock.getsockname()
        self.received_lines: list[str] = []
        self._running = True
        self.client_conn: socket.socket | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                conn, _ = self.server_sock.accept()
                conn.settimeout(0.2)
                self.client_conn = conn
                buffer = ""
                while self._running:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            break
                        buffer += data.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                self.received_lines.append(line.strip())
                    except TimeoutError:
                        continue
                    except OSError:
                        break
            except OSError:
                break

    def drop_client(self) -> None:
        if self.client_conn:
            try:
                self.client_conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.client_conn.close()
            except Exception:
                pass
            self.client_conn = None

    def send_line(self, line: str) -> None:
        if self.client_conn:
            self.client_conn.sendall(line.encode("utf-8"))

    def close(self) -> None:
        self._running = False
        self.drop_client()
        try:
            self.server_sock.close()
        except Exception:
            pass


def test_tcp_client_lifecycle_and_chunking() -> None:
    """Test connection, fragmented TCP chunk accumulation, and event queueing."""
    server = MockServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port, connect_timeout=2.0)
        client.connect()
        assert client.is_connected

        # Send command to server
        client.send_command(Join(username="TestAgent"))
        time.sleep(0.1)
        assert "JOIN TestAgent" in server.received_lines

        # Server sends event in split chunks: "WEL" then "COME p-1 Welcome\n"
        server.send_line("WEL")
        time.sleep(0.05)
        server.send_line("COME p-1 Welcome\n")

        event = client.get_event(timeout=2.0)
        assert isinstance(event, Welcome)
        assert event.assigned_player_id == "p-1"
        assert event.message == "Welcome"

        # Server sends multiple events in single TCP transmission
        server.send_line("TOP_CARD RED_SEVEN RED\nTURN_ANNOUNCED TestAgent\n")

        ev1 = client.get_event(timeout=2.0)
        assert isinstance(ev1, TopCard)
        assert ev1.card == Card(CardColor.RED, CardValue.SEVEN)

        ev2 = client.get_event(timeout=2.0)
        assert isinstance(ev2, TurnAnnounced)
        assert ev2.active_player_name == "TestAgent"

        # Non-blocking poll returns None when queue empty
        assert client.poll_event() is None

        client.disconnect()
        assert not client.is_connected
    finally:
        server.close()


def test_tcp_client_handlers_and_ping() -> None:
    """Test event callbacks and ping transmission."""
    server = MockServer()
    try:
        with TcpGameClient(host=server.host, port=server.port) as client:
            received_game_overs: list[str] = []

            client.register_handler(
                EventType.GAME_OVER,
                lambda ev: received_game_overs.append(
                    ev.winner_name if isinstance(ev, GameOver) else ""
                ),
            )

            client.ping()
            time.sleep(0.1)
            assert "PING" in server.received_lines

            server.send_line("GAME_OVER Alice\n")
            time.sleep(0.1)
            assert received_game_overs == ["Alice"]

    finally:
        server.close()


def test_tcp_client_disconnected_send_raises() -> None:
    """Sending commands on a disconnected client must raise ConnectionError."""
    client = TcpGameClient(host="127.0.0.1", port=65432)
    with pytest.raises(ConnectionError):
        client.send_command(Pass())


def test_codec_unsupported_types() -> None:
    """Unsupported command/event objects must raise ProtocolException."""

    class FakeCommand(ClientCommand):
        pass

    with pytest.raises(ProtocolException, match="Unsupported client command type"):
        ProtocolCodec.encode_command(FakeCommand())

    class FakeEvent(ServerEvent):
        pass

    with pytest.raises(ProtocolException, match="Unsupported server event type"):
        ProtocolCodec.encode_event(FakeEvent())


def test_codec_command_decode_edges() -> None:
    """Edge cases in command decoding."""
    with pytest.raises(ProtocolException, match="JOIN command requires a username"):
        ProtocolCodec.decode_command("JOIN")

    with pytest.raises(ProtocolException, match="JOIN command requires a username"):
        ProtocolCodec.decode_command("JOIN    ")

    with pytest.raises(ProtocolException, match="PLAY command requires card"):
        ProtocolCodec.decode_command("PLAY")


def test_codec_event_decode_edges() -> None:
    """Edge cases in server event decoding."""
    with pytest.raises(ProtocolException, match="Malformed TURN_ANNOUNCED"):
        ProtocolCodec.decode_event("TURN_ANNOUNCED")

    with pytest.raises(ProtocolException, match="Malformed DRAW_RESULT"):
        ProtocolCodec.decode_event("DRAW_RESULT")

    with pytest.raises(ProtocolException, match="Malformed CHAT_BROADCAST"):
        ProtocolCodec.decode_event("CHAT_BROADCAST")

    chat_empty = ProtocolCodec.decode_event("CHAT_BROADCAST Alice")
    assert isinstance(chat_empty, ChatBroadcast)
    assert chat_empty.sender == "Alice"
    assert chat_empty.message == ""


def test_tcp_client_double_connect_and_handler_exception() -> None:
    """Test idempotent connect and resilient handler exception recovery."""
    server = MockServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port)
        client.connect()
        # Second connect should be no-op
        client.connect()
        assert client.is_connected

        # Register a faulty handler
        def faulty_handler(_ev: ServerEvent) -> None:
            raise RuntimeError("Handler failure!")

        client.register_handler(EventType.PONG, faulty_handler)

        # Sending pong should not crash the client
        server.send_line("PONG\n")
        time.sleep(0.1)
        assert client.is_connected

        client.disconnect()
    finally:
        server.close()


def test_tcp_client_auto_reconnect() -> None:
    """Test automatic exponential backoff reconnection when server drops connection."""
    server = MockServer()
    try:
        client = TcpGameClient(
            host=server.host,
            port=server.port,
            auto_reconnect=True,
            initial_backoff=0.05,
            max_backoff=0.1,
            max_reconnect_attempts=3,
        )
        client.connect()
        client.send_command(Join(username="ReconnectBot"))
        time.sleep(0.1)
        assert "JOIN ReconnectBot" in server.received_lines

        # Force close server client conn to simulate abrupt network drop
        server.drop_client()

        # Give client a moment to detect EOF, backoff and reconnect
        time.sleep(0.4)
        assert client.is_connected
        # Verify JOIN command was automatically re-sent upon reconnect
        assert server.received_lines.count("JOIN ReconnectBot") >= 2

        client.disconnect()
    finally:
        server.close()


def test_command_and_event_type_properties() -> None:
    """Verify command_type and event_type on all concrete classes and base errors."""
    with pytest.raises(NotImplementedError):
        _ = ClientCommand().command_type

    with pytest.raises(NotImplementedError):
        _ = ServerEvent().event_type

    commands = [
        (Join("test"), CommandType.JOIN),
        (Play([]), CommandType.PLAY),
        (Draw(), CommandType.DRAW),
        (Pass(), CommandType.PASS),
        (Chat("msg"), CommandType.CHAT),
        (Ping(), CommandType.PING),
        (Start(), CommandType.START),
    ]
    for cmd, expected_cmd_type in commands:
        assert cmd.command_type == expected_cmd_type

    events = [
        (Welcome("id"), EventType.WELCOME),
        (RoomStatus("r", 1, 4), EventType.ROOM_STATUS),
        (HandUpdate([]), EventType.HAND_UPDATE),
        (TopCard(Card(CardColor.RED, CardValue.ZERO), CardColor.RED), EventType.TOP_CARD),
        (TurnAnnounced("p"), EventType.TURN_ANNOUNCED),
        (OpponentUpdate("opp", 1), EventType.OPPONENT_UPDATE),
        (DrawResult(Card(CardColor.RED, CardValue.ZERO)), EventType.DRAW_RESULT),
        (ErrorEvent("err"), EventType.ERROR),
        (ChatBroadcast("p", "m"), EventType.CHAT_BROADCAST),
        (GameOver("p"), EventType.GAME_OVER),
        (Pong(), EventType.PONG),
    ]
    for ev, expected_ev_type in events:
        assert ev.event_type == expected_ev_type
