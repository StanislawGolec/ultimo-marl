"""Tests for 120-second reconnect window configuration, credential caching, and socket recovery."""

from __future__ import annotations

import socket
import threading
import time

from ultimo_marl.agents.heuristic_agent import RuleBasedHeuristicAgent
from ultimo_marl.bot import LiveBot
from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.events import Join


class _ReconnectMockServer:
    """In-process TCP server supporting forced socket drops and session re-attachment."""

    def __init__(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(("127.0.0.1", 0))
        self.server_sock.listen(5)
        self.host, self.port = self.server_sock.getsockname()
        self.received_lines: list[str] = []
        self.connection_count = 0
        self._running = True
        self.client_conn: socket.socket | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                conn, _ = self.server_sock.accept()
                conn.settimeout(0.2)
                with self._lock:
                    self.client_conn = conn
                    self.connection_count += 1
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

    def drop_active_client(self) -> None:
        """Abruptly close the active client connection to simulate a network drop."""
        with self._lock:
            if self.client_conn is not None:
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
        with self._lock:
            if self.client_conn is not None:
                self.client_conn.sendall(line.encode("utf-8"))

    def close(self) -> None:
        self._running = False
        self.drop_active_client()
        try:
            self.server_sock.close()
        except Exception:
            pass


def test_tcp_client_default_reconnect_window_parameters() -> None:
    """Verify TcpGameClient defaults cover the 120-second server disconnect grace window."""
    client = TcpGameClient()
    assert client.max_reconnect_attempts == 10
    assert client.initial_backoff == 0.5
    assert client.max_backoff == 8.0
    assert client.last_username is None
    assert client.last_room_id is None
    assert client.last_access_code is None


def test_tcp_client_reconnects_with_cached_credentials() -> None:
    """Verify socket drop triggers automatic reconnect and re-sends Join with cached credentials."""
    server = _ReconnectMockServer()
    try:
        client = TcpGameClient(
            host=server.host,
            port=server.port,
            auto_reconnect=True,
            initial_backoff=0.05,
            max_backoff=0.2,
            max_reconnect_attempts=10,
        )
        client.connect()
        client.send_command(
            Join(
                username="ResilientBot",
                room_id="private_room_7",
                access_code="CODE42",
            )
        )
        assert client.last_username == "ResilientBot"
        assert client.last_room_id == "private_room_7"
        assert client.last_access_code == "CODE42"

        time.sleep(0.1)
        assert "JOIN ResilientBot private_room_7 CODE42" in server.received_lines

        # Sever the TCP connection
        server.drop_active_client()

        # Wait for client to detect EOF, backoff, reconnect, and re-send JOIN
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if server.received_lines.count("JOIN ResilientBot private_room_7 CODE42") >= 2:
                break
            time.sleep(0.05)

        assert client.is_connected
        assert server.connection_count >= 2
        assert server.received_lines.count("JOIN ResilientBot private_room_7 CODE42") >= 2

        client.disconnect()
    finally:
        server.close()


def test_live_bot_snapshot_recovery_after_reconnect() -> None:
    """Simulate mid-match socket drop, reconnect, state snapshot recovery, and turn resumption."""
    server = _ReconnectMockServer()
    try:
        client = TcpGameClient(
            host=server.host,
            port=server.port,
            auto_reconnect=True,
            initial_backoff=0.05,
            max_backoff=0.2,
            max_reconnect_attempts=10,
        )
        bot = LiveBot(
            client=client,
            agent=RuleBasedHeuristicAgent(),
            bot_name="RecoveryBot",
        )
        client.connect()
        client.send_command(
            Join(
                username="RecoveryBot",
                room_id="match_room_9",
                access_code="VIP1",
            )
        )

        # Initial match state before disconnect
        server.send_line("WELCOME u-1 Connected\n")
        server.send_line("HAND_UPDATE BLUE_ONE GREEN_TWO\n")
        server.send_line("TOP_CARD YELLOW_NINE YELLOW\n")
        for _ in range(3):
            bot.step(timeout=0.2)

        assert len(bot.state.hand) == 2

        # Server drops connection mid-game
        server.drop_active_client()

        # Wait for automatic reconnect and credential re-dispatch
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if server.received_lines.count("JOIN RecoveryBot match_room_9 VIP1") >= 2:
                break
            time.sleep(0.05)

        assert server.received_lines.count("JOIN RecoveryBot match_room_9 VIP1") >= 2

        # Server sends authoritative state recovery snapshot upon re-attachment
        server.send_line(
            "HAND_UPDATE RED_SKIP BLUE_FIVE\n"
            "TOP_CARD RED_TWO RED\n"
            "OPPONENT_UPDATE RivalPlayer 3\n"
            "TURN_ANNOUNCED RecoveryBot\n"
        )

        for _ in range(4):
            bot.step(timeout=0.2)

        time.sleep(0.1)
        assert bot.state.top_card == Card(CardColor.RED, CardValue.TWO)
        assert bot.state.opponent_counts["RivalPlayer"] == 3
        assert any("PLAY RED_SKIP" in line for line in server.received_lines)

        client.disconnect()
    finally:
        server.close()
