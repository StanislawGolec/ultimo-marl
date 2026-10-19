"""Unit and integration tests for bot username validation and NAME_TAKEN collision recovery."""

from __future__ import annotations

import re
import socket
import threading
import time

import pytest

import run_live_bot
from ultimo_marl.agents.heuristic_agent import RuleBasedHeuristicAgent
from ultimo_marl.agents.random_agent import RandomAgent
from ultimo_marl.bot import (
    LiveBot,
    generate_fallback_name,
    validate_bot_username,
)
from ultimo_marl.bot import (
    run_live_bot as bot_cli,
)
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.events import ErrorEvent, Join


@pytest.mark.parametrize(
    "valid_name",
    [
        "Ab",
        "DeepUltimo_v1",
        "Bot_01",
        "A" * 24,
        "player_123_xyz",
    ],
)
def test_validate_bot_username_accepts_valid_names(valid_name: str) -> None:
    """Usernames with 2-24 alphanumeric or underscore characters must pass validation."""
    assert validate_bot_username(valid_name) is True


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "A",
        "A" * 25,
        "Bot With Spaces",
        "Bot-With-Hyphen",
        "Bot!@#",
        "Bot\nNewline",
        "Bot\tTab",
        "Bot🎲Emoji",
    ],
)
def test_validate_bot_username_rejects_invalid_names(invalid_name: str) -> None:
    """Usernames violating ^[a-zA-Z0-9_]{2,24}$ must be rejected."""
    assert validate_bot_username(invalid_name) is False


@pytest.mark.parametrize(
    "bad_cli_name",
    [
        "x",
        "too_long_username_exceeding_twenty_four_chars",
        "invalid space",
        "bad-char!",
    ],
)
def test_cli_parse_args_rejects_invalid_username(
    monkeypatch: pytest.MonkeyPatch,
    bad_cli_name: str,
) -> None:
    """CLI argument parser in both entrypoints must reject invalid --name values."""
    monkeypatch.setattr("sys.argv", ["run_live_bot.py", "--name", bad_cli_name])
    with pytest.raises(SystemExit):
        run_live_bot.parse_args()

    with pytest.raises(SystemExit):
        bot_cli.parse_args()


def test_cli_parse_args_accepts_valid_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI argument parser must accept valid alphanumeric and underscore usernames."""
    monkeypatch.setattr("sys.argv", ["run_live_bot.py", "--name", "AlphaBot_99"])
    args_root = run_live_bot.parse_args()
    assert args_root.name == "AlphaBot_99"

    args_pkg = bot_cli.parse_args()
    assert args_pkg.name == "AlphaBot_99"


def test_generate_fallback_name_format() -> None:
    """Fallback name generator must append a 4-character lowercase hex suffix."""
    base = "DeepUltimo"
    fallback = generate_fallback_name(base)
    assert re.match(r"^DeepUltimo_[0-9a-f]{4}$", fallback)
    assert validate_bot_username(fallback)


def test_live_bot_intercepts_name_taken_and_returns_fallback_join() -> None:
    """LiveBot must intercept ERROR NAME_TAKEN, update bot_name, and return a retry Join command."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    client.last_username = "DeepUltimo_v1"
    client.last_room_id = "arena-1"
    client.last_access_code = "PASS"

    bot = LiveBot(client=client, agent=RandomAgent(), bot_name="DeepUltimo_v1")
    error_ev = ErrorEvent(
        error_message="NAME_TAKEN: Player with name 'DeepUltimo_v1' already exists in this room"
    )

    cmd = bot.handle_event(error_ev)
    assert isinstance(cmd, Join)
    assert cmd.username != "DeepUltimo_v1"
    assert re.match(r"^DeepUltimo_v1_[0-9a-f]{4}$", cmd.username)
    assert cmd.room_id == "arena-1"
    assert cmd.access_code == "PASS"
    assert bot.bot_name == cmd.username
    assert bot.state.bot_name == cmd.username


class _CollisionMockServer:
    """In-process TCP mock server that rejects the first JOIN with NAME_TAKEN."""

    def __init__(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(("127.0.0.1", 0))
        self.server_sock.listen(5)
        self.host, self.port = self.server_sock.getsockname()
        self.received_lines: list[str] = []
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            conn, _ = self.server_sock.accept()
            conn.settimeout(0.2)
            buffer = ""
            join_attempts = 0
            accepted_name = "TakenBot"
            while self._running:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        self.received_lines.append(line)
                        if line.startswith("JOIN "):
                            join_attempts += 1
                            if join_attempts == 1:
                                conn.sendall(b"ERROR NAME_TAKEN: Player with name already exists\n")
                            else:
                                parts = line.split()
                                accepted_name = parts[1]
                                conn.sendall(f"WELCOME p-2 Joined as {accepted_name}\n".encode())
                                conn.sendall(
                                    f"HAND_UPDATE RED_SEVEN\n"
                                    f"TOP_CARD RED_THREE RED\n"
                                    f"TURN_ANNOUNCED {accepted_name}\n".encode()
                                )
                        elif line.startswith("PLAY "):
                            conn.sendall(f"GAME_OVER {accepted_name}\n".encode())
                except TimeoutError:
                    continue
                except OSError:
                    break
            conn.close()
        except OSError:
            pass

    def close(self) -> None:
        self._running = False
        try:
            self.server_sock.close()
        except Exception:
            pass


def test_live_bot_socket_name_taken_retry_e2e() -> None:
    """Verify end-to-end NAME_TAKEN recovery over TCP socket without crashing."""
    server = _CollisionMockServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port, auto_reconnect=False)
        bot = LiveBot(client=client, agent=RuleBasedHeuristicAgent(), bot_name="TakenBot")

        client.connect()
        client.send_command(Join(username="TakenBot", room_id="room-1"))

        bot.run(stop_on_game_over=True)

        join_cmds = [line for line in server.received_lines if line.startswith("JOIN ")]
        assert len(join_cmds) == 2
        assert join_cmds[0] == "JOIN TakenBot room-1"
        assert re.match(r"^JOIN TakenBot_[0-9a-f]{4} room-1$", join_cmds[1])
        assert any(line.startswith("PLAY RED_SEVEN") for line in server.received_lines)

        client.disconnect()
    finally:
        time.sleep(0.05)
        server.close()
