"""Comprehensive test suite for LiveGameState, LiveBot, and CLI live inference runner."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ultimo_marl.agents.heuristic_agent import RuleBasedHeuristicAgent
from ultimo_marl.agents.random_agent import RandomAgent
from ultimo_marl.bot import LiveBot, LiveGameState, create_bot_agent
from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.domain.player import PlayerHand
from ultimo_marl.env.action_mapping import ACTION_DRAW
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.events import (
    DrawResult,
    ErrorEvent,
    EventType,
    GameOver,
    HandUpdate,
    Join,
    OpponentUpdate,
    Pass,
    Play,
    RoomStatus,
    ServerEvent,
    ServerShutdown,
    Start,
    TopCard,
    TurnAnnounced,
    Welcome,
)

# ============================================================================
# 1. LiveGameState Unit Tests
# ============================================================================


def test_live_game_state_event_processing() -> None:
    """Test state mutations across wire protocol server events."""
    state = LiveGameState(bot_name="BotAlpha")

    # Welcome
    assert not state.apply_event(Welcome(assigned_player_id="UUID-123", message="Welcome"))
    assert state.player_id == "UUID-123"

    # RoomStatus
    assert not state.apply_event(
        RoomStatus(
            room_id="room-42",
            current_players=3,
            max_players=4,
            player_names=["Alice", "BotAlpha", "Charlie"],
        )
    )
    assert state.room_id == "room-42"
    assert state.player_names == ["Alice", "BotAlpha", "Charlie"]
    assert "Alice" in state.opponent_counts
    assert "Charlie" in state.opponent_counts

    # HandUpdate
    cards = [
        Card(CardColor.RED, CardValue.SEVEN),
        Card(CardColor.BLUE, CardValue.SKIP),
        Card(CardColor.WILD, CardValue.WILD),
    ]
    assert not state.apply_event(HandUpdate(cards=cards))
    assert len(state.hand) == 3

    # TopCard
    top = Card(CardColor.RED, CardValue.THREE)
    assert not state.apply_event(TopCard(card=top, active_color=CardColor.RED))
    assert state.top_card == top
    assert state.active_color == CardColor.RED

    # OpponentUpdate
    assert not state.apply_event(OpponentUpdate(opponent_name="Alice", card_count=2))
    assert state.opponent_counts["Alice"] == 2

    # DrawResult
    drawn = Card(CardColor.GREEN, CardValue.NINE)
    assert not state.apply_event(DrawResult(drawn_card=drawn))
    assert len(state.hand) == 4
    assert state.has_drawn_this_turn is True

    # ErrorEvent
    assert not state.apply_event(ErrorEvent(error_message="Action rejected"))

    # TurnAnnounced for another player -> returns False
    res_charlie = state.apply_event(TurnAnnounced(active_player_name="Charlie"))
    assert res_charlie is False
    assert state.current_turn_player == "Charlie"

    # TurnAnnounced for bot -> returns True!
    res_bot = state.apply_event(TurnAnnounced(active_player_name="BotAlpha"))
    assert bool(res_bot) is True
    assert state.current_turn_player == "BotAlpha"
    assert bool(state.has_drawn_this_turn) is False

    # GameOver
    res_game_over = state.apply_event(GameOver(winner_name="BotAlpha"))
    assert res_game_over is False
    assert state.is_game_over
    assert state.winner == "BotAlpha"


def test_live_game_state_observation_and_mask() -> None:
    """Verify observation vector bounds and action mask shape."""
    state = LiveGameState(bot_name="BotAlpha")
    state.apply_event(
        RoomStatus(
            room_id="r1",
            current_players=2,
            max_players=2,
            player_names=["BotAlpha", "Bob"],
        )
    )
    state.apply_event(
        HandUpdate(
            cards=[
                Card(CardColor.RED, CardValue.ZERO),
                Card(CardColor.WILD, CardValue.WILD_DRAW_FOUR),
            ]
        )
    )
    state.apply_event(
        TopCard(
            card=Card(CardColor.RED, CardValue.FIVE),
            active_color=CardColor.RED,
        )
    )

    obs = state.encode_observation()
    assert obs.shape == (78,)
    assert obs.dtype == np.float32
    assert (obs >= 0.0).all() and (obs <= 1.0).all()

    mask = state.compute_action_mask()
    assert mask.shape == (61,)
    assert mask.dtype == np.int8
    assert mask.sum() >= 1

    # Drawing disables ACTION_DRAW in subsequent action mask calculation
    state.has_drawn_this_turn = True
    mask_drawn = state.compute_action_mask()
    assert mask_drawn[ACTION_DRAW] == 0


# ============================================================================
# 2. Agent Factory Tests
# ============================================================================


def test_create_bot_agent() -> None:
    """Verify create_bot_agent instantiates the intended agent types."""
    rand_agent = create_bot_agent("random")
    assert isinstance(rand_agent, RandomAgent)

    heur_agent = create_bot_agent("heuristic")
    assert isinstance(heur_agent, RuleBasedHeuristicAgent)

    dqn_agent = create_bot_agent("dqn")
    assert dqn_agent.name == "DQNAgent"

    with pytest.raises(ValueError, match="Unknown bot strategy"):
        create_bot_agent("invalid_strategy")


# ============================================================================
# 3. LiveBot Decision Logic Tests
# ============================================================================


def test_live_bot_auto_start() -> None:
    """When auto_start is True and room reaches max capacity, bot emits START command."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    bot = LiveBot(client=client, agent=RandomAgent(), bot_name="BotAlpha", auto_start=True)

    # Room not full yet
    event_partial = RoomStatus(
        room_id="r1",
        current_players=2,
        max_players=4,
        player_names=["p1", "p2"],
    )
    assert bot.handle_event(event_partial) is None

    # Room full
    event_full = RoomStatus(
        room_id="r1",
        current_players=4,
        max_players=4,
        player_names=["p1", "p2", "p3", "p4"],
    )
    cmd = bot.handle_event(event_full)
    assert isinstance(cmd, Start)


def test_live_bot_turn_decision_play_or_draw() -> None:
    """When bot turn is announced, bot selects a legal card play or draws."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    agent = RuleBasedHeuristicAgent()
    bot = LiveBot(client=client, agent=agent, bot_name="BotAlpha")

    # Set up hand holding RED_SEVEN and table having RED_THREE
    bot.state.hand = PlayerHand([Card(CardColor.RED, CardValue.SEVEN)])
    bot.state.top_card = Card(CardColor.RED, CardValue.THREE)
    bot.state.active_color = CardColor.RED

    turn_ev = TurnAnnounced(active_player_name="BotAlpha")
    cmd = bot.handle_event(turn_ev)

    assert isinstance(cmd, Play)
    assert cmd.card == Card(CardColor.RED, CardValue.SEVEN)
    assert bot.total_decisions == 1
    assert bot.avg_decision_latency_ms >= 0.0


def test_live_bot_post_draw_playable_card() -> None:
    """When bot draws a card that matches the discard pile, it plays it immediately."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    bot = LiveBot(client=client, agent=RuleBasedHeuristicAgent(), bot_name="BotAlpha")

    bot.state.hand = PlayerHand([])
    bot.state.top_card = Card(CardColor.BLUE, CardValue.ZERO)
    bot.state.active_color = CardColor.BLUE
    bot.state.has_drawn_this_turn = True

    # Drawn card matches table color
    drawn_card = Card(CardColor.BLUE, CardValue.FIVE)
    cmd = bot.handle_event(DrawResult(drawn_card=drawn_card))

    assert isinstance(cmd, Play)
    assert cmd.card == drawn_card


def test_live_bot_post_draw_unplayable_card_passes() -> None:
    """When bot draws an unplayable card, it passes the turn."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    bot = LiveBot(client=client, agent=RuleBasedHeuristicAgent(), bot_name="BotAlpha")

    bot.state.hand = PlayerHand([])
    bot.state.top_card = Card(CardColor.BLUE, CardValue.ZERO)
    bot.state.active_color = CardColor.BLUE
    bot.state.has_drawn_this_turn = True

    # Drawn card does not match color or value
    drawn_card = Card(CardColor.RED, CardValue.FIVE)
    cmd = bot.handle_event(DrawResult(drawn_card=drawn_card))

    assert isinstance(cmd, Pass)


# ============================================================================
# 4. Live TCP Bot End-to-End Match Integration Test
# ============================================================================


class MockGameServer:
    """In-process mock server simulating a match session with the bot."""

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

    def send_line(self, line: str) -> None:
        if self.client_conn:
            self.client_conn.sendall(line.encode("utf-8"))

    def close(self) -> None:
        self._running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except Exception:
                pass
        try:
            self.server_sock.close()
        except Exception:
            pass


def test_live_bot_e2e_match_simulation() -> None:
    """Simulate a complete match loop from connection through GAME_OVER."""
    server = MockGameServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port)
        agent = RuleBasedHeuristicAgent()
        bot = LiveBot(client=client, agent=agent, bot_name="DeepUltimo")

        client.connect()
        # Verify registration
        client.send_command(Join("DeepUltimo"))

        # Server sends welcome and game setup
        server.send_line("WELCOME bot-uuid MatchStarting\n")
        server.send_line("ROOM_STATUS room-101 2 2 DeepUltimo,HumanPlayer\n")
        server.send_line("HAND_UPDATE RED_SEVEN GREEN_ZERO BLUE_SKIP\n")
        server.send_line("TOP_CARD RED_NINE RED\n")
        server.send_line("TURN_ANNOUNCED DeepUltimo\n")

        # Process events through bot
        for _ in range(5):
            bot.step(timeout=0.2)

        time.sleep(0.1)
        # DeepUltimo should have played RED_SEVEN
        assert any("PLAY RED_SEVEN" in line for line in server.received_lines)

        # Server announces game over
        server.send_line("GAME_OVER DeepUltimo\n")
        bot.step(timeout=0.2)

        assert bot.state.is_game_over
        assert bot.state.winner == "DeepUltimo"
        assert bot.avg_decision_latency_ms < 5.0  # Ultra-fast <5ms decision requirement!

        client.disconnect()
    finally:
        server.close()


def test_live_bot_run_loop_and_termination() -> None:
    """Test LiveBot run loop with max_turns and stop_on_game_over."""
    server = MockGameServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port)
        agent = RandomAgent()
        bot = LiveBot(client=client, agent=agent, bot_name="RunnerBot")
        client.connect()

        # Send an event that triggers GameOver
        server.send_line("GAME_OVER RunnerBot\n")
        time.sleep(0.1)

        # Run with stop_on_game_over=True should exit immediately upon seeing GAME_OVER
        bot.run(stop_on_game_over=True)
        assert not bot.is_running
        assert bot.state.is_game_over

        client.disconnect()
    finally:
        server.close()


def test_live_bot_empty_mask_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fallback when no legal actions exist, and zero latency property."""
    client = TcpGameClient(host="127.0.0.1", port=65000)
    bot = LiveBot(client=client, agent=RandomAgent(), bot_name="BotAlpha")

    # Initial latency should be 0.0
    assert bot.avg_decision_latency_ms == 0.0

    # Monkeypatch mask to all zeros
    monkeypatch.setattr(bot.state, "compute_action_mask", lambda: np.zeros(61, dtype=np.int8))
    cmd = bot.handle_event(TurnAnnounced(active_player_name="BotAlpha"))
    assert isinstance(cmd, Pass)


def test_create_bot_agent_checkpoint(tmp_path: Path) -> None:
    """Verify DQNAgent creation with a saved model checkpoint."""
    import torch

    from ultimo_marl.training.q_network import DuelingQNetwork

    ckpt_file = tmp_path / "agent.pt"
    net = DuelingQNetwork()
    torch.save(net.state_dict(), ckpt_file)

    agent = create_bot_agent("dqn", model_path=ckpt_file)
    assert agent.name == "DQNAgent"


def test_run_live_bot_cli_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parse_args in run_live_bot.py."""
    import run_live_bot

    test_args = [
        "run_live_bot.py",
        "--host",
        "10.0.0.1",
        "--port",
        "9000",
        "--name",
        "TestRunnerBot",
        "--room",
        "tournament_room_alpha",
        "--strategy",
        "heuristic",
        "--single-game",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = run_live_bot.parse_args()

    assert parsed.host == "10.0.0.1"
    assert parsed.port == 9000
    assert parsed.name == "TestRunnerBot"
    assert parsed.room == "tournament_room_alpha"
    assert parsed.strategy == "heuristic"
    assert parsed.single_game is True

    from ultimo_marl.bot import run_live_bot as bot_runner

    parsed_mod = bot_runner.parse_args()
    assert parsed_mod.room == "tournament_room_alpha"

    # Test main() failure exit code 1
    def mock_connect_fail(self: Any) -> None:
        raise ConnectionError("Mock fail")

    monkeypatch.setattr("ultimo_marl.bot.run_live_bot.TcpGameClient.connect", mock_connect_fail)
    assert bot_runner.main() == 1


def test_live_bot_server_shutdown_handling() -> None:
    """Test LiveBot clean disconnection and loop exit upon ServerShutdown event."""
    server = MockGameServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port)
        agent = RandomAgent()
        bot = LiveBot(client=client, agent=agent, bot_name="ShutdownBot")
        client.connect()

        # Send SERVER_SHUTDOWN event from server
        server.send_line("SERVER_SHUTDOWN 3\n")
        time.sleep(0.1)

        # Run bot; it should receive SERVER_SHUTDOWN and exit cleanly
        bot.run()
        assert not bot.is_running
        assert not client.is_connected
    finally:
        server.close()


def test_run_live_bot_server_shutdown_exit() -> None:
    """Test run_live_bot ServerShutdown handler triggers clean disconnect and exit 0."""
    server = MockGameServer()
    try:
        client = TcpGameClient(host=server.host, port=server.port)
        agent = RandomAgent()
        bot = LiveBot(client=client, agent=agent, bot_name="ExitBot")
        client.connect()

        exited = False

        def mock_exit(code: int) -> None:
            nonlocal exited
            exited = code == 0

        import sys

        old_exit = sys.exit
        sys.exit = mock_exit  # type: ignore[assignment]
        try:

            def shutdown_handler(event: ServerEvent) -> None:
                if isinstance(event, ServerShutdown):
                    bot.stop()
                    client.disconnect()
                    sys.exit(0)

            client.register_handler(EventType.SERVER_SHUTDOWN, shutdown_handler)
            server.send_line("SERVER_SHUTDOWN 5\n")
            time.sleep(0.15)
            assert exited is True
            assert not client.is_connected
        finally:
            sys.exit = old_exit
    finally:
        server.close()
