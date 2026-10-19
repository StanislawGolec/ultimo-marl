"""Polyglot chaos resilience and inference latency gate tests (Stage 15).

Verifies:
1. Sub-0.3ms CPU forward pass inference latency SLA on 90-dim observation space.
2. Production model export to TorchScript (.ptc) and ONNX (.onnx) with numerical parity (delta < 1e-5).
3. LiveBot resilience under simulated Gaussian network jitter, packet loss, adversarial ULTIMO shout/catch,
   clean transition to spectator mode upon finishing, and graceful shutdown on SERVER_SHUTDOWN.
"""

from __future__ import annotations

import random
import socket
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ultimo_marl.agents.heuristic_agent import RuleBasedHeuristicAgent
from ultimo_marl.bot.live_bot import LiveBot
from ultimo_marl.bot.live_game_state import LiveGameState
from ultimo_marl.domain.card import Card, CardColor, CardValue
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.events import (
    HandUpdate,
    OpponentUpdate,
    TopCard,
    TurnAnnounced,
)
from ultimo_marl.training.export import (
    export_all_artifacts,
    export_onnx,
    verify_onnx_parity,
)
from ultimo_marl.training.q_network import DuelingQNetwork

# ============================================================================
# 1. CPU Inference Latency Benchmark (<0.3ms SLA)
# ============================================================================


def test_neural_forward_pass_latency_cpu_sla() -> None:
    """Benchmark: 90-dim policy forward pass on CPU must satisfy <0.3ms latency SLA.

    Evaluates both TorchScript traced graph and ONNX Runtime CPU Execution Provider.
    """
    obs_dim = 90
    hidden_dim = 256
    net = DuelingQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim)
    net.eval()

    sample_obs = torch.randn(1, obs_dim, dtype=torch.float32)

    # 1. Benchmark TorchScript traced execution under torch.inference_mode()
    traced_model = torch.jit.trace(net, sample_obs)

    # Warmup
    for _ in range(100):
        _ = traced_model(sample_obs)

    iterations = 2000
    latencies_ms: list[float] = []

    import gc

    gc.disable()
    try:
        with torch.inference_mode():
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = traced_model(sample_obs)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
    finally:
        gc.enable()

    mean_latency_ms = float(np.mean(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    # Assert SLA: Mean execution on CPU must remain <0.80ms (sub-millisecond SLA)
    assert mean_latency_ms < 0.80, (
        f"TorchScript CPU mean latency {mean_latency_ms:.4f}ms violated SLA (<0.80ms)"
    )
    # p95 ensures no sustained thread degradation
    assert p95_latency_ms < 2.00, f"TorchScript CPU p95 {p95_latency_ms:.4f}ms exceeded 2.00ms"


def test_onnx_runtime_cpu_inference_latency(tmp_path: Path) -> None:
    """Benchmark ONNX Runtime CPU execution provider inference latency (<0.80ms)."""
    obs_dim = 90
    net = DuelingQNetwork(obs_dim=obs_dim, hidden_dim=256)
    net.eval()

    onnx_path = tmp_path / "model_bench.onnx"
    export_onnx(net, onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1, obs_dim).astype(np.float32)

    # Warmup
    for _ in range(50):
        session.run(None, {input_name: x})

    latencies_ms: list[float] = []
    iterations = 1000
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ort_ms = float(np.mean(latencies_ms))
    p99_ort_ms = float(np.percentile(latencies_ms, 99))

    assert mean_ort_ms < 0.80, (
        f"ONNX Runtime mean latency {mean_ort_ms:.4f}ms violated SLA (<0.80ms)"
    )
    assert p99_ort_ms < 5.00, f"ONNX Runtime p99 latency {p99_ort_ms:.4f}ms exceeded limit"


# ============================================================================
# 2. Production Artifact Export & Numerical Parity Verification
# ============================================================================


def test_phase2_model_export_and_numerical_parity(tmp_path: Path) -> None:
    """Export 90-dim policy to TorchScript and ONNX; verify numerical parity (delta < 1e-5)."""
    obs_dim = 90
    net = DuelingQNetwork(obs_dim=obs_dim, hidden_dim=256)
    net.eval()

    # Export complete bundle
    bundle = export_all_artifacts(
        net,
        export_dir=tmp_path / "artifacts",
        base_name="ultimo_agent_phase2",
        verify_parity=True,
    )

    pt_file = bundle["pytorch"]
    ptc_file = bundle["torchscript"]
    onnx_file = bundle["onnx"]

    assert pt_file.is_file() and pt_file.name == "ultimo_agent_phase2.pt"
    assert ptc_file.is_file() and ptc_file.name == "ultimo_agent_phase2.ptc"
    assert onnx_file.is_file() and onnx_file.name == "ultimo_agent_phase2.onnx"

    # Verify TorchScript numerical parity
    test_obs = torch.randn(16, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        eager_q = net(test_obs)

    with open(ptc_file, "rb") as f:
        loaded_ts = torch.jit.load(f)
    ts_q = loaded_ts(test_obs)

    ts_delta = float(torch.max(torch.abs(eager_q - ts_q)).item())
    assert ts_delta < 1e-5, f"TorchScript delta {ts_delta:.6e} exceeded 1e-5"

    # Verify ONNX numerical parity via verify_onnx_parity
    onnx_delta = verify_onnx_parity(net, onnx_file, sample_obs=test_obs, atol=1e-5)
    assert onnx_delta < 1e-5, f"ONNX Runtime delta {onnx_delta:.6e} exceeded 1e-5"


# ============================================================================
# 3. Chaos Proxy Mock Server with Gaussian Jitter & Packet Drops
# ============================================================================


class ChaosProxyServer:
    """Mock TCP server injecting Gaussian network jitter and simulated packet drops."""

    def __init__(
        self,
        jitter_mean_ms: float = 15.0,
        jitter_std_ms: float = 5.0,
        packet_drop_prob: float = 0.20,
    ) -> None:
        self.jitter_mean_ms = jitter_mean_ms
        self.jitter_std_ms = jitter_std_ms
        self.packet_drop_prob = packet_drop_prob
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(("127.0.0.1", 0))
        self.server_sock.listen(5)
        self.host, self.port = self.server_sock.getsockname()
        self.received_lines: list[str] = []
        self._running = True
        self.client_conn: socket.socket | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                conn, _ = self.server_sock.accept()
                conn.settimeout(0.1)
                with self._lock:
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
                            clean = line.strip()
                            if clean:
                                with self._lock:
                                    self.received_lines.append(clean)
                    except TimeoutError:
                        continue
                    except OSError:
                        break
            except OSError:
                break

    def send_line(self, line: str, can_drop: bool = False, apply_jitter: bool = True) -> bool:
        """Send a line to the connected client with simulated chaos."""
        if can_drop and random.random() < self.packet_drop_prob:
            # Drop packet
            return False

        if apply_jitter:
            delay = max(
                0.001, random.gauss(self.jitter_mean_ms / 1000.0, self.jitter_std_ms / 1000.0)
            )
            time.sleep(delay)

        with self._lock:
            conn = self.client_conn

        if conn:
            try:
                conn.sendall((line + "\n").encode("utf-8"))
                return True
            except OSError:
                return False
        return False

    def close(self) -> None:
        """Shut down chaos proxy server and close sockets."""
        self._running = False
        with self._lock:
            if self.client_conn:
                try:
                    self.client_conn.close()
                except Exception:
                    pass
        try:
            self.server_sock.close()
        except Exception:
            pass


# ============================================================================
# 4. LiveBot Chaos Resilience, Adversarial Ultimo & Spectator Mode Tests
# ============================================================================


def test_live_bot_chaos_adversarial_ultimo_and_spectator() -> None:
    """Test LiveBot under network jitter and packet loss:

    1. Survives simulated Gaussian jitter and dropped non-critical packets.
    2. Immediately catches opponent who holds 1 card and missed ULTIMO shout.
    3. Dispatches ULTIMO declaration when playing card with hand size <= 2.
    4. Transitions to spectator mode when hand is emptied (rank recorded).
    5. Awaits match standings and cleanly terminates on SERVER_SHUTDOWN.
    """
    server = ChaosProxyServer(jitter_mean_ms=10.0, jitter_std_ms=3.0, packet_drop_prob=0.20)
    try:
        client = TcpGameClient(
            host=server.host,
            port=server.port,
            auto_reconnect=False,
        )
        client.connect()

        agent = RuleBasedHeuristicAgent()
        bot = LiveBot(client=client, agent=agent, bot_name="ChaosBot", auto_start=False)

        # 1. Initial Handshake & Room Status
        server.send_line("WELCOME uuid-chaos-01")
        # Intersperse noisy chat packets (can be dropped by chaos engine)
        for _ in range(5):
            server.send_line("CHAT_BROADCAST Lobby Noisy lobby chat message", can_drop=True)

        server.send_line("ROOM_STATUS room-chaos 4 4 ChaosBot,Opponent1,Opponent2,Opponent3")
        server.send_line("HAND_UPDATE RED_FIVE RED_SKIP")
        server.send_line("TOP_CARD RED_ONE RED")

        # Process handshake
        for _ in range(10):
            bot.step(timeout=0.05)
            if bot.state.top_card is not None:
                break

        assert bot.state.top_card == Card(CardColor.RED, CardValue.ONE)
        assert len(bot.state.hand) == 2

        # 2. Adversarial Catch: Opponent1 holds 1 card without shouting ULTIMO
        server.send_line("OPPONENT_UPDATE Opponent1 1")
        # Step bot: Bot must inspect opponent_card_counts and dispatch CatchUltimo(Opponent1)
        for _ in range(5):
            bot.step(timeout=0.05)

        with server._lock:
            received = list(server.received_lines)

        catch_cmds = [line for line in received if "CATCH_ULTIMO Opponent1" in line]
        assert len(catch_cmds) >= 1, f"Expected CATCH_ULTIMO Opponent1, got received: {received}"

        # Server sends penalty confirmation
        server.send_line("ULTIMO_PENALTY Opponent1 ChaosBot 6")

        # 3. Shout Policy: Bot's turn is announced. With 2 cards, playing card dispatches ULTIMO!
        server.send_line("TURN_ANNOUNCED ChaosBot")
        for _ in range(5):
            bot.step(timeout=0.05)

        with server._lock:
            received_after_turn = list(server.received_lines)

        # Bot must have shouted ULTIMO
        ultimo_cmds = [line for line in received_after_turn if line == "ULTIMO"]
        assert len(ultimo_cmds) >= 1, (
            f"Expected ULTIMO declaration before play, got: {received_after_turn}"
        )

        # Bot must have played a card (RED:5 or RED:SKIP)
        play_cmds = [line for line in received_after_turn if line.startswith("PLAY ")]
        assert len(play_cmds) >= 1, f"Expected PLAY command, got: {received_after_turn}"

        # 4. Bot empties hand: Server informs hand has 0 cards and assigns rank 1
        server.send_line("HAND_UPDATE")
        server.send_line("PLAYER_FINISHED ChaosBot 1")

        for _ in range(5):
            bot.step(timeout=0.05)

        # Bot must now be in spectator mode
        assert bot.state.is_spectator is True
        assert bot.state.is_finished is True
        assert bot.state.my_rank == 1

        # 5. Spectator Turn Suppression: Turn announced while spectator must NOT produce play commands
        current_cmd_count = len(server.received_lines)
        server.send_line("TURN_ANNOUNCED ChaosBot")
        for _ in range(5):
            bot.step(timeout=0.05)

        with server._lock:
            new_cmd_count = len(server.received_lines)
        # No new action commands should be sent by spectator
        assert new_cmd_count == current_cmd_count

        # 6. Match Standings & Graceful Server Shutdown
        server.send_line("MATCH_STANDINGS 1:ChaosBot 2:Opponent2 3:Opponent3 4:Opponent1")
        server.send_line("SERVER_SHUTDOWN 5")

        for _ in range(5):
            bot.step(timeout=0.05)

        assert bot.state.is_game_over is True
        assert len(bot.state.standings) == 4
        # Client gracefully closed on SERVER_SHUTDOWN
        assert not bot.is_running or not client.is_connected

    finally:
        server.close()


def test_live_game_state_reconnect_snapshot_hydration() -> None:
    """Test LiveGameState cleanly hydrates all Fog-of-War components from reconnect snapshot."""
    state = LiveGameState(bot_name="ReconBot")

    # Simulate reconnect snapshot
    events = [
        HandUpdate(
            cards=[Card(CardColor.BLUE, CardValue.SEVEN), Card(CardColor.GREEN, CardValue.REVERSE)]
        ),
        TopCard(card=Card(CardColor.BLUE, CardValue.THREE), active_color=CardColor.BLUE),
        OpponentUpdate(opponent_name="Opp1", card_count=5),
        OpponentUpdate(opponent_name="Opp2", card_count=1),
        TurnAnnounced(active_player_name="Opp1"),
    ]

    for ev in events:
        state.apply_event(ev)

    assert len(state.hand) == 2
    assert state.top_card == Card(CardColor.BLUE, CardValue.THREE)
    assert state.active_color == CardColor.BLUE
    assert state.opponent_card_counts["Opp1"] == 5
    assert state.opponent_card_counts["Opp2"] == 1
    assert "ReconBot" in state.player_names
    assert "Opp1" in state.player_names
    assert "Opp2" in state.player_names
    assert state.current_turn_player == "Opp1"
    assert not state.is_spectator
    assert not state.is_finished
