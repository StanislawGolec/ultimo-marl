"""Resilient raw TCP client for the Ultimo game engine text wire protocol."""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from ultimo_marl.protocol.codec import ProtocolCodec
from ultimo_marl.protocol.events import (
    ClientCommand,
    EventType,
    Join,
    Ping,
    Pong,
    ProtocolException,
    ServerEvent,
)

logger = logging.getLogger(__name__)


class TcpGameClient:
    """Threaded, resilient raw TCP client for communicating with the Ultimo server.

    Features:
    - Zero-allocation line-delimited stream accumulator splitting on '\\n'.
    - Disables Nagle's algorithm (TCP_NODELAY) for ultra-low latency action dispatch.
    - Automatic exponential backoff reconnection on socket drops (Agent 08/09 resilience).
    - Thread-safe outbound command queuing and inbound event dispatching.
    - Automatic PING/PONG heartbeat management.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        connect_timeout: float = 5.0,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        initial_backoff: float = 0.2,
        max_backoff: float = 4.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier

        self._socket: socket.socket | None = None
        self._running = False
        self._connected = False
        self._write_lock = threading.Lock()
        self._event_queue: queue.Queue[ServerEvent] = queue.Queue()
        self._rx_thread: threading.Thread | None = None

        self._last_join_cmd: Join | None = None
        self._handlers: dict[EventType, list[Callable[[ServerEvent], None]]] = {}

    @property
    def is_connected(self) -> bool:
        """Return True if currently connected to the server."""
        return self._connected and self._socket is not None

    def register_handler(
        self,
        event_type: EventType,
        callback: Callable[[ServerEvent], None],
    ) -> None:
        """Register a callback invoked when a specific ServerEvent type is received."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(callback)

    def connect(self) -> None:
        """Establish TCP connection and start background receiver thread."""
        if self._connected:
            return

        self._connect_socket()
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._receive_loop,
            name=f"TcpClient-RX-{self.host}:{self.port}",
            daemon=True,
        )
        self._rx_thread.start()
        logger.info("Connected to Ultimo server at %s:%d", self.host, self.port)

    def _connect_socket(self) -> None:
        """Create and connect raw TCP stream socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((self.host, self.port))
            # Enable TCP_NODELAY to bypass Nagle's algorithm for real-time card plays
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Switch to blocking mode with timeout for subsequent operations
            sock.settimeout(1.0)
            self._socket = sock
            self._connected = True
        except Exception:
            sock.close()
            raise

    def disconnect(self) -> None:
        """Gracefully disconnect socket and stop receiver thread."""
        self._running = False
        self._connected = False

        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if self._rx_thread is not None and self._rx_thread.is_alive():
            if threading.current_thread() != self._rx_thread:
                self._rx_thread.join(timeout=2.0)
            self._rx_thread = None

        logger.info("Disconnected from Ultimo server %s:%d", self.host, self.port)

    def send_command(self, command: ClientCommand) -> None:
        """Encode and transmit a ClientCommand over the TCP wire."""
        if isinstance(command, Join):
            self._last_join_cmd = command

        wire_line = ProtocolCodec.encode_command(command)
        self.send_raw(wire_line)

    def send_raw(self, line: str) -> None:
        """Send raw line terminated by newline to the socket."""
        if not self._connected or self._socket is None:
            raise ConnectionError("Cannot send command: client is disconnected")

        if not line.endswith("\n"):
            line += "\n"

        payload = line.encode("utf-8")
        with self._write_lock:
            try:
                self._socket.sendall(payload)
            except OSError as err:
                logger.warning("Socket write failed: %s", err)
                self._handle_disconnect()
                raise ConnectionError(f"Failed to transmit data: {err}") from err

    def get_event(self, timeout: float | None = None) -> ServerEvent:
        """Wait for and retrieve the next ServerEvent from the incoming queue.

        Raises queue.Empty if timeout expires without an event.
        """
        return self._event_queue.get(block=True, timeout=timeout)

    def poll_event(self) -> ServerEvent | None:
        """Non-blocking poll for the next available ServerEvent."""
        try:
            return self._event_queue.get_nowait()
        except queue.Empty:
            return None

    def _receive_loop(self) -> None:
        """Background thread accumulating TCP stream chunks and dispatching events."""
        buffer = ""

        while self._running:
            if not self._connected or self._socket is None:
                if self.auto_reconnect:
                    reconnected = self._attempt_reconnect()
                    if not reconnected:
                        break
                    buffer = ""
                    continue
                break

            try:
                chunk = self._socket.recv(4096)
                if not chunk:
                    logger.warning("Server closed TCP connection (EOF received)")
                    self._handle_disconnect()
                    if self.auto_reconnect:
                        continue
                    break

                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    self._process_wire_line(line)

            except TimeoutError:
                # Normal read timeout, continue loop
                continue
            except OSError as err:
                if self._running:
                    logger.warning("Socket read error encountered: %s", err)
                    self._handle_disconnect()
                    if self.auto_reconnect:
                        continue
                break

        self._connected = False
        self._running = False

    def _process_wire_line(self, line: str) -> None:
        """Decode line into event, manage heartbeats, and notify handlers."""
        try:
            event = ProtocolCodec.decode_event(line)
        except ProtocolException as pe:
            logger.error("Failed to decode wire line '%s': %s", line, pe)
            return

        # Automatic heartbeat handling
        if isinstance(event, Pong):
            logger.debug("Received PONG heartbeat from server")

        # Put in FIFO event queue
        self._event_queue.put(event)

        # Notify registered callbacks
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as ex:
                    logger.exception("Error executing handler for %s: %s", event.event_type, ex)

    def _handle_disconnect(self) -> None:
        """Mark socket disconnected."""
        self._connected = False
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _attempt_reconnect(self) -> bool:
        """Attempt to re-establish connection using exponential backoff."""
        backoff = self.initial_backoff

        for attempt in range(1, self.max_reconnect_attempts + 1):
            if not self._running:
                return False

            logger.info(
                "Attempting TCP reconnect (%d/%d) to %s:%d in %.2fs...",
                attempt,
                self.max_reconnect_attempts,
                self.host,
                self.port,
                backoff,
            )
            time.sleep(backoff)

            try:
                self._connect_socket()
                logger.info("Successfully reconnected to %s:%d", self.host, self.port)

                # Automatically re-join lobby if previously registered
                if self._last_join_cmd is not None:
                    try:
                        self.send_command(self._last_join_cmd)
                        logger.info("Re-sent JOIN command for %s", self._last_join_cmd.username)
                    except Exception as err:
                        logger.error("Failed to re-send JOIN command: %s", err)

                return True
            except Exception as err:
                logger.warning("Reconnect attempt %d failed: %s", attempt, err)
                backoff = min(backoff * self.backoff_multiplier, self.max_backoff)

        logger.error("Exceeded maximum reconnect attempts (%d)", self.max_reconnect_attempts)
        return False

    def ping(self) -> None:
        """Send a PING heartbeat command to the server."""
        self.send_command(Ping())

    def __enter__(self) -> TcpGameClient:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
