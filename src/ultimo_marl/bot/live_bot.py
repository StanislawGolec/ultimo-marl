"""Production live TCP inference bot orchestrator."""

from __future__ import annotations

import logging
import queue
import re
import time
import uuid
from typing import TYPE_CHECKING

from ultimo_marl.agents.base import BaseAgent
from ultimo_marl.bot.live_game_state import LiveGameState
from ultimo_marl.domain.card import CardColor
from ultimo_marl.env.action_mapping import ACTION_DRAW, action_id_to_card_and_color
from ultimo_marl.protocol.client import TcpGameClient
from ultimo_marl.protocol.events import (
    ClientCommand,
    Draw,
    DrawResult,
    ErrorEvent,
    GameOver,
    Join,
    Pass,
    Play,
    RoomOwnerChanged,
    RoomStatus,
    ServerEvent,
    ServerShutdown,
    Start,
    TurnAnnounced,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{2,24}$")


def validate_bot_username(name: str) -> bool:
    """Validate that a bot username conforms to ^[a-zA-Z0-9_]{2,24}$."""
    return bool(USERNAME_REGEX.match(name))


def generate_fallback_name(name: str) -> str:
    """Generate a fallback username with a random 4-character hex suffix."""
    return f"{name}_{uuid.uuid4().hex[:4]}"


class LiveBot:
    """Autonomous live TCP inference bot playing Ultimo matches against mobile and engine players."""

    def __init__(
        self,
        client: TcpGameClient,
        agent: BaseAgent,
        bot_name: str,
        *,
        auto_start: bool = True,
        min_start_players: int = 2,
    ) -> None:
        self.client = client
        self.agent = agent
        self.base_name = bot_name
        self.bot_name = bot_name
        self.auto_start = auto_start
        self.min_start_players = min_start_players
        self.state = LiveGameState(bot_name=bot_name)
        self.is_running = False
        self.total_decisions = 0
        self.cumulative_latency_ms = 0.0

    @property
    def avg_decision_latency_ms(self) -> float:
        """Average action selection latency in milliseconds."""
        if self.total_decisions == 0:
            return 0.0
        return self.cumulative_latency_ms / self.total_decisions

    def handle_event(self, event: ServerEvent) -> ClientCommand | None:
        """Process an inbound event and return the appropriate reactive command, if any."""
        is_my_turn = self.state.apply_event(event)

        if is_my_turn:
            return self._decide_turn_action()

        if isinstance(event, DrawResult) and self.state.has_drawn_this_turn:
            return self._decide_post_draw_action()

        if isinstance(event, ErrorEvent) and "NAME_TAKEN" in event.error_message.upper():
            fallback_name = generate_fallback_name(self.base_name)
            logger.warning(
                "Username '%s' is already taken. Retrying with fallback '%s'",
                self.bot_name,
                fallback_name,
            )
            self.bot_name = fallback_name
            self.state.bot_name = fallback_name
            room_id = self.client.last_room_id or self.state.room_id
            return Join(
                username=fallback_name,
                room_id=room_id,
                access_code=self.client.last_access_code,
                max_players=self.client.last_max_players,
            )

        if isinstance(event, RoomStatus):
            if self.auto_start and event.current_players >= self.min_start_players:
                if self.state.is_room_owner or event.current_players == event.max_players:
                    logger.info(
                        "Auto-starting match (%d/%d players, is_room_owner=%s)",
                        event.current_players,
                        event.max_players,
                        self.state.is_room_owner,
                    )
                    return Start()

        if isinstance(event, RoomOwnerChanged):
            if (
                self.auto_start
                and self.state.is_room_owner
                and len(self.state.player_names) >= self.min_start_players
                and self.state.top_card is None
            ):
                logger.info(
                    "Promoted to room owner with %d players, auto-starting match",
                    len(self.state.player_names),
                )
                return Start()

        if isinstance(event, GameOver):
            logger.info(
                "Match finished! Winner: %s. Avg decision latency: %.2f ms",
                event.winner_name,
                self.avg_decision_latency_ms,
            )

        if isinstance(event, ServerShutdown):
            logger.warning(
                "Server is performing graceful shutdown (%ds drain). Disconnecting cleanly.",
                event.drain_seconds,
            )
            self.stop()
            self.client.disconnect()

        return None

    def _decide_turn_action(self) -> ClientCommand:
        """Perform neural or heuristic policy inference on bot's turn."""
        start_time = time.perf_counter()

        obs = self.state.encode_observation()
        mask = self.state.compute_action_mask()

        # If somehow no actions are legal (e.g. after drawing), pass turn
        if not mask.any():
            logger.info("No legal actions available, passing turn")
            return Pass()

        action = self.agent.select_action(obs, mask)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.total_decisions += 1
        self.cumulative_latency_ms += elapsed_ms

        logger.debug(
            "Selected action ID %d in %.2f ms (mask legal actions: %d)",
            action,
            elapsed_ms,
            int(mask.sum()),
        )

        if action == ACTION_DRAW:
            self.state.has_drawn_this_turn = True
            return Draw()

        card, wild_color = action_id_to_card_and_color(action)
        if card is None:
            return Draw()

        self.state.has_drawn_this_turn = False
        return Play(cards=[card], chosen_wild_color=wild_color or CardColor.RED)

    def _decide_post_draw_action(self) -> ClientCommand:
        """Decide whether to play the drawn card or pass turn."""
        mask = self.state.compute_action_mask()

        # Check if any standard/wild card play is legal
        if mask[:ACTION_DRAW].any():
            start_time = time.perf_counter()
            obs = self.state.encode_observation()
            action = self.agent.select_action(obs, mask)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.total_decisions += 1
            self.cumulative_latency_ms += elapsed_ms

            if action != ACTION_DRAW:
                card, wild_color = action_id_to_card_and_color(action)
                if card is not None:
                    self.state.has_drawn_this_turn = False
                    logger.info("Playing drawn card %s (chosen color: %s)", card, wild_color)
                    return Play(cards=[card], chosen_wild_color=wild_color or CardColor.RED)

        # Default to passing turn
        self.state.has_drawn_this_turn = False
        logger.debug("Passing turn post-draw")
        return Pass()

    def step(self, timeout: float = 0.1) -> ServerEvent | None:
        """Fetch and process a single event from the TCP client event queue."""
        try:
            event = self.client.get_event(timeout=timeout)
        except queue.Empty:
            return None

        cmd = self.handle_event(event)
        if cmd is not None:
            try:
                self.client.send_command(cmd)
            except ConnectionError as err:
                if not self.client.auto_reconnect:
                    raise
                logger.warning("Transient disconnect while sending %s: %s", cmd.command_type, err)

        return event

    def run(
        self,
        max_turns: int | None = None,
        stop_on_game_over: bool = False,
    ) -> None:
        """Run bot processing loop until stopped or connection terminates."""
        self.is_running = True
        logger.info("Live bot '%s' loop started", self.bot_name)

        turns = 0
        while self.is_running and (
            self.client.is_connected or (self.client.auto_reconnect and self.client._running)
        ):
            event = self.step(timeout=0.2)
            if event is None:
                continue

            if isinstance(event, TurnAnnounced) and event.active_player_name == self.bot_name:
                turns += 1
                if max_turns is not None and turns >= max_turns:
                    logger.info("Reached maximum turn limit (%d), stopping bot", max_turns)
                    break

            if isinstance(event, GameOver) and stop_on_game_over:
                logger.info("Game over encountered with stop_on_game_over=True, stopping bot")
                break

            if isinstance(event, ServerShutdown):
                break

        self.is_running = False
        logger.info("Live bot '%s' loop concluded", self.bot_name)

    def stop(self) -> None:
        """Signal bot loop to stop execution."""
        self.is_running = False
