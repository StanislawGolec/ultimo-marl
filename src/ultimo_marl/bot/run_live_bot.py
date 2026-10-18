#!/usr/bin/env python3
"""CLI entrypoint for running the live Ultimo TCP inference bot."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

from ultimo_marl.bot import LiveBot, create_bot_agent
from ultimo_marl.protocol import (
    EventType,
    Join,
    ServerEvent,
    ServerShutdown,
    TcpGameClient,
)

logger = logging.getLogger("ultimo_bot")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run live Ultimo MARL inference bot over raw TCP socket"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Target Ultimo server hostname or IP (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Target Ultimo server TCP port (default: 8080)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="DeepUltimo_v1",
        help="Bot username to register with server (default: DeepUltimo_v1)",
    )
    parser.add_argument(
        "--room",
        type=str,
        default=None,
        help="Target room ID to join (default: None)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ultimo_agent_best.pt",
        help="Path to trained PyTorch policy weights (default: checkpoints/ultimo_agent_best.pt)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["dqn", "heuristic", "random"],
        default="dqn",
        help="Bot decision strategy (default: dqn)",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_false",
        dest="auto_start",
        help="Disable automatic START dispatch when room reaches capacity",
    )
    parser.add_argument(
        "--single-game",
        action="store_true",
        help="Terminate cleanly after the first match concludes with GAME_OVER",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI execution routine."""
    args = parse_args()

    # Configure structured logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Initializing Ultimo bot '%s'", args.name)
    logger.info("Target: %s:%d | Strategy: %s", args.host, args.port, args.strategy)

    agent = create_bot_agent(strategy=args.strategy, model_path=args.model_path)
    client = TcpGameClient(host=args.host, port=args.port)
    bot = LiveBot(
        client=client,
        agent=agent,
        bot_name=args.name,
        auto_start=args.auto_start,
    )

    # Register OS signal handlers for graceful shutdown
    def signal_handler(signum: int, _frame: Any) -> None:
        logger.info("Signal %d received, stopping bot...", signum)
        bot.stop()
        client.disconnect()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register ServerShutdown handler to cleanly disconnect and exit code 0
    def shutdown_handler(event: ServerEvent) -> None:
        if isinstance(event, ServerShutdown):
            logger.warning(
                "Server is performing graceful shutdown (%ds drain). Disconnecting cleanly.",
                event.drain_seconds,
            )
            bot.stop()
            client.disconnect()
            sys.exit(0)

    client.register_handler(EventType.SERVER_SHUTDOWN, shutdown_handler)

    try:
        client.connect()
        logger.info("Registering bot username '%s' with server...", args.name)
        client.send_command(Join(username=args.name, room_id=args.room))
        bot.run(stop_on_game_over=args.single_game)
    except ConnectionError as err:
        logger.error("Connection failed: %s", err)
        return 1
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user.")
    finally:
        bot.stop()
        client.disconnect()
        logger.info("Bot shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
