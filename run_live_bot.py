#!/usr/bin/env python3
"""CLI entrypoint for running the live Ultimo TCP inference bot."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run live Ultimo MARL inference bot over raw TCP socket"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Target Ultimo server hostname or IP (default: localhost)",
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
        help="Bot username to register with server",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ultimo_agent_best.pt",
        help="Path to trained PyTorch policy weights",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["dqn", "heuristic", "random"],
        default="dqn",
        help="Bot decision strategy (default: dqn)",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_args()
    print(
        f"[Ultimo-MARL] Initializing live bot '{args.name}' connecting to {args.host}:{args.port}"
    )
    print(f"[Ultimo-MARL] Strategy: {args.strategy} (weights: {args.model_path})")
    # Implemented in Stage 8
    return 0


if __name__ == "__main__":
    sys.exit(main())
