#!/usr/bin/env python3
"""CLI entrypoint for running evaluation tournaments between agents."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Run tournament evaluation between Ultimo agent policies"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of tournament games to simulate (default: 1000)",
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default="dqn",
        choices=["dqn", "heuristic", "random"],
        help="Agent policy A (default: dqn)",
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="heuristic",
        choices=["dqn", "heuristic", "random"],
        help="Agent policy B (default: heuristic)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ultimo_agent_best.pt",
        help="Path to neural policy weights (if using dqn)",
    )
    return parser.parse_args()


def main() -> int:
    """Main evaluation tournament entrypoint."""
    args = parse_args()
    print(
        f"[Ultimo-MARL] Starting evaluation tournament: {args.agent_a} vs {args.agent_b} ({args.games} games)"
    )
    # Implemented in Stage 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
