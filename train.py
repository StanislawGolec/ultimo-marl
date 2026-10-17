#!/usr/bin/env python3
"""CLI entrypoint for running self-play training of Ultimo agents."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(
        description="Train Ultimo MARL policy using Dueling DQN and self-play"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Total training environment steps (default: 500000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for gradient updates (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW optimizer (default: 0.0003)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computing device: auto, cpu, or cuda (default: auto)",
    )
    return parser.parse_args()


def main() -> int:
    """Main training orchestrator entrypoint."""
    args = parse_args()
    print(
        f"[Ultimo-MARL] Starting self-play training pipeline: {args.steps} steps, device: {args.device}"
    )
    # Implemented in Stage 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
