#!/usr/bin/env python3
"""CLI entrypoint for running self-play training of Ultimo agents."""

import argparse
import sys

import torch

from ultimo_marl.training import (
    DDQNLearner,
    DuelingQNetwork,
    EpsilonSchedule,
    PrioritizedReplayBuffer,
    SelfPlayTrainer,
)


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
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimensionality (default: 256)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1_000,
        help="Warmup steps before gradient descent starts (default: 1000)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5_000,
        help="Step frequency for evaluation tournaments (default: 5000)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10_000,
        help="Step frequency for archiving league snapshots (default: 10000)",
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


def resolve_device(device_arg: str) -> torch.device:
    """Resolve target torch device from CLI string."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> int:
    """Main training orchestrator entrypoint."""
    args = parse_args()
    device = resolve_device(args.device)
    print(
        f"[Ultimo-MARL] Launching self-play training: {args.steps:,} steps on {device} (batch={args.batch_size}, lr={args.lr})"
    )

    online_net = DuelingQNetwork(hidden_dim=args.hidden_dim).to(device)
    target_net = DuelingQNetwork(hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())

    learner = DDQNLearner(
        online_net=online_net,
        target_net=target_net,
        lr=args.lr,
        device=device,
    )

    buffer = PrioritizedReplayBuffer(capacity=100_000)
    schedule = EpsilonSchedule(start=1.0, end=0.05, decay_steps=max(args.steps // 2, 10_000))

    trainer = SelfPlayTrainer(
        learner=learner,
        replay_buffer=buffer,
        save_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        snapshot_interval=args.snapshot_interval,
        epsilon_schedule=schedule,
        device=device,
    )

    results = trainer.train(total_steps=args.steps)
    print(f"\n[Ultimo-MARL] Training finished. Summary:\n{results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
