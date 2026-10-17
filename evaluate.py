#!/usr/bin/env python3
"""CLI entrypoint for running evaluation tournaments between agents."""

import argparse
import sys

from ultimo_marl.agents import (
    BaseAgent,
    RandomAgent,
    RuleBasedHeuristicAgent,
    Tournament,
)


def make_agent(name: str, strategy: str) -> BaseAgent:
    """Instantiate agent from strategy identifier."""
    if strategy == "heuristic":
        return RuleBasedHeuristicAgent(name=name)
    if strategy == "random":
        return RandomAgent(name=name)
    if strategy == "dqn":
        # Placeholder until Stage 5
        print(f"[Warning] DQN agent not loaded yet, falling back to Heuristic for {name}")
        return RuleBasedHeuristicAgent(name=name)
    raise ValueError(f"Unknown strategy: {strategy}")


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
        default="heuristic",
        choices=["dqn", "heuristic", "random"],
        help="Agent policy A (default: heuristic)",
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="random",
        choices=["dqn", "heuristic", "random"],
        help="Agent policy B (default: random)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ultimo_agent_best.pt",
        help="Path to neural policy weights (if using dqn)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tournament games (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    """Main evaluation tournament entrypoint."""
    args = parse_args()
    print(
        f"[Ultimo-MARL] Launching tournament: {args.agent_a} vs {args.agent_b} ({args.games:,} games)"
    )

    # In a 4-player match, create 1 Agent A and 3 Agent B to evaluate win rate of A vs 3 B
    agent_a = make_agent("Candidate_A", args.agent_a)
    opponents = [make_agent(f"Baseline_B{i}", args.agent_b) for i in range(1, 4)]
    all_agents: list[BaseAgent] = [agent_a, *opponents]

    tournament = Tournament(agents=all_agents)
    results = tournament.play(num_games=args.games, seed=args.seed)

    print("\n" + results.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
