"""Production live TCP inference bot package for Ultimo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ultimo_marl.agents.base import BaseAgent
from ultimo_marl.agents.heuristic_agent import RuleBasedHeuristicAgent
from ultimo_marl.agents.random_agent import RandomAgent
from ultimo_marl.bot.live_bot import (
    USERNAME_REGEX,
    LiveBot,
    generate_fallback_name,
    validate_bot_username,
)
from ultimo_marl.bot.live_game_state import LiveGameState

if TYPE_CHECKING:
    import torch


def create_bot_agent(
    strategy: str = "dqn",
    model_path: str | Path | None = None,
    device: torch.device | None = None,
) -> BaseAgent:
    """Create agent instance based on specified strategy name.

    Supports:
    - 'dqn': Deep Q-Network policy initialized or restored from checkpoint.
    - 'heuristic': Rule-based domain expert card game agent.
    - 'random': Uniform random legal action sampler.
    """
    normalized = strategy.lower().strip()
    if normalized == "random":
        return RandomAgent()
    if normalized == "heuristic":
        return RuleBasedHeuristicAgent()
    if normalized == "dqn":
        from ultimo_marl.agents.dqn_agent import DQNAgent

        # If a checkpoint path is specified and exists, load it
        ckpt = Path(model_path) if model_path is not None else None
        if ckpt is not None and ckpt.exists():
            return DQNAgent(checkpoint_path=ckpt, device=device)
        return DQNAgent(device=device)

    raise ValueError(
        f"Unknown bot strategy '{strategy}'. Supported strategies: 'dqn', 'heuristic', 'random'."
    )


__all__ = [
    "USERNAME_REGEX",
    "LiveBot",
    "LiveGameState",
    "create_bot_agent",
    "generate_fallback_name",
    "validate_bot_username",
]
