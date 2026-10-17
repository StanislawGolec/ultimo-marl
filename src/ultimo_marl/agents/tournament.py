"""Tournament evaluation engine comparing agent policies with statistical rigor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ultimo_marl.agents.base import BaseAgent
from ultimo_marl.domain.simulator import UltimoSimulator
from ultimo_marl.env.action_mapping import (
    ACTION_DRAW,
    action_id_to_card_and_color,
    compute_action_mask,
)
from ultimo_marl.env.observation import encode_observation


def wilson_score_interval(
    successes: int,
    total: int,
    z: float = 1.959964,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of wins (k).
    total : int
        Total matches played (n).
    z : float
        1 - alpha/2 quantile of standard normal distribution (default: 1.95996 for 95% confidence).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) in [0.0, 1.0].
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4 * total)) / total)) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


@dataclass
class TournamentResult:
    """Statistical summary of a tournament evaluation."""

    games_played: int
    agent_names: list[str]
    wins: dict[str, int]
    win_rates: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    average_turns: float
    total_scores: dict[str, int]

    def summary(self) -> str:
        """Render a formatted markdown table summarizing tournament performance."""
        lines = [
            f"### Tournament Results ({self.games_played:,} games | Avg turns: {self.average_turns:.1f})",
            "",
            "| Agent | Wins | Win Rate | 95% Wilson Confidence Interval | Total Points |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for name in self.agent_names:
            w = self.wins.get(name, 0)
            rate = self.win_rates.get(name, 0.0) * 100.0
            low, high = self.confidence_intervals.get(name, (0.0, 0.0))
            pts = self.total_scores.get(name, 0)
            lines.append(
                f"| **{name}** | {w:,} | {rate:.1f}% | [{low * 100.0:.1f}%, {high * 100.0:.1f}%] | {pts:,} |"
            )
        return "\n".join(lines)


class Tournament:
    """Executes evaluation matches between arbitrary agent policies.

    Rotates table seating across matches to eliminate seat position bias.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        max_turns_per_game: int = 500,
    ) -> None:
        if len(agents) < 2:
            raise ValueError(f"At least 2 agents are required, got {len(agents)}")
        self.agents = list(agents)
        self.num_players = len(agents)
        self.max_turns_per_game = max_turns_per_game

    def play(self, num_games: int = 1000, seed: int = 42) -> TournamentResult:
        """Run tournament simulation and compute statistical metrics."""
        wins = {agent.name: 0 for agent in self.agents}
        total_scores = {agent.name: 0 for agent in self.agents}
        total_turns = 0

        sim = UltimoSimulator(num_players=self.num_players)

        for game_idx in range(num_games):
            # Rotate agent seating to prevent position bias
            # Seat offset rotates every game
            seat_offset = game_idx % self.num_players
            assigned_agents = [
                self.agents[(seat_idx + seat_offset) % self.num_players]
                for seat_idx in range(self.num_players)
            ]

            # Reset agents
            for a in self.agents:
                a.reset()

            sim.reset(seed=seed + game_idx)
            turn_count = 0

            while not sim.is_game_over and turn_count < self.max_turns_per_game:
                curr_idx = sim.current_player_idx
                agent = assigned_agents[curr_idx]

                # Fog of War observation and action mask
                obs = encode_observation(sim, curr_idx)
                hand = sim.get_player(curr_idx).hand
                mask = compute_action_mask(hand, sim.top_card, sim.active_color)

                action_id = agent.select_action(obs, mask)

                # Execute action on headless simulator
                card, wild_color = action_id_to_card_and_color(action_id)
                if action_id == ACTION_DRAW or card is None:
                    sim.draw_card(curr_idx)
                    sim.pass_turn(curr_idx)
                else:
                    play_res = sim.play_cards(
                        curr_idx,
                        [card],
                        chosen_wild_color=wild_color,
                    )
                    if not play_res.success:
                        # Fallback if invalid action produced
                        sim.draw_card(curr_idx)
                        sim.pass_turn(curr_idx)

                turn_count += 1

            total_turns += turn_count

            # Record match outcome
            if sim.is_game_over and sim.winner_idx is not None:
                winner_agent = assigned_agents[sim.winner_idx]
                wins[winner_agent.name] += 1

                scores = sim.calculate_scores()
                for seat_idx, pts in scores.items():
                    agent_name = assigned_agents[seat_idx].name
                    total_scores[agent_name] += pts

        # Calculate win rates and confidence intervals
        win_rates = {name: wins[name] / num_games for name in wins}
        confidence_intervals = {name: wilson_score_interval(wins[name], num_games) for name in wins}
        avg_turns = total_turns / num_games if num_games > 0 else 0.0

        return TournamentResult(
            games_played=num_games,
            agent_names=[a.name for a in self.agents],
            wins=wins,
            win_rates=win_rates,
            confidence_intervals=confidence_intervals,
            average_turns=avg_turns,
            total_scores=total_scores,
        )
