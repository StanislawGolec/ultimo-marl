"""Self-play league training pipeline with fictitious play and tournament evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultimo_marl.agents.base import BaseAgent
from ultimo_marl.agents.heuristic_agent import RuleBasedHeuristicAgent
from ultimo_marl.agents.random_agent import RandomAgent
from ultimo_marl.agents.tournament import Tournament
from ultimo_marl.env.action_mapping import NUM_ACTIONS
from ultimo_marl.env.ultimo_aec_env import UltimoAECEnv
from ultimo_marl.training.export import export_all_artifacts
from ultimo_marl.training.learner import DDQNLearner
from ultimo_marl.training.q_network import DuelingQNetwork
from ultimo_marl.training.replay_buffer import PrioritizedReplayBuffer
from ultimo_marl.training.telemetry import MetricsLogger, generate_training_plots


@dataclass
class EpsilonSchedule:
    """Linear epsilon decay exploration schedule."""

    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 200_000

    def get_epsilon(self, step: int) -> float:
        """Calculate epsilon exploration rate at given global step."""
        if step >= self.decay_steps:
            return self.end
        progress = step / max(self.decay_steps, 1)
        return float(self.start - progress * (self.start - self.end))


class LeaguePool:
    """Fictitious play league population pool storing historical policy checkpoints.

    Matchmaking distribution:
    - 70% chance: historical snapshot from pool (fictitious self-play).
    - 20% chance: clone of latest active policy.
    - 10% chance: RuleBasedHeuristicAgent.

    Parameters
    ----------
    league_dir : str | Path
        Directory where policy snapshot files (.pt) are stored.
    """

    def __init__(self, league_dir: str | Path) -> None:
        self.league_dir = Path(league_dir)
        self.league_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[Path] = sorted(self.league_dir.glob("snapshot_*.pt"))

    def save_snapshot(self, step: int, network: DuelingQNetwork) -> Path:
        """Save network weights as a historical league snapshot."""
        path = self.league_dir / f"snapshot_{step:07d}.pt"
        checkpoint = {
            "online_net": network.state_dict(),
            "obs_dim": network.obs_dim,
            "action_dim": network.action_dim,
            "hidden_dim": network.hidden_dim,
            "step": step,
        }
        torch.save(checkpoint, path)
        self.snapshots.append(path)
        return path

    def sample_opponent(
        self,
        active_network: DuelingQNetwork,
        device: torch.device,
    ) -> BaseAgent:
        """Sample an opponent agent based on the league matchmaking distribution."""
        from ultimo_marl.agents.dqn_agent import DQNAgent

        roll = random.random()

        # 70%: historical snapshot if available
        if roll < 0.70 and len(self.snapshots) > 0:
            chosen_snap = random.choice(self.snapshots)
            snap_name = chosen_snap.stem
            return DQNAgent(checkpoint_path=chosen_snap, name=f"League_{snap_name}", device=device)

        # 20%: latest active policy clone (or fallback if no snapshots)
        if roll < 0.90 or len(self.snapshots) == 0:
            return DQNAgent(network=active_network, name="LatestClone", device=device)

        # 10%: heuristic sparring partner
        return RuleBasedHeuristicAgent(name="HeuristicSparring")


class SelfPlayTrainer:
    """Orchestrates multi-agent self-play reinforcement learning and telemetry.

    Parameters
    ----------
    learner : DDQNLearner
        DDQN gradient learner module.
    replay_buffer : PrioritizedReplayBuffer
        Prioritized experience replay buffer.
    save_dir : str | Path
        Directory for storing checkpoints, logs, and plots.
    batch_size : int
        Mini-batch size for gradient updates (default: 64).
    warmup_steps : int
        Number of steps to collect before starting neural updates (default: 1,000).
    eval_interval : int
        Frequency (in steps) of running evaluation tournaments (default: 5,000).
    snapshot_interval : int
        Frequency (in steps) of archiving league snapshots (default: 10,000).
    epsilon_schedule : EpsilonSchedule | None
        Exploration decay schedule.
    device : torch.device | None
        Target PyTorch computation device.
    """

    def __init__(
        self,
        learner: DDQNLearner,
        replay_buffer: PrioritizedReplayBuffer,
        save_dir: str | Path = "checkpoints",
        batch_size: int = 64,
        warmup_steps: int = 1_000,
        eval_interval: int = 5_000,
        snapshot_interval: int = 10_000,
        epsilon_schedule: EpsilonSchedule | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device if device is not None else next(learner.online_net.parameters()).device
        self.learner = learner
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.snapshot_interval = snapshot_interval

        self.epsilon_schedule = (
            epsilon_schedule if epsilon_schedule is not None else EpsilonSchedule()
        )
        self.league_pool = LeaguePool(self.save_dir / "league")
        self.logger = MetricsLogger(self.save_dir / "logs")

        self.best_heuristic_win_rate = 0.0
        self.best_model_path = self.save_dir / "ultimo_agent_best.pt"

    def evaluate_policy(self, num_games: int = 50) -> dict[str, float]:
        """Run tournament evaluations against Heuristic and Random baselines."""
        from ultimo_marl.agents.dqn_agent import DQNAgent

        candidate = DQNAgent(
            network=self.learner.online_net, name="CandidateDQN", device=self.device
        )

        # 1. Matchup against 3 RuleBasedHeuristicAgents
        heuristic_opponents = [RuleBasedHeuristicAgent(name=f"Heuristic_{i}") for i in range(1, 4)]
        t_heuristic = Tournament(agents=[candidate, *heuristic_opponents])
        res_h = t_heuristic.play(num_games=num_games)
        win_rate_h = res_h.win_rates.get("CandidateDQN", 0.0)

        # 2. Matchup against 3 RandomAgents
        random_opponents = [RandomAgent(name=f"Random_{i}") for i in range(1, 4)]
        t_random = Tournament(agents=[candidate, *random_opponents])
        res_r = t_random.play(num_games=num_games)
        win_rate_r = res_r.win_rates.get("CandidateDQN", 0.0)

        return {
            "win_rate_vs_heuristic": win_rate_h,
            "win_rate_vs_random": win_rate_r,
            "avg_turns_vs_heuristic": res_h.average_turns,
            "avg_turns_vs_random": res_r.average_turns,
        }

    def train(self, total_steps: int = 20_000) -> dict[str, Any]:
        """Execute the multi-agent self-play training loop.

        Parameters
        ----------
        total_steps : int
            Total environment transition steps to collect and train on.

        Returns
        -------
        dict[str, Any]
            Final training summary statistics and artifact paths.
        """
        env = UltimoAECEnv()
        global_step = 0
        episode_idx = 0
        recent_losses: list[float] = []

        print(f"[SelfPlayTrainer] Starting training for {total_steps:,} steps on {self.device}...")

        while global_step < total_steps:
            episode_idx += 1
            env.reset()

            # Active learner always plays as 'player_0'
            learner_agent_id = "player_0"

            # Sample 3 sparring opponents from the League Pool
            opponents: dict[str, BaseAgent] = {
                f"player_{i}": self.league_pool.sample_opponent(
                    self.learner.online_net, self.device
                )
                for i in range(1, 4)
            }

            # State tracking for pending transition of the active learner
            pending_obs: np.ndarray | None = None
            pending_action: int | None = None

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                mask = info["action_mask"]

                if termination or truncation:
                    if (
                        agent == learner_agent_id
                        and pending_obs is not None
                        and pending_action is not None
                    ):
                        # Terminal transition for learner
                        zero_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
                        zero_mask[0] = 1
                        self.replay_buffer.push(
                            state=pending_obs,
                            action=pending_action,
                            reward=float(reward),
                            next_state=obs,
                            done=True,
                            next_action_mask=zero_mask,
                        )
                        pending_obs = None
                        pending_action = None

                    env.step(None)
                    continue

                if agent == learner_agent_id:
                    # If learner previously acted, store the transition: (s, a, r, s', done=False, mask')
                    if pending_obs is not None and pending_action is not None:
                        self.replay_buffer.push(
                            state=pending_obs,
                            action=pending_action,
                            reward=float(reward),
                            next_state=obs,
                            done=False,
                            next_action_mask=mask,
                        )

                    epsilon = self.epsilon_schedule.get_epsilon(global_step)
                    action = self.learner.online_net.act(obs, mask, epsilon=epsilon)
                    pending_obs = obs
                    pending_action = action

                    global_step += 1

                    # Take gradient update step once warmup threshold is met
                    if len(self.replay_buffer) >= self.warmup_steps:
                        batch, indices, weights = self.replay_buffer.sample(
                            batch_size=self.batch_size, device=self.device
                        )
                        loss, td_errors = self.learner.train_step(batch, weights)
                        self.replay_buffer.update_priorities(indices, td_errors)
                        recent_losses.append(loss)

                    # Periodic Telemetry & Logging
                    if global_step % 100 == 0 and recent_losses:
                        avg_loss = float(np.mean(recent_losses[-100:]))
                        self.logger.log(
                            step=global_step,
                            metrics={
                                "loss": avg_loss,
                                "epsilon": epsilon,
                                "buffer_size": len(self.replay_buffer),
                                "episode": episode_idx,
                            },
                        )

                    # Periodic League Snapshot Archival
                    if global_step % self.snapshot_interval == 0:
                        snap_path = self.league_pool.save_snapshot(
                            step=global_step, network=self.learner.online_net
                        )
                        print(f"[{global_step:,} steps] Archived League Snapshot: {snap_path.name}")

                    # Periodic Tournament Evaluation
                    if global_step % self.eval_interval == 0:
                        eval_metrics = self.evaluate_policy(num_games=30)
                        self.logger.log(step=global_step, metrics=eval_metrics)
                        win_rate_h = eval_metrics["win_rate_vs_heuristic"]
                        win_rate_r = eval_metrics["win_rate_vs_random"]
                        print(
                            f"[{global_step:,} steps] Eval | vs Heuristic: {win_rate_h:.1%} | vs Random: {win_rate_r:.1%}"
                        )

                        # Checkpoint best policy
                        if win_rate_h >= self.best_heuristic_win_rate:
                            self.best_heuristic_win_rate = win_rate_h
                            self.learner.save(self.best_model_path)
                            print(f"  -> Saved new best checkpoint: {self.best_model_path.name}")

                    env.step(action)

                    if global_step >= total_steps:
                        break
                else:
                    # Opponent move execution
                    opp = opponents[agent]
                    opp_action = opp.select_action(obs, mask)
                    env.step(opp_action)

        # ---------------------------------------------------------------------
        # Finalization: Save checkpoint, generate plots, export deployment artifacts
        # ---------------------------------------------------------------------
        final_ckpt = self.save_dir / "ultimo_agent_final.pt"
        self.learner.save(final_ckpt)
        if not self.best_model_path.is_file():
            self.learner.save(self.best_model_path)

        self.logger.close()

        # Generate automated training visualization curves
        loss_plot, winrate_plot = generate_training_plots(
            jsonl_path=self.logger.jsonl_path,
            output_dir=self.save_dir / "plots",
        )

        # Export TorchScript and ONNX production deployment bundles
        export_bundle = export_all_artifacts(
            network=self.learner.online_net,
            export_dir=self.save_dir / "exported",
            base_name="ultimo_agent",
        )

        print("\n[SelfPlayTrainer] Training completed successfully!")
        print(f"  - Best Checkpoint: {self.best_model_path}")
        print(f"  - Training Curves: {loss_plot}, {winrate_plot}")
        print(f"  - ONNX Model: {export_bundle['onnx']}")
        print(f"  - TorchScript Model: {export_bundle['torchscript']}")

        return {
            "total_steps": global_step,
            "total_episodes": episode_idx,
            "best_win_rate_heuristic": self.best_heuristic_win_rate,
            "best_checkpoint": str(self.best_model_path),
            "loss_plot": str(loss_plot),
            "winrate_plot": str(winrate_plot),
            "onnx_artifact": str(export_bundle["onnx"]),
            "torchscript_artifact": str(export_bundle["torchscript"]),
        }
