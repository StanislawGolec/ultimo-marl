"""Unit and integration tests for self-play training, telemetry, and model export (Stage 6)."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch

from ultimo_marl.agents import BaseAgent
from ultimo_marl.env import OBSERVATION_DIM
from ultimo_marl.training import (
    DDQNLearner,
    DuelingQNetwork,
    EpsilonSchedule,
    LeaguePool,
    MetricsLogger,
    PrioritizedReplayBuffer,
    SelfPlayTrainer,
    export_all_artifacts,
    export_onnx,
    export_torchscript,
    generate_training_plots,
)

# ============================================================================
# 1. Epsilon Schedule Tests
# ============================================================================


def test_epsilon_schedule() -> None:
    """Verify linear decay schedule between start and end values."""
    sched = EpsilonSchedule(start=1.0, end=0.1, decay_steps=1000)

    assert sched.get_epsilon(0) == 1.0
    assert pytest.approx(sched.get_epsilon(500), abs=1e-5) == 0.55
    assert sched.get_epsilon(1000) == 0.1
    assert sched.get_epsilon(2000) == 0.1


# ============================================================================
# 2. League Pool Tests
# ============================================================================


def test_league_pool_snapshot_archival_and_sampling(tmp_path: Path) -> None:
    """Verify LeaguePool archives policy checkpoints and samples valid BaseAgent opponents."""
    pool = LeaguePool(tmp_path / "league")
    assert len(pool.snapshots) == 0

    net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, hidden_dim=64)
    device = torch.device("cpu")

    # When no snapshots exist, opponent sampling must return a valid BaseAgent
    opp_fallback = pool.sample_opponent(net, device)
    assert isinstance(opp_fallback, BaseAgent)

    # Archive snapshots
    snap1 = pool.save_snapshot(step=1000, network=net)
    snap2 = pool.save_snapshot(step=2000, network=net)

    assert snap1.is_file()
    assert snap2.is_file()
    assert len(pool.snapshots) == 2

    # Sample multiple opponents and verify all conform to BaseAgent
    for _ in range(10):
        opp = pool.sample_opponent(net, device)
        assert isinstance(opp, BaseAgent)


# ============================================================================
# 3. Telemetry & Plot Generation Tests
# ============================================================================


def test_telemetry_and_plot_generation(tmp_path: Path) -> None:
    """Verify MetricsLogger logs JSONL and generate_training_plots creates valid PNG files."""
    log_dir = tmp_path / "telemetry"
    logger = MetricsLogger(log_dir=log_dir, use_tensorboard=True)

    # Log several steps
    for step in range(10):
        logger.log(step=step * 10, metrics={"loss": 1.0 / (step + 1), "epsilon": 1.0 - step * 0.05})

    # Log evaluation steps
    logger.log(step=50, metrics={"win_rate_vs_heuristic": 0.45, "win_rate_vs_random": 0.70})
    logger.log(step=100, metrics={"win_rate_vs_heuristic": 0.75, "win_rate_vs_random": 0.95})
    logger.close()

    assert logger.jsonl_path.is_file()

    # Validate JSONL content
    lines = [line.strip() for line in open(logger.jsonl_path, encoding="utf-8") if line.strip()]
    assert len(lines) == 12
    first_record = json.loads(lines[0])
    assert first_record["step"] == 0
    assert "loss" in first_record

    # Generate training curves
    loss_png, winrate_png = generate_training_plots(
        jsonl_path=logger.jsonl_path,
        output_dir=tmp_path / "plots",
    )

    assert loss_png.is_file()
    assert loss_png.stat().st_size > 1000
    assert winrate_png.is_file()
    assert winrate_png.stat().st_size > 1000


# ============================================================================
# 4. TorchScript and ONNX Export Tests
# ============================================================================


def test_torchscript_and_onnx_export(tmp_path: Path) -> None:
    """Verify zero-dependency export to TorchScript and ONNX with numerical consistency."""
    net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, hidden_dim=64)
    net.eval()

    sample_obs = torch.randn(1, OBSERVATION_DIM)
    expected_q = net(sample_obs)

    # 1. Test TorchScript export
    ts_path = tmp_path / "model.ptc"
    export_torchscript(net, ts_path, sample_obs=sample_obs)
    assert ts_path.is_file()

    with open(ts_path, "rb") as f:
        traced_model = torch.jit.load(f)
    ts_q = traced_model(sample_obs)
    assert torch.allclose(expected_q, ts_q, atol=1e-5)

    # 2. Test ONNX export
    onnx_path = tmp_path / "model.onnx"
    export_onnx(net, onnx_path, sample_obs=sample_obs)
    assert onnx_path.is_file()

    # Check ONNX model structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 3. Test export_all_artifacts bundle
    bundle = export_all_artifacts(net, tmp_path / "bundle", base_name="test_bot")
    assert bundle["pytorch"].is_file()
    assert bundle["torchscript"].is_file()
    assert bundle["onnx"].is_file()


# ============================================================================
# 5. End-to-End Self-Play Trainer Integration Test
# ============================================================================


def test_self_play_trainer_integration(tmp_path: Path) -> None:
    """Run a fast self-play training loop verifying end-to-end multi-agent rollouts and artifacts."""
    device = torch.device("cpu")
    online_net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, hidden_dim=64).to(device)
    target_net = DuelingQNetwork(obs_dim=OBSERVATION_DIM, hidden_dim=64).to(device)
    target_net.load_state_dict(online_net.state_dict())

    learner = DDQNLearner(online_net=online_net, target_net=target_net, lr=1e-3, device=device)
    replay_buffer = PrioritizedReplayBuffer(capacity=1000, obs_dim=OBSERVATION_DIM)
    schedule = EpsilonSchedule(start=1.0, end=0.1, decay_steps=50)

    trainer = SelfPlayTrainer(
        learner=learner,
        replay_buffer=replay_buffer,
        save_dir=tmp_path / "trainer_run",
        batch_size=4,
        warmup_steps=10,
        eval_interval=30,
        snapshot_interval=30,
        epsilon_schedule=schedule,
        device=device,
    )

    # Run 60 fast training steps
    results = trainer.train(total_steps=60)

    assert results["total_steps"] >= 60
    assert len(replay_buffer) > 10
    assert Path(results["best_checkpoint"]).is_file()
    assert Path(results["loss_plot"]).is_file()
    assert Path(results["winrate_plot"]).is_file()
    assert Path(results["onnx_artifact"]).is_file()
    assert Path(results["torchscript_artifact"]).is_file()
