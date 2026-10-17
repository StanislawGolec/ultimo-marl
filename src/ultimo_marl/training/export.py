"""Model artifact exporter for TorchScript and ONNX zero-dependency deployment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ultimo_marl.env.observation import OBSERVATION_DIM

if TYPE_CHECKING:
    from ultimo_marl.training.q_network import DuelingQNetwork


def export_torchscript(
    network: DuelingQNetwork,
    output_path: str | Path,
    sample_obs: torch.Tensor | None = None,
) -> Path:
    """Trace and export PyTorch model to optimized TorchScript format.

    Parameters
    ----------
    network : DuelingQNetwork
        Trained neural network to trace.
    output_path : str | Path
        Target file path (.ptc or .ts).
    sample_obs : torch.Tensor | None
        Sample input observation tensor. If None, generated as zeros of shape (1, obs_dim).

    Returns
    -------
    Path
        Absolute path to the saved TorchScript artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    network.eval()
    device = next(network.parameters()).device

    if sample_obs is None:
        sample_obs = torch.zeros(1, network.obs_dim, dtype=torch.float32, device=device)
    else:
        sample_obs = sample_obs.to(device)

    with torch.no_grad():
        traced_model = torch.jit.trace(network, sample_obs)
        with open(path, "wb") as f:
            torch.jit.save(traced_model, f)

    return path.resolve()


def export_onnx(
    network: DuelingQNetwork,
    output_path: str | Path,
    sample_obs: torch.Tensor | None = None,
    opset_version: int = 18,
) -> Path:
    """Export PyTorch policy to Open Neural Network Exchange (ONNX) format.

    Enables deployment in C++, Rust, Go, Java (ONNX Runtime) without PyTorch runtime.

    Parameters
    ----------
    network : DuelingQNetwork
        Trained neural network to export.
    output_path : str | Path
        Target file path (.onnx).
    sample_obs : torch.Tensor | None
        Sample input observation tensor.
    opset_version : int
        ONNX operator set version (default: 18).

    Returns
    -------
    Path
        Absolute path to the saved ONNX artifact.
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    network.eval()
    device = next(network.parameters()).device

    if sample_obs is None:
        sample_obs = torch.zeros(1, network.obs_dim, dtype=torch.float32, device=device)
    else:
        sample_obs = sample_obs.to(device)

    with torch.no_grad():
        torch.onnx.export(
            network,
            (sample_obs,),
            str(path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["q_values"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "q_values": {0: "batch_size"},
            },
        )

    return path.resolve()


def export_all_artifacts(
    network: DuelingQNetwork,
    export_dir: str | Path,
    base_name: str = "ultimo_agent",
) -> dict[str, Path]:
    """Export complete production deployment bundle (PyTorch, TorchScript, and ONNX).

    Parameters
    ----------
    network : DuelingQNetwork
        Trained policy network.
    export_dir : str | Path
        Directory where artifacts will be deposited.
    base_name : str
        Prefix for exported files.

    Returns
    -------
    dict[str, Path]
        Mapping from format ('pytorch', 'torchscript', 'onnx') to absolute artifact paths.
    """
    dir_path = Path(export_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    sample_obs = torch.zeros(1, OBSERVATION_DIM, dtype=torch.float32)

    # 1. PyTorch checkpoint
    pt_path = dir_path / f"{base_name}.pt"
    checkpoint = {
        "online_net": network.state_dict(),
        "obs_dim": network.obs_dim,
        "action_dim": network.action_dim,
        "hidden_dim": network.hidden_dim,
    }
    torch.save(checkpoint, pt_path)

    # 2. TorchScript traced model
    ts_path = dir_path / f"{base_name}.ptc"
    export_torchscript(network, ts_path, sample_obs=sample_obs)

    # 3. ONNX model
    onnx_path = dir_path / f"{base_name}.onnx"
    export_onnx(network, onnx_path, sample_obs=sample_obs)

    return {
        "pytorch": pt_path.resolve(),
        "torchscript": ts_path.resolve(),
        "onnx": onnx_path.resolve(),
    }
