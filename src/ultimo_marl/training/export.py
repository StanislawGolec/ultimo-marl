"""Model artifact exporter for TorchScript and ONNX zero-dependency deployment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

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


def verify_onnx_parity(
    network: DuelingQNetwork,
    onnx_path: str | Path,
    sample_obs: torch.Tensor | None = None,
    atol: float = 1e-5,
) -> float:
    """Verify numerical parity between PyTorch eager forward pass and ONNX Runtime.

    Parameters
    ----------
    network : DuelingQNetwork
        PyTorch network to compare against.
    onnx_path : str | Path
        Path to exported .onnx model.
    sample_obs : torch.Tensor | None
        Test observation tensor. If None, generated as random float32 tensor of shape (8, obs_dim).
    atol : float
        Maximum allowed absolute tolerance (default: 1e-5).

    Returns
    -------
    float
        Maximum absolute delta observed between PyTorch and ONNX Runtime outputs.
    """
    import numpy as np
    import onnxruntime as ort

    network.eval()
    device = next(network.parameters()).device

    if sample_obs is None:
        sample_obs = torch.randn(8, network.obs_dim, dtype=torch.float32, device=device)
    else:
        sample_obs = sample_obs.to(device)

    with torch.no_grad():
        pt_out = network(sample_obs).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: sample_obs.cpu().numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    max_delta = float(np.max(np.abs(pt_out - ort_out)))
    if max_delta > atol:
        raise ValueError(
            f"ONNX parity violation! Max absolute delta {max_delta:.6e} exceeds tolerance {atol:.6e}"
        )
    return max_delta


def export_all_artifacts(
    network: DuelingQNetwork,
    export_dir: str | Path,
    base_name: str = "ultimo_agent",
    verify_parity: bool = True,
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
    verify_parity : bool
        Whether to verify numerical parity with ONNX Runtime (default: True).

    Returns
    -------
    dict[str, Path]
        Mapping from format ('pytorch', 'torchscript', 'onnx') to absolute artifact paths.
    """
    dir_path = Path(export_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    sample_obs = torch.zeros(1, network.obs_dim, dtype=torch.float32)

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

    # 4. Verify ONNX numerical parity
    if verify_parity:
        verify_onnx_parity(network, onnx_path, sample_obs=sample_obs)

    return {
        "pytorch": pt_path.resolve(),
        "torchscript": ts_path.resolve(),
        "onnx": onnx_path.resolve(),
    }
