"""Runtime utilities: device selection, seeding, mixed precision.

DESIGN GOAL (from MASTER_PLAN.md §4): the same code must run unchanged on
Apple Silicon (MPS), an NVIDIA GPU (CUDA), or CPU/cloud. Device is *resolved*,
never hardcoded. No `.cuda()` calls appear anywhere in this codebase.

References
----------
- PyTorch MPS backend (Apple Silicon GPU acceleration):
  https://pytorch.org/docs/stable/notes/mps.html
- Automatic Mixed Precision (AMP), torch.autocast / GradScaler:
  https://pytorch.org/docs/stable/amp.html
  NOTE: AMP autocast is enabled ONLY on CUDA here. As of the torch versions we
  target, fp16/bf16 autocast on the MPS backend is partial/unstable, so we
  intentionally run full-precision on MPS for correctness over speed.
"""
from __future__ import annotations

import contextlib
import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Resolve the best available device.

    Resolution order (when ``prefer`` is None or "auto"): CUDA -> MPS -> CPU.
    Pass prefer="cpu"/"mps"/"cuda" to force a specific backend (falls back to
    CPU with a warning if the requested backend is unavailable).

    Parameters
    ----------
    prefer : str | None
        One of {"auto", "cuda", "mps", "cpu", None}.

    Returns
    -------
    torch.device
    """
    prefer = (prefer or "auto").lower()

    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if prefer == "cuda":
        return torch.device("cuda") if cuda_ok else torch.device("cpu")
    if prefer == "mps":
        return torch.device("mps") if mps_ok else torch.device("cpu")
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if cuda_ok:
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch RNGs for reproducible experiments.

    Reproducibility matters for publication: every reported number must be
    traceable to a seed. We avoid forcing cudnn.deterministic globally on CPU/MPS
    where it is irrelevant; on CUDA it is set when ``deterministic`` is True.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def amp_autocast(device: torch.device, enabled: bool = True):
    """Mixed-precision context that is a no-op except on CUDA.

    On CUDA we use torch.autocast(fp16) for speed/memory. On MPS/CPU we yield a
    null context (full precision) for numerical stability. This keeps train.py
    identical across platforms.
    """
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def describe_runtime(device: torch.device) -> str:
    """Human-readable one-liner about the active runtime (for logs/reports)."""
    parts = [f"device={device.type}", f"torch={torch.__version__}", f"numpy={np.__version__}"]
    if device.type == "cuda":
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    return " | ".join(parts)
