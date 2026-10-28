"""Torch device selection: cpu | mps (Apple) | cuda | auto. Defaults to cpu (as before).
Can be set via the --device argument of the scripts or the NCMOL_DEVICE environment variable."""
from __future__ import annotations

import os

import torch


def resolve_device(name: str | None = None) -> torch.device:
    name = (name or os.environ.get("NCMOL_DEVICE") or "cpu").lower()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if mps_ok else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda: CUDA is not available on this machine")
    if name == "mps" and not mps_ok:
        raise RuntimeError("--device mps: MPS is not available (requires macOS on Apple Silicon and a torch build with MPS support)")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unknown device: {name} (cpu | mps | cuda | auto)")
    return torch.device(name)
