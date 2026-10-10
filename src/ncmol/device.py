"""Выбор устройства для torch: cpu | mps (Apple) | cuda | auto. По умолчанию cpu (как раньше).
Можно задать аргументом --device у скриптов или переменной окружения NCMOL_DEVICE."""
from __future__ import annotations

import os

import torch


def resolve_device(name: str | None = None) -> torch.device:
    name = (name or os.environ.get("NCMOL_DEVICE") or "cpu").lower()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if mps_ok else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda: CUDA недоступна на этой машине")
    if name == "mps" and not mps_ok:
        raise RuntimeError("--device mps: MPS недоступен (нужен macOS на Apple Silicon и torch с поддержкой MPS)")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"неизвестное устройство: {name} (cpu | mps | cuda | auto)")
    return torch.device(name)
