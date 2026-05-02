"""NVIDIA GPU compute-capability probe via ``nvidia-smi`` (v4.0 §11.x).

Two consumers share this helper:

* :mod:`packages.ml_core.transcription` — :func:`resolve_speech_device`
  picks ``"cuda"`` only when at least one detected GPU has compute
  capability ≥ 7.5 (Turing+).
* :mod:`services.desktop_launcher.preflight` — uses the full inventory
  (name + capability) to render the ``PASCAL_DEV_MODE_REQUIRED``
  warning text and to decide whether to fail closed in production
  mode.

The probe shells out to ``nvidia-smi --query-gpu=name,compute_cap
--format=csv,noheader`` with a 5 s timeout. Every failure mode
(missing binary, nonzero exit, timeout, unparseable rows) collapses
to an empty inventory — callers translate "empty inventory" into the
appropriate domain-level decision (CPU fallback, hardware-rejection,
etc.). The probe never raises.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

_NVIDIA_SMI_TIMEOUT_S: float = 5.0


@dataclass(frozen=True)
class GpuInfo:
    """A single NVIDIA GPU's name and SM compute capability.

    ``compute_cap`` is reported as a float (e.g. ``7.5`` for Turing,
    ``6.1`` for Pascal GP102/GTX 1080 Ti) so caller comparisons against
    the Turing floor stay numeric instead of string-parsing.
    """

    name: str
    compute_cap: float


def query_gpu_inventory() -> list[GpuInfo]:
    """Return every NVIDIA GPU ``nvidia-smi`` reports on this host.

    Empty list when ``nvidia-smi`` is missing, exits non-zero, times
    out, or every row fails to parse. Multi-GPU hosts return rows in
    the order ``nvidia-smi`` enumerates them.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    inventory: list[GpuInfo] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            cap = float(parts[1])
        except ValueError:
            continue
        inventory.append(GpuInfo(name=parts[0], compute_cap=cap))
    return inventory


def query_max_compute_capability() -> float | None:
    """Return the highest compute capability reported, or ``None``.

    Convenience over :func:`query_gpu_inventory` for callers that only
    care about the device-routing decision and not the GPU's name.
    """
    inventory = query_gpu_inventory()
    if not inventory:
        return None
    return max(g.compute_cap for g in inventory)
