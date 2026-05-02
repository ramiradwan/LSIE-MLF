"""Preflight warning codes and operator-readable messages (v4.0 §11.x.3).

The preflight gate uses a small, deterministic enumeration of warning
codes so the operator-console health page (services/operator_console/
views/health_view.py) can render canonical strings instead of
free-form preflight chatter. Codes are stable identifiers; messages
are surface text the operator sees.
"""

from __future__ import annotations

from typing import Final

PASCAL_DEV_MODE_REQUIRED: Final[str] = "PASCAL_DEV_MODE_REQUIRED"
"""Sub-Turing GPU detected; running in developer mode (CPU speech).

The launcher proceeds because the operator has declared this host a
developer machine — either by creating the ``.dev_machine`` marker or
by setting ``LSIE_DEV_FORCE_CPU_SPEECH=1`` in the environment. The
production preflight gate hard-rejects this same condition without
the developer declaration.
"""

NO_GPU_DETECTED: Final[str] = "NO_GPU_DETECTED"
"""nvidia-smi reported no NVIDIA GPU (or is missing).

In developer mode this routes both speech and cross-encoder to CPU.
In production it is a hard failure — the launcher aborts with
:class:`HardwareUnsupportedError`.
"""

PASCAL_DEV_MODE_REQUIRED_MESSAGE: Final[str] = (
    "Unsupported production GPU detected: {gpu_name} (compute capability "
    "{compute_cap:.1f}). LSIE-MLF v4.0 production requires Turing (7.5+) "
    "for GPU speech acceleration. Running in developer mode: "
    "LSIE_DEV_FORCE_CPU_SPEECH=1 will route speech to the CPU. PyTorch "
    "(semantic scorer) and MediaPipe will continue to use this GPU."
)

NO_GPU_DEV_MODE_MESSAGE: Final[str] = (
    "No NVIDIA GPU detected. Running in developer mode: every ML path "
    "(speech, semantic scorer, MediaPipe) routes to the CPU. Production "
    "deployments require an NVIDIA Turing (7.5+) GPU."
)

HARDWARE_UNSUPPORTED_PRODUCTION_MESSAGE: Final[str] = (
    "Unsupported production GPU: {gpu_name} (compute capability "
    "{compute_cap:.1f}). LSIE-MLF v4.0 production requires NVIDIA "
    "Turing (7.5+) for the GPU speech path. To run in developer mode "
    "on this host, create the marker file at "
    "{marker_path} and re-run the launcher."
)

HARDWARE_NO_GPU_PRODUCTION_MESSAGE: Final[str] = (
    "No NVIDIA GPU detected. LSIE-MLF v4.0 production requires NVIDIA "
    "Turing (7.5+). To run in developer mode on this host, create the "
    "marker file at {marker_path} and re-run the launcher."
)

PYTHON_VERSION_UNSUPPORTED_MESSAGE: Final[str] = (
    "Unsupported Python interpreter: {detected}. LSIE-MLF v4.0 requires "
    "Python 3.11.x because mediapipe == 0.10.9 ships no PyPI wheel for "
    "Python 3.12+. Reinstall with the bundled python-build-standalone "
    "runtime (the launcher provisions this automatically) or activate "
    "a 3.11.x interpreter and re-run."
)
