"""Preflight gate for the Windows desktop runtime (§9, §10.1, §10.2)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from packages.ml_core import gpu_probe
from services.desktop_app import os_adapter
from services.desktop_launcher import preflight_codes

_TURING_COMPUTE_CAP: Final[float] = 7.5
_DEV_FORCE_CPU_ENV: Final[str] = "LSIE_DEV_FORCE_CPU_SPEECH"
_PREFLIGHT_COMPLETE_ENV: Final[str] = "LSIE_DESKTOP_PREFLIGHT_COMPLETE"


class HardwareUnsupportedError(RuntimeError):
    """Raised when production-mode hardware cannot satisfy the desktop runtime floor."""


class PythonVersionUnsupportedError(RuntimeError):
    """Raised when the interpreter is outside the supported 3.11.x runtime."""


@dataclass(frozen=True)
class PreflightWarning:
    code: str
    message: str


@dataclass(frozen=True)
class PreflightResult:
    python_version: tuple[int, int, int]
    developer_mode: bool
    speech_device: str
    selected_gpu: gpu_probe.GpuInfo | None
    warnings: tuple[PreflightWarning, ...]


def dev_machine_marker_path() -> Path:
    return os_adapter.resolve_state_dir().parent / ".dev_machine"


def run_preflight(
    *,
    python_version: tuple[int, int, int] | None = None,
) -> PreflightResult:
    detected_version = python_version or sys.version_info[:3]
    _validate_python_version(detected_version)

    inventory = gpu_probe.query_gpu_inventory()
    selected_gpu = _select_gpu(inventory)
    developer_mode = _developer_mode_enabled()

    if selected_gpu is None:
        result = _handle_no_gpu(detected_version, developer_mode)
    elif selected_gpu.compute_cap >= _TURING_COMPUTE_CAP:
        result = PreflightResult(
            python_version=detected_version,
            developer_mode=developer_mode,
            speech_device="cuda",
            selected_gpu=selected_gpu,
            warnings=(),
        )
    else:
        result = _handle_sub_turing_gpu(detected_version, selected_gpu, developer_mode)
    _mark_preflight_complete()
    return result


def _validate_python_version(version: tuple[int, int, int]) -> None:
    if version[:2] == (3, 11):
        return
    detected = ".".join(str(part) for part in version)
    raise PythonVersionUnsupportedError(
        preflight_codes.PYTHON_VERSION_UNSUPPORTED_MESSAGE.format(detected=detected)
    )


def _select_gpu(inventory: list[gpu_probe.GpuInfo]) -> gpu_probe.GpuInfo | None:
    if not inventory:
        return None
    return max(inventory, key=lambda gpu: gpu.compute_cap)


def _developer_mode_enabled() -> bool:
    return os.environ.get(_DEV_FORCE_CPU_ENV) == "1" or os_adapter.is_dev_machine()


def _handle_no_gpu(
    python_version: tuple[int, int, int],
    developer_mode: bool,
) -> PreflightResult:
    if not developer_mode:
        raise HardwareUnsupportedError(
            preflight_codes.HARDWARE_NO_GPU_PRODUCTION_MESSAGE.format(
                marker_path=dev_machine_marker_path()
            )
        )
    _force_cpu_speech()
    return PreflightResult(
        python_version=python_version,
        developer_mode=True,
        speech_device="cpu",
        selected_gpu=None,
        warnings=(
            PreflightWarning(
                code=preflight_codes.NO_GPU_DETECTED,
                message=preflight_codes.NO_GPU_DEV_MODE_MESSAGE,
            ),
        ),
    )


def _handle_sub_turing_gpu(
    python_version: tuple[int, int, int],
    selected_gpu: gpu_probe.GpuInfo,
    developer_mode: bool,
) -> PreflightResult:
    if not developer_mode:
        raise HardwareUnsupportedError(
            preflight_codes.HARDWARE_UNSUPPORTED_PRODUCTION_MESSAGE.format(
                gpu_name=selected_gpu.name,
                compute_cap=selected_gpu.compute_cap,
                marker_path=dev_machine_marker_path(),
            )
        )
    _force_cpu_speech()
    return PreflightResult(
        python_version=python_version,
        developer_mode=True,
        speech_device="cpu",
        selected_gpu=selected_gpu,
        warnings=(
            PreflightWarning(
                code=preflight_codes.PASCAL_DEV_MODE_REQUIRED,
                message=preflight_codes.PASCAL_DEV_MODE_REQUIRED_MESSAGE.format(
                    gpu_name=selected_gpu.name,
                    compute_cap=selected_gpu.compute_cap,
                ),
            ),
        ),
    )


def ensure_preflight() -> PreflightResult:
    if os.environ.get(_PREFLIGHT_COMPLETE_ENV) == "1":
        return PreflightResult(
            python_version=sys.version_info[:3],
            developer_mode=_developer_mode_enabled(),
            speech_device="cpu" if os.environ.get(_DEV_FORCE_CPU_ENV) == "1" else "cuda",
            selected_gpu=None,
            warnings=(),
        )
    return run_preflight()


def _mark_preflight_complete() -> None:
    os.environ[_PREFLIGHT_COMPLETE_ENV] = "1"


def _force_cpu_speech() -> None:
    os.environ[_DEV_FORCE_CPU_ENV] = "1"
