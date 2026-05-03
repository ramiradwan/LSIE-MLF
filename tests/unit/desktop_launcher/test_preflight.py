"""Launcher preflight hardware gate tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from packages.ml_core import gpu_probe
from packages.ml_core.gpu_probe import GpuInfo
from services.desktop_app import os_adapter
from services.desktop_launcher import preflight, preflight_codes


@pytest.fixture(autouse=True)
def clear_preflight_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LSIE_DEV_FORCE_CPU_SPEECH", raising=False)
    monkeypatch.delenv("LSIE_DESKTOP_PREFLIGHT_COMPLETE", raising=False)


def _gpu(name: str, cap: float) -> GpuInfo:
    return GpuInfo(name=name, compute_cap=cap)


def _patch_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inventory: list[GpuInfo],
    dev_machine: bool,
    marker_path: Path,
) -> None:
    monkeypatch.setattr(gpu_probe, "query_gpu_inventory", lambda: inventory)
    monkeypatch.setattr(os_adapter, "is_dev_machine", lambda: dev_machine)
    monkeypatch.setattr(preflight, "dev_machine_marker_path", lambda: marker_path)


def test_pascal_hard_rejects_in_production(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / ".dev_machine"
    _patch_preflight(
        monkeypatch,
        inventory=[_gpu("NVIDIA GeForce GTX 1080 Ti", 6.1)],
        dev_machine=False,
        marker_path=marker,
    )

    with pytest.raises(preflight.HardwareUnsupportedError) as exc_info:
        preflight.run_preflight(python_version=(3, 11, 9))

    message = str(exc_info.value)
    assert "NVIDIA GeForce GTX 1080 Ti" in message
    assert "6.1" in message
    assert str(marker) in message
    assert os.environ.get("LSIE_DEV_FORCE_CPU_SPEECH") is None


def test_pascal_soft_warns_in_developer_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight(
        monkeypatch,
        inventory=[_gpu("NVIDIA GeForce GTX 1080 Ti", 6.1)],
        dev_machine=True,
        marker_path=tmp_path / ".dev_machine",
    )

    result = preflight.run_preflight(python_version=(3, 11, 9))

    assert result.developer_mode is True
    assert result.speech_device == "cpu"
    assert result.selected_gpu == _gpu("NVIDIA GeForce GTX 1080 Ti", 6.1)
    assert os.environ["LSIE_DEV_FORCE_CPU_SPEECH"] == "1"
    assert result.warnings[0].code == preflight_codes.PASCAL_DEV_MODE_REQUIRED
    assert "PyTorch" in result.warnings[0].message


def test_env_override_counts_as_developer_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LSIE_DEV_FORCE_CPU_SPEECH", "1")
    _patch_preflight(
        monkeypatch,
        inventory=[_gpu("NVIDIA GeForce GTX 1080 Ti", 6.1)],
        dev_machine=False,
        marker_path=tmp_path / ".dev_machine",
    )

    result = preflight.run_preflight(python_version=(3, 11, 9))

    assert result.developer_mode is True
    assert result.speech_device == "cpu"
    assert result.warnings[0].code == preflight_codes.PASCAL_DEV_MODE_REQUIRED


def test_no_gpu_hard_rejects_in_production(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / ".dev_machine"
    _patch_preflight(
        monkeypatch,
        inventory=[],
        dev_machine=False,
        marker_path=marker,
    )

    with pytest.raises(preflight.HardwareUnsupportedError) as exc_info:
        preflight.run_preflight(python_version=(3, 11, 9))

    assert "No NVIDIA GPU detected" in str(exc_info.value)
    assert str(marker) in str(exc_info.value)


def test_no_gpu_soft_warns_in_developer_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight(
        monkeypatch,
        inventory=[],
        dev_machine=True,
        marker_path=tmp_path / ".dev_machine",
    )

    result = preflight.run_preflight(python_version=(3, 11, 9))

    assert result.developer_mode is True
    assert result.speech_device == "cpu"
    assert result.selected_gpu is None
    assert os.environ["LSIE_DEV_FORCE_CPU_SPEECH"] == "1"
    assert result.warnings[0].code == preflight_codes.NO_GPU_DETECTED


def test_turing_accepts_without_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_preflight(
        monkeypatch,
        inventory=[_gpu("NVIDIA T4", 7.5)],
        dev_machine=False,
        marker_path=tmp_path / ".dev_machine",
    )

    result = preflight.run_preflight(python_version=(3, 11, 9))

    assert result.developer_mode is False
    assert result.speech_device == "cuda"
    assert result.selected_gpu == _gpu("NVIDIA T4", 7.5)
    assert result.warnings == ()
    assert os.environ.get("LSIE_DEV_FORCE_CPU_SPEECH") is None


def test_multi_gpu_selects_highest_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight(
        monkeypatch,
        inventory=[_gpu("NVIDIA GeForce GTX 1080 Ti", 6.1), _gpu("NVIDIA RTX 3060", 8.6)],
        dev_machine=False,
        marker_path=tmp_path / ".dev_machine",
    )

    result = preflight.run_preflight(python_version=(3, 11, 9))

    assert result.speech_device == "cuda"
    assert result.selected_gpu == _gpu("NVIDIA RTX 3060", 8.6)


def test_python_312_rejects_before_hardware_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: list[bool] = []

    def query_inventory() -> list[GpuInfo]:
        called.append(True)
        return [_gpu("NVIDIA T4", 7.5)]

    monkeypatch.setattr(gpu_probe, "query_gpu_inventory", query_inventory)
    monkeypatch.setattr(preflight, "dev_machine_marker_path", lambda: tmp_path / ".dev_machine")

    with pytest.raises(preflight.PythonVersionUnsupportedError) as exc_info:
        preflight.run_preflight(python_version=(3, 12, 0))

    assert "Python 3.11.x" in str(exc_info.value)
    assert "3.12.0" in str(exc_info.value)
    assert called == []


def test_python_310_rejects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_preflight(
        monkeypatch,
        inventory=[_gpu("NVIDIA T4", 7.5)],
        dev_machine=False,
        marker_path=tmp_path / ".dev_machine",
    )

    with pytest.raises(preflight.PythonVersionUnsupportedError):
        preflight.run_preflight(python_version=(3, 10, 13))
