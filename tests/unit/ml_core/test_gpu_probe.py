"""WS2 P3 — shared NVIDIA GPU inventory probe tests."""

from __future__ import annotations

import subprocess

import pytest

from packages.ml_core import gpu_probe


def test_query_gpu_inventory_parses_single_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA T4, 7.5\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_gpu_inventory() == [
        gpu_probe.GpuInfo(name="NVIDIA T4", compute_cap=7.5),
    ]


def test_query_max_compute_capability_returns_max_across_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce GTX 1080 Ti, 6.1\nNVIDIA RTX 3060, 8.6\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_max_compute_capability() == 8.6


def test_missing_nvidia_smi_returns_empty_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi not on PATH")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_gpu_inventory() == []
    assert gpu_probe.query_max_compute_capability() is None


def test_nonzero_exit_returns_empty_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=9,
            stdout="",
            stderr="No devices found",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_gpu_inventory() == []


def test_timeout_returns_empty_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=5.0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_gpu_inventory() == []


def test_unparseable_rows_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="bad row\nNVIDIA T4, not-a-float\nNVIDIA RTX 3060, 8.6\n\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_gpu_inventory() == [
        gpu_probe.GpuInfo(name="NVIDIA RTX 3060", compute_cap=8.6),
    ]


def test_empty_stdout_returns_empty_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gpu_probe.query_gpu_inventory() == []
