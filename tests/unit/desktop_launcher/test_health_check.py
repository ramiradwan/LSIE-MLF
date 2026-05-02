"""WS1 P2 — runtime health check and handoff tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from services.desktop_launcher import health_check, manifest


def test_runtime_python_prefers_staged_venv(tmp_path: Path) -> None:
    if sys.platform == "win32":
        python_exe = tmp_path / ".venv" / "Scripts" / "python.exe"
    else:
        python_exe = tmp_path / ".venv" / "bin" / "python3"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    assert health_check.runtime_python(tmp_path) == python_exe


def test_run_runtime_smoke_test_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr="missing torch",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(health_check.RuntimeHealthCheckError, match="missing torch"):
        health_check.run_runtime_smoke_test(tmp_path)


def test_run_runtime_smoke_test_returns_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="lsie-mlf runtime smoke ok\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert health_check.run_runtime_smoke_test(tmp_path) == "lsie-mlf runtime smoke ok"


def test_launch_desktop_app_uses_hydrated_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_exe = (
        tmp_path
        / ".venv"
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python3")
    )
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd: list[str], **_kwargs: object) -> None:
            calls.append(cmd)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    health_check.launch_desktop_app(tmp_path)
    assert calls == [[str(python_exe), "-m", "services.desktop_app"]]


def test_finalize_install_promotes_staging_and_writes_manifest(tmp_path: Path) -> None:
    staging = tmp_path / "runtime.staging"
    active = tmp_path / "runtime"
    staging.mkdir()
    (staging / "payload.txt").write_text("ok", encoding="utf-8")

    result = health_check.finalize_install(
        staging_dir=staging,
        active_runtime_dir=active,
        python_runtime="cpython-3.11.15+20260414",
        scrcpy_version="v3.3.4",
    )

    assert result == active
    assert not staging.exists()
    assert (active / "payload.txt").read_text(encoding="utf-8") == "ok"
    loaded = manifest.read_manifest(active)
    assert loaded.python_runtime == "cpython-3.11.15+20260414"
    assert loaded.scrcpy_version == "v3.3.4"
