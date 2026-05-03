"""Runtime health check and handoff tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from services.desktop_launcher import health_check, manifest, preflight


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
    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cast(int, kwargs["creationflags"]) == subprocess.CREATE_NO_WINDOW
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
    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cast(int, kwargs["creationflags"]) == subprocess.CREATE_NO_WINDOW
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="lsie-mlf runtime smoke ok\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert health_check.run_runtime_smoke_test(tmp_path) == "lsie-mlf runtime smoke ok"


def test_launch_desktop_app_runs_preflight_before_handoff(
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
    preflight_calls: list[str] = []
    calls: list[tuple[list[str], Path, dict[str, str], int, object, object]] = []
    app_root = tmp_path / "app"
    (app_root / "services" / "desktop_app").mkdir(parents=True)
    (app_root / "services" / "desktop_app" / "__main__.py").write_text("", encoding="utf-8")

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            calls.append(
                (
                    cmd,
                    cast(Path, kwargs["cwd"]),
                    cast(dict[str, str], kwargs["env"]),
                    cast(int, kwargs["creationflags"]),
                    kwargs["stdout"],
                    kwargs["stderr"],
                )
            )

    monkeypatch.setattr(
        preflight,
        "ensure_preflight",
        lambda: preflight_calls.append("called"),
    )
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    health_check.launch_desktop_app(tmp_path, app_root=app_root)

    assert preflight_calls == ["called"]
    assert calls == [
        (
            [str(python_exe), "-m", "services.desktop_app"],
            app_root,
            {**os.environ, "PYTHONPATH": str(app_root)},
            subprocess.CREATE_NO_WINDOW,
            calls[0][4],
            subprocess.STDOUT,
        )
    ]


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
    calls: list[tuple[list[str], Path, str, int]] = []
    app_root = tmp_path / "app"
    (app_root / "services" / "desktop_app").mkdir(parents=True)
    (app_root / "services" / "desktop_app" / "__main__.py").write_text("", encoding="utf-8")
    existing_pythonpath = f"old{os.pathsep}path"
    monkeypatch.setenv("PYTHONPATH", existing_pythonpath)

    class FakePopen:
        def __init__(self, cmd: list[str], **kwargs: object) -> None:
            calls.append(
                (
                    cmd,
                    cast(Path, kwargs["cwd"]),
                    cast(dict[str, str], kwargs["env"])["PYTHONPATH"],
                    cast(int, kwargs["creationflags"]),
                )
            )

    monkeypatch.setattr(preflight, "ensure_preflight", lambda: None)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    health_check.launch_desktop_app(tmp_path, app_root=app_root)
    assert calls == [
        (
            [str(python_exe), "-m", "services.desktop_app"],
            app_root,
            f"{app_root}{os.pathsep}{existing_pythonpath}",
            subprocess.CREATE_NO_WINDOW,
        )
    ]


def test_launch_desktop_app_rejects_invalid_app_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(preflight, "ensure_preflight", lambda: None)

    with pytest.raises(RuntimeError, match="LSIE-MLF app source root was not found"):
        health_check.launch_desktop_app(tmp_path, app_root=tmp_path / "dist" / "LSIE-MLF-Launcher")


def test_finalize_install_promotes_staging_and_writes_manifest(tmp_path: Path) -> None:
    staging = tmp_path / "runtime.staging"
    active = tmp_path / "runtime"
    staging.mkdir()
    (staging / "payload.txt").write_text("ok", encoding="utf-8")
    pyvenv_cfg = staging / ".venv" / "pyvenv.cfg"
    pyvenv_cfg.parent.mkdir()
    pyvenv_cfg.write_text(
        f"home = {staging / 'python' / 'python'}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    result = health_check.finalize_install(
        staging_dir=staging,
        active_runtime_dir=active,
        python_runtime="cpython-3.11.15+20260414",
        scrcpy_version="v3.3.4",
    )

    assert result == active
    assert not staging.exists()
    assert (active / "payload.txt").read_text(encoding="utf-8") == "ok"
    assert (active / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()[0] == (
        f"home = {active / 'python' / 'python'}"
    )
    loaded = manifest.read_manifest(active)
    assert loaded.python_runtime == "cpython-3.11.15+20260414"
    assert loaded.scrcpy_version == "v3.3.4"
