"""Runtime health check and handoff tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from services.desktop_app import os_adapter
from services.desktop_app.startup_timing import STARTUP_EPOCH_ENV
from services.desktop_launcher import health_check, manifest, preflight


def _assert_subprocess_policy(kwargs: dict[str, object]) -> None:
    if sys.platform == "win32":
        creationflags = cast(int, kwargs["creationflags"])
        assert creationflags & subprocess.CREATE_NO_WINDOW
        assert creationflags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert "creationflags" not in kwargs


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
        _assert_subprocess_policy(kwargs)
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
        _assert_subprocess_policy(kwargs)
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="lsie-mlf runtime smoke ok\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert health_check.run_runtime_smoke_test(tmp_path) == "lsie-mlf runtime smoke ok"


def test_describe_missing_desktop_tooling_uses_shared_contract() -> None:
    missing = (
        os_adapter.MissingExternalTool(
            spec=os_adapter.ADB_TOOL,
            resolver_detail="could not locate executable 'adb': PATH lookup failed",
        ),
        os_adapter.MissingExternalTool(
            spec=os_adapter.SCRCPY_TOOL,
            resolver_detail=(
                "LSIE_SCRCPY_PATH='C:/missing/scrcpy.exe' does not point at an existing file"
            ),
        ),
    )

    message = health_check.describe_missing_desktop_tooling(missing)

    assert "Missing required external tools:" in message
    assert "Android Device Bridge (adb) is unavailable" in message
    assert "scrcpy is unavailable" in message
    assert "Install Android Platform Tools or set LSIE_ADB_PATH" in message
    assert "Install scrcpy or set LSIE_SCRCPY_PATH" in message


def test_build_source_launch_command_uses_hydrated_runtime(
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
    app_root = tmp_path / "app"
    (app_root / "services" / "desktop_app").mkdir(parents=True)
    (app_root / "services" / "desktop_app" / "__main__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", f"old{os.pathsep}path")
    monkeypatch.setattr(
        health_check,
        "ensure_startup_epoch",
        lambda env: env.__setitem__(STARTUP_EPOCH_ENV, "123456789"),
    )
    monkeypatch.setattr(
        health_check,
        "resolve_desktop_runtime_tools",
        lambda: ({}, ()),
    )

    command, cwd, env = health_check.build_source_launch_command(
        tmp_path,
        app_root=app_root,
        module_args=("--operator-api",),
    )

    assert command == [str(python_exe), "-m", "services.desktop_app", "--operator-api"]
    assert cwd == app_root
    assert env["PYTHONPATH"] == f"{app_root}{os.pathsep}old{os.pathsep}path"
    assert env[STARTUP_EPOCH_ENV] == "123456789"


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
    calls: list[tuple[list[str], Path, dict[str, str], dict[str, object]]] = []
    log_path = tmp_path / "desktop-launch.log"
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
                    kwargs,
                )
            )

    monkeypatch.setattr(
        preflight,
        "ensure_preflight",
        lambda: preflight_calls.append("called"),
    )
    monkeypatch.setattr(health_check, "_launch_log_path", lambda: log_path)
    monkeypatch.setattr(
        health_check,
        "format_startup_milestone",
        lambda milestone, *, environ=None, now_ns=None: (
            f"startup milestone={milestone} elapsed_ms=12.3"
        ),
    )
    monkeypatch.setattr(
        health_check,
        "resolve_desktop_runtime_tools",
        lambda: (
            {
                "adb": "/resolved/adb",
                "scrcpy": "/resolved/scrcpy",
                "ffmpeg": "/resolved/ffmpeg",
            },
            (),
        ),
    )
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    health_check.launch_desktop_app(tmp_path, app_root=app_root)

    assert preflight_calls == ["called"]
    cmd, cwd, env, kwargs = calls[0]
    assert cmd == [str(python_exe), "-m", "services.desktop_app"]
    assert cwd == app_root
    assert env["PYTHONPATH"] == str(app_root)
    assert env["LSIE_ADB_PATH"] == "/resolved/adb"
    assert env["LSIE_SCRCPY_PATH"] == "/resolved/scrcpy"
    assert env["LSIE_FFMPEG_PATH"] == "/resolved/ffmpeg"
    assert STARTUP_EPOCH_ENV in env
    assert kwargs["stdout"] is not None
    assert kwargs["stderr"] == subprocess.STDOUT
    assert log_path.read_text(encoding="utf-8").splitlines()[-1] == (
        "startup milestone=launcher_handoff elapsed_ms=12.3"
    )
    _assert_subprocess_policy(kwargs)


def test_launch_desktop_app_rejects_invalid_app_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(preflight, "ensure_preflight", lambda: None)
    monkeypatch.setattr(health_check, "resolve_desktop_runtime_tools", lambda: ({}, ()))

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
