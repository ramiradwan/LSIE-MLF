from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest

import scripts.run_real_device_benchmark as benchmark
from packages.schemas.operator_console import SessionCreateRequest, SessionEndRequest
from services.desktop_app import os_adapter
from services.desktop_launcher import health_check

_NOW = datetime(2026, 5, 12, 12, 0, tzinfo=UTC)
_SESSION_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")


class _FakePopen:
    def __init__(self, cmd: list[str], **kwargs: object) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 4242


class _FakeClient:
    instances: list[_FakeClient] = []

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.started_requests: list[SessionCreateRequest] = []
        self.ended_requests: list[tuple[uuid.UUID, SessionEndRequest]] = []
        _FakeClient.instances.append(self)

    def post_session_start(
        self,
        request: SessionCreateRequest,
    ) -> SimpleNamespace:
        self.started_requests.append(request)
        return SimpleNamespace(
            accepted=True,
            message="session start accepted",
            session_id=_SESSION_ID,
        )

    def post_session_end(
        self,
        session_id: uuid.UUID,
        request: SessionEndRequest,
    ) -> SimpleNamespace:
        self.ended_requests.append((session_id, request))
        return SimpleNamespace(
            accepted=True,
            message="session end accepted",
            session_id=session_id,
        )


class _FakeWatcher:
    instances: list[_FakeWatcher] = []

    def __init__(
        self,
        adb_path: str,
        interval_s: float = benchmark.MEDIA_WATCHER_INTERVAL_S,
    ) -> None:
        self.adb_path = adb_path
        self.interval_s = interval_s
        self.started = False
        _FakeWatcher.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> tuple[tuple[datetime, datetime | None], ...]:
        return ()


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeClient.instances.clear()
    _FakeWatcher.instances.clear()


def test_spawn_desktop_app_uses_shared_launch_command_and_tool_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}
    stdout_handles: list[BinaryIO] = []

    def fake_build_source_launch_command(
        runtime_dir: Path,
        *,
        app_root: Path | None = None,
        module_args: tuple[str, ...] = (),
    ) -> tuple[list[str], Path, dict[str, str]]:
        observed["build"] = (runtime_dir, app_root, module_args)
        return (
            ["python", "-m", "services.desktop_app", "--operator-api"],
            tmp_path,
            {"PYTHONPATH": "existing-path", "BASE_ENV": "1"},
        )

    def fake_apply_windows_child_process_policy(
        popen_kwargs: dict[str, Any],
        *,
        hide_window: bool = True,
    ) -> dict[str, Any]:
        observed["policy"] = (dict(popen_kwargs), hide_window)
        return {**popen_kwargs, "creationflags": 123}

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        observed["popen"] = (cmd, kwargs)
        stdout = cast(BinaryIO, kwargs["stdout"])
        stdout_handles.append(stdout)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(
        health_check,
        "build_source_launch_command",
        fake_build_source_launch_command,
    )
    monkeypatch.setattr(
        os_adapter,
        "_apply_windows_child_process_policy",
        fake_apply_windows_child_process_policy,
    )
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "platform", "linux")

    log_path = tmp_path / "desktop.log"
    tooling = benchmark.RealDeviceTooling(
        adb_path="/resolved/adb",
        scrcpy_path="/resolved/scrcpy",
        ffmpeg_path="/resolved/ffmpeg",
    )

    proc, job_handle = benchmark._spawn_desktop_app(
        8765,
        log_path,
        tooling,
        runtime_dir=tmp_path / "runtime",
    )

    assert observed["build"] == (
        tmp_path / "runtime",
        benchmark.REPO_ROOT,
        ("--operator-api",),
    )
    policy_kwargs, hide_window = observed["policy"]
    assert hide_window is False
    assert policy_kwargs["cwd"] == tmp_path
    env = policy_kwargs["env"]
    assert isinstance(env, dict)
    assert env["PYTHONPATH"] == "existing-path"
    assert env["BASE_ENV"] == "1"
    assert env["LSIE_API_PORT"] == "8765"
    assert env["LSIE_ADB_PATH"] == "/resolved/adb"
    assert env["LSIE_SCRCPY_PATH"] == "/resolved/scrcpy"
    assert env["LSIE_FFMPEG_PATH"] == "/resolved/ffmpeg"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] == "1"

    cmd, kwargs = observed["popen"]
    assert cmd == ["python", "-m", "services.desktop_app", "--operator-api"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["creationflags"] == 123
    stdout = kwargs["stdout"]
    assert getattr(stdout, "name", None) == str(log_path)
    assert log_path.exists()
    launch_log = log_path.read_text(encoding="utf-8")
    assert "startup milestone=launcher_handoff" in launch_log
    assert proc.pid == 4242
    assert job_handle is None

    for handle in stdout_handles:
        handle.close()


def test_run_benchmark_resolves_tooling_through_shared_runtime_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observe_calls: list[str] = []
    spawn_calls: list[tuple[int, Path, benchmark.RealDeviceTooling]] = []
    stop_calls: list[tuple[int, object | None]] = []
    wait_calls: list[tuple[str, object]] = []
    proc = _FakePopen([], stdout=None)
    measurement = benchmark.StimulusMeasurement(
        client_action_id=uuid.UUID("11111111-1111-4111-8111-111111111111"),
        submitted_at=_NOW,
        encounter_visible_at=_NOW,
        stimulus_time_utc=_NOW,
        segment_timestamp_utc=_NOW,
        end_to_end_s=0.5,
        segment_to_visible_ms=25.0,
    )

    def fake_ensure_adb_device(adb_path: str) -> str:
        assert adb_path == "/resolved/adb"
        return "serial-1"

    def fake_observe_tiktok_state(adb_path: str) -> benchmark.MediaState:
        observe_calls.append(adb_path)
        return benchmark.MediaState(foregrounded=True, audio_started=True)

    def fake_spawn_desktop_app(
        api_port: int,
        log_path: Path,
        tooling: benchmark.RealDeviceTooling,
        *,
        runtime_dir: Path = benchmark.REPO_ROOT,
    ) -> tuple[_FakePopen, object | None]:
        del runtime_dir
        spawn_calls.append((api_port, log_path, tooling))
        return proc, None

    def fake_stop_desktop_app(current_proc: _FakePopen, job_handle: object | None = None) -> None:
        stop_calls.append((current_proc.pid, job_handle))

    def fake_wait_until_api_up(client: _FakeClient, deadline: float) -> None:
        wait_calls.append(("api", client.base_url))
        assert deadline > 0

    def fake_wait_until_capture_ready(
        client: _FakeClient,
        session_id: uuid.UUID,
        deadline: float,
    ) -> SimpleNamespace:
        wait_calls.append(("capture", session_id))
        assert client.base_url == "http://127.0.0.1:8765"
        assert deadline > 0
        return SimpleNamespace(is_calibrating=False)

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
    monkeypatch.setattr(benchmark, "_ensure_adb_device", fake_ensure_adb_device)
    monkeypatch.setattr(benchmark, "_observe_tiktok_state", fake_observe_tiktok_state)
    monkeypatch.setattr(benchmark, "_spawn_desktop_app", fake_spawn_desktop_app)
    monkeypatch.setattr(benchmark, "_stop_desktop_app", fake_stop_desktop_app)
    monkeypatch.setattr(benchmark, "ApiClient", _FakeClient)
    monkeypatch.setattr(benchmark, "MediaWatcher", _FakeWatcher)
    monkeypatch.setattr(benchmark, "_wait_until_api_up", fake_wait_until_api_up)
    monkeypatch.setattr(benchmark, "_wait_until_capture_ready", fake_wait_until_capture_ready)
    monkeypatch.setattr(benchmark, "_drive_stimulus", lambda *_args: measurement)

    result = benchmark.run_benchmark(
        stimuli=1,
        api_port=8765,
        stream_url="tiktok://benchmark.local/real-device",
        experiment_id="greeting_line_v1",
        log_path=tmp_path / "real-device.log",
    )

    assert spawn_calls == [
        (
            8765,
            tmp_path / "real-device.log",
            benchmark.RealDeviceTooling(
                adb_path="/resolved/adb",
                scrcpy_path="/resolved/scrcpy",
                ffmpeg_path="/resolved/ffmpeg",
            ),
        )
    ]
    assert len(_FakeClient.instances) == 1
    client = _FakeClient.instances[0]
    assert client.base_url == "http://127.0.0.1:8765"
    assert client.timeout_seconds == 15.0
    assert len(client.started_requests) == 1
    assert client.started_requests[0].stream_url == "tiktok://benchmark.local/real-device"
    assert client.started_requests[0].experiment_id == "greeting_line_v1"
    assert client.ended_requests and client.ended_requests[0][0] == _SESSION_ID
    assert len(_FakeWatcher.instances) == 1
    assert _FakeWatcher.instances[0].adb_path == "/resolved/adb"
    assert _FakeWatcher.instances[0].started is True
    assert wait_calls == [
        ("api", "http://127.0.0.1:8765"),
        ("capture", _SESSION_ID),
    ]
    assert observe_calls == ["/resolved/adb", "/resolved/adb"]
    assert stop_calls == [(4242, None)]
    assert result.measurements == (measurement,)
    assert result.media_pre.playing is True
    assert result.media_post.playing is True
    assert result.media_pause_intervals == ()


def test_run_benchmark_reuses_shared_missing_tool_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        health_check,
        "resolve_desktop_runtime_tools",
        lambda: (
            {},
            (
                os_adapter.MissingExternalTool(
                    spec=os_adapter.ADB_TOOL,
                    resolver_detail="could not locate executable 'adb': PATH lookup failed",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        health_check,
        "describe_missing_desktop_tooling",
        lambda missing: "shared tooling guidance",
    )

    with pytest.raises(RuntimeError, match="shared tooling guidance"):
        benchmark.run_benchmark(
            stimuli=1,
            api_port=8765,
            stream_url="tiktok://benchmark.local/real-device",
            experiment_id="greeting_line_v1",
            log_path=tmp_path / "real-device.log",
        )
