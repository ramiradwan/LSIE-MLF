from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.processes import operator_api_runtime


class FakeQueue:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, payload: object) -> None:
        self.items.append(payload)


class FakeThread:
    def __init__(self, *, target: object, name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.join_timeout: float | None = None

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout


class FakeUvicornServer:
    def __init__(self, _config: object) -> None:
        self.started = True
        self.should_exit = False

    def run(self) -> None:
        return


class FakeHeartbeat:
    def __init__(self, db_path: object, process_key: str) -> None:
        self.db_path = db_path
        self.process_key = process_key
        self.started = False
        self.stopped = False
        created_heartbeats.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeShutdownEvent:
    def __init__(self) -> None:
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls > 1


created_heartbeats: list[FakeHeartbeat] = []
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DESKTOP_PROCESS_DIR = _REPO_ROOT / "services" / "desktop_app" / "processes"
_ALLOWED_ROUTE_IMPORTER = "services/desktop_app/processes/operator_api_runtime.py"


def _imports_module(path: Path, module_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module_name for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == module_name:
            return True
    return False


def _channels(
    *,
    live_control: FakeQueue | None = None,
    segment_control: FakeQueue | None = None,
) -> IpcChannels:
    queue = FakeQueue()
    return IpcChannels(
        ml_inbox=cast(Any, queue),
        drift_updates=cast(Any, queue),
        analytics_inbox=cast(Any, queue),
        pcm_acks=cast(Any, queue),
        live_control=cast(Any, live_control),
        segment_control=cast(Any, segment_control),
    )


def test_only_operator_api_runtime_imports_retained_fastapi_app_in_desktop_processes() -> None:
    offenders: list[str] = []
    for path in sorted(_DESKTOP_PROCESS_DIR.glob("*.py")):
        rel_path = path.relative_to(_REPO_ROOT).as_posix()
        if rel_path == _ALLOWED_ROUTE_IMPORTER:
            continue
        if _imports_module(path, "services.api.main"):
            offenders.append(rel_path)

    assert offenders == []


def test_control_publisher_serializes_to_live_and_segment_queues() -> None:
    live_queue = FakeQueue()
    segment_queue = FakeQueue()
    publisher = operator_api_runtime._QueueLiveSessionControlPublisher(
        _channels(live_control=live_queue, segment_control=segment_queue)
    )
    message = LiveSessionControlMessage(
        action="start",
        session_id=UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="android://device",
        experiment_id="greeting_line_v1",
        active_arm="compliment_content",
        expected_greeting="Love the energy on this stream!",
        timestamp_utc=datetime(2026, 5, 5, 12, 0, tzinfo=UTC),
    )

    publisher.publish(message)

    expected = message.model_dump(mode="json")
    assert live_queue.items == [expected]
    assert segment_queue.items == [expected]


def _imported_qt_roots(target_import: str) -> list[str]:
    code = (
        "import sys, json\n"
        f"import {target_import}\n"
        "found = sorted({k.split('.')[0] for k in sys.modules if k.split('.')[0] == 'PySide6'})\n"
        "print(json.dumps(found))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return list(json.loads(proc.stdout.strip()))


# Qt-heavy test conftests import PySide6 during collection, so this canary
# must use a clean interpreter instead of the shared pytest process.
def test_importing_operator_api_runtime_does_not_import_pyside() -> None:
    assert _imported_qt_roots("services.desktop_app.processes.operator_api_runtime") == []


def test_start_operator_api_runtime_wires_api_and_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []
    created_heartbeats.clear()

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("uvicorn_config", kwargs))

    fake_uvicorn = SimpleNamespace(Config=FakeConfig, Server=FakeUvicornServer)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(
        "services.desktop_app.processes.operator_api_runtime.threading.Thread",
        FakeThread,
    )
    monkeypatch.setattr(
        "services.desktop_app.processes.operator_api_runtime.time.monotonic",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        "services.desktop_app.processes.operator_api_runtime.time.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(operator_api_runtime, "_allocate_port", lambda _preferred: 8123)
    monkeypatch.setenv("LSIE_API_PORT", "8123")
    monkeypatch.setattr(
        "services.desktop_app.os_adapter.resolve_state_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_api_overrides.bootstrap_sqlite_api_store",
        lambda db_path: calls.append(("bootstrap", db_path)),
    )
    monkeypatch.setattr(
        "services.desktop_app.state.sqlite_api_overrides.configure_sqlite_api_overrides",
        lambda api_app, db_path, *, control_publisher: calls.append(
            ("configure", (api_app, db_path, control_publisher))
        ),
    )
    monkeypatch.setattr(
        "services.desktop_app.state.heartbeats.HeartbeatRecorder",
        FakeHeartbeat,
    )

    runtime = operator_api_runtime.start_operator_api_runtime(_channels())
    runtime.stop()

    assert os.environ["LSIE_API_URL"] == "http://127.0.0.1:8123"
    assert calls[0] == ("bootstrap", tmp_path / "desktop.sqlite")
    assert calls[1][0] == "configure"
    assert calls[2][0] == "uvicorn_config"
    assert created_heartbeats[0].process_key == "ui_api_shell"
    assert created_heartbeats[0].started is True
    assert created_heartbeats[0].stopped is True
    assert runtime._uv_server.should_exit is True
    assert cast(FakeThread, runtime._uv_thread).join_timeout == 5.0


def test_run_waits_for_shutdown_and_stops_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeRuntime:
        api_url = "http://127.0.0.1:8000"

        def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(
        operator_api_runtime,
        "start_operator_api_runtime",
        lambda _channels: FakeRuntime(),
    )
    monkeypatch.setattr(
        "services.desktop_app.processes.operator_api_runtime.time.sleep",
        lambda _seconds: None,
    )

    operator_api_runtime.run(cast(Any, FakeShutdownEvent()), _channels())

    assert events == ["stop"]
