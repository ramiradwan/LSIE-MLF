"""Loopback operator API/control runtime for CLI and GUI shells."""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Protocol, cast

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage

logger = logging.getLogger(__name__)

API_HOST = "127.0.0.1"
API_PORT_PREFERRED = 8000
SHUTDOWN_POLL_INTERVAL_S = 0.25
UVICORN_READY_TIMEOUT_S = 5.0
SQLITE_FILENAME = "desktop.sqlite"
HEARTBEAT_PROCESS_KEY = "ui_api_shell"


class _UvicornServer(Protocol):
    started: bool
    should_exit: bool

    def run(self) -> None: ...


class _Heartbeat(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class _QueueLiveSessionControlPublisher:
    def __init__(self, channels: IpcChannels) -> None:
        self._queues = tuple(
            queue
            for queue in (channels.live_control, channels.segment_control)
            if queue is not None
        )

    def publish(self, message: LiveSessionControlMessage) -> None:
        payload = message.model_dump(mode="json")
        for queue in self._queues:
            queue.put(payload)


@dataclass
class RunningOperatorApiRuntime:
    api_url: str
    _uv_server: _UvicornServer
    _uv_thread: threading.Thread
    _heartbeat: _Heartbeat

    def stop(self) -> None:
        self._uv_server.should_exit = True
        self._uv_thread.join(timeout=5.0)
        self._heartbeat.stop()


def _allocate_port(preferred: int) -> int:
    for candidate in (preferred, 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((API_HOST, candidate))
            port = int(sock.getsockname()[1])
            sock.close()
            return port
        except OSError:
            sock.close()
    raise RuntimeError("operator API runtime failed to allocate any local port")


def start_operator_api_runtime(channels: IpcChannels) -> RunningOperatorApiRuntime:
    import uvicorn

    from services.api.main import app as api_app
    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder
    from services.desktop_app.state.sqlite_api_overrides import (
        bootstrap_sqlite_api_store,
        configure_sqlite_api_overrides,
    )

    state_dir = resolve_state_dir()
    db_path = state_dir / SQLITE_FILENAME
    bootstrap_sqlite_api_store(db_path)
    logger.info("desktop sqlite store bootstrapped at %s", db_path)
    configure_sqlite_api_overrides(
        api_app,
        db_path,
        control_publisher=_QueueLiveSessionControlPublisher(channels),
    )

    heartbeat = HeartbeatRecorder(db_path, HEARTBEAT_PROCESS_KEY)
    heartbeat.start()

    requested_port_raw = os.environ.get("LSIE_API_PORT", "").strip()
    requested_port = int(requested_port_raw) if requested_port_raw else API_PORT_PREFERRED
    port = _allocate_port(requested_port)
    if port != requested_port:
        logger.warning("requested port %d unavailable; using %d", requested_port, port)

    api_url = f"http://{API_HOST}:{port}"
    os.environ["LSIE_API_URL"] = api_url

    uv_config = uvicorn.Config(
        app=api_app,
        host=API_HOST,
        port=port,
        log_level="warning",
        lifespan="on",
    )
    uv_server = cast(_UvicornServer, uvicorn.Server(uv_config))
    uv_thread = threading.Thread(target=uv_server.run, name="uvicorn-server", daemon=True)
    uv_thread.start()

    deadline = time.monotonic() + UVICORN_READY_TIMEOUT_S
    while time.monotonic() < deadline and not uv_server.started:
        time.sleep(0.05)
    if uv_server.started:
        logger.info("uvicorn listening on %s", api_url)
    else:
        logger.warning("uvicorn did not report ready within %.1fs", UVICORN_READY_TIMEOUT_S)

    return RunningOperatorApiRuntime(
        api_url=api_url,
        _uv_server=uv_server,
        _uv_thread=uv_thread,
        _heartbeat=heartbeat,
    )


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("operator API runtime starting")
    runtime = start_operator_api_runtime(channels)
    try:
        while not shutdown_event.is_set():
            time.sleep(SHUTDOWN_POLL_INTERVAL_S)
    finally:
        runtime.stop()
        logger.info("operator API runtime stopped")
