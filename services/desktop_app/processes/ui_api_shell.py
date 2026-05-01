"""UI + API shell process (v4.0 §9 / WS3 P1b + WS4 P1b).

Hosts the in-process FastAPI server (``services.api.main.app``) on a
daemon thread bound to ``127.0.0.1:8000`` and the PySide6 ``MainWindow``
on the main thread. The console's API client points at the same loopback
port via ``LSIE_API_URL`` so the existing
``services.operator_console.api_client`` reaches the in-process API
without code changes.

WS4 P1b wires the FastAPI route layer to the local SQLite store:

* Bootstrap ``services.desktop_app.state.sqlite_schema`` under the
  resolved app-data directory if the file is fresh.
* Replace ``services.api.main.app``'s lifespan (which calls
  ``init_pool()`` on the psycopg2 pool) with a no-op — the desktop
  runtime owns its persistence layer through the SQLite store.
* Override ``services.api.routes.operator.get_read_service`` with
  :class:`SqliteOperatorReadService` so the operator console's
  ``/api/v1/operator/*`` aggregate endpoints render real seed-experiment
  rows + any analytics_state_worker writes that have landed.

ML import discipline: this module MUST NOT import ``torch`` /
``mediapipe`` / ``faster_whisper`` / ``ctranslate2`` at any scope. The
heavy framework imports (FastAPI, uvicorn, PySide6) are deferred into
``run`` so the canary subprocess that imports this module pays only the
stdlib cost.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync
import os
import socket
import threading
import time

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.cleanup import recover_orphan_ipc_blocks

logger = logging.getLogger(__name__)

API_HOST = "127.0.0.1"
API_PORT_PREFERRED = 8000
SHUTDOWN_POLL_INTERVAL_MS = 250
UVICORN_READY_TIMEOUT_S = 5.0
SQLITE_FILENAME = "desktop.sqlite"


def _allocate_port(preferred: int) -> int:
    """Return ``preferred`` if free, else an OS-assigned ephemeral port.

    Brief TOCTOU window between our close and uvicorn's bind — acceptable
    for a dev-mode smoke surface where the only goal is "GUI pops and
    talks to a local FastAPI". Production packaging (WS1 P2) can pin a
    fixed port via ``LSIE_API_PORT`` if the operator host requires it.
    """
    for candidate in (preferred, 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((API_HOST, candidate))
            port = int(sock.getsockname()[1])
            sock.close()
            return port
        except OSError:
            sock.close()
    raise RuntimeError("ui_api_shell: failed to allocate any local port")


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    del channels  # ui_api_shell does not consume IPC channels directly.
    logger.info("ui_api_shell starting")
    recover_orphan_ipc_blocks()

    # Late imports — preserves the WS3 P1 ML-isolation canary contract
    # and keeps the parent process free of FastAPI/Qt state.
    import sqlite3
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from services.api.main import app as api_app
    from services.api.routes.operator import get_read_service
    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.sqlite_operator_read_service import (
        SqliteOperatorReadService,
    )
    from services.desktop_app.state.sqlite_reader import SqliteReader
    from services.desktop_app.state.sqlite_schema import bootstrap_schema
    from services.operator_console.app import (
        build_api_client,
        build_main_window,
        build_polling_coordinator,
        build_store,
    )
    from services.operator_console.config import load_config
    from services.operator_console.theme import build_stylesheet

    # ------------------------------------------------------------------
    # WS4 P1b — local SQLite bootstrap + read service wiring
    # ------------------------------------------------------------------
    # Bootstrap the schema (and the four-arm seed) under the platform's
    # standard application-data directory. Idempotent: re-running on a
    # fresh install creates the file and tables; re-running on a
    # populated store is a no-op (CREATE TABLE IF NOT EXISTS + INSERT
    # OR IGNORE on the seed). The writer-only connection lives just
    # long enough to bootstrap; the long-lived writer is owned by
    # analytics_state_worker.
    state_dir = resolve_state_dir()
    db_path = state_dir / SQLITE_FILENAME
    bootstrap_conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(bootstrap_conn)
    finally:
        bootstrap_conn.close()
    logger.info("desktop sqlite store bootstrapped at %s", db_path)

    reader = SqliteReader(db_path)
    read_service = SqliteOperatorReadService(reader)

    def _read_service_dependency() -> SqliteOperatorReadService:
        return read_service

    api_app.dependency_overrides[get_read_service] = _read_service_dependency

    # The v3.4 lifespan in services.api.main calls init_pool() which
    # KeyErrors when POSTGRES_USER is unset. The desktop runtime owns
    # its persistence layer through SqliteReader/Writer, so the FastAPI
    # lifespan becomes a no-op.
    @asynccontextmanager
    async def _desktop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("ui_api_shell FastAPI lifespan: SQLite-backed (Postgres pool skipped)")
        yield

    api_app.router.lifespan_context = _desktop_lifespan

    # Resolve port: env override → preferred 8000 → ephemeral fallback.
    requested_port_raw = os.environ.get("LSIE_API_PORT", "").strip()
    requested_port = int(requested_port_raw) if requested_port_raw else API_PORT_PREFERRED
    port = _allocate_port(requested_port)
    if port != requested_port:
        logger.warning("requested port %d unavailable; using %d", requested_port, port)

    # Point the operator console's API client at our loopback uvicorn.
    api_url = f"http://{API_HOST}:{port}"
    os.environ["LSIE_API_URL"] = api_url

    # uvicorn must run off the main thread because Qt owns the main
    # thread for the duration of the GUI session. lifespan="on" forces
    # ASGI lifespan dispatch even though our patched lifespan is a
    # no-op — keeps behaviour explicit when WS4 P1 lands.
    uv_config = uvicorn.Config(
        app=api_app,
        host=API_HOST,
        port=port,
        log_level="warning",
        lifespan="on",
    )
    uv_server = uvicorn.Server(uv_config)
    uv_thread = threading.Thread(target=uv_server.run, name="uvicorn-server", daemon=True)
    uv_thread.start()

    # Wait briefly for uvicorn's startup to complete; logs a warning if
    # the server never reports ready (e.g. port already in use). The
    # GUI still launches so the operator can see the failure surface.
    deadline = time.monotonic() + UVICORN_READY_TIMEOUT_S
    while time.monotonic() < deadline and not uv_server.started:
        time.sleep(0.05)
    if uv_server.started:
        logger.info("uvicorn listening on %s", api_url)
    else:
        logger.warning("uvicorn did not report ready within %.1fs", UVICORN_READY_TIMEOUT_S)

    qt_app = QApplication([])
    qt_app.setApplicationName("LSIE-MLF Operator Console")
    qt_app.setOrganizationName("LSIE-MLF")
    qt_app.setStyleSheet(build_stylesheet())

    config = load_config()
    store = build_store()
    client = build_api_client(config)
    coordinator = build_polling_coordinator(config, client, store)
    window = build_main_window(config, store, coordinator)
    window.show()
    coordinator.start()
    qt_app.aboutToQuit.connect(coordinator.stop)

    # Bridge the multiprocessing shutdown event to QApplication.quit so
    # the parent's stop_all() reaches the Qt event loop.
    def _check_shutdown() -> None:
        if shutdown_event.is_set():
            qt_app.quit()

    shutdown_timer = QTimer()
    shutdown_timer.setInterval(SHUTDOWN_POLL_INTERVAL_MS)
    shutdown_timer.timeout.connect(_check_shutdown)
    shutdown_timer.start()

    logger.info("ui_api_shell Qt event loop active")
    qt_app.exec()

    uv_server.should_exit = True
    uv_thread.join(timeout=5.0)
    logger.info("ui_api_shell stopped")
