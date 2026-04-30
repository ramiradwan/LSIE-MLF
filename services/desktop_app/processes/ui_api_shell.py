"""UI + API shell process (v4.0 §9 / WS3 P1b).

Hosts the in-process FastAPI server (``services.api.main.app``) on a
daemon thread bound to ``127.0.0.1:8000`` and the PySide6 ``MainWindow``
on the main thread. The console's API client points at the same loopback
port via ``LSIE_API_URL`` so the existing
``services.operator_console.api_client`` reaches the in-process API
without code changes.

Phase 1b transitional accommodation: if ``POSTGRES_USER`` is unset, the
FastAPI app's lifespan is replaced with a no-op so uvicorn can start
without a backing Persistent Store. This lets the GUI render the
empty/default state on a developer host with no Postgres running. WS4
P1 swaps in the SQLite-backed lifespan.

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

logger = logging.getLogger(__name__)

API_HOST = "127.0.0.1"
API_PORT_PREFERRED = 8000
SHUTDOWN_POLL_INTERVAL_MS = 250
UVICORN_READY_TIMEOUT_S = 5.0


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


class _EmptyCursor:
    """psycopg2-shaped cursor that returns no rows for every query.

    Phase 1b transitional stand-in for ``services.api.db.connection``
    when ``POSTGRES_USER`` is unset. Read routes get empty payloads
    instead of ``RuntimeError("Connection pool not initialized")`` so
    the operator console renders its empty/default state. WS4 P1
    replaces this with the real SQLite reader.
    """

    rowcount = 0
    description: object = None

    def execute(self, *_args: object, **_kwargs: object) -> None:
        return None

    def executemany(self, *_args: object, **_kwargs: object) -> None:
        return None

    def fetchall(self) -> list[object]:
        return []

    def fetchone(self) -> object | None:
        return None

    def fetchmany(self, *_args: object, **_kwargs: object) -> list[object]:
        return []

    def close(self) -> None:
        return None

    def __enter__(self) -> _EmptyCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> _EmptyCursor:
        return self

    def __next__(self) -> object:
        raise StopIteration


class _EmptyConnection:
    """psycopg2-shaped connection that hands out :class:`_EmptyCursor`."""

    def cursor(self, *_args: object, **_kwargs: object) -> _EmptyCursor:
        return _EmptyCursor()

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> _EmptyConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _EmptyPool:
    """Pool that vends :class:`_EmptyConnection`. ``closeall`` is a no-op."""

    def getconn(self) -> _EmptyConnection:
        return _EmptyConnection()

    def putconn(self, _conn: object) -> None:
        return None

    def closeall(self) -> None:
        return None


def run(shutdown_event: mpsync.Event) -> None:
    logger.info("ui_api_shell starting")

    # Late imports — preserves the WS3 P1 ML-isolation canary contract
    # and keeps the parent process free of FastAPI/Qt state.
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from services.api.main import app as api_app
    from services.operator_console.app import (
        build_api_client,
        build_main_window,
        build_polling_coordinator,
        build_store,
    )
    from services.operator_console.config import load_config
    from services.operator_console.theme import build_stylesheet

    # The v3.4 lifespan in services.api.main calls init_pool() which
    # KeyErrors when POSTGRES_USER is unset, and individual route
    # handlers call get_connection() which raises until the pool is
    # initialized. WS4 P1 replaces both with a SQLite-backed lifespan;
    # until then, install an empty-pool stand-in so the GUI renders its
    # default state instead of "Connection pool not initialized" errors.
    if not os.environ.get("POSTGRES_USER"):
        from services.api.db import connection as _db_conn
        from services.api.routes.operator import get_read_service
        from services.api.services.operator_read_service import OperatorReadService

        _db_conn._pool = _EmptyPool()  # type: ignore[attr-defined]

        # The /api/v1/operator/* routes wire in _default_redis_factory
        # which lazy-imports `redis`. The desktop runtime does not ship
        # redis (the Message Broker is replaced by IPC in WS3 P2), so
        # override the dependency to a service with redis_factory=None.
        # OperatorReadService treats that as "no live-session pub/sub"
        # and yields None inside _live_state_client.
        def _stub_read_service() -> OperatorReadService:
            return OperatorReadService(redis_factory=None)

        api_app.dependency_overrides[get_read_service] = _stub_read_service

        @asynccontextmanager
        async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
            logger.warning(
                "POSTGRES_USER unset — using empty-row pool stub "
                "(transitional; WS4 P1 swaps in the SQLite lifespan)"
            )
            yield

        api_app.router.lifespan_context = _noop_lifespan

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
