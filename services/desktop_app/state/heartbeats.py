"""Per-process 1 Hz heartbeats to the desktop SQLite store (WS4 P2).

Each of the six v4.0 desktop processes instantiates a
:class:`HeartbeatRecorder` near the top of its ``run`` callable,
:meth:`~HeartbeatRecorder.start` it before the main loop, and
:meth:`~HeartbeatRecorder.stop` it inside the ``finally`` clause. The
recorder owns a daemon thread that writes an ``INSERT OR REPLACE`` into
``process_heartbeat`` once a second so the operator console can render
freshness and the next ui_api_shell startup recovery sweep can spot a
process that crashed mid-flight.

Cross-process write safety: WAL mode + ``busy_timeout=5000`` (set in
:func:`apply_writer_pragmas`) lets independent processes serialise their
sub-millisecond ``INSERT OR REPLACE`` without the main
:class:`SqliteWriter` flush thread hitting ``database is locked``. Each
recorder owns its own ``sqlite3.Connection`` and never touches another
process's connection.

Why direct writes (rather than IPC + analytics_state_worker fan-in):
heartbeats need to keep flowing even if ``analytics_state_worker``
itself wedges. A per-process direct write decouples each child's
liveness signal from any other child's health.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from services.desktop_app.state.sqlite_schema import apply_writer_pragmas

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_S: float = 1.0

_UPSERT_HEARTBEAT_SQL: str = (
    "INSERT INTO process_heartbeat "
    "(process_name, pid, started_at_utc, last_heartbeat_utc) "
    "VALUES (?, ?, ?, ?) "
    "ON CONFLICT(process_name) DO UPDATE SET "
    "  pid = excluded.pid, "
    "  last_heartbeat_utc = excluded.last_heartbeat_utc"
)


def _utc_now_iso() -> str:
    """Return a SQLite-compatible UTC ISO-8601 timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


class HeartbeatRecorder:
    """Daemon-thread writer that beats once per ``interval_s``.

    Construct with the resolved SQLite path, the canonical process
    name (one of the six in
    :data:`services.desktop_app.process_graph.PROCESS_MODULES`), and
    optionally a custom interval. Call :meth:`start` to spawn the
    thread; call :meth:`stop` on shutdown to drain a final beat and
    release the connection. The class is idempotent: ``stop`` after
    ``stop`` is a no-op so callers can put it inside ``finally``
    without guarding.
    """

    def __init__(
        self,
        db_path: Path,
        process_name: str,
        *,
        interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"heartbeat interval must be positive, got {interval_s!r}")
        self._db_path = db_path
        self._process_name = process_name
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None
        self._started_at_utc: str | None = None
        self._pid: int | None = None

    @property
    def process_name(self) -> str:
        return self._process_name

    def start(self) -> None:
        """Open the SQLite connection, write the first beat, spawn the thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        apply_writer_pragmas(self._conn)
        self._pid = os.getpid()
        self._started_at_utc = _utc_now_iso()
        # Write the inaugural beat synchronously so the operator
        # console sees the row come up immediately rather than after
        # the first interval tick.
        self._write_beat()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"heartbeat-{self._process_name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread, write a final beat, close the connection."""
        if self._stop_event.is_set() and self._thread is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Final synchronous beat so the recorded ``last_heartbeat_utc``
        # reflects clean shutdown, not the previous tick. Defensive:
        # the loop may have raced past the stop set and already
        # written a final row.
        try:
            self._write_beat()
        except sqlite3.Error:
            logger.debug("final heartbeat write failed", exc_info=True)
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.debug("heartbeat connection close failed", exc_info=True)
            self._conn = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self._interval_s):
                return
            try:
                self._write_beat()
            except sqlite3.Error:
                logger.warning("heartbeat write failed for %s", self._process_name, exc_info=True)

    def _write_beat(self) -> None:
        if self._conn is None or self._pid is None or self._started_at_utc is None:
            return
        self._conn.execute(
            _UPSERT_HEARTBEAT_SQL,
            (self._process_name, self._pid, self._started_at_utc, _utc_now_iso()),
        )


def fetch_all_heartbeats(db_path: Path) -> list[dict[str, object]]:
    """Snapshot every row of ``process_heartbeat`` (read-only helper).

    Used by the recovery sweep to discover which processes were active
    at last shutdown, and by tests to assert beat semantics. The
    operator console reads the same data through ``SqliteReader`` plus
    the §12 freshness rollup.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT process_name, pid, started_at_utc, last_heartbeat_utc "
            "FROM process_heartbeat ORDER BY process_name"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
