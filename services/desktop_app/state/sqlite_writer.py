"""Single-writer SQLite batch flush.

Local analogue of the v3.4 ``services.worker.pipeline.analytics
.MetricsStore``. Only one instance lives per desktop graph (the
``analytics_state_worker`` process owns it); ``ui_api_shell`` and
every other reader uses :class:`SqliteReader` with
``PRAGMA query_only=1`` to enforce the single-writer invariant.

API surface:

  - ``enqueue(table, payload)`` queues a row for batched insertion.
  - A 250 ms flush thread drains the queue and wraps every batch in
    a single transaction, mirroring ``MetricsStore``'s connection
    pool semantics on PostgreSQL.
  - ``flush()`` forces an immediate flush (used by tests and shutdown).
  - ``close()`` stops the flush thread and closes the connection.

Each ``payload`` is a flat ``dict[str, Any]`` whose keys must match
the SQLite columns of ``table`` exactly. Pydantic validation is the
caller's responsibility — the writer only checks that no unknown
columns were supplied.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.desktop_app.state.sqlite_schema import (
    apply_writer_pragmas,
    bootstrap_schema,
)

logger = logging.getLogger(__name__)

DEFAULT_FLUSH_INTERVAL_S: float = 0.25


@dataclass(frozen=True)
class WriteRecord:
    """One pending insert. Fields match SQLite column names exactly."""

    table: str
    payload: Mapping[str, Any]


@dataclass
class _FlushStats:
    """Lightweight counters surfaced via ``SqliteWriter.stats`` for tests."""

    flush_cycles: int = 0
    rows_written: int = 0
    last_flush_monotonic: float = 0.0
    columns_per_table: dict[str, frozenset[str]] = field(default_factory=dict)


class SqliteWriter:
    """The sole writer to the desktop's local SQLite store.

    Construct once at ``analytics_state_worker`` startup with the
    resolved ``db_path``. Call :meth:`start` before any ``enqueue``
    so the flush thread is alive; call :meth:`close` on cooperative
    shutdown to drain the queue and tear the connection down.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        flush_interval_s: float = DEFAULT_FLUSH_INTERVAL_S,
        bootstrap: bool = True,
    ) -> None:
        self._db_path = db_path
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue[WriteRecord] = queue.Queue()
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        # check_same_thread=False because the flush thread reaches in
        # to commit/rollback while ``enqueue`` runs on the caller's
        # thread. Single-writer is enforced by the public API (only the
        # flush thread + flush() itself touch the connection), not by
        # sqlite3's default thread affinity.
        self._conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        if bootstrap:
            bootstrap_schema(self._conn)
        else:
            apply_writer_pragmas(self._conn)
        self._table_columns: dict[str, frozenset[str]] = self._snapshot_columns()
        self.stats = _FlushStats(columns_per_table=dict(self._table_columns))

    def _snapshot_columns(self) -> dict[str, frozenset[str]]:
        """Read each table's columns once; reused to validate payload keys."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        columns: dict[str, frozenset[str]] = {}
        for table in tables:
            info = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            columns[table] = frozenset(row[1] for row in info)
        return columns

    def start(self) -> None:
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="sqlite-writer-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def enqueue(self, table: str, payload: Mapping[str, Any]) -> None:
        """Validate the payload's columns against the schema, then queue it."""
        known = self._table_columns.get(table)
        if known is None:
            raise ValueError(f"sqlite writer: unknown table {table!r}")
        unknown_cols = set(payload).difference(known)
        if unknown_cols:
            raise ValueError(
                f"sqlite writer: unknown columns for {table!r}: {sorted(unknown_cols)}"
            )
        self._queue.put_nowait(WriteRecord(table=table, payload=dict(payload)))

    def flush(self) -> int:
        """Drain whatever is currently queued and run a single transaction.

        Returns the number of rows written. Safe to call from the
        flush thread or from a test.
        """
        batch: list[WriteRecord] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return 0

        rows_written = 0
        try:
            self._conn.execute("BEGIN")
            for record in batch:
                cols = list(record.payload)
                placeholders = ",".join("?" * len(cols))
                column_list = ",".join(cols)
                stmt = f"INSERT INTO {record.table} ({column_list}) VALUES ({placeholders})"
                self._conn.execute(stmt, [record.payload[c] for c in cols])
                rows_written += 1
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        self.stats.flush_cycles += 1
        self.stats.rows_written += rows_written
        self.stats.last_flush_monotonic = time.monotonic()
        return rows_written

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.flush()
            except Exception:  # noqa: BLE001
                logger.exception("sqlite writer flush failed")
            self._stop_event.wait(timeout=self._flush_interval_s)
        # Drain on the way out so a clean shutdown does not lose pending work.
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            logger.exception("sqlite writer final flush failed")

    def close(self) -> None:
        """Stop the flush thread, drain pending records, close the connection."""
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            logger.debug("sqlite writer close failed", exc_info=True)
