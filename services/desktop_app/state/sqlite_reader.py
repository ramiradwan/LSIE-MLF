"""Read-only SQLite adapter (WS4 P1).

Mirrors the API surface that ``services/api/repos/operator_queries.py``
exposes today, so the existing Qt viewmodels in
``services/operator_console/viewmodels/`` continue to work unchanged
when WS4 P1b swaps the FastAPI dependency from the Postgres-shaped
empty-pool stub (``ui_api_shell._EmptyPool``) to a SqliteReader-backed
adapter.

Phase 1 minimum: connection-vending helpers + a small set of read
functions covering the seed-experiments path so the Experiments page
of the operator console can render real data immediately. The full
17-function operator_queries.py port lands in P1b.

Each connection is opened with ``PRAGMA query_only=1`` to enforce the
WS4 P1 single-writer invariant; an accidental ``INSERT`` from a route
handler raises ``OperationalError: attempt to write a readonly database``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from services.desktop_app.state.sqlite_schema import apply_reader_pragmas

logger = logging.getLogger(__name__)


class SqliteReader:
    """Vend read-only connections backed by the desktop SQLite store."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a query-only sqlite3.Connection scoped to this call."""
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            apply_reader_pragmas(conn)
            yield conn
        finally:
            conn.close()

    def fetch_experiment_arms(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return all arms for a given experiment_id, ordered by arm name.

        Mirrors ``services.api.repos.operator_queries.fetch_experiment_arms``
        in shape so an upstream wrap can swap implementations without
        touching the route layer.
        """
        sql = (
            "SELECT id, experiment_id, label, arm, greeting_text, "
            "alpha_param, beta_param, enabled, end_dated_at, updated_at "
            "FROM experiments "
            "WHERE experiment_id = ? "
            "ORDER BY arm"
        )
        with self.connection() as conn:
            cursor = conn.execute(sql, (experiment_id,))
            return [dict(row) for row in cursor.fetchall()]

    def fetch_active_arm_for_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Return the arm with the highest alpha_param posterior, or None."""
        sql = (
            "SELECT id, experiment_id, label, arm, greeting_text, "
            "alpha_param, beta_param, enabled, end_dated_at, updated_at "
            "FROM experiments "
            "WHERE experiment_id = ? AND enabled = 1 "
            "ORDER BY alpha_param DESC "
            "LIMIT 1"
        )
        with self.connection() as conn:
            cursor = conn.execute(sql, (experiment_id,))
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def fetch_recent_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recently started sessions, newest first."""
        sql = (
            "SELECT session_id, stream_url, started_at, ended_at "
            "FROM sessions "
            "ORDER BY started_at DESC "
            "LIMIT ?"
        )
        with self.connection() as conn:
            cursor = conn.execute(sql, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def fetch_active_session(self) -> dict[str, Any] | None:
        """Return the latest session whose ended_at is NULL, if any."""
        sql = (
            "SELECT session_id, stream_url, started_at, ended_at "
            "FROM sessions "
            "WHERE ended_at IS NULL "
            "ORDER BY started_at DESC "
            "LIMIT 1"
        )
        with self.connection() as conn:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            return dict(row) if row is not None else None
