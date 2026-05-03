"""SQLite-backed session lifecycle service for the desktop API shell."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from packages.schemas.operator_console import (
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
)
from services.api.services.session_lifecycle_service import (
    SessionLifecycleConflictError,
    SessionLifecycleService,
    _stable_session_id_for_action,
)
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas, bootstrap_schema


class SqliteSessionLifecycleService(SessionLifecycleService):
    """Accept session lifecycle writes directly into desktop SQLite."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._db_path = db_path
        self._clock = clock

    def request_session_start(self, request: SessionCreateRequest) -> SessionLifecycleAccepted:
        session_id = _stable_session_id_for_action(request.client_action_id)
        now = self._clock()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    session_id, stream_url, experiment_id, started_at, ended_at
                )
                VALUES (?, ?, ?, ?, NULL)
                """,
                (str(session_id), request.stream_url, request.experiment_id, _iso_utc(now)),
            )
        return SessionLifecycleAccepted(
            action="start",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=now,
        )

    def request_session_end(
        self,
        session_id: UUID,
        request: SessionEndRequest,
    ) -> SessionLifecycleAccepted:
        now = self._clock()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT session_id, ended_at
                FROM sessions
                WHERE session_id = ?
                """,
                (str(session_id),),
            ).fetchone()
            if row is None or row["ended_at"] is not None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not active; end not accepted"
                )
            active = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE ended_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
            if active is None or str(active["session_id"]) != str(session_id):
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not the active session; end not accepted"
                )
            conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?
                WHERE session_id = ? AND ended_at IS NULL
                """,
                (_iso_utc(now), str(session_id)),
            )
        return SessionLifecycleAccepted(
            action="end",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=now,
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            bootstrap_schema(conn)
            apply_writer_pragmas(conn)
            yield conn
        finally:
            conn.close()


def _iso_utc(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
