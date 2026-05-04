"""SQLite-backed session lifecycle service for the desktop API shell."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
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
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas, bootstrap_schema


class LiveSessionControlPublisher(Protocol):
    def publish(self, message: LiveSessionControlMessage) -> None: ...


class SqliteSessionLifecycleService(SessionLifecycleService):
    """Accept session lifecycle writes directly into desktop SQLite."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        control_publisher: LiveSessionControlPublisher | None = None,
    ) -> None:
        self._db_path = db_path
        self._clock = clock
        self._control_publisher = control_publisher

    def request_session_start(self, request: SessionCreateRequest) -> SessionLifecycleAccepted:
        session_id = _stable_session_id_for_action(request.client_action_id)
        now = self._clock()
        with self._connection() as conn:
            existing = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE ended_at IS NULL AND session_id != ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (str(session_id),),
            ).fetchone()
            if existing is not None:
                raise SessionLifecycleConflictError(
                    f"session {existing['session_id']} is already active; start not accepted"
                )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    session_id, stream_url, experiment_id, started_at, ended_at
                )
                VALUES (?, ?, ?, ?, NULL)
                """,
                (str(session_id), request.stream_url, request.experiment_id, _iso_utc(now)),
            )
            active_arm = _fetch_active_arm(conn, request.experiment_id)
        if cursor.rowcount > 0:
            self._publish_control(
                LiveSessionControlMessage(
                    action="start",
                    session_id=session_id,
                    stream_url=request.stream_url,
                    experiment_id=request.experiment_id,
                    active_arm=str(active_arm["arm"]) if active_arm is not None else None,
                    expected_greeting=(
                        str(active_arm["greeting_text"])
                        if active_arm is not None and active_arm["greeting_text"] is not None
                        else None
                    ),
                    timestamp_utc=now,
                )
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
            target_row = conn.execute(
                """
                SELECT session_id, ended_at
                FROM sessions
                WHERE session_id = ?
                """,
                (str(session_id),),
            ).fetchone()
            active_row = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE ended_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
            active_session_id = str(active_row["session_id"]) if active_row is not None else None
            if target_row is None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not active; end not accepted"
                )
            if target_row["ended_at"] is not None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} has already ended; end not accepted"
                )
            if active_row is None:
                raise SessionLifecycleConflictError(
                    f"session {session_id} is not active; end not accepted"
                )
            if active_session_id != str(session_id):
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
        self._publish_control(
            LiveSessionControlMessage(
                action="end",
                session_id=session_id,
                timestamp_utc=now,
            )
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

    def _publish_control(self, message: LiveSessionControlMessage) -> None:
        if self._control_publisher is not None:
            self._control_publisher.publish(message)


def _fetch_active_arm(conn: sqlite3.Connection, experiment_id: str | None) -> sqlite3.Row | None:
    if experiment_id is None:
        return None
    row = conn.execute(
        """
        SELECT arm, greeting_text
        FROM experiments
        WHERE experiment_id = ? AND enabled = 1
        ORDER BY alpha_param DESC, arm ASC
        LIMIT 1
        """,
        (experiment_id,),
    ).fetchone()
    return cast("sqlite3.Row | None", row)


def _iso_utc(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
