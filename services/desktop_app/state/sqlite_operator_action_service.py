"""SQLite-backed operator stimulus action service for the desktop API shell."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from packages.schemas.evaluation import StimulusDefinition
from packages.schemas.operator_console import StimulusAccepted, StimulusRequest
from services.api.services.operator_action_service import (
    SessionAlreadyEndedError,
    SessionNotFoundError,
)
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas, bootstrap_schema

_DESKTOP_STIMULUS_UNAVAILABLE = (
    "Test message accepted. The response measurement is starting now. "
    "Wait for the next result before sending another test message."
)


class LiveSessionControlPublisher(Protocol):
    def publish(self, message: LiveSessionControlMessage) -> None: ...


class SqliteOperatorActionService:
    """Accept desktop stimulus requests without Postgres or Redis coupling."""

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

    def submit_stimulus(self, session_id: UUID, request: StimulusRequest) -> StimulusAccepted:
        now = self._clock()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT s.session_id, s.ended_at, s.stream_url, s.experiment_id,
                       s.active_arm, e.stimulus_definition
                FROM sessions s
                LEFT JOIN experiments e
                    ON e.experiment_id = s.experiment_id
                   AND e.arm = s.active_arm
                   AND e.enabled = 1
                WHERE s.session_id = ?
                ORDER BY e.alpha_param DESC, e.arm ASC
                LIMIT 1
                """,
                (str(session_id),),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(str(session_id))
        if row["ended_at"] is not None:
            raise SessionAlreadyEndedError(str(session_id))
        self._publish_control(
            LiveSessionControlMessage(
                action="stimulus",
                session_id=session_id,
                stream_url=str(row["stream_url"]) if row["stream_url"] is not None else None,
                experiment_id=(
                    str(row["experiment_id"]) if row["experiment_id"] is not None else None
                ),
                active_arm=(str(row["active_arm"]) if row["active_arm"] is not None else None),
                stimulus_definition=(
                    StimulusDefinition.model_validate_json(str(row["stimulus_definition"]))
                    if row["stimulus_definition"] is not None
                    else None
                ),
                stimulus_time_s=now.timestamp(),
                timestamp_utc=now,
            )
        )
        return StimulusAccepted(
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=now,
            stimulus_time_utc=now,
            message=_DESKTOP_STIMULUS_UNAVAILABLE,
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


__all__ = ["SqliteOperatorActionService"]
