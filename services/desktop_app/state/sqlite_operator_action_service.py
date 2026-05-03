"""SQLite-backed operator stimulus action service for the desktop API shell."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from packages.schemas.operator_console import StimulusAccepted, StimulusRequest
from services.api.services.operator_action_service import (
    SessionAlreadyEndedError,
    SessionNotFoundError,
)
from services.desktop_app.state.sqlite_schema import apply_writer_pragmas, bootstrap_schema

_DESKTOP_STIMULUS_UNAVAILABLE = (
    "Desktop live analytics producer is release-gated; stimulus was not dispatched "
    "to the reward pipeline."
)


class SqliteOperatorActionService:
    """Accept desktop stimulus requests without Postgres or Redis coupling."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._db_path = db_path
        self._clock = clock

    def submit_stimulus(self, session_id: UUID, request: StimulusRequest) -> StimulusAccepted:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT session_id, ended_at
                FROM sessions
                WHERE session_id = ?
                """,
                (str(session_id),),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(str(session_id))
        if row["ended_at"] is not None:
            raise SessionAlreadyEndedError(str(session_id))
        return StimulusAccepted(
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=self._clock(),
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


__all__ = ["SqliteOperatorActionService"]
