"""SQLite-backed desktop operator action service tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from packages.schemas.operator_console import StimulusRequest
from services.api.services.operator_action_service import (
    SessionAlreadyEndedError,
    SessionNotFoundError,
)
from services.desktop_app.state.sqlite_operator_action_service import (
    SqliteOperatorActionService,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
CLIENT_ACTION_ID = UUID("00000000-0000-4000-8000-0000000000a1")
_NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


def _seed_session(db: Path, *, ended: bool = False) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        bootstrap_schema(conn)
        conn.execute(
            """
            INSERT INTO sessions (session_id, stream_url, experiment_id, started_at, ended_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(SESSION_ID),
                "test://stream",
                "greeting_line_v1",
                "2026-04-01 12:00:00",
                "2026-04-01 12:05:00" if ended else None,
            ),
        )
    finally:
        conn.close()


def _request() -> StimulusRequest:
    return StimulusRequest(client_action_id=CLIENT_ACTION_ID, operator_note="hello")


def test_submit_stimulus_accepts_active_session_without_pool_or_redis(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_session(db)
    service = SqliteOperatorActionService(db, clock=lambda: _NOW)

    accepted = service.submit_stimulus(SESSION_ID, _request())

    assert accepted.accepted is True
    assert accepted.session_id == SESSION_ID
    assert accepted.client_action_id == CLIENT_ACTION_ID
    assert accepted.received_at_utc == _NOW
    assert accepted.message is not None
    assert "release-gated" in accepted.message
    assert "reward pipeline" in accepted.message


def test_submit_stimulus_rejects_missing_session(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    service = SqliteOperatorActionService(db, clock=lambda: _NOW)

    with pytest.raises(SessionNotFoundError):
        service.submit_stimulus(SESSION_ID, _request())


def test_submit_stimulus_rejects_ended_session(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_session(db, ended=True)
    service = SqliteOperatorActionService(db, clock=lambda: _NOW)

    with pytest.raises(SessionAlreadyEndedError):
        service.submit_stimulus(SESSION_ID, _request())
