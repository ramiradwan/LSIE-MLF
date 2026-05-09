"""SQLite-backed desktop session lifecycle service tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from packages.schemas.operator_console import SessionCreateRequest, SessionEndRequest
from services.api.services.session_lifecycle_service import SessionLifecycleConflictError
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.state.sqlite_session_lifecycle_service import (
    SqliteSessionLifecycleService,
)

CLIENT_ACTION_A = UUID("00000000-0000-4000-8000-0000000000a1")
CLIENT_ACTION_B = UUID("00000000-0000-4000-8000-0000000000b2")


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[LiveSessionControlMessage] = []

    def publish(self, message: LiveSessionControlMessage) -> None:
        self.messages.append(message)


def _fetch_session(db: Path, session_id: UUID) -> sqlite3.Row | None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT session_id, stream_url, experiment_id, started_at, ended_at
            FROM sessions
            WHERE session_id = ?
            """,
            (str(session_id),),
        ).fetchone()
        return cast("sqlite3.Row | None", row)
    finally:
        conn.close()


def test_request_session_start_creates_sqlite_session(tmp_path: Path) -> None:
    now = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    db = tmp_path / "desktop.sqlite"
    publisher = _Publisher()
    service = SqliteSessionLifecycleService(
        db,
        clock=lambda: now,
        control_publisher=publisher,
    )

    accepted = service.request_session_start(
        SessionCreateRequest(
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            client_action_id=CLIENT_ACTION_A,
        )
    )

    assert accepted.accepted is True
    assert accepted.action == "start"
    row = _fetch_session(db, accepted.session_id)
    assert row is not None
    assert row["stream_url"] == "test://stream"
    assert row["experiment_id"] == "greeting_line_v1"
    assert row["started_at"] == "2026-04-01 12:00:00"
    assert row["ended_at"] is None
    assert len(publisher.messages) == 1
    assert publisher.messages[0].action == "start"
    assert publisher.messages[0].session_id == accepted.session_id
    assert publisher.messages[0].stream_url == "test://stream"
    assert publisher.messages[0].experiment_id == "greeting_line_v1"
    assert publisher.messages[0].active_arm is not None
    assert publisher.messages[0].expected_greeting is not None


def test_request_session_start_is_idempotent_for_client_action(tmp_path: Path) -> None:
    current_time = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    db = tmp_path / "desktop.sqlite"
    publisher = _Publisher()
    service = SqliteSessionLifecycleService(
        db,
        clock=lambda: current_time,
        control_publisher=publisher,
    )
    request = SessionCreateRequest(
        stream_url="test://stream",
        experiment_id="greeting_line_v1",
        client_action_id=CLIENT_ACTION_A,
    )

    first = service.request_session_start(request)
    current_time = datetime(2026, 4, 1, 12, 5, tzinfo=UTC)
    second = service.request_session_start(request)

    assert second.session_id == first.session_id
    row = _fetch_session(db, first.session_id)
    assert row is not None
    assert row["started_at"] == "2026-04-01 12:00:00"
    assert row["ended_at"] is None
    assert [message.action for message in publisher.messages] == ["start"]


def test_request_session_end_marks_active_session_ended(tmp_path: Path) -> None:
    started_at = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    ended_at = datetime(2026, 4, 1, 12, 5, tzinfo=UTC)
    current_time = started_at
    db = tmp_path / "desktop.sqlite"
    publisher = _Publisher()
    service = SqliteSessionLifecycleService(
        db,
        clock=lambda: current_time,
        control_publisher=publisher,
    )
    started = service.request_session_start(
        SessionCreateRequest(
            stream_url="test://stream",
            experiment_id="greeting_line_v1",
            client_action_id=CLIENT_ACTION_A,
        )
    )

    current_time = ended_at
    accepted = service.request_session_end(
        started.session_id,
        SessionEndRequest(client_action_id=CLIENT_ACTION_B),
    )

    assert accepted.accepted is True
    assert accepted.action == "end"
    row = _fetch_session(db, started.session_id)
    assert row is not None
    assert row["ended_at"] == "2026-04-01 12:05:00"
    assert [message.action for message in publisher.messages] == ["start", "end"]
    assert publisher.messages[1].session_id == started.session_id


def test_request_session_end_rejects_missing_or_inactive_session(tmp_path: Path) -> None:
    now = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    db = tmp_path / "desktop.sqlite"
    service = SqliteSessionLifecycleService(db, clock=lambda: now)
    missing_session_id = UUID("00000000-0000-4000-8000-000000000001")

    with pytest.raises(SessionLifecycleConflictError):
        service.request_session_end(
            missing_session_id,
            SessionEndRequest(client_action_id=CLIENT_ACTION_A),
        )


def test_request_session_start_rejects_second_active_session(tmp_path: Path) -> None:
    current_time = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    db = tmp_path / "desktop.sqlite"
    service = SqliteSessionLifecycleService(db, clock=lambda: current_time)
    service.request_session_start(
        SessionCreateRequest(
            stream_url="test://older",
            experiment_id="greeting_line_v1",
            client_action_id=CLIENT_ACTION_A,
        )
    )
    current_time = current_time + timedelta(minutes=1)

    with pytest.raises(SessionLifecycleConflictError, match="already active"):
        service.request_session_start(
            SessionCreateRequest(
                stream_url="test://newer",
                experiment_id="greeting_line_v1",
                client_action_id=CLIENT_ACTION_B,
            )
        )


def test_request_session_end_rejects_non_latest_active_session(tmp_path: Path) -> None:
    current_time = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    db = tmp_path / "desktop.sqlite"
    service = SqliteSessionLifecycleService(db, clock=lambda: current_time)
    older = service.request_session_start(
        SessionCreateRequest(
            stream_url="test://older",
            experiment_id="greeting_line_v1",
            client_action_id=CLIENT_ACTION_A,
        )
    )
    current_time = current_time + timedelta(minutes=1)
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute(
            """
            INSERT INTO sessions (session_id, stream_url, experiment_id, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "00000000-0000-4000-8000-0000000000c3",
                "test://newer",
                "greeting_line_v1",
                "2026-04-01 12:01:00",
            ),
        )
    finally:
        conn.close()

    with pytest.raises(SessionLifecycleConflictError, match="is not the active session"):
        service.request_session_end(
            older.session_id,
            SessionEndRequest(client_action_id=CLIENT_ACTION_B),
        )

    row = _fetch_session(db, older.session_id)
    assert row is not None
    assert row["ended_at"] is None
