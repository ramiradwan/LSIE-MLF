"""SQLite-backed desktop operator action service tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.operator_console import StimulusRequest
from services.api.services.operator_action_service import (
    SessionAlreadyEndedError,
    SessionNotFoundError,
)
from services.desktop_app.ipc.control_messages import LiveSessionControlMessage
from services.desktop_app.state.sqlite_operator_action_service import (
    SqliteOperatorActionService,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
CLIENT_ACTION_ID = UUID("00000000-0000-4000-8000-0000000000a1")
_NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[LiveSessionControlMessage] = []

    def publish(self, message: LiveSessionControlMessage) -> None:
        self.messages.append(message)


def _stimulus_definition() -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text="hello creator",
        ),
        expected_stimulus_rule=(
            "Deliver the spoken greeting to the live streamer exactly as written."
        ),
        expected_response_rule=(
            "The live streamer acknowledges the greeting or responds to it on stream."
        ),
    )


def _seed_session(db: Path, *, ended: bool = False) -> None:
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        bootstrap_schema(conn)
        conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, label, arm, stimulus_definition, alpha_param, beta_param, enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id, arm) DO UPDATE SET
                stimulus_definition = excluded.stimulus_definition,
                enabled = excluded.enabled
            """,
            (
                "greeting_line_v1",
                "Greeting line",
                "warm_welcome",
                _stimulus_definition().model_dump_json(),
                1.0,
                1.0,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, stream_url, experiment_id, active_arm, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(SESSION_ID),
                "test://stream",
                "greeting_line_v1",
                "warm_welcome",
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
    publisher = _Publisher()
    service = SqliteOperatorActionService(
        db,
        clock=lambda: _NOW,
        control_publisher=publisher,
    )

    accepted = service.submit_stimulus(SESSION_ID, _request())

    assert accepted.accepted is True
    assert accepted.session_id == SESSION_ID
    assert accepted.client_action_id == CLIENT_ACTION_ID
    assert accepted.received_at_utc == _NOW
    assert accepted.stimulus_time_utc == _NOW
    assert accepted.message is not None
    assert "response measurement is starting" in accepted.message.lower()
    assert "before sending another test message" in accepted.message.lower()
    assert len(publisher.messages) == 1
    assert publisher.messages[0].action == "stimulus"
    assert publisher.messages[0].session_id == SESSION_ID
    assert publisher.messages[0].stimulus_time_s == _NOW.timestamp()
    assert publisher.messages[0].stream_url == "test://stream"
    assert publisher.messages[0].experiment_id == "greeting_line_v1"
    assert publisher.messages[0].active_arm == "warm_welcome"
    assert publisher.messages[0].stimulus_definition is not None
    assert publisher.messages[0].stimulus_definition == _stimulus_definition()


def test_submit_stimulus_rejects_missing_session_without_publishing(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    publisher = _Publisher()
    service = SqliteOperatorActionService(
        db,
        clock=lambda: _NOW,
        control_publisher=publisher,
    )

    with pytest.raises(SessionNotFoundError):
        service.submit_stimulus(SESSION_ID, _request())

    assert publisher.messages == []


def test_submit_stimulus_rejects_ended_session_without_publishing(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    _seed_session(db, ended=True)
    publisher = _Publisher()
    service = SqliteOperatorActionService(
        db,
        clock=lambda: _NOW,
        control_publisher=publisher,
    )

    with pytest.raises(SessionAlreadyEndedError):
        service.submit_stimulus(SESSION_ID, _request())

    assert publisher.messages == []
