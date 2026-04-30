"""
Tests for services/api/routes/sessions.py.

Validates the legacy read handlers plus the lifecycle POST routes that now
publish authoritative start/end intent for the orchestrator.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from packages.schemas.operator_console import (
    SessionCreateRequest,
    SessionEndRequest,
    SessionLifecycleAccepted,
)
from services.api.routes.sessions import (
    _row_to_dict,
    _rows_to_dicts,
    _serialize,
    create_session,
    end_session,
    get_session,
    list_sessions,
)
from services.api.services.session_lifecycle_service import (
    SessionLifecycleConflictError,
    SessionLifecyclePublishError,
)


def _make_mock_cursor(
    columns: list[str],
    rows: list[tuple[Any, ...]],
    *,
    single_row: tuple[Any, ...] | None = None,
) -> MagicMock:
    """Create a mock cursor with description and fetch methods."""
    cursor = MagicMock()
    cursor.description = [(col,) for col in columns]
    cursor.fetchall.return_value = rows
    cursor.fetchone.return_value = single_row
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


class TestSerialize:
    """Helper serialization tests."""

    def test_datetime_serialized(self) -> None:
        """Datetime objects converted to ISO format."""
        dt = datetime(2026, 3, 13, 12, 0, 0, tzinfo=UTC)
        assert _serialize(dt) == "2026-03-13T12:00:00+00:00"

    def test_plain_value_passthrough(self) -> None:
        """Non-datetime values pass through unchanged."""
        assert _serialize(42) == 42
        assert _serialize("hello") == "hello"


class TestRowToDict:
    """Single row conversion."""

    def test_converts_single_row(self) -> None:
        """Single row converted to dict."""
        cursor = MagicMock()
        cursor.description = [("id",), ("name",)]
        cursor.fetchone.return_value = (1, "test")
        result = _row_to_dict(cursor)
        assert result == {"id": 1, "name": "test"}

    def test_none_row(self) -> None:
        """None fetchone returns None."""
        cursor = MagicMock()
        cursor.description = [("id",)]
        cursor.fetchone.return_value = None
        assert _row_to_dict(cursor) is None

    def test_none_description(self) -> None:
        """None description returns None."""
        cursor = MagicMock()
        cursor.description = None
        assert _row_to_dict(cursor) is None


class TestRowsToDicts:
    """Multi-row conversion."""

    def test_converts_rows(self) -> None:
        """Rows converted to list of dicts."""
        cursor = _make_mock_cursor(
            ["id", "name"],
            [(1, "a"), (2, "b")],
        )
        result = _rows_to_dicts(cursor)
        assert len(result) == 2
        assert result[0] == {"id": 1, "name": "a"}

    def test_empty_result(self) -> None:
        """Empty rows return empty list."""
        cursor = _make_mock_cursor(["id"], [])
        assert _rows_to_dicts(cursor) == []


class TestListSessions:
    """§2 step 7 — List sessions endpoint."""

    def test_returns_sessions_list(self) -> None:
        """Returns list of session dicts with metric counts."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["session_id", "stream_url", "started_at", "ended_at", "metric_count"],
            [("sess-1", "rtmp://test", "2026-03-13T12:00:00Z", None, 5)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.sessions.get_connection", return_value=mock_conn),
            patch("services.api.routes.sessions.put_connection"),
        ):
            result = asyncio.run(list_sessions())

        assert len(result) == 1
        assert result[0]["session_id"] == "sess-1"
        assert result[0]["metric_count"] == 5

    def test_connection_returned(self) -> None:
        """Connection returned to pool after query."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["session_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.sessions.get_connection", return_value=mock_conn),
            patch("services.api.routes.sessions.put_connection") as mock_put,
        ):
            asyncio.run(list_sessions())

        mock_put.assert_called_once_with(mock_conn)


class TestGetSession:
    """§11 — Session detail with summary metrics."""

    def test_returns_session_with_summary(self) -> None:
        """§11 — Session includes aggregated metrics summary."""
        mock_conn = MagicMock()

        session_cursor = MagicMock()
        call_count = 0

        def mock_execute(sql: str, params: dict[str, Any] | None = None) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                session_cursor.description = [
                    ("session_id",),
                    ("stream_url",),
                    ("started_at",),
                    ("ended_at",),
                ]
                session_cursor.fetchone.return_value = (
                    "sess-1",
                    "rtmp://test",
                    "2026-03-13T12:00:00Z",
                    None,
                )
            else:
                session_cursor.description = [
                    ("total_segments",),
                    ("avg_au12",),
                    ("avg_f0_mean_measure_hz",),
                    ("avg_jitter_mean_measure",),
                    ("avg_shimmer_mean_measure",),
                    ("first_segment_at",),
                    ("last_segment_at",),
                ]
                session_cursor.fetchone.return_value = (
                    10,
                    2.5,
                    180.0,
                    0.02,
                    0.05,
                    "2026-03-13T12:00:00Z",
                    "2026-03-13T12:05:00Z",
                )

        session_cursor.execute = mock_execute
        session_cursor.__enter__ = MagicMock(return_value=session_cursor)
        session_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = session_cursor

        with (
            patch("services.api.routes.sessions.get_connection", return_value=mock_conn),
            patch("services.api.routes.sessions.put_connection"),
        ):
            result = asyncio.run(get_session("sess-1"))

        assert result["session_id"] == "sess-1"
        assert "summary" in result
        assert result["summary"]["avg_au12"] == 2.5
        assert result["summary"]["total_segments"] == 10

    def test_session_not_found(self) -> None:
        """Returns 404 when session not found."""
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.description = [("session_id",)]
        cursor.fetchone.return_value = None
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = cursor

        with (
            patch("services.api.routes.sessions.get_connection", return_value=mock_conn),
            patch("services.api.routes.sessions.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_session("nonexistent"))

        assert exc_info.value.status_code == 404  # type: ignore[attr-defined]

    def test_parameterized_query(self) -> None:
        """§2 step 7 — Uses parameterized session_id."""
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.description = [("session_id",)]
        cursor.fetchone.return_value = None
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = cursor

        with (
            patch("services.api.routes.sessions.get_connection", return_value=mock_conn),
            patch("services.api.routes.sessions.put_connection"),
            contextlib.suppress(Exception),
        ):
            asyncio.run(get_session("my-session"))

        call_args = cursor.execute.call_args_list[0]
        assert "%(session_id)s" in call_args[0][0]
        assert call_args[0][1]["session_id"] == "my-session"

    def test_connection_returned_on_error(self) -> None:
        """Connection returned to pool even on error."""
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.description = [("session_id",)]
        cursor.fetchone.return_value = None
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = cursor

        with (
            patch("services.api.routes.sessions.get_connection", return_value=mock_conn),
            patch("services.api.routes.sessions.put_connection") as mock_put,
            contextlib.suppress(Exception),
        ):
            asyncio.run(get_session("nonexistent"))

        mock_put.assert_called_once_with(mock_conn)


class TestCreateSessionLifecycleRoute:
    def test_returns_acceptance_from_service(self) -> None:
        service = MagicMock()
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )
        accepted = SessionLifecycleAccepted(
            action="start",
            session_id=uuid.uuid4(),
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        )
        service.request_session_start.return_value = accepted

        result = asyncio.run(create_session(request, service=service))

        assert result == accepted
        service.request_session_start.assert_called_once_with(request)

    def test_publish_error_surfaces_as_503(self) -> None:
        service = MagicMock()
        service.request_session_start.side_effect = SessionLifecyclePublishError("broker down")
        request = SessionCreateRequest(
            stream_url="https://example.com/live",
            experiment_id="exp-1",
            client_action_id=uuid.uuid4(),
        )

        with pytest.raises(Exception) as exc_info:
            asyncio.run(create_session(request, service=service))

        assert exc_info.value.status_code == 503  # type: ignore[attr-defined]


class TestEndSessionLifecycleRoute:
    def test_returns_acceptance_from_service(self) -> None:
        service = MagicMock()
        session_id = uuid.uuid4()
        request = SessionEndRequest(client_action_id=uuid.uuid4())
        accepted = SessionLifecycleAccepted(
            action="end",
            session_id=session_id,
            client_action_id=request.client_action_id,
            accepted=True,
            received_at_utc=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        )
        service.request_session_end.return_value = accepted

        result = asyncio.run(end_session(session_id, request, service=service))

        assert result == accepted
        service.request_session_end.assert_called_once_with(session_id, request)

    def test_conflict_surfaces_as_409(self) -> None:
        service = MagicMock()
        session_id = uuid.uuid4()
        service.request_session_end.side_effect = SessionLifecycleConflictError(
            f"session {session_id} is not active; end not accepted"
        )
        request = SessionEndRequest(client_action_id=uuid.uuid4())

        with pytest.raises(Exception) as exc_info:
            asyncio.run(end_session(session_id, request, service=service))

        assert exc_info.value.status_code == 409  # type: ignore[attr-defined]

    def test_publish_error_surfaces_as_503(self) -> None:
        service = MagicMock()
        session_id = uuid.uuid4()
        service.request_session_end.side_effect = SessionLifecyclePublishError("broker down")
        request = SessionEndRequest(client_action_id=uuid.uuid4())

        with pytest.raises(Exception) as exc_info:
            asyncio.run(end_session(session_id, request, service=service))

        assert exc_info.value.status_code == 503  # type: ignore[attr-defined]
