"""
Tests for services/api/routes/sessions.py — Phase 6.3 validation.

Verifies session endpoints against:
  §2 step 7 — Parameterized queries
  §11 — Summary metric aggregation
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tests.unit.api.routes.conftest as _conftest  # noqa: F401 — trigger sys.modules mocks
from services.api.routes.sessions import (
    _row_to_dict,
    _rows_to_dicts,
    _serialize,
    get_session,
    list_sessions,
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
        from datetime import UTC, datetime

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
                # Session query
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
                # Summary query
                session_cursor.description = [
                    ("total_segments",),
                    ("avg_au12",),
                    ("avg_pitch_f0",),
                    ("avg_jitter",),
                    ("avg_shimmer",),
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
