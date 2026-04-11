"""
Tests for services/api/routes/metrics.py — Phase 6.2 validation.

Verifies metrics endpoints against:
  §2 step 7 — Parameterized queries
  §11 — Variable Extraction Matrix (AU12, acoustic)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import tests.unit.api.routes.conftest  # noqa: F401 — trigger sys.modules mocks
from services.api.routes.metrics import (
    _rows_to_dicts,
    _serialize,
    get_acoustic_timeseries,
    get_au12_timeseries,
    get_metrics,
)


def _make_mock_cursor(
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> MagicMock:
    """Create a mock cursor with description and fetchall."""
    cursor = MagicMock()
    cursor.description = [(col,) for col in columns]
    cursor.fetchall.return_value = rows
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
        assert _serialize(None) is None


class TestRowsToDicts:
    """Cursor result conversion."""

    def test_converts_rows(self) -> None:
        """Rows converted to list of dicts with column names."""
        cursor = _make_mock_cursor(
            ["id", "name"],
            [(1, "test"), (2, "test2")],
        )
        result = _rows_to_dicts(cursor)
        assert len(result) == 2
        assert result[0] == {"id": 1, "name": "test"}

    def test_empty_result(self) -> None:
        """Empty rows return empty list."""
        cursor = _make_mock_cursor(["id"], [])
        assert _rows_to_dicts(cursor) == []

    def test_none_description(self) -> None:
        """None description returns empty list."""
        cursor = MagicMock()
        cursor.description = None
        assert _rows_to_dicts(cursor) == []


class TestGetMetrics:
    """§2 step 7 / §4.E — Metrics query endpoint."""

    def test_returns_metrics_list(self) -> None:
        """§2 step 7 — Returns list of metric dicts."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["id", "session_id", "segment_id", "au12_intensity"],
            [(1, "sess-1", "seg-001", 2.5)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_metrics(session_id=None, limit=100))

        assert len(result) == 1
        assert result[0]["au12_intensity"] == 2.5

    def test_filters_by_session_id(self) -> None:
        """§2 step 7 — Parameterized session_id filter."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            asyncio.run(get_metrics(session_id="test-uuid", limit=50))

        # Verify parameterized query was used
        call_args = mock_cursor.execute.call_args
        assert "%(session_id)s" in call_args[0][0]
        assert call_args[0][1]["session_id"] == "test-uuid"

    def test_connection_returned_on_success(self) -> None:
        """Connection returned to pool after query."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn) as _,
            patch("services.api.routes.metrics.put_connection") as mock_put,
        ):
            asyncio.run(get_metrics(session_id=None, limit=10))

        mock_put.assert_called_once_with(mock_conn)


class TestGetAU12Timeseries:
    """§11 — AU12 intensity time-series."""

    def test_returns_au12_data(self) -> None:
        """§11 — AU12 time-series ordered by timestamp."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["segment_id", "timestamp_utc", "au12_intensity"],
            [("seg-001", "2026-03-13T12:00:00Z", 1.5)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_au12_timeseries("test-uuid"))

        assert len(result) == 1
        assert result[0]["au12_intensity"] == 1.5

    def test_parameterized_query(self) -> None:
        """§2 step 7 — Uses parameterized session_id."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["segment_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            asyncio.run(get_au12_timeseries("my-session"))

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1]["session_id"] == "my-session"


class TestGetAcousticTimeseries:
    """§11 — Acoustic metrics time-series."""

    def test_returns_acoustic_data(self) -> None:
        """§11 — Pitch, jitter, shimmer time-series."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["segment_id", "timestamp_utc", "pitch_f0", "jitter", "shimmer"],
            [("seg-001", "2026-03-13T12:00:00Z", 180.0, 0.02, 0.05)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_acoustic_timeseries("test-uuid"))

        assert len(result) == 1
        assert result[0]["pitch_f0"] == 180.0
        assert result[0]["jitter"] == 0.02
