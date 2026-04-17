"""
Tests for the GET /physiology/{session_id} endpoint in
services/api/routes/physiology.py.

§4.E.2 — Per-segment physiological snapshot readback.
§2 step 7 — Parameterized queries only.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from services.api.routes.physiology import get_physiology


def _make_mock_cursor(
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> MagicMock:
    cursor = MagicMock()
    cursor.description = [(col,) for col in columns]
    cursor.fetchall.return_value = rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


_COLUMNS = [
    "session_id",
    "segment_id",
    "subject_role",
    "rmssd_ms",
    "heart_rate_bpm",
    "freshness_s",
    "is_stale",
    "provider",
    "source_timestamp_utc",
    "created_at",
]


class TestGetPhysiologyLatest:
    """Default (series=False) returns the latest snapshot per subject_role."""

    def test_returns_latest_per_subject(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            _COLUMNS,
            [
                (
                    "sess-1",
                    "seg-12",
                    "streamer",
                    52.1,
                    68,
                    42.0,
                    False,
                    "oura",
                    "2026-04-01T12:00:00Z",
                    "2026-04-01T12:00:02Z",
                ),
                (
                    "sess-1",
                    "seg-12",
                    "operator",
                    61.5,
                    72,
                    38.0,
                    False,
                    "oura",
                    "2026-04-01T12:00:00Z",
                    "2026-04-01T12:00:02Z",
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.physiology.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.physiology.put_connection"),
        ):
            result = asyncio.run(get_physiology(session_id="sess-1", series=False, limit=500))

        assert len(result) == 2
        roles = {r["subject_role"] for r in result}
        assert roles == {"streamer", "operator"}
        assert result[0]["session_id"] == "sess-1"
        assert result[0]["provider"] == "oura"

    def test_parameterized_query_uses_session_id(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(_COLUMNS, [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.physiology.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.physiology.put_connection"),
        ):
            asyncio.run(get_physiology(session_id="abc", series=False, limit=500))

        call_args = mock_cursor.execute.call_args
        assert "%(session_id)s" in call_args[0][0]
        assert "DISTINCT ON (p.subject_role)" in call_args[0][0]
        assert call_args[0][1]["session_id"] == "abc"

    def test_connection_returned(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(_COLUMNS, [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.physiology.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.physiology.put_connection") as mock_put,
        ):
            asyncio.run(get_physiology(session_id="sess-x", series=False, limit=500))

        mock_put.assert_called_once_with(mock_conn)


class TestGetPhysiologySeries:
    """series=True returns the full time-series for the session."""

    def test_series_orders_by_created_at_asc(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(_COLUMNS, [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.physiology.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.physiology.put_connection"),
        ):
            asyncio.run(get_physiology(session_id="sess-1", series=True, limit=200))

        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        assert "ORDER BY p.created_at ASC" in sql
        assert call_args[0][1] == {"session_id": "sess-1", "limit": 200}

    def test_empty_session_returns_empty_list(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(_COLUMNS, [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.physiology.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.physiology.put_connection"),
        ):
            result = asyncio.run(get_physiology(session_id="none", series=True, limit=100))

        assert result == []


class TestGetPhysiologyFailures:
    def test_runtime_error_returns_503(self) -> None:
        with (
            patch(
                "services.api.routes.physiology.get_connection",
                side_effect=RuntimeError("pool down"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_physiology(session_id="sess-1", series=False, limit=500))

        exc: Any = exc_info.value
        assert exc.status_code == 503
