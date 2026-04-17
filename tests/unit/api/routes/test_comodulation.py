"""
Tests for the GET /comodulation/{session_id} endpoint.

§7C — Rolling Co-Modulation Index readback.
§2 step 7 — Parameterized queries only.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from services.api.routes.comodulation import get_comodulation


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
    "window_end_utc",
    "window_minutes",
    "co_modulation_index",
    "n_paired_observations",
    "coverage_ratio",
    "streamer_rmssd_mean",
    "operator_rmssd_mean",
    "created_at",
]


class TestGetComodulation:
    def test_returns_rows(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            _COLUMNS,
            [
                (
                    "sess-1",
                    "2026-04-01T12:05:00Z",
                    5,
                    0.61,
                    60,
                    0.95,
                    52.1,
                    60.8,
                    "2026-04-01T12:05:01Z",
                ),
                (
                    "sess-1",
                    "2026-04-01T12:10:00Z",
                    5,
                    None,  # §7C — insufficient paired observations
                    12,
                    0.25,
                    None,
                    None,
                    "2026-04-01T12:10:01Z",
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.comodulation.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.comodulation.put_connection"),
        ):
            result = asyncio.run(get_comodulation(session_id="sess-1", limit=100))

        assert len(result) == 2
        assert result[0]["co_modulation_index"] == 0.61
        assert result[1]["co_modulation_index"] is None
        assert result[1]["coverage_ratio"] == 0.25

    def test_parameterized_query(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(_COLUMNS, [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.comodulation.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.comodulation.put_connection"),
        ):
            asyncio.run(get_comodulation(session_id="abc", limit=50))

        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        assert "%(session_id)s" in sql
        assert "ORDER BY c.window_end_utc DESC" in sql
        assert call_args[0][1] == {"session_id": "abc", "limit": 50}

    def test_connection_returned(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(_COLUMNS, [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.comodulation.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.comodulation.put_connection") as mock_put,
        ):
            asyncio.run(get_comodulation(session_id="sess-x", limit=10))

        mock_put.assert_called_once_with(mock_conn)

    def test_runtime_error_returns_503(self) -> None:
        with (
            patch(
                "services.api.routes.comodulation.get_connection",
                side_effect=RuntimeError("pool down"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_comodulation(session_id="sess-1", limit=100))

        exc: Any = exc_info.value
        assert exc.status_code == 503
