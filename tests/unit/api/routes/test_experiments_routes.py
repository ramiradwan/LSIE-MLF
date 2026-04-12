"""Tests for services/api/routes/experiments.py.

Verifies experiment state endpoints against:
  §4.E.1 — Thompson Sampling arm state inspection
  §2.7 — Parameterized queries
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from services.api.routes.experiments import (
    _rows_to_dicts,
    _serialize,
    get_experiment,
    list_experiments,
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
        dt = datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC)
        assert _serialize(dt) == "2025-04-01T12:00:00+00:00"

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
            ["experiment_id", "arm", "alpha_param"],
            [("greeting_line_v1", "warm_welcome", 3.5)],
        )

        result = _rows_to_dicts(cursor)

        assert result == [
            {
                "experiment_id": "greeting_line_v1",
                "arm": "warm_welcome",
                "alpha_param": 3.5,
            }
        ]

    def test_empty_result(self) -> None:
        """Empty rows return empty list."""
        cursor = _make_mock_cursor(["experiment_id"], [])
        assert _rows_to_dicts(cursor) == []

    def test_none_description(self) -> None:
        """None description returns empty list."""
        cursor = MagicMock()
        cursor.description = None
        assert _rows_to_dicts(cursor) == []


class TestListExperiments:
    """§4.E.1 — Experiment list endpoint."""

    def test_returns_experiment_ids(self) -> None:
        """Returns list of experiment IDs."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id"],
            [("greeting_line_v1",), ("greeting_line_v2",)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
        ):
            result = asyncio.run(list_experiments())

        assert len(result) == 2
        assert result[0]["experiment_id"] == "greeting_line_v1"
        assert result[1]["experiment_id"] == "greeting_line_v2"

    def test_returns_empty_list(self) -> None:
        """Returns empty list when no experiments exist."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["experiment_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
        ):
            result = asyncio.run(list_experiments())

        assert result == []

    def test_503_when_pool_not_initialized(self) -> None:
        """Returns 503 when the DB pool is unavailable."""
        with (
            patch(
                "services.api.routes.experiments.get_connection",
                side_effect=RuntimeError("Connection pool not initialized"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(list_experiments())

        exc: Any = exc_info.value
        assert exc.status_code == 503
        assert "Connection pool not initialized" in exc.detail

    def test_500_on_unexpected_error(self) -> None:
        """Returns 500 on unexpected query failure."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["experiment_id"], [])
        mock_cursor.execute.side_effect = Exception("boom")
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(list_experiments())

        exc: Any = exc_info.value
        assert exc.status_code == 500
        assert exc.detail == "Internal server error"

    def test_connection_returned(self) -> None:
        """Connection returned to pool after query."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["experiment_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection") as mock_put,
        ):
            asyncio.run(list_experiments())

        mock_put.assert_called_once_with(mock_conn)


class TestGetExperiment:
    """§4.E.1 — Experiment detail endpoint."""

    def test_returns_arm_state(self) -> None:
        """Returns experiment ID with arm-level alpha/beta posteriors."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [
                (
                    "greeting_line_v1",
                    "direct_ask",
                    1.8,
                    4.2,
                    datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC),
                ),
                (
                    "greeting_line_v1",
                    "warm_welcome",
                    3.5,
                    2.1,
                    datetime(2025, 4, 1, 12, 0, 0, tzinfo=UTC),
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
        ):
            result = asyncio.run(get_experiment("greeting_line_v1"))

        assert result["experiment_id"] == "greeting_line_v1"
        assert len(result["arms"]) == 2
        assert result["arms"][0]["arm"] == "direct_ask"
        assert result["arms"][0]["alpha_param"] == 1.8
        assert result["arms"][0]["beta_param"] == 4.2
        assert result["arms"][0]["updated_at"] == "2025-04-01T12:00:00+00:00"
        assert result["arms"][1]["arm"] == "warm_welcome"
        assert result["arms"][1]["alpha_param"] == 3.5
        assert result["arms"][1]["beta_param"] == 2.1

    def test_404_unknown_experiment(self) -> None:
        """Returns 404 when no rows exist for the given experiment ID."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_experiment("nonexistent"))

        exc: Any = exc_info.value
        assert exc.status_code == 404
        assert "No experiment found" in exc.detail

    def test_parameterized_query(self) -> None:
        """§2 step 7 — Uses parameterized experiment_id query."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [("greeting_line_v1", "arm_a", 1.0, 1.0, None)],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
        ):
            asyncio.run(get_experiment("greeting_line_v1"))

        call_args = mock_cursor.execute.call_args
        assert "%(experiment_id)s" in call_args[0][0]
        assert call_args[0][1]["experiment_id"] == "greeting_line_v1"

    def test_503_when_pool_not_initialized(self) -> None:
        """Returns 503 when the DB pool is unavailable."""
        with (
            patch(
                "services.api.routes.experiments.get_connection",
                side_effect=RuntimeError("Connection pool not initialized"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_experiment("greeting_line_v1"))

        exc: Any = exc_info.value
        assert exc.status_code == 503
        assert "Connection pool not initialized" in exc.detail

    def test_500_on_unexpected_error(self) -> None:
        """Returns 500 on unexpected query failure."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [],
        )
        mock_cursor.execute.side_effect = Exception("boom")
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_experiment("greeting_line_v1"))

        exc: Any = exc_info.value
        assert exc.status_code == 500
        assert exc.detail == "Internal server error"

    def test_connection_returned_on_error(self) -> None:
        """Connection returned to pool even when endpoint raises 404."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["experiment_id", "arm", "alpha_param", "beta_param", "updated_at"],
            [],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.experiments.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.experiments.put_connection") as mock_put,
            pytest.raises(HTTPException),
        ):
            asyncio.run(get_experiment("nonexistent"))

        mock_put.assert_called_once_with(mock_conn)
