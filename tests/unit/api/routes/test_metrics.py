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

from services.api.routes.metrics import (
    _rows_to_dicts,
    _serialize,
    get_acoustic_timeseries,
    get_au12_timeseries,
    get_metrics,
)

_ACOUSTIC_COLUMNS: list[str] = [
    "f0_valid_measure",
    "f0_valid_baseline",
    "perturbation_valid_measure",
    "perturbation_valid_baseline",
    "voiced_coverage_measure_s",
    "voiced_coverage_baseline_s",
    "f0_mean_measure_hz",
    "f0_mean_baseline_hz",
    "f0_delta_semitones",
    "jitter_mean_measure",
    "jitter_mean_baseline",
    "jitter_delta",
    "shimmer_mean_measure",
    "shimmer_mean_baseline",
    "shimmer_delta",
]


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

    def test_returns_metrics_list_with_extended_acoustic_projection(self) -> None:
        """§4.E — Returns canonical observational acoustic fields."""
        mock_conn = MagicMock()
        columns = [
            "id",
            "session_id",
            "segment_id",
            "au12_intensity",
            *_ACOUSTIC_COLUMNS,
            "created_at",
        ]
        mock_cursor = _make_mock_cursor(
            columns,
            [
                (
                    1,
                    "sess-1",
                    "seg-001",
                    2.5,
                    True,
                    False,
                    True,
                    False,
                    2.4,
                    0.0,
                    180.0,
                    None,
                    None,
                    0.02,
                    None,
                    None,
                    0.05,
                    None,
                    None,
                    "2026-03-13T12:00:00Z",
                )
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_metrics(session_id=None, limit=100))

        assert len(result) == 1
        assert result[0]["au12_intensity"] == 2.5
        assert result[0]["f0_valid_measure"] is True
        assert isinstance(result[0]["f0_valid_measure"], bool)
        assert result[0]["voiced_coverage_measure_s"] == 2.4
        assert result[0]["f0_mean_baseline_hz"] is None

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

        call_args = mock_cursor.execute.call_args
        assert "%(session_id)s" in call_args[0][0]
        assert call_args[0][1]["session_id"] == "test-uuid"

    def test_connection_returned_on_success(self) -> None:
        """Connection returned to pool after query."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection") as mock_put,
        ):
            asyncio.run(get_metrics(session_id=None, limit=10))

        mock_put.assert_called_once_with(mock_conn)

    def test_response_exposes_canonical_acoustic_fields_and_omits_sensitive_voice_data(
        self,
    ) -> None:
        """§13.22 / §13.23 — Metrics JSON includes canonical acoustic fields only."""
        mock_conn = MagicMock()
        columns = [
            "id",
            "session_id",
            "segment_id",
            "timestamp_utc",
            "au12_intensity",
            *_ACOUSTIC_COLUMNS,
            "created_at",
        ]
        mock_cursor = _make_mock_cursor(
            columns,
            [
                (
                    7,
                    "sess-1",
                    "seg-007",
                    datetime(2026, 3, 13, 12, 5, 0, tzinfo=UTC),
                    0.75,
                    False,
                    True,
                    False,
                    True,
                    0.0,
                    1.2,
                    None,
                    175.0,
                    None,
                    None,
                    0.01,
                    None,
                    None,
                    0.03,
                    None,
                    datetime(2026, 3, 13, 12, 5, 1, tzinfo=UTC),
                )
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_metrics(session_id="sess-1", limit=5))

        assert len(result) == 1
        row = result[0]
        assert set(row) == {
            "id",
            "session_id",
            "segment_id",
            "timestamp_utc",
            "au12_intensity",
            *_ACOUSTIC_COLUMNS,
            "created_at",
        }
        assert row["timestamp_utc"] == "2026-03-13T12:05:00+00:00"
        assert row["created_at"] == "2026-03-13T12:05:01+00:00"
        assert row["f0_valid_measure"] is False
        assert isinstance(row["f0_valid_measure"], bool)
        assert row["f0_valid_baseline"] is True
        assert isinstance(row["f0_valid_baseline"], bool)
        assert row["perturbation_valid_measure"] is False
        assert isinstance(row["perturbation_valid_measure"], bool)
        assert row["perturbation_valid_baseline"] is True
        assert isinstance(row["perturbation_valid_baseline"], bool)
        assert row["voiced_coverage_measure_s"] == 0.0
        assert row["voiced_coverage_baseline_s"] == 1.2
        assert row["f0_mean_measure_hz"] is None
        assert row["f0_mean_baseline_hz"] == 175.0
        assert row["f0_delta_semitones"] is None
        assert row["jitter_mean_measure"] is None
        assert row["jitter_mean_baseline"] == 0.01
        assert row["jitter_delta"] is None
        assert row["shimmer_mean_measure"] is None
        assert row["shimmer_mean_baseline"] == 0.03
        assert row["shimmer_delta"] is None
        assert "raw_audio" not in row
        assert "voiceprint_embedding" not in row
        assert "reconstructive_voiceprint" not in row

        sql, params = mock_cursor.execute.call_args[0]
        assert params["session_id"] == "sess-1"
        for column in _ACOUSTIC_COLUMNS:
            assert f"m.{column}" in sql
        assert "raw_audio" not in sql
        assert "voiceprint_embedding" not in sql
        assert "reconstructive_voiceprint" not in sql


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

    def test_returns_canonical_acoustic_fields(self) -> None:
        """§7D — Route passes through the canonical observational acoustic payload."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["segment_id", "timestamp_utc", *_ACOUSTIC_COLUMNS],
            [
                (
                    "seg-001",
                    "2026-03-13T12:00:00Z",
                    True,
                    True,
                    True,
                    True,
                    2.5,
                    2.0,
                    220.0,
                    180.0,
                    3.5,
                    0.01,
                    0.008,
                    0.002,
                    0.02,
                    0.018,
                    0.002,
                )
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_acoustic_timeseries("test-uuid"))

        assert len(result) == 1
        assert result[0]["f0_valid_measure"] is True
        assert result[0]["perturbation_valid_baseline"] is True
        assert result[0]["f0_mean_measure_hz"] == 220.0
        assert result[0]["shimmer_delta"] == 0.002

    def test_preserves_false_zero_and_null_canonical_values(self) -> None:
        """§13.22 — Deterministic false/0.0/null values survive JSON projection."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            ["segment_id", "timestamp_utc", *_ACOUSTIC_COLUMNS],
            [
                (
                    "seg-null-acoustic",
                    "2026-03-13T12:01:00Z",
                    False,
                    False,
                    False,
                    False,
                    0.0,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_acoustic_timeseries("test-uuid"))

        assert len(result) == 1
        row = result[0]
        assert row["f0_valid_measure"] is False
        assert isinstance(row["f0_valid_measure"], bool)
        assert row["perturbation_valid_measure"] is False
        assert isinstance(row["perturbation_valid_measure"], bool)
        assert row["voiced_coverage_measure_s"] == 0.0
        assert row["voiced_coverage_baseline_s"] == 0.0
        assert row["f0_mean_measure_hz"] is None
        assert row["jitter_delta"] is None
        assert row["shimmer_delta"] is None

    def test_query_uses_expanded_acoustic_presence_predicate(self) -> None:
        """Rows with canonical-only false/0.0 outputs remain eligible for the endpoint."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["segment_id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            asyncio.run(get_acoustic_timeseries("test-uuid"))

        sql, params = mock_cursor.execute.call_args[0]
        assert params["session_id"] == "test-uuid"
        assert "m.f0_valid_measure IS NOT NULL" in sql
        assert "m.voiced_coverage_measure_s IS NOT NULL" in sql

    def test_response_uses_canonical_field_names_and_omits_sensitive_voice_data(self) -> None:
        """§13.22 / §13.23 — Acoustic endpoint preserves canonical names and safe omissions."""
        mock_conn = MagicMock()
        columns = ["segment_id", "timestamp_utc", *_ACOUSTIC_COLUMNS]
        mock_cursor = _make_mock_cursor(
            columns,
            [
                (
                    "seg-serialize",
                    datetime(2026, 3, 13, 12, 2, 0, tzinfo=UTC),
                    False,
                    True,
                    False,
                    True,
                    0.0,
                    1.8,
                    None,
                    182.0,
                    None,
                    None,
                    0.011,
                    None,
                    None,
                    0.027,
                    None,
                )
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch("services.api.routes.metrics.get_connection", return_value=mock_conn),
            patch("services.api.routes.metrics.put_connection"),
        ):
            result = asyncio.run(get_acoustic_timeseries("test-uuid"))

        assert len(result) == 1
        row = result[0]
        assert set(row) == {"segment_id", "timestamp_utc", *_ACOUSTIC_COLUMNS}
        assert row["timestamp_utc"] == "2026-03-13T12:02:00+00:00"
        assert row["f0_valid_measure"] is False
        assert isinstance(row["f0_valid_measure"], bool)
        assert row["f0_valid_baseline"] is True
        assert isinstance(row["f0_valid_baseline"], bool)
        assert row["perturbation_valid_measure"] is False
        assert isinstance(row["perturbation_valid_measure"], bool)
        assert row["perturbation_valid_baseline"] is True
        assert isinstance(row["perturbation_valid_baseline"], bool)
        assert row["voiced_coverage_measure_s"] == 0.0
        assert row["voiced_coverage_baseline_s"] == 1.8
        assert row["f0_mean_measure_hz"] is None
        assert row["f0_mean_baseline_hz"] == 182.0
        assert row["f0_delta_semitones"] is None
        assert row["jitter_mean_measure"] is None
        assert row["jitter_mean_baseline"] == 0.011
        assert row["jitter_delta"] is None
        assert row["shimmer_mean_measure"] is None
        assert row["shimmer_mean_baseline"] == 0.027
        assert row["shimmer_delta"] is None
        assert "raw_audio" not in row
        assert "voiceprint_embedding" not in row
        assert "reconstructive_voiceprint" not in row

        sql, params = mock_cursor.execute.call_args[0]
        assert params["session_id"] == "test-uuid"
        for column in _ACOUSTIC_COLUMNS:
            assert f"m.{column}" in sql
        assert "raw_audio" not in sql
        assert "voiceprint_embedding" not in sql
        assert "reconstructive_voiceprint" not in sql
