"""
Tests for services/api/routes/encounters.py

Verifies encounter log endpoints against:
  §4.E.1 — Thompson Sampling encounter audit trail
  §11 — Variable Extraction Matrix queries
  §2 step 7 — Parameterized queries
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from services.api.routes.encounters import (
    _rows_to_dicts,
    get_encounter_summary,
    list_encounters,
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


class TestListEncounters:
    """§4.E.1 — Encounter log query endpoint."""

    def test_returns_encounter_list(self) -> None:
        """Returns list of encounter log dicts with reward trace fields."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            [
                "id",
                "session_id",
                "segment_id",
                "experiment_id",
                "arm",
                "timestamp_utc",
                "gated_reward",
                "p90_intensity",
                "semantic_gate",
                "is_valid",
                "n_frames",
                "baseline_neutral",
                "stimulus_time",
                "created_at",
            ],
            [
                (
                    1,
                    "sess-1",
                    "seg-1",
                    "greeting_line_v1",
                    "warm_welcome",
                    "2026-04-01T12:00:00Z",
                    0.72,
                    0.81,
                    1,
                    True,
                    130,
                    0.12,
                    3.5,
                    "2026-04-01T12:00:02Z",
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
        ):
            result = asyncio.run(
                list_encounters(
                    experiment_id=None,
                    arm=None,
                    valid_only=False,
                    limit=100,
                )
            )

        assert len(result) == 1
        assert result[0]["experiment_id"] == "greeting_line_v1"
        assert result[0]["arm"] == "warm_welcome"
        assert result[0]["gated_reward"] == 0.72
        assert result[0]["p90_intensity"] == 0.81
        assert result[0]["semantic_gate"] == 1
        assert result[0]["n_frames"] == 130

    def test_row_serializer_preserves_additive_semantic_attribution_columns(self) -> None:
        """Pass-through serialization keeps legacy fields plus additive readbacks."""
        cursor = _make_mock_cursor(
            [
                "id",
                "semantic_gate",
                "semantic_reasoning",
                "semantic_is_match",
                "semantic_confidence_score",
                "soft_reward_candidate",
                "outcome_link_lag_s",
            ],
            [
                (
                    1,
                    0,
                    "cross_encoder_high_nonmatch",
                    False,
                    0.0,
                    0.0,
                    0.0,
                )
            ],
        )

        [row] = _rows_to_dicts(cursor)

        assert row["id"] == 1
        assert row["semantic_gate"] == 0
        assert row["semantic_reasoning"] == "cross_encoder_high_nonmatch"
        assert row["semantic_is_match"] is False
        assert row["semantic_confidence_score"] == 0.0
        assert row["soft_reward_candidate"] == 0.0
        assert row["outcome_link_lag_s"] == 0.0

    def test_filters_by_experiment_id(self) -> None:
        """GET /encounters?experiment_id=X uses parameterized filter."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
        ):
            asyncio.run(
                list_encounters(
                    experiment_id="greeting_line_v1",
                    arm=None,
                    valid_only=False,
                    limit=50,
                )
            )

        call_args = mock_cursor.execute.call_args
        assert "%(experiment_id)s" in call_args[0][0]
        assert call_args[0][1]["experiment_id"] == "greeting_line_v1"

    def test_filters_by_arm(self) -> None:
        """GET /encounters?arm=X uses parameterized filter."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
        ):
            asyncio.run(
                list_encounters(
                    experiment_id=None,
                    arm="warm_welcome",
                    valid_only=False,
                    limit=50,
                )
            )

        call_args = mock_cursor.execute.call_args
        assert "%(arm)s" in call_args[0][0]
        assert call_args[0][1]["arm"] == "warm_welcome"

    def test_valid_only_filter(self) -> None:
        """valid_only=True adds the is_valid condition."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
        ):
            asyncio.run(
                list_encounters(
                    experiment_id=None,
                    arm=None,
                    valid_only=True,
                    limit=50,
                )
            )

        call_args = mock_cursor.execute.call_args
        assert "e.is_valid = TRUE" in call_args[0][0]

    def test_connection_returned(self) -> None:
        """Connection returned to pool after query."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["id"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection") as mock_put,
        ):
            asyncio.run(
                list_encounters(
                    experiment_id=None,
                    arm=None,
                    valid_only=False,
                    limit=10,
                )
            )

        mock_put.assert_called_once_with(mock_conn)


class TestGetEncounterSummary:
    """§4.E.1 — Per-arm encounter summary."""

    def test_returns_per_arm_aggregation(self) -> None:
        """Summary includes avg_reward, encounter_count, gate_rate, avg_frames."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            [
                "arm",
                "encounter_count",
                "valid_count",
                "avg_reward",
                "avg_valid_reward",
                "avg_p90",
                "gate_rate",
                "avg_frames",
                "first_encounter",
                "last_encounter",
            ],
            [
                (
                    "warm_welcome",
                    10,
                    8,
                    0.65,
                    0.72,
                    0.68,
                    0.80,
                    130,
                    "2026-04-01T12:00:00Z",
                    "2026-04-01T14:00:00Z",
                ),
                (
                    "simple_hello",
                    10,
                    7,
                    0.45,
                    0.52,
                    0.48,
                    0.70,
                    125,
                    "2026-04-01T12:05:00Z",
                    "2026-04-01T14:05:00Z",
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
        ):
            result = asyncio.run(get_encounter_summary("greeting_line_v1"))

        assert len(result) == 2
        assert result[0]["arm"] == "warm_welcome"
        assert result[0]["avg_reward"] == 0.65
        assert result[0]["encounter_count"] == 10
        assert result[0]["gate_rate"] == 0.80
        assert result[0]["avg_frames"] == 130

    def test_experiment_not_found(self) -> None:
        """Returns 404 when no encounters exist for experiment."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(["arm"], [])
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
            pytest.raises(Exception) as exc_info,
        ):
            asyncio.run(get_encounter_summary("nonexistent"))

        exc: Any = exc_info.value
        assert exc.status_code == 404
        assert "No encounters found" in exc.detail

    def test_parameterized_query(self) -> None:
        """Summary query parameterizes experiment_id."""
        mock_conn = MagicMock()
        mock_cursor = _make_mock_cursor(
            [
                "arm",
                "encounter_count",
                "valid_count",
                "avg_reward",
                "avg_valid_reward",
                "avg_p90",
                "gate_rate",
                "avg_frames",
                "first_encounter",
                "last_encounter",
            ],
            [
                (
                    "arm_a",
                    5,
                    5,
                    0.5,
                    0.5,
                    0.5,
                    1.0,
                    135,
                    "2026-04-01T12:00:00Z",
                    "2026-04-01T12:00:00Z",
                ),
            ],
        )
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "services.api.routes.encounters.get_connection",
                return_value=mock_conn,
            ),
            patch("services.api.routes.encounters.put_connection"),
        ):
            asyncio.run(get_encounter_summary("greeting_line_v1"))

        call_args = mock_cursor.execute.call_args
        assert "%(experiment_id)s" in call_args[0][0]
        assert call_args[0][1]["experiment_id"] == "greeting_line_v1"
