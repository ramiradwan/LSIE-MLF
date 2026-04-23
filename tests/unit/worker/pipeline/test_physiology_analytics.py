"""Physiology analytics tests for MetricsStore (§4.E.2, §7C, §12.5)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from typing import Self
from unittest.mock import MagicMock, patch

import pytest

# Install mock psycopg2 before analytics methods attempt lazy imports.
_mock_psycopg2 = MagicMock()
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED = 1
_mock_psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE = 6


@pytest.fixture(autouse=True)
def _patch_psycopg2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure psycopg2 mock is available for lazy imports."""
    monkeypatch.setitem(sys.modules, "psycopg2", _mock_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", _mock_psycopg2.pool)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", _mock_psycopg2.extensions)


from services.worker.pipeline.analytics import (  # noqa: E402
    COMOD_WINDOW_MINUTES,
    MetricsStore,
    ThompsonSamplingEngine,
)


@pytest.fixture()
def mock_conn() -> MagicMock:
    """Create a mock connection with cursor context-manager support."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture()
def store(mock_conn: MagicMock) -> MetricsStore:
    """MetricsStore with mocked pool and psycopg2 module."""
    metrics_store = MetricsStore()
    mock_pool = MagicMock()
    metrics_store._pool = mock_pool
    metrics_store._psycopg2 = _mock_psycopg2
    mock_pool.getconn.return_value = mock_conn
    return metrics_store


def _resampled_rows(n_bins: int) -> list[tuple[str, float, datetime]]:
    """Build recent misaligned 1-minute RMSSD samples within the 10-minute window."""
    base_time = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=n_bins)
    rows: list[tuple[str, float, datetime]] = []
    for idx in range(n_bins):
        bin_start = base_time + timedelta(minutes=idx)
        rows.extend(
            [
                ("streamer", 10.0 + idx * 2.0, bin_start + timedelta(seconds=5)),
                ("operator", 6.0 + idx * 2.0, bin_start + timedelta(seconds=35)),
                ("streamer", 14.0 + idx * 2.0, bin_start + timedelta(seconds=25)),
                ("operator", 8.0 + idx * 2.0, bin_start + timedelta(seconds=50)),
            ]
        )
    return rows


def _zero_variance_rows(n_bins: int) -> list[tuple[str, float, datetime]]:
    """Build recent aligned 1-minute bins with zero variance for one subject series."""
    base_time = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=n_bins)
    rows: list[tuple[str, float, datetime]] = []
    for idx in range(n_bins):
        bin_start = base_time + timedelta(minutes=idx)
        rows.extend(
            [
                ("streamer", 12.0, bin_start + timedelta(seconds=5)),
                ("operator", 7.0 + float(idx), bin_start + timedelta(seconds=35)),
                ("streamer", 12.0, bin_start + timedelta(seconds=25)),
                ("operator", 7.0 + float(idx), bin_start + timedelta(seconds=50)),
            ]
        )
    return rows


def test_persist_physiology_snapshot_inserts_row(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """persist_physiology_snapshot writes only scalar physiology_log fields."""
    snapshot = {
        "rmssd_ms": 41.2,
        "heart_rate_bpm": 72,
        "freshness_s": 2.5,
        "is_stale": False,
        "provider": "oura",
        "source_kind": "ibi",
        "derivation_method": "server",
        "window_s": 300,
        "validity_ratio": 0.92,
        "is_valid": True,
        "source_timestamp_utc": "2026-03-13T12:00:00+00:00",
        "payload": {
            "ibi_ms_items": [810.0, 815.0, 805.0],
            "heart_rate_items_bpm": [71, 72, 73],
        },
        "provider_body": {"raw": "sensitive"},
    }

    store.persist_physiology_snapshot("session-1", "seg-001", "streamer", snapshot)

    cursor = mock_conn.cursor.return_value.__enter__.return_value
    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO physiology_log" in sql
    assert list(params.keys()) == [
        "session_id",
        "segment_id",
        "subject_role",
        "rmssd_ms",
        "heart_rate_bpm",
        "freshness_s",
        "is_stale",
        "provider",
        "source_kind",
        "derivation_method",
        "window_s",
        "validity_ratio",
        "is_valid",
        "source_timestamp_utc",
    ]
    assert params == {
        "session_id": "session-1",
        "segment_id": "seg-001",
        "subject_role": "streamer",
        "rmssd_ms": 41.2,
        "heart_rate_bpm": 72,
        "freshness_s": 2.5,
        "is_stale": False,
        "provider": "oura",
        "source_kind": "ibi",
        "derivation_method": "server",
        "window_s": 300,
        "validity_ratio": 0.92,
        "is_valid": True,
        "source_timestamp_utc": "2026-03-13T12:00:00+00:00",
    }
    assert "payload" not in params
    assert "provider_body" not in params
    mock_conn.set_isolation_level.assert_called_once_with(
        _mock_psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED
    )
    mock_conn.commit.assert_called_once()


def test_persist_physiology_snapshot_handles_missing_metadata_defensively(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """Missing metadata binds as nulls without leaking legacy/raw fields."""
    snapshot = {
        "rmssd_ms": None,
        "heart_rate_bpm": 68,
        "freshness_s": 12.0,
        "is_stale": False,
        "provider": "oura",
        "window_length_s": 300,
        "source_timestamp_utc": "2026-03-13T12:00:00+00:00",
        "payload": {"rmssd_items_ms": [40.0, 41.0]},
        "provider_body": {"event": "session"},
    }

    store.persist_physiology_snapshot("session-1", "seg-002", "operator", snapshot)

    cursor = mock_conn.cursor.return_value.__enter__.return_value
    _, params = cursor.execute.call_args.args
    assert list(params.keys()) == [
        "session_id",
        "segment_id",
        "subject_role",
        "rmssd_ms",
        "heart_rate_bpm",
        "freshness_s",
        "is_stale",
        "provider",
        "source_kind",
        "derivation_method",
        "window_s",
        "validity_ratio",
        "is_valid",
        "source_timestamp_utc",
    ]
    assert params == {
        "session_id": "session-1",
        "segment_id": "seg-002",
        "subject_role": "operator",
        "rmssd_ms": None,
        "heart_rate_bpm": 68,
        "freshness_s": 12.0,
        "is_stale": False,
        "provider": "oura",
        "source_kind": None,
        "derivation_method": None,
        "window_s": 300,
        "validity_ratio": None,
        "is_valid": None,
        "source_timestamp_utc": "2026-03-13T12:00:00+00:00",
    }
    assert "payload" not in params
    assert "provider_body" not in params


def test_compute_comodulation_returns_none_insufficient_data(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """compute_comodulation returns and persists a null-valid result below 4 aligned bins."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = _resampled_rows(3)

    result = store.compute_comodulation("session-1")

    query_sql, query_params = cursor.execute.call_args_list[0].args
    assert "FROM physiology_log" in query_sql
    assert query_params["session_id"] == "session-1"
    assert query_params["window_start_utc"] < query_params["window_end_utc"] <= datetime.now(UTC)

    assert result["session_id"] == "session-1"
    assert result["window_start_utc"] == query_params["window_start_utc"]
    assert result["window_end_utc"] == query_params["window_end_utc"]
    assert result["window_minutes"] == COMOD_WINDOW_MINUTES
    assert result["co_modulation_index"] is None
    assert result["n_paired_observations"] == 3
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["streamer_rmssd_mean"] == pytest.approx(14.0)
    assert result["operator_rmssd_mean"] == pytest.approx(9.0)

    assert cursor.execute.call_count == 2
    insert_sql, insert_params = cursor.execute.call_args_list[-1].args
    assert "INSERT INTO comodulation_log" in insert_sql
    assert insert_params["session_id"] == result["session_id"]
    assert insert_params["window_start_utc"] == result["window_start_utc"]
    assert insert_params["window_end_utc"] == result["window_end_utc"]
    assert insert_params["window_minutes"] == COMOD_WINDOW_MINUTES
    assert insert_params["co_modulation_index"] is None
    assert insert_params["n_paired_observations"] == 3
    assert insert_params["coverage_ratio"] == pytest.approx(1.0)
    assert insert_params["streamer_rmssd_mean"] == pytest.approx(14.0)
    assert insert_params["operator_rmssd_mean"] == pytest.approx(9.0)
    mock_conn.commit.assert_called_once()


def test_compute_comodulation_returns_valid_index(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """compute_comodulation persists a valid index using 1-minute resampled bins."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = _resampled_rows(4)

    result = store.compute_comodulation("session-1")

    assert result is not None
    assert result["session_id"] == "session-1"
    assert result["window_minutes"] == COMOD_WINDOW_MINUTES
    assert result["n_paired_observations"] == 4
    assert result["coverage_ratio"] == 1.0
    assert result["co_modulation_index"] == pytest.approx(1.0)
    assert result["streamer_rmssd_mean"] == pytest.approx(15.0)
    assert result["operator_rmssd_mean"] == pytest.approx(10.0)

    assert cursor.execute.call_count == 2
    insert_sql, insert_params = cursor.execute.call_args_list[-1].args
    assert "INSERT INTO comodulation_log" in insert_sql
    assert insert_params["window_minutes"] == COMOD_WINDOW_MINUTES
    assert insert_params["co_modulation_index"] == pytest.approx(result["co_modulation_index"])
    mock_conn.commit.assert_called_once()


def test_compute_comodulation_persists_null_index_for_zero_variance(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """Zero-variance aligned series return and persist a null-valid co-modulation row."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = _zero_variance_rows(4)

    result = store.compute_comodulation("session-1")

    query_params = cursor.execute.call_args_list[0].args[1]
    assert result["session_id"] == "session-1"
    assert result["window_start_utc"] == query_params["window_start_utc"]
    assert result["window_end_utc"] == query_params["window_end_utc"]
    assert result["window_minutes"] == COMOD_WINDOW_MINUTES
    assert result["co_modulation_index"] is None
    assert result["n_paired_observations"] == 4
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["streamer_rmssd_mean"] == pytest.approx(12.0)
    assert result["operator_rmssd_mean"] == pytest.approx(8.5)

    assert cursor.execute.call_count == 2
    insert_sql, insert_params = cursor.execute.call_args_list[-1].args
    assert "INSERT INTO comodulation_log" in insert_sql
    assert insert_params["session_id"] == result["session_id"]
    assert insert_params["window_start_utc"] == result["window_start_utc"]
    assert insert_params["window_end_utc"] == result["window_end_utc"]
    assert insert_params["window_minutes"] == COMOD_WINDOW_MINUTES
    assert insert_params["co_modulation_index"] is None
    assert insert_params["n_paired_observations"] == 4
    assert insert_params["coverage_ratio"] == pytest.approx(1.0)
    assert insert_params["streamer_rmssd_mean"] == pytest.approx(12.0)
    assert insert_params["operator_rmssd_mean"] == pytest.approx(8.5)
    mock_conn.commit.assert_called_once()


def test_compute_comodulation_resamples_one_minute_bins_before_alignment(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """Misaligned samples align in 1-minute bins while excluding a sample at window_start_utc."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    frozen_now = datetime(2026, 3, 13, 12, 10, tzinfo=UTC)
    window_start_utc = frozen_now - timedelta(minutes=COMOD_WINDOW_MINUTES)
    rows = [
        ("streamer", 10.0, window_start_utc),
        ("operator", 7.0, window_start_utc + timedelta(seconds=30)),
        ("streamer", 12.0, window_start_utc + timedelta(minutes=1, seconds=5)),
        ("operator", 8.0, window_start_utc + timedelta(minutes=1, seconds=35)),
        ("streamer", 14.0, window_start_utc + timedelta(minutes=2, seconds=5)),
        ("operator", 9.0, window_start_utc + timedelta(minutes=2, seconds=35)),
        ("streamer", 16.0, window_start_utc + timedelta(minutes=3, seconds=5)),
        ("operator", 10.0, window_start_utc + timedelta(minutes=3, seconds=35)),
    ]
    cursor.fetchall.return_value = rows

    streamer_ts = {ts for role, _, ts in rows if role == "streamer"}
    operator_ts = {ts for role, _, ts in rows if role == "operator"}
    assert streamer_ts.isdisjoint(operator_ts)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> Self:
            del tz
            return cls.fromtimestamp(frozen_now.timestamp(), tz=UTC)

    with patch("services.worker.pipeline.analytics.datetime", FrozenDateTime):
        result = store.compute_comodulation("session-1")

    query_sql, query_params = cursor.execute.call_args_list[0].args
    assert "FROM physiology_log" in query_sql
    assert query_params["window_start_utc"] == window_start_utc
    assert query_params["window_end_utc"] == frozen_now

    assert result["window_start_utc"] == window_start_utc
    assert result["window_end_utc"] == frozen_now
    assert result["window_minutes"] == COMOD_WINDOW_MINUTES
    assert result["co_modulation_index"] is None
    assert result["n_paired_observations"] == 3
    assert result["coverage_ratio"] == pytest.approx(0.75)
    assert result["streamer_rmssd_mean"] == pytest.approx(14.0)
    assert result["operator_rmssd_mean"] == pytest.approx(8.5)

    insert_sql, insert_params = cursor.execute.call_args_list[-1].args
    assert "INSERT INTO comodulation_log" in insert_sql
    assert insert_params["window_start_utc"] == window_start_utc
    assert insert_params["window_end_utc"] == frozen_now
    assert insert_params["co_modulation_index"] is None
    assert insert_params["n_paired_observations"] == 3
    assert insert_params["coverage_ratio"] == pytest.approx(0.75)


def test_compute_comodulation_excludes_stale_samples(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """The physiology query excludes stale and null-RMSSD samples at the SQL layer."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = _resampled_rows(1)

    store.compute_comodulation("session-1")

    query_sql = cursor.execute.call_args_list[0].args[0]
    assert "AND is_stale = FALSE" in query_sql
    assert "AND rmssd_ms IS NOT NULL" in query_sql


def test_thompson_sampling_engine_unchanged() -> None:
    """ThompsonSamplingEngine keeps only its existing public methods."""
    public_methods = {
        name
        for name, value in ThompsonSamplingEngine.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert public_methods == {"select_arm", "update"}
    assert not hasattr(ThompsonSamplingEngine, "persist_physiology_snapshot")
    assert not hasattr(ThompsonSamplingEngine, "compute_comodulation")
