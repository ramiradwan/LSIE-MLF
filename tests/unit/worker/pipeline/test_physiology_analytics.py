"""Physiology analytics tests for MetricsStore (§4.E.2, §7C, §12.5)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

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
    """Build misaligned raw RMSSD samples that align only after 5-minute resampling."""
    base_time = datetime(2026, 3, 13, 12, 0, tzinfo=UTC)
    rows: list[tuple[str, float, datetime]] = []
    for idx in range(n_bins):
        bin_start = base_time + timedelta(minutes=idx * 5)
        rows.extend(
            [
                ("streamer", 10.0 + idx * 2.0, bin_start + timedelta(seconds=20)),
                ("operator", 6.0 + idx * 2.0, bin_start + timedelta(seconds=50)),
                (
                    "streamer",
                    14.0 + idx * 2.0,
                    bin_start + timedelta(minutes=4, seconds=5),
                ),
                (
                    "operator",
                    8.0 + idx * 2.0,
                    bin_start + timedelta(minutes=3, seconds=45),
                ),
            ]
        )
    return rows


def test_persist_physiology_snapshot_inserts_row(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """persist_physiology_snapshot writes one parameterized physiology_log row."""
    snapshot = {
        "rmssd_ms": 41.2,
        "heart_rate_bpm": 72,
        "freshness_s": 2.5,
        "is_stale": False,
        "provider": "oura",
        "source_timestamp_utc": "2026-03-13T12:00:00+00:00",
    }

    store.persist_physiology_snapshot("session-1", "seg-001", "streamer", snapshot)

    cursor = mock_conn.cursor.return_value.__enter__.return_value
    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO physiology_log" in sql
    assert params == {
        "session_id": "session-1",
        "segment_id": "seg-001",
        "subject_role": "streamer",
        "rmssd_ms": 41.2,
        "heart_rate_bpm": 72,
        "freshness_s": 2.5,
        "is_stale": False,
        "provider": "oura",
        "source_timestamp_utc": "2026-03-13T12:00:00+00:00",
    }
    mock_conn.set_isolation_level.assert_called_once_with(
        _mock_psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED
    )
    mock_conn.commit.assert_called_once()


def test_compute_comodulation_returns_none_insufficient_data(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """compute_comodulation returns None when fewer than 4 aligned 5-minute bins exist."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = _resampled_rows(3)

    result = store.compute_comodulation("session-1")

    assert result is None
    sql, params = cursor.execute.call_args.args
    assert "FROM physiology_log" in sql
    assert params["session_id"] == "session-1"
    assert params["window_start_utc"] <= datetime.now(UTC)
    assert cursor.execute.call_count == 1
    mock_conn.commit.assert_not_called()


def test_compute_comodulation_returns_valid_index(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """compute_comodulation persists a valid index using 5-minute resampled bins."""
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
    assert insert_params["co_modulation_index"] == pytest.approx(result["co_modulation_index"])
    mock_conn.commit.assert_called_once()


def test_compute_comodulation_resamples_five_minute_bins_before_alignment(
    store: MetricsStore,
    mock_conn: MagicMock,
) -> None:
    """Misaligned raw timestamps still align after 5-minute binning as required by §7C."""
    cursor = mock_conn.cursor.return_value.__enter__.return_value
    rows = _resampled_rows(4)
    cursor.fetchall.return_value = rows

    streamer_ts = {ts for role, _, ts in rows if role == "streamer"}
    operator_ts = {ts for role, _, ts in rows if role == "operator"}
    assert streamer_ts.isdisjoint(operator_ts)

    result = store.compute_comodulation("session-1")

    assert result is not None
    assert result["n_paired_observations"] == 4
    assert result["coverage_ratio"] == 1.0


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
