"""SqliteWriter batch flush tests."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from services.desktop_app.state.sqlite_writer import SqliteWriter


def _open_reader(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def test_enqueue_then_flush_inserts_row(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000001",
                "stream_url": "test://1",
            },
        )
        rows_written = writer.flush()
        assert rows_written == 1
        assert writer.stats.rows_written == 1
    finally:
        writer.close()

    reader = _open_reader(db)
    row = reader.execute(
        "SELECT session_id, stream_url FROM sessions WHERE session_id = ?",
        ("00000000-0000-4000-8000-000000000001",),
    ).fetchone()
    reader.close()
    assert row is not None
    assert row["stream_url"] == "test://1"


def test_metrics_full_section_7d_columns_round_trip(tmp_path: Path) -> None:
    """A metrics row with every §7D acoustic column must round-trip."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000002",
                "stream_url": "test://metrics",
            },
        )
        writer.enqueue(
            "metrics",
            {
                "session_id": "00000000-0000-4000-8000-000000000002",
                "segment_id": "abc" * 21 + "a",
                "timestamp_utc": "2026-05-01T10:00:00Z",
                "au12_intensity": 0.42,
                "f0_valid_measure": 1,
                "f0_valid_baseline": 0,
                "perturbation_valid_measure": 1,
                "perturbation_valid_baseline": 1,
                "voiced_coverage_measure_s": 4.5,
                "voiced_coverage_baseline_s": 3.0,
                "f0_mean_measure_hz": 220.0,
                "f0_mean_baseline_hz": 200.0,
                "f0_delta_semitones": 1.66,
                "jitter_mean_measure": 0.012,
                "jitter_mean_baseline": 0.010,
                "jitter_delta": 0.002,
                "shimmer_mean_measure": 0.05,
                "shimmer_mean_baseline": 0.04,
                "shimmer_delta": 0.01,
            },
        )
        writer.flush()
    finally:
        writer.close()

    reader = _open_reader(db)
    row = reader.execute(
        "SELECT * FROM metrics WHERE session_id = ?",
        ("00000000-0000-4000-8000-000000000002",),
    ).fetchone()
    reader.close()
    assert row is not None
    assert row["f0_delta_semitones"] == pytest.approx(1.66, abs=1e-9)
    assert row["jitter_delta"] == pytest.approx(0.002, abs=1e-9)
    assert row["shimmer_delta"] == pytest.approx(0.01, abs=1e-9)


def test_unknown_table_rejected(tmp_path: Path) -> None:
    writer = SqliteWriter(tmp_path / "x.sqlite")
    try:
        with pytest.raises(ValueError, match="unknown table"):
            writer.enqueue("not_a_real_table", {"foo": 1})
    finally:
        writer.close()


def test_unknown_column_rejected(tmp_path: Path) -> None:
    writer = SqliteWriter(tmp_path / "x.sqlite")
    try:
        with pytest.raises(ValueError, match="unknown columns"):
            writer.enqueue(
                "sessions",
                {
                    "session_id": "00000000-0000-4000-8000-000000000003",
                    "stream_url": "test://3",
                    "stowaway": "should fail",
                },
            )
    finally:
        writer.close()


def test_flush_thread_drains_periodically(tmp_path: Path) -> None:
    """Calling start() must spawn a thread that eventually flushes."""
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db, flush_interval_s=0.05)
    try:
        writer.start()
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000004",
                "stream_url": "test://4",
            },
        )

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if writer.stats.rows_written >= 1:
                break
            time.sleep(0.05)

        assert writer.stats.rows_written >= 1
        assert writer.stats.flush_cycles >= 1
    finally:
        writer.close()


def test_close_drains_pending_records(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db, flush_interval_s=10.0)
    writer.start()
    writer.enqueue(
        "sessions",
        {
            "session_id": "00000000-0000-4000-8000-000000000005",
            "stream_url": "test://5",
        },
    )
    writer.close()

    reader = _open_reader(db)
    row = reader.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?",
        ("00000000-0000-4000-8000-000000000005",),
    ).fetchone()
    reader.close()
    assert row is not None
