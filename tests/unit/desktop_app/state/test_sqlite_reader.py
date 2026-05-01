"""WS4 P1 — SqliteReader read-only contract tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from services.desktop_app.state.sqlite_reader import SqliteReader
from services.desktop_app.state.sqlite_writer import SqliteWriter


def _bootstrap(tmp_path: Path) -> Path:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    return db


def test_fetch_experiment_arms_returns_seed_rows(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)

    rows = reader.fetch_experiment_arms("greeting_line_v1")

    assert len(rows) == 4
    arms = sorted(row["arm"] for row in rows)
    assert arms == [
        "compliment_content",
        "direct_question",
        "simple_hello",
        "warm_welcome",
    ]
    for row in rows:
        assert row["enabled"] == 1
        assert row["alpha_param"] == pytest.approx(1.0)
        assert row["beta_param"] == pytest.approx(1.0)


def test_fetch_experiment_arms_unknown_id_returns_empty(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    assert reader.fetch_experiment_arms("never_seeded") == []


def test_fetch_active_arm_returns_one_row(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    row = reader.fetch_active_arm_for_experiment("greeting_line_v1")
    assert row is not None
    assert row["experiment_id"] == "greeting_line_v1"
    assert row["enabled"] == 1


def test_fetch_recent_sessions_empty_on_fresh_db(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    assert reader.fetch_recent_sessions(limit=10) == []


def test_fetch_recent_sessions_orders_newest_first(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000010",
                "stream_url": "test://older",
                "started_at": "2026-01-01T00:00:00Z",
            },
        )
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000011",
                "stream_url": "test://newer",
                "started_at": "2026-05-01T00:00:00Z",
            },
        )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    rows = reader.fetch_recent_sessions(limit=10)
    assert len(rows) == 2
    assert rows[0]["stream_url"] == "test://newer"
    assert rows[1]["stream_url"] == "test://older"


def test_fetch_active_session_returns_unended(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    try:
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000020",
                "stream_url": "test://ended",
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T01:00:00Z",
            },
        )
        writer.enqueue(
            "sessions",
            {
                "session_id": "00000000-0000-4000-8000-000000000021",
                "stream_url": "test://active",
                "started_at": "2026-05-01T00:00:00Z",
            },
        )
        writer.flush()
    finally:
        writer.close()

    reader = SqliteReader(db)
    active = reader.fetch_active_session()
    assert active is not None
    assert active["stream_url"] == "test://active"
    assert active["ended_at"] is None


def test_reader_connection_is_query_only(tmp_path: Path) -> None:
    """The vended connection must reject INSERTs (the single-writer rule)."""
    db = _bootstrap(tmp_path)
    reader = SqliteReader(db)
    with (
        reader.connection() as conn,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        conn.execute(
            "INSERT INTO sessions (session_id, stream_url) VALUES (?, ?)",
            ("00000000-0000-4000-8000-000000000099", "violator"),
        )
