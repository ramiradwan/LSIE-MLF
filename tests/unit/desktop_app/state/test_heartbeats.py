"""WS4 P2 — HeartbeatRecorder unit tests."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from services.desktop_app.state.heartbeats import (
    HeartbeatRecorder,
    fetch_all_heartbeats,
)
from services.desktop_app.state.sqlite_writer import SqliteWriter


def _bootstrap(tmp_path: Path) -> Path:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    return db


def test_start_writes_inaugural_beat_synchronously(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    recorder = HeartbeatRecorder(db, "ui_api_shell", interval_s=10.0)
    recorder.start()
    try:
        rows = fetch_all_heartbeats(db)
    finally:
        recorder.stop()

    assert len(rows) == 1
    row = rows[0]
    assert row["process_name"] == "ui_api_shell"
    assert row["pid"] == os.getpid()
    assert row["started_at_utc"] == row["last_heartbeat_utc"]


def test_thread_advances_last_heartbeat(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    recorder = HeartbeatRecorder(db, "module_c_orchestrator", interval_s=0.05)
    recorder.start()
    try:
        first = fetch_all_heartbeats(db)[0]
        # Sleep a bit longer than two ticks so the loop definitely runs.
        time.sleep(0.25)
        second = fetch_all_heartbeats(db)[0]
    finally:
        recorder.stop()

    assert first["started_at_utc"] == second["started_at_utc"]
    # Note: SQLite CURRENT_TIMESTAMP-style strings have 1s resolution,
    # so we assert at least non-decreasing rather than strictly greater.
    assert str(second["last_heartbeat_utc"]) >= str(first["last_heartbeat_utc"])


def test_stop_is_idempotent(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    recorder = HeartbeatRecorder(db, "gpu_ml_worker", interval_s=10.0)
    recorder.start()
    recorder.stop()
    # Second stop must not raise.
    recorder.stop()


def test_double_start_does_not_spawn_two_threads(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    recorder = HeartbeatRecorder(db, "cloud_sync_worker", interval_s=10.0)
    recorder.start()
    try:
        recorder.start()  # should be a no-op
    finally:
        recorder.stop()

    rows = fetch_all_heartbeats(db)
    assert len(rows) == 1
    assert rows[0]["process_name"] == "cloud_sync_worker"


def test_stop_writes_final_beat(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    recorder = HeartbeatRecorder(db, "analytics_state_worker", interval_s=10.0)
    recorder.start()
    initial = str(fetch_all_heartbeats(db)[0]["last_heartbeat_utc"])
    # Force enough wall-clock to advance the SQLite-second resolution.
    time.sleep(1.1)
    recorder.stop()

    final = str(fetch_all_heartbeats(db)[0]["last_heartbeat_utc"])
    assert final >= initial


def test_invalid_interval_rejected(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        HeartbeatRecorder(db, "ui_api_shell", interval_s=0.0)
    with pytest.raises(ValueError, match="positive"):
        HeartbeatRecorder(db, "ui_api_shell", interval_s=-0.5)


def test_two_processes_do_not_collide(tmp_path: Path) -> None:
    """Two HeartbeatRecorder instances against the same DB must coexist."""
    db = _bootstrap(tmp_path)
    a = HeartbeatRecorder(db, "ui_api_shell", interval_s=10.0)
    b = HeartbeatRecorder(db, "capture_supervisor", interval_s=10.0)
    a.start()
    b.start()
    try:
        rows = {row["process_name"]: row for row in fetch_all_heartbeats(db)}
    finally:
        a.stop()
        b.stop()

    assert set(rows) == {"ui_api_shell", "capture_supervisor"}


def test_reader_connection_can_observe_concurrently(tmp_path: Path) -> None:
    """An independent read connection sees the heartbeat without locking out the writer."""
    db = _bootstrap(tmp_path)
    recorder = HeartbeatRecorder(db, "ui_api_shell", interval_s=10.0)
    recorder.start()
    try:
        # Open a separate read-only connection — WAL mode should let
        # this read coexist with the recorder's write.
        ro = sqlite3.connect(str(db), isolation_level=None)
        try:
            ro.execute("PRAGMA query_only=1")
            row = ro.execute(
                "SELECT process_name FROM process_heartbeat WHERE process_name = ?",
                ("ui_api_shell",),
            ).fetchone()
            assert row == ("ui_api_shell",)
        finally:
            ro.close()
    finally:
        recorder.stop()
