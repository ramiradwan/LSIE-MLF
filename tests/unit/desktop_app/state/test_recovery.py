"""Recovery sweep unit tests.

Cover the three sub-steps independently plus the composed
``run_recovery_sweep`` entry point. Process-reaping is verified by
spawning a short-lived Python child with ``subprocess`` and asserting
``psutil.NoSuchProcess`` after the sweep.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from services.desktop_app.state.recovery import (
    RecoveryReport,
    forget_capture_pid,
    reap_orphan_capture_processes,
    record_capture_pid,
    run_recovery_sweep,
    wal_checkpoint,
)
from services.desktop_app.state.sqlite_writer import SqliteWriter


def _bootstrap(tmp_path: Path) -> Path:
    db = tmp_path / "desktop.sqlite"
    writer = SqliteWriter(db)
    writer.close()
    return db


# ---------------------------------------------------------------------------
# wal_checkpoint
# ---------------------------------------------------------------------------


def test_wal_checkpoint_truncates_wal(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    # Push something into the WAL by inserting via a fresh connection
    # that does NOT immediately checkpoint.
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "INSERT INTO sessions (session_id, stream_url, started_at) VALUES (?, ?, ?)",
            ("00000000-0000-4000-8000-000000000001", "x", "2026-04-01 12:00:00"),
        )
    finally:
        conn.close()

    pages = wal_checkpoint(db)
    assert pages >= 0  # busy=0 at startup time
    # After a TRUNCATE checkpoint, a re-checkpoint sees zero pending pages.
    assert wal_checkpoint(db) == 0


def test_wal_checkpoint_idempotent_on_empty_db(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    assert wal_checkpoint(db) >= 0
    assert wal_checkpoint(db) >= 0


# ---------------------------------------------------------------------------
# record_capture_pid / forget_capture_pid
# ---------------------------------------------------------------------------


def _all_manifest(db: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM capture_pid_manifest").fetchall()]
    finally:
        conn.close()


def test_record_capture_pid_inserts_row(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    record_capture_pid(db, 12345, process_kind="scrcpy")
    rows = _all_manifest(db)
    assert len(rows) == 1
    assert rows[0]["pid"] == 12345
    assert rows[0]["process_kind"] == "scrcpy"
    assert rows[0]["parent_process"] == "capture_supervisor"


def test_record_capture_pid_rejects_unknown_kind(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="process_kind"):
        record_capture_pid(db, 12345, process_kind="rogue")


def test_forget_capture_pid_removes_row(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    record_capture_pid(db, 12345, process_kind="scrcpy")
    forget_capture_pid(db, 12345)
    assert _all_manifest(db) == []


def test_record_capture_pid_replaces_on_pid_collision(tmp_path: Path) -> None:
    """OS PID reuse must not trip the PRIMARY KEY constraint."""
    db = _bootstrap(tmp_path)
    record_capture_pid(db, 12345, process_kind="scrcpy")
    record_capture_pid(db, 12345, process_kind="adb", parent_process="capture_supervisor_v2")
    rows = _all_manifest(db)
    assert len(rows) == 1
    assert rows[0]["process_kind"] == "adb"
    assert rows[0]["parent_process"] == "capture_supervisor_v2"


# ---------------------------------------------------------------------------
# reap_orphan_capture_processes
# ---------------------------------------------------------------------------


def test_reap_returns_empty_lists_on_clean_db(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    reaped, survived = reap_orphan_capture_processes(db)
    assert reaped == []
    assert survived == []


def test_reap_handles_already_dead_pid(tmp_path: Path) -> None:
    """A dead PID in the manifest is treated as successfully reaped."""
    db = _bootstrap(tmp_path)
    # Pick a PID very unlikely to exist on this host. PIDs near max
    # u32 are practically guaranteed unallocated.
    fake_pid = 2_147_483_640
    if psutil.pid_exists(fake_pid):
        pytest.skip(f"chosen sentinel pid {fake_pid} happens to exist")
    record_capture_pid(db, fake_pid, process_kind="scrcpy")

    reaped, survived = reap_orphan_capture_processes(db)
    assert reaped == [fake_pid]
    assert survived == []
    # Manifest must be cleared so the next sweep is a no-op.
    assert _all_manifest(db) == []


def test_reap_terminates_live_orphan_child(tmp_path: Path) -> None:
    """Spawn a long-running Python child, register it, then reap."""
    db = _bootstrap(tmp_path)
    # A Python sleep loop is portable across Windows + POSIX.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    try:
        record_capture_pid(db, proc.pid, process_kind="scrcpy")
        # Brief sleep so psutil is sure to see the process up.
        time.sleep(0.2)
        assert psutil.pid_exists(proc.pid)

        reaped, survived = reap_orphan_capture_processes(db)
        assert proc.pid in reaped
        assert survived == []

        # Wait for the OS to clean up the zombie / exit code so the
        # next assertion is stable.
        proc.wait(timeout=5.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# run_recovery_sweep — composed entry
# ---------------------------------------------------------------------------


def test_run_recovery_sweep_returns_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _bootstrap(tmp_path)
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(capture_dir))
    report = run_recovery_sweep(db)

    assert isinstance(report, RecoveryReport)
    assert report.unlinked_ipc_blocks >= 0
    assert report.wal_checkpoint_pages is not None
    assert report.reaped_capture_pids == []
    assert report.survived_capture_pids == []
    assert report.deleted_capture_files == []
    assert report.retained_capture_files == []


def test_run_recovery_sweep_reaps_dead_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _bootstrap(tmp_path)
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(capture_dir))
    fake_pid = 2_147_483_641
    if psutil.pid_exists(fake_pid):
        pytest.skip(f"chosen sentinel pid {fake_pid} happens to exist")
    record_capture_pid(db, fake_pid, process_kind="scrcpy")

    report = run_recovery_sweep(db)
    assert fake_pid in report.reaped_capture_pids
    assert report.survived_capture_pids == []


def test_run_recovery_sweep_continues_on_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in one step must not abort the others."""
    db = _bootstrap(tmp_path)
    capture_dir = tmp_path / "capture"
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(capture_dir))

    from services.desktop_app.state import recovery as recovery_mod

    def boom(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(recovery_mod, "wal_checkpoint", boom)
    report = run_recovery_sweep(db)
    assert report.wal_checkpoint_pages is None
    assert report.reaped_capture_pids == []
    assert report.survived_capture_pids == []


def test_run_recovery_sweep_deletes_stale_capture_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _bootstrap(tmp_path)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    audio = capture_dir / "audio_stream.wav"
    video = capture_dir / "video_stream.mkv"
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(capture_dir))

    report = run_recovery_sweep(db)

    assert report.deleted_capture_files == [str(audio), str(video)]
    assert report.retained_capture_files == []
    assert not audio.exists()
    assert not video.exists()
