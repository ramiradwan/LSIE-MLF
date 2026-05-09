"""Dirty-state recovery sweep at desktop startup.

Composes the §9.2 Dirty State Recovery steps once before the rest of
the desktop graph spawns:

1. **Orphan IPC blocks** — unlink leftover ``lsie_ipc_*`` SharedMemory
   entries from a previous ungraceful exit. Delegates to
   :mod:`services.desktop_app.ipc.cleanup`.
2. **SQLite WAL checkpoint** — run ``PRAGMA wal_checkpoint(TRUNCATE)``
   so WAL pages from the previous boot land in the main DB and the
   ``-wal`` / ``-shm`` sidecars truncate. Without this the operator
   console's first read could return a partial view if a crash left
   the WAL non-empty.
3. **Reap orphan capture children** — read the
   ``capture_pid_manifest`` rows persisted by ``capture_supervisor``
   and terminate any PIDs that are still alive. Win32 Job Objects
   handle this on Windows automatically; POSIX has no kernel
   equivalent and would otherwise leave scrcpy / adb / ffmpeg
   reparented to init, holding the USB device open.

Each step is non-fatal — a step that fails logs the failure and
continues so the desktop boot still proceeds. The :class:`RecoveryReport`
return shape lets ui_api_shell surface what was actually cleaned.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from services.desktop_app.ipc.cleanup import recover_orphan_ipc_blocks
from services.desktop_app.os_adapter import resolve_capture_dir
from services.desktop_app.privacy.zeroize import cleanup_capture_files

logger = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    """Counts and detail from one recovery sweep."""

    unlinked_ipc_blocks: int = 0
    wal_checkpoint_pages: int | None = None
    reaped_capture_pids: list[int] = field(default_factory=list)
    survived_capture_pids: list[int] = field(default_factory=list)
    deleted_capture_files: list[str] = field(default_factory=list)
    retained_capture_files: list[str] = field(default_factory=list)


def run_recovery_sweep(db_path: Path) -> RecoveryReport:
    """Execute every recovery step in order, never raising on a single failure.

    ``db_path`` must point at an already-bootstrapped SQLite store;
    the parent startup sequence calls
    :func:`services.desktop_app.state.sqlite_schema.bootstrap_schema`
    immediately before this function so the ``capture_pid_manifest``
    table is guaranteed present.
    """
    report = RecoveryReport()

    try:
        report.unlinked_ipc_blocks = recover_orphan_ipc_blocks()
    except Exception:  # noqa: BLE001 — non-fatal during boot
        logger.warning("orphan ipc block sweep failed", exc_info=True)

    try:
        report.wal_checkpoint_pages = wal_checkpoint(db_path)
    except Exception:  # noqa: BLE001 — non-fatal during boot
        logger.warning("sqlite wal_checkpoint failed", exc_info=True)

    try:
        reaped, survived = reap_orphan_capture_processes(db_path)
        report.reaped_capture_pids = reaped
        report.survived_capture_pids = survived
    except Exception:  # noqa: BLE001 — non-fatal during boot
        logger.warning("capture pid manifest reap failed", exc_info=True)

    try:
        deleted, retained = cleanup_capture_files(resolve_capture_dir())
        report.deleted_capture_files = [str(path) for path in deleted]
        report.retained_capture_files = [str(path) for path in retained]
    except Exception:  # noqa: BLE001 — non-fatal during boot
        logger.warning("capture file cleanup failed", exc_info=True)

    logger.info(
        "recovery sweep complete: ipc=%d wal_pages=%s reaped=%s survived=%s "
        "deleted_files=%s retained_files=%s",
        report.unlinked_ipc_blocks,
        report.wal_checkpoint_pages,
        report.reaped_capture_pids,
        report.survived_capture_pids,
        report.deleted_capture_files,
        report.retained_capture_files,
    )
    return report


def wal_checkpoint(db_path: Path) -> int:
    """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` against ``db_path``.

    Returns the number of WAL frames moved into the main DB
    (``checkpointed_frames`` in SQLite's three-tuple result). Truncate
    mode also resets the WAL file to zero size — what we want at boot
    so the operator console's first read sees a settled DB.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()
    # SQLite returns (busy, log_pages, checkpointed_frames). ``busy=1``
    # means another connection holds a write lock — at startup that
    # should never happen, so we surface it as a logged warning.
    if row is None:
        return 0
    busy = int(row[0]) if row[0] is not None else 0
    log_pages = int(row[1]) if row[1] is not None else 0
    if busy:
        logger.warning("wal_checkpoint reported busy=%d log_pages=%d", busy, log_pages)
    return log_pages


def reap_orphan_capture_processes(db_path: Path) -> tuple[list[int], list[int]]:
    """Terminate any capture-child PID still alive from a previous boot.

    Reads every row of ``capture_pid_manifest`` and, for each, tries to
    terminate the process. On Windows this is mostly belt-and-suspenders
    (Job Objects already cleaned the descendant tree); on POSIX it is
    the only mechanism that catches scrcpy / adb / ffmpeg reparented to
    init after an ungraceful supervisor exit.

    Returns ``(reaped, survived)`` PID lists for telemetry. The manifest
    is cleared on the way out — fresh capture_supervisor invocations
    repopulate it.
    """
    import psutil

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT pid, process_kind, parent_process, spawned_at_utc FROM capture_pid_manifest"
        ).fetchall()
    finally:
        conn.close()

    reaped: list[int] = []
    survived: list[int] = []

    for row in rows:
        pid = int(row["pid"])
        kind = str(row["process_kind"])
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            # Already dead — that's the expected path. Treat as
            # successfully reaped so the manifest gets cleared.
            reaped.append(pid)
            continue
        # Terminate gracefully, escalate to kill if it ignores SIGTERM.
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
            reaped.append(pid)
            logger.info("recovery reaped orphan %s pid=%d", kind, pid)
        except psutil.NoSuchProcess:
            reaped.append(pid)
        except (psutil.AccessDenied, psutil.TimeoutExpired):
            survived.append(pid)
            logger.warning("recovery could not reap %s pid=%d", kind, pid)

    if rows:
        clear_conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            clear_conn.execute("DELETE FROM capture_pid_manifest")
        finally:
            clear_conn.close()

    return reaped, survived


def record_capture_pid(
    db_path: Path,
    pid: int,
    *,
    process_kind: str,
    parent_process: str = "capture_supervisor",
) -> None:
    """Insert a row into ``capture_pid_manifest`` so the next sweep can find it.

    Capture supervisor calls this immediately after each
    :class:`SupervisedProcess` spawn. The CHECK constraint on
    ``process_kind`` ensures only the three documented kinds land —
    typos at the call site fail loudly rather than silently bypass the
    recovery sweep.
    """
    if process_kind not in ("scrcpy", "adb", "ffmpeg"):
        raise ValueError(f"unknown capture process_kind: {process_kind!r}")
    from datetime import UTC, datetime

    spawned_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO capture_pid_manifest "
            "(pid, process_kind, parent_process, spawned_at_utc) VALUES (?, ?, ?, ?)",
            (pid, process_kind, parent_process, spawned_at),
        )
    finally:
        conn.close()


def forget_capture_pid(db_path: Path, pid: int) -> None:
    """Remove a manifest row when its child exits cleanly.

    Capture supervisor calls this after a graceful
    :meth:`SupervisedProcess.terminate`; the manifest then only
    contains PIDs that were alive at the time of an ungraceful exit.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("DELETE FROM capture_pid_manifest WHERE pid = ?", (pid,))
    finally:
        conn.close()
