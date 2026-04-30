"""Cross-platform OS adapter (Platform Abstraction Rule).

This is the **only** module in the desktop app that branches on
``sys.platform``. Every Win32 integration (Job Objects, Credential
Manager, ``SetErrorMode``, Windows-specific path resolution) and every
POSIX fallback (``os.killpg``, ``setrlimit``, ``/dev/shm`` enumeration,
XDG paths) lives behind this interface. Consumer modules
(``capture_supervisor``, ``ui_api_shell``, ``secrets``, ``cleanup``,
etc.) MUST call the public symbols below and stay OS-agnostic.

Phase boundaries: WS3 P2 introduces the IPC-block recovery sweep here.
WS3 P3a (this module) adds :class:`SupervisedProcess`. WS4 P3 adds
crash-dump suppression and memory zeroisation. WS4 P4 adds the
secret-store wrapper. WS1 P2/P4 add path resolution.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def cleanup_orphan_ipc_blocks(prefix: str) -> int:
    """Unlink leftover SharedMemory blocks whose name starts with ``prefix``.

    POSIX: enumerate ``/dev/shm/`` and ``shm_unlink`` each matching name
    so a crashed parent does not leak rebooted-tmpfs entries.

    Windows: no-op. Anonymous-named ``CreateFileMapping`` mappings are
    reference-counted by the kernel and reclaimed on last-handle close;
    a crashed process leaves no orphaned shared region. Tracking-file
    schemes (writing block names to disk) are deferred until WS4 P1's
    SQLite manifest gives us a durable place to record them.

    Returns the number of blocks unlinked. A failure to remove an
    individual block is logged and skipped — the caller's start path
    must remain non-fatal.
    """
    if sys.platform == "win32":
        return 0

    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return 0

    unlinked = 0
    for entry in os.listdir(shm_dir):
        if not entry.startswith(prefix):
            continue
        path = os.path.join(shm_dir, entry)
        try:
            os.unlink(path)
            unlinked += 1
        except OSError as exc:
            logger.warning("orphan ipc block unlink failed: %s (%s)", path, exc)
    return unlinked


class SupervisedProcess:
    """A subprocess whose entire descendant tree is force-killed on close.

    Cross-platform replacement for the v3.4 ``services/stream_ingest/
    entrypoint.sh`` cleanup contract. ``capture_supervisor`` (WS3 P3b)
    will spawn scrcpy / ADB / FFmpeg through this primitive so a crash
    of the supervisor leaves zero zombies holding the USB device open.

    Windows: a per-instance Job Object with
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` is created and the child is
    assigned to it immediately after launch. ``terminate`` calls
    ``TerminateJobObject`` so every descendant inheriting the job dies
    in one syscall. Closing the supervisor process — graceful or not —
    drops the only outstanding job-handle reference, which fires
    ``KILL_ON_JOB_CLOSE`` and reclaims orphans.

    POSIX: the child is launched in a fresh session via
    ``Popen(start_new_session=True)`` so it becomes the leader of its
    own process group. ``terminate`` sends ``SIGTERM`` to the whole
    group via ``os.killpg``, waits ``grace_s`` for graceful shutdown,
    and escalates to ``SIGKILL``. POSIX provides no kernel equivalent
    of Windows' ``KILL_ON_JOB_CLOSE`` — the supervisor MUST call
    :meth:`terminate` (or :meth:`close`) explicitly on shutdown to
    avoid orphaning descendants when its parent process dies.

    Constructor signature mirrors :func:`subprocess.Popen`. Use
    ``stdout=subprocess.PIPE`` and friends as you normally would.
    """

    def __init__(
        self,
        args: list[str],
        **popen_kwargs: Any,
    ) -> None:
        self._closed: bool = False
        self._job: Any = None  # Win32 PyHANDLE; None on POSIX

        if sys.platform == "win32":
            self._proc, self._job = self._spawn_windows(args, popen_kwargs)
        else:
            self._proc = self._spawn_posix(args, popen_kwargs)

    @staticmethod
    def _spawn_windows(
        args: list[str],
        popen_kwargs: dict[str, Any],
    ) -> tuple[subprocess.Popen[Any], Any]:
        """Spawn the child, then assign it to a fresh Job Object.

        AssignProcessToJobObject on a running process is supported on
        Windows 7+ and incurs only a microsecond race window between
        Popen and the assignment. For scrcpy / ADB / FFmpeg — none of
        which spawn grandchildren in their first few instructions —
        this is acceptable. The CREATE_SUSPENDED + ResumeThread dance
        was considered and rejected as over-engineering: the resume
        path requires going around ``subprocess.Popen``'s closed
        thread handle and dramatically complicates the implementation.
        """
        import win32api
        import win32con
        import win32job

        proc = subprocess.Popen(args, **popen_kwargs)

        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
        )
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
            info,
        )

        proc_handle = win32api.OpenProcess(
            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
            False,
            proc.pid,
        )
        try:
            win32job.AssignProcessToJobObject(job, proc_handle)
        finally:
            win32api.CloseHandle(proc_handle)

        return proc, job

    @staticmethod
    def _spawn_posix(
        args: list[str],
        popen_kwargs: dict[str, Any],
    ) -> subprocess.Popen[Any]:
        kwargs = {**popen_kwargs, "start_new_session": True}
        return subprocess.Popen(args, **kwargs)

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def poll(self) -> int | None:
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self._proc.wait(timeout=timeout)

    def terminate(self, grace_s: float = 3.0) -> None:
        """Force-kill the child and every descendant. Idempotent."""
        if self._closed:
            return
        if sys.platform == "win32":
            self._terminate_windows(grace_s)
        else:
            self._terminate_posix(grace_s)

    def _terminate_windows(self, grace_s: float) -> None:
        import win32api
        import win32job

        if self._job is not None:
            try:
                # Exit code 1 — distinguishes job-terminated from natural exit.
                win32job.TerminateJobObject(self._job, 1)
            except Exception:  # noqa: BLE001
                logger.debug("TerminateJobObject failed for pid=%d", self.pid, exc_info=True)
        try:
            self._proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._proc.kill()
            try:
                self._proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                logger.warning("supervised pid=%d ignored TerminateJobObject", self.pid)
        if self._job is not None:
            with contextlib.suppress(Exception):
                win32api.CloseHandle(self._job)
            self._job = None

    def _terminate_posix(self, grace_s: float) -> None:
        # POSIX-only branch — gated by sys.platform != "win32" upstream.
        # Windows-host mypy needs the # type: ignore on the SIGKILL /
        # getpgid / killpg references because they don't exist in the
        # win32 build of the signal / os modules.
        if self._proc.poll() is None:
            try:
                pgid = os.getpgid(self._proc.pid)  # type: ignore[attr-defined]
                os.killpg(pgid, signal.SIGTERM)  # type: ignore[attr-defined]
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self._proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(self._proc.pid)  # type: ignore[attr-defined]
                os.killpg(pgid, signal.SIGKILL)  # type: ignore[attr-defined]
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                logger.warning("supervised pid=%d ignored SIGKILL", self.pid)

    def close(self) -> None:
        """Alias for :meth:`terminate` that also marks the handle closed."""
        if self._closed:
            return
        self.terminate()
        self._closed = True

    def __enter__(self) -> SupervisedProcess:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
