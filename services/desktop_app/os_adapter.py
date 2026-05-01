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
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_state_dir() -> Path:
    """Return the directory the desktop app uses for its local state.

    Resolution order:

      1. ``LSIE_STATE_DIR`` env override (operator-set, e.g. for tests).
      2. Windows: ``%LOCALAPPDATA%\\LSIE-MLF\\state`` (the standard
         per-user roaming-free path on Windows 10+).
      3. POSIX: ``$XDG_DATA_HOME/lsie-mlf/state`` if the XDG variable
         is set, else ``~/.local/share/lsie-mlf/state``.

    The directory is created (with parents) before return so callers
    can immediately ``open(path / 'foo.sqlite', 'a')`` without an
    ``os.makedirs`` of their own.
    """
    override = os.environ.get("LSIE_STATE_DIR", "").strip()
    if override:
        target = Path(override).expanduser()
    elif sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            target = Path(local_appdata) / "LSIE-MLF" / "state"
        else:
            target = Path.home() / "AppData" / "Local" / "LSIE-MLF" / "state"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        if xdg:
            target = Path(xdg) / "lsie-mlf" / "state"
        else:
            target = Path.home() / ".local" / "share" / "lsie-mlf" / "state"

    target.mkdir(parents=True, exist_ok=True)
    return target


def find_executable(name: str, env_override: str | None = None) -> str:
    """Resolve a system tool's full path. ``PATH`` first, then known fallbacks.

    Designed for ``capture_supervisor`` (WS3 P3) to locate ``scrcpy`` /
    ``adb`` / ``ffmpeg`` reliably even when the shell PATH was not
    refreshed after a winget install. Resolution order:

      1. If ``env_override`` is set in the environment, use it verbatim.
         (E.g. ``LSIE_SCRCPY_PATH`` for operator overrides.)
      2. ``shutil.which(name)`` — honours the current process's PATH.
      3. Windows-specific package-manager scan: walk
         ``%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages`` for
         ``{name}*.exe`` so a freshly winget-installed tool resolves
         even before the shell picks up the PATH update.
      4. Common fixed installation paths
         (``C:\\Program Files\\<name>\\<name>.exe`` etc.).

    Raises ``FileNotFoundError`` with a friendly message if every
    candidate fails — the caller turns that into an operator-readable
    health-page surface.
    """
    if env_override:
        override_path = os.environ.get(env_override, "").strip()
        if override_path:
            if not os.path.isfile(override_path):
                raise FileNotFoundError(
                    f"{env_override}={override_path!r} does not point at an existing file"
                )
            return override_path

    found = shutil.which(name)
    if found:
        return found

    if sys.platform == "win32":
        winget_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.is_dir():
            target = f"{name}.exe"
            for candidate in winget_root.rglob(target):
                if candidate.is_file():
                    return str(candidate)

        program_files = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        for layout in (
            program_files / name / f"{name}.exe",
            program_files / name.capitalize() / f"{name}.exe",
        ):
            if layout.is_file():
                return str(layout)

    raise FileNotFoundError(
        f"could not locate executable {name!r}: PATH lookup failed and no fallback path matched"
    )


def zeroize_shared_memory(shm: Any) -> int:
    """Overwrite a :class:`multiprocessing.shared_memory.SharedMemory`'s bytes.

    Calls ``ctypes.memset`` against the raw ``SharedMemory.buf``-backing
    buffer so transient PCM media never survives a leaked block.
    Returns the number of bytes zeroed (or ``0`` if the buffer is already
    detached). Idempotent: repeated calls are safe; a closed buffer
    short-circuits.

    Why ctypes.memset rather than ``buf[:] = b'\\x00' * len``: the
    former is a single libc call against the mapping address and works
    even when the producer has already closed its Python-side memoryview
    handle but the underlying mapping is still live. The latter goes
    through Python's bytes-allocation path and asserts the memoryview
    is writable.

    POSIX and Windows behave identically — ``shm.buf`` is a
    ``memoryview`` over an mmap on POSIX and a ``CreateFileMapping``
    on Windows, but the underlying ctypes pointer is uniform.
    """
    import ctypes

    buf = getattr(shm, "buf", None)
    if buf is None:
        return 0
    try:
        nbytes = len(buf)
    except (TypeError, ValueError):
        return 0
    if nbytes == 0:
        return 0
    addr = ctypes.addressof(ctypes.c_byte.from_buffer(buf))
    ctypes.memset(addr, 0, nbytes)
    return int(nbytes)


def suppress_crash_dialogs() -> None:
    """Disable the OS crash-dialog popup so a child crash dies silently.

    Windows: ``SetErrorMode(SEM_NOGPFAULTERRORBOX |
    SEM_FAILCRITICALERRORS)`` suppresses the "application has stopped
    working" dialog and the "no disk in drive" critical-error popup.
    Without this, an ungraceful exit in any child process surfaces a
    modal dialog that prevents the parent from cleanly observing the
    child exit code.

    POSIX: no-op. Linux / macOS do not raise modal dialogs on crash.

    Idempotent: callable multiple times without side effects.
    """
    if sys.platform != "win32":
        return
    import ctypes

    # Win32 SetErrorMode flag values from winbase.h:
    # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX.
    flags = 0x0001 | 0x0002 | 0x8000
    try:
        ctypes.windll.kernel32.SetErrorMode(flags)
    except OSError as exc:
        logger.debug("SetErrorMode failed: %s", exc)


def register_localdumps_exclusion(app_name: str = "lsie-mlf-desktop.exe") -> bool:
    """Suppress Windows Error Reporting LocalDumps for our app binary.

    Per §5.2 the Ephemeral Vault contract is that no biometric media
    survives the 24h secure-deletion window. WER's LocalDumps feature,
    when configured globally, will write a ``.dmp`` of any crashing
    process under ``%LOCALAPPDATA%\\CrashDumps\\``. A crash mid-segment
    would therefore freeze raw PCM into a dump file outside the vault
    perimeter. Adding our own ``...\\LocalDumps\\<app_name>`` subkey
    with ``DumpType=0`` overrides any global config and keeps WER from
    writing the dump.

    Returns ``True`` if the key was successfully written, ``False`` on
    POSIX or any registry failure. Non-fatal: a failure to write the
    key logs and continues — the absence of the override is no worse
    than a default Windows install.
    """
    if sys.platform != "win32":
        return False
    try:
        import winreg
    except ImportError:  # pragma: no cover — winreg is stdlib on Windows
        return False

    subkey = r"Software\Microsoft\Windows\Windows Error Reporting\LocalDumps\\" + app_name
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, subkey, 0, winreg.KEY_SET_VALUE) as key:
            # DumpType=0 means "custom" with the dump-file count below;
            # combined with DumpCount=0 this is the documented "do not
            # write a dump" combination.
            winreg.SetValueEx(key, "DumpType", 0, winreg.REG_DWORD, 0)
            winreg.SetValueEx(key, "DumpCount", 0, winreg.REG_DWORD, 0)
        return True
    except OSError as exc:
        logger.warning("LocalDumps exclusion for %s failed: %s", app_name, exc)
        return False


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
