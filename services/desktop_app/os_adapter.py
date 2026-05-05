"""Cross-platform OS adapter.

This is the only desktop-app module that branches on ``sys.platform``.
Win32 integrations and POSIX fallbacks live behind this interface so
consumer modules stay OS-agnostic.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_state_dir() -> Path:
    """Return the directory the desktop app uses for its local state."""
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


def _discover_git_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_capture_dir() -> Path:
    """Return the governed transient capture directory for raw media files."""
    override = os.environ.get("LSIE_CAPTURE_DIR", "").strip()
    if override:
        target = Path(override).expanduser().resolve()
        forbidden = {Path.cwd().resolve()}
        git_root = _discover_git_root(Path.cwd())
        if git_root is not None:
            forbidden.add(git_root)
        if target in forbidden:
            raise ValueError(
                "LSIE_CAPTURE_DIR must not point to the current working directory "
                "or repository root"
            )
    else:
        target = (resolve_state_dir().parent / "capture").resolve()

    target.mkdir(parents=True, exist_ok=True)
    return target


def is_dev_machine() -> bool:
    """Return ``True`` if the operator has declared this host a developer machine.

    The marker is the file ``.dev_machine`` placed at the parent of
    :func:`resolve_state_dir` — i.e. ``%LOCALAPPDATA%\\LSIE-MLF\\.dev_machine``
    on Windows or ``$XDG_DATA_HOME/lsie-mlf/.dev_machine`` on POSIX.
    The operator creates the marker once with ``touch`` (or
    ``ni -ItemType File`` on PowerShell) to declare "I accept the
    documented dev-mode constraints" — most notably the
    ``LSIE_DEV_FORCE_CPU_SPEECH=1`` override needed on Pascal hosts.
    The preflight gate fails closed on sub-Turing hardware unless the
    marker is present.

    The marker is per-machine, never per-user-session — it is
    deliberately a filesystem artefact rather than an env variable so
    that opening a fresh shell does not silently re-enable production
    semantics on a dev box.
    """
    return (resolve_state_dir().parent / ".dev_machine").is_file()


def find_executable(name: str, env_override: str | None = None) -> str:
    """Resolve a system tool's full path. ``PATH`` first, then known fallbacks.

    Designed for ``capture_supervisor`` to locate ``scrcpy`` /
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

    Per §5.1.7 the volatile-memory controls require child-process crash
    dumps to be disabled or redirected to scrubbed diagnostics. WER's
    LocalDumps feature, when configured globally, will write a ``.dmp``
    of any crashing process under ``%LOCALAPPDATA%\\CrashDumps\\``. A
    crash mid-segment would therefore freeze raw PCM into a dump file
    outside the governed cleanup path. Adding our own
    ``...\\LocalDumps\\<app_name>`` subkey with ``DumpType=0`` overrides
    any global config and keeps WER from writing the dump.

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


class SecretStoreUnavailableError(RuntimeError):
    """No functional keyring backend exists on this host.

    Raised by the secret-store primitives when ``keyring`` resolves to
    ``keyring.backends.fail.Keyring`` (the well-known sentinel that every
    operation will throw ``NoKeyringError``). Callers — most notably
    ``cloud_sync_worker`` — turn this into a hard health-page failure
    because the OAuth refresh token cannot be persisted across restarts
    without a Credential Manager / DPAPI-backed store.
    """


def set_secret(service: str, key: str, value: str) -> None:
    """Persist ``value`` under ``(service, key)`` in the OS secret store.

    Windows: routed to ``WinVaultKeyring`` which is DPAPI-backed (and
    TPM-backed where available). POSIX: routed to whichever recommended
    backend ``keyring`` resolves at process start (Secret Service /
    KWallet on Linux, macOS Keychain — though macOS is Tier 2 deferred).

    Raises :class:`SecretStoreUnavailableError` if ``keyring`` resolves
    to the fail backend. Other ``keyring.errors.KeyringError`` subclasses
    propagate so the caller sees the real fault rather than a misleading
    ``unavailable`` translation.
    """
    import keyring
    import keyring.errors

    try:
        keyring.set_password(service, key, value)
    except keyring.errors.NoKeyringError as exc:
        raise SecretStoreUnavailableError(str(exc)) from exc


def get_secret(service: str, key: str) -> str | None:
    """Read the secret stored under ``(service, key)``; ``None`` if absent.

    The ``None`` sentinel matches ``keyring.get_password``'s native
    contract for missing entries — calling code typically chains a
    ``token = get_secret(...) or refresh_via_oauth(...)``.

    Raises :class:`SecretStoreUnavailableError` if no keyring backend is
    available. Other backend faults propagate.
    """
    import keyring
    import keyring.errors

    try:
        return keyring.get_password(service, key)
    except keyring.errors.NoKeyringError as exc:
        raise SecretStoreUnavailableError(str(exc)) from exc


def delete_secret(service: str, key: str) -> bool:
    """Remove the secret at ``(service, key)``. Returns ``True`` if removed.

    A missing entry returns ``False`` rather than raising — this makes
    teardown paths idempotent without forcing every caller to wrap the
    call in a try/except. ``keyring.delete_password`` raises
    ``PasswordDeleteError`` (a subclass of ``KeyringError``, NOT of
    ``NoKeyringError``) for both "key not present" and "backend failure"
    on most backends, so we conservatively translate any
    ``PasswordDeleteError`` to ``False``. ``NoKeyringError`` still
    surfaces as :class:`SecretStoreUnavailableError`.
    """
    import keyring
    import keyring.errors

    try:
        keyring.delete_password(service, key)
    except keyring.errors.NoKeyringError as exc:
        raise SecretStoreUnavailableError(str(exc)) from exc
    except keyring.errors.PasswordDeleteError:
        return False
    return True


def create_shortcut(*, target: Path, shortcut: Path, working_dir: Path, description: str) -> bool:
    if sys.platform != "win32":
        return False
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    try:
        import win32com.client
    except ImportError:
        return False
    shell = win32com.client.Dispatch("WScript.Shell")
    link = shell.CreateShortcut(str(shortcut))
    link.TargetPath = str(target)
    link.WorkingDirectory = str(working_dir)
    link.Description = description
    link.Save()
    return True


def cleanup_orphan_ipc_blocks(prefix: str) -> int:
    """Unlink leftover SharedMemory blocks whose name starts with ``prefix``."""
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


def _apply_windows_child_process_policy(popen_kwargs: dict[str, Any]) -> dict[str, Any]:
    if sys.platform != "win32":
        return dict(popen_kwargs)

    kwargs = dict(popen_kwargs)
    kwargs["creationflags"] = (
        int(kwargs.get("creationflags", 0))
        | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    )

    startupinfo = kwargs.get("startupinfo")
    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo is None and startupinfo_factory is not None:
        startupinfo = startupinfo_factory()

    show_flag = int(getattr(subprocess, "STARTF_USESHOWWINDOW", 0))
    if startupinfo is not None and show_flag:
        startupinfo.dwFlags = int(getattr(startupinfo, "dwFlags", 0)) | show_flag
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
        kwargs["startupinfo"] = startupinfo

    return kwargs


def secure_delete_file(
    path: Path,
    *,
    attempts: int = 6,
    retry_delay_s: float = 0.25,
) -> bool:
    """Best-effort secure delete for a raw-media file; returns ``True`` if removed."""
    if not path.exists():
        return False
    if not path.is_file():
        return False

    if sys.platform != "win32":
        shred = shutil.which("shred")
        if shred is not None:
            result = subprocess.run(
                [shred, "-u", "-z", "-n", "3", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return True
            logger.warning("secure delete via shred failed for %s", path)

    last_error: OSError | None = None
    for attempt in range(max(1, attempts)):
        try:
            with path.open("r+b") as file:
                size = file.seek(0, os.SEEK_END)
                file.seek(0)
                chunk = b"\x00" * min(size, 1024 * 1024)
                remaining = size
                while remaining > 0:
                    write_size = min(remaining, len(chunk))
                    file.write(chunk[:write_size])
                    remaining -= write_size
                file.flush()
                os.fsync(file.fileno())
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            last_error = exc
            if attempt < max(1, attempts) - 1:
                time.sleep(retry_delay_s)

    logger.warning("capture file delete failed: %s (%s)", path, last_error)
    return False


class SupervisedProcess:
    """A subprocess whose entire descendant tree is force-killed on close.

    Cross-platform native-subprocess supervision for the desktop capture
    lane. ``capture_supervisor`` spawns
    scrcpy / ADB / FFmpeg through this primitive so a crash
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

        kwargs = _apply_windows_child_process_policy(popen_kwargs)
        proc = subprocess.Popen(args, **kwargs)

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

    @property
    def stdout(self) -> Any:
        return self._proc.stdout

    def wait(self, timeout: float | None = None) -> int:
        return self._proc.wait(timeout=timeout)

    def terminate_root(self, grace_s: float = 3.0) -> bool:
        if self._closed:
            return True
        if self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                self._proc.terminate()
        try:
            self._proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            return False
        return True

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
