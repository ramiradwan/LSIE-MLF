"""SupervisedProcess descendant-tree cleanup tests.

Spawns synthetic Python subprocesses to validate that
``SupervisedProcess.terminate`` reliably kills the child and any
grandchildren on both Windows and POSIX. That cleanup contract lets
``capture_supervisor`` launch scrcpy, ADB, and FFmpeg without leaking
USB-holding zombies on a crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast
from unittest.mock import patch

import psutil
import pytest

from services.desktop_app import os_adapter
from services.desktop_app.os_adapter import (
    SupervisedProcess,
    _apply_windows_child_process_policy,
    resolve_capture_dir,
)


def _grandchild_spawning_script(pid_file: Path) -> str:
    """Return a Python source snippet that prints its pid + a child pid.

    Layout: parent prints its own pid, spawns a long-running grandchild
    via ``subprocess.Popen``, prints the grandchild pid, then sleeps.
    Each pid is written on its own line to ``pid_file`` so the test
    can read them after the subprocess starts.
    """
    sleeper = sys.executable.replace("\\", "\\\\")
    pid_path = str(pid_file).replace("\\", "\\\\")
    return textwrap.dedent(
        f"""
        import os
        import subprocess
        import sys
        import time

        pid_file = r"{pid_path}"
        sleeper = r"{sleeper}"
        grandchild = subprocess.Popen(
            [sleeper, "-c", "import time; time.sleep(120)"]
        )
        with open(pid_file, "w") as f:
            f.write(f"{{os.getpid()}}\\n")
            f.write(f"{{grandchild.pid}}\\n")
            f.flush()
        time.sleep(120)
        """
    ).strip()


def _read_pids(pid_file: Path, timeout_s: float = 10.0) -> tuple[int, int]:
    """Wait for the subprocess to write both pids, then return them."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pid_file.exists():
            content = pid_file.read_text().strip().splitlines()
            if len(content) == 2:
                return int(content[0]), int(content[1])
        time.sleep(0.1)
    raise AssertionError(f"pids never written to {pid_file}")


def _wait_for_dead(pid: int, timeout_s: float = 5.0) -> bool:
    """Return True if ``pid`` becomes non-existent within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        try:
            proc = psutil.Process(pid)
            if proc.status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


def test_is_dev_machine_checks_marker_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("LSIE_STATE_DIR", str(state_dir))

    assert os_adapter.is_dev_machine() is False
    (tmp_path / ".dev_machine").write_text("", encoding="utf-8")
    assert os_adapter.is_dev_machine() is True


def test_resolve_capture_dir_defaults_beside_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("LSIE_STATE_DIR", str(state_dir))
    monkeypatch.delenv("LSIE_CAPTURE_DIR", raising=False)

    capture_dir = resolve_capture_dir()

    assert capture_dir == tmp_path / "capture"
    assert capture_dir.is_dir()


def test_resolve_capture_dir_rejects_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("LSIE_CAPTURE_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="current working directory or repository root"):
        resolve_capture_dir()


def test_create_shortcut_is_noop_on_posix(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "platform", "linux")
        created = os_adapter.create_shortcut(
            target=tmp_path / "launcher",
            shortcut=tmp_path / "LSIE-MLF.lnk",
            working_dir=tmp_path,
            description="Launch LSIE-MLF",
        )

    assert created is False
    assert not (tmp_path / "LSIE-MLF.lnk").exists()


def test_secure_delete_file_zeroes_before_unlink(tmp_path: Path) -> None:
    target = tmp_path / "video_stream.mkv"
    target.write_bytes(b"secret raw media")
    writes: list[bytes] = []
    original_open = Path.open
    original_unlink = Path.unlink

    def recording_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> BinaryIO:
        file_obj = cast(
            BinaryIO,
            original_open(path, mode, buffering, encoding, errors, newline),
        )
        original_write = cast(Callable[[bytes], int], file_obj.write)

        def recording_write(payload: bytes) -> int:
            writes.append(bytes(payload))
            return original_write(payload)

        file_obj.write = recording_write  # type: ignore[assignment, method-assign]
        return file_obj

    with (
        patch.object(Path, "open", recording_open),
        patch.object(Path, "unlink", lambda path: original_unlink(path)),
    ):
        assert os_adapter.secure_delete_file(target, attempts=1) is True

    assert writes
    assert all(set(chunk) <= {0} for chunk in writes)
    assert not target.exists()


def test_secure_delete_file_retries_locked_unlink(tmp_path: Path) -> None:
    target = tmp_path / "video_stream.mkv"
    target.write_bytes(b"video")
    calls = 0
    original_unlink = Path.unlink

    def flaky_unlink(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("locked")
        original_unlink(path)

    with (
        patch.object(Path, "unlink", flaky_unlink),
        patch("services.desktop_app.os_adapter.time.sleep") as sleep,
    ):
        assert os_adapter.secure_delete_file(target, attempts=2, retry_delay_s=0.01) is True

    sleep.assert_called_once_with(0.01)
    assert calls == 2
    assert not target.exists()


def test_apply_windows_child_process_policy_is_noop_on_posix() -> None:
    original = {"creationflags": 123}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "platform", "linux")
        result = _apply_windows_child_process_policy(original)

    assert result == original
    assert result is not original


def test_apply_windows_child_process_policy_hides_windows_children() -> None:
    startupinfo = SimpleNamespace(dwFlags=0, wShowWindow=99)
    fake_subprocess = SimpleNamespace(
        CREATE_NEW_PROCESS_GROUP=0x200,
        CREATE_NO_WINDOW=0x08000000,
        STARTF_USESHOWWINDOW=0x1,
        SW_HIDE=0,
        STARTUPINFO=lambda: startupinfo,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(os_adapter, "subprocess", fake_subprocess)
        result = _apply_windows_child_process_policy({})

    assert result["creationflags"] & fake_subprocess.CREATE_NEW_PROCESS_GROUP
    assert result["creationflags"] & fake_subprocess.CREATE_NO_WINDOW
    assert result["startupinfo"] is startupinfo
    assert startupinfo.dwFlags & fake_subprocess.STARTF_USESHOWWINDOW
    assert startupinfo.wShowWindow == fake_subprocess.SW_HIDE


def test_terminate_root_waits_for_direct_child_without_tree_kill(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.txt"
    sp = SupervisedProcess(
        [sys.executable, "-c", _grandchild_spawning_script(pid_file)],
    )
    try:
        child_pid, _grandchild_pid = _read_pids(pid_file)
        assert sp.is_alive()

        assert sp.terminate_root(grace_s=5.0) is True
        assert _wait_for_dead(child_pid), f"child pid={child_pid} survived terminate_root()"
        assert _wait_for_dead(sp.pid), f"launcher pid={sp.pid} survived terminate_root()"
    finally:
        sp.close()


def test_terminate_kills_child(tmp_path: Path) -> None:
    """A long-running supervised child must die on terminate().

    Note: on Windows, ``.venv\\Scripts\\python.exe`` is a launcher stub
    that re-execs the real interpreter, so ``sp.pid`` (the launcher)
    differs from ``os.getpid()`` reported by the script (the real
    interpreter). The supervised tree includes both processes and both
    must die — that is exactly what the Job Object guarantees.
    """
    pid_file = tmp_path / "pids.txt"
    sp = SupervisedProcess(
        [sys.executable, "-c", _grandchild_spawning_script(pid_file)],
    )
    try:
        child_pid, _grandchild_pid = _read_pids(pid_file)
        assert sp.is_alive()

        sp.terminate(grace_s=5.0)
        assert _wait_for_dead(child_pid), f"child pid={child_pid} survived terminate()"
        assert _wait_for_dead(sp.pid), f"launcher pid={sp.pid} survived terminate()"
    finally:
        sp.close()


def test_terminate_kills_grandchild(tmp_path: Path) -> None:
    """The whole descendant tree dies on terminate(), not just the direct child."""
    pid_file = tmp_path / "pids.txt"
    sp = SupervisedProcess(
        [sys.executable, "-c", _grandchild_spawning_script(pid_file)],
    )
    try:
        child_pid, grandchild_pid = _read_pids(pid_file)
        assert psutil.pid_exists(grandchild_pid), "grandchild never started"

        sp.terminate(grace_s=5.0)
        assert _wait_for_dead(child_pid), f"child pid={child_pid} survived terminate()"
        assert _wait_for_dead(grandchild_pid), (
            f"grandchild pid={grandchild_pid} survived terminate() — "
            "Job Object / process group did not propagate the signal"
        )
    finally:
        sp.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Multiple close() calls must not raise even after the child has exited."""
    sp = SupervisedProcess([sys.executable, "-c", "import sys; sys.exit(0)"])
    sp.wait(timeout=5.0)
    sp.close()
    sp.close()


def test_context_manager_terminates_on_exit(tmp_path: Path) -> None:
    """Using SupervisedProcess as a context manager closes on __exit__."""
    pid_file = tmp_path / "pids.txt"
    with SupervisedProcess(
        [sys.executable, "-c", _grandchild_spawning_script(pid_file)],
    ) as sp:
        child_pid, grandchild_pid = _read_pids(pid_file)
        assert sp.is_alive()

    assert _wait_for_dead(child_pid)
    assert _wait_for_dead(grandchild_pid)


def test_natural_exit_does_not_raise() -> None:
    """A child that exits on its own must surface a clean exit code."""
    sp = SupervisedProcess([sys.executable, "-c", "import sys; sys.exit(42)"])
    rc = sp.wait(timeout=5.0)
    assert rc == 42
    sp.close()


def test_pid_and_is_alive_track_lifecycle() -> None:
    sp = SupervisedProcess([sys.executable, "-c", "import time; time.sleep(0.5)"])
    try:
        assert isinstance(sp.pid, int) and sp.pid > 0
        assert sp.is_alive() is True
        sp.wait(timeout=5.0)
        assert sp.is_alive() is False
        assert sp.poll() == 0
    finally:
        sp.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 Job Object behaviour")
def test_windows_job_object_handle_release_kills_orphans(tmp_path: Path) -> None:
    """On Windows, dropping the Job Object handle without explicit terminate
    should still kill the child via KILL_ON_JOB_CLOSE.

    We simulate this by force-deleting the SupervisedProcess instance
    (the GC closes the job handle, which fires KILL_ON_JOB_CLOSE).
    """
    pid_file = tmp_path / "pids.txt"
    sp = SupervisedProcess(
        [sys.executable, "-c", _grandchild_spawning_script(pid_file)],
    )
    child_pid, grandchild_pid = _read_pids(pid_file)
    assert psutil.pid_exists(child_pid)

    # Explicit handle close (mirrors what __del__ + GC would do, deterministically).
    import win32api

    win32api.CloseHandle(sp._job)
    sp._job = None

    assert _wait_for_dead(child_pid, timeout_s=5.0), "child survived Job Object handle close"
    assert _wait_for_dead(grandchild_pid, timeout_s=5.0), (
        "grandchild survived Job Object handle close"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group behaviour")
def test_posix_child_runs_in_new_session() -> None:
    """POSIX path: child must be the leader of its own session."""
    sp = SupervisedProcess(
        [sys.executable, "-c", "import os, time; print(os.getpgid(0)); time.sleep(60)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # Read the child's pgid; it must equal the child's pid (group leader).
        line = sp._proc.stdout.readline().strip()  # type: ignore[union-attr]
        assert int(line) == sp.pid
        assert os.getpgid(sp.pid) == sp.pid  # type: ignore[attr-defined]
    finally:
        sp.terminate()
