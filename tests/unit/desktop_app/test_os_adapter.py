"""WS3 P3a — SupervisedProcess descendant-tree cleanup tests.

Spawns synthetic Python subprocesses to validate that
``SupervisedProcess.terminate`` reliably kills the child and any
grandchildren on both Windows (Win32 Job Object) and POSIX (process
group). The cleanup contract is the load-bearing guarantee that lets
``capture_supervisor`` (WS3 P3b) launch scrcpy / ADB / FFmpeg without
leaking USB-holding zombies on a crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from services.desktop_app import os_adapter
from services.desktop_app.os_adapter import SupervisedProcess


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
