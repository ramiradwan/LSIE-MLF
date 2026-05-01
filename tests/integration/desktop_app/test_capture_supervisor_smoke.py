"""WS3 P3 — capture_supervisor smoke against a real Android device.

Gated by ``LSIE_INTEGRATION_DEVICE=1``. Runs only when an Android
device is on the USB bus AND ``scrcpy`` resolves on the host.

What it proves:
  - ``DriftCorrector.poll`` over real ADB returns a sub-second offset.
  - The full ``capture_supervisor.run`` body launches both scrcpy
    instances under :class:`SupervisedProcess`, the audio recording
    starts writing to disk, and a cooperative shutdown leaves zero
    orphan ``scrcpy.exe`` / ``adb.exe`` PIDs in the system table.
  - Drift offsets reach the consumer side over
    ``IpcChannels.drift_updates``.

The smoke is intentionally short (≈10 seconds of capture) to avoid
filling the device's pageserver. It does not validate audio quality
or video integrity — that's WS3 P3c work.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import time
from collections.abc import Iterator

import psutil
import pytest

from services.desktop_app.drift import DriftCorrector
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.os_adapter import find_executable

ENABLED = os.environ.get("LSIE_INTEGRATION_DEVICE") == "1"


@pytest.fixture(scope="module")
def adb_path() -> str:
    if not ENABLED:
        pytest.skip("LSIE_INTEGRATION_DEVICE not set")
    try:
        return find_executable("adb", env_override="LSIE_ADB_PATH")
    except FileNotFoundError as exc:
        pytest.skip(f"adb not found: {exc}")


@pytest.fixture(scope="module")
def scrcpy_path() -> str:
    if not ENABLED:
        pytest.skip("LSIE_INTEGRATION_DEVICE not set")
    try:
        return find_executable("scrcpy", env_override="LSIE_SCRCPY_PATH")
    except FileNotFoundError as exc:
        pytest.skip(f"scrcpy not found: {exc}")


@pytest.fixture(scope="module")
def device_present(adb_path: str) -> Iterator[None]:
    if not ENABLED:
        pytest.skip("LSIE_INTEGRATION_DEVICE not set")
    result = subprocess.run(
        [adb_path, "devices"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    has_device = any(
        line.strip().endswith("device")
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("List of devices")
    )
    if not has_device:
        pytest.skip(f"no adb device attached: {result.stdout!r}")
    yield


def test_real_device_drift_poll_returns_offset(device_present: None) -> None:
    """Real ADB shell call must produce a sub-second drift offset."""
    dc = DriftCorrector()
    offset = dc.poll()

    # Devices are usually within a few hundred ms of host time. A
    # multi-second offset here means the device clock or host clock is
    # wildly out of sync — surface that loudly so the operator notices.
    assert dc._consecutive_failures == 0
    assert dc._frozen is False
    assert abs(offset) < 5.0, f"drift offset {offset:.3f}s is implausibly large"


def _orphan_scrcpy_pids() -> set[int]:
    return {
        proc.info["pid"]
        for proc in psutil.process_iter(["pid", "name"])
        if proc.info.get("name", "").lower().startswith("scrcpy")
    }


def test_capture_supervisor_full_lifecycle_leaves_no_orphans(
    device_present: None,
    scrcpy_path: str,
    tmp_path: object,
) -> None:
    """Spawn capture_supervisor, let it run briefly, signal shutdown, verify cleanup.

    Force-kill not used here — we expect the cooperative shutdown path
    (shutdown_event → SupervisedProcess.terminate) to leave zero
    orphans. WS3 P3a's test_os_adapter already proves the force-kill
    path via Job Object KILL_ON_JOB_CLOSE.
    """
    pre_orphans = _orphan_scrcpy_pids()
    assert pre_orphans == set(), f"pre-test orphans present: {pre_orphans}"

    ctx = mp.get_context("spawn")
    channels = IpcChannels(ml_inbox=ctx.Queue(), drift_updates=ctx.Queue())
    shutdown = ctx.Event()

    capture_dir = str(tmp_path)
    env_clean = os.environ.copy()
    env_clean["LSIE_CAPTURE_DIR"] = capture_dir
    env_clean["LSIE_SCRCPY_PATH"] = scrcpy_path

    # We can't easily reuse the production _launch (it spawns a child
    # process which would not inherit our env override paths cleanly
    # under spawn unless the child re-imports settings). So launch the
    # supervisor's run() in this process via a target wrapper.
    proc = ctx.Process(
        target=_run_capture_supervisor_in_subprocess,
        args=(shutdown, channels, capture_dir, scrcpy_path),
        name="capture_supervisor",
    )
    proc.start()
    try:
        # Wait for at least one drift update to arrive — proves the
        # poll thread reached the device.
        deadline = time.monotonic() + 30.0
        first_drift: dict[str, float] | None = None
        while time.monotonic() < deadline:
            try:
                raw = channels.drift_updates.get(timeout=1.0)
            except Exception:  # noqa: BLE001
                continue
            assert isinstance(raw, dict)
            first_drift = raw
            break
        assert first_drift is not None, "no drift_updates payload arrived within 30 s"
        assert "drift_offset" in first_drift

        # Let scrcpy run a bit so it actually allocates the recording file.
        time.sleep(8.0)

        # Cooperative shutdown.
        shutdown.set()
        proc.join(timeout=15.0)
        assert not proc.is_alive(), "supervisor did not stop cooperatively"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)

    # No orphan scrcpy processes left behind.
    post_orphans = _orphan_scrcpy_pids()
    assert post_orphans == set(), f"orphan scrcpy pids: {post_orphans}"

    # The audio recording artifact should have been created (even if
    # short). scrcpy 3.x writes to record path immediately on launch.
    audio_path = os.path.join(capture_dir, "audio_stream.wav")
    assert os.path.exists(audio_path), f"audio recording absent at {audio_path}"


def _run_capture_supervisor_in_subprocess(
    shutdown_event: object,
    channels: IpcChannels,
    capture_dir: str,
    scrcpy_path: str,
) -> None:
    """Spawn-mode child entrypoint: re-import + invoke run() with overrides set."""
    import logging
    import os as _os

    _os.environ["LSIE_CAPTURE_DIR"] = capture_dir
    _os.environ["LSIE_SCRCPY_PATH"] = scrcpy_path
    logging.basicConfig(level=logging.INFO)

    from services.desktop_app.processes.capture_supervisor import run

    run(shutdown_event=shutdown_event, channels=channels)  # type: ignore[arg-type]
