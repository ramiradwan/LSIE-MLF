"""Capture supervisor process.

Owns the desktop capture lifecycle described by §9.1, §9.3, and §12:

  - ADB device-presence wait (USB loss polls every 2 s for 60 s before
    the capture graph gives up on revival).
  - :class:`services.desktop_app.drift.DriftCorrector` poll loop
    running every ``DRIFT_POLL_INTERVAL`` seconds. Updated offsets are
    pushed onto :attr:`IpcChannels.drift_updates` for
    ``module_c_orchestrator`` to apply via ``correct_timestamp``.
  - Dual-instance scrcpy supervision: audio on port range 27100:27199,
    video on 27200:27299, four-second staggered startup. Each scrcpy is
    launched under
    :class:`services.desktop_app.os_adapter.SupervisedProcess` so a
    crash of this process leaves zero zombies holding the USB device
    open. The audio recording is shielded across scrcpy restarts; the
    video recording is intentionally re-created on each restart so a
    PyAV crash downstream triggers a fresh MKV header.
  - Restart loop: when either scrcpy exits, both are torn down and the
    device-wait + spawn cycle restarts after a short backoff.

It also persists two pieces of recovery state:

  - ``capture_pid_manifest`` — every spawned scrcpy PID is recorded
    against the SQLite store via
    :func:`services.desktop_app.state.recovery.record_capture_pid`.
    The next ui_api_shell startup recovery sweep reads this manifest
    and reaps any PID still alive from an ungraceful prior exit
    (Job Objects cover Windows; this is the POSIX gap).
  - :class:`HeartbeatRecorder` — 1 Hz process-heartbeat row.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from services.desktop_app.drift import (
    DRIFT_POLL_INTERVAL,
    DriftCorrector,
)
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.os_adapter import (
    SupervisedProcess,
    find_executable,
    resolve_capture_dir,
    resolve_state_dir,
)
from services.desktop_app.privacy.zeroize import cleanup_capture_files
from services.desktop_app.state.heartbeats import HeartbeatRecorder
from services.desktop_app.state.recovery import (
    forget_capture_pid,
    record_capture_pid,
)

logger = logging.getLogger(__name__)

# §12 hardware-device-loss cadence
DEVICE_POLL_INTERVAL_S: float = 2.0
DEVICE_POLL_TIMEOUT_S: float = 60.0

# Dual-instance scrcpy port ranges + stagger
AUDIO_SCRCPY_PORT_RANGE: str = "27100:27199"
VIDEO_SCRCPY_PORT_RANGE: str = "27200:27299"
SCRCPY_STAGGER_S: float = 4.0

# Restart backoff after one of the scrcpy instances dies.
RESTART_BACKOFF_S: float = 2.0

SQLITE_FILENAME = "desktop.sqlite"


@dataclass(frozen=True)
class CaptureLayout:
    """File-system locations the supervisor writes to.

    The desktop runtime records capture output into regular files under
    the governed capture directory rather than through the legacy IPC
    capture path. ``module_c_orchestrator`` consumes both files via
    ``services.worker.pipeline.orchestrator.AudioResampler`` and
    ``VideoCapture``, both of which already tolerate file-style reads
    via the existing ``services.worker.pipeline.replay_capture`` shim.
    """

    capture_dir: Path
    audio_path: Path
    video_path: Path


def _resolve_capture_layout() -> CaptureLayout:
    base = resolve_capture_dir()
    return CaptureLayout(
        capture_dir=base,
        audio_path=base / "audio_stream.wav",
        video_path=base / "video_stream.mkv",
    )


def _wait_for_device(
    deadline_s: float = DEVICE_POLL_TIMEOUT_S,
    shutdown_event: mpsync.Event | None = None,
) -> bool:
    """Block until ADB reports at least one device, or ``deadline_s`` elapses.

    Returns ``True`` when a device is observed, ``False`` on timeout or
    cooperative shutdown. Logs each missing-device tick so the operator
    console can surface the wait state in the §12 health rollup.
    """
    import subprocess as _subprocess

    adb = find_executable("adb", env_override="LSIE_ADB_PATH")
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if shutdown_event is not None and shutdown_event.is_set():
            return False
        try:
            result = _subprocess.run(
                [adb, "devices"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith("List of devices") and line.endswith("device"):
                    logger.info("adb sees device: %s", line)
                    return True
        except (_subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("adb devices probe failed: %s", exc)
        wait_s = min(DEVICE_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        if wait_s <= 0:
            break
        if shutdown_event is not None:
            if shutdown_event.wait(timeout=wait_s):
                return False
        else:
            time.sleep(wait_s)
    logger.error("adb device wait timed out after %ds", int(deadline_s))
    return False


def _build_audio_scrcpy_args(scrcpy: str, layout: CaptureLayout) -> list[str]:
    """Audio scrcpy: raw 48 kHz WAV record, no video."""
    return [
        scrcpy,
        "--no-video",
        "--no-playback",
        "--no-window",
        "--audio-codec=raw",
        "--audio-buffer=30",
        "--audio-dup",
        f"--record={layout.audio_path}",
        "--record-format=wav",
        f"--port={AUDIO_SCRCPY_PORT_RANGE}",
    ]


def _build_video_scrcpy_args(scrcpy: str, layout: CaptureLayout) -> list[str]:
    """Video scrcpy: H.264 30 fps MKV record, no audio."""
    return [
        scrcpy,
        "--no-audio",
        "--no-playback",
        "--no-window",
        "--video-codec=h264",
        "--max-fps=30",
        f"--record={layout.video_path}",
        "--record-format=mkv",
        f"--port={VIDEO_SCRCPY_PORT_RANGE}",
    ]


class _DriftPollThread:
    """Background thread that polls ADB drift and ships offsets over IPC.

    Splits the §4.C.1 poll cadence from the supervisor's main loop so
    a slow ADB call cannot stall scrcpy restart logic, and so the
    supervisor process can drive both surfaces from a single import.
    Sends ``{"drift_offset": float, "ts_monotonic": float}`` payloads
    to ``module_c_orchestrator``.
    """

    def __init__(
        self,
        corrector: DriftCorrector,
        channels: IpcChannels,
        shutdown_event: mpsync.Event,
        interval_s: float = float(DRIFT_POLL_INTERVAL),
    ) -> None:
        self._corrector = corrector
        self._channels = channels
        self._shutdown_event = shutdown_event
        self._interval_s = interval_s
        self._thread = threading.Thread(
            target=self._run,
            name="capture-supervisor-drift-poll",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._shutdown_event.is_set():
            offset = self._corrector.poll()
            try:
                self._channels.drift_updates.put_nowait(
                    {"drift_offset": float(offset), "ts_monotonic": time.monotonic()}
                )
            except Exception:  # noqa: BLE001
                logger.debug("drift_updates put failed", exc_info=True)
            self._shutdown_event.wait(timeout=self._interval_s)


def _spawn_scrcpy_pair(
    scrcpy_path: str,
    layout: CaptureLayout,
    db_path: Path,
    shutdown_event: mpsync.Event,
) -> tuple[SupervisedProcess, SupervisedProcess] | None:
    """Spawn audio scrcpy first, wait the stagger, then spawn video.

    Each PID is registered in ``capture_pid_manifest`` immediately
    after spawn so the next boot's recovery sweep can reap the child
    if this supervisor exits ungracefully.
    """
    audio = SupervisedProcess(_build_audio_scrcpy_args(scrcpy_path, layout))
    record_capture_pid(db_path, audio.pid, process_kind="scrcpy")
    logger.info("audio scrcpy launched pid=%s", audio.pid)
    if shutdown_event.wait(timeout=SCRCPY_STAGGER_S):
        _kill_pair(audio, None, db_path)
        return None
    video = SupervisedProcess(_build_video_scrcpy_args(scrcpy_path, layout))
    record_capture_pid(db_path, video.pid, process_kind="scrcpy")
    logger.info("video scrcpy launched pid=%s", video.pid)
    return audio, video


def _kill_pair(
    audio: SupervisedProcess | None,
    video: SupervisedProcess | None,
    db_path: Path,
) -> None:
    for proc in (audio, video):
        if proc is None:
            continue
        try:
            proc.terminate(grace_s=3.0)
        except Exception:  # noqa: BLE001
            logger.debug("terminate failed for pid=%s", proc.pid, exc_info=True)
        # Always forget — even on terminate failure the manifest entry
        # is stale by definition once we've left the spawn pair.
        try:
            forget_capture_pid(db_path, proc.pid)
        except Exception:  # noqa: BLE001
            logger.debug("manifest cleanup failed for pid=%s", proc.pid, exc_info=True)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("capture_supervisor started")

    layout = _resolve_capture_layout()
    deleted, retained = cleanup_capture_files(layout.capture_dir)
    if deleted or retained:
        logger.info(
            "capture startup cleanup: deleted=%s retained=%s",
            [str(path) for path in deleted],
            [str(path) for path in retained],
        )
    logger.info("capture layout: %s", layout.capture_dir)

    db_path = resolve_state_dir() / SQLITE_FILENAME
    heartbeat = HeartbeatRecorder(db_path, "capture_supervisor")
    heartbeat.start()

    corrector = DriftCorrector()
    drift_thread = _DriftPollThread(corrector, channels, shutdown_event)
    drift_thread.start()

    try:
        scrcpy_path = find_executable("scrcpy", env_override="LSIE_SCRCPY_PATH")
    except FileNotFoundError as exc:
        logger.error("capture_supervisor cannot run: %s", exc)
        # Without scrcpy the supervisor still ships drift updates so
        # the orchestrator's correct_timestamp keeps working with
        # whatever ADB-derived offset is available; bail out of the
        # scrcpy restart loop only.
        shutdown_event.wait()
        drift_thread.join(timeout=5.0)
        heartbeat.stop()
        logger.info("capture_supervisor stopped (no scrcpy)")
        return

    audio_proc: SupervisedProcess | None = None
    video_proc: SupervisedProcess | None = None
    try:
        while not shutdown_event.is_set():
            if not _wait_for_device(shutdown_event=shutdown_event):
                if shutdown_event.is_set():
                    break
                logger.warning("device wait timed out; backing off %.1fs", RESTART_BACKOFF_S)
                if shutdown_event.wait(timeout=RESTART_BACKOFF_S):
                    break
                continue

            pair = _spawn_scrcpy_pair(scrcpy_path, layout, db_path, shutdown_event)
            if pair is None:
                break
            audio_proc, video_proc = pair

            # §12 worker-crash behavior: whichever scrcpy exits first
            # triggers a full pair teardown before the device-wait loop
            # restarts.
            while not shutdown_event.is_set():
                if not audio_proc.is_alive() or not video_proc.is_alive():
                    logger.info(
                        "pipeline break: audio_alive=%s video_alive=%s",
                        audio_proc.is_alive(),
                        video_proc.is_alive(),
                    )
                    break
                if shutdown_event.wait(timeout=0.5):
                    break

            _kill_pair(audio_proc, video_proc, db_path)
            audio_proc = None
            video_proc = None

            if shutdown_event.is_set():
                break
            shutdown_event.wait(timeout=RESTART_BACKOFF_S)
    finally:
        _kill_pair(audio_proc, video_proc, db_path)
        deleted, retained = cleanup_capture_files(layout.capture_dir)
        if deleted or retained:
            logger.info(
                "capture shutdown cleanup: deleted=%s retained=%s",
                [str(path) for path in deleted],
                [str(path) for path in retained],
            )
        drift_thread.join(timeout=5.0)
        heartbeat.stop()
        logger.info("capture_supervisor stopped")
