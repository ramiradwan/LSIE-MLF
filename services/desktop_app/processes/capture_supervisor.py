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
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from services.desktop_app.drift import (
    DRIFT_POLL_INTERVAL,
    DriftCorrector,
)
from services.desktop_app.ipc import IpcChannels
from services.desktop_app.os_adapter import (
    DESKTOP_CAPTURE_TOOL_SPECS,
    SupervisedProcess,
    missing_external_tools_detail,
    missing_external_tools_hint,
    resolve_capture_dir,
    resolve_external_tools,
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
_WAV_HEADER_BYTES = 44
_AUDIO_SILENCE_SAMPLE_BYTES = 48_000 * 2 * 3


@dataclass(frozen=True)
class _AdbDevice:
    serial: str
    model: str | None
    active_app: str | None


class _AliveProcess(Protocol):
    def is_alive(self) -> bool: ...


@dataclass(frozen=True)
class CaptureStatusRecord:
    status_key: str
    state: str
    label: str
    detail: str | None
    operator_action_hint: str | None


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


def _run_adb(adb: str, args: list[str], *, timeout_s: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            [adb, *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("adb %s probe failed: %s", " ".join(args), exc)
        return None
    if result.returncode != 0:
        logger.debug("adb %s returned %s: %s", " ".join(args), result.returncode, result.stderr)
        return None
    return result.stdout.strip()


def _connected_adb_serials(adb: str) -> list[str]:
    output = _run_adb(adb, ["devices"], timeout_s=5.0)
    if output is None:
        return []
    serials: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    return serials


def _read_adb_device(adb: str) -> _AdbDevice | None:
    serials = _connected_adb_serials(adb)
    if not serials:
        return None
    serial = serials[0]
    model = _run_adb(adb, ["-s", serial, "shell", "getprop", "ro.product.model"], timeout_s=2.0)
    active_app = _run_adb(
        adb,
        ["-s", serial, "shell", "dumpsys", "window", "windows"],
        timeout_s=2.0,
    )
    return _AdbDevice(
        serial=serial,
        model=model or None,
        active_app=_parse_active_app(active_app),
    )


def _parse_active_app(output: str | None) -> str | None:
    if not output:
        return None
    for line in output.splitlines():
        if "mCurrentFocus" not in line and "mFocusedApp" not in line:
            continue
        marker = " u0 "
        if marker in line:
            suffix = line.split(marker, 1)[1]
            return suffix.split("/", 1)[0].strip(" }") or None
        for token in line.replace("}", " ").split():
            if "/" in token and "." in token:
                return token.split("/", 1)[0]
    return None


def _format_device_detail(device: _AdbDevice) -> str:
    name = device.model or device.serial
    if device.active_app is None:
        return f"Connected device: {name} ({device.serial})"
    return f"Connected device: {name} ({device.serial}) · Active app: {device.active_app}"


def _capture_file_detail(path: Path, *, label: str) -> str:
    if not path.exists():
        return f"{label} stream file pending: {path.name}"
    size = path.stat().st_size
    return f"{label} stream recording: {path.name} · {size:,} bytes"


def _audio_capture_is_silent(path: Path) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    if size <= _WAV_HEADER_BYTES:
        return False
    sample_size = min(_AUDIO_SILENCE_SAMPLE_BYTES, size - _WAV_HEADER_BYTES)
    with path.open("rb") as audio_file:
        audio_file.seek(size - sample_size)
        sample = audio_file.read(sample_size)
    return bool(sample) and all(byte == 0 for byte in sample)


def _audio_capture_status(
    audio_alive: bool,
    layout: CaptureLayout | None,
) -> CaptureStatusRecord:
    if not audio_alive or layout is None:
        return CaptureStatusRecord(
            status_key="audio_capture",
            state="recovering",
            label="Audio Capture",
            detail="Audio scrcpy recorder is starting or restarting",
            operator_action_hint="Keep the target app producing audio",
        )
    if _audio_capture_is_silent(layout.audio_path):
        return CaptureStatusRecord(
            status_key="audio_capture",
            state="recovering",
            label="Audio Capture",
            detail="Audio stream is recording but the captured signal is silent",
            operator_action_hint=(
                "Make sure the phone is playing audible media; "
                "some apps block Android playback capture"
            ),
        )
    return CaptureStatusRecord(
        status_key="audio_capture",
        state="ok",
        label="Audio Capture",
        detail=_capture_file_detail(layout.audio_path, label="Audio"),
        operator_action_hint=None,
    )


def _write_capture_statuses(
    db_path: Path,
    device: _AdbDevice | None,
    audio_proc: _AliveProcess | None,
    video_proc: _AliveProcess | None,
    layout: CaptureLayout | None,
) -> None:
    if device is None:
        records = [
            CaptureStatusRecord(
                status_key="adb",
                state="unknown",
                label="Android Device Bridge",
                detail="No Android device connected",
                operator_action_hint="Connect Android device via USB and allow debugging",
            ),
            CaptureStatusRecord(
                status_key="audio_capture",
                state="unknown",
                label="Audio Capture",
                detail="Audio stream is not recording because no phone is connected",
                operator_action_hint="Connect the phone, then keep the target app producing audio",
            ),
            CaptureStatusRecord(
                status_key="video_capture",
                state="unknown",
                label="Video Capture",
                detail="Video stream is not recording because no phone is connected",
                operator_action_hint=(
                    "Connect the phone and open a stream or video with a visible face"
                ),
            ),
        ]
    else:
        audio_alive = audio_proc is not None and audio_proc.is_alive()
        video_alive = video_proc is not None and video_proc.is_alive()
        records = [
            CaptureStatusRecord(
                status_key="adb",
                state="ok",
                label="Android Device Bridge",
                detail=_format_device_detail(device),
                operator_action_hint=None,
            ),
            _audio_capture_status(audio_alive, layout),
            CaptureStatusRecord(
                status_key="video_capture",
                state="ok" if video_alive else "recovering",
                label="Video Capture",
                detail=(
                    _capture_file_detail(layout.video_path, label="Replay video")
                    if layout is not None and video_alive
                    else "Replay video scrcpy recorder is starting or restarting"
                ),
                operator_action_hint=(
                    None if video_alive else "Open a stream or video with a visible face"
                ),
            ),
        ]
    _upsert_capture_statuses(db_path, records)


def _upsert_capture_statuses(db_path: Path, records: list[CaptureStatusRecord]) -> None:
    from datetime import UTC, datetime

    updated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.executemany(
            "INSERT INTO capture_status "
            "(status_key, state, label, detail, operator_action_hint, updated_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(status_key) DO UPDATE SET "
            "state = excluded.state, "
            "label = excluded.label, "
            "detail = excluded.detail, "
            "operator_action_hint = excluded.operator_action_hint, "
            "updated_at_utc = excluded.updated_at_utc",
            [
                (
                    record.status_key,
                    record.state,
                    record.label,
                    record.detail,
                    record.operator_action_hint,
                    updated_at,
                )
                for record in records
            ],
        )
    finally:
        conn.close()


def _wait_for_device(
    adb: str,
    db_path: Path,
    deadline_s: float = DEVICE_POLL_TIMEOUT_S,
    shutdown_event: mpsync.Event | None = None,
) -> _AdbDevice | None:
    """Block until ADB reports at least one device, or ``deadline_s`` elapses."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if shutdown_event is not None and shutdown_event.is_set():
            return None
        device = _read_adb_device(adb)
        _write_capture_statuses(db_path, device, None, None, None)
        if device is not None:
            logger.info("adb sees device: %s", device.serial)
            return device
        wait_s = min(DEVICE_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        if wait_s <= 0:
            break
        if shutdown_event is not None:
            if shutdown_event.wait(timeout=wait_s):
                return None
        else:
            time.sleep(wait_s)
    logger.error("adb device wait timed out after %ds", int(deadline_s))
    _write_capture_statuses(db_path, None, None, None, None)
    return None


def _build_audio_scrcpy_args(scrcpy: str, layout: CaptureLayout) -> list[str]:
    """Audio scrcpy: raw 48 kHz WAV record, no video."""
    return [
        scrcpy,
        "--no-video",
        "--no-playback",
        "--no-window",
        "--audio-codec=raw",
        "--audio-source=playback",
        "--audio-buffer=30",
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
    record_capture_pid(
        db_path,
        audio.pid,
        process_kind="scrcpy",
        parent_process="capture_supervisor_audio",
    )
    logger.info("audio scrcpy launched pid=%s", audio.pid)
    if shutdown_event.wait(timeout=SCRCPY_STAGGER_S):
        _kill_pair(audio, None, db_path)
        return None
    video = SupervisedProcess(_build_video_scrcpy_args(scrcpy_path, layout))
    record_capture_pid(
        db_path,
        video.pid,
        process_kind="scrcpy",
        parent_process="capture_supervisor_video",
    )
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
            if not proc.terminate_root(grace_s=3.0):
                proc.terminate(grace_s=3.0)
        except Exception:  # noqa: BLE001
            logger.debug("terminate failed for pid=%s", proc.pid, exc_info=True)
        try:
            if proc.is_alive():
                logger.warning("capture child pid=%s survived termination", proc.pid)
                continue
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

    resolved_tools, missing_tools = resolve_external_tools(DESKTOP_CAPTURE_TOOL_SPECS)
    if missing_tools:
        detail = missing_external_tools_detail(missing_tools)
        hint = missing_external_tools_hint(missing_tools)
        logger.error("capture_supervisor cannot run: %s %s", detail, hint)
        _upsert_capture_statuses(
            db_path,
            [
                CaptureStatusRecord(
                    status_key="adb",
                    state="unknown",
                    label="Android Device Bridge",
                    detail=detail,
                    operator_action_hint=hint,
                ),
                CaptureStatusRecord(
                    status_key="audio_capture",
                    state="unknown",
                    label="Audio Capture",
                    detail=detail,
                    operator_action_hint=hint,
                ),
                CaptureStatusRecord(
                    status_key="video_capture",
                    state="unknown",
                    label="Video Capture",
                    detail=detail,
                    operator_action_hint=hint,
                ),
            ],
        )
        # Without capture tooling the supervisor still ships drift updates so
        # the orchestrator's correct_timestamp keeps working with
        # whatever ADB-derived offset is available; bail out of the
        # scrcpy restart loop only.
        shutdown_event.wait()
        drift_thread.join(timeout=5.0)
        heartbeat.stop()
        logger.info("capture_supervisor stopped (missing capture tooling)")
        return
    scrcpy_path = resolved_tools["scrcpy"]
    adb_path = resolved_tools["adb"]

    audio_proc: SupervisedProcess | None = None
    video_proc: SupervisedProcess | None = None
    try:
        while not shutdown_event.is_set():
            device = _wait_for_device(adb_path, db_path, shutdown_event=shutdown_event)
            if device is None:
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
            _write_capture_statuses(db_path, device, audio_proc, video_proc, layout)

            # §12 worker-crash behavior: whichever scrcpy exits first
            # triggers a full pair teardown before the device-wait loop
            # restarts.
            while not shutdown_event.is_set():
                device = _read_adb_device(adb_path)
                _write_capture_statuses(db_path, device, audio_proc, video_proc, layout)
                if device is None or not audio_proc.is_alive() or not video_proc.is_alive():
                    logger.info(
                        "pipeline break: audio_alive=%s video_alive=%s",
                        audio_proc.is_alive(),
                        video_proc.is_alive(),
                    )
                    if device is None:
                        logger.info("pipeline break: device disconnected")
                    break
                if shutdown_event.wait(timeout=0.5):
                    break

            _kill_pair(audio_proc, video_proc, db_path)
            _write_capture_statuses(db_path, device, None, None, None)
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
