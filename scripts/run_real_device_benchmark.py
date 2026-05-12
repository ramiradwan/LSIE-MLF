#!/usr/bin/env python3
"""Real-device end-to-end performance benchmark for the v4 desktop runtime.

Spawns ``python -m services.desktop_app --operator-api`` against a connected
Android device, drives it through the loopback FastAPI surface that backs the
Operator Console, fires N stimuli through the same path the GUI uses, and
records the wall-clock latency until each encounter row becomes readable.

Distinct from ``scripts/run_fixture_benchmark.py`` (the deterministic fixture
benchmark): this script needs real ADB, scrcpy, ffmpeg, and a phone with the
target app foregrounded.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, cast

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.schemas.operator_console import (  # noqa: E402
    EncounterSummary,
    SessionCreateRequest,
    SessionEndRequest,
    SessionSummary,
    StimulusRequest,
)
from services.desktop_app import os_adapter  # noqa: E402
from services.desktop_app.startup_timing import format_startup_milestone  # noqa: E402
from services.desktop_launcher import health_check  # noqa: E402
from services.operator_console.api_client import ApiClient, ApiError  # noqa: E402

logger = logging.getLogger("real_device_benchmark")

DEFAULT_BASELINE_PATH: Path = REPO_ROOT / "docs" / "artifacts" / "real_device_baseline.md"
SCENARIO_LABEL: str = "v4-real-device:@"
DEFAULT_STIMULI: int = 3
DEFAULT_API_PORT: int = 8765
DEFAULT_STREAM_URL: str = "tiktok://benchmark.local/real-device"
DEFAULT_EXPERIMENT_ID: str = "greeting_line_v1"
TIKTOK_PACKAGE: str = "com.zhiliaoapp.musically"

API_READY_TIMEOUT_S: float = 60.0
CAPTURE_READY_TIMEOUT_S: float = 120.0
SESSION_LIFECYCLE_TIMEOUT_S: float = 30.0
ENCOUNTER_TIMEOUT_S: float = 90.0
# Cooperative ProcessGraph teardown joins six children with a 15s per-child
# budget (process_graph.COOPERATIVE_SHUTDOWN_TIMEOUT_S). Worst case a clean
# shutdown needs ~90s; we use 120s so the fallback TerminateProcess path
# never runs. That fallback is destructive on Windows: the Job Object reaps
# scrcpy/ADB/FFmpeg as a tree, which pauses TikTok playback per the
# real-device-desktop-verification skill's known finding.
SHUTDOWN_GRACE_S: float = 120.0
POLL_INTERVAL_S: float = 0.5
STIMULUS_INTERVAL_S: float = 12.0
MEDIA_WATCHER_INTERVAL_S: float = 3.0

BASELINE_COLUMNS: tuple[str, ...] = (
    "Date",
    "Commit SHA",
    "Scenario",
    "Stimuli",
    "API up (s)",
    "Capture ready (s)",
    "Stimulus->encounter p50 (s)",
    "Stimulus->encounter p95 (s)",
    "Segment age @ visible p50 (ms)",
    "Segment age @ visible p95 (ms)",
    "Notes",
)


@dataclass(frozen=True)
class StimulusMeasurement:
    client_action_id: uuid.UUID
    submitted_at: datetime
    encounter_visible_at: datetime
    stimulus_time_utc: datetime | None
    segment_timestamp_utc: datetime
    end_to_end_s: float
    segment_to_visible_ms: float


@dataclass(frozen=True)
class MediaState:
    foregrounded: bool
    audio_started: bool

    @property
    def playing(self) -> bool:
        return self.foregrounded and self.audio_started

    def reason(self) -> str:
        if self.playing:
            return "playing"
        parts: list[str] = []
        if not self.foregrounded:
            parts.append(f"{TIKTOK_PACKAGE} not foregrounded")
        if not self.audio_started:
            parts.append("no AudioTrack in state:started")
        return "; ".join(parts) or "unknown"


@dataclass(frozen=True)
class BenchmarkSummary:
    api_up_s: float
    capture_ready_s: float
    measurements: tuple[StimulusMeasurement, ...]
    media_pre: MediaState
    media_post: MediaState
    media_pause_intervals: tuple[tuple[datetime, datetime | None], ...] = ()


@dataclass(frozen=True)
class RealDeviceTooling:
    adb_path: str
    scrcpy_path: str
    ffmpeg_path: str


class MediaWatcher:
    """Background poller that records TikTok pause intervals during a run.

    Per the real-device-desktop-verification skill, scrcpy's
    ``--audio-source=playback`` mode can pause Android playback while
    capturing. A run that catches a pause silently still produces honest
    latency numbers but processes garbage capture data, so the benchmark
    must surface the event.
    """

    def __init__(self, adb_path: str, interval_s: float = MEDIA_WATCHER_INTERVAL_S) -> None:
        self._adb_path = adb_path
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._intervals: list[list[datetime | None]] = []
        self._current_pause_start: datetime | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="lsie-media-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> tuple[tuple[datetime, datetime | None], ...]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        with self._lock:
            if self._current_pause_start is not None:
                # Pause ongoing at stop — close it with None to signal "still paused".
                self._intervals.append([self._current_pause_start, None])
                self._current_pause_start = None
            return tuple(
                (interval[0], interval[1])
                for interval in self._intervals
                if interval[0] is not None
            )

    def any_pause_observed(self) -> bool:
        with self._lock:
            return bool(self._intervals) or self._current_pause_start is not None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = _observe_tiktok_state(self._adb_path)
            except Exception:  # noqa: BLE001
                logger.debug("media watcher poll failed", exc_info=True)
                self._stop_event.wait(self._interval_s)
                continue
            now = _now_utc()
            with self._lock:
                if not state.playing and self._current_pause_start is None:
                    self._current_pause_start = now
                    logger.warning("media watcher: pause observed at %s (%s)", now, state.reason())
                elif state.playing and self._current_pause_start is not None:
                    self._intervals.append([self._current_pause_start, now])
                    self._current_pause_start = None
                    logger.warning("media watcher: pause cleared at %s", now)
            self._stop_event.wait(self._interval_s)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _format_s(value: float) -> str:
    return f"{value:.3f}"


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _git_sha_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"


def _adb_shell(adb_path: str, command: str) -> str:
    result = subprocess.run(
        [adb_path, "shell", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"adb shell {command!r} failed: {result.stderr.strip()}")
    return result.stdout


def _observe_tiktok_state(adb_path: str) -> MediaState:
    foreground = _adb_shell(adb_path, "dumpsys activity activities")
    foregrounded = False
    for line in foreground.splitlines():
        stripped = line.strip()
        if stripped.startswith("topResumedActivity=") and TIKTOK_PACKAGE in stripped:
            foregrounded = True
            break

    pid_raw = _adb_shell(adb_path, f"pidof {TIKTOK_PACKAGE}").strip()
    audio_started = False
    if pid_raw:
        target_pids = {token for token in pid_raw.split() if token}
        audio_dump = _adb_shell(adb_path, "dumpsys audio")
        for line in audio_dump.splitlines():
            if "type:android.media.AudioTrack" not in line:
                continue
            if "state:started" not in line:
                continue
            for pid in target_pids:
                if f"/{pid} " in line:
                    audio_started = True
                    break
            if audio_started:
                break
    return MediaState(foregrounded=foregrounded, audio_started=audio_started)


def _ensure_adb_device(adb_path: str) -> str:
    result = subprocess.run(
        [adb_path, "devices"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"adb devices failed: {result.stderr.strip()}")
    serials: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    if not serials:
        raise RuntimeError(
            "no ADB device in 'device' state — connect a phone with USB debugging enabled"
        )
    if len(serials) > 1:
        logger.warning("multiple ADB devices visible (%s); using %s", serials, serials[0])
    return serials[0]


def _allocate_port(preferred: int) -> int:
    for candidate in (preferred, 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", candidate))
            return int(sock.getsockname()[1])
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("could not allocate a local port for the operator API")


def _spawn_desktop_app(
    api_port: int,
    log_path: Path,
    tooling: RealDeviceTooling,
    *,
    runtime_dir: Path = REPO_ROOT,
) -> tuple[subprocess.Popen[bytes], object | None]:
    """Spawn the desktop subprocess + (Windows) wrap it in a Job Object.

    The Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` ensures
    that when the benchmark gives up and force-terminates the desktop
    (after the cooperative-shutdown grace expires), Windows reaps every
    multiprocessing child + scrcpy descendant in one syscall. Without
    this, force-kill leaves the desktop's children as orphans that
    keep spawning scrcpy and recreating the §5.2 transient capture
    artefacts that the cleanup just deleted.
    """
    command, cwd, env = health_check.build_source_launch_command(
        runtime_dir,
        app_root=REPO_ROOT,
        module_args=("--operator-api",),
    )
    env["LSIE_API_PORT"] = str(api_port)
    env["LSIE_ADB_PATH"] = tooling.adb_path
    env["LSIE_SCRCPY_PATH"] = tooling.scrcpy_path
    env["LSIE_FFMPEG_PATH"] = tooling.ffmpeg_path
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
    with log_path.open("a", encoding="utf-8") as launch_log:
        launch_log.write(f"\n--- launching LSIE-MLF from {cwd} with {command[0]} ---\n")
        launch_log.write(format_startup_milestone("launcher_handoff", environ=env) + "\n")
    stdout_handle = log_path.open("ab")
    popen_kwargs: dict[str, Any] = os_adapter._apply_windows_child_process_policy(
        {
            "cwd": cwd,
            "env": env,
            "stdout": stdout_handle,
            "stderr": subprocess.STDOUT,
        },
        hide_window=False,
    )
    proc = subprocess.Popen(
        command,
        cwd=cast(Path, popen_kwargs["cwd"]),
        env=cast(dict[str, str], popen_kwargs["env"]),
        stdout=cast(BinaryIO, popen_kwargs["stdout"]),
        stderr=cast(int, popen_kwargs["stderr"]),
        creationflags=int(popen_kwargs.get("creationflags", 0)),
    )
    job_handle: object | None = None
    if sys.platform == "win32":
        try:
            job_handle = _attach_to_kill_on_close_job(proc.pid)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Job Object attach failed for desktop pid=%s; orphans on force-kill possible: %s",
                proc.pid,
                exc,
            )
    return proc, job_handle


def _attach_to_kill_on_close_job(pid: int) -> object:
    import win32api  # type: ignore[import-not-found]
    import win32con  # type: ignore[import-not-found]
    import win32job  # type: ignore[import-not-found]

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
        pid,
    )
    try:
        win32job.AssignProcessToJobObject(job, proc_handle)
    finally:
        win32api.CloseHandle(proc_handle)
    return job


def _close_job_handle(handle: object | None) -> None:
    if handle is None:
        return
    try:
        import win32api  # type: ignore[import-not-found]

        win32api.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        logger.debug("Job Object close failed", exc_info=True)


def _stop_desktop_app(
    proc: subprocess.Popen[bytes],
    job_handle: object | None = None,
) -> None:
    try:
        if proc.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            logger.exception("failed to send graceful shutdown signal")
        try:
            proc.wait(timeout=SHUTDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            logger.warning(
                "desktop app did not exit within %.0fs of graceful signal; terminating",
                SHUTDOWN_GRACE_S,
            )
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        # Closing the Job Object handle reaps any descendants still
        # alive via KILL_ON_JOB_CLOSE. This is what catches the
        # multiprocessing children that survive the desktop parent's
        # force-termination.
        _close_job_handle(job_handle)


def _wait_until_api_up(client: ApiClient, deadline: float) -> None:
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.get_health()
            return
        except (ApiError, OSError) as exc:
            last_exc = exc
            time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"loopback API never came up; last error: {last_exc!r}")


def _wait_until_capture_ready(
    client: ApiClient,
    session_id: uuid.UUID,
    deadline: float,
) -> SessionSummary:
    last: SessionSummary | None = None
    while time.monotonic() < deadline:
        try:
            session = client.get_session(session_id)
        except ApiError:
            time.sleep(POLL_INTERVAL_S)
            continue
        last = session
        if session.is_calibrating is False:
            return session
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"capture never finished calibrating; last session readback: {last!r}")


def _await_encounter_for_stimulus(
    client: ApiClient,
    session_id: uuid.UUID,
    stimulus_time_utc: datetime | None,
    submitted_at: datetime,
    deadline: float,
) -> tuple[EncounterSummary, datetime]:
    while time.monotonic() < deadline:
        try:
            encounters = client.list_session_encounters(session_id, limit=20)
        except ApiError:
            time.sleep(POLL_INTERVAL_S)
            continue
        match = _match_stimulus_encounter(encounters, stimulus_time_utc, submitted_at)
        if match is not None:
            return match, _now_utc()
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"no encounter row visible within {ENCOUNTER_TIMEOUT_S:.0f}s of stimulus submit"
    )


def _match_stimulus_encounter(
    encounters: Sequence[EncounterSummary],
    stimulus_time_utc: datetime | None,
    submitted_at: datetime,
) -> EncounterSummary | None:
    for enc in encounters:
        if stimulus_time_utc is not None and enc.stimulus_time_utc == stimulus_time_utc:
            return enc
        # Fall back to "first encounter whose segment closed after submit"
        # for environments where the stimulus_time_utc round-trip differs
        # by sub-second formatting between the API and local datetime.
        if stimulus_time_utc is None and enc.segment_timestamp_utc >= submitted_at:
            return enc
    return None


def _drive_stimulus(
    client: ApiClient,
    session_id: uuid.UUID,
    iteration: int,
) -> StimulusMeasurement:
    client_action_id = uuid.uuid4()
    request = StimulusRequest(
        operator_note=f"benchmark stimulus #{iteration}",
        client_action_id=client_action_id,
    )
    submit_t = time.monotonic()
    submitted_at = _now_utc()
    accepted = client.post_stimulus(session_id, request)
    if not accepted.accepted:
        raise RuntimeError(f"stimulus #{iteration} rejected: {accepted.message!r}")
    deadline = submit_t + ENCOUNTER_TIMEOUT_S
    encounter, visible_at = _await_encounter_for_stimulus(
        client,
        session_id,
        accepted.stimulus_time_utc,
        submitted_at,
        deadline,
    )
    end_to_end_s = (visible_at - submitted_at).total_seconds()
    segment_to_visible_ms = (visible_at - encounter.segment_timestamp_utc).total_seconds() * 1000.0
    return StimulusMeasurement(
        client_action_id=client_action_id,
        submitted_at=submitted_at,
        encounter_visible_at=visible_at,
        stimulus_time_utc=accepted.stimulus_time_utc,
        segment_timestamp_utc=encounter.segment_timestamp_utc,
        end_to_end_s=end_to_end_s,
        segment_to_visible_ms=max(0.0, segment_to_visible_ms),
    )


def _format_markdown_row(cells: Sequence[str]) -> str:
    escaped = [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in cells]
    return "| " + " | ".join(escaped) + " |"


def _build_row(summary: BenchmarkSummary, *, notes_suffix: str = "") -> tuple[str, ...]:
    end_to_end = [m.end_to_end_s for m in summary.measurements]
    seg_to_vis = [m.segment_to_visible_ms for m in summary.measurements]
    notes_parts = [
        "v4 real-device end-to-end benchmark",
        "operator API (no GUI)",
        f"stimuli={len(summary.measurements)}",
        f"media-pre={summary.media_pre.reason()}",
        f"media-post={summary.media_post.reason()}",
    ]
    if summary.media_pause_intervals:
        notes_parts.append(
            f"INVALID: {len(summary.media_pause_intervals)} mid-run pause(s) observed"
        )
    if not summary.media_post.playing:
        notes_parts.append("INVALID: media paused after teardown")
    if notes_suffix:
        notes_parts.append(notes_suffix)
    return (
        _now_utc().date().isoformat(),
        f"`{_git_sha_short()}`",
        SCENARIO_LABEL,
        str(len(summary.measurements)),
        _format_s(summary.api_up_s),
        _format_s(summary.capture_ready_s),
        _format_s(_percentile(end_to_end, 50.0)),
        _format_s(_percentile(end_to_end, 95.0)),
        _format_ms(_percentile(seg_to_vis, 50.0)),
        _format_ms(_percentile(seg_to_vis, 95.0)),
        "; ".join(notes_parts),
    )


def _append_baseline_row(path: Path, row: str) -> None:
    text = path.read_text(encoding="utf-8")
    sep = "" if text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}{row}\n", encoding="utf-8")


def run_benchmark(
    *,
    stimuli: int,
    api_port: int,
    stream_url: str,
    experiment_id: str,
    log_path: Path,
) -> BenchmarkSummary:
    resolved_tools, missing_tools = health_check.resolve_desktop_runtime_tools()
    if missing_tools:
        raise RuntimeError(health_check.describe_missing_desktop_tooling(missing_tools))
    tooling = RealDeviceTooling(
        adb_path=resolved_tools["adb"],
        scrcpy_path=resolved_tools["scrcpy"],
        ffmpeg_path=resolved_tools["ffmpeg"],
    )
    serial = _ensure_adb_device(tooling.adb_path)
    logger.info("ADB device ready: %s", serial)

    media_pre = _observe_tiktok_state(tooling.adb_path)
    logger.info("device media state (pre): %s", media_pre.reason())
    if not media_pre.playing:
        raise RuntimeError(
            f"device media state is not playing: {media_pre.reason()}. "
            f"Foreground {TIKTOK_PACKAGE} and start a video before retrying — "
            "a paused source invalidates capture/playback-preservation conclusions."
        )

    proc, job_handle = _spawn_desktop_app(api_port, log_path, tooling)
    logger.info("desktop app launched pid=%s api_port=%d log=%s", proc.pid, api_port, log_path)
    try:
        client = ApiClient(f"http://127.0.0.1:{api_port}", timeout_seconds=15.0)

        api_up_t0 = time.monotonic()
        _wait_until_api_up(client, deadline=api_up_t0 + API_READY_TIMEOUT_S)
        api_up_s = time.monotonic() - api_up_t0
        logger.info("loopback API up after %.2fs", api_up_s)

        client_action_id = uuid.uuid4()
        session_request = SessionCreateRequest(
            stream_url=stream_url,
            experiment_id=experiment_id,
            client_action_id=client_action_id,
        )
        session_lifecycle = client.post_session_start(session_request)
        if not session_lifecycle.accepted:
            raise RuntimeError(f"session start rejected: {session_lifecycle.message!r}")
        session_id = session_lifecycle.session_id
        logger.info("session %s started", session_id)

        capture_t0 = time.monotonic()
        _wait_until_capture_ready(
            client,
            session_id,
            deadline=capture_t0 + CAPTURE_READY_TIMEOUT_S,
        )
        capture_ready_s = time.monotonic() - capture_t0
        logger.info("capture stabilized after %.2fs", capture_ready_s)

        watcher = MediaWatcher(tooling.adb_path)
        watcher.start()
        try:
            measurements: list[StimulusMeasurement] = []
            for iteration in range(1, stimuli + 1):
                measurement = _drive_stimulus(client, session_id, iteration)
                measurements.append(measurement)
                logger.info(
                    "stimulus #%d: end_to_end=%.3fs, segment_to_visible=%.1fms",
                    iteration,
                    measurement.end_to_end_s,
                    measurement.segment_to_visible_ms,
                )
                if iteration < stimuli:
                    time.sleep(STIMULUS_INTERVAL_S)
        finally:
            pause_intervals = watcher.stop()

        # Best-effort session end.
        try:
            client.post_session_end(
                session_id,
                SessionEndRequest(client_action_id=uuid.uuid4()),
            )
        except ApiError as exc:
            logger.warning("session end rejected: %s", exc)

        partial = BenchmarkSummary(
            api_up_s=api_up_s,
            capture_ready_s=capture_ready_s,
            measurements=tuple(measurements),
            media_pre=media_pre,
            media_post=MediaState(foregrounded=False, audio_started=False),
            media_pause_intervals=pause_intervals,
        )
    finally:
        _stop_desktop_app(proc, job_handle)

    media_post = _observe_tiktok_state(tooling.adb_path)
    logger.info("device media state (post): %s", media_post.reason())
    return BenchmarkSummary(
        api_up_s=partial.api_up_s,
        capture_ready_s=partial.capture_ready_s,
        measurements=partial.measurements,
        media_pre=media_pre,
        media_post=media_post,
        media_pause_intervals=partial.media_pause_intervals,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-device end-to-end benchmark for the v4 desktop runtime.",
    )
    parser.add_argument(
        "-n",
        "--stimuli",
        type=int,
        default=DEFAULT_STIMULI,
        help=f"Number of stimuli to fire per run (default: {DEFAULT_STIMULI})",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=DEFAULT_API_PORT,
        help=f"Loopback API port to bind (default: {DEFAULT_API_PORT})",
    )
    parser.add_argument(
        "--stream-url",
        default=DEFAULT_STREAM_URL,
        help="Stream URL (label) for the session record",
    )
    parser.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help="Experiment ID; must match a row in the local seed",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Markdown baseline log to append the run row to",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Print the row but do not append to the baseline log",
    )
    parser.add_argument(
        "--notes-suffix",
        default="",
        help="Optional suffix appended to the Notes cell",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Path for the desktop subprocess log (default: temp file)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    if args.log_path is not None:
        log_path = args.log_path.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, raw = tempfile.mkstemp(prefix="lsie-real-device-", suffix=".log")
        os.close(fd)
        log_path = Path(raw)

    api_port = _allocate_port(int(args.api_port))
    if api_port != int(args.api_port):
        logger.warning("requested API port %d unavailable; using %d", args.api_port, api_port)

    try:
        summary = run_benchmark(
            stimuli=int(args.stimuli),
            api_port=api_port,
            stream_url=str(args.stream_url),
            experiment_id=str(args.experiment_id),
            log_path=log_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("benchmark failed: %s", exc)
        logger.error("desktop subprocess log retained at: %s", log_path)
        return 1

    row_cells = _build_row(summary, notes_suffix=str(args.notes_suffix))
    row = _format_markdown_row(row_cells)
    print(row)
    post_paused = not summary.media_post.playing
    mid_run_paused = bool(summary.media_pause_intervals)
    invalidated = post_paused or mid_run_paused
    if mid_run_paused:
        logger.error(
            "media watcher recorded %d pause interval(s) during the run; refusing to append.",
            len(summary.media_pause_intervals),
        )
    if post_paused:
        logger.error(
            "device media state changed during run (post: %s); refusing to append.",
            summary.media_post.reason(),
        )
    if not invalidated and not args.no_append:
        _append_baseline_row(args.baseline_path.resolve(), row)
    logger.info("desktop subprocess log: %s", log_path)
    return 1 if invalidated else 0


if __name__ == "__main__":
    raise SystemExit(main())
