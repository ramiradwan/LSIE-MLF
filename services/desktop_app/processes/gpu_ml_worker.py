"""GPU ML worker process.

This is the **only** module in the desktop graph that may import
``torch``, ``mediapipe``, ``faster_whisper``, or ``ctranslate2``. The
parent process never imports this module — it is launched by string
through :func:`services.desktop_app.process_graph._launch`, so the ML
libraries are pulled into a dedicated child process and never into the
UI / API / orchestrator / state / cloud-sync surfaces.

The ML imports stay at module top level so the isolation canary can
prove they remain confined to this process. ``run`` stays intentionally
narrow around that boundary.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import math
import multiprocessing.synchronize as mpsync
import os
import queue
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The four ML library imports below are intentional at module top
# level: they prove the v4.0 §9 isolation contract. The canary test
# imports each non-ML process module in a clean subprocess and asserts
# these names are absent from ``sys.modules``; re-importing this
# module DOES bring them in — that is the whole point of routing ML
# inference into a dedicated child process.
import ctranslate2  # noqa: F401
import faster_whisper  # noqa: F401
import mediapipe  # noqa: F401
import torch  # noqa: F401

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.ipc.shared_buffers import read_pcm_block
from services.desktop_app.os_adapter import SupervisedProcess, find_executable

logger = logging.getLogger(__name__)

INBOX_POLL_TIMEOUT_S = 0.5
SQLITE_FILENAME = "desktop.sqlite"
_PCM_SAMPLE_WIDTH_BYTES = 2
_VIDEO_FILENAME = "video_stream.mkv"
_VISUAL_CALIBRATION_FRAMES_REQUIRED = 45
_VISUAL_AU12_RING_MAXLEN = 1800
_VISUAL_FRAME_WIDTH = 540
_VISUAL_FRAME_HEIGHT = 960
_VISUAL_OUTPUT_FPS = 15
_VISUAL_FRAME_BYTES = _VISUAL_FRAME_WIDTH * _VISUAL_FRAME_HEIGHT * 3
_VISUAL_TICK_INTERVAL_S = 1.0 / float(_VISUAL_OUTPUT_FPS)
_DESKTOP_STREAM_STOP_TIMEOUT_S = 1.0
_DESKTOP_SCREENRECORD_TIME_LIMIT_S = 180
_DESKTOP_SCREENRECORD_BIT_RATE = 2_000_000
_GPU_STATUS_KEY = "gpu_ml_worker"
_NO_FRAME_DETAIL = "No shared desktop video frames available yet for live face tracking."
_NO_FRAME_HINT = "Verify Video Capture is recording and keep a visible face on screen."


@dataclass(frozen=True)
class _GpuWorkerStatus:
    state: str
    detail: str
    operator_action_hint: str | None = None


@dataclass(frozen=True)
class _Au12Observation:
    timestamp_s: float
    intensity: float

    def as_payload(self) -> dict[str, float]:
        return {"timestamp_s": self.timestamp_s, "intensity": self.intensity}


@dataclass(frozen=True)
class _VisualTickOutcome:
    visual: VisualAnalyticsStateMessage | None
    missing_frame: bool = False


class DesktopScreenrecordCapture:
    def __init__(self, adb_path: str, ffmpeg_path: str) -> None:
        self._adb_path = adb_path
        self._ffmpeg_path = ffmpeg_path
        self._frame_buffer: deque[Any] = deque(maxlen=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_proc: SupervisedProcess | None = None
        self._decoder_proc: SupervisedProcess | None = None

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="desktop-screenrecord-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_stream()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_DESKTOP_STREAM_STOP_TIMEOUT_S)
            if thread.is_alive():
                logger.warning("Desktop screenrecord capture thread did not stop within timeout")
            else:
                self._thread = None
        self._frame_buffer.clear()

    def get_latest_frame(self) -> Any | None:
        try:
            return self._frame_buffer[-1]
        except IndexError:
            return None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._start_stream()
                stdout = self._decoder_proc.stdout if self._decoder_proc is not None else None
                if stdout is None:
                    raise EOFError("desktop visual decoder has no stdout")
                while not self._stop_event.is_set():
                    payload = _read_exact(stdout, _VISUAL_FRAME_BYTES)
                    if len(payload) != _VISUAL_FRAME_BYTES:
                        raise EOFError("desktop visual stream ended")
                    self._frame_buffer.append(_bgr_frame_from_raw(payload))
            except (EOFError, OSError):
                if not self._stop_event.is_set():
                    logger.debug("Desktop screenrecord stream restarting", exc_info=True)
                    self._stop_stream()
                    self._stop_event.wait(timeout=0.5)
            except Exception:
                if not self._stop_event.is_set():
                    logger.warning("Desktop screenrecord stream failed", exc_info=True)
                    self._stop_stream()
                    self._stop_event.wait(timeout=1.0)
        self._stop_stream()

    def _start_stream(self) -> None:
        self._stop_stream()
        source = SupervisedProcess(
            [
                self._adb_path,
                "exec-out",
                "screenrecord",
                "--output-format=h264",
                f"--time-limit={_DESKTOP_SCREENRECORD_TIME_LIMIT_S}",
                f"--size={_VISUAL_FRAME_WIDTH}x{_VISUAL_FRAME_HEIGHT}",
                f"--bit-rate={_DESKTOP_SCREENRECORD_BIT_RATE}",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if source.stdout is None:
            source.close()
            raise EOFError("desktop screenrecord has no stdout")
        decoder = SupervisedProcess(
            [
                self._ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                (
                    f"fps={_VISUAL_OUTPUT_FPS},"
                    f"scale={_VISUAL_FRAME_WIDTH}:{_VISUAL_FRAME_HEIGHT}:"
                    "force_original_aspect_ratio=decrease,"
                    f"pad={_VISUAL_FRAME_WIDTH}:{_VISUAL_FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
                ),
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdin=source.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._source_proc = source
        self._decoder_proc = decoder

    def _stop_stream(self) -> None:
        for proc in (self._decoder_proc, self._source_proc):
            if proc is not None:
                proc.close()
        self._decoder_proc = None
        self._source_proc = None


@dataclass
class LiveVisualTracker:
    capture_path: str | None = None
    video_capture_factory: Callable[[str], Any] | None = None
    face_mesh_factory: Callable[[], Any] | None = None
    au12_factory: Callable[[], Any] | None = None
    session_id: uuid.UUID | None = None
    active_arm: str | None = None
    expected_greeting: str | None = None
    stimulus_time_s: float | None = None
    _capture: Any = field(default=None, init=False, repr=False)
    _face_mesh: Any = field(default=None, init=False, repr=False)
    _au12: Any = field(default=None, init=False, repr=False)
    _calibration_frames_accumulated: int = field(default=0, init=False)
    _latest_au12_intensity: float | None = field(default=None, init=False)
    _latest_au12_timestamp_s: float | None = field(default=None, init=False)
    _message_sequence: int = field(default=0, init=False)
    _au12_observations: deque[_Au12Observation] = field(
        default_factory=lambda: deque(maxlen=_VISUAL_AU12_RING_MAXLEN),
        init=False,
        repr=False,
    )

    def handle_control(
        self,
        control: LiveSessionControlMessage,
    ) -> VisualAnalyticsStateMessage:
        if control.action == "end":
            ended_session_id = control.session_id
            self.close()
            return self._visual_state(
                session_id=ended_session_id,
                timestamp_utc=control.timestamp_utc,
                status="no_session",
                face_present=False,
            )
        if control.action == "start":
            self.close()
            self.session_id = control.session_id
            self.active_arm = control.active_arm
            self.expected_greeting = control.expected_greeting
            self.stimulus_time_s = None
            self._ensure_capture_started()
            return self._visual_state(
                session_id=control.session_id,
                timestamp_utc=control.timestamp_utc,
                status="waiting_for_face",
                face_present=False,
            )
        self.session_id = control.session_id
        self.active_arm = control.active_arm or self.active_arm
        self.expected_greeting = control.expected_greeting or self.expected_greeting
        self.stimulus_time_s = control.stimulus_time_s
        ready = self._calibration_frames_accumulated >= _VISUAL_CALIBRATION_FRAMES_REQUIRED
        return self._visual_state(
            session_id=control.session_id,
            timestamp_utc=control.timestamp_utc,
            status="post_stimulus" if ready else "waiting_for_face",
            face_present=ready,
        )

    def tick(self, now: datetime | None = None) -> _VisualTickOutcome:
        if self.session_id is None:
            return _VisualTickOutcome(visual=None)
        timestamp_utc = now or datetime.now(UTC)
        frame = self._latest_frame()
        if frame is None:
            calibrated = self._calibration_frames_accumulated >= _VISUAL_CALIBRATION_FRAMES_REQUIRED
            status = "waiting_for_face"
            if calibrated:
                status = "post_stimulus" if self.stimulus_time_s is not None else "ready"
            return _VisualTickOutcome(
                visual=self._visual_state(
                    session_id=self.session_id,
                    timestamp_utc=timestamp_utc,
                    status=status,
                    face_present=calibrated,
                ),
                missing_frame=not calibrated,
            )
        landmarks = self._extract_landmarks(frame)
        if landmarks is None:
            return _VisualTickOutcome(
                visual=self._visual_state(
                    session_id=self.session_id,
                    timestamp_utc=timestamp_utc,
                    status="waiting_for_face",
                    face_present=False,
                )
            )

        if self._calibration_frames_accumulated < _VISUAL_CALIBRATION_FRAMES_REQUIRED:
            if self._compute_au12(landmarks, is_calibrating=True) is not None:
                self._calibration_frames_accumulated = min(
                    self._calibration_frames_accumulated + 1,
                    _VISUAL_CALIBRATION_FRAMES_REQUIRED,
                )
                self._sync_calibration_count_from_normalizer()
            status = (
                "ready"
                if self._calibration_frames_accumulated >= _VISUAL_CALIBRATION_FRAMES_REQUIRED
                else "calibrating"
            )
            return _VisualTickOutcome(
                visual=self._visual_state(
                    session_id=self.session_id,
                    timestamp_utc=timestamp_utc,
                    status=status,
                    face_present=True,
                )
            )

        intensity = self._compute_au12(landmarks, is_calibrating=False)
        if intensity is None:
            return _VisualTickOutcome(
                visual=self._visual_state(
                    session_id=self.session_id,
                    timestamp_utc=timestamp_utc,
                    status="ready",
                    face_present=True,
                )
            )
        timestamp_s = timestamp_utc.timestamp()
        self._latest_au12_intensity = intensity
        self._latest_au12_timestamp_s = timestamp_s
        status = "post_stimulus" if self.stimulus_time_s is not None else "ready"
        self._au12_observations.append(
            _Au12Observation(timestamp_s=timestamp_s, intensity=intensity)
        )
        return _VisualTickOutcome(
            visual=self._visual_state(
                session_id=self.session_id,
                timestamp_utc=timestamp_utc,
                status=status,
                face_present=True,
            )
        )

    def drain_au12_observations(
        self,
        *,
        start_s: float,
        end_s: float,
    ) -> list[dict[str, float]]:
        selected: list[_Au12Observation] = []
        retained: deque[_Au12Observation] = deque(maxlen=_VISUAL_AU12_RING_MAXLEN)
        for observation in self._au12_observations:
            if start_s <= observation.timestamp_s <= end_s:
                selected.append(observation)
            if observation.timestamp_s > end_s:
                retained.append(observation)
        self._au12_observations = retained
        return [observation.as_payload() for observation in selected]

    def close(self) -> None:
        if self._capture is not None:
            with contextlib.suppress(Exception):
                self._capture.stop()
            self._capture = None
        if self._face_mesh is not None:
            with contextlib.suppress(Exception):
                self._face_mesh.close()
            self._face_mesh = None
        self._au12 = None
        self.session_id = None
        self.active_arm = None
        self.expected_greeting = None
        self.stimulus_time_s = None
        self._calibration_frames_accumulated = 0
        self._latest_au12_intensity = None
        self._latest_au12_timestamp_s = None
        self._au12_observations.clear()

    def _visual_state(
        self,
        *,
        session_id: uuid.UUID,
        timestamp_utc: datetime,
        status: str,
        face_present: bool,
    ) -> VisualAnalyticsStateMessage:
        self._message_sequence += 1
        is_calibrating = (
            status in {"waiting_for_face", "calibrating"}
            and self._calibration_frames_accumulated < _VISUAL_CALIBRATION_FRAMES_REQUIRED
        )
        return VisualAnalyticsStateMessage.model_validate(
            {
                "message_id": str(
                    _visual_state_message_id(
                        session_id=session_id,
                        status=status,
                        timestamp_utc=timestamp_utc,
                        sequence=self._message_sequence,
                    )
                ),
                "session_id": str(session_id),
                "timestamp_utc": timestamp_utc,
                "face_present": face_present,
                "is_calibrating": is_calibrating,
                "calibration_frames_accumulated": self._calibration_frames_accumulated,
                "calibration_frames_required": _VISUAL_CALIBRATION_FRAMES_REQUIRED,
                "active_arm": self.active_arm,
                "expected_greeting": self.expected_greeting,
                "latest_au12_intensity": self._latest_au12_intensity,
                "latest_au12_timestamp_s": self._latest_au12_timestamp_s,
                "status": status,
            }
        )

    def _ensure_capture_started(self) -> None:
        if self._capture is None:
            factory = self.video_capture_factory or _default_video_capture_factory
            self._capture = factory(self._capture_path())
        with contextlib.suppress(Exception):
            self._capture.start()

    def _capture_path(self) -> str:
        if self.capture_path is not None:
            return self.capture_path
        from services.desktop_app.os_adapter import resolve_capture_dir

        return str(resolve_capture_dir() / _VIDEO_FILENAME)

    def _latest_frame(self) -> Any | None:
        self._ensure_capture_started()
        try:
            return self._capture.get_latest_frame()
        except Exception:
            logger.warning("Visual tracker failed to read latest frame", exc_info=True)
            return None

    def _extract_landmarks(self, frame: Any) -> Any | None:
        if self._face_mesh is None:
            factory = self.face_mesh_factory or _default_face_mesh_factory
            self._face_mesh = factory()
        try:
            return self._face_mesh.extract_landmarks(frame)
        except Exception:
            logger.warning("Visual tracker failed to extract face landmarks", exc_info=True)
            return None

    def _compute_au12(self, landmarks: Any, *, is_calibrating: bool) -> float | None:
        if self._au12 is None:
            factory = self.au12_factory or _default_au12_factory
            self._au12 = factory()
        try:
            intensity = float(
                self._au12.compute_bounded_intensity(
                    landmarks,
                    is_calibrating=is_calibrating,
                )
            )
        except Exception:
            logger.warning("Visual tracker failed to compute AU12 intensity", exc_info=True)
            return None
        return min(1.0, max(0.0, intensity))

    def _sync_calibration_count_from_normalizer(self) -> None:
        buffer = getattr(self._au12, "calibration_buffer", None)
        if isinstance(buffer, list):
            self._calibration_frames_accumulated = min(
                max(self._calibration_frames_accumulated, len(buffer)),
                _VISUAL_CALIBRATION_FRAMES_REQUIRED,
            )


def _default_video_capture_factory(path: str) -> Any:
    del path
    adb_path = find_executable("adb", env_override="LSIE_ADB_PATH")
    ffmpeg_path = find_executable("ffmpeg", env_override="LSIE_FFMPEG_PATH")
    return DesktopScreenrecordCapture(adb_path, ffmpeg_path)


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _bgr_frame_from_raw(payload: bytes) -> Any:
    import numpy as np

    return (
        np.frombuffer(payload, dtype=np.uint8)
        .reshape((_VISUAL_FRAME_HEIGHT, _VISUAL_FRAME_WIDTH, 3))
        .copy()
    )


def _default_face_mesh_factory() -> Any:
    from packages.ml_core.face_mesh import FaceMeshProcessor

    return FaceMeshProcessor()


def _default_au12_factory() -> Any:
    from packages.ml_core.au12 import AU12Normalizer

    return AU12Normalizer()


def _visual_state_message_id(
    *,
    session_id: uuid.UUID,
    status: str,
    timestamp_utc: datetime,
    sequence: int,
) -> uuid.UUID:
    digest = bytearray(
        hashlib.sha256(
            f"visual-state:{session_id}:{status}:{timestamp_utc.isoformat()}:{sequence}".encode()
        ).digest()[:16]
    )
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(digest))


def _default_acoustic_payload() -> dict[str, Any]:
    return {
        "f0_valid_measure": False,
        "f0_valid_baseline": False,
        "perturbation_valid_measure": False,
        "perturbation_valid_baseline": False,
        "voiced_coverage_measure_s": 0.0,
        "voiced_coverage_baseline_s": 0.0,
        "f0_mean_measure_hz": None,
        "f0_mean_baseline_hz": None,
        "f0_delta_semitones": None,
        "jitter_mean_measure": None,
        "jitter_mean_baseline": None,
        "jitter_delta": None,
        "shimmer_mean_measure": None,
        "shimmer_mean_baseline": None,
        "shimmer_delta": None,
    }


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_nonnegative_float_or_default(value: Any, default: float = 0.0) -> float:
    result = _finite_float_or_none(value)
    if result is None or result < 0.0:
        return default
    return result


def _serialize_acoustic_metrics(metrics: Any) -> dict[str, Any]:
    acoustic_payload = _default_acoustic_payload()
    acoustic_payload.update(asdict(metrics))
    for metric_field in (
        "f0_mean_measure_hz",
        "f0_mean_baseline_hz",
        "f0_delta_semitones",
        "jitter_mean_measure",
        "jitter_mean_baseline",
        "jitter_delta",
        "shimmer_mean_measure",
        "shimmer_mean_baseline",
        "shimmer_delta",
    ):
        acoustic_payload[metric_field] = _finite_float_or_none(acoustic_payload.get(metric_field))
    for metric_field in ("voiced_coverage_measure_s", "voiced_coverage_baseline_s"):
        acoustic_payload[metric_field] = _finite_nonnegative_float_or_default(
            acoustic_payload.get(metric_field)
        )
    return acoustic_payload


def _derive_segment_start_time_s(
    *,
    timestamp_utc: str,
    audio_data: bytes,
    sample_rate: int,
) -> float | None:
    try:
        segment_end_time_s = (
            datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00")).astimezone(UTC).timestamp()
        )
    except ValueError:
        return None
    if sample_rate <= 0 or len(audio_data) < _PCM_SAMPLE_WIDTH_BYTES:
        return None
    segment_duration_s = len(audio_data) / (_PCM_SAMPLE_WIDTH_BYTES * sample_rate)
    return segment_end_time_s - segment_duration_s


def _parse_utc_timestamp_s(value: Any) -> float | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC).timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp()
        except ValueError:
            return None
    return None


def _handoff_window_s(
    handoff: dict[str, Any],
    audio_data: bytes,
) -> tuple[float, float] | None:
    start_s = _parse_utc_timestamp_s(handoff.get("segment_window_start_utc"))
    end_s = _parse_utc_timestamp_s(handoff.get("segment_window_end_utc"))
    if start_s is not None and end_s is not None:
        return (start_s, end_s)
    timestamp_utc = str(handoff.get("timestamp_utc", ""))
    derived_start = _derive_segment_start_time_s(
        timestamp_utc=timestamp_utc,
        audio_data=audio_data,
        sample_rate=16000,
    )
    derived_end = _parse_utc_timestamp_s(timestamp_utc)
    if derived_start is None or derived_end is None:
        return None
    return (derived_start, derived_end)


def _handoff_with_tracker_au12(
    handoff: dict[str, Any],
    *,
    audio_data: bytes,
    tracker: LiveVisualTracker | None,
) -> dict[str, Any]:
    enriched = dict(handoff)
    if enriched.get("_physiological_context") is None:
        enriched.pop("_physiological_context", None)
    if enriched.get("physiological_context") is None:
        enriched.pop("physiological_context", None)
    if tracker is None:
        return enriched
    window = _handoff_window_s(enriched, audio_data)
    observations = (
        []
        if window is None
        else tracker.drain_au12_observations(
            start_s=window[0],
            end_s=window[1],
        )
    )
    enriched["_au12_series"] = observations
    return enriched


def _normalize_semantic_result(
    semantic: dict[str, Any] | None,
    *,
    semantic_method: str | None = None,
    semantic_method_version: str | None = None,
) -> dict[str, Any] | None:
    if semantic is None:
        return None
    from packages.schemas.evaluation import SEMANTIC_METHODS, SEMANTIC_REASON_CODES

    method = semantic_method or semantic.get("semantic_method") or "cross_encoder"
    if method not in SEMANTIC_METHODS:
        method = "cross_encoder"
    reasoning = semantic.get("reasoning")
    if reasoning not in SEMANTIC_REASON_CODES:
        reasoning = "semantic_error"
    confidence = _finite_float_or_none(semantic.get("confidence_score"))
    if confidence is None:
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return {
        "reasoning": reasoning,
        "is_match": bool(semantic.get("is_match", False)),
        "confidence_score": confidence,
        "semantic_method": method,
        "semantic_method_version": semantic_method_version
        or semantic.get("semantic_method_version")
        or "desktop-gpu-worker-v1",
    }


def _analytics_message_id(segment_id: str) -> uuid.UUID:
    digest = bytearray(hashlib.sha256(f"analytics-result:{segment_id}".encode()).digest()[:16])
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return uuid.UUID(bytes=bytes(digest))


def _upsert_gpu_worker_status(db_path: Path, status: _GpuWorkerStatus) -> None:
    updated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO capture_status "
            "(status_key, state, label, detail, operator_action_hint, updated_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(status_key) DO UPDATE SET "
            "state = excluded.state, "
            "label = excluded.label, "
            "detail = excluded.detail, "
            "operator_action_hint = excluded.operator_action_hint, "
            "updated_at_utc = excluded.updated_at_utc",
            (
                _GPU_STATUS_KEY,
                status.state,
                "gpu_ml_worker",
                status.detail,
                status.operator_action_hint,
                updated_at,
            ),
        )
    finally:
        conn.close()


def _initial_worker_status() -> _GpuWorkerStatus:
    detail = "gpu_ml_worker is starting speech and live visual pipelines."
    if os.environ.get("LSIE_DEV_FORCE_CPU_SPEECH") == "1":
        detail = "gpu_ml_worker is starting with LSIE_DEV_FORCE_CPU_SPEECH=1."
    return _GpuWorkerStatus(state="recovering", detail=detail)


def _warm_transcription_engine() -> tuple[Any | None, _GpuWorkerStatus]:
    from packages.ml_core.transcription import TranscriptionEngine

    try:
        engine = TranscriptionEngine()
        engine.load_model()
    except Exception:
        logger.warning("gpu_ml_worker failed to warm transcription engine", exc_info=True)
        return (
            None,
            _GpuWorkerStatus(
                state="recovering",
                detail="Speech model warmup is still in progress.",
                operator_action_hint=(
                    "Wait for speech model load to complete or inspect gpu_ml_worker logs."
                ),
            ),
        )
    device = str(getattr(engine, "device", "unknown"))
    compute_type = str(getattr(engine, "compute_type", "unknown"))
    return (
        engine,
        _GpuWorkerStatus(
            state="ok",
            detail=(f"Speech model warmed on device={device} compute_type={compute_type}."),
        ),
    )


def _ack_pcm_block(channels: IpcChannels, name: str) -> None:
    if channels.pcm_acks is None:
        return
    channels.pcm_acks.put(PcmBlockAckMessage(name=name).model_dump(mode="json"))


def _build_analytics_result(
    msg: InferenceControlMessage,
    audio: bytes,
    tracker: LiveVisualTracker | None = None,
    *,
    transcription_engine: Any | None = None,
) -> AnalyticsResultMessage | None:
    timings: dict[str, float] = {}
    pipeline_started = time.perf_counter()

    handoff_start = time.perf_counter()
    handoff = _handoff_with_tracker_au12(
        msg.handoff,
        audio_data=audio,
        tracker=tracker,
    )
    timings["handoff_au12"] = (time.perf_counter() - handoff_start) * 1000.0
    segment_id = str(handoff.get("segment_id", "unknown"))
    timestamp_utc = str(handoff["timestamp_utc"])
    stimulus_time = handoff.get("_stimulus_time")

    transcription = ""
    try:
        from packages.ml_core.audio_pipe import pcm_to_wav_bytes

        wav_start = time.perf_counter()
        wav_bytes = pcm_to_wav_bytes(audio)
        timings["pcm_to_wav"] = (time.perf_counter() - wav_start) * 1000.0

        engine = transcription_engine
        if engine is None:
            from packages.ml_core.transcription import TranscriptionEngine

            engine_init_start = time.perf_counter()
            engine = TranscriptionEngine()
            timings["transcription_engine_init"] = (
                time.perf_counter() - engine_init_start
            ) * 1000.0

        transcribe_start = time.perf_counter()
        transcription = engine.transcribe(io.BytesIO(wav_bytes))
        timings["transcribe"] = (time.perf_counter() - transcribe_start) * 1000.0
    except Exception:
        logger.warning("Transcription failed for %s", segment_id, exc_info=True)

    acoustic_payload: dict[str, Any]
    if stimulus_time is None:
        acoustic_payload = _default_acoustic_payload()
    else:
        try:
            from packages.ml_core.acoustic import AcousticAnalyzer

            segment_start_time_s = _derive_segment_start_time_s(
                timestamp_utc=timestamp_utc,
                audio_data=audio,
                sample_rate=16000,
            )
            if segment_start_time_s is None:
                acoustic_payload = _default_acoustic_payload()
            else:
                acoustic_init_start = time.perf_counter()
                analyzer = AcousticAnalyzer()
                timings["acoustic_init"] = (
                    time.perf_counter() - acoustic_init_start
                ) * 1000.0

                acoustic_run_start = time.perf_counter()
                metrics = analyzer.analyze(
                    audio,
                    sample_rate=16000,
                    stimulus_time_s=float(stimulus_time),
                    segment_start_time_s=segment_start_time_s,
                )
                timings["acoustic_analyze"] = (
                    time.perf_counter() - acoustic_run_start
                ) * 1000.0
                acoustic_payload = _serialize_acoustic_metrics(metrics)
        except Exception:
            logger.warning("Acoustic analysis failed for %s", segment_id, exc_info=True)
            acoustic_payload = _default_acoustic_payload()

    semantic: dict[str, Any] | None = None
    if not transcription:
        semantic = _normalize_semantic_result(
            {
                "reasoning": "semantic_error",
                "is_match": False,
                "confidence_score": 0.0,
            },
            semantic_method="cross_encoder",
            semantic_method_version="desktop-gpu-worker-empty-transcription-v1",
        )
    else:
        try:
            from packages.ml_core.preprocessing import TextPreprocessor
            from packages.ml_core.semantic import SemanticEvaluator

            preproc_start = time.perf_counter()
            preprocessed_text = TextPreprocessor().preprocess(transcription)
            timings["text_preprocess"] = (time.perf_counter() - preproc_start) * 1000.0

            evaluator_init_start = time.perf_counter()
            evaluator = SemanticEvaluator()
            timings["semantic_evaluator_init"] = (
                time.perf_counter() - evaluator_init_start
            ) * 1000.0

            evaluate_start = time.perf_counter()
            live_semantic = evaluator.evaluate(
                str(handoff.get("_expected_greeting", "Hello, welcome to the stream!")),
                preprocessed_text,
            )
            timings["semantic_evaluate"] = (
                time.perf_counter() - evaluate_start
            ) * 1000.0

            semantic = _normalize_semantic_result(
                live_semantic,
                semantic_method=getattr(evaluator, "last_semantic_method", None),
                semantic_method_version=getattr(evaluator, "last_semantic_method_version", None),
            )
        except Exception:
            logger.warning("Semantic evaluation failed for %s", segment_id, exc_info=True)
            semantic = _normalize_semantic_result(
                {
                    "reasoning": "semantic_error",
                    "is_match": False,
                    "confidence_score": 0.0,
                },
                semantic_method="cross_encoder",
                semantic_method_version="desktop-gpu-worker-semantic-fallback-v1",
            )

    if semantic is None:
        return None

    timings["pipeline_total"] = (time.perf_counter() - pipeline_started) * 1000.0
    breakdown = ", ".join(f"{name}={ms:.1f}ms" for name, ms in sorted(timings.items()))
    # WARNING level so the per-segment breakdown reaches the parent's
    # default child logger config; INFO would be filtered. The string is
    # cheap and the cadence is bounded by the 30 s segment window.
    logger.warning("gpu_ml_worker segment_id=%s timings: %s", segment_id, breakdown)

    return AnalyticsResultMessage.model_validate(
        {
            "message_id": str(_analytics_message_id(segment_id)),
            "handoff": handoff,
            "semantic": semantic,
            "transcription": transcription,
            "acoustic": acoustic_payload,
        }
    )


def _publish_analytics_result(
    channels: IpcChannels,
    msg: InferenceControlMessage,
    tracker: LiveVisualTracker | None = None,
    *,
    transcription_engine: Any | None = None,
    status_callback: Callable[[_GpuWorkerStatus], None] | None = None,
) -> None:
    audio_name = msg.audio.name
    try:
        audio = read_pcm_block(msg.audio.to_metadata())
    except FileNotFoundError:
        _ack_pcm_block(channels, audio_name)
        segment_id = str(msg.handoff.get("segment_id", "unknown"))
        logger.warning("Dropped expired PCM block for segment_id=%s", segment_id)
        return
    _ack_pcm_block(channels, audio_name)
    analytics = _build_analytics_result(
        msg,
        audio,
        tracker,
        transcription_engine=transcription_engine,
    )
    if analytics is None:
        return
    if status_callback is not None:
        status_callback(
            _GpuWorkerStatus(
                state="ok",
                detail="gpu_ml_worker is processing live analytics normally.",
            )
        )
    channels.analytics_inbox.put(analytics.model_dump(mode="json"))


def _drain_live_control(channels: IpcChannels, tracker: LiveVisualTracker) -> None:
    if channels.live_control is None:
        return
    while True:
        try:
            raw = channels.live_control.get_nowait()
        except queue.Empty:
            return
        try:
            control = LiveSessionControlMessage.model_validate(raw)
            visual = tracker.handle_control(control)
        except Exception:  # noqa: BLE001
            logger.exception("gpu_ml_worker discarded malformed live control message")
            continue
        channels.analytics_inbox.put(visual.model_dump(mode="json"))


def _publish_visual_tick(
    channels: IpcChannels,
    tracker: LiveVisualTracker,
    *,
    status_callback: Callable[[_GpuWorkerStatus], None] | None = None,
) -> None:
    try:
        outcome = tracker.tick()
    except Exception:  # noqa: BLE001
        logger.exception("gpu_ml_worker failed to process visual tracking tick")
        if status_callback is not None:
            status_callback(
                _GpuWorkerStatus(
                    state="degraded",
                    detail="Live visual tracking failed on the latest tick.",
                    operator_action_hint=_NO_FRAME_HINT,
                )
            )
        return
    if status_callback is not None:
        if outcome.missing_frame:
            status_callback(
                _GpuWorkerStatus(
                    state="recovering",
                    detail=_NO_FRAME_DETAIL,
                    operator_action_hint=_NO_FRAME_HINT,
                )
            )
        elif outcome.visual is not None and outcome.visual.face_present:
            status_callback(
                _GpuWorkerStatus(
                    state="ok",
                    detail="Live visual tracking is receiving phone frames.",
                )
            )
    if outcome.visual is not None:
        channels.analytics_inbox.put(outcome.visual.model_dump(mode="json"))


def _ml_inbox_poll_timeout(next_visual_tick: float, now_monotonic: float) -> float:
    return min(INBOX_POLL_TIMEOUT_S, max(0.0, next_visual_tick - now_monotonic))


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("gpu_ml_worker started")

    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder

    state_dir = resolve_state_dir()
    db_path = state_dir / SQLITE_FILENAME
    heartbeat = HeartbeatRecorder(db_path, "gpu_ml_worker")
    heartbeat.start()
    _upsert_gpu_worker_status(db_path, _initial_worker_status())
    tracker = LiveVisualTracker()
    warm_runtime: dict[str, Any] = {
        "engine": None,
        "status": _GpuWorkerStatus(
            state="recovering",
            detail="Speech model warmup is still in progress.",
        ),
        "ready": False,
    }

    def _warm_engine() -> None:
        engine, status = _warm_transcription_engine()
        warm_runtime["engine"] = engine
        warm_runtime["status"] = status
        warm_runtime["ready"] = True

    warm_thread = threading.Thread(
        target=_warm_engine,
        name="gpu-ml-speech-warmup",
        daemon=True,
    )
    warm_thread.start()
    warm_status_reported = False

    next_visual_tick = time.monotonic()
    try:
        while not shutdown_event.is_set():
            if bool(warm_runtime["ready"]) and not warm_status_reported:
                _upsert_gpu_worker_status(db_path, warm_runtime["status"])
                warm_status_reported = True
            _drain_live_control(channels, tracker)
            now_monotonic = time.monotonic()
            if now_monotonic >= next_visual_tick:
                _publish_visual_tick(
                    channels,
                    tracker,
                    status_callback=lambda status: _upsert_gpu_worker_status(db_path, status),
                )
                next_visual_tick = now_monotonic + _VISUAL_TICK_INTERVAL_S
            try:
                raw = channels.ml_inbox.get(
                    timeout=_ml_inbox_poll_timeout(next_visual_tick, time.monotonic())
                )
            except queue.Empty:
                continue
            try:
                msg = InferenceControlMessage.model_validate(raw)
            except Exception:  # noqa: BLE001
                logger.exception("gpu_ml_worker discarded malformed control message")
                continue
            logger.info(
                "gpu_ml_worker received segment_id=%s audio=%s/%d bytes",
                msg.handoff.get("segment_id", "?"),
                msg.audio.name,
                msg.audio.byte_length,
            )
            try:
                _publish_analytics_result(
                    channels,
                    msg,
                    tracker,
                    transcription_engine=warm_runtime["engine"],
                    status_callback=lambda status: _upsert_gpu_worker_status(db_path, status),
                )
            except Exception:  # noqa: BLE001
                _upsert_gpu_worker_status(
                    db_path,
                    _GpuWorkerStatus(
                        state="degraded",
                        detail="gpu_ml_worker failed to publish analytics for the latest segment.",
                        operator_action_hint=(
                            "Check gpu_ml_worker logs and retry after "
                            "speech model warmup completes."
                        ),
                    ),
                )
                logger.exception(
                    "gpu_ml_worker failed to publish analytics result for segment_id=%s",
                    msg.handoff.get("segment_id", "?"),
                )
    finally:
        tracker.close()
        warm_thread.join(timeout=1.0)
        heartbeat.stop()
        logger.info("gpu_ml_worker stopped")
