"""
Module C orchestration and synchronization for pipeline handoff (§4.C).

The module coordinates drift correction, FFmpeg audio resampling, video-frame
capture, AU12 telemetry, stimulus timestamps, physiological context, session
lifecycle state, and construction of InferenceHandoffPayloads for Module D. It
applies ADB drift polling every 30 seconds with frozen/reset fallback (§13.5)
and uses canonical component names (§0, §13.15). It does not perform semantic
scoring, reward updates, or direct ML persistence.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta
from math import sqrt
from queue import Empty, Queue
from typing import Any

from pydantic import ValidationError

from packages.schemas.data_tiers import DataTier, mark_data_tier

logger = logging.getLogger(__name__)

# §4.C.1 drift correction now lives in services.desktop_app.drift; it is
# polled by services.desktop_app.processes.capture_supervisor (WS3 P3)
# and the offset is shipped to module_c_orchestrator over the IPC
# drift_updates channel. Orchestrator keeps a DriftCorrector instance
# only for the apply-side correct_timestamp() call.
from services.desktop_app.drift import (  # noqa: E402
    ADB_COMMAND,
    DRIFT_FREEZE_AFTER_FAILURES,
    DRIFT_POLL_INTERVAL,
    DRIFT_RESET_TIMEOUT,
    MAX_TOLERATED_DRIFT_MS,
    DriftCorrector,
)

# Re-export the symbols above for backwards compatibility with the v3.4
# test suite that imports them from this module.
__all__ = [
    "ADB_COMMAND",
    "DRIFT_FREEZE_AFTER_FAILURES",
    "DRIFT_POLL_INTERVAL",
    "DRIFT_RESET_TIMEOUT",
    "MAX_TOLERATED_DRIFT_MS",
    "DriftCorrector",
]

# §4.C.2 FFmpeg resampling command
FFMPEG_RESAMPLE_CMD: list[str] = [
    "ffmpeg",
    "-f",
    "s16le",
    "-ar",
    "48000",
    "-ac",
    "1",
    "-i",
    "/tmp/ipc/audio_stream.raw",
    "-ar",
    "16000",
    "-f",
    "s16le",
    "-ac",
    "1",
    "pipe:1",
]

# §4.C segment window
SEGMENT_WINDOW_SECONDS: int = 30
DEFAULT_MEDIA_SOURCE_URI: str = "file:///tmp/ipc/video_stream.mkv"
DEFAULT_EXPERIMENT_ROW_ID: int = 0
BANDIT_POLICY_VERSION: str = "thompson_sampling_v1"

GREETING_LINES: dict[str, str] = {
    "warm_welcome": "Hey! Thanks for streaming, you're awesome!",
    "direct_question": "Hi! What's the best advice you've gotten today?",
    "compliment_content": "Love the energy on this stream! How long have you been live?",
    "simple_hello": "Hello! Just joined, happy to be here!",
}
DEFAULT_GREETING_TEXT: str = "Hello, welcome to the stream!"

# §2 step 3 — FFmpeg crash restart delay
FFMPEG_RESTART_DELAY: float = 1.0  # seconds

# §4.C.4 — Physiological State Buffer configuration
PHYSIO_QUEUE_KEY: str = "physio:events"
PHYSIO_BUFFER_RETENTION_S: int = 900
PHYSIO_DERIVE_WINDOW_S: int = 300
PHYSIO_VALIDITY_MIN: float = 0.80
PHYSIO_STALENESS_THRESHOLD_S: float = 600.0  # 10 minutes
MAX_PHYSIO_DRAIN_PER_CYCLE: int = 100  # Bounded drain to prevent stall

# Session lifecycle pub/sub bridge (API Server -> Orchestrator).
SESSION_LIFECYCLE_CHANNEL: str = "session:lifecycle"
SESSION_LIFECYCLE_POLL_TIMEOUT_S: float = 1.0

# Operator-read live-session overlay — a Redis key, not a pub/sub channel.
LIVE_SESSION_STATE_KEY_PREFIX: str = "operator:live_session:"
LIVE_SESSION_STATE_TTL_S: int = 24 * 3600
# Read-only operator readiness threshold. Calibration math still continues
# until stimulus injection, so §7A.4 behaviour is unchanged.
LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED: int = 45

# Operator health heartbeat — written by the orchestrator, read by the API probe.
ORCHESTRATOR_HEARTBEAT_KEY: str = "operator:orchestrator:heartbeat"
ORCHESTRATOR_HEARTBEAT_TTL_S: int = 30
ORCHESTRATOR_HEARTBEAT_INTERVAL_S: float = 1.0


def _live_session_state_key(session_id: str) -> str:
    return f"{LIVE_SESSION_STATE_KEY_PREFIX}{session_id}"


class AudioResampler:
    """
    Manage the persistent FFmpeg resampling pipe for audio handoff.

    Accepts no in-memory audio input; it reads the shared 48 kHz PCM IPC stream,
    starts/restarts the configured FFmpeg subprocess, and produces 16 kHz mono
    PCM chunks from stdout. It does not transcribe audio, own the IPC writer, or
    persist audio bytes.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def is_running(self) -> bool:
        """Check if FFmpeg subprocess is alive."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """
        Launch FFmpeg resampling subprocess.

        §4.C.2 — Exact command from spec: ffmpeg -f s16le -ar 48000 -ac 1
        -i /tmp/ipc/audio_stream.raw -ar 16000 -f s16le -ac 1 pipe:1
        """
        if self.is_running:
            return

        # §4.C.2 — Spawn FFmpeg with stdout piped for reading
        self._process = subprocess.Popen(
            FFMPEG_RESAMPLE_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        logger.info("FFmpeg resampler started (PID %d)", self._process.pid)

    def read_chunk(self, num_bytes: int) -> bytes:
        """
        Read resampled 16 kHz PCM from FFmpeg stdout.

        §2 step 3 — If FFmpeg crashed, restart within 1 second.

        Args:
            num_bytes: Number of bytes to read (e.g. 32000 for 1s at 16kHz mono s16le).

        Returns:
            PCM bytes, or empty bytes if unavailable.
        """
        if not self.is_running:
            # §2 step 3 / §12 Worker crash C — restart FFmpeg
            logger.warning("FFmpeg not running, restarting in %.1fs", FFMPEG_RESTART_DELAY)
            time.sleep(FFMPEG_RESTART_DELAY)
            self.start()

        if self._process is None or self._process.stdout is None:
            return b""

        try:
            data = mark_data_tier(
                self._process.stdout.read(num_bytes),
                DataTier.TRANSIENT,
                spec_ref="§5.2.1",
                purpose="Raw PCM audio buffer from IPC/FFmpeg boundary",
            )  # §5.2.1 Transient Data
            if not data:
                # §12 Worker crash C — EOF means FFmpeg exited
                logger.warning("FFmpeg stdout EOF, process likely crashed")
                self.stop()
                return b""
            return data
        except (OSError, ValueError) as exc:
            logger.error("FFmpeg read error: %s", exc)
            self.stop()
            return b""

    def stop(self) -> None:
        """Terminate FFmpeg subprocess gracefully."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self._process.kill()
            finally:
                self._process = None
            logger.info("FFmpeg resampler stopped")


class Orchestrator:
    """
    Coordinate Module C capture state and assemble inference handoff payloads.

    Accepts stream/session/experiment configuration plus live or replay IPC
    media, ground-truth events, stimulus triggers, and optional physiology
    events. Produces 30-second segment payloads with drift-corrected windows,
    active arm context, per-frame AU12 telemetry, stimulus timestamps, and a
    pre-update BanditDecisionSnapshot for Module D. It does not perform
    transcription, semantic evaluation, reward/posterior updates, or persist raw
    media.
    """

    def __init__(
        self,
        stream_url: str = "",
        session_id: str | None = None,
        experiment_id: str = "greeting_line_v1",
        ipc_queue: Any = None,
    ) -> None:
        """
        Initialize orchestrator with session, experiment, video, and AU12 state.

        Args:
            stream_url: TikTok stream URL for this session.
            session_id: UUID for this session (auto-generated if None).
            experiment_id: Thompson Sampling experiment ID (§4.E.1).
            ipc_queue: ``multiprocessing.Queue``-like sink for v4.0 desktop
                IPC dispatch. When ``None``, ``_dispatch_payload`` logs a
                warning and discards the segment — useful for unit tests
                that exercise other parts of the pipeline. The v4.0
                ``module_c_orchestrator`` process supplies the real queue
                via ``IpcChannels.ml_inbox``.
        """
        self.drift_corrector = DriftCorrector()
        replay_fixture = os.environ.get("REPLAY_CAPTURE_FIXTURE")
        replay_source: Any = None
        self._using_replay_capture: bool = bool(replay_fixture)
        self._replay_capture_source: Any = None
        if replay_fixture:
            from services.worker.pipeline.replay_capture import ReplayCaptureSource

            realtime_value = os.environ.get("REPLAY_CAPTURE_REALTIME", "1").strip().lower()
            replay_realtime = realtime_value not in {"0", "false", "no", "off"}
            replay_source = ReplayCaptureSource(replay_fixture, realtime=replay_realtime)
            self._replay_capture_source = replay_source
            self.audio_resampler = replay_source
            logger.info("Replay capture fixture configured: %s", replay_fixture)
        else:
            self.audio_resampler = AudioResampler()
        # §12 Queue overload B/C — deque eviction
        self.event_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self._stream_url = stream_url
        self._session_id = session_id or str(uuid.uuid4())
        self._segment_counter: int = 0
        self._audio_buffer: bytearray = bytearray()
        self._running: bool = False

        # Thompson Sampling state. _experiment_id keeps the string experiment
        # key used by the reward updater; handoff emits the selected Persistent
        # Store row ID via _experiment_row_id.
        self._experiment_id: str = experiment_id
        self._experiment_row_id: int = DEFAULT_EXPERIMENT_ROW_ID
        self._active_arm: str = ""
        self._expected_greeting: str = ""
        self._bandit_decision_snapshot: dict[str, Any] | None = None
        self._segment_window_anchor_utc: datetime | None = None

        # WS3 P2 — IPC dispatch state. The orchestrator keeps the most
        # recent N PcmBlock handles alive so consumers (gpu_ml_worker)
        # can attach to them; the kernel auto-reclaims older blocks
        # when they fall out of the bounded buffer. N=8 covers ~4 min
        # of 30-second segments, far longer than ML inference latency.
        self._ipc_queue: Any = ipc_queue
        self._inflight_blocks: deque[Any] = deque(maxlen=8)

        # Video capture (lazy-init in run() for live capture).
        # Replay mode binds the same source to the video and audio surfaces.
        self.video_capture: Any = replay_source

        # AU12 telemetry accumulator for continuous reward pipeline.
        # Each entry: {"timestamp_s": float, "intensity": float}
        # Drained into the payload on each assemble_segment() call.
        self._au12_series: list[dict[str, float]] = []

        # Stimulus injection timestamp (drift-corrected UTC epoch).
        # Set by record_stimulus_injection(). None until operator sends greeting.
        self._stimulus_time: float | None = None

        # Physiological rolling buffers — §4.C.4.
        # Keep time-ordered chunk events per subject_role for trailing-window derivation.
        self._physio_buffer: dict[str, deque[dict[str, Any]]] = {
            "streamer": deque(),
            "operator": deque(),
        }

        # Redis client for non-blocking physiological drains.
        # Client creation is best-effort; actual I/O happens on LPOP.
        self._redis: Any = None
        try:
            import redis as redis_lib

            self._redis = redis_lib.Redis.from_url(
                os.environ.get("REDIS_URL", "redis://redis:6379/0"),
                decode_responses=True,
            )
        except Exception:
            logger.debug("Physiology Redis client unavailable during init", exc_info=True)

        # Per-session AU12 normalizer — lazy-init on first frame
        self._au12_normalizer: Any = None

        # Calibration state: True until operator injects greeting.
        # While True, AU12Normalizer accumulates the pre-stimulus AU12 baseline (§7A.4).
        self._is_calibrating: bool = True

        # FaceMesh processor — lazy-init on first frame
        self._face_mesh: Any = None

        # Session lifecycle state — boot session starts during run(), but
        # subsequent create/end intents are delivered through Redis pub/sub.
        self._session_active: bool = False
        self._session_lifecycle_received: bool = False
        self._session_lifecycle_queue: Queue[dict[str, Any]] = Queue()
        self._session_lifecycle_stop = threading.Event()
        self._session_lifecycle_thread: threading.Thread | None = None

        # Read-side live-session publish dedupe. The API merges this JSON
        # onto SessionSummary without exposing any raw frame data.
        self._last_live_session_state_payload: str | None = None
        self._last_heartbeat_monotonic: float = 0.0

    def _calibration_frames_accumulated(self) -> int:
        normalizer = self._au12_normalizer
        if normalizer is None:
            return 0
        buffer = getattr(normalizer, "calibration_buffer", None)
        return len(buffer) if isinstance(buffer, list) else 0

    def _live_session_state_payload(self) -> dict[str, Any]:
        accumulated = self._calibration_frames_accumulated()
        required = LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED
        return {
            "active_arm": self._active_arm or None,
            "expected_greeting": self._expected_greeting or None,
            "is_calibrating": self._is_calibrating,
            "calibration_frames_accumulated": accumulated,
            "calibration_frames_required": required,
        }

    def _publish_live_session_state(self) -> None:
        if self._redis is None:
            return
        raw_payload = json.dumps(self._live_session_state_payload(), sort_keys=True)
        if raw_payload == self._last_live_session_state_payload:
            return
        try:
            self._redis.set(
                _live_session_state_key(self._session_id),
                raw_payload,
                ex=LIVE_SESSION_STATE_TTL_S,
            )
        except Exception:
            logger.debug("Live-session state publish unavailable", exc_info=True)
            return
        self._last_live_session_state_payload = raw_payload

    def _publish_orchestrator_heartbeat(self) -> None:
        """Publish a small heartbeat consumed by the operator health probe."""
        if self._redis is None:
            return
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "session_active": self._session_active,
        }
        try:
            self._redis.set(
                ORCHESTRATOR_HEARTBEAT_KEY,
                json.dumps(payload, sort_keys=True),
                ex=ORCHESTRATOR_HEARTBEAT_TTL_S,
            )
        except Exception:
            logger.debug("Orchestrator heartbeat publish unavailable", exc_info=True)

    def _publish_orchestrator_heartbeat_if_due(self, now_monotonic: float) -> None:
        if now_monotonic - self._last_heartbeat_monotonic < ORCHESTRATOR_HEARTBEAT_INTERVAL_S:
            return
        self._publish_orchestrator_heartbeat()
        self._last_heartbeat_monotonic = now_monotonic

    def record_stimulus_injection(self) -> None:
        """
        Record the exact moment the greeting line was sent.

        Must be called by the operator interface when the greeting text
        is injected into the live stream chat. This timestamp anchors
        the stimulus-locked measurement window [+0.5s, +5.0s] used by
        the reward pipeline (services/worker/pipeline/reward.py).

        Also transitions the AU12 normalizer from calibration mode to
        inference mode. Pre-stimulus frames establish the AU12 baseline;
        post-stimulus frames are scored against that baseline.

        §7A.4 — Calibration phase ends at stimulus onset.
        §4.E.1 — Thompson Sampling experiment arm deployment trigger.
        """
        self._stimulus_time = self.drift_corrector.correct_timestamp(time.time())
        self._is_calibrating = False
        self._publish_live_session_state()
        logger.info(
            "Stimulus injected at t=%.3f (arm=%s, greeting='%s')",
            self._stimulus_time,
            self._active_arm,
            self._expected_greeting,
        )

    def _process_video_frame(self) -> None:
        """
        Grab the latest video frame and compute AU12 intensity.

        §4.D.2 — MediaPipe FaceMesh landmark extraction
        §7A.4 — AU12 baseline calibration (pre-stimulus) and scoring (post-stimulus)

        During calibration (before stimulus injection):
          - compute_bounded_intensity(is_calibrating=True) accumulates the AU12 baseline
          - Returned intensity is 0.0 (calibration phase always returns zero)
          - These frames are NOT appended to _au12_series (no value for reward)

        After stimulus injection:
          - compute_bounded_intensity(is_calibrating=False) returns [0.0, 1.0]
          - {timestamp_s, intensity} appended to _au12_series for reward pipeline

        §5.2 — Frame data exists only in volatile memory (numpy arrays).
        """
        if self.video_capture is None:
            return

        frame = mark_data_tier(
            self.video_capture.get_latest_frame(),
            DataTier.TRANSIENT,
            spec_ref="§5.2.1",
            purpose="Decoded video frame in volatile AU12 processing memory",
        )  # §5.2.1 Transient Data
        if frame is None:
            return

        try:
            # Lazy-init FaceMesh processor on first frame
            if self._face_mesh is None:
                from packages.ml_core.face_mesh import FaceMeshProcessor

                self._face_mesh = FaceMeshProcessor()
                logger.info("FaceMesh processor initialized for AU12 pipeline")

            # Lazy-init AU12 normalizer on first frame
            if self._au12_normalizer is None:
                from packages.ml_core.au12 import AU12Normalizer

                self._au12_normalizer = AU12Normalizer()  # α_scale = 6.0 default
                logger.info(
                    "AU12 normalizer initialized (α_scale=%.1f)",
                    self._au12_normalizer.alpha,
                )

            # Convert to numpy array: PyAV VideoFrame has .to_ndarray(),
            # but VideoCapture.get_latest_frame() already returns ndarray.
            # The hasattr guard handles both paths — direct PyAV frames
            # (from _decode_loop) and pre-converted ndarrays (from buffer).
            frame_array = (
                frame.to_ndarray(format="bgr24") if hasattr(frame, "to_ndarray") else frame
            )

            # §4.D.2 — Extract 478-vertex 3D landmarks from BGR frame
            landmarks = self._face_mesh.extract_landmarks(frame_array.copy())
            if landmarks is None:
                # §4.D contract — missing face returns null facial metrics
                return

            # §7A.4 — Compute bounded AU12 intensity in [0.0, 1.0]
            intensity: float = self._au12_normalizer.compute_bounded_intensity(
                landmarks,
                is_calibrating=self._is_calibrating,
            )
            self._publish_live_session_state()

            # Only accumulate post-stimulus frames (calibration returns 0.0
            # and doesn't produce meaningful reward signal)
            if not self._is_calibrating:
                now_utc = self.drift_corrector.correct_timestamp(time.time())
                self._au12_series.append({"timestamp_s": now_utc, "intensity": float(intensity)})

        except Exception as e:
            # §12 — AU12 extraction failure is non-critical (expected on
            # frame drops, partial faces, or pipe jitter). Must not crash
            # the main loop; subsequent frames will retry.
            logger.debug("AU12 frame processing skipped: %s", e)

    def _prune_physio_buffer(self, role: str, now_wall: float | None = None) -> None:
        """Drop retained physiological chunks older than the retention horizon."""
        buffer = self._physio_buffer[role]
        if not buffer:
            return

        wall_now = time.time() if now_wall is None else now_wall
        cutoff = wall_now - PHYSIO_BUFFER_RETENTION_S
        while buffer and buffer[0]["window_end_ts"] < cutoff:
            buffer.popleft()

    def _derive_physio_snapshot(
        self,
        role: str,
        now_wall: float | None = None,
    ) -> dict[str, Any] | None:
        """Derive a scalar physiological snapshot over the trailing derivation window."""
        buffer = self._physio_buffer[role]
        if not buffer:
            return None

        wall_now = time.time() if now_wall is None else now_wall
        self._prune_physio_buffer(role, now_wall=wall_now)
        if not buffer:
            return None

        window_start_ts = wall_now - PHYSIO_DERIVE_WINDOW_S
        overlapping = [chunk for chunk in buffer if chunk["window_end_ts"] >= window_start_ts]
        if not overlapping:
            latest_chunk = buffer[-1]
            freshness_s = max(0.0, wall_now - latest_chunk["window_end_ts"])
            return {
                "rmssd_ms": None,
                "heart_rate_bpm": None,
                "source_timestamp_utc": latest_chunk["window_end_utc"],
                "freshness_s": round(freshness_s, 1),
                "is_stale": freshness_s > PHYSIO_STALENESS_THRESHOLD_S,
                "provider": latest_chunk["provider"],
                "source_kind": latest_chunk["source_kind"],
                "derivation_method": latest_chunk["payload_derivation_method"],
                "window_s": PHYSIO_DERIVE_WINDOW_S,
                "validity_ratio": 0.0,
                "is_valid": False,
            }

        ibi_chunks = [chunk for chunk in overlapping if chunk["source_kind"] == "ibi"]
        session_chunks = [chunk for chunk in overlapping if chunk["source_kind"] == "session"]
        contributing = ibi_chunks or session_chunks
        if not contributing:
            return None

        source_kind = "ibi" if ibi_chunks else "session"
        latest_chunk = max(contributing, key=lambda chunk: chunk["window_end_ts"])
        heart_rate_values = [
            hr
            for chunk in contributing
            for hr in chunk["payload"].get("heart_rate_items_bpm") or []
        ]
        valid_total = sum(
            int(chunk["payload"].get("valid_sample_count", 0)) for chunk in contributing
        )
        expected_total = sum(
            int(chunk["payload"].get("expected_sample_count", 0)) for chunk in contributing
        )
        validity_ratio = valid_total / max(expected_total, 1)
        is_valid = validity_ratio >= PHYSIO_VALIDITY_MIN

        rmssd_ms: float | None = None
        derivation_method = latest_chunk["payload_derivation_method"]
        if source_kind == "ibi":
            ibi_values = [
                ibi for chunk in contributing for ibi in chunk["payload"].get("ibi_ms_items") or []
            ]
            if len(ibi_values) >= 2:
                squared_diffs = [
                    (ibi_values[index + 1] - ibi_values[index]) ** 2
                    for index in range(len(ibi_values) - 1)
                ]
                rmssd_ms = round(sqrt(sum(squared_diffs) / len(squared_diffs)), 3)
                derivation_method = "server"
        else:
            session_rmssd = [
                value
                for chunk in contributing
                for value in chunk["payload"].get("rmssd_items_ms") or []
            ]
            if session_rmssd:
                rmssd_ms = round(sum(session_rmssd) / len(session_rmssd), 3)

        if not is_valid:
            rmssd_ms = None

        freshness_s = max(0.0, wall_now - latest_chunk["window_end_ts"])
        return {
            "rmssd_ms": rmssd_ms,
            "heart_rate_bpm": round(sum(heart_rate_values) / len(heart_rate_values))
            if heart_rate_values
            else None,
            "source_timestamp_utc": latest_chunk["window_end_utc"],
            "freshness_s": round(freshness_s, 1),
            "is_stale": freshness_s > PHYSIO_STALENESS_THRESHOLD_S,
            "provider": latest_chunk["provider"],
            "source_kind": source_kind,
            "derivation_method": derivation_method,
            "window_s": PHYSIO_DERIVE_WINDOW_S,
            "validity_ratio": validity_ratio,
            "is_valid": is_valid,
        }

    def _drain_physio_events(self) -> None:
        """
        §4.C.4 — Drain pending physiological events from Redis.

        Called at the start of each segment assembly cycle. Non-blocking:
        uses LPOP (not BLPOP) and processes at most
        MAX_PHYSIO_DRAIN_PER_CYCLE events to avoid stalling dispatch.

        Valid chunk events are appended to the per-role rolling buffer and
        old retained chunks are pruned by retention horizon.
        """
        if self._redis is None:
            return

        drained = 0
        wall_now = time.time()
        while drained < MAX_PHYSIO_DRAIN_PER_CYCLE:
            try:
                raw = mark_data_tier(
                    self._redis.lpop(PHYSIO_QUEUE_KEY),
                    DataTier.TRANSIENT,
                    spec_ref="§5.2.1",
                    purpose="PhysiologicalChunkEvent JSON while in Redis transit",
                )  # §5.2.1 Transient Data
            except Exception:
                logger.warning("Physiological Redis drain unavailable", exc_info=True)
                return

            if raw is None:
                break
            drained += 1

            try:
                from packages.schemas.physiology import PhysiologicalChunkEvent

                event = PhysiologicalChunkEvent.model_validate_json(raw)
                chunk_record = {
                    "provider": event.provider,
                    "subject_role": event.subject_role,
                    "source_kind": event.source_kind,
                    "window_start_utc": event.window_start_utc.isoformat(),
                    "window_end_utc": event.window_end_utc.isoformat(),
                    "window_start_ts": event.window_start_utc.timestamp(),
                    "window_end_ts": event.window_end_utc.timestamp(),
                    "payload_derivation_method": event.payload.derivation_method,
                    "payload": event.payload.model_dump(mode="json"),
                }
                self._physio_buffer[event.subject_role].append(chunk_record)
                self._prune_physio_buffer(event.subject_role, now_wall=wall_now)
            except Exception:
                logger.warning("Malformed physiological event discarded", exc_info=True)

        if drained > 0:
            logger.debug("Drained %d physiological events from Redis", drained)

        for role in ("streamer", "operator"):
            self._prune_physio_buffer(role, now_wall=wall_now)

    @staticmethod
    def _canonical_utc_timestamp(value: datetime) -> str:
        """Return the canonical UTC string used for stable segment identity."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def _segment_window_for_counter(
        self,
        segment_number: int,
        timestamp_utc: datetime,
    ) -> tuple[datetime, datetime]:
        """Compute deterministic segment window boundaries for a segment ordinal."""
        if self._segment_window_anchor_utc is None:
            self._segment_window_anchor_utc = timestamp_utc - timedelta(
                seconds=SEGMENT_WINDOW_SECONDS
            )

        anchor = self._segment_window_anchor_utc
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=UTC)
        anchor = anchor.astimezone(UTC)
        self._segment_window_anchor_utc = anchor

        start = anchor + timedelta(seconds=(segment_number - 1) * SEGMENT_WINDOW_SECONDS)
        end = start + timedelta(seconds=SEGMENT_WINDOW_SECONDS)
        return start, end

    def _stable_segment_id(self, window_start_utc: datetime, window_end_utc: datetime) -> str:
        """Return SHA-256(session_id|window_start|window_end) using canonical UTC."""
        stable_identity = "|".join(
            (
                f"{uuid.UUID(self._session_id)}",
                self._canonical_utc_timestamp(window_start_utc),
                self._canonical_utc_timestamp(window_end_utc),
            )
        )
        return hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()

    def _lookup_selected_experiment_row_id(self, store: Any, arm: str) -> int:
        """Best-effort lookup of the Persistent Store row ID for the selected arm."""
        if not (hasattr(store, "_get_conn") and hasattr(store, "_put_conn")):
            return DEFAULT_EXPERIMENT_ROW_ID

        conn: Any | None = None
        try:
            conn = store._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id FROM experiments
                    WHERE experiment_id = %(experiment_id)s
                      AND arm = %(arm)s
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    {"experiment_id": self._experiment_id, "arm": arm},
                )
                row = cur.fetchone()
            if row is None:
                return DEFAULT_EXPERIMENT_ROW_ID
            if isinstance(row, dict):
                return int(row.get("id") or DEFAULT_EXPERIMENT_ROW_ID)
            return int(row[0])
        except Exception:
            logger.debug(
                "Unable to resolve experiment row ID for experiment=%s arm=%s",
                self._experiment_id,
                arm,
                exc_info=True,
            )
            return DEFAULT_EXPERIMENT_ROW_ID
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    store._put_conn(conn)

    def _decision_context_hash(
        self,
        *,
        candidate_arm_ids: list[str],
        posterior_by_arm: dict[str, dict[str, float]],
        selected_arm_id: str,
    ) -> str:
        """Hash the stable pre-update selection context for attribution linkage."""
        context = {
            "experiment_code": self._experiment_id,
            "experiment_row_id": self._experiment_row_id,
            "candidate_arm_ids": candidate_arm_ids,
            "posterior_by_arm": posterior_by_arm,
            "selected_arm_id": selected_arm_id,
            "policy_version": BANDIT_POLICY_VERSION,
        }
        encoded = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _bandit_random_seed(
        self,
        *,
        segment_window_start_utc: datetime,
        stimulus_time: float | None,
    ) -> int:
        seed_material = "".join(
            (
                f"{uuid.UUID(self._session_id)}",
                self._canonical_utc_timestamp(segment_window_start_utc),
                str(stimulus_time),
                BANDIT_POLICY_VERSION,
            )
        )
        return int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")

    def _capture_bandit_decision_snapshot(
        self,
        *,
        selection_time_utc: datetime,
        segment_window_start_utc: datetime,
        candidate_arm_ids: list[str],
        posterior_by_arm: dict[str, dict[str, float]],
        sampled_theta_by_arm: dict[str, float] | None,
        random_seed: int | None = None,
    ) -> None:
        """Capture the pre-update Thompson Sampling BanditDecisionSnapshot.

        The caller invokes this immediately after arm selection. Copy every
        mutable selection input before storing it so later posterior updates,
        test mutations, or store-side object reuse cannot alter the pre-update
        evidence carried on the handoff payload.
        """
        if not self._active_arm:
            self._active_arm = "simple_hello"
        if not self._expected_greeting:
            self._expected_greeting = GREETING_LINES.get(self._active_arm, DEFAULT_GREETING_TEXT)

        normalized_candidates = [str(arm_id) for arm_id in candidate_arm_ids]
        if not normalized_candidates:
            normalized_candidates = [self._active_arm]
        if self._active_arm not in normalized_candidates:
            normalized_candidates.append(self._active_arm)
        normalized_candidates = list(dict.fromkeys(normalized_candidates))

        posterior_copy: dict[str, dict[str, float]] = {}
        for arm_id, posterior in posterior_by_arm.items():
            posterior_copy[str(arm_id)] = {
                "alpha": float(posterior["alpha"]),
                "beta": float(posterior["beta"]),
            }
        if self._active_arm not in posterior_copy:
            posterior_copy[self._active_arm] = {"alpha": 1.0, "beta": 1.0}

        resolved_random_seed = random_seed
        if resolved_random_seed is None:
            resolved_random_seed = self._bandit_random_seed(
                segment_window_start_utc=segment_window_start_utc,
                stimulus_time=self._stimulus_time,
            )

        normalized_sampled_theta_by_arm = {
            str(arm_id): float(theta) for arm_id, theta in (sampled_theta_by_arm or {}).items()
        }

        snapshot: dict[str, Any] = {
            "selection_method": "thompson_sampling",
            "selection_time_utc": selection_time_utc,
            "experiment_id": int(self._experiment_row_id),
            "policy_version": BANDIT_POLICY_VERSION,
            "selected_arm_id": self._active_arm,
            "candidate_arm_ids": normalized_candidates,
            "posterior_by_arm": posterior_copy,
            "sampled_theta_by_arm": normalized_sampled_theta_by_arm,
            "expected_greeting": self._expected_greeting,
            "decision_context_hash": self._decision_context_hash(
                candidate_arm_ids=normalized_candidates,
                posterior_by_arm=posterior_copy,
                selected_arm_id=self._active_arm,
            ),
            "random_seed": int(resolved_random_seed),
        }

        self._bandit_decision_snapshot = snapshot

    def _ensure_bandit_decision_snapshot(
        self,
        *,
        selection_time_utc: datetime,
        segment_window_start_utc: datetime,
    ) -> None:
        """Populate a fallback snapshot when tests assemble without live arm selection."""
        if self._bandit_decision_snapshot is not None:
            self._bandit_decision_snapshot = {
                **self._bandit_decision_snapshot,
                "sampled_theta_by_arm": {
                    str(arm_id): float(theta)
                    for arm_id, theta in (
                        self._bandit_decision_snapshot.get("sampled_theta_by_arm") or {}
                    ).items()
                },
            }
            return
        fallback_arm = self._active_arm or "simple_hello"
        self._active_arm = fallback_arm
        self._expected_greeting = self._expected_greeting or GREETING_LINES.get(
            fallback_arm,
            DEFAULT_GREETING_TEXT,
        )
        self._capture_bandit_decision_snapshot(
            selection_time_utc=selection_time_utc,
            segment_window_start_utc=segment_window_start_utc,
            candidate_arm_ids=[fallback_arm],
            posterior_by_arm={fallback_arm: {"alpha": 1.0, "beta": 1.0}},
            sampled_theta_by_arm=None,
        )

    def assemble_segment(
        self,
        audio_data: bytes,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Assemble a 30-second segment as a fully validated InferenceHandoffPayload.

        §2 step 5 — Validates against Pydantic schema before returning.
        §6.1 — InferenceHandoffPayload JSON Schema Draft 07 contract.
        §7E/§11.3 — Stable segment_id = SHA-256 over session/window identity.
        """
        from packages.schemas.inference_handoff import InferenceHandoffPayload
        from services.worker.pipeline.serialization import sanitize_json_payload

        assembly_started = time.perf_counter()
        self._segment_counter += 1
        segment_number = self._segment_counter
        # Callers have already drained audio/events for this attempted window.
        # Consume AU12 before validation so invalid payloads cannot leak forward.
        au12_series = list(self._au12_series)
        self._au12_series.clear()

        # §4.C.1 — Apply drift correction to current timestamp.
        now_utc = self.drift_corrector.correct_timestamp(time.time())
        timestamp = datetime.fromtimestamp(now_utc, tz=UTC)
        segment_window_start_utc, segment_window_end_utc = self._segment_window_for_counter(
            segment_number,
            timestamp,
        )
        segment_id = self._stable_segment_id(
            segment_window_start_utc,
            segment_window_end_utc,
        )

        # §2 step 5 — Build segment dict for the segments array.
        segment_data: dict[str, Any] = {
            "segment_id": segment_id,
            "audio_bytes": len(audio_data),
            "events": events,
        }

        # Determine codec based on video availability.
        has_video = self.video_capture is not None and getattr(
            self.video_capture,
            "is_running",
            False,
        )
        codec = "h264" if has_video else "raw"
        resolution = [1920, 1080] if has_video else [1, 1]

        # Extract latest video frame before building the handoff payload.
        frame_data: bytes | None = None
        if self.video_capture is not None:
            try:
                frame = mark_data_tier(
                    self.video_capture.get_latest_frame(),
                    DataTier.TRANSIENT,
                    spec_ref="§5.2.1",
                    purpose="Decoded video frame in volatile segment assembly memory",
                )  # §5.2.1 Transient Data
                if frame is not None:
                    frame_array = (
                        frame.to_ndarray(format="bgr24") if hasattr(frame, "to_ndarray") else frame
                    )
                    frame_data = frame_array.tobytes()
                    resolution = [frame_array.shape[1], frame_array.shape[0]]
            except Exception:
                logger.warning("Assemble segment frame extraction failed", exc_info=True)

        self._ensure_bandit_decision_snapshot(
            selection_time_utc=timestamp,
            segment_window_start_utc=segment_window_start_utc,
        )

        physiological_context: dict[str, Any] | None = None
        now_wall = time.time()
        if any(self._physio_buffer[role] for role in ("streamer", "operator")):
            context = {
                "streamer": self._derive_physio_snapshot("streamer", now_wall=now_wall),
                "operator": self._derive_physio_snapshot("operator", now_wall=now_wall),
            }
            if any(snapshot is not None for snapshot in context.values()):
                physiological_context = context

        payload_data: dict[str, Any] = {
            "session_id": uuid.UUID(self._session_id),
            "segment_id": segment_id,
            "segment_window_start_utc": segment_window_start_utc,
            "segment_window_end_utc": segment_window_end_utc,
            "timestamp_utc": timestamp,
            "media_source": {
                "stream_url": self._stream_url or DEFAULT_MEDIA_SOURCE_URI,
                "codec": codec,
                "resolution": resolution,
            },
            "segments": [segment_data],
            "_active_arm": self._active_arm,
            "_experiment_id": int(self._experiment_row_id),
            "_expected_greeting": self._expected_greeting,
            "_stimulus_time": self._stimulus_time,
            "_au12_series": au12_series,
            "_bandit_decision_snapshot": self._bandit_decision_snapshot,
        }
        if physiological_context is not None:
            payload_data["_physiological_context"] = physiological_context

        payload = InferenceHandoffPayload.model_validate(sanitize_json_payload(payload_data))
        result: dict[str, Any] = payload.model_dump(mode="json", by_alias=True)

        # Transport-only binary and reward-wiring fields are added after schema
        # validation so they do not weaken InferenceHandoffPayload.extra='forbid'.
        result["_audio_data"] = audio_data
        result["_frame_data"] = frame_data
        result["_experiment_code"] = self._experiment_id

        result = sanitize_json_payload(result)
        # WS3 P2: the v3.4 base64 round-trip on _audio_data / _frame_data is
        # retired. The desktop IPC path moves audio through SharedMemory
        # (services.desktop_app.ipc.shared_buffers) and frames stay
        # process-local per the v4.0 §9 invariant that decoded video
        # frames never cross a process boundary.
        logger.info(
            "BENCHMARK segment_assembly_ms=%.3f segment_id=%s",
            (time.perf_counter() - assembly_started) * 1000.0,
            segment_id,
        )

        return result

    def _dispatch_payload(self, payload: dict[str, Any]) -> None:
        """Validate the typed handoff and dispatch via v4.0 IPC.

        WS3 P2 retires the v3.4 Celery + Redis dispatch path. The 30 s
        PCM window travels via ``shared_buffers.write_pcm_block``; its
        SharedMemory metadata is wrapped into an
        ``InferenceControlMessage`` and pushed onto the IPC queue. The
        ``_audio_data`` base64 round-trip is dropped on the desktop
        path; ``sanitize_json_payload`` is retained because it also
        prunes empty ``_physiological_context`` and absent
        ``_bandit_decision_snapshot`` optionals.
        """
        try:
            from packages.schemas.inference_handoff import InferenceHandoffPayload
            from services.desktop_app.ipc.control_messages import (
                AudioBlockRef,
                InferenceControlMessage,
            )
            from services.desktop_app.ipc.shared_buffers import write_pcm_block
            from services.worker.pipeline.serialization import sanitize_json_payload

            transport_fields = ("_audio_data", "_frame_data", "_experiment_code")
            transport_payload = {key: payload[key] for key in transport_fields if key in payload}
            handoff_data = {
                key: value for key, value in payload.items() if key not in transport_fields
            }

            validated = InferenceHandoffPayload.model_validate(sanitize_json_payload(handoff_data))
            handoff_dump: dict[str, Any] = sanitize_json_payload(
                validated.model_dump(mode="json", by_alias=True)
            )

            audio = transport_payload.pop("_audio_data", None)
            # WS3 P2 / v4.0 §9 invariant: decoded video frames never cross
            # a process boundary. Drop _frame_data unconditionally — it
            # was a v3.4 Celery-path artifact and is observed nowhere on
            # the desktop graph.
            transport_payload.pop("_frame_data", None)
            if not isinstance(audio, bytes | bytearray) or not audio:
                logger.warning(
                    "dispatch skipped: missing or empty _audio_data on segment %s",
                    handoff_dump.get("segment_id"),
                )
                return

            if self._ipc_queue is None:
                logger.warning(
                    "dispatch skipped: no IPC queue configured on Orchestrator (segment_id=%s)",
                    handoff_dump.get("segment_id"),
                )
                return

            block = write_pcm_block(bytes(audio))
            evicted = self._track_block(block)
            if evicted is not None:
                evicted.close_and_unlink()

            msg = InferenceControlMessage(
                handoff=handoff_dump,
                audio=AudioBlockRef.from_metadata(block.metadata),
                forward_fields=transport_payload,
            )
            self._ipc_queue.put(msg.model_dump(mode="json"))
        except Exception as exc:
            logger.error("Failed to dispatch segment via IPC: %s", exc)

    def _track_block(self, block: Any) -> Any | None:
        """Append ``block`` to the bounded inflight buffer; return any eviction."""
        evicted: Any | None = None
        if (
            self._inflight_blocks.maxlen is not None
            and len(self._inflight_blocks) == self._inflight_blocks.maxlen
        ):
            evicted = self._inflight_blocks[0]
        self._inflight_blocks.append(block)
        return evicted

    def close_inflight_blocks(self) -> None:
        """Release every retained PcmBlock handle. Idempotent."""
        while self._inflight_blocks:
            block = self._inflight_blocks.popleft()
            try:
                block.close_and_unlink()
            except Exception:  # noqa: BLE001
                logger.debug("inflight block cleanup failed", exc_info=True)

    def _drain_event_buffer(self) -> list[dict[str, Any]]:
        """Drain queued ground-truth events into the current segment payload."""
        events: list[dict[str, Any]] = []
        while self.event_buffer:
            events.append(self.event_buffer.popleft())
        return events

    def _flush_inflight_segment(self) -> None:
        """Flush the current partial segment before a session transitions."""
        has_inflight_state = bool(self._audio_buffer or self.event_buffer or self._au12_series)
        if not has_inflight_state:
            self._audio_buffer.clear()
            self.event_buffer.clear()
            return

        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        events = self._drain_event_buffer()
        self._drain_physio_events()
        candidate_segment_number = self._segment_counter + 1
        try:
            payload = self.assemble_segment(audio_data, events)
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Discarding invalid assembled handoff segment source=flush_inflight "
                "session_id=%s segment_number=%d segment_counter=%d "
                "window_anchor_utc=%s audio_bytes=%d event_count=%d error=%s",
                self._session_id,
                candidate_segment_number,
                self._segment_counter,
                self._segment_window_anchor_utc,
                len(audio_data),
                len(events),
                exc,
                exc_info=True,
            )
            return

        self._dispatch_payload(payload)

    def _reset_session_state(self) -> None:
        """Reset per-session accumulators without restarting capture surfaces."""
        self._segment_counter = 0
        self._audio_buffer.clear()
        self.event_buffer.clear()
        self._au12_series.clear()
        self._stimulus_time = None
        self._active_arm = ""
        self._expected_greeting = ""
        self._experiment_row_id = DEFAULT_EXPERIMENT_ROW_ID
        self._bandit_decision_snapshot = None
        self._segment_window_anchor_utc = None
        self._is_calibrating = True
        self._au12_normalizer = None

    def _begin_session(self, *, session_id: str, stream_url: str, experiment_id: str) -> None:
        """Make a session active and register it authoritatively in Postgres."""
        self._reset_session_state()
        self._session_id = session_id
        self._stream_url = stream_url
        self._experiment_id = experiment_id or "greeting_line_v1"
        self._segment_window_anchor_utc = datetime.fromtimestamp(
            self.drift_corrector.correct_timestamp(time.time()),
            tz=UTC,
        )
        self._session_active = True
        self._register_session()
        self._select_experiment_arm()
        self._publish_live_session_state()
        self._publish_orchestrator_heartbeat()
        logger.info(
            "Session lifecycle started: session_id=%s stream_url=%s experiment_id=%s",
            self._session_id,
            self._stream_url or "(none)",
            self._experiment_id,
        )

    def _mark_session_ended(self, session_id: str) -> None:
        """Persist an authoritative `ended_at` timestamp for the session."""
        update_session_sql = """
            UPDATE sessions
            SET ended_at = NOW()
            WHERE session_id = %(session_id)s
              AND ended_at IS NULL
        """

        try:
            import psycopg2

            conn = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST", "postgres"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                dbname=os.environ["POSTGRES_DB"],
            )
            try:
                with conn.cursor() as cur:
                    cur.execute(update_session_sql, {"session_id": session_id})
                conn.commit()
                logger.info("Session marked ended in Persistent Store: %s", session_id)
            finally:
                conn.close()
        except Exception as exc:
            logger.error(
                "Failed to mark session ended %s: %s",
                session_id,
                exc,
                exc_info=True,
            )

    def _finish_active_session(self, *, flush: bool) -> None:
        """Flush and close the currently active session, if any."""
        if not self._session_active:
            return

        session_id = self._session_id
        if flush:
            self._flush_inflight_segment()
        self._mark_session_ended(session_id)
        self._session_active = False
        self._reset_session_state()
        self._publish_live_session_state()
        self._publish_orchestrator_heartbeat()
        logger.info("Session lifecycle ended: session_id=%s", session_id)

    def _handle_session_lifecycle_intent(self, intent: dict[str, Any]) -> None:
        """Apply a queued lifecycle intent published by the API Server."""
        action = str(intent.get("action") or "").strip().lower()
        session_id = str(intent.get("session_id") or "").strip()
        if action not in {"start", "end"} or not session_id:
            logger.warning("Malformed session lifecycle intent ignored: %r", intent)
            return

        self._session_lifecycle_received = True

        if action == "start":
            if session_id == self._session_id:
                logger.info("Duplicate/stale session start ignored: %s", session_id)
                return
            if self._session_active:
                self._finish_active_session(flush=True)
            self._begin_session(
                session_id=session_id,
                stream_url=str(intent.get("stream_url") or ""),
                experiment_id=str(intent.get("experiment_id") or self._experiment_id),
            )
            return

        if not self._session_active:
            logger.info("Session end ignored; no active session is running")
            return
        if session_id != self._session_id:
            logger.info(
                "Session end ignored for non-active session_id=%s (active=%s)",
                session_id,
                self._session_id,
            )
            return
        self._finish_active_session(flush=True)

    def _drain_session_lifecycle_intents(self) -> None:
        """Apply all queued lifecycle intents without blocking the main loop."""
        while True:
            try:
                intent = self._session_lifecycle_queue.get_nowait()
            except Empty:
                break
            try:
                self._handle_session_lifecycle_intent(intent)
            except Exception:
                logger.warning("Session lifecycle intent failed", exc_info=True)

    def _start_session_lifecycle_listener(self) -> None:
        """Listen for API-published session lifecycle JSON on Redis pub/sub."""
        if self._session_lifecycle_thread is not None and self._session_lifecycle_thread.is_alive():
            return

        redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        self._session_lifecycle_stop.clear()

        def _listen() -> None:
            client: Any = None
            pubsub: Any = None
            try:
                import redis

                client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                )
                pubsub = client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(SESSION_LIFECYCLE_CHANNEL)
                logger.info(
                    "Redis session lifecycle listener subscribed to '%s'",
                    SESSION_LIFECYCLE_CHANNEL,
                )

                while not self._session_lifecycle_stop.is_set():
                    message = pubsub.get_message(timeout=SESSION_LIFECYCLE_POLL_TIMEOUT_S)
                    if message is None:
                        continue
                    if message.get("type") != "message":
                        continue

                    raw_message = message.get("data")
                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode("utf-8")

                    try:
                        intent = json.loads(str(raw_message))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Malformed session lifecycle payload ignored: %r",
                            raw_message,
                        )
                        continue

                    if not isinstance(intent, dict):
                        logger.warning("Unexpected session lifecycle payload ignored: %r", intent)
                        continue
                    self._session_lifecycle_queue.put(intent)
            except Exception:
                logger.warning("Redis session lifecycle listener unavailable", exc_info=True)
            finally:
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        pubsub.unsubscribe(SESSION_LIFECYCLE_CHANNEL)
                    with contextlib.suppress(Exception):
                        pubsub.close()
                if client is not None and hasattr(client, "close"):
                    with contextlib.suppress(Exception):
                        client.close()

        thread = threading.Thread(
            target=_listen,
            name="session-lifecycle-listener",
            daemon=True,
        )
        thread.start()
        self._session_lifecycle_thread = thread

    def stop(self) -> None:
        """Stop the orchestration loop and clean up all resources."""
        self._running = False
        self._session_lifecycle_stop.set()
        if (
            self._session_lifecycle_thread is not None
            and self._session_lifecycle_thread.is_alive()
            and self._session_lifecycle_thread is not threading.current_thread()
        ):
            with contextlib.suppress(Exception):
                self._session_lifecycle_thread.join(timeout=SESSION_LIFECYCLE_POLL_TIMEOUT_S + 1.0)
        self.audio_resampler.stop()
        # Stop video capture thread
        if self.video_capture is not None and self.video_capture is not self.audio_resampler:
            with contextlib.suppress(Exception):
                self.video_capture.stop()
        # Release Redis client used for physiological drain.
        if self._redis is not None and hasattr(self._redis, "close"):
            with contextlib.suppress(Exception):
                self._redis.close()
            self._redis = None
        # Release FaceMesh resources
        if self._face_mesh is not None:
            with contextlib.suppress(Exception):
                self._face_mesh.close()
            self._face_mesh = None
        # Clear AU12 state
        self._au12_normalizer = None
        self._au12_series.clear()

    async def run(self) -> None:
        """
        Main orchestration loop for lifecycle-aware session ownership.

        §4.C — Coordinates all Module C responsibilities:
        0. Preserve boot-time session auto-create from STREAM_URL/EXPERIMENT_ID.
        0b. Start background Redis lifecycle listener for authoritative start/end.
        0c. Start video capture thread.
        1. Poll drift every DRIFT_POLL_INTERVAL seconds (§4.C.1).
        2. Continuously read resampled audio chunks (§4.C.2).
        2b. Process latest video frame → AU12 accumulation.
        3. Assemble fixed 30-second segments only while a session is active.
        4. Flush/rotate sessions on lifecycle intents from the API Server.
        """
        # --- Pre-loop initialization ---
        self._running = True
        self._start_session_lifecycle_listener()

        # Start the configured boot session until/unless lifecycle messages arrive.
        self._begin_session(
            session_id=self._session_id,
            stream_url=self._stream_url,
            experiment_id=self._experiment_id,
        )
        self._publish_live_session_state()
        self._publish_orchestrator_heartbeat()

        # Start video capture from IPC Pipe unless replay is opt-in.
        if self._using_replay_capture:
            logger.info("Using replay capture source instead of live IPC pipes")
        else:
            try:
                from services.worker.pipeline.video_capture import VideoCapture

                self.video_capture = VideoCapture()
                self.video_capture.start()
            except Exception as exc:
                logger.warning("Video capture unavailable: %s", exc)
                self.video_capture = None

        self.audio_resampler.start()

        # §4.C.2 — 16 kHz, mono, s16le = 2 bytes/sample = 32000 bytes/second
        bytes_per_second = 16000 * 2  # 16 kHz * 2 bytes (s16le)
        segment_bytes = bytes_per_second * SEGMENT_WINDOW_SECONDS
        # Read audio in 1/30th second chunks to match 30 FPS video
        chunk_size = int(bytes_per_second / 30)

        while self._running:
            self._drain_session_lifecycle_intents()
            now = time.monotonic()
            self._publish_orchestrator_heartbeat_if_due(now)

            # §4.C.1 / WS3 P3 — Drift correction is polled by
            # services.desktop_app.processes.capture_supervisor and
            # delivered to this process over IpcChannels.drift_updates;
            # module_c_orchestrator drains the queue and updates
            # self.drift_corrector.drift_offset directly. In replay
            # mode this loop runs in a unit-test process without a
            # supervisor, so the offset stays at its zero default.

            # §4.C.2 — Read resampled audio chunk. When no session is active we
            # still drain the pipe, but intentionally discard the bytes so a later
            # start intent begins from a clean lifecycle boundary.
            chunk = self.audio_resampler.read_chunk(chunk_size)
            if chunk and self._session_active:
                self._audio_buffer.extend(chunk)

            # §12 Worker crash C — Self-heal video capture thread.
            # PyAV crashes silently if the IPC Pipe drops a keyframe.
            # If the thread dies, automatically revive it so we don't lose telemetry.
            video_dead = self.video_capture is None or not getattr(
                self.video_capture,
                "is_running",
                False,
            )
            if not self._using_replay_capture and video_dead:
                try:
                    from services.worker.pipeline.video_capture import VideoCapture

                    if self.video_capture is not None:
                        with contextlib.suppress(Exception):
                            self.video_capture.stop()
                    self.video_capture = VideoCapture()
                    self.video_capture.start()
                    logger.info("Video capture thread revived after pipe crash")
                except Exception:
                    # §12 Worker crash — video revival is best-effort; if it
                    # fails, the next loop iteration will retry.
                    logger.debug("Video revival failed", exc_info=True)

            # §4.D.2 + §7A.4 — Process latest video frame for AU12 only
            # while a session is actively running.
            if self._session_active:
                self._process_video_frame()

            # §2 step 5 — When we have 30s of audio for an active session,
            # assemble and dispatch a segment.
            if self._session_active and len(self._audio_buffer) >= segment_bytes:
                audio_data = bytes(self._audio_buffer[:segment_bytes])
                self._audio_buffer = bytearray(self._audio_buffer[segment_bytes:])
                events = self._drain_event_buffer()
                self._drain_physio_events()
                candidate_segment_number = self._segment_counter + 1
                try:
                    payload = self.assemble_segment(audio_data, events)
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Discarding invalid assembled handoff segment source=run_dispatch "
                        "session_id=%s segment_number=%d segment_counter=%d "
                        "window_anchor_utc=%s audio_bytes=%d event_count=%d error=%s",
                        self._session_id,
                        candidate_segment_number,
                        self._segment_counter,
                        self._segment_window_anchor_utc,
                        len(audio_data),
                        len(events),
                        exc,
                        exc_info=True,
                    )
                else:
                    self._dispatch_payload(payload)

            # Yield control to event loop
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # Session registration and experiment arm selection
    # ------------------------------------------------------------------

    def _register_session(self) -> None:
        """
        Register this session in the Persistent Store.

        §2 step 7 — Parameterized INSERT with ON CONFLICT guard
        (idempotent in case of worker restart with same session_id).

        Must execute synchronously before the first segment dispatch.
        Uses a direct psycopg2 connection (not the MetricsStore pool)
        to avoid circular dependency with the Celery task layer.
        """
        insert_session_sql = mark_data_tier(
            """
            INSERT INTO sessions (session_id, stream_url, started_at)
            VALUES (%(session_id)s, %(stream_url)s, NOW())
            ON CONFLICT (session_id) DO NOTHING
        """,
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
            purpose="Authoritative session metadata INSERT",
        )  # §5.2.3 Permanent Analytical Storage

        try:
            import psycopg2

            conn = psycopg2.connect(
                host=os.environ.get("POSTGRES_HOST", "postgres"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                dbname=os.environ["POSTGRES_DB"],
            )
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_session_sql,
                        mark_data_tier(
                            {
                                "session_id": self._session_id,
                                "stream_url": self._stream_url or "unknown",
                            },
                            DataTier.PERMANENT,
                            spec_ref="§5.2.3",
                            purpose="Normalized session metadata row parameters",
                        ),
                    )
                conn.commit()
                logger.info(
                    "Session registered in Persistent Store: %s",
                    self._session_id,
                )
            finally:
                conn.close()
        except Exception as exc:
            # §12.1 Module E — log but don't abort; persist_metrics will
            # also fail with FK violation, but the buffer/CSV fallback
            # will capture the data. The operator can manually create
            # the session row if needed.
            logger.error(
                "Failed to register session %s: %s",
                self._session_id,
                exc,
                exc_info=True,
            )

    def _select_experiment_arm(self) -> None:
        """
        §4.E.1 — Select the active greeting line via Thompson Sampling.

        Captures the pre-update BanditDecisionSnapshot at arm-selection
        time while keeping the string experiment key available for the
        §7B reward update path.
        """
        selection_time = datetime.fromtimestamp(
            self.drift_corrector.correct_timestamp(time.time()),
            tz=UTC,
        )
        segment_window_start_utc = self._segment_window_anchor_utc or selection_time
        if segment_window_start_utc.tzinfo is None:
            segment_window_start_utc = segment_window_start_utc.replace(tzinfo=UTC)
        segment_window_start_utc = segment_window_start_utc.astimezone(UTC)
        random_seed = self._bandit_random_seed(
            segment_window_start_utc=segment_window_start_utc,
            stimulus_time=self._stimulus_time,
        )

        try:
            import numpy as np

            from services.worker.pipeline.analytics import MetricsStore

            store = MetricsStore()
            store.connect()
            try:
                arms = sorted(
                    store.get_experiment_arms(self._experiment_id),
                    key=lambda arm_data: str(arm_data["arm"]),
                )
                if not arms:
                    raise ValueError(f"No arms found for experiment '{self._experiment_id}'")

                posterior_by_arm: dict[str, dict[str, float]] = {}
                sampled_theta_by_arm: dict[str, float] = {}
                selected_arm_data: dict[str, Any] | None = None
                best_sample = -1.0
                rng = np.random.Generator(np.random.PCG64(random_seed))

                for arm_data in arms:
                    arm_id = str(arm_data["arm"])
                    alpha = float(arm_data["alpha_param"])
                    beta_param = float(arm_data["beta_param"])
                    posterior_by_arm[arm_id] = {"alpha": alpha, "beta": beta_param}
                    sample = float(rng.beta(alpha, beta_param))
                    sampled_theta_by_arm[arm_id] = sample
                    if sample > best_sample:
                        best_sample = sample
                        selected_arm_data = arm_data

                if selected_arm_data is None:
                    raise ValueError(
                        f"No selectable arms found for experiment '{self._experiment_id}'"
                    )

                self._active_arm = str(selected_arm_data["arm"])
                self._expected_greeting = str(
                    selected_arm_data.get("greeting_text")
                    or GREETING_LINES.get(self._active_arm, DEFAULT_GREETING_TEXT)
                )

                row_id = selected_arm_data.get("id") or selected_arm_data.get("experiment_row_id")
                self._experiment_row_id = (
                    int(row_id)
                    if row_id is not None
                    else self._lookup_selected_experiment_row_id(store, self._active_arm)
                )
                candidate_arm_ids = list(posterior_by_arm)
                self._capture_bandit_decision_snapshot(
                    selection_time_utc=selection_time,
                    segment_window_start_utc=segment_window_start_utc,
                    candidate_arm_ids=candidate_arm_ids,
                    posterior_by_arm=posterior_by_arm,
                    sampled_theta_by_arm=sampled_theta_by_arm,
                    random_seed=random_seed,
                )

                logger.info(
                    "Thompson Sampling selected arm '%s' for session %s: \"%s\"",
                    self._active_arm,
                    self._session_id,
                    self._expected_greeting,
                )
            finally:
                store.close()
        except Exception as exc:
            # Fallback: use the stable default greeting if TS unavailable.
            self._active_arm = "simple_hello"
            self._expected_greeting = GREETING_LINES["simple_hello"]
            self._experiment_row_id = DEFAULT_EXPERIMENT_ROW_ID
            self._capture_bandit_decision_snapshot(
                selection_time_utc=selection_time,
                segment_window_start_utc=segment_window_start_utc,
                candidate_arm_ids=[self._active_arm],
                posterior_by_arm={self._active_arm: {"alpha": 1.0, "beta": 1.0}},
                sampled_theta_by_arm=None,
                random_seed=random_seed,
            )
            logger.warning(
                "Thompson Sampling unavailable, using fallback arm: %s",
                exc,
            )
