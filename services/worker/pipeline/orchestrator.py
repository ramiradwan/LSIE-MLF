"""
Orchestration & Synchronization — §4.C Module C

Aligns timestamps between device hardware clocks, IPC media streams,
and external WebSocket events. Manages audio resampling pipeline
and segment assembly for handoff to Module D.

v3.0 additions:
  - Per-frame AU12 accumulation via compute_bounded_intensity() (§7A.4)
  - Stimulus injection timestamping for reward window anchoring (§4.E.1)
  - Calibration lifecycle: pre-stimulus B_neutral → post-stimulus scoring
  - Payload wiring: _au12_series, _stimulus_time, _x_max fields
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from math import sqrt
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)

# §4.C.1 Drift polling specification
DRIFT_POLL_INTERVAL: int = 30  # seconds
MAX_TOLERATED_DRIFT_MS: int = 150  # milliseconds
ADB_COMMAND: str = "adb shell 'echo $EPOCHREALTIME'"
DRIFT_FREEZE_AFTER_FAILURES: int = 3
DRIFT_RESET_TIMEOUT: int = 300  # 5 minutes in seconds

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


class DriftCorrector:
    """
    §4.C.1 — Temporal drift correction.

    Polls Android hardware clock via ADB every 30 seconds, computes
    drift_offset = host_utc_time - android_epoch_time, and applies
    correction to all timestamps.

    Fallback: freeze drift_offset after 3 ADB failures; reset to
    zero after 5 minutes.
    """

    def __init__(self) -> None:
        self.drift_offset: float = 0.0
        self._consecutive_failures: int = 0
        self._frozen: bool = False
        self._frozen_at: float = 0.0

    def poll(self) -> float:
        """
        Execute ADB epoch poll and update drift_offset.

        §4.C.1 — drift_offset = host_utc - android_epoch.
        §12 Hardware loss C: freeze drift after 3 failures; reset to
        zero after 5 minutes of frozen state.

        Returns:
            Current drift_offset in seconds.
        """
        # §12 Hardware loss C — check if frozen drift should reset
        if self._frozen:
            elapsed = time.monotonic() - self._frozen_at
            if elapsed >= DRIFT_RESET_TIMEOUT:
                # §12 — reset to zero after 5 minutes
                logger.warning("Drift frozen for %ds, resetting to zero", int(elapsed))
                self.drift_offset = 0.0
                self._frozen = False
                self._consecutive_failures = 0
            return self.drift_offset

        try:
            # §4.C.1 — Execute ADB epoch command
            host_utc = time.time()
            result = subprocess.run(
                ADB_COMMAND,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ADB returned code {result.returncode}")

            android_epoch = float(result.stdout.strip())

            # §4.C.1 — drift_offset = host_utc - android_epoch
            self.drift_offset = host_utc - android_epoch
            self._consecutive_failures = 0

            drift_ms = abs(self.drift_offset * 1000)
            if drift_ms > MAX_TOLERATED_DRIFT_MS:
                logger.warning(
                    "Drift %.1fms exceeds %dms tolerance",
                    drift_ms,
                    MAX_TOLERATED_DRIFT_MS,
                )

        except Exception as exc:
            self._consecutive_failures += 1
            logger.error(
                "ADB poll failed (%d/%d): %s",
                self._consecutive_failures,
                DRIFT_FREEZE_AFTER_FAILURES,
                exc,
            )

            # §12 Hardware loss C — freeze after 3 failures
            if self._consecutive_failures >= DRIFT_FREEZE_AFTER_FAILURES and not self._frozen:
                logger.warning("Freezing drift at %.6f", self.drift_offset)
                self._frozen = True
                self._frozen_at = time.monotonic()

        return self.drift_offset

    def correct_timestamp(self, original_ts: float) -> float:
        """Apply drift correction: corrected_ts = original_ts + drift_offset."""
        return original_ts + self.drift_offset


class AudioResampler:
    """
    §4.C.2 — FFmpeg audio resampling subprocess.

    Continuously resamples audio from 48 kHz → 16 kHz via pipe:1.
    Spawns persistent FFmpeg subprocess (§4.C contract side effects).

    Error: FFmpeg crashes restart automatically within 1 second (§2 step 3).
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
            data = self._process.stdout.read(num_bytes)
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
                self._process = None  # type: ignore[assignment]
            logger.info("FFmpeg resampler stopped")


class Orchestrator:
    """
    §4.C — Main orchestration loop.

    Coordinates drift correction, audio resampling, event buffering,
    and segment assembly into InferenceHandoffPayload objects for
    dispatch to Module D via process_segment().

    Segment windows are fixed at 30 seconds and validated before
    inference (§2 step 5).

    v3.0 additions:
      - Per-frame AU12 telemetry accumulation (§7A.4, §4.D.2)
      - Stimulus injection timestamping (§4.E.1)
      - Calibration lifecycle: B_neutral accumulation → scoring transition
      - Payload wiring for reward pipeline (_au12_series, _stimulus_time)
    """

    def __init__(
        self,
        stream_url: str = "",
        session_id: str | None = None,
        experiment_id: str = "greeting_line_v1",
    ) -> None:
        """
        Initialize orchestrator with session, experiment, video, and AU12 state.

        Args:
            stream_url: TikTok stream URL for this session.
            session_id: UUID for this session (auto-generated if None).
            experiment_id: Thompson Sampling experiment ID (§4.E.1).
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

        # Gap G-05 / Stage 2 — Thompson Sampling state
        self._experiment_id: str = experiment_id
        self._active_arm: str = ""
        self._expected_greeting: str = ""

        # Gap G-03 — Video capture (lazy-init in run() for live capture).
        # Replay mode binds the same source to the video and audio surfaces.
        self.video_capture: Any = replay_source

        # [v3.0] AU12 telemetry accumulator for continuous reward pipeline.
        # Each entry: {"timestamp_s": float, "intensity": float}
        # Drained into the payload on each assemble_segment() call.
        self._au12_series: list[dict[str, float]] = []

        # [v3.0] Stimulus injection timestamp (drift-corrected UTC epoch).
        # Set by record_stimulus_injection(). None until operator sends greeting.
        self._stimulus_time: float | None = None

        # [v3.2] Physiological rolling buffers — §4.C.4.
        # Keep time-ordered chunk events per subject_role for trailing-window derivation.
        self._physio_buffer: dict[str, deque[dict[str, Any]]] = {
            "streamer": deque(),
            "operator": deque(),
        }

        # [v3.1] Redis client for non-blocking physiological drains.
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

        # [v3.0] Per-session AU12 normalizer — lazy-init on first frame
        self._au12_normalizer: Any = None

        # [v3.0] Calibration state: True until operator injects greeting.
        # While True, AU12Normalizer accumulates B_neutral baseline (§7A.4).
        self._is_calibrating: bool = True

        # [v3.0] FaceMesh processor — lazy-init on first frame
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
        [v3.0] Record the exact moment the greeting line was sent.

        Must be called by the operator interface when the greeting text
        is injected into the live stream chat. This timestamp anchors
        the stimulus-locked measurement window [+0.5s, +5.0s] used by
        the reward pipeline (services/worker/pipeline/reward.py).

        Also transitions the AU12 normalizer from calibration mode to
        inference mode. All pre-stimulus frames contributed to B_neutral;
        all post-stimulus frames will be scored against that baseline.

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
        [v3.0] Grab the latest video frame and compute AU12 intensity.

        §4.D.2 — MediaPipe FaceMesh landmark extraction
        §7A.4 — AU12 baseline calibration (pre-stimulus) and scoring (post-stimulus)

        During calibration (before stimulus injection):
          - compute_bounded_intensity(is_calibrating=True) accumulates B_neutral
          - Returned intensity is 0.0 (calibration phase always returns zero)
          - These frames are NOT appended to _au12_series (no value for reward)

        After stimulus injection:
          - compute_bounded_intensity(is_calibrating=False) returns [0.0, 1.0]
          - {timestamp_s, intensity} appended to _au12_series for reward pipeline

        §5.2 — Frame data exists only in volatile memory (numpy arrays).
        """
        if self.video_capture is None:
            return

        frame = self.video_capture.get_latest_frame()
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
                raw = self._redis.lpop(PHYSIO_QUEUE_KEY)
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

    def assemble_segment(
        self,
        audio_data: bytes,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Assemble a 30-second segment as InferenceHandoffPayload.

        §2 step 5 — Validates against Pydantic schema before returning.
        §6.1 — InferenceHandoffPayload JSON Schema Draft 07 contract.

        Stage 2 fields: _frame_data, _active_arm, _experiment_id, _expected_greeting
        v3.0 fields: _au12_series, _stimulus_time, _x_max

        Args:
            audio_data: Raw 16 kHz s16le PCM bytes for this segment.
            events: Ground truth events within the segment window.

        Returns:
            Validated payload dict ready for Module D dispatch.
        """
        from packages.schemas.inference_handoff import (
            InferenceHandoffPayload,
            MediaSource,
        )

        assembly_started = time.perf_counter()
        self._segment_counter += 1
        segment_id = f"seg-{self._segment_counter:04d}"

        # §4.C.1 — Apply drift correction to current timestamp
        now_utc = self.drift_corrector.correct_timestamp(time.time())
        timestamp = datetime.fromtimestamp(now_utc, tz=UTC)

        # §2 step 5 — Build segment dict for the segments array
        segment_data: dict[str, Any] = {
            "segment_id": segment_id,
            "audio_bytes": len(audio_data),
            "events": events,
        }

        # [Stage 2] Gap G-03 — Determine codec based on video availability
        has_video = self.video_capture is not None and getattr(
            self.video_capture, "is_running", False
        )
        codec = "h264" if has_video else "raw"
        resolution = [1920, 1080] if has_video else [1, 1]

        # [Stage 2] Gap G-03 — Extract latest video frame FIRST
        frame_data: bytes | None = None
        if self.video_capture is not None:
            try:
                frame = self.video_capture.get_latest_frame()
                if frame is not None:
                    frame_array = (
                        frame.to_ndarray(format="bgr24") if hasattr(frame, "to_ndarray") else frame
                    )
                    frame_data = frame_array.tobytes()
                    resolution = [frame_array.shape[1], frame_array.shape[0]]
            except Exception:
                logger.warning("Assemble segment frame extraction failed", exc_info=True)

        payload = InferenceHandoffPayload(
            session_id=uuid.UUID(self._session_id),
            timestamp_utc=timestamp,
            media_source=MediaSource(
                stream_url=self._stream_url or "unknown",
                codec=codec,
                resolution=resolution,
            ),
            segments=[segment_data],
        )

        result: dict[str, Any] = payload.model_dump(mode="json")
        result["_audio_data"] = audio_data
        result["_segment_id"] = segment_id
        result["_frame_data"] = frame_data

        result["_active_arm"] = self._active_arm
        result["_experiment_id"] = self._experiment_id
        result["_expected_greeting"] = self._expected_greeting

        result["_au12_series"] = list(self._au12_series)
        self._au12_series.clear()

        result["_stimulus_time"] = self._stimulus_time

        now_wall = time.time()
        if any(self._physio_buffer[role] for role in ("streamer", "operator")):
            context = {
                "streamer": self._derive_physio_snapshot("streamer", now_wall=now_wall),
                "operator": self._derive_physio_snapshot("operator", now_wall=now_wall),
            }
            result["_physiological_context"] = context

        result["_x_max"] = None

        from services.worker.pipeline.serialization import encode_bytes_fields

        result = encode_bytes_fields(result, ["_audio_data", "_frame_data"])
        logger.info(
            "BENCHMARK segment_assembly_ms=%.3f segment_id=%s",
            (time.perf_counter() - assembly_started) * 1000.0,
            segment_id,
        )

        return result

    def _dispatch_payload(self, payload: dict[str, Any]) -> None:
        """Dispatch a validated payload to Module D."""
        try:
            from services.worker.tasks.inference import process_segment

            process_segment.delay(payload)
        except Exception as exc:
            logger.error("Failed to dispatch segment: %s", exc)

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
        payload = self.assemble_segment(audio_data, events)
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
        self._is_calibrating = True
        self._au12_normalizer = None

    def _begin_session(self, *, session_id: str, stream_url: str, experiment_id: str) -> None:
        """Make a session active and register it authoritatively in Postgres."""
        self._reset_session_state()
        self._session_id = session_id
        self._stream_url = stream_url
        self._experiment_id = experiment_id or "greeting_line_v1"
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

                client = redis.from_url(  # type: ignore[no-untyped-call]
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
        # [Stage 2] Gap G-03 — Stop video capture thread
        if self.video_capture is not None and self.video_capture is not self.audio_resampler:
            with contextlib.suppress(Exception):
                self.video_capture.stop()
        # [v3.1] Release Redis client used for physiological drain.
        if self._redis is not None and hasattr(self._redis, "close"):
            with contextlib.suppress(Exception):
                self._redis.close()
            self._redis = None
        # [v3.0] Release FaceMesh resources
        if self._face_mesh is not None:
            with contextlib.suppress(Exception):
                self._face_mesh.close()
            self._face_mesh = None
        # [v3.0] Clear AU12 state
        self._au12_normalizer = None
        self._au12_series.clear()

    async def run(self) -> None:
        """
        Main orchestration loop — revised for lifecycle-aware session ownership.

        §4.C — Coordinates all Module C responsibilities:
        0. Preserve boot-time session auto-create from STREAM_URL/EXPERIMENT_ID.
        0b. Start background Redis lifecycle listener for authoritative start/end.
        0c. Start video capture thread (Gap G-03).
        1. Poll drift every DRIFT_POLL_INTERVAL seconds (§4.C.1).
        2. Continuously read resampled audio chunks (§4.C.2).
        2b. [v3.0] Process latest video frame → AU12 accumulation.
        3. Assemble fixed 30-second segments only while a session is active.
        4. Flush/rotate sessions on lifecycle intents from the API Server.
        """
        # --- Pre-loop initialization ---
        self._running = True
        self._start_session_lifecycle_listener()

        # Preserve the historical boot path until/unless lifecycle messages arrive.
        self._begin_session(
            session_id=self._session_id,
            stream_url=self._stream_url,
            experiment_id=self._experiment_id,
        )
        self._publish_live_session_state()
        self._publish_orchestrator_heartbeat()

        # Gap G-03 — Start video capture from IPC Pipe unless replay is opt-in.
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

        last_drift_poll = 0.0

        while self._running:
            self._drain_session_lifecycle_intents()
            now = time.monotonic()
            self._publish_orchestrator_heartbeat_if_due(now)

            # §4.C.1 — Poll drift at configured interval. Replay mode keeps the
            # zero/cached offset and intentionally avoids ADB/USB dependencies.
            if not self._using_replay_capture and now - last_drift_poll >= DRIFT_POLL_INTERVAL:
                self.drift_corrector.poll()
                last_drift_poll = now

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

            # [v3.0] §4.D.2 + §7A.4 — Process latest video frame for AU12 only
            # while a lifecycle session is actively running.
            if self._session_active:
                self._process_video_frame()

            # §2 step 5 — When we have 30s of audio for an active session,
            # assemble and dispatch a segment.
            if self._session_active and len(self._audio_buffer) >= segment_bytes:
                audio_data = bytes(self._audio_buffer[:segment_bytes])
                self._audio_buffer = bytearray(self._audio_buffer[segment_bytes:])
                events = self._drain_event_buffer()
                self._drain_physio_events()
                payload = self.assemble_segment(audio_data, events)
                self._dispatch_payload(payload)

            # Yield control to event loop
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # Stage 2 — Session registration and experiment arm selection
    # ------------------------------------------------------------------

    def _register_session(self) -> None:
        """
        Gap G-02 — Register this session in the Persistent Store.

        §2 step 7 — Parameterized INSERT with ON CONFLICT guard
        (idempotent in case of worker restart with same session_id).

        Must execute synchronously before the first segment dispatch.
        Uses a direct psycopg2 connection (not the MetricsStore pool)
        to avoid circular dependency with the Celery task layer.
        """
        insert_session_sql = """
            INSERT INTO sessions (session_id, stream_url, started_at)
            VALUES (%(session_id)s, %(stream_url)s, NOW())
            ON CONFLICT (session_id) DO NOTHING
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
                    cur.execute(
                        insert_session_sql,
                        {
                            "session_id": self._session_id,
                            "stream_url": self._stream_url or "unknown",
                        },
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

        Queries the Persistent Store for the experiment arms and draws
        from the Beta(alpha, beta) posterior to select the arm with the
        highest sample. The selected arm and its corresponding greeting
        line are stored on the orchestrator instance and injected into
        every InferenceHandoffPayload for the duration of this session.

        If no arms exist (experiment not seeded), falls back to the
        first greeting line and logs a warning.
        """
        # Mapping of arm names to greeting line strings.
        # These must match the arms seeded in 02-seed-experiments.sql.
        greeting_lines: dict[str, str] = {
            "warm_welcome": "Hey! Thanks for streaming, you're awesome!",
            "direct_question": "Hi! What's the best advice you've gotten today?",
            "compliment_content": "Love the energy on this stream! How long have you been live?",
            "simple_hello": "Hello! Just joined, happy to be here!",
        }

        try:
            from services.worker.pipeline.analytics import MetricsStore, ThompsonSamplingEngine

            store = MetricsStore()
            store.connect()
            try:
                engine = ThompsonSamplingEngine(store)
                self._active_arm = engine.select_arm(self._experiment_id)
                self._expected_greeting = greeting_lines.get(
                    self._active_arm,
                    "Hello, welcome to the stream!",
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
            # Fallback: use first greeting line if TS unavailable
            self._active_arm = "simple_hello"
            self._expected_greeting = greeting_lines["simple_hello"]
            logger.warning(
                "Thompson Sampling unavailable, using fallback arm: %s",
                exc,
            )
