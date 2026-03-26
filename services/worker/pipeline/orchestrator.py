"""
Orchestration & Synchronization — §4.C Module C

Aligns timestamps between device hardware clocks, IPC media streams,
and external WebSocket events. Manages audio resampling pipeline
and segment assembly for handoff to Module D.

v3.0 additions:
  - Per-frame AU12 accumulation via compute_bounded_intensity() (§7.4)
  - Stimulus injection timestamping for reward window anchoring (§4.E.1)
  - Calibration lifecycle: pre-stimulus B_neutral → post-stimulus scoring
  - Payload wiring: _au12_series, _stimulus_time, _x_max fields
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import time
import uuid
from collections import deque
from datetime import UTC, datetime
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
        §12 Hardware loss C: freeze drift after 3 failures, reset to
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
                self._process = None
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
      - Per-frame AU12 telemetry accumulation (§7.4, §4.D.2)
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

        # Gap G-03 — Video capture (lazy-init in run())
        self.video_capture: Any = None

        # [v3.0] AU12 telemetry accumulator for continuous reward pipeline.
        # Each entry: {"timestamp_s": float, "intensity": float}
        # Drained into the payload on each assemble_segment() call.
        self._au12_series: list[dict[str, float]] = []

        # [v3.0] Stimulus injection timestamp (drift-corrected UTC epoch).
        # Set by record_stimulus_injection(). None until operator sends greeting.
        self._stimulus_time: float | None = None

        # [v3.0] Per-session AU12 normalizer — lazy-init on first frame
        self._au12_normalizer: Any = None

        # [v3.0] Calibration state: True until operator injects greeting.
        # While True, AU12Normalizer accumulates B_neutral baseline (§7.4).
        self._is_calibrating: bool = True

        # [v3.0] FaceMesh processor — lazy-init on first frame
        self._face_mesh: Any = None

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

        §7.4 — Calibration phase ends at stimulus onset.
        §4.E.1 — Thompson Sampling experiment arm deployment trigger.
        """
        self._stimulus_time = self.drift_corrector.correct_timestamp(time.time())
        self._is_calibrating = False
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
        §7.4 — AU12 baseline calibration (pre-stimulus) and scoring (post-stimulus)

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

            # §4.D.2 — Extract 478-vertex 3D landmarks from BGR frame
            landmarks = self._face_mesh.extract_landmarks(frame)
            if landmarks is None:
                # §4.D contract — missing face returns null facial metrics
                return

            # §7.4 — Compute bounded AU12 intensity in [0.0, 1.0]
            intensity: float = self._au12_normalizer.compute_bounded_intensity(
                landmarks,
                is_calibrating=self._is_calibrating,
            )

            # Only accumulate post-stimulus frames (calibration returns 0.0
            # and doesn't produce meaningful reward signal)
            if not self._is_calibrating:
                now_utc = self.drift_corrector.correct_timestamp(time.time())
                self._au12_series.append({"timestamp_s": now_utc, "intensity": float(intensity)})

        except Exception:
            # §12 — AU12 extraction failure must not crash the main loop.
            # Frame is silently dropped; subsequent frames will retry.
            logger.debug("AU12 frame processing failed", exc_info=True)

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

        # §6.1 — Construct and validate via Pydantic
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

        # Return as dict for serialization to Module D
        result: dict[str, Any] = payload.model_dump(mode="json")
        result["_audio_data"] = audio_data
        result["_segment_id"] = segment_id

        # [Stage 2] Gap G-03 — Attach latest video frame (or None)
        frame_data: bytes | None = None
        if self.video_capture is not None:
            try:
                frame = self.video_capture.get_latest_frame()
                if frame is not None:
                    frame_data = frame.tobytes()
            except Exception:
                pass
        result["_frame_data"] = frame_data

        # [Stage 2] §4.E.1 — Inject experiment arm for downstream evaluation
        result["_active_arm"] = self._active_arm
        result["_experiment_id"] = self._experiment_id
        result["_expected_greeting"] = self._expected_greeting

        # [v3.0] Attach AU12 time series accumulated since last segment.
        # The reward pipeline (reward.py) uses this to compute the P90
        # within the stimulus-locked window [+0.5s, +5.0s].
        # Drain the accumulator so each segment gets exactly its frames.
        result["_au12_series"] = list(self._au12_series)
        self._au12_series.clear()

        # [v3.0] Attach stimulus injection timestamp for window alignment.
        # None if operator hasn't injected the greeting yet — persist_metrics
        # will skip the Thompson Sampling update for this segment.
        result["_stimulus_time"] = self._stimulus_time

        # [v3.0] Per-subject maximum response capability (future calibration).
        # Currently None; will be populated when explicit calibration gesture
        # (e.g., "please smile broadly") is implemented in the operator UI.
        result["_x_max"] = None

        return result

    def stop(self) -> None:
        """Stop the orchestration loop and clean up all resources."""
        self._running = False
        self.audio_resampler.stop()
        # [Stage 2] Gap G-03 — Stop video capture thread
        if self.video_capture is not None:
            with contextlib.suppress(Exception):
                self.video_capture.stop()
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
        Main orchestration loop — revised for v3.0.

        §4.C — Coordinates all Module C responsibilities:
        0. Register session in Persistent Store (Gap G-02)
        0b. Select Thompson Sampling arm for this session (§4.E.1)
        0c. Start video capture thread (Gap G-03)
        1. Poll drift every DRIFT_POLL_INTERVAL seconds (§4.C.1)
        2. Continuously read resampled audio chunks (§4.C.2)
        2b. [v3.0] Process latest video frame → AU12 accumulation
        3. Accumulate 30s of audio, drain event buffer (§2 step 5)
        4. Assemble InferenceHandoffPayload and dispatch (§6.1)
        """
        # --- Pre-loop initialization (Stage 2 + v3.0) ---

        # Gap G-02 — Must execute before any segment dispatch
        self._register_session()

        # §4.E.1 — Select greeting line for this session
        self._select_experiment_arm()

        # Gap G-03 — Start video capture from IPC Pipe
        try:
            from services.worker.pipeline.video_capture import VideoCapture

            self.video_capture = VideoCapture()
            self.video_capture.start()
        except Exception as exc:
            logger.warning("Video capture unavailable: %s", exc)
            self.video_capture = None

        # --- Main loop ---
        self._running = True
        self.audio_resampler.start()

        # §4.C.2 — 16 kHz, mono, s16le = 2 bytes/sample = 32000 bytes/second
        bytes_per_second = 16000 * 2  # 16 kHz * 2 bytes (s16le)
        segment_bytes = bytes_per_second * SEGMENT_WINDOW_SECONDS
        chunk_size = bytes_per_second  # Read 1 second at a time

        last_drift_poll = 0.0

        while self._running:
            now = time.monotonic()

            # §4.C.1 — Poll drift at configured interval
            if now - last_drift_poll >= DRIFT_POLL_INTERVAL:
                self.drift_corrector.poll()
                last_drift_poll = now

            # §4.C.2 — Read resampled audio chunk
            chunk = self.audio_resampler.read_chunk(chunk_size)
            if chunk:
                self._audio_buffer.extend(chunk)

            # [v3.0] §4.D.2 + §7.4 — Process latest video frame for AU12
            self._process_video_frame()

            # §2 step 5 — When we have 30s of audio, assemble segment
            if len(self._audio_buffer) >= segment_bytes:
                audio_data = bytes(self._audio_buffer[:segment_bytes])
                self._audio_buffer = bytearray(self._audio_buffer[segment_bytes:])

                # Drain events from buffer for this segment window
                events: list[dict[str, Any]] = []
                while self.event_buffer:
                    events.append(self.event_buffer.popleft())

                payload = self.assemble_segment(audio_data, events)

                # §2 step 5 → §2 step 6 — Dispatch to Module D
                try:
                    from services.worker.tasks.inference import process_segment

                    process_segment.delay(payload)
                except Exception as exc:
                    logger.error("Failed to dispatch segment: %s", exc)

            # Yield control to event loop
            await asyncio.sleep(0.1)

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
        import os

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
