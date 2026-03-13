"""
Orchestration & Synchronization — §4.C Module C

Aligns timestamps between device hardware clocks, IPC media streams,
and external WebSocket events. Manages audio resampling pipeline
and segment assembly for handoff to Module D.
"""

from __future__ import annotations

import asyncio
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
    """

    def __init__(
        self,
        stream_url: str = "",
        session_id: str | None = None,
    ) -> None:
        self.drift_corrector = DriftCorrector()
        self.audio_resampler = AudioResampler()
        # §12 Queue overload B/C — deque eviction
        self.event_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self._stream_url = stream_url
        self._session_id = session_id or str(uuid.uuid4())
        self._segment_counter: int = 0
        self._audio_buffer: bytearray = bytearray()
        self._running: bool = False

    def assemble_segment(
        self,
        audio_data: bytes,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Assemble a 30-second segment as InferenceHandoffPayload.

        §2 step 5 — Validates against Pydantic schema before returning.
        §6.1 — InferenceHandoffPayload JSON Schema Draft 07 contract.

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

        # §6.1 — Construct and validate via Pydantic
        payload = InferenceHandoffPayload(
            session_id=uuid.UUID(self._session_id),
            timestamp_utc=timestamp,
            media_source=MediaSource(
                stream_url=self._stream_url or "unknown",
                codec="raw",
                resolution=[1, 1],  # Audio-only; placeholder for schema
            ),
            segments=[segment_data],
        )

        # Return as dict for serialization to Module D
        result: dict[str, Any] = payload.model_dump(mode="json")
        # Attach raw audio reference for Module D processing
        result["_audio_data"] = audio_data
        result["_segment_id"] = segment_id
        return result

    def stop(self) -> None:
        """Stop the orchestration loop and clean up resources."""
        self._running = False
        self.audio_resampler.stop()

    async def run(self) -> None:
        """
        Main orchestration loop: poll drift, read audio, assemble
        segments, dispatch to Module D.

        §4.C — Coordinates all Module C responsibilities:
        1. Poll drift every DRIFT_POLL_INTERVAL seconds (§4.C.1)
        2. Continuously read resampled audio chunks (§4.C.2)
        3. Accumulate 30s of audio, drain event buffer (§2 step 5)
        4. Assemble InferenceHandoffPayload and dispatch (§6.1)
        """
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
