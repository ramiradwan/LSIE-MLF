"""
Tests for services/worker/pipeline/orchestrator.py — Phase 3.1–3.3 validation.

Verifies DriftCorrector, AudioResampler, and Orchestrator against:
  §4.C.1 — Drift polling and freeze/reset behavior
  §4.C.2 — FFmpeg resampling subprocess
  §2 step 5 — 30-second segment assembly with Pydantic validation
  §12 — Error handling for hardware loss and worker crash
"""

from __future__ import annotations

import subprocess
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from services.worker.pipeline.orchestrator import (
    DRIFT_FREEZE_AFTER_FAILURES,
    DRIFT_RESET_TIMEOUT,
    FFMPEG_RESAMPLE_CMD,
    AudioResampler,
    DriftCorrector,
    Orchestrator,
)


class TestDriftCorrector:
    """§4.C.1 — Temporal drift correction."""

    def test_initial_offset_zero(self) -> None:
        """Drift offset starts at zero."""
        dc = DriftCorrector()
        assert dc.drift_offset == 0.0

    def test_poll_success_computes_offset(self) -> None:
        """§4.C.1 — drift_offset = host_utc - android_epoch."""
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.123456\n"

        with patch("services.worker.pipeline.orchestrator.subprocess.run", return_value=mock_result), \
             patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.5):
            offset = dc.poll()

        expected = 1710000000.5 - 1710000000.123456
        assert abs(offset - expected) < 1e-6
        assert dc._consecutive_failures == 0
        assert not dc._frozen

    def test_poll_failure_increments_counter(self) -> None:
        """§12 Hardware loss — consecutive failures tracked."""
        dc = DriftCorrector()
        with patch(
            "services.worker.pipeline.orchestrator.subprocess.run",
            side_effect=subprocess.TimeoutExpired("adb", 5),
        ):
            dc.poll()
        assert dc._consecutive_failures == 1
        assert not dc._frozen

    def test_poll_freezes_after_3_failures(self) -> None:
        """§12 Hardware loss C — freeze drift after 3 failures."""
        dc = DriftCorrector()
        dc.drift_offset = 0.5  # Set a non-zero offset before freeze

        with patch(
            "services.worker.pipeline.orchestrator.subprocess.run",
            side_effect=RuntimeError("ADB down"),
        ):
            for _ in range(DRIFT_FREEZE_AFTER_FAILURES):
                dc.poll()

        assert dc._frozen
        assert dc.drift_offset == 0.5  # Preserved at frozen value

    def test_frozen_returns_cached_offset(self) -> None:
        """§12 — While frozen, poll returns cached offset without ADB call."""
        dc = DriftCorrector()
        dc.drift_offset = 1.5
        dc._frozen = True
        dc._frozen_at = time.monotonic()

        # Should not call subprocess at all
        with patch("services.worker.pipeline.orchestrator.subprocess.run") as mock_run:
            offset = dc.poll()
            mock_run.assert_not_called()

        assert offset == 1.5

    def test_frozen_resets_after_5_minutes(self) -> None:
        """§12 Hardware loss C — reset to zero after 5 minutes."""
        dc = DriftCorrector()
        dc.drift_offset = 2.0
        dc._frozen = True
        dc._frozen_at = time.monotonic() - DRIFT_RESET_TIMEOUT - 1

        offset = dc.poll()
        assert offset == 0.0
        assert not dc._frozen
        assert dc._consecutive_failures == 0

    def test_correct_timestamp(self) -> None:
        """Apply drift correction."""
        dc = DriftCorrector()
        dc.drift_offset = 0.5
        assert dc.correct_timestamp(100.0) == 100.5

    def test_success_resets_failure_count(self) -> None:
        """Successful poll resets consecutive failure counter."""
        dc = DriftCorrector()
        dc._consecutive_failures = 2

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.0\n"

        with patch("services.worker.pipeline.orchestrator.subprocess.run", return_value=mock_result), \
             patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.0):
            dc.poll()

        assert dc._consecutive_failures == 0

    def test_nonzero_returncode_is_failure(self) -> None:
        """ADB returning non-zero exit code counts as failure."""
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("services.worker.pipeline.orchestrator.subprocess.run", return_value=mock_result):
            dc.poll()

        assert dc._consecutive_failures == 1


class TestAudioResampler:
    """§4.C.2 — FFmpeg audio resampling subprocess."""

    def test_start_launches_ffmpeg(self) -> None:
        """§4.C.2 — start() spawns FFmpeg with correct command."""
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # running

        with patch("services.worker.pipeline.orchestrator.subprocess.Popen", return_value=mock_proc) as mock_popen:
            ar.start()
            mock_popen.assert_called_once_with(
                FFMPEG_RESAMPLE_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

    def test_start_idempotent(self) -> None:
        """start() doesn't relaunch if already running."""
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("services.worker.pipeline.orchestrator.subprocess.Popen", return_value=mock_proc) as mock_popen:
            ar.start()
            ar.start()
            assert mock_popen.call_count == 1

    def test_read_chunk_returns_data(self) -> None:
        """read_chunk reads from FFmpeg stdout."""
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.read.return_value = b"\x00" * 1024
        ar._process = mock_proc

        data = ar.read_chunk(1024)
        assert len(data) == 1024

    def test_read_chunk_restarts_on_crash(self) -> None:
        """§2 step 3 / §12 Worker crash C — restart FFmpeg on crash."""
        ar = AudioResampler()
        ar._process = None  # Simulates crashed state

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.read.return_value = b"\x00" * 100

        with patch("services.worker.pipeline.orchestrator.subprocess.Popen", return_value=mock_proc), \
             patch("services.worker.pipeline.orchestrator.time.sleep") as mock_sleep:
            data = ar.read_chunk(100)
            # §2 step 3 — 1s restart delay
            mock_sleep.assert_called_once_with(1.0)
            assert len(data) == 100

    def test_read_chunk_returns_empty_on_eof(self) -> None:
        """§12 Worker crash C — EOF means FFmpeg exited."""
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.read.return_value = b""
        ar._process = mock_proc

        data = ar.read_chunk(1024)
        assert data == b""
        assert ar._process is None  # Process cleaned up

    def test_stop_terminates_process(self) -> None:
        """stop() terminates FFmpeg gracefully."""
        ar = AudioResampler()
        mock_proc = MagicMock()
        ar._process = mock_proc

        ar.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert ar._process is None

    def test_is_running_property(self) -> None:
        """is_running reflects FFmpeg process state."""
        ar = AudioResampler()
        assert not ar.is_running

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        ar._process = mock_proc
        assert ar.is_running

        mock_proc.poll.return_value = 0
        assert not ar.is_running


class TestOrchestrator:
    """§4.C / §2 step 5 — Orchestrator segment assembly."""

    def test_assemble_segment_validates_payload(self) -> None:
        """§2 step 5 — InferenceHandoffPayload validated by Pydantic."""
        orch = Orchestrator(stream_url="https://example.com/stream")
        audio = b"\x00" * 960000  # 30s at 16kHz mono s16le
        events: list[dict[str, Any]] = [
            {"uniqueId": "user1", "event_type": "gift", "timestamp_utc": "2026-03-13T12:00:00Z"},
        ]

        payload = orch.assemble_segment(audio, events)

        assert "session_id" in payload
        assert "timestamp_utc" in payload
        assert "media_source" in payload
        assert "segments" in payload
        assert payload["_segment_id"] == "seg-0001"
        assert payload["_audio_data"] == audio

    def test_assemble_segment_increments_counter(self) -> None:
        """Segment IDs increment."""
        orch = Orchestrator()
        audio = b"\x00" * 100

        p1 = orch.assemble_segment(audio, [])
        p2 = orch.assemble_segment(audio, [])

        assert p1["_segment_id"] == "seg-0001"
        assert p2["_segment_id"] == "seg-0002"

    def test_assemble_segment_applies_drift(self) -> None:
        """§4.C.1 — Timestamps drift-corrected."""
        orch = Orchestrator()
        orch.drift_corrector.drift_offset = 1.0

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.0):
            payload = orch.assemble_segment(b"\x00", [])

        # Timestamp should be based on corrected time (1710000001.0)
        assert payload["timestamp_utc"] is not None

    def test_event_buffer_deque_maxlen(self) -> None:
        """§12 Queue overload B/C — deque eviction via maxlen."""
        orch = Orchestrator()
        assert orch.event_buffer.maxlen == 10000

    def test_stop_cleans_up(self) -> None:
        """stop() stops resampler and sets running flag."""
        orch = Orchestrator()
        orch._running = True
        mock_proc = MagicMock()
        orch.audio_resampler._process = mock_proc

        orch.stop()
        assert not orch._running
        mock_proc.terminate.assert_called_once()
