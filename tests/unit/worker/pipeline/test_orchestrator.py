"""
Tests for services/worker/pipeline/orchestrator.py — Phase 3.2 validation.

Verifies DriftCorrector, AudioResampler, and Orchestrator against:
  §4.C.1 — Drift polling and freeze/reset behavior
  §4.C.2 — FFmpeg resampling subprocess
  §2 step 5 — 30-second segment assembly with Pydantic validation
  §12 — Error handling for hardware loss and worker crash
  §4.C.4 — Rolling physiological buffering and trailing-window derivation
"""

from __future__ import annotations

import base64
import subprocess
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal
from unittest.mock import MagicMock, patch

from packages.schemas.physiology import PhysiologicalChunkEvent, PhysiologicalChunkPayload
from services.worker.pipeline.orchestrator import (
    DRIFT_FREEZE_AFTER_FAILURES,
    DRIFT_RESET_TIMEOUT,
    FFMPEG_RESAMPLE_CMD,
    MAX_PHYSIO_DRAIN_PER_CYCLE,
    PHYSIO_BUFFER_RETENTION_S,
    PHYSIO_DERIVE_WINDOW_S,
    PHYSIO_STALENESS_THRESHOLD_S,
    AudioResampler,
    DriftCorrector,
    Orchestrator,
)


def _physio_chunk_json(
    *,
    subject_role: Literal["streamer", "operator"] = "streamer",
    source_kind: Literal["ibi", "session"] = "ibi",
    window_start_utc: datetime | None = None,
    window_end_utc: datetime | None = None,
    ibi_ms_items: list[float] | None = None,
    rmssd_items_ms: list[float] | None = None,
    heart_rate_items_bpm: list[int] | None = None,
    valid_sample_count: int = 4,
    expected_sample_count: int = 4,
    derivation_method: Literal["provider", "server"] = "provider",
) -> str:
    start = window_start_utc or datetime.now(UTC)
    end = window_end_utc or start
    event = PhysiologicalChunkEvent(
        unique_id=uuid.uuid4(),
        provider="oura",
        subject_role=subject_role,
        source_kind=source_kind,
        window_start_utc=start,
        window_end_utc=end,
        ingest_timestamp_utc=end,
        payload=PhysiologicalChunkPayload(
            sample_interval_s=60,
            valid_sample_count=valid_sample_count,
            expected_sample_count=expected_sample_count,
            derivation_method=derivation_method,
            ibi_ms_items=ibi_ms_items,
            rmssd_items_ms=rmssd_items_ms,
            heart_rate_items_bpm=heart_rate_items_bpm,
        ),
    )
    return event.model_dump_json()


def _buffer_chunk(
    *,
    role: Literal["streamer", "operator"],
    source_kind: Literal["ibi", "session"],
    end_ts: float,
    payload: dict[str, Any],
    derivation_method: Literal["provider", "server"] = "provider",
) -> dict[str, Any]:
    end = datetime.fromtimestamp(end_ts, tz=UTC)
    start = datetime.fromtimestamp(end_ts - 60, tz=UTC)
    return {
        "provider": "oura",
        "subject_role": role,
        "source_kind": source_kind,
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "window_start_ts": start.timestamp(),
        "window_end_ts": end.timestamp(),
        "payload_derivation_method": derivation_method,
        "payload": payload,
    }


class TestDriftCorrector:
    """§4.C.1 — Temporal drift correction."""

    def test_initial_offset_zero(self) -> None:
        dc = DriftCorrector()
        assert dc.drift_offset == 0.0

    def test_poll_success_computes_offset(self) -> None:
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.123456\n"

        with (
            patch("services.worker.pipeline.orchestrator.subprocess.run", return_value=mock_result),
            patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.5),
        ):
            offset = dc.poll()

        expected = 1710000000.5 - 1710000000.123456
        assert abs(offset - expected) < 1e-6
        assert dc._consecutive_failures == 0
        assert not dc._frozen

    def test_poll_failure_increments_counter(self) -> None:
        dc = DriftCorrector()
        with patch(
            "services.worker.pipeline.orchestrator.subprocess.run",
            side_effect=subprocess.TimeoutExpired("adb", 5),
        ):
            dc.poll()
        assert dc._consecutive_failures == 1
        assert not dc._frozen

    def test_poll_freezes_after_3_failures(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 0.5

        with patch(
            "services.worker.pipeline.orchestrator.subprocess.run",
            side_effect=RuntimeError("ADB down"),
        ):
            for _ in range(DRIFT_FREEZE_AFTER_FAILURES):
                dc.poll()

        assert dc._frozen
        assert dc.drift_offset == 0.5

    def test_frozen_returns_cached_offset(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 1.5
        dc._frozen = True
        dc._frozen_at = time.monotonic()

        with patch("services.worker.pipeline.orchestrator.subprocess.run") as mock_run:
            offset = dc.poll()
            mock_run.assert_not_called()

        assert offset == 1.5

    def test_frozen_resets_after_5_minutes(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 2.0
        dc._frozen = True
        dc._frozen_at = time.monotonic() - DRIFT_RESET_TIMEOUT - 1

        offset = dc.poll()
        assert offset == 0.0
        assert not dc._frozen
        assert dc._consecutive_failures == 0

    def test_correct_timestamp(self) -> None:
        dc = DriftCorrector()
        dc.drift_offset = 0.5
        assert dc.correct_timestamp(100.0) == 100.5

    def test_success_resets_failure_count(self) -> None:
        dc = DriftCorrector()
        dc._consecutive_failures = 2

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1710000000.0\n"

        with (
            patch("services.worker.pipeline.orchestrator.subprocess.run", return_value=mock_result),
            patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.0),
        ):
            dc.poll()

        assert dc._consecutive_failures == 0

    def test_nonzero_returncode_is_failure(self) -> None:
        dc = DriftCorrector()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch(
            "services.worker.pipeline.orchestrator.subprocess.run", return_value=mock_result
        ):
            dc.poll()

        assert dc._consecutive_failures == 1


class TestAudioResampler:
    """§4.C.2 — FFmpeg audio resampling subprocess."""

    def test_start_launches_ffmpeg(self) -> None:
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch(
            "services.worker.pipeline.orchestrator.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            ar.start()
            mock_popen.assert_called_once_with(
                FFMPEG_RESAMPLE_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

    def test_start_idempotent(self) -> None:
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch(
            "services.worker.pipeline.orchestrator.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            ar.start()
            ar.start()
            assert mock_popen.call_count == 1

    def test_read_chunk_returns_data(self) -> None:
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.read.return_value = b"\x00" * 1024
        ar._process = mock_proc

        data = ar.read_chunk(1024)
        assert len(data) == 1024

    def test_read_chunk_restarts_on_crash(self) -> None:
        ar = AudioResampler()
        ar._process = None  # type: ignore[assignment]

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.read.return_value = b"\x00" * 100

        with (
            patch("services.worker.pipeline.orchestrator.subprocess.Popen", return_value=mock_proc),
            patch("services.worker.pipeline.orchestrator.time.sleep") as mock_sleep,
        ):
            data = ar.read_chunk(100)
            mock_sleep.assert_called_once_with(1.0)
            assert len(data) == 100

    def test_read_chunk_returns_empty_on_eof(self) -> None:
        ar = AudioResampler()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.read.return_value = b""
        ar._process = mock_proc

        data = ar.read_chunk(1024)
        assert data == b""
        assert ar._process is None

    def test_stop_terminates_process(self) -> None:
        ar = AudioResampler()
        mock_proc = MagicMock()
        ar._process = mock_proc

        ar.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert ar._process is None

    def test_is_running_property(self) -> None:
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

    def test_physio_buffer_initialized_empty(self) -> None:
        orch = Orchestrator()
        assert set(orch._physio_buffer.keys()) == {"streamer", "operator"}
        assert all(
            isinstance(buffer, deque) and len(buffer) == 0
            for buffer in orch._physio_buffer.values()
        )

    def test_drain_physio_events_appends_chunk_and_prunes_retention(self) -> None:
        orch = Orchestrator()
        now_wall = 2000.0
        old_end = datetime.fromtimestamp(now_wall - PHYSIO_BUFFER_RETENTION_S - 1, tz=UTC)
        fresh_end = datetime.fromtimestamp(now_wall - 10, tz=UTC)
        old_start = datetime.fromtimestamp(old_end.timestamp() - 60, tz=UTC)
        fresh_start = datetime.fromtimestamp(fresh_end.timestamp() - 60, tz=UTC)
        mock_redis = MagicMock()
        mock_redis.lpop.side_effect = [
            _physio_chunk_json(
                source_kind="session",
                window_start_utc=old_start,
                window_end_utc=old_end,
                rmssd_items_ms=[30.0],
                heart_rate_items_bpm=[65],
            ),
            _physio_chunk_json(
                source_kind="session",
                window_start_utc=fresh_start,
                window_end_utc=fresh_end,
                rmssd_items_ms=[42.0],
                heart_rate_items_bpm=[72],
            ),
            None,
        ]
        orch._redis = mock_redis

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            orch._drain_physio_events()

        buffer = orch._physio_buffer["streamer"]
        assert len(buffer) == 1
        assert buffer[0]["window_end_utc"] == fresh_end.isoformat()
        assert mock_redis.lpop.call_count == 3

    def test_drain_physio_events_bounded(self) -> None:
        orch = Orchestrator()
        now_wall = 2000.0
        end = datetime.fromtimestamp(now_wall - 1, tz=UTC)
        start = datetime.fromtimestamp(end.timestamp() - 60, tz=UTC)
        mock_redis = MagicMock()
        mock_redis.lpop.side_effect = [
            _physio_chunk_json(
                window_start_utc=start,
                window_end_utc=end,
                ibi_ms_items=[800.0, 810.0],
                heart_rate_items_bpm=[70],
            )
        ] * 200
        orch._redis = mock_redis

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            orch._drain_physio_events()

        assert mock_redis.lpop.call_count == MAX_PHYSIO_DRAIN_PER_CYCLE

    def test_assemble_segment_validates_payload(self) -> None:
        orch = Orchestrator(stream_url="https://example.com/stream")
        audio = b"\x00" * 960000
        events: list[dict[str, Any]] = [
            {
                "uniqueId": "user1",
                "event_type": "gift",
                "timestamp_utc": "2026-03-13T12:00:00Z",
            },
        ]

        payload = orch.assemble_segment(audio, events)

        assert "session_id" in payload
        assert "timestamp_utc" in payload
        assert "media_source" in payload
        assert "segments" in payload
        assert payload["_segment_id"] == "seg-0001"
        assert base64.b64decode(payload["_audio_data"]) == audio

    def test_assemble_segment_prefers_ibi_over_session(self) -> None:
        orch = Orchestrator()
        now_wall = 1000.0
        ibi_end = now_wall - 20
        session_end = now_wall - 10
        orch._physio_buffer["streamer"].extend(
            [
                _buffer_chunk(
                    role="streamer",
                    source_kind="session",
                    end_ts=session_end,
                    payload={
                        "sample_interval_s": 60,
                        "valid_sample_count": 4,
                        "expected_sample_count": 4,
                        "derivation_method": "provider",
                        "ibi_ms_items": None,
                        "rmssd_items_ms": [99.0],
                        "heart_rate_items_bpm": [80],
                        "motion_items": None,
                    },
                ),
                _buffer_chunk(
                    role="streamer",
                    source_kind="ibi",
                    end_ts=ibi_end,
                    payload={
                        "sample_interval_s": 60,
                        "valid_sample_count": 4,
                        "expected_sample_count": 4,
                        "derivation_method": "provider",
                        "ibi_ms_items": [800.0, 1000.0, 900.0],
                        "rmssd_items_ms": None,
                        "heart_rate_items_bpm": [70, 74],
                        "motion_items": None,
                    },
                ),
            ]
        )

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            payload = orch.assemble_segment(b"\x00", [])

        expected_rmssd = round(((((1000 - 800) ** 2) + ((900 - 1000) ** 2)) / 2) ** 0.5, 3)
        snapshot = payload["_physiological_context"]["streamer"]
        assert snapshot["rmssd_ms"] == expected_rmssd
        assert snapshot["heart_rate_bpm"] == 72
        assert snapshot["source_kind"] == "ibi"
        assert snapshot["derivation_method"] == "server"
        assert (
            snapshot["source_timestamp_utc"]
            == datetime.fromtimestamp(
                ibi_end,
                tz=UTC,
            ).isoformat()
        )
        assert snapshot["freshness_s"] == 20.0
        assert snapshot["is_valid"] is True
        assert snapshot["is_stale"] is False
        assert payload["_physiological_context"]["operator"] is None

    def test_invalid_window_keeps_metadata_but_nulls_rmssd(self) -> None:
        orch = Orchestrator()
        now_wall = 1000.0
        end = now_wall - 15
        orch._physio_buffer["streamer"].append(
            _buffer_chunk(
                role="streamer",
                source_kind="ibi",
                end_ts=end,
                payload={
                    "sample_interval_s": 60,
                    "valid_sample_count": 1,
                    "expected_sample_count": 4,
                    "derivation_method": "provider",
                    "ibi_ms_items": [800.0, 810.0, 805.0],
                    "rmssd_items_ms": None,
                    "heart_rate_items_bpm": [69],
                    "motion_items": None,
                },
            )
        )

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            payload = orch.assemble_segment(b"\x00", [])

        snapshot = payload["_physiological_context"]["streamer"]
        assert snapshot["rmssd_ms"] is None
        assert snapshot["heart_rate_bpm"] == 69
        assert snapshot["source_kind"] == "ibi"
        assert snapshot["derivation_method"] == "server"
        assert (
            snapshot["source_timestamp_utc"]
            == datetime.fromtimestamp(
                end,
                tz=UTC,
            ).isoformat()
        )
        assert snapshot["validity_ratio"] == 0.25
        assert snapshot["is_valid"] is False
        assert snapshot["window_s"] == PHYSIO_DERIVE_WINDOW_S
        assert snapshot["freshness_s"] == 15.0

    def test_session_snapshot_uses_mean_rmssd_and_rounds_to_3_decimals(self) -> None:
        orch = Orchestrator()
        now_wall = 2000.0
        orch._physio_buffer["operator"].extend(
            [
                _buffer_chunk(
                    role="operator",
                    source_kind="session",
                    end_ts=now_wall - 40,
                    payload={
                        "sample_interval_s": 60,
                        "valid_sample_count": 4,
                        "expected_sample_count": 4,
                        "derivation_method": "provider",
                        "ibi_ms_items": None,
                        "rmssd_items_ms": [40.1114, 40.1115],
                        "heart_rate_items_bpm": [66],
                        "motion_items": None,
                    },
                ),
                _buffer_chunk(
                    role="operator",
                    source_kind="session",
                    end_ts=now_wall - 10,
                    payload={
                        "sample_interval_s": 60,
                        "valid_sample_count": 4,
                        "expected_sample_count": 4,
                        "derivation_method": "provider",
                        "ibi_ms_items": None,
                        "rmssd_items_ms": [40.1116],
                        "heart_rate_items_bpm": [68],
                        "motion_items": None,
                    },
                ),
            ]
        )

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            payload = orch.assemble_segment(b"\x00", [])

        snapshot = payload["_physiological_context"]["operator"]
        assert snapshot is not None
        assert snapshot["rmssd_ms"] == 40.112
        assert snapshot["heart_rate_bpm"] == 67
        assert snapshot["source_kind"] == "session"
        assert snapshot["derivation_method"] == "provider"

    def test_stale_snapshot_marked_but_attached(self) -> None:
        orch = Orchestrator()
        now_wall = 2000.0
        end = now_wall - PHYSIO_STALENESS_THRESHOLD_S - 5
        orch._physio_buffer["operator"].append(
            _buffer_chunk(
                role="operator",
                source_kind="session",
                end_ts=end,
                payload={
                    "sample_interval_s": 60,
                    "valid_sample_count": 0,
                    "expected_sample_count": 4,
                    "derivation_method": "provider",
                    "ibi_ms_items": None,
                    "rmssd_items_ms": [45.0],
                    "heart_rate_items_bpm": [66],
                    "motion_items": None,
                },
            )
        )

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            payload = orch.assemble_segment(b"\x00", [])

        snapshot = payload["_physiological_context"]["operator"]
        assert snapshot is not None
        assert snapshot["is_stale"] is True
        assert snapshot["freshness_s"] == PHYSIO_STALENESS_THRESHOLD_S + 5
        assert snapshot["rmssd_ms"] is None
        assert payload["_segment_id"] == "seg-0001"

    def test_context_omitted_only_when_neither_role_has_state(self) -> None:
        orch = Orchestrator()

        payload = orch.assemble_segment(b"\x00", [])

        assert "_physiological_context" not in payload

        now_wall = 1000.0
        end = now_wall - 30
        orch._physio_buffer["streamer"].append(
            _buffer_chunk(
                role="streamer",
                source_kind="session",
                end_ts=end,
                payload={
                    "sample_interval_s": 60,
                    "valid_sample_count": 0,
                    "expected_sample_count": 4,
                    "derivation_method": "provider",
                    "ibi_ms_items": None,
                    "rmssd_items_ms": [40.0],
                    "heart_rate_items_bpm": [64],
                    "motion_items": None,
                },
            )
        )

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            payload_with_state = orch.assemble_segment(b"\x00", [])

        assert "_physiological_context" in payload_with_state
        assert payload_with_state["_physiological_context"]["streamer"] is not None
        assert payload_with_state["_physiological_context"]["streamer"]["rmssd_ms"] is None
        assert payload_with_state["_physiological_context"]["operator"] is None

    def test_assemble_segment_increments_counter(self) -> None:
        orch = Orchestrator()
        audio = b"\x00" * 100

        p1 = orch.assemble_segment(audio, [])
        p2 = orch.assemble_segment(audio, [])

        assert p1["_segment_id"] == "seg-0001"
        assert p2["_segment_id"] == "seg-0002"

    def test_assemble_segment_applies_drift(self) -> None:
        orch = Orchestrator()
        orch.drift_corrector.drift_offset = 1.0

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.0):
            payload = orch.assemble_segment(b"\x00", [])

        timestamp = datetime.fromisoformat(payload["timestamp_utc"].replace("Z", "+00:00"))
        assert timestamp.timestamp() == 1710000001.0

    def test_reward_fields_unchanged_when_physiology_attached(self) -> None:
        orch = Orchestrator()
        orch._au12_series = [{"timestamp_s": 1.23, "intensity": 0.8}]
        orch._stimulus_time = 456.7
        now_wall = 1000.0
        end = now_wall - 20
        orch._physio_buffer["streamer"].append(
            _buffer_chunk(
                role="streamer",
                source_kind="session",
                end_ts=end,
                payload={
                    "sample_interval_s": 60,
                    "valid_sample_count": 4,
                    "expected_sample_count": 4,
                    "derivation_method": "provider",
                    "ibi_ms_items": None,
                    "rmssd_items_ms": [44.0],
                    "heart_rate_items_bpm": [65],
                    "motion_items": None,
                },
            )
        )

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=now_wall):
            payload = orch.assemble_segment(b"\x00", [])

        assert payload["_au12_series"] == [{"timestamp_s": 1.23, "intensity": 0.8}]
        assert payload["_stimulus_time"] == 456.7
        assert orch._au12_series == []

    def test_stop_cleans_up(self) -> None:
        orch = Orchestrator()
        orch._running = True
        orch.audio_resampler = MagicMock()
        orch.video_capture = MagicMock()
        orch._redis = MagicMock()
        orch._face_mesh = MagicMock()
        orch._au12_series = [{"timestamp_s": 1.0, "intensity": 0.5}]

        redis_client = orch._redis
        face_mesh = orch._face_mesh

        orch.stop()

        assert not orch._running
        orch.audio_resampler.stop.assert_called_once()
        orch.video_capture.stop.assert_called_once()
        redis_client.close.assert_called_once()
        face_mesh.close.assert_called_once()
        assert orch._redis is None
        assert orch._face_mesh is None
        assert orch._au12_series == []
