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

import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from packages.schemas.inference_handoff import InferenceHandoffPayload
from packages.schemas.physiology import PhysiologicalChunkEvent, PhysiologicalChunkPayload
from services.worker.pipeline.orchestrator import (
    DRIFT_FREEZE_AFTER_FAILURES,
    DRIFT_RESET_TIMEOUT,
    FFMPEG_RESAMPLE_CMD,
    LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED,
    LIVE_SESSION_STATE_TTL_S,
    MAX_PHYSIO_DRAIN_PER_CYCLE,
    PHYSIO_BUFFER_RETENTION_S,
    PHYSIO_DERIVE_WINDOW_S,
    PHYSIO_STALENESS_THRESHOLD_S,
    SEGMENT_WINDOW_SECONDS,
    AudioResampler,
    DriftCorrector,
    Orchestrator,
)


def _canonical_utc_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _expected_segment_id(payload: dict[str, Any]) -> str:
    stable_identity = "|".join(
        (
            f"{uuid.UUID(payload['session_id'])}",
            _canonical_utc_timestamp(payload["segment_window_start_utc"]),
            _canonical_utc_timestamp(payload["segment_window_end_utc"]),
        )
    )
    return hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()


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
        event_type="physiological_chunk",
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


# DriftCorrector tests moved to tests/unit/desktop_app/test_drift.py
# alongside the class itself (services.desktop_app.drift). The
# orchestrator now only calls correct_timestamp on its corrector
# instance — the poll loop runs in capture_supervisor.


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

    def test_publish_live_session_state_writes_calibration_progress(self) -> None:
        orch = Orchestrator(session_id="11111111-1111-4111-8111-111111111111")
        mock_redis = MagicMock()
        orch._redis = mock_redis
        orch._active_arm = "arm-a"
        orch._expected_greeting = "hello there"
        orch._au12_normalizer = type("_Norm", (), {"calibration_buffer": [0.1] * 12})()

        orch._publish_live_session_state()

        key, raw_payload = mock_redis.set.call_args.args[:2]
        assert key == "operator:live_session:11111111-1111-4111-8111-111111111111"
        assert mock_redis.set.call_args.kwargs["ex"] == LIVE_SESSION_STATE_TTL_S
        payload = json.loads(raw_payload)
        assert payload["active_arm"] == "arm-a"
        assert payload["expected_greeting"] == "hello there"
        assert payload["is_calibrating"] is True
        assert payload["calibration_frames_accumulated"] == 12
        assert payload["calibration_frames_required"] == LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED

    def test_publish_live_session_state_keeps_calibrating_until_stimulus(self) -> None:
        orch = Orchestrator(session_id="22222222-2222-4222-8222-222222222222")
        mock_redis = MagicMock()
        orch._redis = mock_redis
        orch._au12_normalizer = type(
            "_Norm",
            (),
            {"calibration_buffer": [0.1] * LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED},
        )()

        orch._publish_live_session_state()

        payload = json.loads(mock_redis.set.call_args.args[1])
        assert payload["is_calibrating"] is True
        assert payload["calibration_frames_accumulated"] == LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED

        orch._is_calibrating = False
        orch._publish_live_session_state()

        payload = json.loads(mock_redis.set.call_args.args[1])
        assert payload["is_calibrating"] is False

    def test_begin_session_republishes_blank_calibration_for_new_session(self) -> None:
        orch = Orchestrator(session_id="33333333-3333-4333-8333-333333333333")
        mock_redis = MagicMock()
        orch._redis = mock_redis
        orch._register_session = MagicMock()  # type: ignore[method-assign]

        def select_arm() -> None:
            orch._active_arm = "arm-new"
            orch._expected_greeting = "hello new session"

        orch._select_experiment_arm = MagicMock(side_effect=select_arm)  # type: ignore[method-assign]
        old_reset_payload = {
            "active_arm": None,
            "expected_greeting": None,
            "is_calibrating": True,
            "calibration_frames_accumulated": 0,
            "calibration_frames_required": LIVE_SESSION_CALIBRATION_FRAMES_REQUIRED,
        }
        # Simulate the exact JSON having been published for the prior session;
        # lifecycle start must still write it to the new session-scoped key.
        orch._last_live_session_state_payload = json.dumps(old_reset_payload, sort_keys=True)
        orch._active_arm = "arm-old"
        orch._expected_greeting = "old greeting"
        orch._is_calibrating = False
        orch._au12_normalizer = type("_Norm", (), {"calibration_buffer": [0.1] * 99})()

        new_session_id = "44444444-4444-4444-8444-444444444444"
        orch._begin_session(
            session_id=new_session_id,
            stream_url="rtmp://example/new",
            experiment_id="exp-new",
        )

        # FILTER OUT BACKGROUND HEARTBEATS
        session_calls = [
            call
            for call in mock_redis.set.call_args_list
            if str(call.args[0]).startswith("operator:live_session:")
        ]

        # The orchestrator publishes the finalized state with the new arm and reset calibration.
        assert len(session_calls) >= 1

        last_key, last_raw = session_calls[-1].args[:2]
        assert last_key == f"operator:live_session:{new_session_id}"

        last_payload = json.loads(last_raw)
        assert last_payload["active_arm"] == "arm-new"
        assert last_payload["expected_greeting"] == "hello new session"
        assert last_payload["is_calibrating"] is True
        assert last_payload["calibration_frames_accumulated"] == 0

    def test_run_boot_path_begins_session_from_constructor_stream_and_experiment(self) -> None:
        session_id = "55555555-5555-4555-8555-555555555555"
        orch = Orchestrator(
            stream_url="rtmp://example/boot",
            session_id=session_id,
            experiment_id="exp-boot",
        )
        orch._using_replay_capture = True
        orch._start_session_lifecycle_listener = MagicMock()  # type: ignore[method-assign]
        orch._begin_session = MagicMock()  # type: ignore[method-assign]
        orch.audio_resampler = MagicMock()
        orch.audio_resampler.read_chunk.return_value = b""

        async def stop_after_one_tick(_delay: float) -> None:
            orch._running = False

        with patch("services.worker.pipeline.orchestrator.asyncio.sleep", stop_after_one_tick):
            asyncio.run(orch.run())

        orch._start_session_lifecycle_listener.assert_called_once()
        orch._begin_session.assert_called_once_with(
            session_id=session_id,
            stream_url="rtmp://example/boot",
            experiment_id="exp-boot",
        )
        orch.audio_resampler.start.assert_called_once()

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
        assert "segment_id" in payload
        assert "_segment_id" not in payload
        assert payload["segment_id"] == _expected_segment_id(payload)
        assert payload["_experiment_id"] == 0
        assert payload["_bandit_decision_snapshot"]["experiment_id"] == 0
        # WS3 P2: assemble_segment no longer base64-encodes _audio_data;
        # the desktop IPC path moves audio through SharedMemory.
        assert payload["_audio_data"] == audio

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
        assert _canonical_utc_timestamp(snapshot["source_timestamp_utc"]) == (
            datetime.fromtimestamp(ibi_end, tz=UTC).isoformat().replace("+00:00", "Z")
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
        assert _canonical_utc_timestamp(snapshot["source_timestamp_utc"]) == (
            datetime.fromtimestamp(end, tz=UTC).isoformat().replace("+00:00", "Z")
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
        assert payload["segment_id"] == _expected_segment_id(payload)
        assert "_segment_id" not in payload

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

        assert p1["segment_id"] == _expected_segment_id(p1)
        assert p2["segment_id"] == _expected_segment_id(p2)
        assert p1["segment_id"] != p2["segment_id"]
        p1_end = datetime.fromisoformat(p1["segment_window_end_utc"].replace("Z", "+00:00"))
        p2_start = datetime.fromisoformat(p2["segment_window_start_utc"].replace("Z", "+00:00"))
        assert (p2_start - p1_end).total_seconds() == 0
        assert (
            datetime.fromisoformat(p1["segment_window_end_utc"].replace("Z", "+00:00"))
            - datetime.fromisoformat(p1["segment_window_start_utc"].replace("Z", "+00:00"))
        ).total_seconds() == SEGMENT_WINDOW_SECONDS

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

    def test_flush_inflight_segment_dispatches_partial_payload(self) -> None:
        orch = Orchestrator()
        orch._audio_buffer = bytearray(b"partial-audio")
        orch.event_buffer.append({"event_type": "gift", "uniqueId": "u1"})
        orch.assemble_segment = MagicMock(return_value={"payload": "ok"})  # type: ignore[method-assign]
        orch._dispatch_payload = MagicMock()  # type: ignore[method-assign]
        orch._drain_physio_events = MagicMock()  # type: ignore[method-assign]

        orch._flush_inflight_segment()

        orch._drain_physio_events.assert_called_once()
        orch.assemble_segment.assert_called_once_with(
            b"partial-audio",
            [{"event_type": "gift", "uniqueId": "u1"}],
        )
        orch._dispatch_payload.assert_called_once_with({"payload": "ok"})
        assert orch._audio_buffer == bytearray()
        assert list(orch.event_buffer) == []

    def test_flush_inflight_segment_discards_invalid_assembly(self, caplog: Any) -> None:
        session_id = "99999999-9999-4999-8999-999999999999"
        orch = Orchestrator(session_id=session_id)
        failed_au12 = {"timestamp_s": 1.0, "intensity": 1.5}
        orch._audio_buffer = bytearray(b"partial-audio")
        orch.event_buffer.append({"event_type": "gift", "uniqueId": "u1"})
        orch._au12_series = [failed_au12]
        original_assemble_segment = orch.assemble_segment
        orch.assemble_segment = MagicMock(wraps=original_assemble_segment)  # type: ignore[method-assign]
        orch._dispatch_payload = MagicMock()  # type: ignore[method-assign]
        orch._drain_physio_events = MagicMock()  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger="services.worker.pipeline.orchestrator"):
            orch._flush_inflight_segment()

        orch._drain_physio_events.assert_called_once()
        orch.assemble_segment.assert_called_once_with(
            b"partial-audio",
            [{"event_type": "gift", "uniqueId": "u1"}],
        )
        orch._dispatch_payload.assert_not_called()
        assert orch._audio_buffer == bytearray()
        assert list(orch.event_buffer) == []
        assert orch._au12_series == []
        assert "Discarding invalid assembled handoff segment" in caplog.text
        assert "source=flush_inflight" in caplog.text
        assert session_id in caplog.text
        assert "intensity" in caplog.text

    def test_run_discards_invalid_segment_and_dispatches_later_valid_work(
        self,
        caplog: Any,
    ) -> None:
        session_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        orch = Orchestrator(session_id=session_id)
        orch._using_replay_capture = True
        orch._redis = None
        orch._start_session_lifecycle_listener = MagicMock()  # type: ignore[method-assign]
        orch._publish_live_session_state = MagicMock()  # type: ignore[method-assign]
        orch._publish_orchestrator_heartbeat = MagicMock()  # type: ignore[method-assign]
        orch._publish_orchestrator_heartbeat_if_due = MagicMock()  # type: ignore[method-assign]
        orch._drain_session_lifecycle_intents = MagicMock()  # type: ignore[method-assign]
        orch._process_video_frame = MagicMock()  # type: ignore[method-assign]
        orch._drain_physio_events = MagicMock()  # type: ignore[method-assign]

        def begin_session(**_kwargs: Any) -> None:
            orch._session_active = True

        orch._begin_session = MagicMock(side_effect=begin_session)  # type: ignore[method-assign]
        segment_bytes = 16000 * 2 * SEGMENT_WINDOW_SECONDS
        first_audio = b"\x00" * segment_bytes
        second_audio = b"\x01" * segment_bytes
        failed_au12 = {"timestamp_s": 1.0, "intensity": 1.5}
        orch._au12_series = [failed_au12]
        orch.audio_resampler = MagicMock()
        read_chunks = deque([first_audio, second_audio, b""])
        orch.audio_resampler.read_chunk.side_effect = lambda _chunk_size: (
            read_chunks.popleft() if read_chunks else b""
        )
        bad_segment_event = {"event_type": "gift", "uniqueId": "bad-segment-event"}
        orch.event_buffer.append(bad_segment_event)
        original_assemble_segment = orch.assemble_segment
        orch.assemble_segment = MagicMock(wraps=original_assemble_segment)  # type: ignore[method-assign]

        def dispatch_payload(_payload: dict[str, Any]) -> None:
            orch._running = False

        orch._dispatch_payload = MagicMock(side_effect=dispatch_payload)  # type: ignore[method-assign]

        sleep_count = 0

        async def fast_sleep(_delay: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                orch._running = False

        with (
            caplog.at_level(logging.WARNING, logger="services.worker.pipeline.orchestrator"),
            patch("services.worker.pipeline.orchestrator.asyncio.sleep", fast_sleep),
        ):
            asyncio.run(orch.run())

        assert orch.assemble_segment.call_count == 2
        first_call, second_call = orch.assemble_segment.call_args_list
        assert first_call.args == (
            first_audio,
            [bad_segment_event],
        )
        assert second_call.args == (second_audio, [])
        orch._dispatch_payload.assert_called_once()
        dispatched_payload = orch._dispatch_payload.call_args.args[0]
        assert dispatched_payload["_au12_series"] == []
        assert failed_au12 not in dispatched_payload["_au12_series"]
        assert orch._au12_series == []
        assert "Discarding invalid assembled handoff segment" in caplog.text
        assert "source=run_dispatch" in caplog.text
        assert session_id in caplog.text
        assert "intensity" in caplog.text

    def test_start_intent_rotates_active_session(self) -> None:
        original_session_id = str(uuid.uuid4())
        orch = Orchestrator(
            stream_url="https://example.com/original",
            session_id=original_session_id,
            experiment_id="exp-original",
        )
        orch._session_active = True
        orch._segment_counter = 8
        orch._audio_buffer = bytearray(b"old")
        orch.event_buffer.append({"event_type": "comment"})
        orch._au12_series = [{"timestamp_s": 1.0, "intensity": 0.5}]
        orch._flush_inflight_segment = MagicMock()  # type: ignore[method-assign]
        orch._mark_session_ended = MagicMock()  # type: ignore[method-assign]
        orch._register_session = MagicMock()  # type: ignore[method-assign]
        orch._select_experiment_arm = MagicMock()  # type: ignore[method-assign]

        new_session_id = str(uuid.uuid4())
        orch._handle_session_lifecycle_intent(
            {
                "action": "start",
                "session_id": new_session_id,
                "stream_url": "https://example.com/new",
                "experiment_id": "exp-new",
            }
        )

        orch._flush_inflight_segment.assert_called_once()
        orch._mark_session_ended.assert_called_once_with(original_session_id)
        orch._register_session.assert_called_once()
        orch._select_experiment_arm.assert_called_once()
        assert orch._session_active is True
        assert orch._session_id == new_session_id
        assert orch._stream_url == "https://example.com/new"
        assert orch._experiment_id == "exp-new"
        assert orch._segment_counter == 0
        assert orch._au12_series == []
        assert list(orch.event_buffer) == []

    def test_end_intent_flushes_and_marks_session_ended(self) -> None:
        session_id = str(uuid.uuid4())
        orch = Orchestrator(
            stream_url="https://example.com/live",
            session_id=session_id,
            experiment_id="exp-1",
        )
        orch._session_active = True
        orch._flush_inflight_segment = MagicMock()  # type: ignore[method-assign]
        orch._mark_session_ended = MagicMock()  # type: ignore[method-assign]

        orch._handle_session_lifecycle_intent(
            {
                "action": "end",
                "session_id": session_id,
                "stream_url": "https://example.com/live",
                "experiment_id": "exp-1",
            }
        )

        orch._flush_inflight_segment.assert_called_once()
        orch._mark_session_ended.assert_called_once_with(session_id)
        assert orch._session_active is False
        assert orch._segment_counter == 0
        assert orch._au12_series == []

    def test_end_intent_for_non_active_session_is_ignored(self) -> None:
        active_session_id = str(uuid.uuid4())
        orch = Orchestrator(
            stream_url="https://example.com/live",
            session_id=active_session_id,
            experiment_id="exp-1",
        )
        orch._session_active = True
        orch._flush_inflight_segment = MagicMock()  # type: ignore[method-assign]
        orch._mark_session_ended = MagicMock()  # type: ignore[method-assign]

        orch._handle_session_lifecycle_intent(
            {
                "action": "end",
                "session_id": str(uuid.uuid4()),
                "stream_url": "https://example.com/live",
                "experiment_id": "exp-1",
            }
        )

        orch._flush_inflight_segment.assert_not_called()
        orch._mark_session_ended.assert_not_called()
        assert orch._session_active is True

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

    @pytest.mark.audit_item("13.25")
    def test_segment_id_replay_stable_for_identical_window_boundaries(self) -> None:
        session_id = "66666666-6666-4666-8666-666666666666"
        anchor = datetime(2026, 3, 13, 12, 0, 0, tzinfo=UTC)

        first = Orchestrator(session_id=session_id)
        first._segment_window_anchor_utc = anchor
        second = Orchestrator(session_id=session_id)
        second._segment_window_anchor_utc = anchor

        with patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000000.0):
            first_payload = first.assemble_segment(b"\x00", [])
        with patch("services.worker.pipeline.orchestrator.time.time", return_value=1710000999.0):
            second_payload = second.assemble_segment(b"\x00", [])

        expected_serialization = "|".join(
            (
                session_id,
                _canonical_utc_timestamp(first_payload["segment_window_start_utc"]),
                _canonical_utc_timestamp(first_payload["segment_window_end_utc"]),
            )
        )
        unseparated_serialization = expected_serialization.replace("|", "")

        assert (
            first_payload["segment_window_start_utc"] == second_payload["segment_window_start_utc"]
        )
        assert first_payload["segment_window_end_utc"] == second_payload["segment_window_end_utc"]
        assert first_payload["segment_id"] == second_payload["segment_id"]
        assert first_payload["segment_id"] == _expected_segment_id(first_payload)
        assert (
            first_payload["segment_id"]
            == hashlib.sha256(expected_serialization.encode("utf-8")).hexdigest()
        )
        assert (
            first_payload["segment_id"]
            != hashlib.sha256(unseparated_serialization.encode("utf-8")).hexdigest()
        )
        assert first_payload["timestamp_utc"] != second_payload["timestamp_utc"]

    @pytest.mark.audit_item("13.26")
    def test_bandit_snapshot_copies_pre_update_state_and_omits_absent_optionals(self) -> None:
        orch = Orchestrator(session_id="77777777-7777-4777-8777-777777777777")
        orch._active_arm = "arm_a"
        orch._expected_greeting = "hello before update"
        orch._experiment_row_id = 17
        selection_time = datetime(2026, 3, 13, 12, 0, 0, tzinfo=UTC)
        candidate_arm_ids = ["arm_a", "arm_b"]
        posterior_by_arm = {
            "arm_a": {"alpha": 2.0, "beta": 3.0},
            "arm_b": {"alpha": 4.0, "beta": 5.0},
        }

        orch._capture_bandit_decision_snapshot(
            selection_time_utc=selection_time,
            candidate_arm_ids=candidate_arm_ids,
            posterior_by_arm=posterior_by_arm,
            sampled_theta_by_arm=None,
        )
        candidate_arm_ids.append("arm_c")
        posterior_by_arm["arm_a"]["alpha"] = 99.0

        snapshot = orch._bandit_decision_snapshot
        assert snapshot is not None
        assert snapshot["selection_method"] == "thompson_sampling"
        assert snapshot["selection_time_utc"] == selection_time
        assert snapshot["experiment_id"] == 17
        assert snapshot["policy_version"] == "thompson_sampling_v1"
        assert snapshot["selected_arm_id"] == "arm_a"
        assert snapshot["candidate_arm_ids"] == ["arm_a", "arm_b"]
        assert snapshot["posterior_by_arm"]["arm_a"] == {"alpha": 2.0, "beta": 3.0}
        assert snapshot["expected_greeting"] == "hello before update"
        assert len(snapshot["decision_context_hash"]) == 64
        assert "sampled_theta_by_arm" not in snapshot
        assert "random_seed" not in snapshot

        payload = orch.assemble_segment(b"\x00", [])
        payload_snapshot = payload["_bandit_decision_snapshot"]
        assert payload_snapshot["posterior_by_arm"]["arm_a"] == {"alpha": 2.0, "beta": 3.0}
        assert "sampled_theta_by_arm" not in payload_snapshot
        assert "random_seed" not in payload_snapshot

    def test_dispatch_payload_validates_model_and_omits_ineligible_physio(self) -> None:
        """WS3 P2: dispatch validates handoff and pushes IPC control message.

        The v3.4 Celery + base64 path is retired; the segment now travels
        as a SharedMemory PCM block plus an ``InferenceControlMessage``.
        ``_physiological_context`` with all-None roles is still pruned by
        ``sanitize_json_payload``.
        """
        from queue import Queue as ThreadQueue

        from services.desktop_app.ipc.control_messages import InferenceControlMessage
        from services.desktop_app.ipc.shared_buffers import (
            PcmBlockMetadata,
            read_pcm_block,
        )

        ipc_queue: ThreadQueue[Any] = ThreadQueue()
        orch = Orchestrator(
            session_id="88888888-8888-4888-8888-888888888888",
            ipc_queue=ipc_queue,
        )
        try:
            payload = orch.assemble_segment(b"\x01\x02", [])
            payload["_physiological_context"] = {"streamer": None, "operator": None}

            orch._dispatch_payload(payload)

            assert ipc_queue.qsize() == 1
            raw = ipc_queue.get_nowait()
            msg = InferenceControlMessage.model_validate(raw)

            InferenceHandoffPayload.model_validate(msg.handoff)
            assert "_physiological_context" not in msg.handoff
            assert "sampled_theta_by_arm" not in msg.handoff["_bandit_decision_snapshot"]
            assert "random_seed" not in msg.handoff["_bandit_decision_snapshot"]

            recovered = read_pcm_block(
                PcmBlockMetadata(
                    name=msg.audio.name,
                    byte_length=msg.audio.byte_length,
                    sha256=msg.audio.sha256,
                )
            )
            assert recovered == b"\x01\x02"
        finally:
            orch.close_inflight_blocks()
