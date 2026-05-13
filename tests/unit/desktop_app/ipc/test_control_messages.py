"""InferenceControlMessage Pydantic validation tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from services.desktop_app.ipc.control_messages import (
    AnalyticsResultMessage,
    AudioBlockRef,
    InferenceControlMessage,
    LiveSessionControlMessage,
    PcmBlockAckMessage,
    VisualAnalyticsStateMessage,
)
from services.desktop_app.ipc.shared_buffers import PcmBlockMetadata


def _valid_audio_ref() -> AudioBlockRef:
    return AudioBlockRef(
        name="lsie_ipc_pcm_abcdef0123456789",
        byte_length=960_000,
        sha256="0" * 64,
    )


def _stimulus_definition() -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text="hello creator",
        ),
        expected_stimulus_rule=(
            "Deliver the spoken greeting to the live streamer exactly as written."
        ),
        expected_response_rule=(
            "The live streamer acknowledges the greeting or responds to it on stream."
        ),
    )


def _bandit_snapshot(now: datetime) -> dict[str, Any]:
    stimulus_definition = _stimulus_definition()
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": now,
        "experiment_id": 7,
        "policy_version": "desktop-v1",
        "selected_arm_id": "arm_a",
        "candidate_arm_ids": ["arm_a", "arm_b"],
        "posterior_by_arm": {
            "arm_a": {"alpha": 2.0, "beta": 1.0},
            "arm_b": {"alpha": 1.0, "beta": 2.0},
        },
        "sampled_theta_by_arm": {"arm_a": 0.9, "arm_b": 0.1},
        "stimulus_modality": stimulus_definition.stimulus_modality,
        "stimulus_payload": stimulus_definition.stimulus_payload.model_dump(mode="json"),
        "expected_stimulus_rule": stimulus_definition.expected_stimulus_rule,
        "expected_response_rule": stimulus_definition.expected_response_rule,
        "decision_context_hash": "1" * 64,
        "random_seed": 42,
    }


def _valid_handoff(now: datetime) -> dict[str, Any]:
    stimulus_definition = _stimulus_definition()
    return {
        "session_id": "00000000-0000-4000-8000-000000000001",
        "segment_id": "a" * 64,
        "segment_window_start_utc": now,
        "segment_window_end_utc": now + timedelta(seconds=30),
        "timestamp_utc": now + timedelta(seconds=31),
        "media_source": {
            "stream_url": "https://example.com/stream",
            "codec": "h264",
            "resolution": [1920, 1080],
        },
        "segments": [],
        "_active_arm": "arm_a",
        "_experiment_id": 7,
        "_stimulus_modality": stimulus_definition.stimulus_modality,
        "_stimulus_payload": stimulus_definition.stimulus_payload.model_dump(mode="json"),
        "_expected_stimulus_rule": stimulus_definition.expected_stimulus_rule,
        "_expected_response_rule": stimulus_definition.expected_response_rule,
        "_stimulus_time": now.timestamp() + 3.0,
        "_au12_series": [{"timestamp_s": now.timestamp() + 4.0, "intensity": 0.8}],
        "_bandit_decision_snapshot": _bandit_snapshot(now),
    }


def _valid_analytics_message() -> AnalyticsResultMessage:
    now = datetime(2026, 5, 2, tzinfo=UTC)
    return AnalyticsResultMessage.model_validate(
        {
            "message_id": uuid.UUID("00000000-0000-4000-8000-000000000002"),
            "handoff": _valid_handoff(now),
            "semantic": {
                "reasoning": "cross_encoder_high_match",
                "is_match": True,
                "confidence_score": 0.91,
                "semantic_method": "cross_encoder",
                "semantic_method_version": "ce-v1",
            },
            "transcription": "hello creator",
            "reward": {
                "gated_reward": 0.8,
                "p90_intensity": 0.8,
                "semantic_gate": 1,
                "n_frames_in_window": 1,
                "au12_baseline_pre": None,
                "stimulus_time": now.timestamp() + 3.0,
            },
            "acoustic": {
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
            },
        }
    )


def test_audio_ref_metadata_round_trip() -> None:
    metadata = PcmBlockMetadata(
        name="lsie_ipc_pcm_abc",
        byte_length=42,
        sha256="a" * 64,
    )
    ref = AudioBlockRef.from_metadata(metadata)
    assert ref.to_metadata() == metadata


def test_audio_ref_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError):
        AudioBlockRef(name="ok", byte_length=1, sha256="not-a-hash")


def test_audio_ref_rejects_zero_byte_length() -> None:
    with pytest.raises(ValidationError):
        AudioBlockRef(name="ok", byte_length=0, sha256="0" * 64)


def test_audio_ref_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AudioBlockRef(
            name="ok",
            byte_length=1,
            sha256="0" * 64,
            unexpected="should fail",  # type: ignore[call-arg]
        )


def test_control_message_validates_minimal_handoff() -> None:
    msg = InferenceControlMessage(
        handoff={"segment_id": "abc"},
        audio=_valid_audio_ref(),
    )
    assert msg.forward_fields == {}
    assert msg.audio.byte_length == 960_000


def test_control_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        InferenceControlMessage(
            handoff={},
            audio=_valid_audio_ref(),
            stowaway="forbidden",  # type: ignore[call-arg]
        )


def test_control_message_round_trips_through_dump_validate() -> None:
    """Mirrors the IPC path: producer dumps, queue ships, consumer validates."""
    msg = InferenceControlMessage(
        handoff={"segment_id": "deadbeef"},
        audio=_valid_audio_ref(),
        forward_fields={"_experiment_code": "greeting_line_v1"},
    )
    dumped = msg.model_dump(mode="json")
    re_validated = InferenceControlMessage.model_validate(dumped)
    assert re_validated == msg


def test_pcm_block_ack_message_round_trips_through_dump_validate() -> None:
    msg = PcmBlockAckMessage(name="lsie_ipc_pcm_abcdef0123456789")

    dumped = msg.model_dump(mode="json")
    re_validated = PcmBlockAckMessage.model_validate(dumped)

    assert re_validated == msg
    assert dumped["schema_version"] == "ws5.p4.pcm_block_ack.v1"


def test_live_session_control_message_round_trips_through_dump_validate() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    msg = LiveSessionControlMessage(
        action="stimulus",
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="desktop-capture://current",
        experiment_id="greeting_line_v1",
        active_arm="warm_welcome",
        stimulus_definition=_stimulus_definition(),
        stimulus_time_s=now.timestamp(),
        timestamp_utc=now,
    )

    dumped = msg.model_dump(mode="json")
    re_validated = LiveSessionControlMessage.model_validate(dumped)

    assert re_validated == msg
    assert dumped["schema_version"] == "ws5.p4.live_session_control.v1"
    assert "_audio_data" not in dumped
    assert "_frame_data" not in dumped


def test_live_session_control_message_rejects_inconsistent_action_fields() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)

    with pytest.raises(ValidationError):
        LiveSessionControlMessage(
            action="start",
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            timestamp_utc=now,
        )
    with pytest.raises(ValidationError):
        LiveSessionControlMessage(
            action="stimulus",
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            timestamp_utc=now,
        )
    with pytest.raises(ValidationError):
        LiveSessionControlMessage(
            action="start",
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            stream_url="desktop-capture://current",
            stimulus_time_s=now.timestamp(),
            timestamp_utc=now,
        )


def test_live_session_control_message_rejects_raw_media_fields() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    dumped = LiveSessionControlMessage(
        action="start",
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        stream_url="desktop-capture://current",
        timestamp_utc=now,
    ).model_dump(mode="json")
    dumped["_frame_data"] = "raw-media-must-not-cross"

    with pytest.raises(ValidationError):
        LiveSessionControlMessage.model_validate(dumped)


def test_visual_analytics_state_message_round_trips_through_dump_validate() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    msg = VisualAnalyticsStateMessage(
        message_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        timestamp_utc=now,
        face_present=True,
        is_calibrating=False,
        calibration_frames_accumulated=45,
        calibration_frames_required=45,
        latest_au12_intensity=0.82,
        latest_au12_timestamp_s=now.timestamp(),
        status="ready",
    )

    dumped = msg.model_dump(mode="json")
    re_validated = VisualAnalyticsStateMessage.model_validate(dumped)

    assert re_validated == msg
    assert dumped["schema_version"] == "ws5.p4.visual_analytics_state.v1"
    assert "_audio_data" not in dumped
    assert "_frame_data" not in dumped


def test_visual_analytics_state_message_rejects_invalid_calibration_state() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)

    with pytest.raises(ValidationError):
        VisualAnalyticsStateMessage(
            message_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            timestamp_utc=now,
            face_present=True,
            is_calibrating=True,
            calibration_frames_accumulated=46,
            calibration_frames_required=45,
            status="calibrating",
        )
    with pytest.raises(ValidationError):
        VisualAnalyticsStateMessage(
            message_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            timestamp_utc=now,
            face_present=True,
            is_calibrating=True,
            calibration_frames_accumulated=44,
            calibration_frames_required=45,
            status="ready",
        )
    with pytest.raises(ValidationError):
        VisualAnalyticsStateMessage(
            message_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            timestamp_utc=now,
            face_present=False,
            is_calibrating=False,
            calibration_frames_accumulated=45,
            calibration_frames_required=45,
            status="ready",
        )
    with pytest.raises(ValidationError):
        VisualAnalyticsStateMessage(
            message_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
            session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
            timestamp_utc=now,
            face_present=True,
            is_calibrating=True,
            calibration_frames_accumulated=45,
            calibration_frames_required=45,
            status="ready",
        )


def test_visual_analytics_state_message_rejects_raw_media_fields() -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    dumped = VisualAnalyticsStateMessage(
        message_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        timestamp_utc=now,
        face_present=False,
        is_calibrating=True,
        calibration_frames_accumulated=0,
        calibration_frames_required=45,
        status="waiting_for_face",
    ).model_dump(mode="json")
    dumped["_audio_data"] = "raw-media-must-not-cross"

    with pytest.raises(ValidationError):
        VisualAnalyticsStateMessage.model_validate(dumped)


def test_analytics_result_message_round_trips_through_dump_validate() -> None:
    msg = _valid_analytics_message()

    dumped = msg.model_dump(mode="json")
    re_validated = AnalyticsResultMessage.model_validate(dumped)

    assert re_validated == msg
    assert dumped["schema_version"] == "ws5.p4.analytics_result.v1"
    assert "_audio_data" not in dumped
    assert "_frame_data" not in dumped


def test_analytics_result_message_rejects_raw_media_fields() -> None:
    dumped = _valid_analytics_message().model_dump(mode="json")
    dumped["_audio_data"] = "raw-media-must-not-cross"

    with pytest.raises(ValidationError):
        AnalyticsResultMessage.model_validate(dumped)


def test_analytics_result_message_rejects_unbounded_semantic_reasoning() -> None:
    dumped = _valid_analytics_message().model_dump(mode="json")
    dumped["semantic"]["reasoning"] = "the creator greeted us with arbitrary text"

    with pytest.raises(ValidationError):
        AnalyticsResultMessage.model_validate(dumped)


def test_analytics_result_message_rejects_reversed_window_bounds() -> None:
    dumped = _valid_analytics_message().model_dump(mode="json")
    start = dumped["handoff"]["segment_window_start_utc"]
    dumped["handoff"]["segment_window_start_utc"] = dumped["handoff"]["segment_window_end_utc"]
    dumped["handoff"]["segment_window_end_utc"] = start

    with pytest.raises(ValidationError):
        AnalyticsResultMessage.model_validate(dumped)
