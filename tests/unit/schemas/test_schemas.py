"""
Tests for packages/schemas/ — Phase 0 validation.

Verifies InferenceHandoffPayload, event models, and evaluation schema
conform to their spec sections (§6.1, §4.B, §8.2).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from packages.schemas.evaluation import SemanticEvaluationResult
from packages.schemas.events import ComboEvent, GiftEvent, LiveEvent
from packages.schemas.inference_handoff import InferenceHandoffPayload, MediaSource


class TestInferenceHandoffPayload:
    """§6.1 — InferenceHandoffPayload JSON Schema Draft 07 contract."""

    def test_valid_payload(self, sample_session_id: str, sample_timestamp: datetime) -> None:
        payload = InferenceHandoffPayload(
            session_id=uuid.UUID(sample_session_id),
            timestamp_utc=sample_timestamp,
            media_source=MediaSource(
                stream_url="https://example.com/stream",
                codec="h264",
                resolution=[1920, 1080],
            ),
            segments=[],
        )
        assert str(payload.session_id) == sample_session_id

    def test_invalid_codec_rejected(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        with pytest.raises(ValidationError):
            InferenceHandoffPayload(
                session_id=uuid.UUID(sample_session_id),
                timestamp_utc=sample_timestamp,
                media_source=MediaSource(
                    stream_url="https://example.com/stream",
                    codec="vp9",
                    resolution=[1920, 1080],
                ),
                segments=[],
            )

    def test_resolution_must_be_positive(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        with pytest.raises(ValidationError):
            MediaSource(
                stream_url="https://example.com/stream",
                codec="h264",
                resolution=[0, 1080],
            )

    def test_module_c_to_d_contract_excludes_module_d_acoustic_outputs(self) -> None:
        """Module C → D stays on the stable InferenceHandoffPayload contract."""
        schema = InferenceHandoffPayload.model_json_schema()
        properties = schema["properties"]

        for field in (
            "f0_valid_measure",
            "f0_valid_baseline",
            "perturbation_valid_measure",
            "perturbation_valid_baseline",
            "voiced_coverage_measure_s",
            "voiced_coverage_baseline_s",
            "f0_mean_measure_hz",
            "f0_mean_baseline_hz",
            "f0_delta_semitones",
            "jitter_mean_measure",
            "jitter_mean_baseline",
            "jitter_delta",
            "shimmer_mean_measure",
            "shimmer_mean_baseline",
            "shimmer_delta",
            "pitch_f0",
            "jitter",
            "shimmer",
        ):
            assert field not in properties


class TestSemanticEvaluationResult:
    """§8.2 — additionalProperties: false enforcement."""

    def test_valid_result(self) -> None:
        result = SemanticEvaluationResult(
            reasoning="Semantically equivalent greeting.",
            is_match=True,
            confidence_score=0.95,
        )
        assert result.is_match is True
        assert 0.0 <= result.confidence_score <= 1.0

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationResult(  # type: ignore[call-arg]
                reasoning="Test",
                is_match=True,
                confidence_score=0.5,
                extra_field="should fail",
            )

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationResult(
                reasoning="Test",
                is_match=True,
                confidence_score=1.5,
            )


class TestLiveEvent:
    """§4.B — Ground truth event models."""

    def test_live_event_creation(self) -> None:
        event = LiveEvent(
            uniqueId="user123",
            event_type="gift",
            timestamp_utc=datetime.now(UTC),
        )
        assert event.unique_id == "user123"

    def test_gift_event_value(self) -> None:
        event = GiftEvent(
            uniqueId="user456",
            event_type="gift",
            timestamp_utc=datetime.now(UTC),
            gift_value=100,
        )
        assert event.gift_value == 100


class TestComboEvent:
    """§4.B.1 — Action_Combo constraint validation."""

    def test_combo_requires_at_least_two_events(self) -> None:
        """§4.B.1 — ComboEvent must have min_length=2 events."""
        now = datetime.now(UTC)
        single_event = LiveEvent(
            uniqueId="user1",
            event_type="gift",
            timestamp_utc=now,
        )
        with pytest.raises(ValidationError):
            ComboEvent(
                events=[single_event],
                window_start=now,
                window_end=now,
            )

    def test_valid_combo_event(self) -> None:
        """§4.B.1 — Valid combo with two+ events."""
        now = datetime.now(UTC)
        e1 = LiveEvent(uniqueId="u1", event_type="gift", timestamp_utc=now)
        e2 = LiveEvent(uniqueId="u2", event_type="like", timestamp_utc=now)
        combo = ComboEvent(events=[e1, e2], window_start=now, window_end=now)
        assert len(combo.events) == 2
        assert combo.is_valid is False


class TestInferenceHandoffRoundTrip:
    """§6.1 — JSON Schema Draft 07 export and round-trip serialization."""

    def test_json_roundtrip(self, sample_session_id: str, sample_timestamp: datetime) -> None:
        payload = InferenceHandoffPayload(
            session_id=uuid.UUID(sample_session_id),
            timestamp_utc=sample_timestamp,
            media_source=MediaSource(
                stream_url="https://example.com/stream",
                codec="raw",
                resolution=[640, 480],
            ),
            segments=[{"frame": 1}],
        )
        json_str = payload.model_dump_json()
        restored = InferenceHandoffPayload.model_validate_json(json_str)
        assert restored.session_id == payload.session_id
        assert restored.segments == payload.segments

    def test_json_schema_has_draft07(self) -> None:
        schema = InferenceHandoffPayload.model_json_schema()
        assert schema.get("$schema") == "http://json-schema.org/draft-07/schema#"
