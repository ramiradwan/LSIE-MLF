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
from packages.schemas.events import GiftEvent, LiveEvent
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
            SemanticEvaluationResult(
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
