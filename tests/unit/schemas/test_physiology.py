"""Unit tests for physiological schema models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from packages.schemas.physiology import (
    PhysiologicalContext,
    PhysiologicalPayload,
    PhysiologicalSampleEvent,
    PhysiologicalSnapshot,
)

DRAFT_07_SCHEMA_URI = "http://json-schema.org/draft-07/schema#"


def _sample_timestamp() -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _valid_payload_data(
    rmssd_ms: float | None = 42.5,
    heart_rate_bpm: int | None = 68,
    sample_window_s: int = 300,
) -> dict[str, Any]:
    return {
        "rmssd_ms": rmssd_ms,
        "heart_rate_bpm": heart_rate_bpm,
        "sample_window_s": sample_window_s,
    }


def _valid_event_data(subject_role: str = "streamer") -> dict[str, Any]:
    return {
        "unique_id": uuid.uuid4(),
        "provider": "oura",
        "subject_role": subject_role,
        "source_timestamp_utc": _sample_timestamp(),
        "ingest_timestamp_utc": _sample_timestamp(),
        "payload": _valid_payload_data(),
    }


class TestPhysiologicalSampleEvent:
    def test_valid_event_creation_and_json_roundtrip(self) -> None:
        event = PhysiologicalSampleEvent.model_validate(_valid_event_data())

        json_str = event.model_dump_json()
        restored = PhysiologicalSampleEvent.model_validate_json(json_str)

        assert restored == event
        assert restored.subject_role == "streamer"
        assert restored.event_type == "physiological_sample"
        assert restored.payload.heart_rate_bpm == 68

    def test_subject_role_literal_enforcement_rejects_viewer(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalSampleEvent.model_validate(_valid_event_data(subject_role="viewer"))

    def test_model_json_schema_has_draft07(self) -> None:
        schema = PhysiologicalSampleEvent.model_json_schema()

        assert schema.get("$schema") == DRAFT_07_SCHEMA_URI


class TestPhysiologicalPayload:
    @pytest.mark.parametrize("invalid_bpm", [10, 400])
    def test_hr_bounds_validation_rejects_out_of_range_values(self, invalid_bpm: int) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalPayload(
                rmssd_ms=42.5,
                heart_rate_bpm=invalid_bpm,
                sample_window_s=300,
            )

    def test_rmssd_non_negative_validation(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalPayload(
                rmssd_ms=-0.1,
                heart_rate_bpm=70,
                sample_window_s=300,
            )

    def test_null_rmssd_and_heart_rate_are_accepted(self) -> None:
        payload = PhysiologicalPayload(
            rmssd_ms=None,
            heart_rate_bpm=None,
            sample_window_s=300,
        )

        assert payload.rmssd_ms is None
        assert payload.heart_rate_bpm is None


class TestPhysiologicalSnapshot:
    def test_freshness_non_negative_validation(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalSnapshot(
                rmssd_ms=25.0,
                heart_rate_bpm=64,
                source_timestamp_utc=_sample_timestamp(),
                freshness_s=-1.0,
                is_stale=False,
                provider="oura",
            )


class TestPhysiologicalContext:
    def test_both_null_subjects_are_accepted(self) -> None:
        context = PhysiologicalContext(streamer=None, operator=None)

        assert context.streamer is None
        assert context.operator is None
