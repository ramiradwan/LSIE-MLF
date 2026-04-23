"""Unit tests for physiological schema models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from packages.schemas.physiology import (
    PhysiologicalChunkEvent,
    PhysiologicalChunkPayload,
    PhysiologicalContext,
    PhysiologicalSnapshot,
)

DRAFT_07_SCHEMA_URI = "http://json-schema.org/draft-07/schema#"


def _sample_timestamp() -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _valid_payload_data(**overrides: Any) -> dict[str, Any]:
    payload = {
        "sample_interval_s": 300,
        "valid_sample_count": 5,
        "expected_sample_count": 5,
        "derivation_method": "provider",
        "ibi_ms_items": [810.0, 820.0, 830.0],
    }
    payload.update(overrides)
    return payload


def _valid_event_data(**overrides: Any) -> dict[str, Any]:
    event = {
        "unique_id": uuid.uuid4(),
        "provider": "oura",
        "subject_role": "streamer",
        "source_kind": "ibi",
        "window_start_utc": _sample_timestamp(),
        "window_end_utc": _sample_timestamp(),
        "ingest_timestamp_utc": _sample_timestamp(),
        "payload": _valid_payload_data(),
    }
    event.update(overrides)
    return event


def _valid_snapshot_data(**overrides: Any) -> dict[str, Any]:
    snapshot = {
        "rmssd_ms": 42.5,
        "heart_rate_bpm": 68,
        "source_timestamp_utc": _sample_timestamp(),
        "freshness_s": 30.0,
        "is_stale": False,
        "provider": "oura",
        "source_kind": "ibi",
        "derivation_method": "server",
        "window_s": 300,
        "validity_ratio": 0.9,
        "is_valid": True,
    }
    snapshot.update(overrides)
    return snapshot


class TestPhysiologicalChunkEvent:
    def test_valid_event_creation(self) -> None:
        event = PhysiologicalChunkEvent.model_validate(_valid_event_data())

        assert event.event_type == "physiological_chunk"
        assert event.source_kind == "ibi"
        assert event.payload.ibi_ms_items == [810.0, 820.0, 830.0]

    def test_event_literal_constraints_reject_invalid_values(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalChunkEvent.model_validate(_valid_event_data(subject_role="viewer"))

        with pytest.raises(ValidationError):
            PhysiologicalChunkEvent.model_validate(_valid_event_data(source_kind="daily"))

    def test_event_forbids_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalChunkEvent.model_validate(_valid_event_data(unexpected_field=True))

    def test_model_json_schema_has_draft07(self) -> None:
        schema = PhysiologicalChunkEvent.model_json_schema()

        assert schema.get("$schema") == DRAFT_07_SCHEMA_URI


class TestPhysiologicalChunkPayload:
    def test_payload_forbids_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalChunkPayload.model_validate(_valid_payload_data(unexpected_field=True))

    def test_payload_requires_at_least_one_primary_series(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalChunkPayload.model_validate(
                _valid_payload_data(
                    ibi_ms_items=None,
                    rmssd_items_ms=None,
                    heart_rate_items_bpm=None,
                )
            )

    def test_payload_rejects_zero_ibi_item(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalChunkPayload.model_validate(_valid_payload_data(ibi_ms_items=[0.0]))

    def test_payload_accepts_zero_rmssd_item(self) -> None:
        payload = PhysiologicalChunkPayload.model_validate(
            _valid_payload_data(
                ibi_ms_items=None,
                rmssd_items_ms=[0.0, 31.2],
                heart_rate_items_bpm=[65, 66],
                derivation_method="server",
            )
        )

        assert payload.rmssd_items_ms == [0.0, 31.2]

    def test_payload_accepts_numeric_motion_items(self) -> None:
        payload = PhysiologicalChunkPayload.model_validate(
            _valid_payload_data(motion_items=[0.0, 1.25, 2.5])
        )

        assert payload.motion_items == [0.0, 1.25, 2.5]

    def test_payload_rejects_negative_motion_items(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalChunkPayload.model_validate(_valid_payload_data(motion_items=[-0.1]))

    def test_payload_accepts_alternate_primary_series(self) -> None:
        payload = PhysiologicalChunkPayload.model_validate(
            _valid_payload_data(
                ibi_ms_items=None,
                rmssd_items_ms=[30.5, 31.2],
                heart_rate_items_bpm=[65, 66],
                derivation_method="server",
            )
        )

        assert payload.derivation_method == "server"
        assert payload.rmssd_items_ms == [30.5, 31.2]
        assert payload.heart_rate_items_bpm == [65, 66]


class TestPhysiologicalSnapshot:
    def test_snapshot_requires_enriched_metadata_fields(self) -> None:
        snapshot = PhysiologicalSnapshot.model_validate(_valid_snapshot_data())

        assert snapshot.source_kind == "ibi"
        assert snapshot.derivation_method == "server"
        assert snapshot.window_s == 300
        assert snapshot.validity_ratio == pytest.approx(0.9)
        assert snapshot.is_valid is True

    def test_snapshot_forbids_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalSnapshot.model_validate(_valid_snapshot_data(unexpected_field=True))

    def test_freshness_non_negative_validation(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalSnapshot.model_validate(_valid_snapshot_data(freshness_s=-1.0))

    def test_validity_ratio_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            PhysiologicalSnapshot.model_validate(_valid_snapshot_data(validity_ratio=1.5))


class TestPhysiologicalContext:
    def test_context_accepts_missing_roles(self) -> None:
        context = PhysiologicalContext.model_validate({})

        assert context.streamer is None
        assert context.operator is None

    def test_context_allows_single_role_snapshot(self) -> None:
        context = PhysiologicalContext.model_validate(
            {"streamer": _valid_snapshot_data(), "operator": None}
        )

        assert context.streamer is not None
        assert context.operator is None
