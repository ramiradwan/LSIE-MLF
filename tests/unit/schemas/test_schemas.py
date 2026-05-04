"""
Tests for packages/schemas/ contracts.

Verifies InferenceHandoffPayload, event models, semantic evaluation, and
attribution ledger schemas conform to their v3.4 contract surfaces.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import ValidationError

from packages.schemas.attribution import (
    AttributionEvent,
    AttributionScore,
    BanditDecisionSnapshot,
    EventOutcomeLink,
    OutcomeEvent,
)
from packages.schemas.evaluation import SEMANTIC_REASON_CODES, SemanticEvaluationResult
from packages.schemas.events import ComboEvent, GiftEvent, LiveEvent
from packages.schemas.inference_handoff import InferenceHandoffPayload, MediaSource

SEGMENT_ID = "a" * 64
EXPECTED_RULE_TEXT_HASH = "b" * 64
DECISION_CONTEXT_HASH = "c" * 64


def _bandit_snapshot_data(sample_timestamp: datetime) -> dict[str, Any]:
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": sample_timestamp,
        "experiment_id": 101,
        "policy_version": "ts-v1",
        "selected_arm_id": "arm_a",
        "candidate_arm_ids": ["arm_a", "arm_b"],
        "posterior_by_arm": {
            "arm_a": {"alpha": 2.0, "beta": 3.0},
            "arm_b": {"alpha": 1.0, "beta": 1.0},
        },
        "sampled_theta_by_arm": {"arm_a": 0.72, "arm_b": 0.44},
        "expected_greeting": "Say hello to the creator",
        "decision_context_hash": DECISION_CONTEXT_HASH,
        "random_seed": 42,
    }


def _physiological_snapshot_data(sample_timestamp: datetime) -> dict[str, Any]:
    return {
        "rmssd_ms": 42.5,
        "heart_rate_bpm": 68,
        "source_timestamp_utc": sample_timestamp,
        "freshness_s": 30.0,
        "is_stale": False,
        "provider": "oura",
        "source_kind": "ibi",
        "derivation_method": "server",
        "window_s": 300,
        "validity_ratio": 0.9,
        "is_valid": True,
    }


def _handoff_payload_data(
    sample_session_id: str,
    sample_timestamp: datetime,
    **overrides: Any,
) -> dict[str, Any]:
    payload = {
        "session_id": sample_session_id,
        "segment_id": SEGMENT_ID,
        "segment_window_start_utc": sample_timestamp,
        "segment_window_end_utc": sample_timestamp,
        "timestamp_utc": sample_timestamp,
        "media_source": {
            "stream_url": "https://example.com/stream",
            "codec": "h264",
            "resolution": [1920, 1080],
        },
        "segments": [],
        "_active_arm": "arm_a",
        "_experiment_id": 101,
        "_expected_greeting": "Say hello to the creator",
        "_stimulus_time": None,
        "_au12_series": [{"timestamp_s": 0.0, "intensity": 0.62}],
        "_bandit_decision_snapshot": _bandit_snapshot_data(sample_timestamp),
    }
    payload.update(overrides)
    return payload


def _attribution_event_data(sample_timestamp: datetime) -> dict[str, Any]:
    return {
        "event_id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "segment_id": SEGMENT_ID,
        "event_type": "greeting_interaction",
        "event_time_utc": sample_timestamp,
        "stimulus_time_utc": sample_timestamp,
        "selected_arm_id": "arm_a",
        "expected_rule_text_hash": EXPECTED_RULE_TEXT_HASH,
        "semantic_method": "cross_encoder",
        "semantic_method_version": "cross-encoder-v1",
        "semantic_p_match": 0.94,
        "semantic_reason_code": "cross_encoder_high_match",
        "reward_path_version": "reward-v3.4",
        "bandit_decision_snapshot": _bandit_snapshot_data(sample_timestamp),
        "evidence_flags": ["semantic_match", "au12_lift"],
        "finality": "online_provisional",
        "schema_version": "v3.4",
        "created_at": sample_timestamp,
    }


class TestBanditDecisionSnapshot:
    def test_replay_identity_fields_are_required(self, sample_timestamp: datetime) -> None:
        schema = BanditDecisionSnapshot.model_json_schema()
        required = set(schema["required"])

        assert "sampled_theta_by_arm" in required
        assert "decision_context_hash" in required
        assert "random_seed" in required

        missing_theta = _bandit_snapshot_data(sample_timestamp)
        missing_theta.pop("sampled_theta_by_arm")
        with pytest.raises(ValidationError):
            BanditDecisionSnapshot.model_validate(missing_theta)

        missing_hash = _bandit_snapshot_data(sample_timestamp)
        missing_hash.pop("decision_context_hash")
        with pytest.raises(ValidationError):
            BanditDecisionSnapshot.model_validate(missing_hash)

        missing_seed = _bandit_snapshot_data(sample_timestamp)
        missing_seed.pop("random_seed")
        with pytest.raises(ValidationError):
            BanditDecisionSnapshot.model_validate(missing_seed)


class TestInferenceHandoffPayload:
    """§6.1 — InferenceHandoffPayload JSON Schema Draft 07 contract."""

    def test_valid_payload_exposes_v34_surface(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        payload = InferenceHandoffPayload.model_validate(
            _handoff_payload_data(sample_session_id, sample_timestamp)
        )

        assert str(payload.session_id) == sample_session_id
        assert payload.segment_id == SEGMENT_ID
        assert payload.active_arm == "arm_a"
        assert payload.bandit_decision_snapshot.selected_arm_id == "arm_a"

        dumped = payload.model_dump(by_alias=True, exclude_none=True)
        assert dumped["_active_arm"] == "arm_a"
        assert "active_arm" not in dumped
        assert dumped["_bandit_decision_snapshot"]["selection_method"] == "thompson_sampling"
        assert "_physiological_context" not in dumped

    def test_required_v34_fields_are_in_json_schema(self) -> None:
        schema = InferenceHandoffPayload.model_json_schema(by_alias=True)
        properties = schema["properties"]
        required = set(schema["required"])

        for field in (
            "session_id",
            "segment_id",
            "segment_window_start_utc",
            "segment_window_end_utc",
            "timestamp_utc",
            "_active_arm",
            "_experiment_id",
            "_expected_greeting",
            "_stimulus_time",
            "_au12_series",
            "_bandit_decision_snapshot",
        ):
            assert field in properties
            assert field in required

        assert properties["session_id"]["pattern"].startswith("^[0-9a-f]{8}")
        assert properties["media_source"]["$ref"]
        media_defs = schema.get("$defs", {})
        media_source_schema = media_defs[properties["media_source"]["$ref"].split("/")[-1]]
        assert media_source_schema["properties"]["stream_url"]["format"] == "uri"
        assert "_physiological_context" in properties
        assert "_physiological_context" not in required
        assert "bandit_decision_snapshot" not in properties

    def test_bandit_snapshot_is_required_and_not_free_form_extra(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        missing_snapshot = _handoff_payload_data(sample_session_id, sample_timestamp)
        missing_snapshot.pop("_bandit_decision_snapshot")

        with pytest.raises(ValidationError):
            InferenceHandoffPayload.model_validate(missing_snapshot)

        with_extra = _handoff_payload_data(
            sample_session_id,
            sample_timestamp,
            _unmodeled_snapshot={"selected_arm_id": "arm_a"},
        )
        with pytest.raises(ValidationError):
            InferenceHandoffPayload.model_validate(with_extra)

    def test_session_id_must_be_uuid4(self, sample_timestamp: datetime) -> None:
        non_v4_session_id = "11111111-1111-1111-8111-111111111111"

        with pytest.raises(ValidationError):
            InferenceHandoffPayload.model_validate(
                _handoff_payload_data(non_v4_session_id, sample_timestamp)
            )

    def test_invalid_codec_rejected(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        data = _handoff_payload_data(sample_session_id, sample_timestamp)
        data["media_source"]["codec"] = "vp9"

        with pytest.raises(ValidationError):
            InferenceHandoffPayload.model_validate(data)

    def test_stream_url_must_be_uri(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        data = _handoff_payload_data(sample_session_id, sample_timestamp)
        data["media_source"]["stream_url"] = "not a uri"

        with pytest.raises(ValidationError):
            InferenceHandoffPayload.model_validate(data)

    def test_resolution_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            MediaSource.model_validate(
                {
                    "stream_url": "https://example.com/stream",
                    "codec": "h264",
                    "resolution": [0, 1080],
                }
            )

    def test_segment_id_must_be_sha256_hex(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        with pytest.raises(ValidationError):
            InferenceHandoffPayload.model_validate(
                _handoff_payload_data(sample_session_id, sample_timestamp, segment_id="abc")
            )

    def test_optional_physiological_context_accepts_real_partial_snapshot(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        payload = InferenceHandoffPayload.model_validate(
            _handoff_payload_data(
                sample_session_id,
                sample_timestamp,
                _physiological_context={
                    "streamer": _physiological_snapshot_data(sample_timestamp),
                    "operator": None,
                },
            )
        )

        assert payload.physiological_context is not None
        assert payload.physiological_context.streamer is not None
        assert payload.physiological_context.operator is None

    def test_physiological_context_rejects_empty_and_null_placeholders(
        self, sample_session_id: str, sample_timestamp: datetime
    ) -> None:
        for context in ({}, {"streamer": None, "operator": None}, None):
            with pytest.raises(ValidationError):
                InferenceHandoffPayload.model_validate(
                    _handoff_payload_data(
                        sample_session_id,
                        sample_timestamp,
                        _physiological_context=context,
                    )
                )

    def test_module_c_to_d_contract_excludes_module_d_acoustic_outputs(self) -> None:
        """Module C → D stays on the stable InferenceHandoffPayload contract."""
        schema = InferenceHandoffPayload.model_json_schema(by_alias=True)
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
        ):
            assert field not in properties


class TestSemanticEvaluationResult:
    """§8.3 — bounded canonical semantic scorer payload enforcement."""

    def test_valid_result_is_canonical_three_field_payload(self) -> None:
        result = SemanticEvaluationResult(
            reasoning="cross_encoder_high_match",
            is_match=True,
            confidence_score=0.95,
        )

        assert result.reasoning in SEMANTIC_REASON_CODES
        assert result.is_match is True
        assert 0.0 <= result.confidence_score <= 1.0
        assert not hasattr(result, "semantic_method")
        assert not hasattr(result, "semantic_method_version")

    def test_free_form_reasoning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationResult(
                reasoning=cast(Any, "Semantically equivalent greeting."),
                is_match=True,
                confidence_score=0.95,
            )

    def test_extra_noncanonical_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationResult(  # type: ignore[call-arg]
                reasoning="cross_encoder_high_match",
                is_match=True,
                confidence_score=0.5,
                explanatory_text="should fail",
            )

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationResult(
                reasoning="semantic_error",
                is_match=False,
                confidence_score=1.5,
            )

    def test_semantic_method_metadata_rejected_from_canonical_payload(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluationResult(  # type: ignore[call-arg]
                reasoning="semantic_error",
                is_match=False,
                confidence_score=0.0,
                semantic_method=cast(Any, "cross_encoder"),
                semantic_method_version="v1",
            )


class TestAttributionSchemas:
    """§6.4 — Attribution ledger typed Pydantic contracts."""

    def test_attribution_event_validates_required_contract(
        self, sample_timestamp: datetime
    ) -> None:
        event = AttributionEvent.model_validate(_attribution_event_data(sample_timestamp))

        assert event.event_type == "greeting_interaction"
        assert event.finality == "online_provisional"
        assert event.schema_version == "v3.4"
        assert event.bandit_decision_snapshot.selection_method == "thompson_sampling"

    def test_outcome_link_and_score_models_validate(self, sample_timestamp: datetime) -> None:
        event_id = uuid.uuid4()
        outcome_id = uuid.uuid4()
        common = {
            "finality": "offline_final",
            "schema_version": "v3.4",
            "created_at": sample_timestamp,
        }

        outcome = OutcomeEvent.model_validate(
            {
                "outcome_id": outcome_id,
                "session_id": uuid.uuid4(),
                "outcome_type": "creator_follow",
                "outcome_value": 1.0,
                "outcome_time_utc": sample_timestamp,
                "source_system": "tiktok",
                "source_event_ref": "follow-123",
                "confidence": 0.88,
                **common,
            }
        )
        link = EventOutcomeLink.model_validate(
            {
                "link_id": uuid.uuid4(),
                "event_id": event_id,
                "outcome_id": outcome.outcome_id,
                "lag_s": 3.2,
                "horizon_s": 30.0,
                "link_rule_version": "link-v1",
                "eligibility_flags": ["within_horizon"],
                **common,
            }
        )
        score = AttributionScore.model_validate(
            {
                "score_id": uuid.uuid4(),
                "event_id": event_id,
                "outcome_id": outcome_id,
                "attribution_method": "lagged_correlation",
                "method_version": "attr-v1",
                "score_raw": None,
                "score_normalized": 0.5,
                "confidence": 0.75,
                "evidence_flags": ["within_horizon"],
                **common,
            }
        )

        assert link.lag_s == pytest.approx(3.2)
        assert score.score_raw is None
        assert score.finality == "offline_final"

    def test_finality_and_unique_arrays_are_enforced(self, sample_timestamp: datetime) -> None:
        with pytest.raises(ValidationError):
            AttributionEvent.model_validate(
                _attribution_event_data(sample_timestamp) | {"finality": "draft"}
            )

        with pytest.raises(ValidationError):
            EventOutcomeLink.model_validate(
                {
                    "link_id": uuid.uuid4(),
                    "event_id": uuid.uuid4(),
                    "outcome_id": uuid.uuid4(),
                    "lag_s": 0.0,
                    "horizon_s": 30.0,
                    "link_rule_version": "link-v1",
                    "eligibility_flags": ["within_horizon", "within_horizon"],
                    "finality": "online_provisional",
                    "schema_version": "v3.4",
                    "created_at": sample_timestamp,
                }
            )

    def test_attribution_schema_documents_id_finality_version_and_created_at(self) -> None:
        schema = AttributionEvent.model_json_schema()
        properties = schema["properties"]

        assert "deterministic UUIDv5" in properties["event_id"]["description"]
        assert "lifecycle state" in properties["finality"]["description"]
        assert "schema contract version" in properties["schema_version"]["description"]
        assert "does not populate" in properties["created_at"]["description"]


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
        payload = InferenceHandoffPayload.model_validate(
            _handoff_payload_data(
                sample_session_id,
                sample_timestamp,
                media_source={
                    "stream_url": "https://example.com/stream",
                    "codec": "raw",
                    "resolution": [640, 480],
                },
                segments=[{"frame": 1}],
            )
        )
        json_str = payload.model_dump_json(exclude={"physiological_context"})
        restored = InferenceHandoffPayload.model_validate_json(json_str)
        assert restored.session_id == payload.session_id
        assert restored.segments == payload.segments
        assert restored.bandit_decision_snapshot.selected_arm_id == "arm_a"

    def test_json_schema_has_draft07(self) -> None:
        schema = InferenceHandoffPayload.model_json_schema()
        assert schema.get("$schema") == "http://json-schema.org/draft-07/schema#"
