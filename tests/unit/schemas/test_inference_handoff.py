from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from packages.schemas.inference_handoff import InferenceHandoffPayload, ResponseInference

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"


def _handoff_payload(*, response_inference: dict[str, Any] | None = None) -> dict[str, Any]:
    sample_timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    stimulus_payload = {"content_type": "text", "text": "Say hello to the creator"}
    expected_stimulus_rule = "Deliver the spoken greeting to the live streamer."
    expected_response_rule = "The live streamer acknowledges the greeting."
    payload = {
        "session_id": SESSION_ID,
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
        "_active_arm": "warm_welcome",
        "_experiment_id": 1,
        "_stimulus_modality": "spoken_greeting",
        "_stimulus_payload": stimulus_payload,
        "_expected_stimulus_rule": expected_stimulus_rule,
        "_expected_response_rule": expected_response_rule,
        "_stimulus_time": 100.0,
        "_response_observation_horizon_s": 8.0,
        "_au12_series": [{"timestamp_s": 100.1, "intensity": 0.5}],
        "_bandit_decision_snapshot": {
            "selection_method": "thompson_sampling",
            "selection_time_utc": sample_timestamp,
            "experiment_id": 1,
            "policy_version": "ts-v1",
            "selected_arm_id": "warm_welcome",
            "candidate_arm_ids": ["warm_welcome", "direct_question"],
            "posterior_by_arm": {
                "warm_welcome": {"alpha": 1.0, "beta": 1.0},
                "direct_question": {"alpha": 1.0, "beta": 1.0},
            },
            "sampled_theta_by_arm": {"warm_welcome": 0.72, "direct_question": 0.44},
            "stimulus_modality": "spoken_greeting",
            "stimulus_payload": stimulus_payload,
            "expected_stimulus_rule": expected_stimulus_rule,
            "expected_response_rule": expected_response_rule,
            "decision_context_hash": DECISION_CONTEXT_HASH,
            "random_seed": 42,
        },
        "_stimulus_id": str(uuid.UUID("00000000-0000-4000-8000-000000000002")),
    }
    if response_inference is not None:
        payload["response_inference"] = response_inference
    return payload


def test_response_inference_accepts_bounded_metadata() -> None:
    response = ResponseInference.model_validate(
        {
            "is_match": True,
            "confidence_score": 0.91,
            "registration_status": "observable_response",
            "response_reason_code": "response_semantic_ack",
            "response_type": "semantic_ack",
            "matched_response_time": 100.75,
            "evidence_span_ref": "segment:response-window",
        }
    )

    assert response.registration_status == "observable_response"
    assert response.response_reason_code == "response_semantic_ack"
    assert response.matched_response_time == pytest.approx(100.75)


def test_response_inference_rejects_unbounded_or_invalid_fields() -> None:
    valid = {
        "is_match": True,
        "confidence_score": 0.91,
        "registration_status": "observable_response",
        "response_reason_code": "response_semantic_ack",
    }

    with pytest.raises(ValidationError):
        ResponseInference.model_validate({**valid, "response_reason_code": "the creator smiled"})
    with pytest.raises(ValidationError):
        ResponseInference.model_validate({**valid, "registration_status": "definitely_replied"})
    with pytest.raises(ValidationError):
        ResponseInference.model_validate({**valid, "rationale": "free-form response note"})
    with pytest.raises(ValidationError):
        ResponseInference.model_validate({**valid, "matched_response_time": -1.0})
    with pytest.raises(ValidationError):
        ResponseInference.model_validate({**valid, "evidence_span_ref": ""})


def test_handoff_accepts_response_inference_and_horizon_alias() -> None:
    response_inference = {
        "is_match": True,
        "confidence_score": 0.91,
        "registration_status": "observable_response",
        "response_reason_code": "response_semantic_ack",
        "matched_response_time": 100.75,
        "evidence_span_ref": "segment:response-window",
    }

    handoff = InferenceHandoffPayload.model_validate(
        _handoff_payload(response_inference=response_inference)
    )

    assert handoff.response_observation_horizon_s == pytest.approx(8.0)
    assert handoff.response_inference is not None
    assert handoff.response_inference.registration_status == "observable_response"
    dumped = handoff.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert dumped["_response_observation_horizon_s"] == pytest.approx(8.0)
    assert dumped["response_inference"] == response_inference
    assert "rationale" not in str(dumped)


def test_handoff_rejects_invalid_response_horizon() -> None:
    with pytest.raises(ValidationError):
        InferenceHandoffPayload.model_validate(
            _handoff_payload(
                response_inference={
                    "is_match": False,
                    "confidence_score": 0.0,
                    "registration_status": "no_observable_response",
                    "response_reason_code": "no_observable_response",
                },
            )
            | {"_response_observation_horizon_s": -1.0}
        )
