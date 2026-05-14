from __future__ import annotations

from typing import Any

import pytest

from packages.schemas.cloud import TelemetrySegmentBatch
from services.cloud_api.services import telemetry_service
from services.cloud_api.services.telemetry_service import TelemetryIngestService


class TransactionConnection:
    def __init__(self) -> None:
        self.cursor_instance = TransactionCursor()

    def cursor(self) -> TransactionCursor:
        return self.cursor_instance


class TransactionCursor:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> TransactionCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.closed = True


def _event_only_batch() -> TelemetrySegmentBatch:
    return TelemetrySegmentBatch.model_validate(
        {
            "segments": [],
            "attribution_events": [
                {
                    "event_id": "00000000-0000-4000-8000-000000000001",
                    "session_id": "00000000-0000-4000-8000-000000000002",
                    "segment_id": "a" * 64,
                    "event_type": "stimulus_interaction",
                    "event_time_utc": "2026-05-02T12:00:00Z",
                    "stimulus_time_utc": "2026-05-02T12:00:00Z",
                    "selected_arm_id": "arm-a",
                    "expected_rule_text_hash": "b" * 64,
                    "semantic_method": "cross_encoder",
                    "semantic_method_version": "ce-v1",
                    "semantic_p_match": 0.91,
                    "semantic_reason_code": "cross_encoder_high_match",
                    "reward_path_version": "reward-v1",
                    "bandit_decision_snapshot": {
                        "selection_method": "thompson_sampling",
                        "selection_time_utc": "2026-05-02T12:00:00Z",
                        "experiment_id": 101,
                        "policy_version": "policy-v1",
                        "selected_arm_id": "arm-a",
                        "candidate_arm_ids": ["arm-a", "arm-b"],
                        "posterior_by_arm": {
                            "arm-a": {"alpha": 2.0, "beta": 1.0},
                            "arm-b": {"alpha": 1.0, "beta": 1.0},
                        },
                        "sampled_theta_by_arm": {"arm-a": 0.7, "arm-b": 0.3},
                        "decision_context_hash": "c" * 64,
                        "random_seed": 42,
                        "stimulus_modality": "spoken_greeting",
                        "stimulus_payload": {"content_type": "text", "text": "Say hello"},
                        "expected_stimulus_rule": "Deliver the stimulus",
                        "expected_response_rule": "Observe the bounded response",
                    },
                    "evidence_flags": [],
                    "finality": "online_provisional",
                    "schema_version": "attribution-v1",
                    "created_at": "2026-05-02T12:00:00Z",
                    "stimulus_id": "00000000-0000-4000-8000-000000000003",
                    "stimulus_modality": "spoken_greeting",
                    "matched_response_time_utc": "2026-05-02T12:00:00Z",
                    "response_registration_status": "observable_response",
                    "response_reason_code": "response_semantic_ack",
                    "expected_response_rule_text_hash": "d" * 64,
                }
            ],
        }
    )


def test_ingest_segments_counts_event_only_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[TelemetrySegmentBatch, str]] = []

    def fake_run_in_transaction(operation: Any) -> int:
        return int(operation(TransactionConnection()))

    def fake_insert_segment_batch(cur: Any, batch: TelemetrySegmentBatch, *, client_id: str) -> int:
        assert isinstance(cur, TransactionCursor)
        calls.append((batch, client_id))
        return len(batch.attribution_events)

    monkeypatch.setattr(telemetry_service, "run_in_transaction", fake_run_in_transaction)
    monkeypatch.setattr(telemetry_service, "insert_segment_batch", fake_insert_segment_batch)

    response = TelemetryIngestService().ingest_segments(_event_only_batch(), client_id="desktop-a")

    assert response.accepted_count == 1
    assert response.inserted_count == 1
    assert len(calls) == 1
    assert calls[0][1] == "desktop-a"


def test_ingest_segments_propagates_validation_write_failure_transactionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rolled_back = False

    def fake_run_in_transaction(operation: Any) -> int:
        nonlocal rolled_back
        try:
            return int(operation(TransactionConnection()))
        except ValueError:
            rolled_back = True
            raise

    def fake_insert_segment_batch(cur: Any, batch: TelemetrySegmentBatch, *, client_id: str) -> int:
        del cur, batch, client_id
        raise ValueError("attribution event failed validation")

    monkeypatch.setattr(telemetry_service, "run_in_transaction", fake_run_in_transaction)
    monkeypatch.setattr(telemetry_service, "insert_segment_batch", fake_insert_segment_batch)

    with pytest.raises(ValueError, match="attribution event failed validation"):
        TelemetryIngestService().ingest_segments(_event_only_batch(), client_id="desktop-a")

    assert rolled_back is True
