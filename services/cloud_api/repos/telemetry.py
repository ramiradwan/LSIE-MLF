"""Repository writes for cloud telemetry ingestion."""

from __future__ import annotations

from typing import Any

from psycopg2.extras import Json

from packages.schemas.attribution import AttributionEvent
from packages.schemas.cloud import PosteriorDelta, TelemetrySegmentBatch
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.cloud_api.repos.common import model_to_json_dict

_INSERT_SEGMENT_SQL = """
    INSERT INTO segment_telemetry (segment_id, session_id, payload, client_id)
    VALUES (%(segment_id)s, %(session_id)s, %(payload)s, %(client_id)s)
    ON CONFLICT (segment_id) DO NOTHING
"""

_INSERT_ATTRIBUTION_EVENT_SQL = """
    INSERT INTO attribution_event (
        event_id,
        session_id,
        segment_id,
        event_type,
        event_time_utc,
        stimulus_time_utc,
        selected_arm_id,
        expected_rule_text_hash,
        semantic_method,
        semantic_method_version,
        semantic_p_match,
        semantic_reason_code,
        reward_path_version,
        bandit_decision_snapshot,
        evidence_flags,
        finality,
        schema_version,
        created_at
    )
    VALUES (
        %(event_id)s,
        %(session_id)s,
        %(segment_id)s,
        %(event_type)s,
        %(event_time_utc)s,
        %(stimulus_time_utc)s,
        %(selected_arm_id)s,
        %(expected_rule_text_hash)s,
        %(semantic_method)s,
        %(semantic_method_version)s,
        %(semantic_p_match)s,
        %(semantic_reason_code)s,
        %(reward_path_version)s,
        %(bandit_decision_snapshot)s,
        %(evidence_flags)s,
        %(finality)s,
        %(schema_version)s,
        %(created_at)s
    )
    ON CONFLICT (event_id) DO NOTHING
"""

_INSERT_POSTERIOR_DELTA_SQL = """
    INSERT INTO posterior_delta_log (
        event_id,
        experiment_id,
        arm_id,
        delta_alpha,
        delta_beta,
        segment_id,
        client_id,
        applied_at_utc,
        decision_context_hash
    )
    VALUES (
        %(event_id)s,
        %(experiment_id)s,
        %(arm_id)s,
        %(delta_alpha)s,
        %(delta_beta)s,
        %(segment_id)s,
        %(client_id)s,
        %(applied_at_utc)s,
        %(decision_context_hash)s
    )
    ON CONFLICT (segment_id, client_id, arm_id) DO NOTHING
"""


def insert_segment_batch(cur: Any, batch: TelemetrySegmentBatch) -> int:
    inserted = 0
    for segment in batch.segments:
        inserted += insert_segment(cur, segment)
    for event in batch.attribution_events:
        inserted += insert_attribution_event(cur, event)
    return inserted


def insert_segment(cur: Any, segment: InferenceHandoffPayload) -> int:
    payload = model_to_json_dict(segment)
    cur.execute(
        _INSERT_SEGMENT_SQL,
        {
            "segment_id": segment.segment_id,
            "session_id": str(segment.session_id),
            "payload": Json(payload),
            "client_id": None,
        },
    )
    return int(cur.rowcount)


def insert_attribution_event(cur: Any, event: AttributionEvent) -> int:
    cur.execute(
        _INSERT_ATTRIBUTION_EVENT_SQL,
        {
            "event_id": str(event.event_id),
            "session_id": str(event.session_id),
            "segment_id": event.segment_id,
            "event_type": event.event_type,
            "event_time_utc": event.event_time_utc,
            "stimulus_time_utc": event.stimulus_time_utc,
            "selected_arm_id": event.selected_arm_id,
            "expected_rule_text_hash": event.expected_rule_text_hash,
            "semantic_method": event.semantic_method,
            "semantic_method_version": event.semantic_method_version,
            "semantic_p_match": event.semantic_p_match,
            "semantic_reason_code": event.semantic_reason_code,
            "reward_path_version": event.reward_path_version,
            "bandit_decision_snapshot": Json(model_to_json_dict(event.bandit_decision_snapshot)),
            "evidence_flags": event.evidence_flags,
            "finality": event.finality,
            "schema_version": event.schema_version,
            "created_at": event.created_at,
        },
    )
    return int(cur.rowcount)


def insert_posterior_delta_batch(cur: Any, deltas: list[PosteriorDelta]) -> int:
    inserted = 0
    for delta in deltas:
        inserted += insert_posterior_delta(cur, delta)
    return inserted


def insert_posterior_delta(cur: Any, delta: PosteriorDelta) -> int:
    cur.execute(
        _INSERT_POSTERIOR_DELTA_SQL,
        {
            "event_id": str(delta.event_id),
            "experiment_id": delta.experiment_id,
            "arm_id": delta.arm_id,
            "delta_alpha": delta.delta_alpha,
            "delta_beta": delta.delta_beta,
            "segment_id": delta.segment_id,
            "client_id": delta.client_id,
            "applied_at_utc": delta.applied_at_utc,
            "decision_context_hash": delta.decision_context_hash,
        },
    )
    return int(cur.rowcount)
