"""Repository writes for cloud telemetry ingestion."""

from __future__ import annotations

from typing import Any

from psycopg2.extras import Json

from packages.schemas.attribution import AttributionEvent
from packages.schemas.cloud import PosteriorDelta, TelemetrySegmentBatch
from packages.schemas.data_tiers import DataTier, mark_data_tier
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.cloud_api.repos.common import model_to_json_dict


class PosteriorDeltaApplyError(RuntimeError):
    pass


_INSERT_SEGMENT_SQL = mark_data_tier(
    """
    INSERT INTO segment_telemetry (segment_id, session_id, payload, client_id)
    VALUES (%(segment_id)s, %(session_id)s, %(payload)s, %(client_id)s)
    ON CONFLICT (segment_id) DO NOTHING
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)

_INSERT_ATTRIBUTION_EVENT_SQL = mark_data_tier(
    """
    INSERT INTO attribution_event (
        event_id,
        session_id,
        segment_id,
        event_type,
        event_time_utc,
        stimulus_time_utc,
        stimulus_id,
        stimulus_modality,
        selected_arm_id,
        expected_rule_text_hash,
        expected_response_rule_text_hash,
        semantic_method,
        semantic_method_version,
        semantic_p_match,
        semantic_reason_code,
        matched_response_time_utc,
        response_registration_status,
        response_reason_code,
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
        %(stimulus_id)s,
        %(stimulus_modality)s,
        %(selected_arm_id)s,
        %(expected_rule_text_hash)s,
        %(expected_response_rule_text_hash)s,
        %(semantic_method)s,
        %(semantic_method_version)s,
        %(semantic_p_match)s,
        %(semantic_reason_code)s,
        %(matched_response_time_utc)s,
        %(response_registration_status)s,
        %(response_reason_code)s,
        %(reward_path_version)s,
        %(bandit_decision_snapshot)s,
        %(evidence_flags)s,
        %(finality)s,
        %(schema_version)s,
        %(created_at)s
    )
    ON CONFLICT (event_id) DO NOTHING
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)

_INSERT_POSTERIOR_DELTA_SQL = mark_data_tier(
    """
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
    ON CONFLICT DO NOTHING
""",
    DataTier.PERMANENT,
    spec_ref="§5.2.3",
)

_SELECT_SEGMENT_AUTH_SQL = """
    SELECT payload, client_id
    FROM segment_telemetry
    WHERE segment_id = %(segment_id)s
"""

_APPLY_POSTERIOR_DELTA_SQL = """
    UPDATE experiments
    SET
        alpha_param = alpha_param + %(delta_alpha)s,
        beta_param = beta_param + %(delta_beta)s,
        updated_at = NOW()
    WHERE id = %(experiment_id)s
      AND arm = %(arm_id)s
"""


def insert_segment_batch(cur: Any, batch: TelemetrySegmentBatch, *, client_id: str) -> int:
    inserted = 0
    for segment in batch.segments:
        inserted += insert_segment(cur, segment, client_id=client_id)
    for event in batch.attribution_events:
        inserted += insert_attribution_event(cur, event)
    return inserted


def insert_segment(cur: Any, segment: InferenceHandoffPayload, *, client_id: str) -> int:
    payload = model_to_json_dict(segment)
    cur.execute(
        _INSERT_SEGMENT_SQL,
        mark_data_tier(
            {
                "segment_id": segment.segment_id,
                "session_id": str(segment.session_id),
                "payload": Json(payload),
                "client_id": client_id,
            },
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
        ),
    )
    return int(cur.rowcount)


def insert_attribution_event(cur: Any, event: AttributionEvent) -> int:
    cur.execute(
        _INSERT_ATTRIBUTION_EVENT_SQL,
        mark_data_tier(
            {
                "event_id": str(event.event_id),
                "session_id": str(event.session_id),
                "segment_id": event.segment_id,
                "event_type": event.event_type,
                "event_time_utc": event.event_time_utc,
                "stimulus_time_utc": event.stimulus_time_utc,
                "stimulus_id": str(event.stimulus_id) if event.stimulus_id is not None else None,
                "stimulus_modality": event.stimulus_modality,
                "selected_arm_id": event.selected_arm_id,
                "expected_rule_text_hash": event.expected_rule_text_hash,
                "expected_response_rule_text_hash": event.expected_response_rule_text_hash,
                "semantic_method": event.semantic_method,
                "semantic_method_version": event.semantic_method_version,
                "semantic_p_match": event.semantic_p_match,
                "semantic_reason_code": event.semantic_reason_code,
                "matched_response_time_utc": event.matched_response_time_utc,
                "response_registration_status": event.response_registration_status,
                "response_reason_code": event.response_reason_code,
                "reward_path_version": event.reward_path_version,
                "bandit_decision_snapshot": Json(
                    model_to_json_dict(event.bandit_decision_snapshot)
                ),
                "evidence_flags": event.evidence_flags,
                "finality": event.finality,
                "schema_version": event.schema_version,
                "created_at": event.created_at,
            },
            DataTier.PERMANENT,
            spec_ref="§5.2.3",
        ),
    )
    return int(cur.rowcount)


def insert_posterior_delta_batch(
    cur: Any,
    deltas: list[PosteriorDelta],
    *,
    authenticated_client_id: str,
) -> int:
    inserted = 0
    for delta in deltas:
        inserted += insert_posterior_delta(
            cur,
            delta,
            authenticated_client_id=authenticated_client_id,
        )
    return inserted


def insert_posterior_delta(
    cur: Any,
    delta: PosteriorDelta,
    *,
    authenticated_client_id: str,
) -> int:
    params = mark_data_tier(
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
        DataTier.PERMANENT,
        spec_ref="§5.2.3",
    )
    _assert_delta_matches_segment(cur, delta, authenticated_client_id=authenticated_client_id)
    cur.execute(_INSERT_POSTERIOR_DELTA_SQL, params)
    inserted = int(cur.rowcount)
    if inserted == 0:
        return 0
    cur.execute(_APPLY_POSTERIOR_DELTA_SQL, params)
    if int(cur.rowcount) != 1:
        raise PosteriorDeltaApplyError(
            f"Posterior delta target arm not found: experiment_id={delta.experiment_id}, "
            f"arm_id={delta.arm_id!r}"
        )
    return inserted


def _assert_delta_matches_segment(
    cur: Any,
    delta: PosteriorDelta,
    *,
    authenticated_client_id: str,
) -> None:
    cur.execute(_SELECT_SEGMENT_AUTH_SQL, {"segment_id": delta.segment_id})
    row = cur.fetchone()
    if row is None:
        raise PosteriorDeltaApplyError(
            f"Posterior delta segment not found: segment_id={delta.segment_id}"
        )
    payload = row[0]
    stored_client_id = row[1]
    if stored_client_id != authenticated_client_id:
        raise PosteriorDeltaApplyError(
            "Posterior delta segment does not belong to authenticated client"
        )
    if not isinstance(payload, dict):
        raise PosteriorDeltaApplyError("Posterior delta segment payload is malformed")
    snapshot = payload.get("_bandit_decision_snapshot")
    if not isinstance(snapshot, dict):
        raise PosteriorDeltaApplyError(
            "Posterior delta segment is missing bandit decision snapshot"
        )
    stored_experiment_id = snapshot.get("experiment_id")
    stored_arm_id = snapshot.get("selected_arm_id")
    stored_decision_hash = snapshot.get("decision_context_hash")
    if stored_experiment_id != delta.experiment_id:
        raise PosteriorDeltaApplyError(
            "Posterior delta experiment_id does not match segment decision snapshot"
        )
    if stored_arm_id != delta.arm_id:
        raise PosteriorDeltaApplyError(
            "Posterior delta arm_id does not match segment decision snapshot"
        )
    if delta.decision_context_hash != stored_decision_hash:
        raise PosteriorDeltaApplyError(
            "Posterior delta decision_context_hash does not match segment decision snapshot"
        )
