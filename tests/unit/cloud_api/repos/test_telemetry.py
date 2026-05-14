from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from psycopg2.extras import Json

from packages.schemas.attribution import ArmPosterior, AttributionEvent, BanditDecisionSnapshot
from packages.schemas.evaluation import StimulusPayload
from services.cloud_api.db import schema
from services.cloud_api.repos import telemetry

SEGMENT_ID = "a" * 64
RULE_HASH = "b" * 64
RESPONSE_RULE_HASH = "c" * 64
DECISION_CONTEXT_HASH = "d" * 64


class AttributionCursor:
    def __init__(self) -> None:
        self.rowcount = 1
        self.sql: str | None = None
        self.params: Mapping[str, object] | None = None

    def execute(self, sql: str, params: Mapping[str, object]) -> None:
        self.sql = sql
        self.params = params


def _decision_snapshot(timestamp: datetime) -> BanditDecisionSnapshot:
    return BanditDecisionSnapshot(
        selection_method="thompson_sampling",
        selection_time_utc=timestamp,
        experiment_id=101,
        policy_version="policy-v1",
        selected_arm_id="arm-a",
        candidate_arm_ids=["arm-a", "arm-b"],
        posterior_by_arm={
            "arm-a": ArmPosterior(alpha=2.0, beta=1.0),
            "arm-b": ArmPosterior(alpha=1.0, beta=1.0),
        },
        sampled_theta_by_arm={"arm-a": 0.7, "arm-b": 0.3},
        decision_context_hash=DECISION_CONTEXT_HASH,
        random_seed=42,
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(text="Say hello"),
        expected_stimulus_rule="Deliver the stimulus",
        expected_response_rule="Observe the bounded response",
    )


def _attribution_event() -> AttributionEvent:
    timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    return AttributionEvent(
        event_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
        session_id=uuid.UUID("00000000-0000-4000-8000-000000000002"),
        segment_id=SEGMENT_ID,
        event_type="stimulus_interaction",
        event_time_utc=timestamp,
        stimulus_time_utc=timestamp,
        selected_arm_id="arm-a",
        expected_rule_text_hash=RULE_HASH,
        semantic_method="cross_encoder",
        semantic_method_version="ce-v1",
        semantic_p_match=0.93,
        semantic_reason_code="cross_encoder_high_match",
        reward_path_version="reward-v1",
        bandit_decision_snapshot=_decision_snapshot(timestamp),
        evidence_flags=["response_window_complete"],
        finality="online_provisional",
        schema_version="attribution-v1",
        created_at=timestamp,
        stimulus_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
        stimulus_modality="spoken_greeting",
        matched_response_time_utc=timestamp,
        response_registration_status="observable_response",
        response_reason_code="response_semantic_ack",
        expected_response_rule_text_hash=RESPONSE_RULE_HASH,
    )


def test_attribution_event_type_constraint_is_migration_safe() -> None:
    assert "DROP CONSTRAINT IF EXISTS attribution_event_event_type_check" in schema.SCHEMA_SQL
    assert "DROP CONSTRAINT IF EXISTS ck_attribution_event_type" in schema.SCHEMA_SQL
    assert "ADD CONSTRAINT ck_attribution_event_type" in schema.SCHEMA_SQL
    assert "CHECK (event_type IN ('stimulus_interaction')) NOT VALID" in schema.SCHEMA_SQL
    assert "greeting_interaction" not in schema.SCHEMA_SQL


def test_attribution_insert_sql_covers_canonical_event_fields() -> None:
    required_fields = {
        "event_id",
        "session_id",
        "segment_id",
        "event_type",
        "event_time_utc",
        "stimulus_time_utc",
        "stimulus_id",
        "stimulus_modality",
        "selected_arm_id",
        "expected_rule_text_hash",
        "expected_response_rule_text_hash",
        "semantic_method",
        "semantic_method_version",
        "semantic_p_match",
        "semantic_reason_code",
        "matched_response_time_utc",
        "response_registration_status",
        "response_reason_code",
        "reward_path_version",
        "bandit_decision_snapshot",
        "evidence_flags",
        "finality",
        "schema_version",
        "created_at",
    }

    for field in required_fields:
        assert field in telemetry._INSERT_ATTRIBUTION_EVENT_SQL
    assert "greeting_interaction" not in telemetry._INSERT_ATTRIBUTION_EVENT_SQL
    assert "greeting_text" not in telemetry._INSERT_ATTRIBUTION_EVENT_SQL


def test_insert_attribution_event_preserves_canonical_metadata() -> None:
    cur = AttributionCursor()
    event = _attribution_event()

    inserted = telemetry.insert_attribution_event(cur, event)

    assert inserted == 1
    assert cur.sql == telemetry._INSERT_ATTRIBUTION_EVENT_SQL
    assert cur.params is not None
    assert cur.params["stimulus_id"] == str(event.stimulus_id)
    assert cur.params["stimulus_modality"] == "spoken_greeting"
    assert cur.params["expected_response_rule_text_hash"] == RESPONSE_RULE_HASH
    assert cur.params["matched_response_time_utc"] == event.matched_response_time_utc
    assert cur.params["response_registration_status"] == "observable_response"
    assert cur.params["response_reason_code"] == "response_semantic_ack"
    assert isinstance(cur.params["bandit_decision_snapshot"], Json)
    assert "greeting_text" not in cur.params
    assert "greeting_interaction" not in cur.params.values()


def test_insert_attribution_event_omits_no_canonical_params() -> None:
    cur = AttributionCursor()

    telemetry.insert_attribution_event(cur, _attribution_event())

    assert cur.params is not None
    insert_columns = {
        column.strip()
        for column in telemetry._INSERT_ATTRIBUTION_EVENT_SQL.split("(", maxsplit=1)[1]
        .split(")", maxsplit=1)[0]
        .split(",")
    }
    assert insert_columns.issubset(set(cur.params))
