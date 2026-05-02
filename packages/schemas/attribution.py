"""Strict Pydantic schemas for deterministic attribution ledger records (§6.4).

The module models BanditDecisionSnapshot records, AttributionEvent records,
OutcomeEvent records, EventOutcomeLink records, and AttributionScore records
using canonical terminology (§0, §13.15). Producers must
supply replay-stable identifiers and audit timestamps; these schemas validate
shape and constraints but do not generate IDs, query storage, or update rewards.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from packages.schemas.evaluation import SemanticMethod, SemanticReasonCode

AttributionFinality = Literal["online_provisional", "offline_final"]
AttributionEventType = Literal["greeting_interaction"]
OutcomeEventType = Literal["creator_follow"]
BanditSelectionMethod = Literal["thompson_sampling"]

_SCHEMA_VERSION_DESCRIPTION = (
    "Required schema contract version for the serialized attribution record; "
    "the producer supplies this value and the schema does not generate it."
)
_FINALITY_DESCRIPTION = (
    "Attribution record lifecycle state: online_provisional for online/streaming "
    "records or offline_final for finalized replay/backfill records."
)
_CREATED_AT_DESCRIPTION = (
    "UTC timestamp when the producing system created the attribution record; "
    "the schema validates this field but does not populate it automatically."
)


def _ensure_unique(values: list[str], field_name: str) -> list[str]:
    """Validate JSON Schema uniqueItems semantics for string arrays."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} entries must be unique")
    return values


class AttributionBaseModel(BaseModel):
    """
    Shared strict base for attribution ledger schemas.

    Accepts Pydantic field definitions from subclasses and produces models that
    forbid unexpected keys. It does not add identity generation, timestamps, or
    persistence behavior.
    """

    model_config = ConfigDict(extra="forbid")


class ArmPosterior(AttributionBaseModel):
    """
    Validate posterior parameters for one Thompson Sampling arm.

    Accepts positive alpha and beta values and produces a strict nested snapshot
    object. It does not sample arms or update posteriors.
    """

    alpha: float = Field(..., gt=0.0)
    beta: float = Field(..., gt=0.0)


class BanditDecisionSnapshot(AttributionBaseModel):
    """
    Validate the pre-update Thompson Sampling BanditDecisionSnapshot (§6.4 / §6.1).

    Accepts selection metadata, selected/candidate arm IDs, posterior parameters,
    optional sampled theta values, and the expected greeting captured at arm
    selection time. Produces a replayable decision record for handoff and
    attribution; it does not perform selection, update posteriors, or compute
    rewards.
    """

    selection_method: BanditSelectionMethod
    selection_time_utc: datetime
    experiment_id: int
    policy_version: str
    selected_arm_id: str
    candidate_arm_ids: list[str] = Field(..., min_length=1)
    posterior_by_arm: dict[str, ArmPosterior] = Field(..., min_length=1)
    sampled_theta_by_arm: dict[str, float] | None = None
    expected_greeting: str
    decision_context_hash: str = Field(..., pattern="^[0-9a-f]{64}$")
    random_seed: int = Field(..., ge=0, le=18446744073709551615)

    @field_validator("candidate_arm_ids")
    @classmethod
    def _candidate_arm_ids_unique(cls, values: list[str]) -> list[str]:
        return _ensure_unique(values, "candidate_arm_ids")

    @field_validator("sampled_theta_by_arm")
    @classmethod
    def _theta_values_are_probabilities(
        cls, values: dict[str, float] | None
    ) -> dict[str, float] | None:
        if values is None:
            return values
        if any(theta < 0.0 or theta > 1.0 for theta in values.values()):
            raise ValueError("sampled theta values must be between 0.0 and 1.0")
        return values


class AttributionEvent(AttributionBaseModel):
    """
    Validate a greeting-interaction AttributionEvent record (§6.4.1).

    Accepts caller-provided event identity, segment/session context, semantic
    summary, reward-path version, evidence flags, finality, schema version, and
    creation time. Produces a strict ledger event model; it does not hash rule
    text, build UUIDs, or mutate the reward path.
    """

    event_id: UUID = Field(
        ...,
        description=(
            "Caller-provided deterministic UUIDv5 derived from session_id, "
            "segment_id, event_type, and reward_path_version; replay-stable "
            "for attribution backfill idempotency."
        ),
    )
    session_id: UUID
    segment_id: str = Field(..., pattern="^[0-9a-f]{64}$")
    event_type: AttributionEventType
    event_time_utc: datetime
    stimulus_time_utc: datetime | None = None
    selected_arm_id: str
    expected_rule_text_hash: str = Field(..., pattern="^[0-9a-f]{64}$")
    semantic_method: SemanticMethod
    semantic_method_version: str
    semantic_p_match: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_reason_code: SemanticReasonCode | None = None
    reward_path_version: str
    bandit_decision_snapshot: BanditDecisionSnapshot
    evidence_flags: list[str] = Field(default_factory=list)
    finality: AttributionFinality = Field(..., description=_FINALITY_DESCRIPTION)
    schema_version: str = Field(..., description=_SCHEMA_VERSION_DESCRIPTION)
    created_at: datetime = Field(..., description=_CREATED_AT_DESCRIPTION)

    @field_validator("evidence_flags")
    @classmethod
    def _evidence_flags_unique(cls, values: list[str]) -> list[str]:
        return _ensure_unique(values, "evidence_flags")


class OutcomeEvent(AttributionBaseModel):
    """
    Validate a delayed downstream outcome record (§6.4.2).

    Accepts caller-provided outcome identity, session context, outcome type/value,
    source metadata, confidence, finality, schema version, and creation time.
    Produces a strict ledger outcome model; it does not discover outcomes or link
    them to events.
    """

    outcome_id: UUID = Field(
        ...,
        description=(
            "Caller-provided deterministic UUIDv5 derived from session_id, "
            "outcome_type, outcome_time_utc, source_system, and "
            "source_event_ref; replay-stable for attribution backfill "
            "idempotency."
        ),
    )
    session_id: UUID
    outcome_type: OutcomeEventType
    outcome_value: float
    outcome_time_utc: datetime
    source_system: str
    source_event_ref: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    finality: AttributionFinality = Field(..., description=_FINALITY_DESCRIPTION)
    schema_version: str = Field(..., description=_SCHEMA_VERSION_DESCRIPTION)
    created_at: datetime = Field(..., description=_CREATED_AT_DESCRIPTION)


class EventOutcomeLink(AttributionBaseModel):
    """
    Validate a deterministic event-to-outcome eligibility link (§6.4.3).

    Accepts caller-provided link identity, event/outcome IDs, non-negative lag,
    horizon, link-rule version, eligibility flags, finality, schema version, and
    creation time. Produces a strict link model; it does not score attribution or
    create OutcomeEvent records.
    """

    link_id: UUID = Field(
        ...,
        description=(
            "Caller-provided deterministic UUIDv5 derived from event_id, "
            "outcome_id, and link_rule_version; replay-stable for attribution "
            "backfill idempotency."
        ),
    )
    event_id: UUID
    outcome_id: UUID
    lag_s: float = Field(..., ge=0.0)
    horizon_s: float = Field(..., gt=0.0)
    link_rule_version: str
    eligibility_flags: list[str]
    finality: AttributionFinality = Field(..., description=_FINALITY_DESCRIPTION)
    schema_version: str = Field(..., description=_SCHEMA_VERSION_DESCRIPTION)
    created_at: datetime = Field(..., description=_CREATED_AT_DESCRIPTION)

    @field_validator("eligibility_flags")
    @classmethod
    def _eligibility_flags_unique(cls, values: list[str]) -> list[str]:
        return _ensure_unique(values, "eligibility_flags")


class AttributionScore(AttributionBaseModel):
    """
    Validate a method-specific AttributionScore record (§6.4.4).

    Accepts caller-provided score identity, related event/outcome IDs, method
    metadata, raw/normalized scores, confidence, evidence flags, finality, schema
    version, and creation time. Produces a strict score model; it does not compute
    the score or alter Thompson Sampling updates.
    """

    score_id: UUID = Field(
        ...,
        description=(
            "Caller-provided deterministic UUIDv5 derived from event_id, "
            "outcome_id, attribution_method, and method_version; finality "
            "is a mutable lifecycle field so replay/finalization can upsert "
            "the same AttributionScore row."
        ),
    )
    event_id: UUID
    outcome_id: UUID | None = None
    attribution_method: str
    method_version: str
    score_raw: float | None = Field(...)
    score_normalized: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_flags: list[str] = Field(default_factory=list)
    finality: AttributionFinality = Field(..., description=_FINALITY_DESCRIPTION)
    schema_version: str = Field(..., description=_SCHEMA_VERSION_DESCRIPTION)
    created_at: datetime = Field(..., description=_CREATED_AT_DESCRIPTION)

    @field_validator("evidence_flags")
    @classmethod
    def _score_evidence_flags_unique(cls, values: list[str]) -> list[str]:
        return _ensure_unique(values, "evidence_flags")
