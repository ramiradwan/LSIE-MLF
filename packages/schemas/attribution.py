"""Attribution ledger schemas — §6.4 v3.4 contracts.

These models define typed Pydantic contracts for deterministic attribution
ledger records. Identifier fields document their replay-stable UUIDv5
semantics, but this module intentionally does not generate identifiers or
audit timestamps at runtime; producing services must supply those values.
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
    """Shared strict configuration for §6.4 attribution contracts."""

    model_config = ConfigDict(extra="forbid")


class ArmPosterior(AttributionBaseModel):
    """Posterior parameters for one candidate bandit arm."""

    alpha: float = Field(..., gt=0.0)
    beta: float = Field(..., gt=0.0)


class BanditDecisionSnapshot(AttributionBaseModel):
    """§6.4 / §6.1 — Pre-update Thompson Sampling selection snapshot."""

    selection_method: BanditSelectionMethod
    selection_time_utc: datetime
    experiment_id: int
    policy_version: str
    selected_arm_id: str
    candidate_arm_ids: list[str] = Field(..., min_length=1)
    posterior_by_arm: dict[str, ArmPosterior] = Field(..., min_length=1)
    sampled_theta_by_arm: dict[str, float] | None = None
    expected_greeting: str
    decision_context_hash: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    random_seed: int | None = None

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
    """§6.4.1 — Greeting interaction event record for attribution analytics."""

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
    """§6.4.2 — Delayed downstream outcome record."""

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
    """§6.4.3 — Deterministic event→outcome eligibility link record."""

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
    """§6.4.4 — Method-specific attribution score record."""

    score_id: UUID = Field(
        ...,
        description=(
            "Caller-provided deterministic UUIDv5 derived from event_id, "
            "outcome_id, attribution_method, and method_version; finality "
            "is a mutable lifecycle field so replay/finalization can upsert "
            "the same score row."
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
