"""
Pydantic contracts for Module C → D → E inference handoff (§6.1).

The schema validates multimodal segment identity, drift-corrected UTC window
bounds, media metadata, active experiment fields, AU12 telemetry, pre-update
BanditDecisionSnapshot records, and optional physiological context. It uses canonical
field aliases for underscore-prefixed transport keys (§0, §13.15) and rejects
null physiological context placeholders; it does not run inference or compute
rewards.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from pydantic import (
    UUID4,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from packages.schemas.attribution import BanditDecisionSnapshot
from packages.schemas.evaluation import (
    ResponseReasonCode,
    ResponseRegistrationStatus,
    StimulusModality,
    StimulusPayload,
)
from packages.schemas.physiology import PhysiologicalChunkEvent, PhysiologicalContext

# §6 compatibility export: preserve the historical field name while carrying
# the canonical PhysiologicalChunkEvent schema definition.
physiological_sample_event_schema: dict[str, Any] = PhysiologicalChunkEvent.model_json_schema()

UUID4_PATTERN = "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


class MediaSource(BaseModel):
    """
    Validate source media metadata for an inference handoff.

    Accepts a stream URI, codec, and positive [width, height] resolution and
    produces strict serialized metadata for the segment payload. It does not
    inspect media bytes or infer codec/resolution values.
    """

    stream_url: AnyUrl = Field(..., description="URI of the source stream.")
    codec: str = Field(..., pattern="^(h264|h265|raw)$")
    resolution: list[int] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="[width, height] in pixels.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("resolution")
    @classmethod
    def _positive_dims(cls, v: list[int]) -> list[int]:
        if any(d < 1 for d in v):
            raise ValueError("Resolution dimensions must be >= 1.")
        return v


class ResponseInference(BaseModel):
    """
    Validate bounded observational subject-response inference output.

    Accepts a response match flag, confidence, bounded registration status and
    reason code, plus optional evidence timing metadata. It does not alter
    delivery validation or reward-path semantics.
    """

    is_match: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    registration_status: ResponseRegistrationStatus
    response_reason_code: ResponseReasonCode
    response_type: str | None = Field(default=None, min_length=1)
    matched_response_time: float | None = Field(default=None, ge=0.0)
    evidence_span_ref: str | None = Field(default=None, min_length=1)

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_absent_optionals(self, handler: Any) -> dict[str, Any]:
        data = cast(dict[str, Any], handler(self))
        for key in ("response_type", "matched_response_time", "evidence_span_ref"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class AU12Observation(BaseModel):
    """
    Validate one bounded AU12 telemetry sample.

    Accepts a drift-corrected epoch timestamp and a [0, 1] intensity value and
    produces the strict item schema used in ``_au12_series``. It does not compute
    or recalibrate AU12 values.
    """

    timestamp_s: float
    intensity: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class InferenceHandoffPayload(BaseModel):
    """
    Validate the §6.1 handoff payload exchanged across pipeline modules.

    Accepts segment IDs/windows, media metadata, Module C experiment context,
    per-frame AU12 observations, the pre-update BanditDecisionSnapshot, and an
    optional non-empty physiological context. Produces a Draft 07-compatible
    Pydantic model using canonical external aliases for underscore-prefixed
    fields. It does not compute segment IDs, run semantic evaluation, calculate
    rewards, or accept ``None`` as a present physiological context.
    """

    session_id: UUID4 = Field(
        ...,
        description="UUID v4 representing the continuous live stream session.",
        json_schema_extra={"pattern": UUID4_PATTERN},
    )
    segment_id: str = Field(
        ...,
        pattern="^[0-9a-f]{64}$",
        description=(
            "Deterministic SHA-256 segment identifier derived from stable segment "
            "identity fields; replay-stable across backfill."
        ),
    )
    segment_window_start_utc: datetime = Field(
        ..., description="RFC 3339 UTC timestamp of the 30-second segment start boundary."
    )
    segment_window_end_utc: datetime = Field(
        ..., description="RFC 3339 UTC timestamp of the 30-second segment end boundary."
    )
    timestamp_utc: datetime = Field(
        ..., description="RFC 3339 UTC timestamp of inference event completion."
    )
    media_source: MediaSource
    segments: list[dict[str, Any]] = Field(..., description="Array of analyzed video segments.")
    active_arm: str = Field(
        ...,
        alias="_active_arm",
        description="Thompson Sampling arm identifier selected for this segment.",
    )
    experiment_id: int = Field(
        ...,
        alias="_experiment_id",
        description="Persistent Store experiment row ID for this session.",
    )
    stimulus_modality: StimulusModality = Field(
        ...,
        alias="_stimulus_modality",
        description="Stimulus modality assigned to the active arm for this segment.",
    )
    stimulus_payload: StimulusPayload = Field(
        ...,
        alias="_stimulus_payload",
        description="Typed stimulus payload assigned to the active arm for this segment.",
    )
    expected_stimulus_rule: str = Field(
        ...,
        alias="_expected_stimulus_rule",
        description="Stimulus rule text assigned to the active arm for validation.",
        min_length=1,
    )
    expected_response_rule: str = Field(
        ...,
        alias="_expected_response_rule",
        description=(
            "Observed-response rule text assigned to the active arm for semantic evaluation."
        ),
        min_length=1,
    )
    stimulus_time: float | None = Field(
        ...,
        alias="_stimulus_time",
        description=(
            "Drift-corrected UTC epoch seconds of stimulus injection, or null if no "
            "stimulus occurred in this segment."
        ),
    )
    au12_series: list[AU12Observation] = Field(
        ...,
        alias="_au12_series",
        description="Per-frame AU12 bounded intensity observations.",
    )
    bandit_decision_snapshot: BanditDecisionSnapshot = Field(
        ...,
        alias="_bandit_decision_snapshot",
        description=(
            "Pre-update Thompson Sampling BanditDecisionSnapshot captured at arm-selection time."
        ),
    )
    physiological_context: PhysiologicalContext | SkipJsonSchema[None] = Field(
        default=None,
        alias="_physiological_context",
        description=(
            "Optional per-subject physiological context snapshot attached when eligible "
            "physiological data exists. Omit the key when no real role snapshot exists."
        ),
    )
    stimulus_id: UUID4 | None = Field(default=None, alias="_stimulus_id")
    response_observation_horizon_s: float | None = Field(
        default=None,
        alias="_response_observation_horizon_s",
        ge=0.0,
    )
    response_inference: ResponseInference | None = None

    model_config = ConfigDict(
        extra="forbid",
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
        json_schema_extra={"$schema": "http://json-schema.org/draft-07/schema#"},
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_null_physiological_context(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("_physiological_context", "physiological_context"):
                if key in data and data[key] is None:
                    raise ValueError(
                        "_physiological_context must be omitted or contain at least "
                        "one real role snapshot"
                    )
        return data
