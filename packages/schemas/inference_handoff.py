"""
InferenceHandoffPayload — §6.1 JSON Schema Draft 07 Contract

Standardized schema for multimodal ML pipeline handoff between
Module C → Module D and Module D → Module E. v3.4 adds stable segment
identity/window fields, first-class bandit decision capture, and optional
strict physiological context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import UUID4, AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from packages.schemas.attribution import BanditDecisionSnapshot
from packages.schemas.physiology import PhysiologicalChunkEvent, PhysiologicalContext

# §6 compatibility export: preserve the historical field name while carrying
# the canonical PhysiologicalChunkEvent schema definition.
physiological_sample_event_schema: dict[str, Any] = PhysiologicalChunkEvent.model_json_schema()

UUID4_PATTERN = "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


class MediaSource(BaseModel):
    """Media source metadata attached to each InferenceHandoffPayload."""

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


class AU12Observation(BaseModel):
    """One bounded AU12 observation attached to the inference handoff."""

    timestamp_s: float
    intensity: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class InferenceHandoffPayload(BaseModel):
    """
    §6.1 — The JSON Schema Draft 07 contract governing data exchange
    between Module C and Module D, and from Module D to Module E.

    Underscore-prefixed payload keys are represented by internal Python field
    names with exact external aliases so Pydantic treats them as normal modeled
    fields rather than private attributes or free-form extras.
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
    expected_greeting: str = Field(
        ...,
        alias="_expected_greeting",
        description="Greeting rule text assigned to the active arm for semantic evaluation.",
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
    x_max: float | None = Field(
        ...,
        alias="_x_max",
        description="Per-subject maximum raw D_mouth/IOD ratio observed during calibration.",
    )
    bandit_decision_snapshot: BanditDecisionSnapshot = Field(
        ...,
        alias="_bandit_decision_snapshot",
        description=(
            "Pre-update Thompson Sampling selection snapshot captured at arm-selection time."
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
