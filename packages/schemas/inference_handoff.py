"""
InferenceHandoffPayload — §6.1 JSON Schema Draft 07 Contract

Standardized schema for multimodal ML pipeline handoff between
Module C → Module D and Module D → Module E.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class MediaSource(BaseModel):
    """Media source metadata attached to each InferenceHandoffPayload."""

    stream_url: str = Field(..., description="URI of the source stream.")
    codec: str = Field(..., pattern="^(h264|h265|raw)$")
    resolution: list[int] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="[width, height] in pixels.",
    )

    @field_validator("resolution")
    @classmethod
    def _positive_dims(cls, v: list[int]) -> list[int]:
        if any(d < 1 for d in v):
            raise ValueError("Resolution dimensions must be >= 1.")
        return v


class InferenceHandoffPayload(BaseModel):
    """
    §6.1 — The JSON Schema Draft 07 contract governing data exchange
    between Module C and Module D, and from Module D to Module E.
    """

    session_id: UUID = Field(
        ..., description="UUID v4 representing the continuous live stream session."
    )
    timestamp_utc: datetime = Field(
        ..., description="RFC 3339 UTC timestamp of inference event completion."
    )
    media_source: MediaSource
    segments: list[dict[str, Any]] = Field(
        default_factory=list, description="Array of analyzed video segments."
    )

    model_config = {"json_schema_extra": {"$schema": "http://json-schema.org/draft-07/schema#"}}
