"""
Physiological Schemas — §6 (v3.2 amendment), §4.C.4, §0.3

Pydantic models for normalized physiological telemetry transport and
the per-subject context snapshot attached to InferenceHandoffPayload.

Design constraints:
  - PhysiologicalChunkEvent mirrors the API Server → Orchestrator transport
    for hydrated wearable chunks rather than retired scalar samples.
  - PhysiologicalSnapshot is validated independently before dict-injection
    into assemble_segment(). It does NOT become a typed field on
    InferenceHandoffPayload — the existing payload uses dict-injection
    for experiment fields (_active_arm, _au12_series, etc.) and physiology
    follows the same pattern.
  - PhysiologicalContext wraps two optional snapshots (streamer, operator)
    and is the shape of the _physiological_context dict key.

Spec references:
  §6     — InferenceHandoffPayload interface contract
  §4.C.4 — Physiological State Buffer
  §0.3   — Canonical terms: Physiological Chunk Event, Physiological
           Context, subject_role
  §5     — Data governance: raw payloads are Transient Data
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class PhysiologicalChunkPayload(BaseModel):
    """§6.2 — Normalized hydrated chunk payload from a wearable provider."""

    sample_interval_s: int = Field(
        ...,
        gt=0,
        description="Provider sample spacing in seconds for the chunk payload.",
    )
    valid_sample_count: int = Field(
        ...,
        ge=0,
        description="Count of valid provider samples contributing to the chunk.",
    )
    expected_sample_count: int = Field(
        ...,
        ge=0,
        description="Expected provider samples for the chunk window.",
    )
    derivation_method: Literal["provider", "server"] = Field(
        ...,
        description="How the chunked values were derived before transport.",
    )
    ibi_ms_items: list[float] | None = Field(
        None,
        description="Optional IBI samples in milliseconds.",
    )
    rmssd_items_ms: list[float] | None = Field(
        None,
        description="Optional RMSSD samples in milliseconds.",
    )
    heart_rate_items_bpm: list[int] | None = Field(
        None,
        description="Optional heart-rate samples in beats per minute.",
    )
    motion_items: list[float] | None = Field(
        None,
        description="Optional motion magnitude samples for the chunk payload.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("ibi_ms_items")
    @classmethod
    def _positive_ibi_items(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if any(item <= 0 for item in value):
            raise ValueError("ibi_ms_items entries must be greater than 0")
        return value

    @field_validator("rmssd_items_ms", "motion_items")
    @classmethod
    def _non_negative_float_items(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if any(item < 0 for item in value):
            raise ValueError("physiological sample arrays cannot contain negative values")
        return value

    @field_validator("heart_rate_items_bpm")
    @classmethod
    def _bounded_heart_rate_items(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if any(item < 20 or item > 300 for item in value):
            raise ValueError("heart_rate_items_bpm entries must be between 20 and 300")
        return value

    @model_validator(mode="after")
    def _require_primary_series(self) -> PhysiologicalChunkPayload:
        if not any(
            series is not None
            for series in (self.ibi_ms_items, self.rmssd_items_ms, self.heart_rate_items_bpm)
        ):
            raise ValueError(
                "at least one of ibi_ms_items, rmssd_items_ms, "
                "or heart_rate_items_bpm must be present"
            )
        return self


class PhysiologicalChunkEvent(BaseModel):
    """§6.2 — Canonical transport event for hydrated physiological chunks."""

    unique_id: UUID = Field(..., description="UUID v4 for idempotency tracking.")
    event_type: Literal["physiological_chunk"] = "physiological_chunk"
    provider: Literal["oura"] = Field(..., description="Wearable data provider identifier.")
    subject_role: Literal["streamer", "operator"] = Field(
        ...,
        description="§0.3 canonical term: identifies the data subject.",
    )
    source_kind: Literal["ibi", "session"] = Field(
        ...,
        description="Normalized provider resource family used for this chunk.",
    )
    window_start_utc: datetime = Field(
        ...,
        description="Inclusive UTC start timestamp for the hydrated provider chunk.",
    )
    window_end_utc: datetime = Field(
        ...,
        description="Inclusive UTC end timestamp for the hydrated provider chunk.",
    )
    ingest_timestamp_utc: datetime = Field(
        ...,
        description="UTC timestamp when normalized transport was produced.",
    )
    payload: PhysiologicalChunkPayload

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"$schema": "http://json-schema.org/draft-07/schema#"},
    }


# Deprecated non-canonical event alias retired during the staged refactor.
# The legacy scalar event name is intentionally not preserved as a live alias.


class PhysiologicalSnapshot(BaseModel):
    """Per-subject derived physiological state snapshot attached to segments."""

    rmssd_ms: float | None = Field(None, ge=0.0)
    heart_rate_bpm: int | None = Field(None, ge=20, le=300)
    source_timestamp_utc: datetime
    freshness_s: float = Field(
        ...,
        ge=0.0,
        description="Seconds since source_timestamp_utc at segment assembly time.",
    )
    is_stale: bool = Field(
        ...,
        description="True if freshness_s exceeds PHYSIO_STALENESS_THRESHOLD_S.",
    )
    provider: str
    source_kind: Literal["ibi", "session"]
    derivation_method: Literal["provider", "server"]
    window_s: int = Field(..., gt=0)
    validity_ratio: float = Field(..., ge=0.0, le=1.0)
    is_valid: bool

    model_config = {"extra": "forbid"}

    @field_validator("freshness_s")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("freshness_s cannot be negative.")
        return value


class PhysiologicalContext(BaseModel):
    """§6.3 — Shape of the optional _physiological_context dict key.

    Streamer and operator entries are independently optional. Either role may be
    omitted or set to null when no derived snapshot is available. The entire
    _physiological_context key is omitted from the payload when physiology is not
    enabled or no subject snapshots are attached for that segment.
    """

    streamer: PhysiologicalSnapshot | None = None
    operator: PhysiologicalSnapshot | None = None

    model_config = {"extra": "forbid"}
