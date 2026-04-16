"""
Physiological Schemas — §6 (v3.1 amendment), §4.B.2, §0.3

Pydantic models for normalized physiological telemetry events and
the per-subject context snapshot attached to InferenceHandoffPayload.

Design constraints:
  - PhysiologicalSampleEvent mirrors the LiveEvent pattern from §4.B
    (uniqueId, event_type, timestamp_utc, payload) but with physiology-
    specific payload fields.
  - PhysiologicalSnapshot is validated independently before dict-injection
    into assemble_segment(). It does NOT become a typed field on
    InferenceHandoffPayload — the existing payload uses dict-injection
    for experiment fields (_active_arm, _au12_series, etc.) and physiology
    follows the same pattern.
  - PhysiologicalContext wraps two optional snapshots (streamer, operator)
    and is the shape of the _physiological_context dict key.

Spec references:
  §6     — Interface contracts and payload schemas
  §4.B.2 — Physiological Ingestion Adapter
  §0.3   — Canonical terms: Physiological Sample Event, Physiological
           Context, subject_role
  §5     — Data governance: raw payloads are Transient Data
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class PhysiologicalPayload(BaseModel):
    """Normalized physiological measurements from a wearable provider.

    §6 — RMSSD and heart rate are the normalized physiological metrics.
    sample_window_s reflects the provider's aggregation interval
    (e.g., 300s for Oura Ring 5-minute epochs).
    """

    rmssd_ms: float | None = Field(
        None,
        ge=0.0,
        description="Root Mean Square of Successive Differences (ms). Null if unavailable.",
    )
    heart_rate_bpm: int | None = Field(
        None,
        ge=20,
        le=300,
        description="Heart rate in beats per minute. Null if unavailable.",
    )
    sample_window_s: int = Field(
        ...,
        gt=0,
        description="Provider aggregation window in seconds (e.g., 300 for Oura).",
    )


class PhysiologicalSampleEvent(BaseModel):
    """§4.B.2 — Normalized physiological event for inter-module transport.

    Produced by the API Server webhook route and enqueued to Redis
    for the Orchestrator to drain. Mirrors the LiveEvent pattern
    from §4.B (uniqueId, event_type, timestamp_utc, payload).
    """

    unique_id: UUID = Field(..., description="UUID v4 for idempotency tracking.")
    event_type: Literal["physiological_sample"] = "physiological_sample"
    provider: Literal["oura"] = Field(..., description="Wearable data provider identifier.")
    subject_role: Literal["streamer", "operator"] = Field(
        ...,
        description="§0.3 canonical term: identifies the data subject.",
    )
    source_timestamp_utc: datetime = Field(
        ...,
        description="Provider-reported UTC timestamp of the measurement.",
    )
    ingest_timestamp_utc: datetime = Field(
        ...,
        description="UTC timestamp when the API Server received the webhook.",
    )
    payload: PhysiologicalPayload

    model_config = {"json_schema_extra": {"$schema": "http://json-schema.org/draft-07/schema#"}}


class PhysiologicalSnapshot(BaseModel):
    """Per-subject physiological state snapshot attached to segments.

    §4.C.3 — The Orchestrator's PhysiologicalStateBuffer maintains
    one of these per subject_role. Freshness is computed at segment
    assembly time, not at ingestion time.
    """

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

    @field_validator("freshness_s")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("freshness_s cannot be negative.")
        return v


class PhysiologicalContext(BaseModel):
    """§6 (v3.1 amendment) — Shape of the _physiological_context dict key.

    Both subjects are optional. Null means no data available for that
    role. The entire _physiological_context key is omitted from the
    payload when physiology is not enabled.
    """

    streamer: PhysiologicalSnapshot | None = None
    operator: PhysiologicalSnapshot | None = None
