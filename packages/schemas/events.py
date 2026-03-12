"""
Event Schemas — §4.B Module B Ground Truth Ingestion

Pydantic models for parsed TikTok Webcast events and composite
Action_Combo triggers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LiveEvent(BaseModel):
    """Single parsed event from TikTokLive WebSocket stream."""

    unique_id: str = Field(..., alias="uniqueId")
    event_type: str
    timestamp_utc: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class GiftEvent(LiveEvent):
    """Gift event with monetary value."""

    gift_value: int = Field(0, ge=0)


class ComboEvent(BaseModel):
    """
    Action_Combo constraint — §4.B.1
    Multiple events that must coincide temporally to form a valid
    ground truth signal.
    """

    events: list[LiveEvent] = Field(..., min_length=2)
    window_start: datetime
    window_end: datetime
    is_valid: bool = False


class GroundTruthRecord(BaseModel):
    """Staged ground truth record for synchronization with inference outputs."""

    unique_id: str
    event_type: str
    timestamp_utc: datetime
    gift_value: int | None = None
    combo_events: list[LiveEvent] | None = None
