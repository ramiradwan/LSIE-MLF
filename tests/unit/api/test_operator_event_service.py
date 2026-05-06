"""Unit tests for OperatorEventService SSE payload behavior."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from packages.schemas.operator_console import (
    HealthSnapshot,
    HealthState,
    OperatorEventType,
    OverviewSnapshot,
    SessionSummary,
)
from services.api.services.operator_event_service import OperatorChangeMarker, OperatorEventService


def _now() -> datetime:
    return datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


class _DisconnectAfter:
    def __init__(self, *, checks_before_disconnect: int) -> None:
        self._remaining = checks_before_disconnect

    async def is_disconnected(self) -> bool:
        if self._remaining <= 0:
            return True
        self._remaining -= 1
        return False


def test_build_bootstrap_reads_full_current_state() -> None:
    session_id = uuid.uuid4()
    health = HealthSnapshot(generated_at_utc=_now(), overall_state=HealthState.OK)
    session = SessionSummary(session_id=session_id, status="active", started_at_utc=_now())
    overview = OverviewSnapshot(generated_at_utc=_now(), active_session=session, health=health)
    svc = MagicMock()
    svc.get_overview.return_value = overview
    svc.list_sessions.return_value = [session]
    svc.list_encounters.return_value = []
    svc.get_session_physiology.return_value = None
    svc.list_experiments.return_value = []
    svc.get_health.return_value = health
    svc.list_alerts.return_value = []

    service = OperatorEventService(
        read_service=svc,
        marker_provider=lambda: {},
        clock=_now,
    )

    bootstrap = asyncio.run(service.build_bootstrap())

    assert bootstrap.overview is overview
    assert bootstrap.live_session is session
    assert bootstrap.health is health
    svc.list_encounters.assert_called_once_with(session_id, limit=100, before_utc=None)


def test_stream_events_reconnect_emits_current_state_and_suppresses_duplicate() -> None:
    health = HealthSnapshot(generated_at_utc=_now(), overall_state=HealthState.OK)
    overview = OverviewSnapshot(generated_at_utc=_now(), health=health)
    svc = MagicMock()
    svc.get_overview.return_value = overview
    svc.list_sessions.return_value = []

    markers: dict[OperatorEventType, OperatorChangeMarker] = {
        "overview": {"version": 1},
        "sessions": {"version": 2},
    }
    duplicate = "overview:3711dbff9ec55bc2aa05744e"
    service = OperatorEventService(
        read_service=svc,
        marker_provider=lambda: markers,
        clock=_now,
        poll_interval_s=0.0,
    )

    async def collect() -> list[Any]:
        events = []
        async for event in service.stream_events(
            _DisconnectAfter(checks_before_disconnect=3),
            last_event_id=duplicate,
        ):
            events.append(event)
            if len(events) == 1:
                break
        return events

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0].event == "sessions"
    assert events[0].id != duplicate
    svc.get_overview.assert_called_once()
    svc.list_sessions.assert_called_once_with(limit=50)


def test_stream_events_emits_only_changed_markers() -> None:
    health = HealthSnapshot(generated_at_utc=_now(), overall_state=HealthState.OK)
    overview = OverviewSnapshot(generated_at_utc=_now(), health=health)
    svc = MagicMock()
    svc.get_overview.return_value = overview

    markers: list[dict[OperatorEventType, OperatorChangeMarker]] = [
        {"overview": {"version": 1}},
        {"overview": {"version": 2}},
    ]

    def marker_provider() -> dict[OperatorEventType, OperatorChangeMarker]:
        return markers.pop(0) if markers else {"overview": {"version": 2}}

    service = OperatorEventService(
        read_service=svc,
        marker_provider=marker_provider,
        clock=_now,
        poll_interval_s=0.0,
    )

    async def collect() -> list[Any]:
        events = []
        async for event in service.stream_events(
            _DisconnectAfter(checks_before_disconnect=3),
            last_event_id=None,
        ):
            events.append(event)
            break
        return events

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0].event == "overview"
    assert str(events[0].id).startswith("overview:")
    assert events[0].retry == 1000
    assert svc.get_overview.call_count == 1
