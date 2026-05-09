"""Operator Console server-sent event service."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import AsyncIterable, Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID

from fastapi.sse import ServerSentEvent

from packages.schemas.operator_console import (
    AlertEvent,
    EncounterSummary,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    OperatorEventEnvelope,
    OperatorEventPayload,
    OperatorEventType,
    OperatorStateBootstrap,
    OverviewSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
)

OperatorMarkerValue = str | int | float | None
OperatorChangeMarker = dict[str, OperatorMarkerValue]
MarkerProvider = Callable[[], Mapping[OperatorEventType, OperatorChangeMarker]]


class DisconnectCheck(Protocol):
    async def is_disconnected(self) -> bool: ...


class OperatorReadServiceLike(Protocol):
    def get_overview(self) -> OverviewSnapshot: ...

    def list_sessions(self, *, limit: int = 50) -> list[SessionSummary]: ...

    def get_session(self, session_id: UUID) -> SessionSummary | None: ...

    def list_encounters(
        self,
        session_id: UUID,
        *,
        limit: int = 100,
        before_utc: datetime | None = None,
    ) -> list[EncounterSummary]: ...

    def list_experiments(self) -> list[ExperimentSummary]: ...

    def get_experiment_detail(self, experiment_id: str) -> ExperimentDetail | None: ...

    def get_session_physiology(self, session_id: UUID) -> SessionPhysiologySnapshot | None: ...

    def get_health(self) -> HealthSnapshot | object: ...

    def list_alerts(
        self, *, limit: int = 50, since_utc: datetime | None = None
    ) -> list[AlertEvent]: ...


class OperatorEventService:
    def __init__(
        self,
        *,
        read_service: OperatorReadServiceLike,
        marker_provider: MarkerProvider | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        poll_interval_s: float = 0.5,
        retry_ms: int = 1000,
        sessions_limit: int = 50,
        encounters_limit: int = 100,
        alerts_limit: int = 50,
    ) -> None:
        self._read_service = read_service
        self._marker_provider = marker_provider
        self._clock = clock
        self._poll_interval_s = poll_interval_s
        self._retry_ms = retry_ms
        self._sessions_limit = sessions_limit
        self._encounters_limit = encounters_limit
        self._alerts_limit = alerts_limit

    def has_event_stream_support(self) -> bool:
        return self._marker_provider is not None

    async def build_bootstrap(self) -> OperatorStateBootstrap:
        overview = self._read_service.get_overview()
        sessions = self._read_service.list_sessions(limit=self._sessions_limit)
        live_session = overview.active_session
        encounters: list[EncounterSummary] = []
        physiology: SessionPhysiologySnapshot | None = None
        experiment: ExperimentDetail | None = None
        if live_session is not None:
            encounters = self._read_service.list_encounters(
                live_session.session_id,
                limit=self._encounters_limit,
                before_utc=None,
            )
            physiology = self._read_service.get_session_physiology(live_session.session_id)
            if live_session.experiment_id is not None:
                experiment = self._read_service.get_experiment_detail(live_session.experiment_id)
        experiment_summaries = self._read_service.list_experiments()
        health = await self._resolve_health()
        alerts = self._read_service.list_alerts(limit=self._alerts_limit, since_utc=None)
        return OperatorStateBootstrap(
            generated_at_utc=self._clock(),
            overview=overview,
            sessions=sessions,
            live_session=live_session,
            encounters=encounters,
            experiment_summaries=experiment_summaries,
            experiment=experiment,
            physiology=physiology,
            health=health,
            alerts=alerts,
        )

    async def stream_events(
        self,
        request: DisconnectCheck,
        *,
        last_event_id: str | None = None,
    ) -> AsyncIterable[ServerSentEvent]:
        previous_markers = {} if last_event_id is not None else self._snapshot_markers()
        while not await request.is_disconnected():
            await asyncio.sleep(self._poll_interval_s)
            if await request.is_disconnected():
                break
            current_markers = self._snapshot_markers()
            for event_type in _EVENT_ORDER:
                marker = current_markers.get(event_type, {})
                if previous_markers.get(event_type, {}) == marker:
                    continue
                if _cursor_for(event_type, marker) == last_event_id:
                    continue
                envelope = await self._build_envelope(event_type, marker)
                yield ServerSentEvent(
                    data=envelope,
                    event=envelope.event_type,
                    id=envelope.event_id,
                    retry=self._retry_ms,
                )
            previous_markers = current_markers

    def _snapshot_markers(self) -> dict[OperatorEventType, OperatorChangeMarker]:
        if self._marker_provider is None:
            raise RuntimeError("operator event stream is not available in this runtime")
        raw_markers = self._marker_provider()
        return {event_type: dict(raw_markers.get(event_type, {})) for event_type in _EVENT_ORDER}

    async def _build_envelope(
        self,
        event_type: OperatorEventType,
        marker: OperatorChangeMarker,
    ) -> OperatorEventEnvelope:
        generated_at = self._clock()
        cursor = _cursor_for(event_type, marker)
        return OperatorEventEnvelope(
            event_id=cursor,
            event_type=event_type,
            cursor=cursor,
            generated_at_utc=generated_at,
            payload=await self._payload_for(event_type),
        )

    async def _payload_for(self, event_type: OperatorEventType) -> OperatorEventPayload:
        overview = self._read_service.get_overview()
        active_session = overview.active_session
        if event_type == "overview":
            return overview
        if event_type == "sessions":
            return self._read_service.list_sessions(limit=self._sessions_limit)
        if event_type == "live_session":
            if active_session is None:
                return []
            live_session = self._read_service.get_session(active_session.session_id)
            return [] if live_session is None else live_session
        if event_type == "encounters":
            if active_session is None:
                return []
            return self._read_service.list_encounters(
                active_session.session_id,
                limit=self._encounters_limit,
                before_utc=None,
            )
        if event_type == "experiment_summaries":
            return self._read_service.list_experiments()
        if event_type == "experiment":
            detail = self._active_experiment_detail(active_session)
            return [] if detail is None else detail
        if event_type == "physiology":
            if active_session is None:
                return []
            physiology = self._read_service.get_session_physiology(active_session.session_id)
            return [] if physiology is None else physiology
        if event_type == "health":
            return await self._resolve_health()
        return self._read_service.list_alerts(limit=self._alerts_limit, since_utc=None)

    def _active_experiment_detail(
        self, active_session: SessionSummary | None
    ) -> ExperimentDetail | None:
        if active_session is None or active_session.experiment_id is None:
            return None
        return self._read_service.get_experiment_detail(active_session.experiment_id)

    async def _resolve_health(self) -> HealthSnapshot:
        result = self._read_service.get_health()
        if inspect.isawaitable(result):
            return cast(HealthSnapshot, await result)
        return cast(HealthSnapshot, result)


def _cursor_for(event_type: OperatorEventType, marker: OperatorChangeMarker) -> str:
    marker_json = json.dumps(marker, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{event_type}:{marker_json}".encode()).hexdigest()[:24]
    return f"{event_type}:{digest}"


_EVENT_ORDER: tuple[OperatorEventType, ...] = (
    "overview",
    "sessions",
    "live_session",
    "encounters",
    "experiment_summaries",
    "experiment",
    "physiology",
    "health",
    "alerts",
)

__all__ = [
    "MarkerProvider",
    "OperatorChangeMarker",
    "OperatorEventService",
    "OperatorReadServiceLike",
]
