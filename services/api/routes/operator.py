"""
Operator Console aggregate routes — `/api/v1/operator/*`.

These endpoints fan in the data the PySide6 Operator Console needs and
fan out a single write surface for stimulus submission. Response models
are DTOs from `packages.schemas.operator_console`; the heavy
lifting lives in `OperatorReadService`/`OperatorActionService` so the
route layer stays thin.

Design constraints:
  - Every response is a Pydantic DTO — no raw dicts cross the wire.
  - Read handlers never call the DB directly; they go through
    `OperatorReadService` which owns connection lifecycle and DTO
    assembly.
  - The single write path (`POST /sessions/{id}/stimulus`) flows
    through `OperatorActionService`; desktop dependency overrides replace
    its default trigger channel with SQLite/IPC-backed service methods.
  - Operator-safe error payloads: RuntimeError → 503, unexpected →
    500, missing resource → 404, state conflict → 409.

Spec references:
  §4.C       — Stimulus lifecycle; authoritative `_stimulus_time`
               stays orchestrator-owned.
  §4.C.4     — Physiological State Buffer freshness semantics.
  §4.E.1     — Operator-facing execution details.
  §4.E.2     — Physiology persistence.
  §7B        — Thompson Sampling reward fields.
  §7C        — Co-Modulation Index (null-valid).
  §12        — Error-handling matrix.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from typing import Annotated, Any, Protocol, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.concurrency import run_in_threadpool

from packages.schemas.operator_console import (
    AlertEvent,
    CloudActionStatus,
    CloudAuthState,
    CloudAuthStatus,
    CloudExperimentRefreshStatus,
    CloudOperatorErrorCode,
    CloudOutboxSummary,
    CloudSignInResult,
    EncounterSummary,
    ExperimentBundleRefreshPreview,
    ExperimentBundleRefreshRequest,
    ExperimentBundleRefreshResult,
    ExperimentDetail,
    ExperimentSummary,
    HealthSnapshot,
    OperatorStateBootstrap,
    OverviewSnapshot,
    SessionPhysiologySnapshot,
    SessionSummary,
    StimulusAccepted,
    StimulusRequest,
)
from services.api.services.operator_action_service import (
    OperatorActionService,
    SessionAlreadyEndedError,
    SessionNotFoundError,
    StimulusPublishError,
)
from services.api.services.operator_event_service import OperatorEventService
from services.api.services.operator_read_service import (
    OperatorReadService,
    _default_redis_factory,
)

router = APIRouter(prefix="/operator", tags=["operator"])
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dependency providers — small factories so tests can override easily.
# ----------------------------------------------------------------------


def get_read_service() -> OperatorReadService:
    return OperatorReadService(redis_factory=_default_redis_factory)


def get_action_service() -> OperatorActionService:
    return OperatorActionService()


class CloudOperatorService(Protocol):
    def get_auth_status(self) -> CloudAuthStatus: ...

    def sign_in(self) -> CloudSignInResult: ...

    def get_outbox_summary(self) -> CloudOutboxSummary: ...

    def get_latest_experiment_refresh(self) -> ExperimentBundleRefreshResult | None: ...

    def preview_experiment_bundle_refresh(self) -> ExperimentBundleRefreshPreview: ...

    def refresh_experiment_bundle(
        self,
        request: ExperimentBundleRefreshRequest,
    ) -> ExperimentBundleRefreshResult: ...


class UnavailableCloudOperatorService:
    def get_auth_status(self) -> CloudAuthStatus:
        return CloudAuthStatus(
            state=CloudAuthState.REFRESH_TOKEN_UNAVAILABLE,
            checked_at_utc=datetime.now(UTC),
            message="Desktop cloud service is not configured.",
        )

    def sign_in(self) -> CloudSignInResult:
        return CloudSignInResult(
            status=CloudActionStatus.FAILED,
            auth_state=CloudAuthState.REFRESH_TOKEN_UNAVAILABLE,
            completed_at_utc=datetime.now(UTC),
            message="Desktop cloud service is not configured.",
            error_code=CloudOperatorErrorCode.CLOUD_UNAVAILABLE,
            retryable=True,
        )

    def get_outbox_summary(self) -> CloudOutboxSummary:
        return CloudOutboxSummary(generated_at_utc=datetime.now(UTC))

    def get_latest_experiment_refresh(self) -> ExperimentBundleRefreshResult | None:
        return None

    def preview_experiment_bundle_refresh(self) -> ExperimentBundleRefreshPreview:
        return ExperimentBundleRefreshPreview(
            status=CloudActionStatus.FAILED,
            checked_at_utc=datetime.now(UTC),
            message="Desktop cloud service is not configured.",
            error_code=CloudOperatorErrorCode.CLOUD_UNAVAILABLE,
            retryable=True,
        )

    def refresh_experiment_bundle(
        self,
        _request: ExperimentBundleRefreshRequest,
    ) -> ExperimentBundleRefreshResult:
        return ExperimentBundleRefreshResult(
            status=CloudExperimentRefreshStatus.FAILED,
            completed_at_utc=datetime.now(UTC),
            message="Desktop cloud service is not configured.",
            error_code=CloudOperatorErrorCode.CLOUD_UNAVAILABLE,
            retryable=True,
        )


def get_cloud_service() -> CloudOperatorService:
    return UnavailableCloudOperatorService()


def get_event_service() -> OperatorEventService:
    return OperatorEventService(read_service=get_read_service())


_RawEventDep = Depends(get_event_service)


def get_supported_event_service(
    service: OperatorEventService = _RawEventDep,
) -> OperatorEventService:
    if not service.has_event_stream_support():
        raise HTTPException(status_code=503, detail="Operator event stream is not available")
    return service


# Module-level singletons for FastAPI defaults — extracted to satisfy
# B008 (no function calls in argument defaults). Behavior is identical:
# FastAPI resolves `Depends(...)` at request time regardless of where
# the sentinel is declared.
_ReadDep = Depends(get_read_service)
_ActionDep = Depends(get_action_service)
_CloudDep = Depends(get_cloud_service)
_EventDep = Depends(get_event_service)
_SupportedEventDep = Depends(get_supported_event_service)
_LimitSessionsQuery = Query(50, ge=1, le=500)
_LimitEncountersQuery = Query(100, ge=1, le=1000)
_BeforeUtcQuery = Query(None)
_LimitAlertsQuery = Query(50, ge=1, le=500)
_SinceUtcQuery = Query(None)
_LastEventIdHeader = Header(alias="Last-Event-ID")


# ----------------------------------------------------------------------
# Read endpoints
# ----------------------------------------------------------------------


@router.get("/state/bootstrap", response_model=OperatorStateBootstrap)
async def get_operator_state_bootstrap(
    service: OperatorEventService = _EventDep,
) -> OperatorStateBootstrap:
    try:
        return await service.build_bootstrap()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator state bootstrap failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/state/events", response_class=EventSourceResponse)
async def stream_operator_state_events(
    request: Request,
    last_event_id: Annotated[str | None, _LastEventIdHeader] = None,
    service: OperatorEventService = _SupportedEventDep,
) -> AsyncIterable[ServerSentEvent]:
    async for event in service.stream_events(request, last_event_id=last_event_id):
        yield event


@router.get("/overview", response_model=OverviewSnapshot)
async def get_overview(
    service: OperatorReadService = _ReadDep,
) -> OverviewSnapshot:
    """§4.E.1 — Six-card operator Overview."""
    try:
        return service.get_overview()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator overview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    limit: int = _LimitSessionsQuery,
    service: OperatorReadService = _ReadDep,
) -> list[SessionSummary]:
    try:
        return service.list_sessions(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator list_sessions failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(
    session_id: UUID,
    service: OperatorReadService = _ReadDep,
) -> SessionSummary:
    try:
        summary = service.get_session(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator get_session failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    if summary is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return summary


@router.get("/sessions/{session_id}/encounters", response_model=list[EncounterSummary])
async def list_session_encounters(
    session_id: UUID,
    limit: int = _LimitEncountersQuery,
    before_utc: datetime | None = _BeforeUtcQuery,
    service: OperatorReadService = _ReadDep,
) -> list[EncounterSummary]:
    """§7B + §7D — per-segment encounter rows with reward explanation.

    EncounterSummary also carries the optional observational acoustic summary.
    """
    try:
        return service.list_encounters(session_id, limit=limit, before_utc=before_utc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator list_encounters failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/experiments", response_model=list[ExperimentSummary])
async def list_experiments(
    service: OperatorReadService = _ReadDep,
) -> list[ExperimentSummary]:
    try:
        return service.list_experiments()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator list_experiments failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment_detail(
    experiment_id: str,
    service: OperatorReadService = _ReadDep,
) -> ExperimentDetail:
    try:
        detail = service.get_experiment_detail(experiment_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator get_experiment failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    if detail is None:
        raise HTTPException(status_code=404, detail=f"experiment {experiment_id!r} not found")
    return detail


@router.get("/sessions/{session_id}/physiology", response_model=SessionPhysiologySnapshot)
async def get_session_physiology(
    session_id: UUID,
    service: OperatorReadService = _ReadDep,
) -> SessionPhysiologySnapshot:
    """§4.E.2 + §7C — per-role physiology + co-modulation (null-valid)."""
    try:
        snap = service.get_session_physiology(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator get_physiology failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    if snap is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return snap


@router.get("/health", response_model=HealthSnapshot)
async def get_health(
    service: OperatorReadService = _ReadDep,
) -> HealthSnapshot:
    """§12 — subsystem rollup with degraded/recovering/error distinction."""
    try:
        result: Any = service.get_health()
        if inspect.isawaitable(result):
            return cast(HealthSnapshot, await result)
        return cast(HealthSnapshot, result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator get_health failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/alerts", response_model=list[AlertEvent])
async def list_alerts(
    limit: int = _LimitAlertsQuery,
    since_utc: datetime | None = _SinceUtcQuery,
    service: OperatorReadService = _ReadDep,
) -> list[AlertEvent]:
    try:
        return service.list_alerts(limit=limit, since_utc=since_utc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator list_alerts failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/cloud/auth/status", response_model=CloudAuthStatus)
async def get_cloud_auth_status(
    service: CloudOperatorService = _CloudDep,
) -> CloudAuthStatus:
    try:
        return await run_in_threadpool(service.get_auth_status)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator cloud auth status failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/cloud/auth/sign-in", response_model=CloudSignInResult)
async def sign_in_to_cloud(
    service: CloudOperatorService = _CloudDep,
) -> CloudSignInResult:
    try:
        return await run_in_threadpool(service.sign_in)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator cloud sign-in failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/cloud/outbox", response_model=CloudOutboxSummary)
async def get_cloud_outbox_summary(
    service: CloudOperatorService = _CloudDep,
) -> CloudOutboxSummary:
    try:
        return await run_in_threadpool(service.get_outbox_summary)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator cloud outbox summary failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get(
    "/cloud/experiments/refresh/latest",
    response_model=ExperimentBundleRefreshResult | None,
)
async def get_latest_cloud_experiment_refresh(
    service: CloudOperatorService = _CloudDep,
) -> ExperimentBundleRefreshResult | None:
    try:
        return await run_in_threadpool(service.get_latest_experiment_refresh)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator cloud experiment refresh latest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post(
    "/cloud/experiments/refresh/preview",
    response_model=ExperimentBundleRefreshPreview,
)
async def preview_cloud_experiment_bundle_refresh(
    service: CloudOperatorService = _CloudDep,
) -> ExperimentBundleRefreshPreview:
    try:
        return await run_in_threadpool(service.preview_experiment_bundle_refresh)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator cloud experiment refresh preview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/cloud/experiments/refresh", response_model=ExperimentBundleRefreshResult)
async def refresh_cloud_experiment_bundle(
    request: ExperimentBundleRefreshRequest,
    service: CloudOperatorService = _CloudDep,
) -> ExperimentBundleRefreshResult:
    try:
        return await run_in_threadpool(service.refresh_experiment_bundle, request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator cloud experiment refresh failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# ----------------------------------------------------------------------
# Write endpoint
# ----------------------------------------------------------------------


@router.post("/sessions/{session_id}/stimulus", response_model=StimulusAccepted)
async def submit_stimulus(
    session_id: UUID,
    request: StimulusRequest,
    service: OperatorActionService = _ActionDep,
) -> StimulusAccepted:
    """§4.C — operator-issued stimulus submission.

    Idempotent on `client_action_id` (Redis SETNX). Never assigns
    authoritative `stimulus_time` — that comes from the orchestrator
    via drift-corrected clock on receipt.
    """
    try:
        return service.submit_stimulus(session_id, request)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found") from exc
    except SessionAlreadyEndedError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"session {session_id} has already ended; stimulus not accepted",
        ) from exc
    except StimulusPublishError as exc:
        raise HTTPException(
            status_code=503,
            detail="broker unavailable — cannot deliver stimulus trigger",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator submit_stimulus failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
