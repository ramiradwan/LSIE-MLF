"""
Operator Console aggregate routes — `/api/v1/operator/*` (Phase 2).

These endpoints fan in the data the PySide6 Operator Console needs and
fan out a single write surface for stimulus submission. Response models
are Phase-1 DTOs from `packages.schemas.operator_console`; the heavy
lifting lives in `OperatorReadService`/`OperatorActionService` so the
route layer stays thin.

Design constraints:
  - Every response is a Pydantic DTO — no raw dicts cross the wire.
  - Read handlers never call the DB directly; they go through
    `OperatorReadService` which owns connection lifecycle and DTO
    assembly.
  - The single write path (`POST /sessions/{id}/stimulus`) flows
    through `OperatorActionService`, which is the only place allowed
    to touch the Redis trigger channel for operator intent.
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
  SPEC-AMEND-008 — PySide6 Operator Console replaces Streamlit.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from packages.schemas.operator_console import (
    AlertEvent,
    EncounterSummary,
    ExperimentDetail,
    HealthSnapshot,
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
from services.api.services.operator_read_service import OperatorReadService

router = APIRouter(prefix="/operator", tags=["operator"])
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dependency providers — small factories so tests can override easily.
# ----------------------------------------------------------------------


def get_read_service() -> OperatorReadService:
    return OperatorReadService()


def get_action_service() -> OperatorActionService:
    return OperatorActionService()


# ----------------------------------------------------------------------
# Read endpoints
# ----------------------------------------------------------------------


@router.get("/overview", response_model=OverviewSnapshot)
async def get_overview(
    service: OperatorReadService = Depends(get_read_service),
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
    limit: int = Query(50, ge=1, le=500),
    service: OperatorReadService = Depends(get_read_service),
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
    service: OperatorReadService = Depends(get_read_service),
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


@router.get(
    "/sessions/{session_id}/encounters", response_model=list[EncounterSummary]
)
async def list_session_encounters(
    session_id: UUID,
    limit: int = Query(100, ge=1, le=1000),
    before_utc: datetime | None = Query(None),
    service: OperatorReadService = Depends(get_read_service),
) -> list[EncounterSummary]:
    """§7B — per-segment encounter rows with reward explanation."""
    try:
        return service.list_encounters(
            session_id, limit=limit, before_utc=before_utc
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator list_encounters failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment_detail(
    experiment_id: str,
    service: OperatorReadService = Depends(get_read_service),
) -> ExperimentDetail:
    try:
        detail = service.get_experiment_detail(experiment_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator get_experiment failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    if detail is None:
        raise HTTPException(
            status_code=404, detail=f"experiment {experiment_id!r} not found"
        )
    return detail


@router.get(
    "/sessions/{session_id}/physiology", response_model=SessionPhysiologySnapshot
)
async def get_session_physiology(
    session_id: UUID,
    service: OperatorReadService = Depends(get_read_service),
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
    service: OperatorReadService = Depends(get_read_service),
) -> HealthSnapshot:
    """§12 — subsystem rollup with degraded/recovering/error distinction."""
    try:
        return service.get_health()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator get_health failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/alerts", response_model=list[AlertEvent])
async def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    since_utc: datetime | None = Query(None),
    service: OperatorReadService = Depends(get_read_service),
) -> list[AlertEvent]:
    try:
        return service.list_alerts(limit=limit, since_utc=since_utc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("operator list_alerts failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# ----------------------------------------------------------------------
# Write endpoint
# ----------------------------------------------------------------------


@router.post("/sessions/{session_id}/stimulus", response_model=StimulusAccepted)
async def submit_stimulus(
    session_id: UUID,
    request: StimulusRequest,
    service: OperatorActionService = Depends(get_action_service),
) -> StimulusAccepted:
    """§4.C — operator-issued stimulus submission.

    Idempotent on `client_action_id` (Redis SETNX). Never assigns
    authoritative `stimulus_time` — that comes from the orchestrator
    via drift-corrected clock on receipt.
    """
    try:
        return service.submit_stimulus(session_id, request)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"session {session_id} not found"
        ) from exc
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
