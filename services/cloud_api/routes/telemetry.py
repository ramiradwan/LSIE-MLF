"""Cloud telemetry ingest routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from packages.schemas.cloud import (
    CloudIngestResponse,
    TelemetryPosteriorDeltaBatch,
    TelemetrySegmentBatch,
)
from services.cloud_api.services.auth_service import (
    AuthenticatedClient,
    require_authenticated_client,
)
from services.cloud_api.services.telemetry_service import (
    PosteriorDeltaAuthorizationError,
    TelemetryIngestService,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_telemetry_service() -> TelemetryIngestService:
    return TelemetryIngestService()


_TelemetryServiceDep = Depends(get_telemetry_service)
_AuthenticatedClientDep = Depends(require_authenticated_client)


@router.post("/telemetry/segments")
async def ingest_segments(
    request: TelemetrySegmentBatch,
    authenticated_client: AuthenticatedClient = _AuthenticatedClientDep,
    service: TelemetryIngestService = _TelemetryServiceDep,
) -> CloudIngestResponse:
    try:
        return service.ingest_segments(request, client_id=authenticated_client.client_id)
    except PosteriorDeltaAuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("cloud segment telemetry ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/telemetry/posterior_deltas")
async def ingest_posterior_deltas(
    request: TelemetryPosteriorDeltaBatch,
    authenticated_client: AuthenticatedClient = _AuthenticatedClientDep,
    service: TelemetryIngestService = _TelemetryServiceDep,
) -> CloudIngestResponse:
    try:
        return service.ingest_posterior_deltas(request, client_id=authenticated_client.client_id)
    except PosteriorDeltaAuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("cloud posterior delta ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
