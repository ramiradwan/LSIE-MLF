"""Cloud session lifecycle routes."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from packages.schemas.cloud import (
    CloudSessionCreateRequest,
    CloudSessionCreateResponse,
    CloudSessionEndRequest,
    CloudSessionEndResponse,
)
from services.cloud_api.services.auth_service import (
    AuthenticatedClient,
    require_authenticated_client,
)
from services.cloud_api.services.session_service import (
    CloudSessionService,
    SessionNotFoundError,
    SessionOwnershipError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_session_service() -> CloudSessionService:
    return CloudSessionService()


_SessionServiceDep = Depends(get_session_service)
_AuthenticatedClientDep = Depends(require_authenticated_client)


@router.post("/sessions", response_model=CloudSessionCreateResponse, status_code=201)
async def create_session(
    request: CloudSessionCreateRequest,
    authenticated_client: AuthenticatedClient = _AuthenticatedClientDep,
    service: CloudSessionService = _SessionServiceDep,
) -> CloudSessionCreateResponse:
    try:
        return service.create_session(request, client_id=authenticated_client.client_id)
    except SessionOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("cloud session create failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/sessions/{session_id}/end", response_model=CloudSessionEndResponse)
async def end_session(
    session_id: UUID,
    request: CloudSessionEndRequest,
    authenticated_client: AuthenticatedClient = _AuthenticatedClientDep,
    service: CloudSessionService = _SessionServiceDep,
) -> CloudSessionEndResponse:
    try:
        return service.end_session(session_id, request, client_id=authenticated_client.client_id)
    except SessionOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("cloud session end failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
