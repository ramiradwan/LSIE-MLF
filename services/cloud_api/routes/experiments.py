"""Cloud experiment bundle routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from packages.schemas.cloud import ExperimentBundle
from services.cloud_api.services.auth_service import (
    AuthenticatedClient,
    require_authenticated_client,
)
from services.cloud_api.services.bundle_service import ExperimentBundleService

router = APIRouter()
logger = logging.getLogger(__name__)


def get_bundle_service() -> ExperimentBundleService:
    return ExperimentBundleService()


_BundleServiceDep = Depends(get_bundle_service)
_AuthenticatedClientDep = Depends(require_authenticated_client)


@router.get("/experiments/bundle", response_model=ExperimentBundle)
async def get_experiment_bundle(
    authenticated_client: AuthenticatedClient = _AuthenticatedClientDep,
    service: ExperimentBundleService = _BundleServiceDep,
) -> ExperimentBundle:
    del authenticated_client
    try:
        return service.build_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("cloud experiment bundle read failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
