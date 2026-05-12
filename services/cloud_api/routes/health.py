"""Bounded cloud API health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from services.cloud_api.db.connection import check_readiness

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(response: Response) -> dict[str, str | dict[str, str]]:
    ready = await check_readiness()
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "checks": {"database": "unavailable"}}
    return {"status": "ok", "checks": {"database": "ok"}}
