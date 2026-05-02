"""Cloud OAuth token routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from packages.schemas.cloud import OAuthTokenRequest, OAuthTokenResponse
from services.cloud_api.services.auth_service import OAuthTokenService

router = APIRouter()


def get_oauth_token_service() -> OAuthTokenService:
    return OAuthTokenService()


_OAuthTokenServiceDep = Depends(get_oauth_token_service)


@router.post("/auth/oauth/token", response_model=OAuthTokenResponse)
async def exchange_oauth_token(
    request: OAuthTokenRequest,
    service: OAuthTokenService = _OAuthTokenServiceDep,
) -> OAuthTokenResponse:
    return service.exchange_token(request)
