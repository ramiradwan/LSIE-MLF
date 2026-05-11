"""Cloud OAuth token routes.

This cloud control-plane surface is consumed by typed desktop cloud helpers,
not imported by desktop runtime processes or the loopback API shell.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from packages.schemas.cloud import OAuthTokenRequest, OAuthTokenResponse
from services.cloud_api.services.auth_service import (
    AuthConfigurationError,
    InvalidClientError,
    InvalidTokenError,
    OAuthTokenService,
)

router = APIRouter()


def get_oauth_token_service() -> OAuthTokenService:
    return OAuthTokenService()


_OAuthTokenServiceDep = Depends(get_oauth_token_service)


@router.post("/auth/oauth/token", response_model=OAuthTokenResponse)
async def exchange_oauth_token(
    request: OAuthTokenRequest,
    service: OAuthTokenService = _OAuthTokenServiceDep,
) -> OAuthTokenResponse:
    try:
        return service.exchange_token(request)
    except (InvalidClientError, InvalidTokenError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
