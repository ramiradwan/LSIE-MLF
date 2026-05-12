"""Cloud OAuth token routes.

This cloud control-plane surface is consumed by typed desktop cloud helpers,
not imported by desktop runtime processes or the loopback API shell.
"""

from __future__ import annotations

from typing import Literal, cast
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from packages.schemas.cloud import OAuthAuthorizationRequest, OAuthTokenRequest, OAuthTokenResponse
from services.cloud_api.services.auth_service import (
    AuthConfigurationError,
    CloudTokenCodec,
    InvalidClientError,
    InvalidTokenError,
    OAuthTokenService,
)

router = APIRouter()


def get_oauth_token_service() -> OAuthTokenService:
    return OAuthTokenService()


def get_authorization_token_codec() -> CloudTokenCodec:
    try:
        return CloudTokenCodec.from_env()
    except AuthConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="cloud authorization is not configured",
        ) from None


_OAuthTokenServiceDep = Depends(get_oauth_token_service)
_AuthorizationTokenCodecDep = Depends(get_authorization_token_codec)


@router.get("/auth/oauth/authorize")
async def authorize_oauth_client(
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    code_challenge: str = Query(...),
    code_challenge_method: str = Query(...),
    state: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    codec: CloudTokenCodec = _AuthorizationTokenCodecDep,
) -> RedirectResponse:
    try:
        request = OAuthAuthorizationRequest(
            response_type=cast(Literal["code"], response_type),
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=cast(Literal["S256"], code_challenge_method),
            state=state,
            scope=scope,
        )
        code = codec.issue_authorization_code(
            client_id=request.client_id,
            redirect_uri=request.redirect_uri,
            code_challenge=request.code_challenge,
        )
    except ValidationError:
        raise HTTPException(status_code=400, detail="invalid authorization request") from None
    except InvalidClientError:
        raise HTTPException(status_code=401, detail="client_id is not allowed") from None
    query: dict[str, str] = {"code": code}
    if request.state is not None:
        query["state"] = request.state
    return RedirectResponse(f"{request.redirect_uri}?{urlencode(query)}", status_code=302)


@router.post("/auth/oauth/token", response_model=OAuthTokenResponse)
async def exchange_oauth_token(
    request: OAuthTokenRequest,
    service: OAuthTokenService = _OAuthTokenServiceDep,
) -> OAuthTokenResponse:
    try:
        return service.exchange_token(request)
    except InvalidClientError:
        raise HTTPException(status_code=401, detail="client_id is not allowed") from None
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except AuthConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="cloud authorization is not configured",
        ) from None
