"""Minimal WS5 P1 OAuth token service boundary."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from packages.schemas.cloud import OAuthTokenRequest, OAuthTokenResponse

_ACCESS_TOKEN_TTL_SECONDS = 3600


class OAuthTokenService:
    def exchange_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        seed = f"{request.grant_type}:{request.client_id}:{request.code or request.refresh_token}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        refresh_seed = f"refresh:{request.client_id}:{datetime.now(UTC).date().isoformat()}"
        refresh_digest = hashlib.sha256(refresh_seed.encode("utf-8")).hexdigest()
        return OAuthTokenResponse(
            access_token=f"ws5p1_{digest}",
            expires_in=_ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=f"ws5p1_refresh_{refresh_digest}",
            scope="telemetry experiments sessions",
        )
