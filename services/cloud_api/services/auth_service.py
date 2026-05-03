"""Cloud bearer token issuance and validation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Annotated, Final, cast

from fastapi import Depends, Header, HTTPException

from packages.schemas.cloud import OAuthTokenRequest, OAuthTokenResponse

_ACCESS_TOKEN_TTL_SECONDS: Final[int] = 3600
_AUTHORIZATION_CODE_TTL_SECONDS: Final[int] = 300
_REFRESH_TOKEN_TTL_SECONDS: Final[int] = 60 * 60 * 24 * 30
_DEFAULT_ALLOWED_CLIENT_IDS: Final[frozenset[str]] = frozenset({"desktop-app"})
_DEFAULT_SCOPE: Final[str] = "telemetry experiments sessions"
_ISSUER: Final[str] = "lsie-cloud-api"
_TOKEN_PREFIX: Final[str] = "lsie_ws5"
_WWW_AUTHENTICATE_BEARER: Final[dict[str, str]] = {"WWW-Authenticate": "Bearer"}


class AuthConfigurationError(RuntimeError):
    pass


class InvalidClientError(RuntimeError):
    pass


class InvalidTokenError(RuntimeError):
    pass


class UnsupportedGrantError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AuthenticatedClient:
    client_id: str
    scope: str
    expires_at_epoch: int


@dataclass(frozen=True, slots=True)
class TokenClaims:
    client_id: str
    token_type: str
    scope: str
    issued_at_epoch: int
    expires_at_epoch: int


class CloudTokenCodec:
    def __init__(self, *, secret: str, allowed_client_ids: frozenset[str]) -> None:
        self._secret = secret
        self._allowed_client_ids = allowed_client_ids

    @classmethod
    def from_env(cls) -> CloudTokenCodec:
        secret = os.environ.get("LSIE_CLOUD_TOKEN_SIGNING_SECRET", "").strip()
        if not secret:
            raise AuthConfigurationError("LSIE_CLOUD_TOKEN_SIGNING_SECRET is not configured")
        allowed_client_ids = _allowed_client_ids_from_env()
        if not allowed_client_ids:
            raise AuthConfigurationError("LSIE_CLOUD_ALLOWED_CLIENT_IDS is empty")
        return cls(secret=secret, allowed_client_ids=allowed_client_ids)

    def issue_token_response(self, client_id: str) -> OAuthTokenResponse:
        self._ensure_allowed_client(client_id)
        now_epoch = int(time.time())
        return OAuthTokenResponse(
            access_token=self._issue_token(
                client_id=client_id,
                token_type="access",
                now_epoch=now_epoch,
                ttl_seconds=_ACCESS_TOKEN_TTL_SECONDS,
            ),
            expires_in=_ACCESS_TOKEN_TTL_SECONDS,
            refresh_token=self._issue_token(
                client_id=client_id,
                token_type="refresh",
                now_epoch=now_epoch,
                ttl_seconds=_REFRESH_TOKEN_TTL_SECONDS,
            ),
            scope=_DEFAULT_SCOPE,
        )

    def issue_authorization_code(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
    ) -> str:
        self._ensure_allowed_client(client_id)
        return self._issue_token(
            client_id=client_id,
            token_type="authorization_code",
            now_epoch=int(time.time()),
            ttl_seconds=_AUTHORIZATION_CODE_TTL_SECONDS,
            extra_claims={
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
            },
        )

    def authenticate_access_token(self, token: str) -> AuthenticatedClient:
        claims = self._authenticate_token(token, expected_token_type="access")
        return AuthenticatedClient(
            client_id=claims.client_id,
            scope=claims.scope,
            expires_at_epoch=claims.expires_at_epoch,
        )

    def authenticate_refresh_token(self, token: str) -> AuthenticatedClient:
        claims = self._authenticate_token(token, expected_token_type="refresh")
        return AuthenticatedClient(
            client_id=claims.client_id,
            scope=claims.scope,
            expires_at_epoch=claims.expires_at_epoch,
        )

    def authenticate_authorization_code(
        self,
        code: str,
        *,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> AuthenticatedClient:
        payload = self._authenticate_payload(code, expected_token_type="authorization_code")
        code_client_id = _require_str_claim(payload, "sub")
        if code_client_id != client_id:
            raise InvalidClientError(
                "authorization code client_id does not match request client_id"
            )
        if _require_str_claim(payload, "redirect_uri") != redirect_uri:
            raise InvalidTokenError("authorization code redirect_uri does not match request")
        try:
            expected_code_challenge = _pkce_code_challenge(code_verifier)
        except UnicodeEncodeError as exc:
            raise InvalidTokenError("authorization code PKCE verification failed") from exc
        if not hmac.compare_digest(
            _require_str_claim(payload, "code_challenge"), expected_code_challenge
        ):
            raise InvalidTokenError("authorization code PKCE verification failed")
        return AuthenticatedClient(
            client_id=code_client_id,
            scope=_require_str_claim(payload, "scope"),
            expires_at_epoch=_require_int_claim(payload, "exp"),
        )

    def _issue_token(
        self,
        *,
        client_id: str,
        token_type: str,
        now_epoch: int,
        ttl_seconds: int,
        extra_claims: dict[str, object] | None = None,
    ) -> str:
        payload = {
            "exp": now_epoch + ttl_seconds,
            "iat": now_epoch,
            "iss": _ISSUER,
            "scope": _DEFAULT_SCOPE,
            "sub": client_id,
            "typ": token_type,
            "ver": 1,
        }
        if extra_claims is not None:
            payload.update(extra_claims)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload_b64 = _urlsafe_b64encode(payload_json)
        signature_b64 = _urlsafe_b64encode(self._sign(payload_b64.encode("utf-8")))
        return f"{_TOKEN_PREFIX}.{payload_b64}.{signature_b64}"

    def _authenticate_token(self, token: str, *, expected_token_type: str) -> TokenClaims:
        payload = self._authenticate_payload(token, expected_token_type=expected_token_type)
        return TokenClaims(
            client_id=_require_str_claim(payload, "sub"),
            token_type=_require_str_claim(payload, "typ"),
            scope=_require_str_claim(payload, "scope"),
            issued_at_epoch=_require_int_claim(payload, "iat"),
            expires_at_epoch=_require_int_claim(payload, "exp"),
        )

    def _authenticate_payload(
        self,
        token: str,
        *,
        expected_token_type: str,
    ) -> dict[str, object]:
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
            raise InvalidTokenError("invalid bearer token")
        payload_b64 = parts[1]
        signature_b64 = parts[2]
        expected_signature = _urlsafe_b64encode(self._sign(payload_b64.encode("utf-8")))
        if not hmac.compare_digest(expected_signature, signature_b64):
            raise InvalidTokenError("invalid bearer token")
        try:
            payload_raw = _urlsafe_b64decode(payload_b64)
            payload_obj = json.loads(payload_raw.decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidTokenError("invalid bearer token") from exc
        if not isinstance(payload_obj, dict):
            raise InvalidTokenError("invalid bearer token")
        payload = cast(dict[str, object], payload_obj)
        token_type = _require_str_claim(payload, "typ")
        if token_type != expected_token_type:
            raise InvalidTokenError("invalid bearer token")
        issuer = _require_str_claim(payload, "iss")
        if issuer != _ISSUER:
            raise InvalidTokenError("invalid bearer token")
        client_id = _require_str_claim(payload, "sub")
        self._ensure_allowed_client(client_id)
        expires_at_epoch = _require_int_claim(payload, "exp")
        if expires_at_epoch <= int(time.time()):
            raise InvalidTokenError("bearer token expired")
        return payload

    def _ensure_allowed_client(self, client_id: str) -> None:
        if client_id not in self._allowed_client_ids:
            raise InvalidClientError(f"client_id {client_id!r} is not allowed")

    def _sign(self, payload: bytes) -> bytes:
        return hmac.new(self._secret.encode("utf-8"), payload, hashlib.sha256).digest()


class OAuthTokenService:
    def __init__(self, codec: CloudTokenCodec | None = None) -> None:
        self._codec = codec

    def exchange_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        codec = self._codec or CloudTokenCodec.from_env()
        codec._ensure_allowed_client(request.client_id)
        if request.grant_type == "authorization_code":
            authenticated_client = codec.authenticate_authorization_code(
                cast(str, request.code),
                client_id=request.client_id,
                redirect_uri=cast(str, request.redirect_uri),
                code_verifier=cast(str, request.code_verifier),
            )
            return codec.issue_token_response(authenticated_client.client_id)
        if request.refresh_token is None:
            raise InvalidTokenError("refresh token is required")
        authenticated_client = codec.authenticate_refresh_token(request.refresh_token)
        if authenticated_client.client_id != request.client_id:
            raise InvalidClientError("refresh token client_id does not match request client_id")
        return codec.issue_token_response(request.client_id)


def get_token_codec() -> CloudTokenCodec:
    try:
        return CloudTokenCodec.from_env()
    except AuthConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


_AuthorizationHeader = Annotated[str | None, Header()]
_TokenCodecDep = Annotated[CloudTokenCodec, Depends(get_token_codec)]


def require_authenticated_client(
    codec: _TokenCodecDep,
    authorization: _AuthorizationHeader = None,
) -> AuthenticatedClient:
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers=_WWW_AUTHENTICATE_BEARER,
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers=_WWW_AUTHENTICATE_BEARER,
        )
    try:
        return codec.authenticate_access_token(token.strip())
    except (InvalidClientError, InvalidTokenError) as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers=_WWW_AUTHENTICATE_BEARER,
        ) from exc


def _allowed_client_ids_from_env() -> frozenset[str]:
    raw = os.environ.get("LSIE_CLOUD_ALLOWED_CLIENT_IDS", "")
    if not raw.strip():
        return _DEFAULT_ALLOWED_CLIENT_IDS
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _require_int_claim(payload: object, name: str) -> int:
    if not isinstance(payload, dict):
        raise InvalidTokenError("invalid bearer token")
    value = payload.get(name)
    if not isinstance(value, int):
        raise InvalidTokenError("invalid bearer token")
    return value


def _require_str_claim(payload: object, name: str) -> str:
    if not isinstance(payload, dict):
        raise InvalidTokenError("invalid bearer token")
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise InvalidTokenError("invalid bearer token")
    return value


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return _urlsafe_b64encode(digest)


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")
