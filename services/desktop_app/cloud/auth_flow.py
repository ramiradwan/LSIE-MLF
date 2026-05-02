from __future__ import annotations

import base64
import hashlib
import secrets as token_secrets
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from packages.schemas.cloud import OAuthTokenRequest, OAuthTokenResponse
from services.desktop_app.privacy.secrets import SECRET_KEY_CLOUD_REFRESH_TOKEN, set_secret

BrowserOpener = Callable[[str], bool]
TokenExchange = Callable[[OAuthTokenRequest], OAuthTokenResponse]
StateGenerator = Callable[[], str]


@dataclass(frozen=True)
class PkceChallenge:
    code_verifier: str
    code_challenge: str


@dataclass(frozen=True)
class AuthorizationRequest:
    authorization_url: str
    code_verifier: str


@dataclass(frozen=True)
class AuthFlowConfig:
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    redirect_uri: str = "http://127.0.0.1:0/oauth/callback"
    scope: str = "telemetry experiments sessions"
    callback_timeout_s: float = 120.0

    def with_redirect_uri(self, redirect_uri: str) -> AuthFlowConfig:
        return AuthFlowConfig(
            authorization_endpoint=self.authorization_endpoint,
            token_endpoint=self.token_endpoint,
            client_id=self.client_id,
            redirect_uri=redirect_uri,
            scope=self.scope,
            callback_timeout_s=self.callback_timeout_s,
        )


class OAuthTokenExchangeError(RuntimeError):
    pass


class OAuthCallbackError(RuntimeError):
    pass


@dataclass(frozen=True)
class OAuthCallbackResult:
    code: str
    state: str


def generate_state(byte_count: int = 32) -> str:
    return _base64_urlsafe(token_secrets.token_bytes(byte_count))


def generate_code_verifier(byte_count: int = 64) -> str:
    if byte_count < 32:
        raise ValueError("byte_count must be at least 32")
    return _base64_urlsafe(token_secrets.token_bytes(byte_count))


def build_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return _base64_urlsafe(digest)


def build_pkce_challenge(byte_count: int = 64) -> PkceChallenge:
    verifier = generate_code_verifier(byte_count)
    return PkceChallenge(code_verifier=verifier, code_challenge=build_code_challenge(verifier))


def build_authorization_url(config: AuthFlowConfig, challenge: PkceChallenge, *, state: str) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": config.scope,
            "state": state,
            "code_challenge": challenge.code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{config.authorization_endpoint}?{query}"


class DesktopAuthFlow:
    def __init__(
        self,
        config: AuthFlowConfig,
        *,
        browser_opener: BrowserOpener | None = None,
        token_exchange: TokenExchange | None = None,
        state_generator: StateGenerator | None = None,
    ) -> None:
        self._config = config
        self._browser_opener = browser_opener or webbrowser.open
        self._token_exchange = token_exchange or self._exchange_with_token_endpoint
        self._state_generator = state_generator or generate_state

    def open_authorization(self, *, state: str) -> AuthorizationRequest:
        challenge = build_pkce_challenge()
        authorization_url = build_authorization_url(self._config, challenge, state=state)
        self._browser_opener(authorization_url)
        return AuthorizationRequest(
            authorization_url=authorization_url,
            code_verifier=challenge.code_verifier,
        )

    def run_loopback_authorization(self) -> OAuthTokenResponse:
        state = self._state_generator()
        challenge = build_pkce_challenge()
        with _LoopbackCallbackServer(self._config.callback_timeout_s) as callback_server:
            callback_config = self._config.with_redirect_uri(callback_server.redirect_uri)
            authorization_url = build_authorization_url(callback_config, challenge, state=state)
            self._browser_opener(authorization_url)
            callback = callback_server.wait_for_callback()
        if callback.state != state:
            raise OAuthCallbackError("OAuth callback state mismatch")
        request = OAuthTokenRequest(
            grant_type="authorization_code",
            client_id=self._config.client_id,
            code=callback.code,
            code_verifier=challenge.code_verifier,
            redirect_uri=callback_config.redirect_uri,
        )
        return self._exchange_and_store_refresh_token(request)

    def exchange_authorization_code(self, *, code: str, code_verifier: str) -> OAuthTokenResponse:
        request = OAuthTokenRequest(
            grant_type="authorization_code",
            client_id=self._config.client_id,
            code=code,
            code_verifier=code_verifier,
            redirect_uri=self._config.redirect_uri,
        )
        return self._exchange_and_store_refresh_token(request)

    def refresh_access_token(self, *, refresh_token: str) -> OAuthTokenResponse:
        request = OAuthTokenRequest(
            grant_type="refresh_token",
            client_id=self._config.client_id,
            refresh_token=refresh_token,
        )
        return self._exchange_and_store_refresh_token(request)

    def _exchange_and_store_refresh_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        response = self._token_exchange(request)
        if response.refresh_token is not None:
            set_secret(SECRET_KEY_CLOUD_REFRESH_TOKEN, response.refresh_token)
        return response

    def _exchange_with_token_endpoint(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        try:
            response = httpx.post(
                self._config.token_endpoint,
                json=request.model_dump(mode="json", exclude_none=True),
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OAuthTokenExchangeError(str(exc)) from exc
        data: object = response.json()
        return OAuthTokenResponse.model_validate(data)


class _LoopbackCallbackHandler(BaseHTTPRequestHandler):
    server: _LoopbackHttpServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        code = query.get("code", [""])[0]
        state = query.get("state", [""])[0]
        error = query.get("error", [""])[0]
        if error:
            self.server.callback_error = error
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authorization failed. You can close this window.")
            return
        if not code or not state:
            self.server.callback_error = "OAuth callback missing code or state"
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authorization callback was incomplete.")
            return
        self.server.callback_result = OAuthCallbackResult(code=code, state=state)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Authorization complete. You can close this window.")

    def log_message(self, format: str, *args: object) -> None:
        return


class _LoopbackHttpServer(HTTPServer):
    callback_result: OAuthCallbackResult | None = None
    callback_error: str | None = None


class _LoopbackCallbackServer:
    def __init__(self, timeout_s: float) -> None:
        self._server = _LoopbackHttpServer(("127.0.0.1", 0), _LoopbackCallbackHandler)
        self._server.timeout = timeout_s

    @property
    def redirect_uri(self) -> str:
        host = str(self._server.server_address[0])
        port = int(self._server.server_address[1])
        return f"http://{host}:{port}/oauth/callback"

    def __enter__(self) -> _LoopbackCallbackServer:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._server.server_close()

    def wait_for_callback(self) -> OAuthCallbackResult:
        self._server.handle_request()
        if self._server.callback_error is not None:
            raise OAuthCallbackError(self._server.callback_error)
        if self._server.callback_result is None:
            raise OAuthCallbackError("OAuth callback timed out")
        return self._server.callback_result


def _base64_urlsafe(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


__all__ = [
    "AuthFlowConfig",
    "AuthorizationRequest",
    "BrowserOpener",
    "DesktopAuthFlow",
    "OAuthCallbackError",
    "OAuthCallbackResult",
    "OAuthTokenExchangeError",
    "PkceChallenge",
    "StateGenerator",
    "TokenExchange",
    "build_authorization_url",
    "build_code_challenge",
    "build_pkce_challenge",
    "generate_code_verifier",
    "generate_state",
]
