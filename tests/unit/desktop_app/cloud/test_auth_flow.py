from __future__ import annotations

import threading

import httpx
import pytest

from packages.schemas.cloud import OAuthTokenRequest, OAuthTokenResponse
from services.desktop_app.cloud.auth_flow import (
    AuthFlowConfig,
    DesktopAuthFlow,
    build_code_challenge,
)
from services.desktop_app.privacy.secrets import SECRET_KEY_CLOUD_REFRESH_TOKEN


def _config() -> AuthFlowConfig:
    return AuthFlowConfig(
        authorization_endpoint="https://cloud.example.test/oauth/authorize",
        token_endpoint="https://cloud.example.test/oauth/token",
        client_id="desktop-a",
        redirect_uri="lsie://oauth/callback",
    )


def test_pkce_challenge_matches_s256_reference_vector() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

    assert build_code_challenge(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_open_authorization_uses_injected_browser_opener() -> None:
    opened_urls: list[str] = []
    flow = DesktopAuthFlow(_config(), browser_opener=lambda url: not opened_urls.append(url))

    request = flow.open_authorization(state="state-a")

    assert opened_urls == [request.authorization_url]
    assert "code_challenge_method=S256" in request.authorization_url
    assert "state=state-a" in request.authorization_url
    assert request.code_verifier


def test_loopback_authorization_validates_state_and_exchanges_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}
    opened_urls: list[str] = []
    seen_requests: list[OAuthTokenRequest] = []

    def exchange(request: OAuthTokenRequest) -> OAuthTokenResponse:
        seen_requests.append(request)
        return OAuthTokenResponse(
            access_token="access-a",
            expires_in=3600,
            refresh_token="refresh-a",
        )

    def open_callback(url: str) -> bool:
        opened_urls.append(url)
        redirect_uri = str(httpx.URL(url).params["redirect_uri"])
        state = str(httpx.URL(url).params["state"])

        def send_callback() -> None:
            response = httpx.get(redirect_uri, params={"code": "code-a", "state": state})
            response.raise_for_status()

        threading.Thread(target=send_callback).start()
        return True

    monkeypatch.setattr(
        "services.desktop_app.cloud.auth_flow.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    flow = DesktopAuthFlow(
        _config(),
        browser_opener=open_callback,
        token_exchange=exchange,
        state_generator=lambda: "state-a",
    )

    response = flow.run_loopback_authorization()

    assert response.access_token == "access-a"
    assert opened_urls
    assert seen_requests == [
        OAuthTokenRequest(
            grant_type="authorization_code",
            client_id="desktop-a",
            code="code-a",
            code_verifier=seen_requests[0].code_verifier,
            redirect_uri=seen_requests[0].redirect_uri,
        )
    ]
    assert seen_requests[0].code_verifier
    assert seen_requests[0].redirect_uri.startswith("http://127.0.0.1:")
    assert stored == {SECRET_KEY_CLOUD_REFRESH_TOKEN: "refresh-a"}


def test_authorization_code_exchange_uses_schema_and_stores_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}
    seen_requests: list[OAuthTokenRequest] = []

    def exchange(request: OAuthTokenRequest) -> OAuthTokenResponse:
        seen_requests.append(request)
        return OAuthTokenResponse(
            access_token="access-a",
            expires_in=3600,
            refresh_token="refresh-a",
        )

    monkeypatch.setattr(
        "services.desktop_app.cloud.auth_flow.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    flow = DesktopAuthFlow(_config(), token_exchange=exchange)

    response = flow.exchange_authorization_code(code="code-a", code_verifier="verifier-a")

    assert response.access_token == "access-a"
    assert seen_requests == [
        OAuthTokenRequest(
            grant_type="authorization_code",
            client_id="desktop-a",
            code="code-a",
            code_verifier="verifier-a",
            redirect_uri="lsie://oauth/callback",
        )
    ]
    assert stored == {SECRET_KEY_CLOUD_REFRESH_TOKEN: "refresh-a"}


def test_refresh_exchange_does_not_overwrite_secret_when_response_omits_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}

    def exchange(request: OAuthTokenRequest) -> OAuthTokenResponse:
        assert request.grant_type == "refresh_token"
        assert request.refresh_token == "refresh-a"
        return OAuthTokenResponse(access_token="access-b", expires_in=3600)

    monkeypatch.setattr(
        "services.desktop_app.cloud.auth_flow.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    flow = DesktopAuthFlow(_config(), token_exchange=exchange)

    response = flow.refresh_access_token(refresh_token="refresh-a")

    assert response.access_token == "access-b"
    assert stored == {}
