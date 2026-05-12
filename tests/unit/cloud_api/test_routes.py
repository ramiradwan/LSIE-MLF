from __future__ import annotations

import base64
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import pytest
from Crypto.PublicKey import ECC
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import Response
from starlette.routing import Route

from packages.schemas.cloud import (
    CloudIngestResponse,
    CloudSessionCreateRequest,
    CloudSessionCreateResponse,
    CloudSessionEndRequest,
    CloudSessionEndResponse,
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
    OAuthTokenRequest,
    OAuthTokenResponse,
    TelemetryPosteriorDeltaBatch,
    TelemetrySegmentBatch,
)
from services.cloud_api import main as cloud_main
from services.cloud_api.main import app
from services.cloud_api.routes import auth, experiments, health, sessions, telemetry
from services.cloud_api.routes.experiments import get_bundle_service
from services.cloud_api.routes.sessions import get_session_service
from services.cloud_api.routes.telemetry import get_telemetry_service
from services.cloud_api.services.auth_service import (
    AuthConfigurationError,
    AuthenticatedClient,
    CloudTokenCodec,
    InvalidTokenError,
    OAuthTokenService,
    get_token_codec,
)
from services.cloud_api.services.bundle_service import ExperimentBundleService
from services.cloud_api.services.experiment_bundle_service import (
    LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY,
    _sign_payload,
)
from services.cloud_api.services.session_service import (
    CloudSessionService,
    SessionNotFoundError,
    SessionOwnershipError,
)
from services.cloud_api.services.telemetry_service import (
    PosteriorDeltaAuthorizationError,
    TelemetryIngestService,
)
from services.desktop_app.cloud.auth_flow import build_code_challenge
from services.desktop_app.cloud.experiment_bundle import BundleVerificationConfig, verify_bundle

SEGMENT_ID = "a" * 64
_AUTHENTICATED_CLIENT = AuthenticatedClient(
    client_id="desktop-a",
    scope="telemetry experiments sessions",
    expires_at_epoch=4102444800,
)


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


class RuntimeErrorTelemetryService(TelemetryIngestService):
    def ingest_posterior_deltas(
        self,
        batch: TelemetryPosteriorDeltaBatch,
        *,
        client_id: str,
    ) -> CloudIngestResponse:
        del batch, client_id
        raise RuntimeError("Persistent Store unavailable")


class ForbiddenTelemetryService(TelemetryIngestService):
    def ingest_posterior_deltas(
        self,
        batch: TelemetryPosteriorDeltaBatch,
        *,
        client_id: str,
    ) -> CloudIngestResponse:
        del batch, client_id
        raise PosteriorDeltaAuthorizationError("posterior delta rejected")


class RuntimeErrorSessionService(CloudSessionService):
    def create_session(
        self,
        request: CloudSessionCreateRequest,
        *,
        client_id: str,
    ) -> CloudSessionCreateResponse:
        del request, client_id
        raise RuntimeError("Persistent Store unavailable")


class ForbiddenSessionService(CloudSessionService):
    def create_session(
        self,
        request: CloudSessionCreateRequest,
        *,
        client_id: str,
    ) -> CloudSessionCreateResponse:
        del request, client_id
        raise SessionOwnershipError("session client_id does not match authenticated client")


class MissingSessionService(CloudSessionService):
    def end_session(
        self,
        session_id: uuid.UUID,
        request: CloudSessionEndRequest,
        *,
        client_id: str,
    ) -> CloudSessionEndResponse:
        del request, client_id
        raise SessionNotFoundError(f"session {session_id} not found")


class RuntimeErrorBundleService(ExperimentBundleService):
    def build_bundle(self) -> ExperimentBundle:
        raise RuntimeError("Persistent Store unavailable")


async def _readiness_ok() -> bool:
    return True


async def _readiness_failure() -> bool:
    return False


class FixedOAuthTokenService(OAuthTokenService):
    def exchange_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        del request
        return OAuthTokenResponse(
            access_token="access-a",
            expires_in=3600,
            refresh_token="refresh-a",
            scope="telemetry experiments sessions",
        )


class InvalidOAuthTokenService(OAuthTokenService):
    def exchange_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        del request
        raise InvalidTokenError("authorization code PKCE verification failed")


class MisconfiguredOAuthTokenService(OAuthTokenService):
    def exchange_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        del request
        raise AuthConfigurationError("LSIE_CLOUD_TOKEN_SIGNING_SECRET contains secret details")


class HappyTelemetryService(TelemetryIngestService):
    def ingest_segments(
        self,
        batch: TelemetrySegmentBatch,
        *,
        client_id: str,
    ) -> CloudIngestResponse:
        del client_id
        accepted = len(batch.segments) + len(batch.attribution_events)
        return CloudIngestResponse(accepted_count=accepted, inserted_count=accepted)

    def ingest_posterior_deltas(self, batch: object, *, client_id: str) -> CloudIngestResponse:
        del batch, client_id
        return CloudIngestResponse(accepted_count=1, inserted_count=1)


class HappySessionService(CloudSessionService):
    def create_session(
        self,
        request: CloudSessionCreateRequest,
        *,
        client_id: str,
    ) -> CloudSessionCreateResponse:
        del request, client_id
        return CloudSessionCreateResponse(
            session_id=uuid.UUID("00000000-0000-4000-8000-0000000000aa"),
            client_id="desktop-a",
            created_at_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        )

    def end_session(
        self,
        session_id: uuid.UUID,
        request: CloudSessionEndRequest,
        *,
        client_id: str,
    ) -> CloudSessionEndResponse:
        del client_id
        return CloudSessionEndResponse(session_id=session_id, ended_at_utc=request.ended_at_utc)


class HappyBundleService(ExperimentBundleService):
    def build_bundle(self) -> ExperimentBundle:
        now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
        return ExperimentBundle(
            bundle_id="bundle-1",
            issued_at_utc=now,
            expires_at_utc=datetime(2026, 5, 2, 13, 0, tzinfo=UTC),
            policy_version="cloud-v1",
            experiments=[
                ExperimentBundleExperiment(
                    experiment_id="exp-1",
                    label="Experiment 1",
                    arms=[
                        ExperimentBundleArm(
                            arm_id="arm-a",
                            greeting_text="Hello",
                            posterior_alpha=1.0,
                            posterior_beta=1.0,
                            selection_count=0,
                            enabled=True,
                        )
                    ],
                )
            ],
            signature="sig",
        )


def _token_codec() -> CloudTokenCodec:
    return CloudTokenCodec(
        secret="unit-test-secret",
        allowed_client_ids=frozenset({"desktop-a"}),
    )


def _auth_headers() -> dict[str, str]:
    access_token = _token_codec().issue_token_response("desktop-a").access_token
    return {"Authorization": f"Bearer {access_token}"}


def test_oauth_service_exchanges_signed_authorization_code() -> None:
    codec = _token_codec()
    service = OAuthTokenService(codec=codec)
    authorization_code = codec.issue_authorization_code(
        client_id="desktop-a",
        redirect_uri="http://127.0.0.1:8765/oauth/callback",
        code_challenge=build_code_challenge("verifier-a"),
    )

    response = service.exchange_token(
        OAuthTokenRequest(
            grant_type="authorization_code",
            client_id="desktop-a",
            code=authorization_code,
            code_verifier="verifier-a",
            redirect_uri="http://127.0.0.1:8765/oauth/callback",
        )
    )

    assert response.token_type == "Bearer"
    assert response.refresh_token is not None


def test_oauth_service_rejects_authorization_code_with_wrong_verifier() -> None:
    codec = _token_codec()
    service = OAuthTokenService(codec=codec)
    authorization_code = codec.issue_authorization_code(
        client_id="desktop-a",
        redirect_uri="http://127.0.0.1:8765/oauth/callback",
        code_challenge=build_code_challenge("verifier-a"),
    )

    with pytest.raises(InvalidTokenError, match="PKCE"):
        service.exchange_token(
            OAuthTokenRequest(
                grant_type="authorization_code",
                client_id="desktop-a",
                code=authorization_code,
                code_verifier="wrong-verifier",
                redirect_uri="http://127.0.0.1:8765/oauth/callback",
            )
        )


def test_oauth_authorize_route_redirects_with_exchangeable_code() -> None:
    with TestClient(_protected_app(), follow_redirects=False) as client:
        response = client.get(
            "/v4/auth/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": "desktop-a",
                "redirect_uri": "http://127.0.0.1:8765/oauth/callback",
                "code_challenge": build_code_challenge("verifier-a"),
                "code_challenge_method": "S256",
                "state": "state-a",
            },
        )

    assert response.status_code == 302
    location = response.headers["location"]
    parsed = urlparse(location)
    query = parse_qs(parsed.query)
    assert (
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "http://127.0.0.1:8765/oauth/callback"
    )
    assert query["state"] == ["state-a"]

    token_response = OAuthTokenService(codec=_token_codec()).exchange_token(
        OAuthTokenRequest(
            grant_type="authorization_code",
            client_id="desktop-a",
            code=query["code"][0],
            code_verifier="verifier-a",
            redirect_uri="http://127.0.0.1:8765/oauth/callback",
        )
    )

    assert token_response.token_type == "Bearer"
    assert token_response.refresh_token is not None


@pytest.mark.parametrize(
    "params",
    [
        {
            "response_type": "token",
            "client_id": "desktop-a",
            "redirect_uri": "http://127.0.0.1:8765/oauth/callback",
            "code_challenge": build_code_challenge("verifier-a"),
            "code_challenge_method": "S256",
        },
        {
            "response_type": "code",
            "client_id": "desktop-a",
            "redirect_uri": "https://cloud.example.test/oauth/callback",
            "code_challenge": build_code_challenge("verifier-a"),
            "code_challenge_method": "S256",
        },
        {
            "response_type": "code",
            "client_id": "desktop-a",
            "redirect_uri": "http://127.0.0.1:8765/oauth/callback",
            "code_challenge": build_code_challenge("verifier-a"),
            "code_challenge_method": "plain",
        },
    ],
)
def test_oauth_authorize_route_rejects_invalid_request(params: dict[str, str]) -> None:
    with TestClient(_protected_app(), follow_redirects=False) as client:
        response = client.get("/v4/auth/oauth/authorize", params=params)

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid authorization request"}


def test_oauth_authorize_route_rejects_disallowed_client_without_raw_id() -> None:
    with TestClient(_protected_app(), follow_redirects=False) as client:
        response = client.get(
            "/v4/auth/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": "desktop-secret-client",
                "redirect_uri": "http://127.0.0.1:8765/oauth/callback",
                "code_challenge": build_code_challenge("verifier-a"),
                "code_challenge_method": "S256",
            },
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "client_id is not allowed"}


def _protected_app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(telemetry.router, prefix="/v4")
    test_app.include_router(sessions.router, prefix="/v4")
    test_app.include_router(experiments.router, prefix="/v4")
    test_app.include_router(auth.router, prefix="/v4")
    test_app.dependency_overrides[get_token_codec] = _token_codec
    test_app.dependency_overrides[auth.get_authorization_token_codec] = _token_codec
    test_app.dependency_overrides[get_telemetry_service] = HappyTelemetryService
    test_app.dependency_overrides[get_session_service] = HappySessionService
    test_app.dependency_overrides[get_bundle_service] = HappyBundleService
    return test_app


@pytest.fixture
def protected_client() -> Generator[TestClient, None, None]:
    with TestClient(_protected_app()) as client:
        yield client


def _posterior_delta_batch() -> TelemetryPosteriorDeltaBatch:
    return TelemetryPosteriorDeltaBatch.model_validate(
        {
            "deltas": [
                {
                    "experiment_id": 101,
                    "arm_id": "arm_a",
                    "delta_alpha": 1.0,
                    "delta_beta": 0.0,
                    "segment_id": SEGMENT_ID,
                    "client_id": "desktop-a",
                    "event_id": uuid.uuid4(),
                    "applied_at_utc": datetime.now(UTC),
                    "decision_context_hash": SEGMENT_ID,
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_telemetry_runtime_error_maps_to_503() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await telemetry.ingest_posterior_deltas(
            _posterior_delta_batch(),
            authenticated_client=_AUTHENTICATED_CLIENT,
            service=RuntimeErrorTelemetryService(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Persistent Store unavailable"


@pytest.mark.asyncio
async def test_telemetry_authorization_error_maps_to_403() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await telemetry.ingest_posterior_deltas(
            _posterior_delta_batch(),
            authenticated_client=_AUTHENTICATED_CLIENT,
            service=ForbiddenTelemetryService(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "posterior delta rejected"


@pytest.mark.asyncio
async def test_session_runtime_error_maps_to_503(sample_timestamp: datetime) -> None:
    request = CloudSessionCreateRequest(
        client_id="desktop-a",
        started_at_utc=sample_timestamp,
        policy_version="cloud-v1",
    )

    with pytest.raises(HTTPException) as exc_info:
        await sessions.create_session(
            request,
            authenticated_client=_AUTHENTICATED_CLIENT,
            service=RuntimeErrorSessionService(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Persistent Store unavailable"


@pytest.mark.asyncio
async def test_session_ownership_error_maps_to_403(sample_timestamp: datetime) -> None:
    request = CloudSessionCreateRequest(
        client_id="desktop-b",
        started_at_utc=sample_timestamp,
        policy_version="cloud-v1",
    )

    with pytest.raises(HTTPException) as exc_info:
        await sessions.create_session(
            request,
            authenticated_client=_AUTHENTICATED_CLIENT,
            service=ForbiddenSessionService(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "session client_id does not match authenticated client"


@pytest.mark.asyncio
async def test_session_not_found_maps_to_404(sample_timestamp: datetime) -> None:
    session_id = uuid.uuid4()
    request = CloudSessionEndRequest(ended_at_utc=sample_timestamp)

    with pytest.raises(HTTPException) as exc_info:
        await sessions.end_session(
            session_id,
            request,
            authenticated_client=_AUTHENTICATED_CLIENT,
            service=MissingSessionService(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == f"session {session_id} not found"


@pytest.mark.asyncio
async def test_experiment_bundle_runtime_error_maps_to_503() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await experiments.get_experiment_bundle(
            authenticated_client=_AUTHENTICATED_CLIENT,
            service=RuntimeErrorBundleService(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Persistent Store unavailable"


def test_experiment_bundle_payload_signs_with_ed25519_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    unsigned_bundle = (
        HappyBundleService()
        .build_bundle()
        .model_copy(
            update={"issued_at_utc": now, "expires_at_utc": datetime(2026, 5, 2, 13, 0, tzinfo=UTC)}
        )
    )
    payload = ExperimentBundlePayload.model_validate(
        unsigned_bundle.model_dump(exclude={"signature"})
    )
    private_key = ECC.generate(curve="Ed25519")
    monkeypatch.setenv(
        LSIE_CLOUD_BUNDLE_ED25519_PRIVATE_KEY,
        _urlsafe_b64encode(private_key.export_key(format="DER")),
    )

    signed = unsigned_bundle.model_copy(update={"signature": _sign_payload(payload)})

    public_key_raw = private_key.public_key().export_key(format="DER")
    verify_bundle(
        signed,
        config=BundleVerificationConfig(
            ed25519_public_key=_urlsafe_b64encode(public_key_raw).encode("utf-8")
        ),
        now_utc=now,
    )


@pytest.mark.asyncio
async def test_oauth_route_returns_token_response_for_refresh_grant(
    sample_timestamp: datetime,
) -> None:
    del sample_timestamp
    request = OAuthTokenRequest(
        grant_type="refresh_token",
        client_id="desktop-a",
        refresh_token="refresh-a",
    )

    response = await auth.exchange_oauth_token(request, service=FixedOAuthTokenService())

    assert response.access_token == "access-a"
    assert response.token_type == "Bearer"
    assert response.refresh_token == "refresh-a"


@pytest.mark.asyncio
async def test_oauth_route_returns_token_response_for_authorization_code_grant(
    sample_timestamp: datetime,
) -> None:
    del sample_timestamp
    request = OAuthTokenRequest(
        grant_type="authorization_code",
        client_id="desktop-a",
        code="code-a",
        code_verifier="verifier-a",
        redirect_uri="http://127.0.0.1:8765/oauth/callback",
    )

    response = await auth.exchange_oauth_token(request, service=FixedOAuthTokenService())

    assert response.access_token == "access-a"
    assert response.token_type == "Bearer"
    assert response.refresh_token == "refresh-a"


@pytest.mark.asyncio
async def test_oauth_route_maps_invalid_token_to_401(sample_timestamp: datetime) -> None:
    del sample_timestamp
    request = OAuthTokenRequest(
        grant_type="authorization_code",
        client_id="desktop-a",
        code="code-a",
        code_verifier="verifier-a",
        redirect_uri="http://127.0.0.1:8765/oauth/callback",
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth.exchange_oauth_token(request, service=InvalidOAuthTokenService())

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "authorization code PKCE verification failed"


@pytest.mark.asyncio
async def test_oauth_route_maps_invalid_client_to_bounded_401() -> None:
    request = OAuthTokenRequest(
        grant_type="refresh_token",
        client_id="desktop-secret-client",
        refresh_token="refresh-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth.exchange_oauth_token(request, service=OAuthTokenService(codec=_token_codec()))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "client_id is not allowed"


@pytest.mark.asyncio
async def test_oauth_route_maps_configuration_error_to_bounded_503() -> None:
    request = OAuthTokenRequest(
        grant_type="refresh_token",
        client_id="desktop-a",
        refresh_token="refresh-a",
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth.exchange_oauth_token(request, service=MisconfiguredOAuthTokenService())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "cloud authorization is not configured"


def test_cloud_app_mounts_health_routes() -> None:
    route_paths = {route.path for route in app.routes if isinstance(route, Route)}

    assert "/healthz" in route_paths
    assert "/readyz" in route_paths


def test_cloud_app_mounts_v4_routes() -> None:
    route_paths = {route.path for route in app.routes if isinstance(route, Route)}

    assert "/v4/telemetry/segments" in route_paths
    assert "/v4/telemetry/posterior_deltas" in route_paths
    assert "/v4/sessions" in route_paths
    assert "/v4/sessions/{session_id}/end" in route_paths
    assert "/v4/experiments/bundle" in route_paths
    assert "/v4/auth/oauth/authorize" in route_paths
    assert "/v4/auth/oauth/token" in route_paths


def test_cloud_app_serves_healthz_when_startup_pool_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_init_pool() -> None:
        raise RuntimeError("postgres://user:secret@db/app")

    monkeypatch.setattr(cloud_main, "init_pool", fail_init_pool)

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_returns_bounded_liveness_without_readiness_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_if_called() -> bool:
        raise AssertionError("healthz must not touch PostgreSQL")

    test_app = FastAPI()
    test_app.include_router(health.router)
    monkeypatch.setattr(health, "check_readiness", fail_if_called)

    with TestClient(test_app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_bounded_success(monkeypatch: pytest.MonkeyPatch) -> None:
    test_app = FastAPI()
    test_app.include_router(health.router)
    monkeypatch.setattr(health, "check_readiness", _readiness_ok)

    with TestClient(test_app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok"}}


def test_readyz_returns_sanitized_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    test_app = FastAPI()
    test_app.include_router(health.router)
    monkeypatch.setattr(health, "check_readiness", _readiness_failure)

    with TestClient(test_app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "checks": {"database": "unavailable"},
    }


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("post", "/v4/telemetry/segments", {"segments": [], "attribution_events": []}),
        ("post", "/v4/telemetry/posterior_deltas", {"deltas": []}),
        (
            "post",
            "/v4/sessions",
            {
                "client_id": "desktop-a",
                "started_at_utc": datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat(),
                "policy_version": "cloud-v1",
            },
        ),
        (
            "post",
            "/v4/sessions/00000000-0000-4000-8000-000000000001/end",
            {"ended_at_utc": datetime(2026, 5, 2, 12, 5, tzinfo=UTC).isoformat()},
        ),
        ("get", "/v4/experiments/bundle", None),
    ],
)
def test_protected_routes_require_bearer_token(
    protected_client: TestClient,
    method: str,
    path: str,
    json_body: dict[str, object] | None,
) -> None:
    response = _request(protected_client, method, path, json=json_body)

    assert response.status_code == 401
    assert response.json() == {"detail": "missing bearer token"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("post", "/v4/telemetry/segments", {"segments": [], "attribution_events": []}),
        ("post", "/v4/telemetry/posterior_deltas", {"deltas": []}),
        (
            "post",
            "/v4/sessions",
            {
                "client_id": "desktop-a",
                "started_at_utc": datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat(),
                "policy_version": "cloud-v1",
            },
        ),
        (
            "post",
            "/v4/sessions/00000000-0000-4000-8000-000000000001/end",
            {"ended_at_utc": datetime(2026, 5, 2, 12, 5, tzinfo=UTC).isoformat()},
        ),
        ("get", "/v4/experiments/bundle", None),
    ],
)
def test_protected_routes_reject_invalid_bearer_token(
    protected_client: TestClient,
    method: str,
    path: str,
    json_body: dict[str, object] | None,
) -> None:
    response = _request(
        protected_client,
        method,
        path,
        json=json_body,
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid bearer token"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        (
            "post",
            "/v4/sessions",
            {
                "client_id": "desktop-a",
                "started_at_utc": datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat(),
                "policy_version": "cloud-v1",
            },
        ),
        (
            "post",
            "/v4/sessions/00000000-0000-4000-8000-000000000001/end",
            {"ended_at_utc": datetime(2026, 5, 2, 12, 5, tzinfo=UTC).isoformat()},
        ),
        ("get", "/v4/experiments/bundle", None),
    ],
)
def test_protected_routes_accept_valid_bearer_token(
    protected_client: TestClient,
    method: str,
    path: str,
    json_body: dict[str, object] | None,
) -> None:
    response = _request(
        protected_client,
        method,
        path,
        json=json_body,
        headers=_auth_headers(),
    )

    assert response.status_code < 400


def test_segment_route_accepts_event_only_batch(protected_client: TestClient) -> None:
    timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat()
    response = protected_client.post(
        "/v4/telemetry/segments",
        json={
            "segments": [],
            "attribution_events": [
                {
                    "event_id": "00000000-0000-4000-8000-000000000003",
                    "session_id": "00000000-0000-4000-8000-000000000001",
                    "segment_id": SEGMENT_ID,
                    "event_type": "greeting_interaction",
                    "event_time_utc": timestamp,
                    "stimulus_time_utc": None,
                    "selected_arm_id": "arm_a",
                    "expected_rule_text_hash": SEGMENT_ID,
                    "semantic_method": "cross_encoder",
                    "semantic_method_version": "ce-v1",
                    "semantic_p_match": 0.91,
                    "semantic_reason_code": None,
                    "reward_path_version": "reward-v1",
                    "bandit_decision_snapshot": {
                        "selection_method": "thompson_sampling",
                        "selection_time_utc": timestamp,
                        "experiment_id": 101,
                        "policy_version": "ts-v1",
                        "selected_arm_id": "arm_a",
                        "candidate_arm_ids": ["arm_a", "arm_b"],
                        "posterior_by_arm": {
                            "arm_a": {"alpha": 2.0, "beta": 3.0},
                            "arm_b": {"alpha": 1.0, "beta": 1.0},
                        },
                        "sampled_theta_by_arm": {"arm_a": 0.72, "arm_b": 0.44},
                        "expected_greeting": "Say hello to the creator",
                        "decision_context_hash": SEGMENT_ID,
                        "random_seed": 42,
                    },
                    "evidence_flags": [],
                    "finality": "online_provisional",
                    "schema_version": "v1",
                    "created_at": timestamp,
                }
            ],
        },
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "accepted_count": 1, "inserted_count": 1}


def _request(
    client: TestClient,
    method: str,
    path: str,
    *,
    json: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    if method == "get":
        return client.get(path, headers=headers)
    return client.post(path, json=json, headers=headers)
