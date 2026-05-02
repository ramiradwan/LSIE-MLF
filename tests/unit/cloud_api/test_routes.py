from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from starlette.routing import Route

from packages.schemas.cloud import (
    CloudIngestResponse,
    CloudSessionCreateRequest,
    CloudSessionCreateResponse,
    CloudSessionEndRequest,
    CloudSessionEndResponse,
    ExperimentBundle,
    OAuthTokenRequest,
    OAuthTokenResponse,
    TelemetryPosteriorDeltaBatch,
)
from services.cloud_api.main import app
from services.cloud_api.routes import auth, experiments, sessions, telemetry
from services.cloud_api.services.auth_service import OAuthTokenService
from services.cloud_api.services.bundle_service import ExperimentBundleService
from services.cloud_api.services.session_service import CloudSessionService, SessionNotFoundError
from services.cloud_api.services.telemetry_service import TelemetryIngestService

SEGMENT_ID = "a" * 64


class RuntimeErrorTelemetryService(TelemetryIngestService):
    def ingest_posterior_deltas(
        self,
        batch: TelemetryPosteriorDeltaBatch,
    ) -> CloudIngestResponse:
        del batch
        raise RuntimeError("Message Broker unavailable")


class RuntimeErrorSessionService(CloudSessionService):
    def create_session(self, request: CloudSessionCreateRequest) -> CloudSessionCreateResponse:
        del request
        raise RuntimeError("Persistent Store unavailable")


class MissingSessionService(CloudSessionService):
    def end_session(
        self,
        session_id: uuid.UUID,
        request: CloudSessionEndRequest,
    ) -> CloudSessionEndResponse:
        del request
        raise SessionNotFoundError(f"session {session_id} not found")


class RuntimeErrorBundleService(ExperimentBundleService):
    def build_bundle(self) -> ExperimentBundle:
        raise RuntimeError("Persistent Store unavailable")


class FixedOAuthTokenService(OAuthTokenService):
    def exchange_token(self, request: OAuthTokenRequest) -> OAuthTokenResponse:
        del request
        return OAuthTokenResponse(
            access_token="access-a",
            expires_in=3600,
            refresh_token="refresh-a",
            scope="telemetry experiments sessions",
        )


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
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_telemetry_runtime_error_maps_to_503() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await telemetry.ingest_posterior_deltas(
            _posterior_delta_batch(),
            service=RuntimeErrorTelemetryService(),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Message Broker unavailable"


@pytest.mark.asyncio
async def test_session_runtime_error_maps_to_503(sample_timestamp: datetime) -> None:
    request = CloudSessionCreateRequest(
        client_id="desktop-a",
        started_at_utc=sample_timestamp,
        policy_version="cloud-v1",
    )

    with pytest.raises(HTTPException) as exc_info:
        await sessions.create_session(request, service=RuntimeErrorSessionService())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Persistent Store unavailable"


@pytest.mark.asyncio
async def test_session_not_found_maps_to_404(sample_timestamp: datetime) -> None:
    session_id = uuid.uuid4()
    request = CloudSessionEndRequest(ended_at_utc=sample_timestamp)

    with pytest.raises(HTTPException) as exc_info:
        await sessions.end_session(session_id, request, service=MissingSessionService())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == f"session {session_id} not found"


@pytest.mark.asyncio
async def test_experiment_bundle_runtime_error_maps_to_503() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await experiments.get_experiment_bundle(service=RuntimeErrorBundleService())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Persistent Store unavailable"


@pytest.mark.asyncio
async def test_oauth_route_returns_token_response(sample_timestamp: datetime) -> None:
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


def test_cloud_app_mounts_v4_routes() -> None:
    route_paths = {route.path for route in app.routes if isinstance(route, Route)}

    assert "/v4/telemetry/segments" in route_paths
    assert "/v4/telemetry/posterior_deltas" in route_paths
    assert "/v4/sessions" in route_paths
    assert "/v4/sessions/{session_id}/end" in route_paths
    assert "/v4/experiments/bundle" in route_paths
    assert "/v4/auth/oauth/token" in route_paths
