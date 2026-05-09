from __future__ import annotations

import copy
import logging
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.schemas.cloud import CloudIngestResponse
from services.cloud_api.middleware.forbid_raw import forbid_raw_payload_middleware
from services.cloud_api.routes import telemetry
from services.cloud_api.routes.telemetry import get_telemetry_service
from services.cloud_api.services.auth_service import CloudTokenCodec, get_token_codec

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"
RAW_SENTINEL = "RAW_SENTINEL_SHOULD_NOT_APPEAR"


class RecordingTelemetryService:
    def __init__(self) -> None:
        self.segment_calls = 0
        self.posterior_calls = 0

    def ingest_segments(self, batch: Any, *, client_id: str) -> CloudIngestResponse:
        del client_id
        self.segment_calls += 1
        return CloudIngestResponse(
            accepted_count=len(batch.segments),
            inserted_count=len(batch.segments),
        )

    def ingest_posterior_deltas(self, batch: Any, *, client_id: str) -> CloudIngestResponse:
        del client_id
        self.posterior_calls += 1
        return CloudIngestResponse(
            accepted_count=len(batch.deltas),
            inserted_count=len(batch.deltas),
        )


@pytest.fixture
def service() -> RecordingTelemetryService:
    return RecordingTelemetryService()


@pytest.fixture
def client(service: RecordingTelemetryService) -> Generator[TestClient]:
    test_app = FastAPI()
    test_app.middleware("http")(forbid_raw_payload_middleware)
    test_app.include_router(telemetry.router, prefix="/v4")
    test_app.dependency_overrides[get_telemetry_service] = lambda: service
    test_app.dependency_overrides[get_token_codec] = lambda: CloudTokenCodec(
        secret="integration-test-secret",
        allowed_client_ids=frozenset({"desktop-a"}),
    )
    with TestClient(test_app) as test_client:
        yield test_client


def _auth_headers() -> dict[str, str]:
    codec = CloudTokenCodec(
        secret="integration-test-secret",
        allowed_client_ids=frozenset({"desktop-a"}),
    )
    token = codec.issue_token_response("desktop-a").access_token
    return {"Authorization": f"Bearer {token}"}


def _segment_batch(*, codec: str = "h264") -> dict[str, Any]:
    timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat()
    return {
        "segments": [
            {
                "session_id": SESSION_ID,
                "segment_id": SEGMENT_ID,
                "segment_window_start_utc": timestamp,
                "segment_window_end_utc": timestamp,
                "timestamp_utc": timestamp,
                "media_source": {
                    "stream_url": "https://example.com/stream",
                    "codec": codec,
                    "resolution": [1920, 1080],
                },
                "segments": [],
                "_active_arm": "warm_welcome",
                "_experiment_id": 1,
                "_expected_greeting": "Say hello to the creator",
                "_stimulus_time": 100.0,
                "_au12_series": [
                    {"timestamp_s": 100.1, "intensity": 0.2},
                    {"timestamp_s": 100.2, "intensity": 0.5},
                ],
                "_bandit_decision_snapshot": {
                    "selection_method": "thompson_sampling",
                    "selection_time_utc": timestamp,
                    "experiment_id": 1,
                    "policy_version": "ts-v1",
                    "selected_arm_id": "warm_welcome",
                    "candidate_arm_ids": ["warm_welcome", "direct_question"],
                    "posterior_by_arm": {
                        "warm_welcome": {"alpha": 1.0, "beta": 1.0},
                        "direct_question": {"alpha": 1.0, "beta": 1.0},
                    },
                    "sampled_theta_by_arm": {
                        "warm_welcome": 0.72,
                        "direct_question": 0.44,
                    },
                    "expected_greeting": "Say hello to the creator",
                    "decision_context_hash": DECISION_CONTEXT_HASH,
                    "random_seed": 42,
                },
            }
        ],
        "attribution_events": [],
    }


def _posterior_delta_batch() -> dict[str, Any]:
    return {
        "deltas": [
            {
                "experiment_id": 1,
                "arm_id": "warm_welcome",
                "delta_alpha": 0.82,
                "delta_beta": 0.18,
                "segment_id": SEGMENT_ID,
                "client_id": "desktop-a",
                "event_id": "00000000-0000-4000-8000-000000000010",
                "applied_at_utc": datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat(),
                "decision_context_hash": DECISION_CONTEXT_HASH,
            }
        ]
    }


def _with_nested_extra(extra: dict[str, Any]) -> dict[str, Any]:
    payload = _segment_batch()
    payload["segments"][0]["privacy_probe"] = extra
    return payload


def _with_allowed_segment_payload(extra: dict[str, Any]) -> dict[str, Any]:
    payload = _segment_batch()
    payload["segments"][0]["segments"] = [extra]
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        _with_nested_extra({"_audio_data": RAW_SENTINEL}),
        _with_nested_extra({"_frame_data": RAW_SENTINEL}),
        _with_nested_extra({"audio_b64": RAW_SENTINEL}),
        _with_nested_extra({"pcm_samples": [0, 1, -1]}),
        _with_nested_extra({"decoded_frames": [[[1, 2, 3]]]}),
        _with_nested_extra({"voice_embedding": [0.1, 0.2, 0.3]}),
        _with_nested_extra({"raw_provider_response": {"body": RAW_SENTINEL}}),
        _with_nested_extra(
            {"event_type": "physiological_chunk", "payload": {"body": RAW_SENTINEL}}
        ),
        _with_nested_extra({"semantic_rationale": RAW_SENTINEL}),
        _with_nested_extra({"audioBase64": RAW_SENTINEL}),
        _with_allowed_segment_payload({"imageBase64": RAW_SENTINEL}),
        _with_allowed_segment_payload({"videoBase64": RAW_SENTINEL}),
        _with_allowed_segment_payload({"voicePrint": [0.1, 0.2, 0.3]}),
    ],
)
def test_privacy_perimeter_rejects_raw_media_before_validation(
    client: TestClient,
    service: RecordingTelemetryService,
    caplog: pytest.LogCaptureFixture,
    payload: dict[str, Any],
) -> None:
    caplog.set_level(logging.WARNING, logger="services.cloud_api.middleware.forbid_raw")

    response = client.post(
        "/v4/telemetry/segments",
        json=payload,
        headers=_auth_headers(),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "raw media payloads are not accepted by cloud API"}
    assert service.segment_calls == 0
    assert RAW_SENTINEL not in response.text
    assert RAW_SENTINEL not in caplog.text


def test_privacy_perimeter_allows_raw_codec_metadata(
    client: TestClient,
    service: RecordingTelemetryService,
) -> None:
    response = client.post(
        "/v4/telemetry/segments",
        json=_segment_batch(codec="raw"),
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "accepted_count": 1, "inserted_count": 1}
    assert service.segment_calls == 1


def test_privacy_perimeter_allows_posterior_deltas(
    client: TestClient,
    service: RecordingTelemetryService,
) -> None:
    response = client.post(
        "/v4/telemetry/posterior_deltas",
        json=_posterior_delta_batch(),
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "accepted_count": 1, "inserted_count": 1}
    assert service.posterior_calls == 1


def test_malformed_schema_still_reaches_pydantic_validation(
    client: TestClient,
    service: RecordingTelemetryService,
) -> None:
    payload = copy.deepcopy(_segment_batch())
    del payload["segments"][0]["segment_id"]

    response = client.post(
        "/v4/telemetry/segments",
        json=payload,
        headers=_auth_headers(),
    )

    assert response.status_code == 422
    assert "segment_id" in response.text
    assert service.segment_calls == 0


def test_malformed_json_still_uses_fastapi_validation(
    client: TestClient,
    service: RecordingTelemetryService,
) -> None:
    response = client.post(
        "/v4/telemetry/segments",
        content="{not json",
        headers={"content-type": "application/json", **_auth_headers()},
    )

    assert response.status_code == 422
    assert service.segment_calls == 0


def test_privacy_perimeter_preserves_auth_barrier_when_bearer_missing(
    client: TestClient,
    service: RecordingTelemetryService,
) -> None:
    response = client.post("/v4/telemetry/segments", json=_segment_batch())

    assert response.status_code == 401
    assert response.json() == {"detail": "missing bearer token"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert service.segment_calls == 0


def test_privacy_perimeter_preserves_auth_barrier_when_bearer_invalid(
    client: TestClient,
    service: RecordingTelemetryService,
) -> None:
    response = client.post(
        "/v4/telemetry/posterior_deltas",
        json=_posterior_delta_batch(),
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid bearer token"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert service.posterior_calls == 0
