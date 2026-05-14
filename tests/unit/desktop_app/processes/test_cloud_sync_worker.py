from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from packages.schemas.attribution import AttributionEvent, BanditDecisionSnapshot
from packages.schemas.cloud import (
    ExperimentBundle,
    ExperimentBundleArm,
    ExperimentBundleExperiment,
    ExperimentBundlePayload,
    PosteriorDelta,
)
from packages.schemas.evaluation import StimulusDefinition, StimulusPayload
from packages.schemas.inference_handoff import InferenceHandoffPayload
from packages.schemas.operator_console import CloudExperimentRefreshStatus, CloudOperatorErrorCode
from services.desktop_app.cloud.experiment_bundle import (
    BundleVerificationConfig,
    sign_bundle_payload,
)
from services.desktop_app.cloud.outbox import CloudOutbox, PendingUpload, canonical_payload_json
from services.desktop_app.os_adapter import SecretStoreUnavailableError
from services.desktop_app.processes.cloud_sync_worker import CloudSyncConfig, CloudSyncWorker

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"
BUNDLE_SECRET = "bundle-secret"


class ScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = responses or []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            return httpx.Response(
                200, json={"status": "accepted", "accepted_count": 1, "inserted_count": 1}
            )
        response = self._responses.pop(0)
        response.request = request
        return response


class TokenThenTimeoutTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._request_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._request_count += 1
        if self._request_count == 1:
            return httpx.Response(
                200,
                json={"access_token": "access-a", "expires_in": 3600},
                request=request,
            )
        raise httpx.ReadTimeout("timed out", request=request)


def _bandit_snapshot_data(sample_timestamp: datetime) -> dict[str, Any]:
    return {
        "selection_method": "thompson_sampling",
        "selection_time_utc": sample_timestamp,
        "experiment_id": 101,
        "policy_version": "ts-v1",
        "selected_arm_id": "arm_a",
        "candidate_arm_ids": ["arm_a", "arm_b"],
        "posterior_by_arm": {
            "arm_a": {"alpha": 2.0, "beta": 3.0},
            "arm_b": {"alpha": 1.0, "beta": 1.0},
        },
        "sampled_theta_by_arm": {"arm_a": 0.72, "arm_b": 0.44},
        "decision_context_hash": DECISION_CONTEXT_HASH,
        "random_seed": 42,
        "stimulus_modality": "spoken_greeting",
        "stimulus_payload": {"content_type": "text", "text": "Say hello to the creator"},
        "expected_stimulus_rule": "Deliver the spoken greeting to the creator",
        "expected_response_rule": "The live streamer acknowledges the greeting",
    }


def _handoff_payload(
    sample_timestamp: datetime,
    *,
    with_physiology: bool = False,
) -> InferenceHandoffPayload:
    payload: dict[str, Any] = {
        "session_id": SESSION_ID,
        "segment_id": SEGMENT_ID,
        "segment_window_start_utc": sample_timestamp,
        "segment_window_end_utc": sample_timestamp,
        "timestamp_utc": sample_timestamp,
        "media_source": {
            "stream_url": "https://example.com/stream",
            "codec": "h264",
            "resolution": [1920, 1080],
        },
        "segments": [],
        "_active_arm": "arm_a",
        "_experiment_id": 101,
        "_stimulus_modality": "spoken_greeting",
        "_stimulus_payload": {"content_type": "text", "text": "Say hello to the creator"},
        "_expected_stimulus_rule": "Deliver the spoken greeting to the creator",
        "_expected_response_rule": "The live streamer acknowledges the greeting",
        "_stimulus_time": None,
        "_au12_series": [{"timestamp_s": 0.0, "intensity": 0.62}],
        "_bandit_decision_snapshot": _bandit_snapshot_data(sample_timestamp),
    }
    if with_physiology:
        payload["_physiological_context"] = {
            "streamer": {
                "rmssd_ms": 42.0,
                "heart_rate_bpm": 72,
                "source_timestamp_utc": sample_timestamp,
                "freshness_s": 3.0,
                "is_stale": False,
                "provider": "oura",
                "source_kind": "ibi",
                "derivation_method": "provider",
                "window_s": 300,
                "validity_ratio": 1.0,
                "is_valid": True,
            }
        }
    return InferenceHandoffPayload.model_validate(payload)


def _attribution_event(sample_timestamp: datetime) -> AttributionEvent:
    return AttributionEvent(
        event_id=uuid.UUID("00000000-0000-4000-8000-000000000003"),
        session_id=uuid.UUID(SESSION_ID),
        segment_id=SEGMENT_ID,
        event_type="stimulus_interaction",
        event_time_utc=sample_timestamp,
        stimulus_time_utc=None,
        selected_arm_id="arm_a",
        expected_rule_text_hash=DECISION_CONTEXT_HASH,
        semantic_method="cross_encoder",
        semantic_method_version="ce-v1",
        semantic_p_match=0.91,
        semantic_reason_code=None,
        reward_path_version="reward-v1",
        bandit_decision_snapshot=BanditDecisionSnapshot.model_validate(
            _bandit_snapshot_data(sample_timestamp)
        ),
        evidence_flags=[],
        finality="online_provisional",
        schema_version="v1",
        created_at=sample_timestamp,
    )


def _posterior_delta(sample_timestamp: datetime) -> PosteriorDelta:
    return PosteriorDelta(
        experiment_id=101,
        arm_id="arm-a",
        delta_alpha=1.0,
        delta_beta=0.0,
        segment_id=SEGMENT_ID,
        client_id="desktop-a",
        event_id=uuid.UUID("00000000-0000-4000-8000-000000000002"),
        applied_at_utc=sample_timestamp,
        decision_context_hash=SEGMENT_ID,
    )


def _bundle_stimulus_definition() -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text="Hello A",
        ),
        expected_stimulus_rule="Deliver the spoken greeting to the creator",
        expected_response_rule="The live streamer acknowledges the greeting",
    )


def _migrated_cached_stimulus_definition(text: str) -> StimulusDefinition:
    return StimulusDefinition(
        stimulus_modality="spoken_greeting",
        stimulus_payload=StimulusPayload(
            content_type="text",
            text=text,
        ),
        expected_stimulus_rule=(
            "Deliver the spoken greeting to the live streamer exactly as written."
        ),
        expected_response_rule=(
            "The live streamer acknowledges the greeting or responds to it on stream."
        ),
    )


def _bundle(*, signature: str | None = None) -> ExperimentBundle:
    payload = ExperimentBundlePayload(
        bundle_id="bundle-a",
        issued_at_utc=datetime(2026, 5, 1, tzinfo=UTC),
        expires_at_utc=datetime(2036, 5, 1, tzinfo=UTC),
        policy_version="v4.0",
        experiments=[
            ExperimentBundleExperiment(
                experiment_id="experiment-a",
                label="Experiment A",
                arms=[
                    ExperimentBundleArm(
                        arm_id="arm-a",
                        stimulus_definition=_bundle_stimulus_definition(),
                        posterior_alpha=2.0,
                        posterior_beta=3.0,
                        selection_count=5,
                    )
                ],
            )
        ],
    )
    effective_signature = signature
    if effective_signature is None:
        effective_signature = sign_bundle_payload(payload, BUNDLE_SECRET)
    return ExperimentBundle(**payload.model_dump(), signature=effective_signature)


def _outbox(tmp_path: Path) -> CloudOutbox:
    return CloudOutbox(tmp_path / "desktop.sqlite")


def _row_statuses(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        return {
            str(row[0]): str(row[1])
            for row in conn.execute("SELECT upload_id, status FROM pending_uploads").fetchall()
        }
    finally:
        conn.close()


def _latest_refresh_state(db_path: Path) -> tuple[str, str | None, int]:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        row = conn.execute(
            """
            SELECT status, error_code, retryable
            FROM cloud_experiment_refresh_state
            WHERE state_key = 'latest'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return str(row[0]), row[1], int(row[2])


async def _sync_once(
    outbox: CloudOutbox, transport: httpx.AsyncBaseTransport
) -> ScriptedTransport | None:
    config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
    worker = CloudSyncWorker(outbox, config)
    async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
        await worker.sync_once(client)
    return transport if isinstance(transport, ScriptedTransport) else None


@pytest.mark.asyncio
async def test_periodic_refresh_applies_verified_bundle_and_records_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    monkeypatch.setattr(
        "services.desktop_app.cloud.experiment_bundle.BundleVerificationConfig.from_env",
        classmethod(
            lambda cls: BundleVerificationConfig(
                signature_mode="hmac-sha256",
                hmac_secret=BUNDLE_SECRET,
            )
        ),
    )
    outbox = _outbox(tmp_path)
    try:
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
        worker = CloudSyncWorker(outbox, config)
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(200, json=_bundle().model_dump(mode="json")),
            ]
        )
        async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
            result = await worker.refresh_experiments_if_due(
                client,
                now=datetime(2026, 5, 2, tzinfo=UTC),
            )
    finally:
        outbox.close()

    assert result is not None
    assert result.status == CloudExperimentRefreshStatus.APPLIED
    assert result.bundle_id == "bundle-a"
    assert [request.url.path for request in transport.requests] == [
        "/v4/auth/oauth/token",
        "/v4/experiments/bundle",
    ]
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    try:
        row = conn.execute(
            """
            SELECT stimulus_definition, alpha_param, beta_param
            FROM experiments
            WHERE experiment_id = 'experiment-a' AND arm = 'arm-a'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert json.loads(str(row[0])) == _bundle_stimulus_definition().model_dump(mode="json")
    assert tuple(row[1:]) == (2.0, 3.0)
    assert _latest_refresh_state(tmp_path / "desktop.sqlite") == ("applied", None, 0)


@pytest.mark.asyncio
async def test_periodic_refresh_401_after_refresh_is_terminal_bounded_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
        worker = CloudSyncWorker(outbox, config)
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(401, json={"detail": "expired access"}),
                httpx.Response(200, json={"access_token": "access-b", "expires_in": 3600}),
                httpx.Response(401, json={"detail": "denied"}),
            ]
        )
        async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
            result = await worker.refresh_experiments_if_due(
                client,
                now=datetime(2026, 5, 2, tzinfo=UTC),
            )
    finally:
        outbox.close()

    assert result is not None
    assert result.status == CloudExperimentRefreshStatus.FAILED
    assert result.error_code == CloudOperatorErrorCode.UNAUTHORIZED
    assert result.retryable is False
    assert result.message == "Cloud authorization was rejected."
    assert _latest_refresh_state(tmp_path / "desktop.sqlite") == ("failed", "unauthorized", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected_code", "retryable"),
    [
        (
            httpx.Response(429, json={"detail": "rate limited"}),
            CloudOperatorErrorCode.RATE_LIMITED,
            True,
        ),
        (
            httpx.Response(503, json={"detail": "unavailable"}),
            CloudOperatorErrorCode.CLOUD_UNAVAILABLE,
            True,
        ),
    ],
)
async def test_periodic_refresh_http_failures_are_bounded_retryable_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    expected_code: CloudOperatorErrorCode,
    retryable: bool,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
        worker = CloudSyncWorker(outbox, config)
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                response,
            ]
        )
        async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
            result = await worker.refresh_experiments_if_due(
                client,
                now=datetime(2026, 5, 2, tzinfo=UTC),
            )
    finally:
        outbox.close()

    assert result is not None
    assert result.status == CloudExperimentRefreshStatus.FAILED
    assert result.error_code == expected_code
    assert result.retryable is retryable
    assert _latest_refresh_state(tmp_path / "desktop.sqlite") == (
        "failed",
        expected_code.value,
        1 if retryable else 0,
    )


@pytest.mark.asyncio
async def test_periodic_refresh_secret_store_failure_is_bounded_without_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_secret(key: str) -> str | None:
        raise SecretStoreUnavailableError("token=secret response body")

    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        fail_get_secret,
    )
    outbox = _outbox(tmp_path)
    try:
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
        worker = CloudSyncWorker(outbox, config)
        transport = ScriptedTransport()
        async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
            result = await worker.refresh_experiments_if_due(
                client,
                now=datetime(2026, 5, 2, tzinfo=UTC),
            )
    finally:
        outbox.close()

    assert result is not None
    assert result.status == CloudExperimentRefreshStatus.FAILED
    assert result.error_code == CloudOperatorErrorCode.SECRET_STORE_UNAVAILABLE
    assert result.message == "Cloud sign-in is temporarily unavailable."
    assert "secret" not in result.message.lower()
    assert "token=" not in result.message
    assert "response body" not in result.message
    assert transport.requests == []
    assert _latest_refresh_state(tmp_path / "desktop.sqlite") == (
        "failed",
        "secret_store_unavailable",
        1,
    )


@pytest.mark.asyncio
async def test_periodic_refresh_verification_failure_preserves_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    monkeypatch.setattr(
        "services.desktop_app.cloud.experiment_bundle.BundleVerificationConfig.from_env",
        classmethod(
            lambda cls: BundleVerificationConfig(
                signature_mode="hmac-sha256",
                hmac_secret=BUNDLE_SECRET,
            )
        ),
    )
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            label TEXT,
            arm TEXT NOT NULL,
            greeting_text TEXT,
            alpha_param REAL NOT NULL DEFAULT 1.0,
            beta_param REAL NOT NULL DEFAULT 1.0,
            UNIQUE (experiment_id, arm)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO experiments (experiment_id, label, arm, greeting_text, alpha_param, beta_param)
        VALUES ('experiment-a', 'Experiment A', 'arm-a', 'Cached A', 9.0, 8.0)
        """
    )
    conn.close()
    outbox = _outbox(tmp_path)
    try:
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
        worker = CloudSyncWorker(outbox, config)
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(
                    200,
                    json=_bundle(signature="bad-signature").model_dump(mode="json"),
                ),
            ]
        )
        async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
            result = await worker.refresh_experiments_if_due(
                client,
                now=datetime(2026, 5, 2, tzinfo=UTC),
            )
    finally:
        outbox.close()

    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    try:
        row = conn.execute(
            """
            SELECT stimulus_definition, alpha_param, beta_param
            FROM experiments
            WHERE experiment_id = 'experiment-a' AND arm = 'arm-a'
            """
        ).fetchone()
    finally:
        conn.close()
    assert result is not None
    assert result.error_code == CloudOperatorErrorCode.SIGNATURE_FAILED
    assert row is not None
    assert json.loads(str(row[0])) == _migrated_cached_stimulus_definition("Cached A").model_dump(
        mode="json"
    )
    assert tuple(row[1:]) == (9.0, 8.0)
    assert _latest_refresh_state(tmp_path / "desktop.sqlite") == ("failed", "signature_failed", 0)


@pytest.mark.asyncio
async def test_sync_deletes_successful_segment_batch_even_when_insert_count_is_lower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        handoff = _handoff_payload(datetime(2026, 5, 2, tzinfo=UTC))
        payload_data = handoff.model_dump(mode="json", by_alias=True, exclude_none=True)
        payload_data["_stimulus_time"] = None
        upload_id = outbox.enqueue_raw(
            endpoint="telemetry_segments",
            payload_type="inference_handoff",
            dedupe_key=handoff.segment_id,
            payload_json=json.dumps(payload_data, sort_keys=True, separators=(",", ":")),
        )
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(
                    200, json={"status": "accepted", "accepted_count": 1, "inserted_count": 0}
                ),
            ]
        )

        seen = await _sync_once(outbox, transport)
    finally:
        outbox.close()

    assert seen is transport
    assert upload_id not in _row_statuses(tmp_path / "desktop.sqlite")
    assert [request.url.path for request in transport.requests] == [
        "/v4/auth/oauth/token",
        "/v4/telemetry/segments",
    ]
    body = json.loads(transport.requests[1].content)
    assert body["segments"][0]["segment_id"] == SEGMENT_ID


@pytest.mark.asyncio
async def test_sync_preserves_physiological_context_in_segment_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        outbox.enqueue_inference_handoff(
            _handoff_payload(datetime(2026, 5, 2, tzinfo=UTC), with_physiology=True)
        )
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(
                    200, json={"status": "accepted", "accepted_count": 1, "inserted_count": 1}
                ),
            ]
        )

        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    body = json.loads(transport.requests[1].content)
    context = body["segments"][0]["_physiological_context"]
    assert context["streamer"]["rmssd_ms"] == 42.0


@pytest.mark.asyncio
async def test_sync_posts_posterior_delta_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_posterior_delta(
            _posterior_delta(datetime(2026, 5, 2, tzinfo=UTC))
        )
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(
                    200, json={"status": "accepted", "accepted_count": 1, "inserted_count": 1}
                ),
            ]
        )

        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    assert upload_id not in _row_statuses(tmp_path / "desktop.sqlite")
    assert transport.requests[1].url.path == "/v4/telemetry/posterior_deltas"
    body = json.loads(transport.requests[1].content)
    assert body["deltas"][0]["segment_id"] == SEGMENT_ID


@pytest.mark.asyncio
async def test_sync_retries_once_after_401_with_refreshed_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    stored: dict[str, str] = {}
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.set_secret",
        lambda key, value: stored.setdefault(key, value),
    )
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_posterior_delta(
            _posterior_delta(datetime(2026, 5, 2, tzinfo=UTC))
        )
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(401, json={"detail": "expired"}),
                httpx.Response(
                    200,
                    json={
                        "access_token": "access-b",
                        "expires_in": 3600,
                        "refresh_token": "refresh-b",
                    },
                ),
                httpx.Response(
                    200, json={"status": "accepted", "accepted_count": 1, "inserted_count": 1}
                ),
            ]
        )

        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    assert upload_id not in _row_statuses(tmp_path / "desktop.sqlite")
    assert [request.url.path for request in transport.requests] == [
        "/v4/auth/oauth/token",
        "/v4/telemetry/posterior_deltas",
        "/v4/auth/oauth/token",
        "/v4/telemetry/posterior_deltas",
    ]
    assert transport.requests[1].headers["authorization"] == "Bearer access-a"
    assert transport.requests[3].headers["authorization"] == "Bearer access-b"
    assert stored == {"cloud_oauth_refresh_token": "refresh-b"}


@pytest.mark.asyncio
async def test_timeout_marks_valid_rows_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    monkeypatch.setattr("services.desktop_app.cloud.outbox.random.uniform", lambda _a, _b: 0.0)
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_posterior_delta(
            _posterior_delta(datetime(2026, 5, 2, tzinfo=UTC))
        )
        await _sync_once(outbox, TokenThenTimeoutTransport())
    finally:
        outbox.close()

    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    try:
        row = conn.execute(
            "SELECT status, attempt_count, last_error FROM pending_uploads WHERE upload_id = ?",
            (upload_id,),
        ).fetchone()
    finally:
        conn.close()
    assert tuple(row) == ("pending", 1, "cloud sync timeout")


@pytest.mark.asyncio
async def test_permanent_4xx_marks_rows_dead_letter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_posterior_delta(
            _posterior_delta(datetime(2026, 5, 2, tzinfo=UTC))
        )
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(422, json={"detail": "invalid"}),
            ]
        )
        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    assert _row_statuses(tmp_path / "desktop.sqlite") == {upload_id: "dead_letter"}


@pytest.mark.asyncio
async def test_same_second_segment_and_event_batch_drains_segment_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        handoff = _handoff_payload(datetime(2026, 5, 2, tzinfo=UTC))
        event = _attribution_event(datetime(2026, 5, 2, tzinfo=UTC))
        segment_payload = canonical_payload_json(handoff)
        event_payload = event.model_dump_json(by_alias=True)
        conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
        try:
            conn.execute(
                """
                INSERT INTO pending_uploads (
                    upload_id, endpoint, payload_type, dedupe_key, payload_json,
                    payload_sha256, created_at_utc, next_attempt_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "z-segment",
                    "telemetry_segments",
                    "inference_handoff",
                    SEGMENT_ID,
                    segment_payload,
                    None,
                    created_at,
                    created_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO pending_uploads (
                    upload_id, endpoint, payload_type, dedupe_key, payload_json,
                    payload_sha256, created_at_utc, next_attempt_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "a-event",
                    "telemetry_segments",
                    "attribution_event",
                    str(event.event_id),
                    event_payload,
                    None,
                    created_at,
                    created_at,
                ),
            )
        finally:
            conn.close()
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(
                    200,
                    json={"status": "accepted", "accepted_count": 1, "inserted_count": 1},
                ),
                httpx.Response(
                    200,
                    json={"status": "accepted", "accepted_count": 1, "inserted_count": 1},
                ),
            ]
        )
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=1)
        worker = CloudSyncWorker(outbox, config)
        async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
            await worker.sync_once(client)
    finally:
        outbox.close()

    assert [request.url.path for request in transport.requests] == [
        "/v4/auth/oauth/token",
        "/v4/telemetry/segments",
        "/v4/telemetry/segments",
    ]
    first_body = json.loads(transport.requests[1].content)
    second_body = json.loads(transport.requests[2].content)
    assert first_body["segments"][0]["segment_id"] == SEGMENT_ID
    assert first_body["attribution_events"] == []
    assert second_body["segments"] == []
    assert second_body["attribution_events"][0]["segment_id"] == SEGMENT_ID


@pytest.mark.asyncio
async def test_event_only_segment_batch_posts_attribution_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_attribution_event(
            _attribution_event(datetime(2026, 5, 2, tzinfo=UTC))
        )
        transport = ScriptedTransport(
            [
                httpx.Response(200, json={"access_token": "access-a", "expires_in": 3600}),
                httpx.Response(
                    200,
                    json={"status": "accepted", "accepted_count": 1, "inserted_count": 1},
                ),
            ]
        )
        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    assert upload_id not in _row_statuses(tmp_path / "desktop.sqlite")
    assert [request.url.path for request in transport.requests] == [
        "/v4/auth/oauth/token",
        "/v4/telemetry/segments",
    ]
    body = json.loads(transport.requests[1].content)
    assert body["segments"] == []
    assert len(body["attribution_events"]) == 1
    assert body["attribution_events"][0]["segment_id"] == SEGMENT_ID


@pytest.mark.asyncio
async def test_secret_store_failure_marks_rows_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_secret(key: str) -> str | None:
        raise SecretStoreUnavailableError("unavailable")

    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        fail_get_secret,
    )
    monkeypatch.setattr("services.desktop_app.cloud.outbox.random.uniform", lambda _a, _b: 0.0)
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_posterior_delta(
            _posterior_delta(datetime(2026, 5, 2, tzinfo=UTC))
        )
        transport = ScriptedTransport()
        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    try:
        row = conn.execute(
            "SELECT status, attempt_count, last_error FROM pending_uploads WHERE upload_id = ?",
            (upload_id,),
        ).fetchone()
    finally:
        conn.close()
    assert tuple(row) == ("pending", 1, "cloud sign-in temporarily unavailable")
    assert transport.requests == []


@pytest.mark.asyncio
async def test_validation_failure_is_dead_lettered_without_posting_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    outbox = _outbox(tmp_path)
    try:
        upload_id = outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="bad",
            payload_json=canonical_payload_json({"not": "valid"}),
        )
        transport = ScriptedTransport()
        await _sync_once(outbox, transport)
    finally:
        outbox.close()

    assert _row_statuses(tmp_path / "desktop.sqlite") == {upload_id: "dead_letter"}
    assert transport.requests == []


def test_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LSIE_CLOUD_BASE_URL", "https://cloud.example.test/")
    monkeypatch.setenv("LSIE_CLOUD_SYNC_INTERVAL_S", "1.25")
    monkeypatch.setenv("LSIE_CLOUD_SYNC_BATCH_SIZE", "7")
    monkeypatch.setenv("LSIE_CLOUD_SYNC_TIMEOUT_S", "3.5")
    monkeypatch.setenv("LSIE_CLOUD_CLIENT_ID", "desktop-a")

    config = CloudSyncConfig.from_env()

    assert config.base_url == "https://cloud.example.test"
    assert config.interval_s == 1.25
    assert config.batch_size == 7
    assert config.timeout_s == 3.5
    assert config.client_id == "desktop-a"


def test_batch_builder_does_not_log_payload_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outbox = _outbox(tmp_path)
    try:
        raw = canonical_payload_json(_posterior_delta(datetime(2026, 5, 2, tzinfo=UTC)))
        upload_id = outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json=raw,
        )
        upload = outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)[0]
        worker = CloudSyncWorker(outbox, CloudSyncConfig())

        batch = worker._build_batch("telemetry_posterior_deltas", [upload])
    finally:
        outbox.close()

    assert batch is not None
    assert batch[1] == [upload_id]
    assert raw not in caplog.text


def test_event_only_batch_builder_includes_attribution_events(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)
    try:
        event = _attribution_event(datetime(2026, 5, 2, tzinfo=UTC))
        upload = PendingUpload(
            upload_id="event-a",
            endpoint="telemetry_segments",
            payload_type="attribution_event",
            dedupe_key="event-a",
            payload_json=event.model_dump_json(by_alias=True),
            payload_sha256=None,
            payload_redacted_at_utc=None,
            created_at_utc="2026-05-02T00:00:00Z",
            next_attempt_at_utc="2026-05-02T00:00:00Z",
            attempt_count=0,
            locked_at_utc=None,
            last_error=None,
            status="pending",
        )
        worker = CloudSyncWorker(outbox, CloudSyncConfig())

        batch = worker._build_batch("telemetry_segments", [upload])
    finally:
        outbox.close()

    assert batch is not None
    payload, upload_ids = batch
    attribution_events = cast(list[dict[str, object]], payload["attribution_events"])
    assert upload_ids == ["event-a"]
    assert payload["segments"] == []
    assert len(attribution_events) == 1
    assert attribution_events[0]["segment_id"] == SEGMENT_ID
