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
from packages.schemas.cloud import PosteriorDelta
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.cloud.outbox import CloudOutbox, PendingUpload, canonical_payload_json
from services.desktop_app.os_adapter import SecretStoreUnavailableError
from services.desktop_app.processes.cloud_sync_worker import CloudSyncConfig, CloudSyncWorker

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"


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
        "expected_greeting": "Say hello to the creator",
        "decision_context_hash": DECISION_CONTEXT_HASH,
        "random_seed": 42,
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
        "_expected_greeting": "Say hello to the creator",
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
        event_type="greeting_interaction",
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


async def _sync_once(
    outbox: CloudOutbox, transport: httpx.AsyncBaseTransport
) -> ScriptedTransport | None:
    config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
    worker = CloudSyncWorker(outbox, config)
    async with httpx.AsyncClient(base_url=config.base_url, transport=transport) as client:
        await worker.sync_once(client)
    return transport if isinstance(transport, ScriptedTransport) else None


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
    assert tuple(row) == ("pending", 1, "cloud secret store unavailable")
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
