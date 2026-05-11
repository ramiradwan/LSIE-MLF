from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import httpx
import pytest

from packages.schemas.attribution import AttributionEvent
from packages.schemas.cloud import PosteriorDelta
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.cloud.outbox import CloudOutbox
from services.desktop_app.processes.cloud_sync_worker import CloudSyncConfig, CloudSyncWorker

DECISION_CONTEXT_HASH: Final[str] = "b" * 64
CLIENT_ID: Final[str] = "desktop-a"


class OfflineReplaySink(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.segment_keys: set[tuple[str, str]] = set()
        self.attribution_event_keys: set[str] = set()
        self.delta_keys: set[tuple[str, str, str]] = set()
        self.telemetry_posts = 0
        self.delta_posts = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v4/auth/oauth/token":
            return httpx.Response(
                200,
                json={"access_token": "access-a", "expires_in": 3600},
                request=request,
            )
        if request.url.path == "/v4/telemetry/segments":
            self.telemetry_posts += 1
            body = json.loads(request.content)
            accepted = len(body["segments"]) + len(body["attribution_events"])
            inserted = 0
            for segment in body["segments"]:
                key = (str(segment["session_id"]), str(segment["segment_id"]))
                if key not in self.segment_keys:
                    self.segment_keys.add(key)
                    inserted += 1
            for event in body["attribution_events"]:
                event_key = str(event["event_id"])
                if event_key not in self.attribution_event_keys:
                    self.attribution_event_keys.add(event_key)
                    inserted += 1
            return httpx.Response(
                200,
                json={
                    "status": "accepted",
                    "accepted_count": accepted,
                    "inserted_count": inserted,
                },
                request=request,
            )
        if request.url.path == "/v4/telemetry/posterior_deltas":
            self.delta_posts += 1
            body = json.loads(request.content)
            accepted = len(body["deltas"])
            inserted = 0
            for delta in body["deltas"]:
                delta_key = (
                    str(delta["segment_id"]),
                    str(delta["client_id"]),
                    str(delta["arm_id"]),
                )
                if delta_key not in self.delta_keys:
                    self.delta_keys.add(delta_key)
                    inserted += 1
            return httpx.Response(
                200,
                json={
                    "status": "accepted",
                    "accepted_count": accepted,
                    "inserted_count": inserted,
                },
                request=request,
            )
        return httpx.Response(404, request=request)


class CrashAfterTelemetrySink(OfflineReplaySink):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        if request.url.path == "/v4/telemetry/segments":
            raise RuntimeError("process crashed after server accepted telemetry")
        return response


def _session_id(session_index: int) -> str:
    return f"00000000-0000-4000-8000-{session_index + 1:012x}"


def _handoff_payload(
    session_index: int,
    segment_index: int,
    sample_timestamp: datetime,
) -> InferenceHandoffPayload:
    segment_id = f"{session_index * 10_000 + segment_index:064x}"
    return InferenceHandoffPayload.model_validate(
        {
            "session_id": _session_id(session_index),
            "segment_id": segment_id,
            "segment_window_start_utc": sample_timestamp,
            "segment_window_end_utc": sample_timestamp + timedelta(seconds=30),
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
            "_bandit_decision_snapshot": {
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
            },
        }
    )


def _posterior_delta(
    session_index: int,
    segment_index: int,
    sample_timestamp: datetime,
) -> PosteriorDelta:
    arm_id = f"arm-{segment_index % 5}"
    segment_id = f"{session_index * 10_000 + segment_index:064x}"
    return PosteriorDelta(
        experiment_id=101,
        arm_id=arm_id,
        delta_alpha=1.0,
        delta_beta=0.0,
        segment_id=segment_id,
        client_id=CLIENT_ID,
        event_id=uuid.UUID(
            f"00000000-0000-4000-8000-{session_index * 10_000 + segment_index:012x}"
        ),
        applied_at_utc=sample_timestamp,
        decision_context_hash=DECISION_CONTEXT_HASH,
    )


def _attribution_event(
    session_index: int,
    segment_index: int,
    sample_timestamp: datetime,
) -> AttributionEvent:
    segment_id = f"{session_index * 10_000 + segment_index:064x}"
    return AttributionEvent(
        event_id=uuid.UUID(
            f"10000000-0000-4000-8000-{session_index * 10_000 + segment_index:012x}"
        ),
        session_id=uuid.UUID(_session_id(session_index)),
        segment_id=segment_id,
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
        bandit_decision_snapshot=_handoff_payload(
            session_index,
            segment_index,
            sample_timestamp,
        ).bandit_decision_snapshot,
        evidence_flags=[],
        finality="online_provisional",
        schema_version="v1",
        created_at=sample_timestamp,
    )


def _pending_upload_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        row = conn.execute("SELECT COUNT(*) FROM pending_uploads").fetchone()
    finally:
        conn.close()
    return int(row[0])


@pytest.mark.asyncio
async def test_offline_replay_drains_exactly_once_after_duplicate_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    db_path = tmp_path / "desktop.sqlite"
    outbox = CloudOutbox(db_path)
    try:
        base_timestamp = datetime(2026, 5, 2, tzinfo=UTC)
        for session_index in range(50):
            for segment_index in range(10):
                timestamp = base_timestamp + timedelta(
                    seconds=(session_index * 10 + segment_index) * 30
                )
                outbox.enqueue_inference_handoff(
                    _handoff_payload(session_index, segment_index, timestamp)
                )
                outbox.enqueue_inference_handoff(
                    _handoff_payload(session_index, segment_index, timestamp)
                )
                if segment_index % 3 == 0:
                    outbox.enqueue_attribution_event(
                        _attribution_event(session_index, segment_index, timestamp)
                    )
                    outbox.enqueue_attribution_event(
                        _attribution_event(session_index, segment_index, timestamp)
                    )
                if segment_index % 2 == 0:
                    outbox.enqueue_posterior_delta(
                        _posterior_delta(session_index, segment_index, timestamp)
                    )
                    outbox.enqueue_posterior_delta(
                        _posterior_delta(session_index, segment_index, timestamp)
                    )

        sink = OfflineReplaySink()
        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=37)
        worker = CloudSyncWorker(outbox, config)
        async with httpx.AsyncClient(base_url=config.base_url, transport=sink) as client:
            await worker.sync_once(client)
            await worker.sync_once(client)
    finally:
        outbox.close()

    assert _pending_upload_count(db_path) == 0
    assert len(sink.segment_keys) == 500
    assert len(sink.attribution_event_keys) == 200
    assert len(sink.delta_keys) == 250
    assert sink.telemetry_posts == 19
    assert sink.delta_posts == 7


@pytest.mark.asyncio
async def test_offline_replay_is_idempotent_after_crash_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.desktop_app.processes.cloud_sync_worker.get_secret",
        lambda key: "refresh-a",
    )
    monkeypatch.setattr("services.desktop_app.cloud.outbox.random.uniform", lambda _a, _b: 0.0)
    db_path = tmp_path / "desktop.sqlite"
    outbox = CloudOutbox(db_path)
    try:
        timestamp = datetime(2026, 5, 2, tzinfo=UTC)
        for segment_index in range(3):
            outbox.enqueue_inference_handoff(_handoff_payload(0, segment_index, timestamp))
        outbox.enqueue_attribution_event(_attribution_event(0, 0, timestamp))

        config = CloudSyncConfig(base_url="https://cloud.example.test", batch_size=10)
        crashy_sink = CrashAfterTelemetrySink()
        worker = CloudSyncWorker(outbox, config)
        async with httpx.AsyncClient(base_url=config.base_url, transport=crashy_sink) as client:
            with pytest.raises(RuntimeError, match="process crashed"):
                await worker.sync_once(client)

        assert _pending_upload_count(db_path) == 4
        outbox.reset_stale_locks(before_utc="9999-12-31T23:59:59Z")
        recovery_sink = OfflineReplaySink()
        recovery_sink.segment_keys.update(crashy_sink.segment_keys)
        recovery_sink.attribution_event_keys.update(crashy_sink.attribution_event_keys)
        recovery_worker = CloudSyncWorker(outbox, config)
        async with httpx.AsyncClient(base_url=config.base_url, transport=recovery_sink) as client:
            await recovery_worker.sync_once(client)
    finally:
        outbox.close()

    assert _pending_upload_count(db_path) == 0
    assert len(recovery_sink.segment_keys) == 3
    assert len(recovery_sink.attribution_event_keys) == 1
    assert crashy_sink.telemetry_posts == 1
    assert recovery_sink.telemetry_posts == 1
