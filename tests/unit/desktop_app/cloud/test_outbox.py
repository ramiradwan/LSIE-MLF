from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.schemas.cloud import PosteriorDelta
from packages.schemas.inference_handoff import InferenceHandoffPayload
from services.desktop_app.cloud.outbox import (
    CloudOutbox,
    OutboxDedupeConflictError,
    RedactedPayloadError,
    deterministic_upload_id,
    payload_sha256,
)
from services.desktop_app.state.sqlite_schema import bootstrap_schema

SEGMENT_ID = "a" * 64
DECISION_CONTEXT_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"


def _handoff_payload(*, with_physiology: bool = False) -> InferenceHandoffPayload:
    sample_timestamp = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    payload: dict[str, object] = {
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


def _posterior_delta() -> PosteriorDelta:
    return PosteriorDelta(
        experiment_id=101,
        arm_id="arm-a",
        delta_alpha=1.0,
        delta_beta=0.0,
        segment_id=SEGMENT_ID,
        client_id="desktop-a",
        event_id="00000000-0000-4000-8000-000000000001",
        applied_at_utc=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
        decision_context_hash=SEGMENT_ID,
    )


def test_outbox_preserves_real_physiological_context(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        upload_id = outbox.enqueue_inference_handoff(_handoff_payload(with_physiology=True))
        upload = outbox.fetch_ready_batch("telemetry_segments", limit=10)[0]
        model = outbox.validate_upload(upload)
    finally:
        outbox.close()

    stored = json.loads(upload.payload_json)
    assert upload.upload_id == upload_id
    assert stored["_physiological_context"]["streamer"]["rmssd_ms"] == 42.0
    assert isinstance(model, InferenceHandoffPayload)
    assert model.physiological_context is not None
    assert model.physiological_context.streamer is not None
    assert model.physiological_context.streamer.rmssd_ms == 42.0


def test_outbox_omits_absent_physiological_context_marker(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        outbox.enqueue_inference_handoff(_handoff_payload(with_physiology=False))
        upload = outbox.fetch_ready_batch("telemetry_segments", limit=10)[0]
    finally:
        outbox.close()

    stored = json.loads(upload.payload_json)
    assert "_stimulus_time" in stored
    assert "_physiological_context" not in stored


def test_outbox_enqueue_dedupes_by_payload_type_and_dedupe_key(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        first = outbox.enqueue_posterior_delta(_posterior_delta())
        second = outbox.enqueue_posterior_delta(_posterior_delta())
        batch = outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)
    finally:
        outbox.close()

    assert first == second
    assert len(batch) == 1
    assert batch[0].payload_type == "posterior_delta"
    assert batch[0].status == "pending"


def test_outbox_rejects_changed_payload_for_existing_dedupe_key(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json='{"value":1}',
        )
        with pytest.raises(OutboxDedupeConflictError):
            outbox.enqueue_raw(
                endpoint="telemetry_posterior_deltas",
                payload_type="posterior_delta",
                dedupe_key="delta-a",
                payload_json='{"value":2}',
            )
    finally:
        outbox.close()


def test_fetch_ready_batch_locks_rows_and_delete_success_removes_them(tmp_path: Path) -> None:
    db = tmp_path / "desktop.sqlite"
    outbox = CloudOutbox(db)
    try:
        upload_id = outbox.enqueue_posterior_delta(_posterior_delta())
        batch = outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)
        assert [row.upload_id for row in batch] == [upload_id]
        assert outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10) == []
        outbox.delete_uploads([upload_id])
        assert outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10) == []
    finally:
        outbox.close()

    conn = sqlite3.connect(str(db), isolation_level=None)
    rows = conn.execute("SELECT COUNT(*) FROM pending_uploads").fetchone()[0]
    conn.close()
    assert rows == 0


def test_mark_retry_reschedules_locked_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.desktop_app.cloud.outbox.random.uniform", lambda _a, _b: 0.0)
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        upload_id = outbox.enqueue_posterior_delta(_posterior_delta())
        outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)
        outbox.mark_retry(
            [upload_id],
            error="HTTP 503",
            now=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
            base_delay_s=2.0,
        )
        row = outbox.fetch_ready_batch(
            "telemetry_posterior_deltas",
            limit=10,
            now_utc="2026-05-02T12:00:01Z",
        )
        assert row == []
        row = outbox.fetch_ready_batch(
            "telemetry_posterior_deltas",
            limit=10,
            now_utc="2026-05-02T12:00:02Z",
        )
    finally:
        outbox.close()

    assert len(row) == 1
    assert row[0].attempt_count == 1
    assert row[0].last_error == "HTTP 503"


def test_invalid_payload_is_marked_dead_letter(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="bad",
            payload_json='{"not":"valid"}',
        )
        upload = outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)[0]
        with pytest.raises(ValidationError):
            outbox.validate_upload(upload)
        conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
        row = conn.execute(
            """
            SELECT status, payload_redacted_at_utc, payload_json, payload_sha256
            FROM pending_uploads
            WHERE upload_id = ?
            """,
            (upload.upload_id,),
        ).fetchone()
        conn.close()
    finally:
        outbox.close()

    status, redacted_at, payload_json, digest = row
    summary = json.loads(payload_json)
    assert status == "dead_letter"
    assert redacted_at is not None
    assert summary["_redacted"] is True
    assert summary["replay_metadata"]["segment_id"] is None
    assert digest == payload_sha256('{"not":"valid"}')


def test_reset_stale_locks_returns_inflight_rows_to_pending(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        outbox.enqueue_posterior_delta(_posterior_delta())
        outbox.fetch_ready_batch("telemetry_posterior_deltas", limit=10)
        assert outbox.reset_stale_locks(before_utc="9999-12-31T23:59:59Z") == 1
        rows = outbox.fetch_ready_batch(
            "telemetry_posterior_deltas",
            limit=10,
            now_utc="9999-12-31T23:59:59Z",
        )
    finally:
        outbox.close()

    assert len(rows) == 1


def test_apply_retention_policy_redacts_old_payloads(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        payload = '{"segment_id":"seg-a","event_id":"event-a","client_id":"desktop-a"}'
        upload_id = outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json=payload,
            created_at_utc="2026-05-01T00:00:00Z",
        )
        redacted = outbox.apply_retention_policy(now=datetime(2026, 5, 2, 12, 0, tzinfo=UTC))
        conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
        row = conn.execute(
            """
            SELECT payload_json, payload_sha256, payload_redacted_at_utc
            FROM pending_uploads
            WHERE upload_id = ?
            """,
            (upload_id,),
        ).fetchone()
        conn.close()
    finally:
        outbox.close()

    payload_json, digest, redacted_at = row
    summary = json.loads(payload_json)
    assert redacted == 1
    assert digest == payload_sha256(payload)
    assert redacted_at == "2026-05-02T12:00:00Z"
    assert summary["_redacted"] is True
    assert summary["replay_metadata"] == {
        "segment_id": "seg-a",
        "client_id": "desktop-a",
        "event_ids": ["event-a"],
        "requires_segment_replay": True,
    }


def test_validate_upload_rejects_redacted_payload(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json='{"segment_id":"seg-a","event_id":"event-a","client_id":"desktop-a"}',
            created_at_utc="2026-05-01T00:00:00Z",
        )
        outbox.apply_retention_policy(now=datetime(2026, 5, 2, 12, 0, tzinfo=UTC))
        upload = outbox.fetch_ready_batch(
            "telemetry_posterior_deltas",
            limit=10,
            now_utc="2026-05-02T12:00:00Z",
        )[0]
        with pytest.raises(RedactedPayloadError, match="has been redacted"):
            outbox.validate_upload(upload)
    finally:
        outbox.close()


def test_outbox_dedupe_accepts_matching_payload_after_redaction(tmp_path: Path) -> None:
    outbox = CloudOutbox(tmp_path / "desktop.sqlite")
    try:
        payload = '{"segment_id":"seg-a","event_id":"event-a","client_id":"desktop-a"}'
        first = outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json=payload,
            created_at_utc="2026-05-01T00:00:00Z",
        )
        outbox.apply_retention_policy(now=datetime(2026, 5, 2, 12, 0, tzinfo=UTC))
        second = outbox.enqueue_raw(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key="delta-a",
            payload_json=payload,
        )
    finally:
        outbox.close()

    assert second == first


def test_bootstrap_exposes_pending_uploads_to_sqlite_writer(tmp_path: Path) -> None:
    from services.desktop_app.state.sqlite_writer import SqliteWriter

    writer = SqliteWriter(tmp_path / "desktop.sqlite")
    try:
        assert "pending_uploads" in writer.stats.columns_per_table
    finally:
        writer.close()


def test_deterministic_upload_id_is_stable() -> None:
    first = deterministic_upload_id("telemetry_segments", "inference_handoff", "seg-a", "{}")
    second = deterministic_upload_id("telemetry_segments", "inference_handoff", "seg-a", "{}")
    different = deterministic_upload_id("telemetry_segments", "attribution_event", "seg-a", "{}")

    assert first == second
    assert first != different


def test_pending_uploads_schema_is_idempotent(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "desktop.sqlite"), isolation_level=None)
    bootstrap_schema(conn)
    bootstrap_schema(conn)
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(pending_uploads)").fetchall()}
    conn.close()

    assert "idx_pending_uploads_ready" in indexes
    assert "idx_pending_uploads_lock" in indexes
