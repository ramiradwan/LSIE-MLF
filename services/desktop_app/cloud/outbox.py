from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from packages.schemas.attribution import AttributionEvent
from packages.schemas.cloud import PosteriorDelta
from packages.schemas.inference_handoff import InferenceHandoffPayload
from packages.schemas.operator_console import CloudOutboxSummary, ExperimentBundleRefreshResult
from services.desktop_app.state.sqlite_schema import bootstrap_schema

UploadEndpoint = Literal["telemetry_segments", "telemetry_posterior_deltas"]
PayloadType = Literal["inference_handoff", "attribution_event", "posterior_delta"]
UploadStatus = Literal["pending", "in_flight", "dead_letter"]

REDACT_AFTER_ATTEMPTS = 5
REDACT_AFTER_AGE = timedelta(days=1)
REDACTED_ENVELOPE_VERSION = 1


class OutboxDedupeConflictError(RuntimeError):
    pass


_READY_BATCH_SQL = """
    SELECT upload_id, endpoint, payload_type, dedupe_key, payload_json,
           payload_sha256, payload_redacted_at_utc,
           created_at_utc, next_attempt_at_utc, attempt_count, locked_at_utc,
           last_error, status
    FROM pending_uploads
    WHERE endpoint = ?
      AND status = 'pending'
      AND next_attempt_at_utc <= ?
    ORDER BY created_at_utc, upload_id
    LIMIT ?
"""


@dataclass(frozen=True)
class PendingUpload:
    upload_id: str
    endpoint: UploadEndpoint
    payload_type: PayloadType
    dedupe_key: str
    payload_json: str
    payload_sha256: str | None
    payload_redacted_at_utc: str | None
    created_at_utc: str
    next_attempt_at_utc: str
    attempt_count: int
    locked_at_utc: str | None
    last_error: str | None
    status: UploadStatus


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def payload_sha256(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def canonical_payload_json(payload: object) -> str:
    if isinstance(payload, InferenceHandoffPayload):
        data = payload.model_dump(mode="json", by_alias=True)
        if data.get("_physiological_context") is None:
            data.pop("_physiological_context", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))
    if isinstance(payload, BaseModel):
        return json.dumps(
            payload.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def deterministic_upload_id(
    endpoint: UploadEndpoint,
    payload_type: PayloadType,
    dedupe_key: str,
    payload_json: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(endpoint.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(dedupe_key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(payload_json.encode("utf-8"))
    return digest.hexdigest()


class CloudOutbox:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        bootstrap_schema(self._conn)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        self._conn.close()

    def enqueue_inference_handoff(self, payload: InferenceHandoffPayload) -> str:
        return self._enqueue_model(
            endpoint="telemetry_segments",
            payload_type="inference_handoff",
            dedupe_key=payload.segment_id,
            payload=payload,
        )

    def enqueue_attribution_event(self, payload: AttributionEvent) -> str:
        return self._enqueue_model(
            endpoint="telemetry_segments",
            payload_type="attribution_event",
            dedupe_key=str(payload.event_id),
            payload=payload,
        )

    def enqueue_posterior_delta(self, payload: PosteriorDelta) -> str:
        return self._enqueue_model(
            endpoint="telemetry_posterior_deltas",
            payload_type="posterior_delta",
            dedupe_key=f"{payload.segment_id}:{payload.client_id}:{payload.arm_id}",
            payload=payload,
        )

    def enqueue_raw(
        self,
        *,
        endpoint: UploadEndpoint,
        payload_type: PayloadType,
        dedupe_key: str,
        payload_json: str,
        created_at_utc: str | None = None,
    ) -> str:
        upload_id = deterministic_upload_id(endpoint, payload_type, dedupe_key, payload_json)
        created = created_at_utc or utc_now_iso()
        digest = payload_sha256(payload_json)
        try:
            self._conn.execute(
                """
                INSERT INTO pending_uploads (
                    upload_id, endpoint, payload_type, dedupe_key, payload_json,
                    payload_sha256, created_at_utc, next_attempt_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    endpoint,
                    payload_type,
                    dedupe_key,
                    payload_json,
                    digest,
                    created,
                    created,
                ),
            )
        except sqlite3.IntegrityError as exc:
            row = self._conn.execute(
                """
                SELECT upload_id, payload_json, payload_sha256
                FROM pending_uploads
                WHERE payload_type = ? AND dedupe_key = ?
                """,
                (payload_type, dedupe_key),
            ).fetchone()
            if row is not None:
                stored_payload_json = str(row["payload_json"])
                stored_digest = row["payload_sha256"] or payload_sha256(stored_payload_json)
                if stored_payload_json == payload_json or str(stored_digest) == digest:
                    return str(row["upload_id"])
                if _is_attribution_finality_promotion(
                    payload_type,
                    stored_payload_json,
                    payload_json,
                ):
                    self._conn.execute(
                        """
                        UPDATE pending_uploads
                        SET upload_id = ?, endpoint = ?, payload_json = ?, payload_sha256 = ?,
                            payload_redacted_at_utc = NULL, created_at_utc = ?,
                            next_attempt_at_utc = ?, attempt_count = 0, status = 'pending',
                            locked_at_utc = NULL, last_error = NULL
                        WHERE payload_type = ? AND dedupe_key = ?
                        """,
                        (
                            upload_id,
                            endpoint,
                            payload_json,
                            digest,
                            created,
                            created,
                            payload_type,
                            dedupe_key,
                        ),
                    )
                    return upload_id
            raise OutboxDedupeConflictError(
                f"payload changed for {payload_type} dedupe key {dedupe_key!r}"
            ) from exc
        return upload_id

    def fetch_ready_batch(
        self,
        endpoint: UploadEndpoint,
        *,
        limit: int,
        now_utc: str | None = None,
    ) -> list[PendingUpload]:
        if limit <= 0:
            return []
        now = now_utc or utc_now_iso()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self._conn.execute(_READY_BATCH_SQL, (endpoint, now, limit)).fetchall()
            upload_ids = [row["upload_id"] for row in rows]
            if upload_ids:
                placeholders = ",".join("?" for _ in upload_ids)
                self._conn.execute(
                    f"""
                    UPDATE pending_uploads
                    SET status = 'in_flight', locked_at_utc = ?
                    WHERE upload_id IN ({placeholders})
                    """,
                    [now, *upload_ids],
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return [_row_to_pending_upload(row) for row in rows]

    def delete_uploads(self, upload_ids: list[str]) -> None:
        if not upload_ids:
            return
        placeholders = ",".join("?" for _ in upload_ids)
        self._conn.execute(
            f"DELETE FROM pending_uploads WHERE upload_id IN ({placeholders})",
            upload_ids,
        )

    def mark_retry(
        self,
        upload_ids: list[str],
        *,
        error: str,
        now: datetime | None = None,
        base_delay_s: float = 1.0,
        max_delay_s: float = 300.0,
    ) -> None:
        if not upload_ids:
            return
        current = now or datetime.now(UTC)
        for upload_id in upload_ids:
            row = self._conn.execute(
                "SELECT attempt_count FROM pending_uploads WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if row is None:
                continue
            next_attempt_count = int(row["attempt_count"]) + 1
            delay = min(max_delay_s, base_delay_s * (2 ** (next_attempt_count - 1)))
            delay += random.uniform(0.0, min(1.0, delay * 0.1))
            next_attempt = current + timedelta(seconds=delay)
            self._conn.execute(
                """
                UPDATE pending_uploads
                SET status = 'pending', locked_at_utc = NULL, attempt_count = ?,
                    next_attempt_at_utc = ?, last_error = ?
                WHERE upload_id = ?
                """,
                (next_attempt_count, _format_dt(next_attempt), _sanitize_error(error), upload_id),
            )

    def mark_dead_letter(self, upload_ids: list[str], *, error: str) -> None:
        if not upload_ids:
            return
        placeholders = ",".join("?" for _ in upload_ids)
        self._conn.execute(
            f"""
            UPDATE pending_uploads
            SET status = 'dead_letter', locked_at_utc = NULL, last_error = ?
            WHERE upload_id IN ({placeholders})
            """,
            [_sanitize_error(error), *upload_ids],
        )
        self._redact_uploads(upload_ids, redacted_at_utc=utc_now_iso())

    def reset_stale_locks(self, *, before_utc: str | None = None) -> int:
        before = before_utc or utc_now_iso()
        cursor = self._conn.execute(
            """
            UPDATE pending_uploads
            SET status = 'pending', locked_at_utc = NULL
            WHERE status = 'in_flight'
              AND locked_at_utc <= ?
            """,
            (before,),
        )
        return int(cursor.rowcount)

    def summarize(self, *, now: datetime | None = None) -> CloudOutboxSummary:
        rows = self._conn.execute(
            """
            SELECT status, next_attempt_at_utc, payload_redacted_at_utc, last_error
            FROM pending_uploads
            """
        ).fetchall()
        pending_count = 0
        in_flight_count = 0
        dead_letter_count = 0
        retry_scheduled_count = 0
        redacted_count = 0
        earliest_next_attempt: datetime | None = None
        last_error: str | None = None
        current = now or datetime.now(UTC)
        for row in rows:
            status = str(row["status"])
            redacted = row["payload_redacted_at_utc"] is not None
            if redacted:
                redacted_count += 1
            if status == "dead_letter":
                dead_letter_count += 1
            elif not redacted and status == "pending":
                pending_count += 1
            elif not redacted and status == "in_flight":
                in_flight_count += 1
            next_attempt = _parse_dt(str(row["next_attempt_at_utc"]))
            if not redacted and status == "pending" and next_attempt > current:
                retry_scheduled_count += 1
            if not redacted and (
                earliest_next_attempt is None or next_attempt < earliest_next_attempt
            ):
                earliest_next_attempt = next_attempt
            if row["last_error"] is not None:
                last_error = str(row["last_error"])
        return CloudOutboxSummary(
            generated_at_utc=current,
            pending_count=pending_count,
            in_flight_count=in_flight_count,
            dead_letter_count=dead_letter_count,
            retry_scheduled_count=retry_scheduled_count,
            redacted_count=redacted_count,
            earliest_next_attempt_utc=earliest_next_attempt,
            last_error=last_error,
            latest_experiment_refresh=self.latest_experiment_refresh(),
        )

    def latest_experiment_refresh(self) -> ExperimentBundleRefreshResult | None:
        row = self._conn.execute(
            """
            SELECT status, completed_at_utc, message, bundle_id,
                   experiment_count, error_code, retryable
            FROM cloud_experiment_refresh_state
            WHERE state_key = 'latest'
            """
        ).fetchone()
        if row is None:
            return None
        return ExperimentBundleRefreshResult(
            status=row["status"],
            completed_at_utc=_parse_dt(str(row["completed_at_utc"])),
            message=str(row["message"]),
            bundle_id=row["bundle_id"],
            experiment_count=int(row["experiment_count"]),
            error_code=row["error_code"],
            retryable=bool(row["retryable"]),
        )

    def record_experiment_refresh_result(self, result: ExperimentBundleRefreshResult) -> None:
        self._conn.execute(
            """
            INSERT INTO cloud_experiment_refresh_state (
                state_key, status, completed_at_utc, message, bundle_id,
                experiment_count, error_code, retryable
            ) VALUES ('latest', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                status = excluded.status,
                completed_at_utc = excluded.completed_at_utc,
                message = excluded.message,
                bundle_id = excluded.bundle_id,
                experiment_count = excluded.experiment_count,
                error_code = excluded.error_code,
                retryable = excluded.retryable
            """,
            (
                result.status.value,
                _format_dt(result.completed_at_utc),
                result.message,
                result.bundle_id,
                result.experiment_count,
                result.error_code.value if result.error_code is not None else None,
                1 if result.retryable else 0,
            ),
        )

    def apply_retention_policy(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        redacted_ids: list[str] = []
        rows = self._conn.execute(
            """
            SELECT upload_id, created_at_utc, attempt_count
            FROM pending_uploads
            WHERE payload_redacted_at_utc IS NULL
            """
        ).fetchall()
        for row in rows:
            upload_id = str(row["upload_id"])
            created_at = _parse_dt(str(row["created_at_utc"]))
            attempt_count = int(row["attempt_count"])
            if attempt_count >= REDACT_AFTER_ATTEMPTS or current - created_at >= REDACT_AFTER_AGE:
                redacted_ids.append(upload_id)
        if not redacted_ids:
            return 0
        self._redact_uploads(redacted_ids, redacted_at_utc=_format_dt(current))
        return len(redacted_ids)

    def validate_upload(self, upload: PendingUpload) -> object:
        if _is_redacted_payload(upload.payload_json):
            raise RedactedPayloadError(
                f"upload {upload.upload_id} payload has been redacted and cannot be replayed"
            )
        try:
            if upload.payload_type == "inference_handoff":
                return InferenceHandoffPayload.model_validate_json(upload.payload_json)
            if upload.payload_type == "attribution_event":
                return AttributionEvent.model_validate_json(upload.payload_json)
            if upload.payload_type == "posterior_delta":
                return PosteriorDelta.model_validate_json(upload.payload_json)
        except ValidationError as exc:
            self.mark_dead_letter([upload.upload_id], error=str(exc).splitlines()[0])
            raise
        raise ValueError(f"unsupported payload_type {upload.payload_type!r}")

    def _enqueue_model(
        self,
        *,
        endpoint: UploadEndpoint,
        payload_type: PayloadType,
        dedupe_key: str,
        payload: object,
    ) -> str:
        return self.enqueue_raw(
            endpoint=endpoint,
            payload_type=payload_type,
            dedupe_key=dedupe_key,
            payload_json=canonical_payload_json(payload),
        )

    def _redact_uploads(self, upload_ids: list[str], *, redacted_at_utc: str) -> None:
        if not upload_ids:
            return
        placeholders = ",".join("?" for _ in upload_ids)
        rows = self._conn.execute(
            f"""
            SELECT upload_id, endpoint, payload_type, dedupe_key, payload_json,
                   payload_sha256, created_at_utc, next_attempt_at_utc,
                   attempt_count, locked_at_utc, last_error, status
            FROM pending_uploads
            WHERE upload_id IN ({placeholders})
            """,
            upload_ids,
        ).fetchall()
        for row in rows:
            original_payload_json = str(row["payload_json"])
            if _is_redacted_payload(original_payload_json):
                continue
            digest = row["payload_sha256"] or payload_sha256(original_payload_json)
            summary = json.dumps(
                _redacted_payload_summary(
                    upload_id=str(row["upload_id"]),
                    endpoint=str(row["endpoint"]),
                    payload_type=str(row["payload_type"]),
                    dedupe_key=str(row["dedupe_key"]),
                    payload_sha256=str(digest),
                    created_at_utc=str(row["created_at_utc"]),
                    next_attempt_at_utc=str(row["next_attempt_at_utc"]),
                    attempt_count=int(row["attempt_count"]),
                    locked_at_utc=row["locked_at_utc"],
                    last_error=row["last_error"],
                    status=str(row["status"]),
                    payload_json=original_payload_json,
                    redacted_at_utc=redacted_at_utc,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            self._conn.execute(
                """
                UPDATE pending_uploads
                SET payload_json = ?, payload_sha256 = ?, payload_redacted_at_utc = ?
                WHERE upload_id = ?
                """,
                (summary, digest, redacted_at_utc, str(row["upload_id"])),
            )


class RedactedPayloadError(RuntimeError):
    pass


def _row_to_pending_upload(row: sqlite3.Row) -> PendingUpload:
    return PendingUpload(
        upload_id=str(row["upload_id"]),
        endpoint=row["endpoint"],
        payload_type=row["payload_type"],
        dedupe_key=str(row["dedupe_key"]),
        payload_json=str(row["payload_json"]),
        payload_sha256=row["payload_sha256"],
        payload_redacted_at_utc=row["payload_redacted_at_utc"],
        created_at_utc=str(row["created_at_utc"]),
        next_attempt_at_utc=str(row["next_attempt_at_utc"]),
        attempt_count=int(row["attempt_count"]),
        locked_at_utc=row["locked_at_utc"],
        last_error=row["last_error"],
        status=row["status"],
    )


def _format_dt(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _sanitize_error(error: str) -> str:
    return error[:500]


def _is_redacted_payload(payload_json: str) -> bool:
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError:
        return False
    return bool(data.get("_redacted")) if isinstance(data, dict) else False


def _is_attribution_finality_promotion(
    payload_type: PayloadType,
    stored_payload_json: str,
    payload_json: str,
) -> bool:
    if payload_type != "attribution_event":
        return False
    try:
        stored = AttributionEvent.model_validate_json(stored_payload_json)
        incoming = AttributionEvent.model_validate_json(payload_json)
    except ValidationError:
        return False
    if stored.event_id != incoming.event_id:
        return False
    stored_data = stored.model_dump(mode="json")
    incoming_data = incoming.model_dump(mode="json")
    stored_data["finality"] = incoming_data["finality"]
    return (
        stored.finality == "online_provisional"
        and incoming.finality == "offline_final"
        and stored_data == incoming_data
    )


def _redacted_payload_summary(
    *,
    upload_id: str,
    endpoint: str,
    payload_type: str,
    dedupe_key: str,
    payload_sha256: str,
    created_at_utc: str,
    next_attempt_at_utc: str,
    attempt_count: int,
    locked_at_utc: str | None,
    last_error: str | None,
    status: str,
    payload_json: str,
    redacted_at_utc: str,
) -> dict[str, object]:
    return {
        "_redacted": True,
        "schema_version": REDACTED_ENVELOPE_VERSION,
        "upload_id": upload_id,
        "endpoint": endpoint,
        "payload_type": payload_type,
        "dedupe_key": dedupe_key,
        "payload_sha256": payload_sha256,
        "payload_redacted_at_utc": redacted_at_utc,
        "created_at_utc": created_at_utc,
        "next_attempt_at_utc": next_attempt_at_utc,
        "attempt_count": attempt_count,
        "locked_at_utc": locked_at_utc,
        "last_error": last_error,
        "status": status,
        "replay_metadata": _replay_metadata(payload_type, payload_json),
    }


def _replay_metadata(payload_type: str, payload_json: str) -> dict[str, object]:
    try:
        data = json.loads(payload_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    if payload_type == "inference_handoff":
        return {
            "segment_id": data.get("segment_id"),
            "session_id": data.get("session_id"),
            "event_ids": [],
            "requires_segment_replay": True,
        }
    if payload_type == "attribution_event":
        event_id = data.get("event_id")
        return {
            "segment_id": data.get("segment_id"),
            "session_id": data.get("session_id"),
            "event_ids": [event_id] if event_id is not None else [],
            "requires_segment_replay": False,
        }
    if payload_type == "posterior_delta":
        event_id = data.get("event_id")
        return {
            "segment_id": data.get("segment_id"),
            "client_id": data.get("client_id"),
            "event_ids": [event_id] if event_id is not None else [],
            "requires_segment_replay": True,
        }
    return {}
