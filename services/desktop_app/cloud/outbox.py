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
from services.desktop_app.state.sqlite_schema import bootstrap_schema

UploadEndpoint = Literal["telemetry_segments", "telemetry_posterior_deltas"]
PayloadType = Literal["inference_handoff", "attribution_event", "posterior_delta"]
UploadStatus = Literal["pending", "in_flight", "dead_letter"]


class OutboxDedupeConflictError(RuntimeError):
    pass


_READY_BATCH_SQL = """
    SELECT upload_id, endpoint, payload_type, dedupe_key, payload_json,
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
    created_at_utc: str
    next_attempt_at_utc: str
    attempt_count: int
    locked_at_utc: str | None
    last_error: str | None
    status: UploadStatus


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        bootstrap_schema(self._conn)

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
        try:
            self._conn.execute(
                """
                INSERT INTO pending_uploads (
                    upload_id, endpoint, payload_type, dedupe_key, payload_json,
                    created_at_utc, next_attempt_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (upload_id, endpoint, payload_type, dedupe_key, payload_json, created, created),
            )
        except sqlite3.IntegrityError as exc:
            row = self._conn.execute(
                """
                SELECT upload_id, payload_json
                FROM pending_uploads
                WHERE payload_type = ? AND dedupe_key = ?
                """,
                (payload_type, dedupe_key),
            ).fetchone()
            if row is not None and str(row["payload_json"]) == payload_json:
                return str(row["upload_id"])
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

    def validate_upload(self, upload: PendingUpload) -> object:
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


def _row_to_pending_upload(row: sqlite3.Row) -> PendingUpload:
    return PendingUpload(
        upload_id=str(row["upload_id"]),
        endpoint=row["endpoint"],
        payload_type=row["payload_type"],
        dedupe_key=str(row["dedupe_key"]),
        payload_json=str(row["payload_json"]),
        created_at_utc=str(row["created_at_utc"]),
        next_attempt_at_utc=str(row["next_attempt_at_utc"]),
        attempt_count=int(row["attempt_count"]),
        locked_at_utc=row["locked_at_utc"],
        last_error=row["last_error"],
        status=row["status"],
    )


def _format_dt(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_error(error: str) -> str:
    return error[:500]
