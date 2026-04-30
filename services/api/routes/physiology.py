"""
Physiological Webhook Ingress — §4.B.2, §2.8 notification-only ingress

API Server route for authenticated Oura Ring v2 webhook notifications.
Authenticates incoming JSON, deduplicates deliveries, and enqueues minimal
hydration metadata to the Message Broker for a later provider-fetch stage.

Design constraints:
  - Lives in services/api/routes/ (API Server container, python:3.11-slim)
  - MUST NOT import anything from packages/ml_core/ (§3.2 image separation)
  - Uses HMAC-SHA256 signature verification against OURA_WEBHOOK_SECRET
  - Idempotency via Message Broker SETNX with 1-hour TTL
  - Enqueue via RPUSH to the physio:hydrate Message Broker list
  - Returns 200 on success, 401 bad sig, 422 bad payload, 503 Message Broker failure

Spec references:
  §4.B.2  — Physiological Ingestion Adapter
  §2.8    — API Server → Orchestrator Container transition
  §5      — Raw payloads are Transient Sensitive Data
  §12     — Failure matrix: invalid sig, duplicate, enqueue failure
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request

from services.api.db.connection import get_connection, put_connection

logger = logging.getLogger(__name__)

router = APIRouter()

# §5 — Secret injected at runtime, never hardcoded.
OURA_WEBHOOK_SECRET: str = os.environ.get("OURA_WEBHOOK_SECRET", "")

# Message Broker key prefix for idempotency tracking.
_IDEMPOTENCY_PREFIX = "physio:seen:"
_IDEMPOTENCY_TTL_S = 3600  # 1 hour

# Message Broker list key for hydration notifications.
_PHYSIO_HYDRATE_QUEUE = "physio:hydrate"


def _get_webhook_secret() -> str:
    """Read the webhook secret from the environment.

    The module-level constant preserves the reference artifact shape while this
    helper ensures tests and long-lived processes observe the current env value.
    """
    return os.environ.get("OURA_WEBHOOK_SECRET", OURA_WEBHOOK_SECRET)


def _verify_oura_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """HMAC-SHA256 verification of the raw Oura webhook payload.

    §4.B.2 — Mandatory authentication before any payload processing.
    """
    if not secret:
        logger.error("OURA_WEBHOOK_SECRET not configured; rejecting all webhooks")
        return False
    if not signature:
        return False

    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _get_redis() -> Any:
    """Create the Redis client for Message Broker access using REDIS_URL."""
    import redis as redis_lib

    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    return redis_lib.Redis.from_url(redis_url, decode_responses=True)


def _derive_event_uuid(raw: dict[str, Any], body: bytes) -> uuid.UUID:
    """Derive a stable event UUID for idempotency.

    If the provider sends a UUID-shaped event_id (or id), preserve it.
    Otherwise derive a deterministic UUID5 from the provider event identifier;
    if no provider event identifier exists, fall back to the raw body digest.
    """
    provider_event_id = raw.get("event_id") or raw.get("id")
    if provider_event_id is None:
        data = raw.get("data")
        if isinstance(data, dict):
            provider_event_id = data.get("event_id") or data.get("id")

    if provider_event_id is not None:
        provider_event_id_str = str(provider_event_id)
        with suppress(ValueError, TypeError, AttributeError):
            return uuid.UUID(provider_event_id_str)
        return uuid.uuid5(uuid.NAMESPACE_URL, f"oura:{provider_event_id_str}")

    digest = hashlib.sha256(body).hexdigest()
    return uuid.uuid5(uuid.NAMESPACE_URL, f"oura:{digest}")


@router.post("/ingest/oura/webhook")
async def oura_webhook(
    request: Request,
    x_oura_signature: str = Header(..., alias="x-oura-signature"),
) -> dict[str, str]:
    """§4.B.2 — Oura Ring v2 webhook receiver.

    Accepts change notifications, verifies authenticity, validates the minimal
    routing metadata, and enqueues hydration work for downstream processing.

    §12 failure matrix:
      - Invalid signature → 401
      - Duplicate delivery → 200 (no-op)
      - Malformed payload → 422
      - Message Broker enqueue failure → 503
    """
    body = await request.body()

    if not _verify_oura_signature(body, x_oura_signature, _get_webhook_secret()):
        logger.warning("Oura webhook signature verification failed")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Malformed JSON") from exc

    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="Malformed JSON")

    subject_role = raw.get("subject_role")
    if subject_role not in ("streamer", "operator"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid or missing subject_role: {subject_role}",
        )

    event_type = raw.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise HTTPException(status_code=422, detail="Invalid or missing event_type")

    data_type = raw.get("data_type")
    if not isinstance(data_type, str) or not data_type.strip():
        raise HTTPException(status_code=422, detail="Invalid or missing data_type")

    start_datetime = raw.get("start_datetime")
    if not isinstance(start_datetime, str) or not start_datetime.strip():
        raise HTTPException(status_code=422, detail="Invalid or missing start_datetime")

    end_datetime = raw.get("end_datetime")
    if not isinstance(end_datetime, str) or not end_datetime.strip():
        raise HTTPException(status_code=422, detail="Invalid or missing end_datetime")

    event_uuid = _derive_event_uuid(raw, body)
    hydration_payload = {
        "unique_id": str(event_uuid),
        "subject_role": subject_role,
        "event_type": event_type,
        "data_type": data_type,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "notification_received_utc": datetime.now(UTC).isoformat(),
    }
    idem_key = f"{_IDEMPOTENCY_PREFIX}{event_uuid}"

    redis_client = None
    try:
        redis_client = _get_redis()
        if not redis_client.set(idem_key, "1", nx=True, ex=_IDEMPOTENCY_TTL_S):
            logger.info("Duplicate physiological webhook: event_id=%s", event_uuid)
            return {"status": "duplicate", "event_id": str(event_uuid)}

        try:
            redis_client.rpush(_PHYSIO_HYDRATE_QUEUE, json.dumps(hydration_payload))
        except Exception:
            with suppress(Exception):
                redis_client.delete(idem_key)
            raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Message Broker enqueue failed for physiological hydration notification: "
            "subject=%s event_id=%s",
            subject_role,
            event_uuid,
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    finally:
        if redis_client is not None and hasattr(redis_client, "close"):
            with suppress(Exception):
                redis_client.close()

    logger.info(
        "Physiological hydration notification enqueued: subject=%s event_id=%s",
        subject_role,
        event_uuid,
    )
    return {"status": "accepted", "event_id": str(event_uuid)}


# §4.E.2 / §7C — Per-segment physiological snapshot readback (latest per subject_role).
_LATEST_PHYSIO_SQL: str = """
    SELECT DISTINCT ON (p.subject_role)
        p.session_id, p.segment_id, p.subject_role, p.rmssd_ms,
        p.heart_rate_bpm, p.freshness_s, p.is_stale, p.provider,
        p.source_timestamp_utc, p.created_at
    FROM physiology_log p
    WHERE p.session_id = %(session_id)s
    ORDER BY p.subject_role, p.created_at DESC
"""

# §4.E.2 — Per-segment physiological snapshot series (time-series form).
_SERIES_PHYSIO_SQL: str = """
    SELECT p.session_id, p.segment_id, p.subject_role, p.rmssd_ms,
           p.heart_rate_bpm, p.freshness_s, p.is_stale, p.provider,
           p.source_timestamp_utc, p.created_at
    FROM physiology_log p
    WHERE p.session_id = %(session_id)s
    ORDER BY p.created_at ASC
    LIMIT %(limit)s
"""


def _serialize_physio(val: Any) -> Any:
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


def _rows_to_dicts(cursor: Any) -> list[dict[str, Any]]:
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    rows: list[Any] = cursor.fetchall()
    return [
        {col: _serialize_physio(val) for col, val in zip(columns, row, strict=True)} for row in rows
    ]


@router.get("/physiology/{session_id}")
async def get_physiology(
    session_id: str,
    series: bool = Query(False),
    limit: int = Query(500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """§4.E.2 — Physiological snapshot readback for a session.

    Default returns the latest snapshot per ``subject_role``
    (streamer and operator). ``?series=true`` returns the full
    time-series of snapshots for the session, ordered by insertion.

    Args:
        session_id: Session UUID whose physiology_log rows to return.
        series: If True, return the time-series instead of latest-per-role.
        limit: Max rows for series mode (1-5000, default 500).
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            if series:
                cur.execute(
                    _SERIES_PHYSIO_SQL,
                    {"session_id": session_id, "limit": limit},
                )
            else:
                cur.execute(
                    _LATEST_PHYSIO_SQL,
                    {"session_id": session_id},
                )
            return _rows_to_dicts(cur)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to query physiology for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        if conn is not None:
            put_connection(conn)
