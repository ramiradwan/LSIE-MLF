"""
Physiological Webhook Ingestion — §4.B.2, §2 (v3.1 API→Orchestrator transition)

FastAPI POST route for authenticated Oura Ring v2 webhook payloads.
Normalizes incoming JSON into PhysiologicalSampleEvent and enqueues
it to Redis for the Orchestrator Container to drain.

Design constraints:
  - Lives in services/api/routes/ (API Server container, python:3.11-slim)
  - MUST NOT import anything from packages/ml_core/ (§3.2 image separation)
  - Uses HMAC-SHA256 signature verification against OURA_WEBHOOK_SECRET
  - Idempotency via Redis SETNX with 1-hour TTL
  - Enqueue via RPUSH to physio:events Redis list
  - Returns 200 on success, 401 bad sig, 422 bad payload, 503 Redis failure

Spec references:
  §4.B.2  — Physiological Ingestion Adapter
  §2      — API Server → Orchestrator Container transition
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

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError

from packages.schemas.physiology import PhysiologicalPayload, PhysiologicalSampleEvent

logger = logging.getLogger(__name__)

router = APIRouter()

# §5 — Secret injected at runtime, never hardcoded.
OURA_WEBHOOK_SECRET: str = os.environ.get("OURA_WEBHOOK_SECRET", "")

# Redis key prefix for idempotency tracking.
_IDEMPOTENCY_PREFIX = "physio:seen:"
_IDEMPOTENCY_TTL_S = 3600  # 1 hour

# Redis list key for inter-container transport.
_PHYSIO_QUEUE_KEY = "physio:events"


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
    """Create a Redis client using REDIS_URL from the environment."""
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


def _normalize_event(raw: dict[str, Any], body: bytes) -> PhysiologicalSampleEvent:
    """Normalize an Oura webhook payload into PhysiologicalSampleEvent."""
    subject_role = raw.get("subject_role")
    if subject_role not in ("streamer", "operator"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid or missing subject_role: {subject_role}",
        )

    hrv_data = raw.get("data")
    if not isinstance(hrv_data, dict):
        raise HTTPException(status_code=422, detail="Payload parsing error: data must be an object")

    source_ts_str = hrv_data.get("timestamp")
    if source_ts_str is None:
        raise HTTPException(status_code=422, detail="Missing timestamp in payload")

    try:
        source_ts = datetime.fromisoformat(str(source_ts_str).replace("Z", "+00:00"))
        event = PhysiologicalSampleEvent(
            unique_id=_derive_event_uuid(raw, body),
            event_type="physiological_sample",
            provider="oura",
            subject_role=subject_role,
            source_timestamp_utc=source_ts,
            ingest_timestamp_utc=datetime.now(UTC),
            payload=PhysiologicalPayload(
                rmssd_ms=float(hrv_data["rmssd"]) if hrv_data.get("rmssd") is not None else None,
                heart_rate_bpm=(
                    int(hrv_data["heart_rate"]) if hrv_data.get("heart_rate") is not None else None
                ),
                sample_window_s=300,
            ),
        )
    except HTTPException:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Payload parsing error: {exc}") from exc

    return event


@router.post("/ingest/oura/webhook")
async def oura_webhook(
    request: Request,
    x_oura_signature: str = Header(..., alias="x-oura-signature"),
) -> dict[str, str]:
    """§4.B.2 — Oura Ring v2 webhook receiver.

    Accepts physiological telemetry, verifies authenticity, normalizes it
    into PhysiologicalSampleEvent, and enqueues it to Redis for the
    Orchestrator Container.

    §12 failure matrix:
      - Invalid signature → 401
      - Duplicate delivery → 200 (no-op)
      - Malformed payload → 422
      - Redis enqueue failure → 503
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

    event = _normalize_event(raw, body)
    idem_key = f"{_IDEMPOTENCY_PREFIX}{event.unique_id}"

    redis_client = None
    try:
        redis_client = _get_redis()
        if not redis_client.set(idem_key, "1", nx=True, ex=_IDEMPOTENCY_TTL_S):
            logger.info("Duplicate physiological webhook: %s", event.unique_id)
            return {"status": "duplicate", "event_id": str(event.unique_id)}

        try:
            redis_client.rpush(_PHYSIO_QUEUE_KEY, event.model_dump_json())
        except Exception:
            with suppress(Exception):
                redis_client.delete(idem_key)
            raise
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Redis enqueue failed for physiological event: subject=%s event_id=%s",
            event.subject_role,
            event.unique_id,
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    finally:
        if redis_client is not None and hasattr(redis_client, "close"):
            with suppress(Exception):
                redis_client.close()

    logger.info(
        "Physiological event enqueued: subject=%s rmssd=%s hr=%s",
        event.subject_role,
        event.payload.rmssd_ms,
        event.payload.heart_rate_bpm,
    )
    return {"status": "accepted", "event_id": str(event.unique_id)}
