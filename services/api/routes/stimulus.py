"""
Stimulus Endpoint — §4.E.1 Operator Intervention

REST endpoint for the operator to trigger greeting line injection.
Publishes to the Message Broker "stimulus:inject" channel; the
Orchestrator Container listener then calls record_stimulus_injection().

Provides the REST bridge from the operator's stimulus decision to Module C
record_stimulus_injection().
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/stimulus")
async def trigger_stimulus() -> dict[str, Any]:
    """
    Trigger stimulus injection through the Message Broker.

    §4.E.1 — The operator calls this endpoint when they send the greeting
    line into the live stream chat. This ends the AU12 calibration phase
    and begins the measurement window for the reward pipeline.

    The API Server publishes the trigger to the Message Broker. The
    Orchestrator Container listener receives it and calls
    record_stimulus_injection().

    Returns:
        JSON with status and number of Message Broker subscribers that
        received the message.

    Raises:
        HTTPException 503: If the Message Broker is unavailable.
    """
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

    try:
        import redis as redis_lib

        client = redis_lib.from_url(redis_url)  # type: ignore[no-untyped-call]
        receivers: int = client.publish("stimulus:inject", "inject")
        client.close()

        if receivers == 0:
            logger.warning("Stimulus trigger published but no subscribers received it")
            return {
                "status": "published",
                "receivers": 0,
                "warning": "No orchestrator instance is currently listening. "
                "Is the orchestrator container running?",
            }

        logger.info("Stimulus trigger published to %d receiver(s)", receivers)
        return {"status": "triggered", "receivers": receivers}

    except Exception as exc:
        logger.error("Failed to publish stimulus trigger: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Redis unavailable — cannot deliver stimulus trigger to orchestrator.",
        ) from exc
