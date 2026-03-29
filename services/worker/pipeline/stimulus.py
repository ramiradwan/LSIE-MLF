"""
Stimulus Trigger — §4.E.1 Operator Intervention Bridge

Provides two mechanisms for triggering record_stimulus_injection()
on the running Orchestrator instance:

1. AUTO-TRIGGER (E2E testing): After a configurable calibration delay,
   the orchestrator automatically injects the stimulus. Controlled by
   the AUTO_STIMULUS_DELAY_S environment variable. Set to 0 to disable.

2. REDIS PUB/SUB (production): The API server or operator dashboard
   publishes a message to the "stimulus:inject" Redis channel. The
   orchestrator subscribes and triggers on receipt.

§4.E.1 — Thompson Sampling experiment arm deployment trigger.
§7.4 — Calibration phase ends at stimulus onset.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Auto-trigger delay in seconds. Set to 0 to disable auto-trigger.
# For E2E testing, 15 seconds gives enough calibration frames at 30fps
# (450 frames for B_neutral accumulation).
AUTO_STIMULUS_DELAY_S: float = float(os.environ.get("AUTO_STIMULUS_DELAY_S", "15"))

# Redis channel for external stimulus triggers
STIMULUS_CHANNEL: str = "stimulus:inject"


def setup_auto_trigger(orchestrator: Any, delay_s: float | None = None) -> threading.Timer | None:
    """
    Schedule automatic stimulus injection after a delay.

    Used for E2E testing to remove the human-in-the-loop dependency.
    Returns the Timer so the caller can cancel it during shutdown.

    Args:
        orchestrator: The Orchestrator instance with record_stimulus_injection().
        delay_s: Override delay in seconds. None uses AUTO_STIMULUS_DELAY_S env var.

    Returns:
        The started Timer, or None if auto-trigger is disabled (delay <= 0).
    """
    delay = delay_s if delay_s is not None else AUTO_STIMULUS_DELAY_S

    if delay <= 0:
        logger.info("Auto-trigger disabled (AUTO_STIMULUS_DELAY_S=0)")
        return None

    def _trigger() -> None:
        if orchestrator._is_calibrating:
            logger.info("Auto-trigger firing after %.1fs calibration delay", delay)
            orchestrator.record_stimulus_injection()
        else:
            logger.debug("Auto-trigger skipped — stimulus already injected")

    timer = threading.Timer(delay, _trigger)
    timer.daemon = True
    timer.start()
    logger.info("Auto-trigger scheduled in %.1fs", delay)
    return timer


def start_redis_listener(orchestrator: Any) -> threading.Thread | None:
    """
    Start a background thread that listens for stimulus trigger messages
    on the Redis pub/sub channel.

    The API server or operator dashboard publishes to "stimulus:inject"
    when the operator sends the greeting line into the chat. The message
    payload is ignored — any message on the channel triggers injection.

    Returns:
        The listener thread, or None if Redis is unavailable.
    """
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

    def _listen() -> None:
        try:
            import redis

            client = redis.from_url(redis_url)  # type: ignore[no-untyped-call]
            pubsub = client.pubsub()
            pubsub.subscribe(STIMULUS_CHANNEL)
            logger.info("Redis stimulus listener subscribed to '%s'", STIMULUS_CHANNEL)

            for message in pubsub.listen():
                if message["type"] == "message":
                    if orchestrator._is_calibrating:
                        logger.info("Redis stimulus trigger received")
                        orchestrator.record_stimulus_injection()
                    else:
                        logger.debug("Redis trigger ignored — stimulus already injected")

                    # Only trigger once per session
                    break

            pubsub.unsubscribe()
            client.close()
        except Exception:
            logger.warning("Redis stimulus listener unavailable", exc_info=True)

    thread = threading.Thread(target=_listen, name="stimulus-listener", daemon=True)
    thread.start()
    return thread


def publish_stimulus_trigger(redis_url: str | None = None) -> bool:
    """
    Publish a stimulus trigger message to the Redis channel.

    Called by the API endpoint or operator dashboard when the operator
    sends the greeting line into the live stream chat.

    Args:
        redis_url: Redis connection URL. Defaults to REDIS_URL env var.

    Returns:
        True if the message was published, False on failure.
    """
    url = redis_url or os.environ.get("REDIS_URL", "redis://redis:6379/0")
    try:
        import redis

        client = redis.from_url(url)  # type: ignore[no-untyped-call]
        receivers = client.publish(STIMULUS_CHANNEL, "inject")
        client.close()
        logger.info("Stimulus trigger published (%d receivers)", receivers)
        return True
    except Exception:
        logger.warning("Failed to publish stimulus trigger", exc_info=True)
        return False
