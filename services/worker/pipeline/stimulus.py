"""Retained server/cloud stimulus trigger bridge.

Provides two mechanisms for triggering record_stimulus_injection() on the
retained orchestrator instance:

1. AUTO-TRIGGER: after a configurable calibration delay, the orchestrator
   automatically injects the stimulus. Controlled by AUTO_STIMULUS_DELAY_S;
   set to 0 to disable.

2. REDIS PUB/SUB: retained API/operator tooling publishes a message to the
   "stimulus:inject" Redis channel. The orchestrator subscribes and triggers on
   receipt. The v4 desktop runtime uses local operator API commands instead.

§4.E.1 — Thompson Sampling experiment arm deployment trigger.
§7A.4 — Calibration phase ends at stimulus onset.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

AUTO_STIMULUS_DELAY_S: float = float(os.environ.get("AUTO_STIMULUS_DELAY_S", "15"))
STIMULUS_CHANNEL: str = "stimulus:inject"


def setup_auto_trigger(orchestrator: Any, delay_s: float | None = None) -> threading.Timer | None:
    """Schedule automatic stimulus injection after a delay."""
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
    """Start a retained Redis listener for stimulus trigger messages."""
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

    def _listen() -> None:
        try:
            import redis

            client = redis.from_url(redis_url)
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
    """Publish a stimulus trigger message to the retained Redis channel."""
    url = redis_url or os.environ.get("REDIS_URL", "redis://redis:6379/0")
    try:
        import redis

        client = redis.from_url(url)
        receivers = client.publish(STIMULUS_CHANNEL, "inject")
        client.close()
        logger.info("Stimulus trigger published (%d receivers)", receivers)
        return True
    except Exception:
        logger.warning("Failed to publish stimulus trigger", exc_info=True)
        return False
