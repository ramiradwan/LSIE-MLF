"""
Orchestrator Entrypoint — §4.C Module C

Starts the Orchestrator.run() asyncio loop for the Orchestrator Container.
The Orchestrator Container reads from IPC Pipes, processes video frames for
AU12, assembles 30-second segments, and dispatches process_segment.delay()
tasks through the Message Broker to the ML Worker.

This module is the CMD target for the Orchestrator Container service in
docker-compose.yml:
    command: ["python3.11", "-m", "services.worker.run_orchestrator"]

Environment variables:
    STREAM_URL — TikTok stream URL (optional, default "")
    EXPERIMENT_ID — Thompson Sampling experiment ID (default "greeting_line_v1")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("orchestrator.main")


def main() -> None:
    """Start the Orchestrator Container loop with graceful shutdown."""
    from services.worker.pipeline.orchestrator import Orchestrator

    stream_url = os.environ.get("STREAM_URL", "")
    experiment_id = os.environ.get("EXPERIMENT_ID", "greeting_line_v1")

    orchestrator = Orchestrator(
        stream_url=stream_url,
        experiment_id=experiment_id,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGTERM (Docker stop) and SIGINT (Ctrl+C)
    def _shutdown(sig: int, frame: object) -> None:
        logger.info("Received signal %d, shutting down orchestrator...", sig)
        orchestrator.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        "Starting orchestrator (stream_url=%s, experiment_id=%s)",
        stream_url or "(none)",
        experiment_id,
    )

    # Set up stimulus injection triggers.
    auto_timer = None
    _redis_thread = None
    try:
        from services.worker.pipeline.stimulus import (
            setup_auto_trigger,
            start_redis_listener,
        )

        # Auto-trigger for E2E testing (controlled by AUTO_STIMULUS_DELAY_S env var)
        auto_timer = setup_auto_trigger(orchestrator)

        # Message Broker listener for production operator stimulus triggers.
        _redis_thread = start_redis_listener(orchestrator)
    except Exception:
        logger.warning("Stimulus trigger setup failed", exc_info=True)

    try:
        loop.run_until_complete(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Orchestrator interrupted")
    finally:
        if auto_timer is not None:
            auto_timer.cancel()
        orchestrator.stop()
        loop.close()
        logger.info("Orchestrator shutdown complete")


if __name__ == "__main__":
    main()
