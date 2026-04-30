"""Analytics + state worker process (v4.0 §4.E / WS3 P1 stub).

Sole writer to the local SQLite store (WS4 P1) and the only process
that runs the §7B reward computation and the Beta-Bernoulli posterior
update. Wraps ``services.worker.pipeline.analytics.MetricsStore`` and
``ThompsonSamplingEngine``, plus
``services.worker.pipeline.reward.compute_reward`` — none of which
transitively touch torch / mediapipe / faster_whisper / ctranslate2.

ML import discipline: this module MUST NOT import any of the four ML
library roots at any scope.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    del channels  # WS5 P4 wires the analytics inbox.
    logger.info("analytics_state_worker started")
    shutdown_event.wait()
    logger.info("analytics_state_worker stopped")
