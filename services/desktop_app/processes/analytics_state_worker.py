"""Analytics + state worker process (v4.0 §4.E / WS3 P1 + WS4 P1b).

Sole writer to the local SQLite store (WS4 P1) and the only process
that runs the §7B reward computation and the Beta-Bernoulli posterior
update. Wraps ``services.worker.pipeline.analytics.MetricsStore`` and
``ThompsonSamplingEngine``, plus
``services.worker.pipeline.reward.compute_reward`` — none of which
transitively touch torch / mediapipe / faster_whisper / ctranslate2.

WS4 P1b owns the SQLite writer's lifecycle here:

* Open and start the :class:`SqliteWriter` against the resolved
  app-data ``desktop.sqlite`` so the operator console reads pick up
  any rows the worker enqueues.
* Drain pending records on cooperative shutdown so the WAL flushes
  cleanly before process teardown.

The full reward-pipeline / analytics-inbox wiring lands in WS5 P4; for
now this process owns the writer and idles on the shutdown event,
which is what the parent ``DesktopGraph.stop_all()`` already drives.

ML import discipline: this module MUST NOT import any of the four ML
library roots at any scope.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)

SQLITE_FILENAME = "desktop.sqlite"


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    del channels  # WS5 P4 wires the analytics inbox.
    logger.info("analytics_state_worker started")

    # Late imports preserve the ML-isolation canary contract and keep
    # the parent process free of SQLite handles.
    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.sqlite_writer import SqliteWriter

    state_dir = resolve_state_dir()
    db_path = state_dir / SQLITE_FILENAME
    writer = SqliteWriter(db_path)
    writer.start()
    logger.info("sqlite writer opened at %s", db_path)

    try:
        shutdown_event.wait()
    finally:
        writer.close()
        logger.info("analytics_state_worker stopped")
