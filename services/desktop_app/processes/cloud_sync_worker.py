"""Cloud sync worker process (v4.0 §4.E.* / WS3 P1 stub, body in WS5 P2).

Drives the offline-tolerant HTTPS telemetry path: drains the
``pending_uploads`` SQLite table populated by
``analytics_state_worker``, batches into ``POST /v4/telemetry/segments``
and ``POST /v4/telemetry/posterior_deltas``, retries with exponential
backoff via ``httpx``. Holds the cloud OAuth refresh token via the
Workstream 4 Phase 4 ``keyring`` wrapper.

WS4 P2 adds a per-process :class:`HeartbeatRecorder` so the operator
console health rollup can render freshness even while the cloud
outbox path is still a stub.

ML import discipline: this module MUST NOT import torch / mediapipe /
faster_whisper / ctranslate2.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)

SQLITE_FILENAME = "desktop.sqlite"


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    del channels  # WS5 P2 wires the cloud outbox drain.
    logger.info("cloud_sync_worker started")

    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder

    state_dir = resolve_state_dir()
    heartbeat = HeartbeatRecorder(state_dir / SQLITE_FILENAME, "cloud_sync_worker")
    heartbeat.start()

    try:
        shutdown_event.wait()
    finally:
        heartbeat.stop()
        logger.info("cloud_sync_worker stopped")
