"""Capture supervisor process (v4.0 §4.A + §4.C.1 / WS3 P1 stub).

Owns scrcpy, ADB, and FFmpeg subprocesses. WS3 P3 wraps them in a
Win32 Job Object so a parent crash cannot leave orphaned device
holders. The ``DriftCorrector`` poll moves here from
``services.worker.pipeline.orchestrator`` — the freeze and reset
thresholds (3 failures, 300 s) survive the move verbatim.

ML import discipline: this module MUST NOT import ``torch``,
``mediapipe``, ``faster_whisper``, or ``ctranslate2`` at any scope.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    del channels  # WS3 P3 wires the orchestrator audio feed.
    logger.info("capture_supervisor started")
    shutdown_event.wait()
    logger.info("capture_supervisor stopped")
