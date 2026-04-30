"""Module C orchestrator process (v4.0 §4.C / WS3 P1 stub).

Wraps the existing ``services.worker.pipeline.orchestrator.Orchestrator``
class (drift correction, segment assembly, AU12 derivation, the
physiological state buffer). The Phase 2 _desktop_ipc_mode flag will
gate whether ``assemble_segment`` dispatches via Celery (legacy, dying)
or via the IPC control queue + SharedMemory route.

ML import discipline: ``Orchestrator`` itself transitively touches
neither torch nor mediapipe nor faster_whisper nor ctranslate2 — the
heavy ML libs live in ``processes.gpu_ml_worker`` only. This stub MUST
not regress that.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    # Phase 2 wires the IPC channel into the orchestrator's dispatch
    # path. Live audio feed from capture_supervisor lands in WS3 P3;
    # until then this stub holds the channel reference for inspection.
    del channels
    logger.info("module_c_orchestrator started")
    shutdown_event.wait()
    logger.info("module_c_orchestrator stopped")
