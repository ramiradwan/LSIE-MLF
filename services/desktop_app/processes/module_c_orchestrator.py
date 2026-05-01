"""Module C orchestrator process (v4.0 §4.C / WS3 P1 + P3).

Wraps the existing ``services.worker.pipeline.orchestrator.Orchestrator``
class (drift-corrected segment assembly, AU12 derivation, the
physiological state buffer). The IPC channel from
``module_c_orchestrator`` to ``gpu_ml_worker`` is wired by the
orchestrator's ``_dispatch_payload`` (WS3 P2). The drift offset is
delivered by ``capture_supervisor`` over
``IpcChannels.drift_updates`` and applied here (WS3 P3).

ML import discipline: ``Orchestrator`` itself transitively touches
neither torch nor mediapipe nor faster_whisper nor ctranslate2 — the
heavy ML libs live in ``processes.gpu_ml_worker`` only.
"""

from __future__ import annotations

import logging
import multiprocessing.synchronize as mpsync
import queue
import threading

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)

DRIFT_DRAIN_POLL_TIMEOUT_S = 0.5
SQLITE_FILENAME = "desktop.sqlite"


def _drain_drift_updates(
    channels: IpcChannels,
    orchestrator: object,
    shutdown_event: mpsync.Event,
) -> None:
    """Apply each ``drift_updates`` payload to the orchestrator's corrector.

    The supervisor pushes ``{"drift_offset": float, ...}`` dicts on
    every poll cycle (~30 s). We update the orchestrator's drift
    corrector inline; the orchestrator's apply-side
    ``correct_timestamp`` calls then pick up the new offset on the
    next read.
    """
    corrector = getattr(orchestrator, "drift_corrector", None)
    if corrector is None:
        logger.error("orchestrator has no drift_corrector — drift updates will be dropped")
        return

    while not shutdown_event.is_set():
        try:
            payload = channels.drift_updates.get(timeout=DRIFT_DRAIN_POLL_TIMEOUT_S)
        except queue.Empty:
            continue
        if not isinstance(payload, dict):
            continue
        offset = payload.get("drift_offset")
        if isinstance(offset, int | float):
            corrector.drift_offset = float(offset)


def run(shutdown_event: mpsync.Event, channels: IpcChannels) -> None:
    logger.info("module_c_orchestrator started")

    # Late imports: keeps the parent's import-isolation canary clean.
    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.heartbeats import HeartbeatRecorder
    from services.worker.pipeline.orchestrator import Orchestrator

    orchestrator = Orchestrator(ipc_queue=channels.ml_inbox)

    drift_thread = threading.Thread(
        target=_drain_drift_updates,
        args=(channels, orchestrator, shutdown_event),
        name="module-c-drift-drain",
        daemon=True,
    )
    drift_thread.start()

    state_dir = resolve_state_dir()
    heartbeat = HeartbeatRecorder(state_dir / SQLITE_FILENAME, "module_c_orchestrator")
    heartbeat.start()

    try:
        # Phase-3 stub: the live capture-supervisor → orchestrator
        # audio/video pipe-through is not wired yet (Orchestrator.run()
        # still expects the v3.4 IPC pipe path under /tmp/ipc/). Until
        # WS3 P3c lands, this process holds the orchestrator instance
        # alive so the drift consumer can keep applying offsets and
        # the segment-id math stays available to test fixtures.
        shutdown_event.wait()
    finally:
        heartbeat.stop()
        try:
            orchestrator.close_inflight_blocks()
        except Exception:  # noqa: BLE001
            logger.debug("inflight cleanup failed", exc_info=True)
        drift_thread.join(timeout=5.0)
        logger.info("module_c_orchestrator stopped")
