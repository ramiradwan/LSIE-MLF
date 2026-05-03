"""Module C desktop shell for drift updates and orchestrator lifecycle.

This process owns the desktop wrapper around
``services.worker.pipeline.orchestrator.Orchestrator`` while keeping the
heavy ML libraries isolated to ``gpu_ml_worker``.
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
        # Live Module C dispatch remains release-gated on the desktop path.
        # services.worker.pipeline.orchestrator.Orchestrator.run() would
        # initialize the AU12/FaceMesh path in this process, which breaks
        # the v4 desktop ML-isolation contract that reserves mediapipe for
        # gpu_ml_worker only. Keep this process as the drift/lifecycle shell
        # until a desktop-safe capture path moves that ML surface behind the
        # GPU worker boundary.
        shutdown_event.wait()
    finally:
        heartbeat.stop()
        try:
            orchestrator.close_inflight_blocks()
        except Exception:  # noqa: BLE001
            logger.debug("inflight cleanup failed", exc_info=True)
        drift_thread.join(timeout=5.0)
        logger.info("module_c_orchestrator stopped")
