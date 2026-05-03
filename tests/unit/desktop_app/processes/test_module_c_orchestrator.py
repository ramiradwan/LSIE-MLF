from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.desktop_app.ipc import IpcChannels
from services.desktop_app.processes import module_c_orchestrator


def _make_channels() -> IpcChannels:
    ctx = mp.get_context("spawn")
    return IpcChannels(
        ml_inbox=ctx.Queue(),
        drift_updates=ctx.Queue(),
        analytics_inbox=ctx.Queue(),
    )


def test_drain_drift_updates_applies_numeric_offsets() -> None:
    channels = _make_channels()
    shutdown = mp.get_context("spawn").Event()
    orchestrator = SimpleNamespace(drift_corrector=SimpleNamespace(drift_offset=0.0))

    thread = threading.Thread(
        target=module_c_orchestrator._drain_drift_updates,
        args=(channels, orchestrator, shutdown),
        daemon=True,
    )
    thread.start()
    try:
        channels.drift_updates.put("ignore-me")
        channels.drift_updates.put({"drift_offset": 0.125})

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if orchestrator.drift_corrector.drift_offset == 0.125:
                break
            time.sleep(0.01)
    finally:
        shutdown.set()
        thread.join(timeout=2.0)

    assert orchestrator.drift_corrector.drift_offset == 0.125


def test_run_keeps_desktop_module_c_release_gated_and_cleans_up(tmp_path: Path) -> None:
    channels = _make_channels()
    shutdown = mp.get_context("spawn").Event()
    heartbeat = MagicMock()
    orchestrator = MagicMock()
    orchestrator.drift_corrector = SimpleNamespace(drift_offset=0.0)

    with (
        patch("services.desktop_app.os_adapter.resolve_state_dir", return_value=tmp_path),
        patch("services.desktop_app.state.heartbeats.HeartbeatRecorder", return_value=heartbeat),
        patch("services.worker.pipeline.orchestrator.Orchestrator", return_value=orchestrator),
    ):
        runner = threading.Thread(
            target=module_c_orchestrator.run,
            kwargs={"shutdown_event": shutdown, "channels": channels},
            daemon=True,
        )
        runner.start()
        try:
            channels.drift_updates.put({"drift_offset": 0.25})
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if orchestrator.drift_corrector.drift_offset == 0.25:
                    break
                time.sleep(0.01)
        finally:
            shutdown.set()
            runner.join(timeout=2.0)

    assert not runner.is_alive()
    assert orchestrator.drift_corrector.drift_offset == 0.25
    orchestrator.run.assert_not_called()
    orchestrator.close_inflight_blocks.assert_called_once_with()
    heartbeat.start.assert_called_once_with()
    heartbeat.stop.assert_called_once_with()
