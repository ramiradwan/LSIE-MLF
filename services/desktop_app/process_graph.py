"""Six-process desktop graph bootstrap.

The desktop runtime runs as six named ``multiprocessing.Process``
children spawned via the ``spawn`` start method, matching §9.1. The
parent never imports a process module directly, so the ML libraries
remain isolated to ``gpu_ml_worker``.

Each process module exposes a ``run(shutdown_event, channels)``
callable. ``channels`` is the :class:`IpcChannels` bundle carrying the
queues that connect the graph.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import multiprocessing.context as mpcontext
import multiprocessing.synchronize as mpsync
import time
from dataclasses import dataclass, field

from services.desktop_app.ipc import IpcChannels

logger = logging.getLogger(__name__)

# Canonical process-name → module-path mapping. The order is the order
# in which processes are spawned and is the same order the v4.0 spec
# §9 process table uses.
PROCESS_MODULES: dict[str, str] = {
    "ui_api_shell": "services.desktop_app.processes.ui_api_shell",
    "capture_supervisor": "services.desktop_app.processes.capture_supervisor",
    "module_c_orchestrator": "services.desktop_app.processes.module_c_orchestrator",
    "gpu_ml_worker": "services.desktop_app.processes.gpu_ml_worker",
    "analytics_state_worker": "services.desktop_app.processes.analytics_state_worker",
    "cloud_sync_worker": "services.desktop_app.processes.cloud_sync_worker",
}

# Modules that MUST NOT appear in the parent's ``sys.modules`` after
# importing process_graph. A canary test re-asserts this in a clean
# subprocess on every CI run.
ML_LIB_ROOTS: frozenset[str] = frozenset({"torch", "mediapipe", "faster_whisper", "ctranslate2"})

SQLITE_FILENAME = "desktop.sqlite"


def _prepare_runtime_state() -> None:
    import sqlite3

    from services.desktop_app.os_adapter import resolve_state_dir
    from services.desktop_app.state.recovery import run_recovery_sweep
    from services.desktop_app.state.sqlite_schema import bootstrap_schema

    db_path = resolve_state_dir() / SQLITE_FILENAME
    bootstrap_conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        bootstrap_schema(bootstrap_conn)
    finally:
        bootstrap_conn.close()
    run_recovery_sweep(db_path)


def _launch(
    module_name: str,
    shutdown_event: mpsync.Event,
    channels: IpcChannels,
) -> None:
    """Pickleable child entrypoint — runs only inside the spawned child.

    Spawn-mode children re-import this module to find ``_launch``, then
    invoke it. The child installs the crash-privacy guards first,
    because crash-dialog state is per-process on Windows and does not
    propagate from the parent, then imports its target module.
    """
    from services.desktop_app.privacy.crash_dumps import install_crash_privacy_guards

    install_crash_privacy_guards()
    mod = importlib.import_module(module_name)
    mod.run(shutdown_event=shutdown_event, channels=channels)


@dataclass
class ProcessGraph:
    """Lifecycle manager for the six v4.0 desktop processes.

    Public surface is intentionally narrow: ``start_all`` to spawn,
    ``stop_all`` to signal cooperative shutdown then force-terminate
    after a grace period, ``wait`` for blocking join. The dict members
    are exposed for the canary tests to introspect PIDs and exit codes
    without needing private accessors.
    """

    children: dict[str, mpcontext.SpawnProcess] = field(default_factory=dict)
    shutdown_events: dict[str, mpsync.Event] = field(default_factory=dict)
    channels: IpcChannels | None = field(default=None)
    _shutdown_requested: bool = field(default=False, init=False)

    def start_all(self) -> None:
        self._shutdown_requested = False
        _prepare_runtime_state()
        ctx = mp.get_context("spawn")
        if self.channels is None:
            self.channels = IpcChannels(
                ml_inbox=ctx.Queue(),
                drift_updates=ctx.Queue(),
                analytics_inbox=ctx.Queue(),
            )
        for name, module in PROCESS_MODULES.items():
            evt = ctx.Event()
            proc = ctx.Process(
                name=name,
                target=_launch,
                args=(module, evt, self.channels),
                daemon=False,
            )
            proc.start()
            self.children[name] = proc
            self.shutdown_events[name] = evt
            logger.info("spawned %s pid=%s", name, proc.pid)

    def request_shutdown(self) -> None:
        self._shutdown_requested = True
        for evt in self.shutdown_events.values():
            evt.set()

    def stop_all(self, timeout: float = 5.0) -> None:
        self.request_shutdown()
        for name, proc in self.children.items():
            proc.join(timeout=timeout)
            if proc.is_alive():
                logger.warning("force-terminating %s pid=%s", name, proc.pid)
                proc.terminate()
                proc.join(timeout=timeout)
        self.children.clear()
        self.shutdown_events.clear()

    def wait(self, poll_interval: float = 0.25, shutdown_timeout: float = 5.0) -> None:
        shutdown_started_at: float | None = None
        while self.children:
            exited: list[tuple[str, int | None]] = []
            alive = False
            for name, proc in self.children.items():
                if proc.is_alive():
                    alive = True
                    continue
                exited.append((name, proc.exitcode))

            if exited and not self._shutdown_requested:
                name, exitcode = exited[0]
                logger.info("%s exited with code %s; requesting graph shutdown", name, exitcode)
                self.request_shutdown()

            if self._shutdown_requested and shutdown_started_at is None:
                shutdown_started_at = time.monotonic()

            if not alive:
                return

            if (
                shutdown_started_at is not None
                and (time.monotonic() - shutdown_started_at) >= shutdown_timeout
            ):
                self.stop_all(timeout=max(1.0, shutdown_timeout))
                return

            time.sleep(poll_interval)
