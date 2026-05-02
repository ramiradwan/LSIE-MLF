"""Six-process desktop graph bootstrap (v4.0 §9 / WS3 P1 + P2).

Replaces the v3.4 docker-compose container topology with six named
``multiprocessing.Process`` children spawned via the ``spawn`` start
method. The parent process never imports a process module directly —
children are launched by *string* through :func:`_launch`, so the ML
import discipline holds: ``torch`` / ``mediapipe`` / ``faster_whisper``
/ ``ctranslate2`` are imported only inside
``services.desktop_app.processes.gpu_ml_worker`` and only after the
spawned child re-imports it.

Each process module exposes a ``run(shutdown_event, channels)``
callable. ``channels`` is the :class:`IpcChannels` bundle carrying the
multiprocessing queues that knit the graph together (Phase 2 wires
``ml_inbox`` between ``module_c_orchestrator`` and ``gpu_ml_worker``).
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import multiprocessing.context as mpcontext
import multiprocessing.synchronize as mpsync
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


def _launch(
    module_name: str,
    shutdown_event: mpsync.Event,
    channels: IpcChannels,
) -> None:
    """Pickleable child entrypoint — runs only inside the spawned child.

    Spawn-mode children re-import this module to find ``_launch``, then
    invoke it. The first thing the child does is install the WS4 P3
    crash-privacy guards (crash-dialog state is per-process on Windows
    so the parent's install does not propagate to spawn-mode children),
    then ``importlib.import_module`` on its target — that brings in
    ``torch`` etc. only for ``gpu_ml_worker``.
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

    def start_all(self) -> None:
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

    def stop_all(self, timeout: float = 5.0) -> None:
        for evt in self.shutdown_events.values():
            evt.set()
        for name, proc in self.children.items():
            proc.join(timeout=timeout)
            if proc.is_alive():
                logger.warning("force-terminating %s pid=%s", name, proc.pid)
                proc.terminate()
                proc.join(timeout=timeout)
        self.children.clear()
        self.shutdown_events.clear()

    def wait(self) -> None:
        for proc in self.children.values():
            proc.join()
